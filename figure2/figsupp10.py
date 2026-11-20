#!/usr/bin/env python3
"""Join organ-specific Human MP-promoter ages to existing maSigPro calls.

Primary figures: 100% stacked bars of Dynamic/Non-dynamic within each age,
separately for each organ and Major/Minor class. Each single-page PDF is 5 x 4
inches. Percentages remain in source tables, but are not annotated on bars.
The figure is descriptive: no new dynamic calls, age assignments or tests.

Default inputs are read-only. Exact promoter coverage, class, gene and locus
annotations must reconcile before anything is exported. A missing age or an
unmatched ID is an error, never a Human age or a Non-dynamic call.

Run with the existing project Python environment:
  python plot_age_dynamic_overlap.py --self-test
  python plot_age_dynamic_overlap.py --check-only
  python plot_age_dynamic_overlap.py
Use --output-dir for a fresh destination; existing outputs are never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np

BASE = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo')
AGE_ROOT = BASE / '3.get_promoter_age/all_tissues_MP'
STATS_ROOT = BASE / '8.alternative_promoter_usage_development/8.1.run_maSigPro/output'
ORGANS = ['Brain', 'Cerebellum', 'Heart', 'Kidney', 'Liver', 'Ovary', 'Testis']
AGE_LABELS = {'Amniote_conserved': 'Amniote', 'Eutherian_conserved': 'Eutherian',
              'Primate_restricted': 'Primate', 'Human_specific': 'Human'}
STATES = ['Dynamic', 'Non-dynamic', 'Not testable']
COLORS = {'Dynamic': '#ef5285', 'Non-dynamic': '#60c5ba'}
COMMON = ['geneId', 'seqnames', 'start', 'strand', 'internalPromoter',
          'promoterPosition', 'txId', 'mean_promoter_activity', 'promoter_class']
def figure_filename(organ, kind):
    return f'Human_{organ}_MP_{kind.lower()}_age_dynamic_overlap.pdf'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def promoter_key(raw):
    match = re.fullmatch(r'(?:promoter_)?([0-9]+)', raw.strip())
    require(match is not None, f'Invalid promoter ID: {raw!r}')
    return str(int(match.group(1)))


def read_table(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), f'Duplicate columns: {path}')
        require(set(required) <= set(fields), f'Missing columns in {path}: {set(required)-set(fields)}')
        rows = list(reader)
    require(rows, f'Empty input: {path}')
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows),
            f'Malformed input rows: {path}')
    index = {}
    for row in rows:
        key = promoter_key(row['promoterId'])
        require(key not in index, f'Duplicate normalized promoter ID {key}: {path}')
        require(all(row[c].strip() for c in ['geneId', 'promoter_class']), f'Empty gene/class: {path}, {key}')
        require(row['promoter_class'] in ['Major', 'Minor'], f'Unknown class: {path}, {key}')
        index[key] = row
    return fields, index


def numeric_or_missing(raw, field):
    if raw.strip().lower() in ['', 'na', 'nan', 'none', 'null', '.', 'inf', '-inf']:
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f'Invalid {field}: {raw!r}') from error
    if not math.isfinite(value):
        return None
    require(0 <= value <= 1, f'{field} outside [0, 1]: {raw}')
    return value


def call_status(row):
    """Preserve upstream flags; verify q <= Q, matching the actual R backend."""
    p = numeric_or_missing(row['p_value'], 'p_value')
    q = numeric_or_missing(row['padj_BH'], 'padj_BH')
    cutoff = numeric_or_missing(row['Q'], 'Q')
    require(cutoff is not None and 0 < cutoff < 1, 'Invalid upstream Q')
    flag = row['dynamic'].strip().upper()
    require(flag in ['TRUE', 'FALSE', 'T', 'F', '1', '0', '', 'NA', 'NAN'],
            f'Unknown dynamic flag: {flag}')
    positive = flag in ['TRUE', 'T', '1']
    if p is None or q is None:
        require(not positive, 'Dynamic=TRUE with missing p/q')
        return 'Not testable', 'missing_or_nonfinite_p_or_q'
    require(flag in ['TRUE', 'FALSE', 'T', 'F', '1', '0'], 'Finite p/q but missing dynamic flag')
    require(positive == (q <= cutoff), f'dynamic flag disagrees with padj_BH <= Q: {row["promoterId"]}')
    return ('Dynamic' if positive else 'Non-dynamic'), ''


def validate_age_groups(index, organ):
    genes = defaultdict(list)
    for key, row in index.items():
        require(row['TSS_age_group'] in AGE_LABELS, f'Unknown age: {organ}, {key}')
        require(row['TSS_age_label'] == AGE_LABELS[row['TSS_age_group']], f'Conflicting age label: {organ}, {key}')
        require(row['internalPromoter'].strip().upper() in ['FALSE', 'F', '0'], f'Internal promoter: {organ}, {key}')
        require(row['strand'] in ['+', '-'] and int(row['start']) > 0, f'Invalid locus: {organ}, {key}')
        try:
            activity = Decimal(row['mean_promoter_activity'])
        except InvalidOperation as error:
            raise ValueError(f'Invalid activity: {organ}, {key}') from error
        require(activity.is_finite() and activity > 0, f'Inactive/invalid age-table promoter: {organ}, {key}')
        genes[row['geneId']].append(row)
    for gene, group in genes.items():
        require(len(group) >= 2, f'Single-promoter gene in MP age table: {organ}, {gene}')
        require(sum(r['promoter_class'] == 'Major' for r in group) == 1,
                f'Expected one Major per MP gene: {organ}, {gene}')
    return genes


def reconcile(ages, stats, organ, kind):
    expected = {k: r for k, r in ages.items() if r['promoter_class'] == kind}
    missing, extra = set(expected)-set(stats), set(stats)-set(expected)
    require(not missing and not extra,
            f'{organ}/{kind} coverage mismatch: missing={len(missing)} {sorted(missing)[:8]}, '
            f'extra={len(extra)} {sorted(extra)[:8]}')
    result = []
    for key, age in expected.items():
        row = stats[key]
        for field in COMMON:
            # No reclassification or averaging: preserve and compare upstream strings.
            require(age[field] == row[field], f'{organ}/{kind} annotation mismatch: {key}, {field}')
        require(row['BH_family'].strip().lower() == kind.lower(), f'Wrong BH family: {organ}/{kind}, {key}')
        status, reason = call_status(row)
        result.append(dict(species='Human', organ=organ, promoterId=age['promoterId'],
                           statistics_promoterId=row['promoterId'],
                           **{c: age[c] for c in COMMON},
                           TSS_age_group=age['TSS_age_group'], TSS_age_label=age['TSS_age_label'],
                           **{c: row[c] for c in ['p_value', 'padj_BH', 'dynamic', 'Q', 'degree', 'BH_family']},
                           analysis_status=status, not_testable_reason=reason))
    return result


def percent(n, total):
    return 100 * n / total if total else None


def summarize(joined, organ, kind):
    counts = Counter((r['TSS_age_group'], r['analysis_status']) for r in joined)
    status_totals = Counter(r['analysis_status'] for r in joined)
    wide, long = [], []
    for age, label in AGE_LABELS.items():
        d, nd, nt = (counts[age, state] for state in STATES)
        tested, total = d+nd, d+nd+nt
        record = dict(species='Human', organ=organ, promoter_class=kind,
                      TSS_age_group=age, TSS_age_label=label,
                      age_promoters_n=total, testable_n=tested, dynamic_n=d,
                      non_dynamic_n=nd, not_testable_n=nt,
                      dynamic_pct=percent(d, tested), non_dynamic_pct=percent(nd, tested),
                      not_testable_pct_of_all_age=percent(nt, total))
        if tested:
            require(math.isclose(record['dynamic_pct']+record['non_dynamic_pct'], 100, abs_tol=1e-10),
                    'Within-age percentages do not sum to 100')
        wide.append(record)
        for state, n in zip(STATES, [d, nd, nt]):
            long.append(dict(species='Human', organ=organ, promoter_class=kind,
                             TSS_age_group=age, TSS_age_label=label, analysis_status=state,
                             overlap_n=n, testable_in_age_n=tested, all_in_age_n=total,
                             pct_within_age_tested=percent(n, tested) if state != 'Not testable' else None,
                             same_status_all_ages_n=status_totals[state],
                             pct_within_status_all_ages=percent(n, status_totals[state])))
    require(sum(r['age_promoters_n'] for r in wide) == len(joined), 'Count reconciliation failed')
    for state in STATES:
        if status_totals[state]:
            require(math.isclose(sum(r['pct_within_status_all_ages'] for r in long
                                     if r['analysis_status'] == state), 100, abs_tol=1e-10),
                    'Within-status age percentages do not sum to 100')
    return wide, long


def collect(age_root, stats_root, requested=None):
    found = {}
    for path in sorted(age_root.glob('*/*_MP_major_minor_promoters_with_age.tsv')):
        match = re.fullmatch(r'Human_([A-Za-z][A-Za-z0-9_-]*)_MP_major_minor_promoters_with_age\.tsv', path.name)
        if not match:
            continue
        organ = match.group(1)
        require(organ == path.parent.name and organ not in found, f'Ambiguous organ input: {path}')
        found[organ] = path
    require(found, f'No Human MP age tables: {age_root}')
    organs = list(requested) if requested else [o for o in ORGANS if o in found]+sorted(set(found)-set(ORGANS))
    require(len(organs) == len(set(organs)) and set(organs) <= set(found), 'Unknown/duplicate selected organs')
    joined, wide, long, audits, hashes = [], [], [], [], {}
    for organ in organs:
        age_file = found[organ]
        hashes[str(age_file.resolve())] = sha256(age_file)
        _, ages = read_table(age_file, ['promoterId', *COMMON, 'TSS_age_group', 'TSS_age_label'])
        genes = validate_age_groups(ages, organ)
        observed = set()
        for kind in ['Major', 'Minor']:
            stats_file = stats_root / f'Human_{organ}_multipromoter' / kind.lower() / f'{organ}_all_promoters_statistics.tsv'
            require(stats_file.is_file(), f'Missing statistics table: {stats_file}')
            hashes[str(stats_file.resolve())] = sha256(stats_file)
            _, stats = read_table(stats_file, ['promoterId', *COMMON, 'p_value', 'padj_BH', 'dynamic', 'Q', 'degree', 'BH_family'])
            rows = reconcile(ages, stats, organ, kind)
            require(not observed.intersection(stats), f'Major/minor overlap: {organ}')
            observed.update(stats)
            rows_wide, rows_long = summarize(rows, organ, kind)
            joined.extend(rows); wide.extend(rows_wide); long.extend(rows_long)
            counts = Counter(r['analysis_status'] for r in rows)
            audit = dict(organ=organ, promoter_class=kind, age_promoters_n=len(rows),
                         statistics_promoters_n=len(stats), matched_n=len(rows),
                         missing_ids_n=0, extra_ids_n=0, duplicate_ids_n=0, class_conflicts_n=0,
                         shared_annotation_conflicts_n=0, MP_genes_n=len(genes),
                         dynamic_n=counts['Dynamic'], non_dynamic_n=counts['Non-dynamic'],
                         not_testable_n=counts['Not testable'],
                         upstream_Q=','.join(sorted({r['Q'] for r in rows})),
                         upstream_degree=','.join(sorted({r['degree'] for r in rows})),
                         age_file=str(age_file.resolve()), statistics_file=str(stats_file.resolve()))
            audits.append(audit)
            print(f'VALIDATED {organ:12s} {kind:5s}: {len(rows):,} matched; '
                  f'Dynamic={counts["Dynamic"]:,}, Non-dynamic={counts["Non-dynamic"]:,}, '
                  f'Not testable={counts["Not testable"]:,}', flush=True)
        require(observed == set(ages), f'Age universe not fully reconciled: {organ}')
    require(all(sha256(Path(p)) == h for p, h in hashes.items()), 'Input changed during reconciliation')
    return organs, joined, wide, long, audits, hashes


def draw_page(organ, kind, rows):
    """A categorical age comparison, not a continuous evolutionary time axis."""
    data = [r for r in rows if r['organ'] == organ and r['promoter_class'] == kind]
    require([r['TSS_age_group'] for r in data] == list(AGE_LABELS), 'Plot order mismatch')
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_axes([.17, .23, .79, .55])
    fig.text(.565, .942, f'Human | {organ} | {kind}', ha='center', fontsize=14)
    fig.legend(handles=[Patch(facecolor=COLORS[s], label=s) for s in STATES[:2]],
               loc='upper center', bbox_to_anchor=(.565, .89), ncol=2,
               frameon=False, fontsize=12, handlelength=1.2, handletextpad=.4, columnspacing=1.2)
    d = np.array([r['dynamic_pct'] if r['testable_n'] else 0 for r in data])
    nd = np.array([r['non_dynamic_pct'] if r['testable_n'] else 0 for r in data])
    x = np.arange(4)
    bars_d = ax.bar(x, d, width=.65, color=COLORS['Dynamic'], edgecolor='none', rasterized=False)
    bars_nd = ax.bar(x, nd, bottom=d, width=.65, color=COLORS['Non-dynamic'], edgecolor='none', rasterized=False)
    for i, row in enumerate(data):
        ax.text(i, 103.5, f'n={row["testable_n"]:,}', ha='center', va='bottom', fontsize=9)
        if not row['testable_n']:
            ax.text(i, 50, 'NA', ha='center', va='center', fontsize=12)
            continue
        # Keep exact stack heights; omit percentage text as requested.
    ax.set(xlim=(-.6, 3.6), ylim=(0, 112), xticks=x, yticks=range(0, 101, 20))
    ax.set_xticklabels(list(AGE_LABELS.values()), fontsize=12)
    ax.set_xlabel('Promoter age', fontsize=14, labelpad=8)
    ax.set_ylabel('Promoters (%)', fontsize=14, labelpad=7)
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis='x', length=0, pad=6)
    ax.spines['left'].set_bounds(0, 100)
    excluded = sum(r['not_testable_n'] for r in data)
    fig.text(.565, .045, f'n = tested promoters; not testable excluded: {excluded:,}',
             ha='center', fontsize=8, color='#444444')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            b = t.get_window_extent(renderer)
            require(b.x0 >= -.5 and b.y0 >= -.5 and b.x1 <= fig.bbox.width+.5
                    and b.y1 <= fig.bbox.height+.5, f'Text outside page: {t.get_text()}')
    for i, (a, b) in enumerate(zip(bars_d, bars_nd)):
        require(math.isclose(a.get_height(), d[i], abs_tol=1e-12)
                and math.isclose(b.get_height(), nd[i], abs_tol=1e-12), 'Bar values changed')
        if data[i]['testable_n']:
            require(math.isclose(a.get_height()+b.get_height(), 100, abs_tol=1e-10), 'Stack does not sum to 100')
    # Reject collisions between adjacent category labels, not just clipping.
    tick_boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
    require(all(a.x1+2 < b.x0 for a, b in zip(tick_boxes, tick_boxes[1:])), 'Age tick labels overlap')
    return fig, dict(organ=organ, promoter_class=kind, figure_size_inches=[5, 4],
                     bars=4, vector_segments=8, excluded_not_testable_n=excluded,
                     text_within_canvas=True, age_tick_labels_do_not_overlap=True,
                     exact_percentages_match_source=True, common_y_scale=[0, 100],
                     percentage_annotations=False, file=figure_filename(organ, kind))


def write_tsv(path, rows):
    require(rows, f'No rows for {path}')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('NA' if v is None else format(v, '.12g') if isinstance(v, float) else v)
                             for k, v in row.items()})


def self_test():
    require(promoter_key('promoter_001') == promoter_key('1') == '1', 'ID normalization test')
    for bad in ['1.0', '', 'P1']:
        try:
            promoter_key(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid ID accepted')
    sample = dict(promoterId='1', p_value='.001', padj_BH='.01', Q='.01', dynamic='TRUE')
    require(call_status(sample)[0] == 'Dynamic', 'q=Q boundary test')
    require(call_status(dict(sample, dynamic='FALSE', padj_BH='.02'))[0] == 'Non-dynamic', 'Non-dynamic test')
    require(call_status(dict(sample, dynamic='FALSE', p_value='NA', padj_BH='NA'))[0] == 'Not testable', 'NA test')
    try:
        call_status(dict(sample, dynamic='FALSE'))
    except ValueError:
        pass
    else:
        raise AssertionError('Conflicting flag accepted')
    fixture = [dict(TSS_age_group='Amniote_conserved', analysis_status=s) for s in STATES]
    wide, long = summarize(fixture, 'Test', 'Major')
    require(wide[0]['dynamic_pct'] == wide[0]['non_dynamic_pct'] == 50, 'Denominator test')
    require(wide[0]['age_promoters_n'] == 3 and wide[0]['testable_n'] == 2, 'Exclude NA only from tested denominator')
    require(wide[1]['dynamic_pct'] is None, 'Empty age group must be NA, not zero')
    age = dict(promoterId='1', **{c:'same' for c in COMMON}, TSS_age_group='Human_specific', TSS_age_label='Human')
    age['promoter_class'] = 'Major'
    stats = dict(age, **sample, BH_family='major', degree='2')
    reconcile({'1':age}, {'1':stats}, 'Test', 'Major')
    for broken in [{}, {'1':dict(stats, promoter_class='Minor')}, {'1':dict(stats, geneId='wrong')}]:
        try:
            reconcile({'1':age}, broken, 'Test', 'Major')
        except ValueError:
            pass
        else:
            raise AssertionError('Broken join accepted')
    print('SELF-TEST PASS: IDs, duplicate-safe joins, label/locus guards, q<=Q boundary, NA exclusion, zero denominators.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--age-root', type=Path, default=AGE_ROOT)
    parser.add_argument('--stats-root', type=Path, default=STATS_ROOT)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'output')
    parser.add_argument('--organs', nargs='+')
    parser.add_argument('--check-only', action='store_true', help='Validate all sources; write nothing')
    parser.add_argument('--self-test', action='store_true', help='Run synthetic checks; write nothing')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    organs, joined, wide, long, audits, hashes = collect(args.age_root, args.stats_root, args.organs)
    if args.check_only:
        print(f'CHECK-ONLY PASS: {len(organs)} organs; {len(joined):,} organ-promoter rows; no files written.', flush=True)
        return
    output = args.output_dir.resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            f'Output not empty; choose a new --output-dir (nothing overwritten): {output}')
    require(output != args.age_root.resolve() and output != args.stats_root.resolve(), 'Output is an input directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    style = {'font.family':'DejaVu Sans', 'pdf.fonttype':42, 'pdf.compression':0,
             'font.size':12, 'axes.spines.top':False, 'axes.spines.right':False,
             'axes.linewidth':.8, 'axes.grid':False, 'figure.facecolor':'white',
             'axes.facecolor':'white', 'legend.frameon':False}
    with tempfile.TemporaryDirectory(prefix='.overlap_staging_', dir=output.parent) as temporary:
        stage = Path(temporary)
        write_tsv(stage/'age_dynamic_overlap_summary.tsv', wide)
        write_tsv(stage/'age_dynamic_overlap_long.tsv', long)
        write_tsv(stage/'promoters_with_age_and_dynamic_status.tsv', joined)
        write_tsv(stage/'input_reconciliation.tsv', audits)
        plot_qa = []
        with plt.rc_context(style):
            for organ in organs:
                for kind in ['Major', 'Minor']:
                    fig, qa = draw_page(organ, kind, wide)
                    fig.savefig(stage/figure_filename(organ, kind), format='pdf', facecolor='white', metadata={
                        'Title':f'Human {organ} {kind}: dynamic fraction within each sequence-age class',
                        'Subject':'100% stacks among testable promoters; percentage text omitted. '
                                  'Existing calls, mapping-age labels, counts and stack heights unchanged. '
                                  'Non-dynamic means not significant, not proven stable.',
                        'CreationDate':None, 'ModDate':None})
                    plt.close(fig)
                    plot_qa.append(qa)
        notes = '''# Human MP promoter age and dynamic-status overlap

One separate, single-page PDF is produced for each organ x Major/Minor.
Dynamic = #ef5285; Non-dynamic = #60c5ba. Percentage text is not drawn on bars.
Each of the four age bars sums to 100% among the TESTABLE promoters in that age.
Dynamic percent = Dynamic / (Dynamic + Non-dynamic) x 100.
Non-dynamic percent = Non-dynamic / (Dynamic + Non-dynamic) x 100.
n above a bar is its testable-promoter denominator, not biological replicates.
Promoters lacking finite p/q are Not testable and excluded from those two percentages,
but retained in the joined table, all-age totals, reconciliation and long-form table.
An age with zero testable promoters has NA percentages and no 100% bar.

The existing dynamic flag is preserved and checked against padj_BH <= its recorded Q.
The upstream maSigPro fits use separate Major/Minor BH families within each organ.
No maSigPro rerun, altered threshold, new BH correction, enrichment test or p-value
is introduced. Non-dynamic means not significant in the existing analysis, not proven stable.
These plots are descriptive; no error bars, confidence intervals or stars are added.
Promoters of the same gene are not independent biological replicates.

Age annotation comes from the specified organ-specific MP tables. The full expected
Major and Minor ID sets must match their statistics files exactly. Gene/locus/class,
internal-promoter flag, transcript IDs and mean activities must also agree.
Missing IDs, age labels, duplicated normalized IDs and annotation conflicts stop the run.
Major/minor are organ-wide labels, not reselected at each developmental stage.
No single-promoter genes are added; internal and inactive promoters are not reintroduced.
Only Human is analyzed because the provided age directory contains Human age tables.

Amniote/Eutherian/Primate/Human are existing sequence-mapping categories, not sample ages
or established functional promoter origins. Human means no mapping detected in five
comparison species, not confirmed de novo human-specific origin.

Outputs:
- age_dynamic_overlap_summary.tsv: one row per organ x class x age; numerator, denominator and plotted percentages.
- age_dynamic_overlap_long.tsv: one row per organ x class x age x status. It also gives the REVERSE denominator,
  pct_within_status_all_ages, describing age composition within Dynamic or Non-dynamic. That is not the plotted quantity.
- promoters_with_age_and_dynamic_status.tsv: complete promoter-level joined annotations and statistics.
- input_reconciliation.tsv: matching counts, original thresholds, dynamic totals and source paths.
- run_manifest.json: source/output hashes, definitions and figure QA.
- Human_<Organ>_MP_<major|minor>_age_dynamic_overlap.pdf: one editable RGB vector PDF per panel; 5 x 4 inches.

Run the script without arguments for all available Human organs. --check-only validates without writing.
--organs Brain Liver restricts the analysis. Use --output-dir for a fresh destination.
Existing output directories are never overwritten, and no input files are modified.
'''
        (stage/'README.md').write_text(notes, encoding='utf-8')
        require(all(sha256(Path(p)) == h for p, h in hashes.items()), 'Input changed before export')
        manifest = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
                        species='Human', organs=organs, source_sha256=hashes,
                        script_sha256=sha256(Path(__file__)), command=sys.argv,
                        figure_contract=dict(question='Dynamic fraction within each age, separately by organ and Major/Minor',
                                             archetype='quantitative grid as separate single-page PDFs',
                                             backend='Python/matplotlib', size_inches=[5,4],
                                             format='Individual PDF files; editable text; RGB vectors; no percentage annotations',
                                             statistics='Descriptive counts and percentages; no new tests or intervals'),
                        denominator='Dynamic + Non-dynamic in the same organ x promoter_class x age',
                        not_testable='Missing/nonfinite p or adjusted p; excluded from plotted denominator',
                        upstream_call='Keep original dynamic flag; validate padj_BH <= recorded Q',
                        palette=COLORS, age_labels=AGE_LABELS, promoter_organ_rows=len(joined),
                        summary_rows=len(wide), long_rows=len(long), PDF_files=len(plot_qa),
                        PDF_pages_per_file=1, percentage_annotations=False,
                        all_input_ids_and_shared_annotations_match=True, plot_qa=plot_qa,
                        output_sha256={p.name:sha256(p) for p in stage.iterdir() if p.is_file()})
        (stage/'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        # Publish only after all matching, plot checks and input-hash checks pass.
        if not output.exists():
            output.mkdir()
        require(not any(output.iterdir()), 'Output changed during processing')
        files = [p for p in stage.iterdir() if p.name != 'run_manifest.json']+[stage/'run_manifest.json']
        for path in files:
            os.link(path, output/path.name)
    print(f'COMPLETE: {len(organs)} organs; {len(plot_qa)} single-page PDFs; {len(wide)} age summaries -> {output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, InvalidOperation) as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
