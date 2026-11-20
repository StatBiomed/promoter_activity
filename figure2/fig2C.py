#!/usr/bin/env python3
"""Human promoter ages in tissue-specific multi-promoter (MP) genes only.

Python >=3.9; numpy, scipy, matplotlib. Reuses the parent age reference.
Input: results_by_organ/TISSUE/Human_TISSUE_major_minor_promoters.tsv.
MP means >=2 retained promoters with the same geneId WITHIN THAT TISSUE.
Validate the defined inputs, then remove whole single-promoter genes. Preserve
all MP promoters, activity values, row order and original Major/Minor labels.
No re-ranking, new expression threshold, cross-tissue union or isoform counting.

Figure contract: quantitative comparison of Major/Minor counts by mapping-based
age within MP genes. Gold Major / blue Minor; four rings show each age's
Major+Minor count divided by ALL MP-gene promoters in the same tissue.
Shared y scale by default; editable PDF text; 5.5 x 5.4 inches. No error bars.
Statistical test: parent within-gene Major-label randomization, recalculated on
MP genes; exact Poisson-binomial tails; BH over four ages separately per tissue.
These are sequence-mapping age categories, not established functional ages.

Outputs: one annotated TSV and PDF per tissue; one combined counts TSV;
run_manifest.json records input/output hashes, exclusions and test definitions.
Default refuses overwrite. --check-only writes no files; --png adds previews.
A tissue without MP genes fails explicitly (an age percentage is undefined).

Usage:
  python3 classify_and_plot_human_MP_promoter_age_by_tissue.py --check-only
  python3 classify_and_plot_human_MP_promoter_age_by_tissue.py
  python3 classify_and_plot_human_MP_promoter_age_by_tissue.py --tissues Testis
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo')
DEFAULT_INPUT = ROOT / '2.define_major_minor_promoter/try_human_liver/results_by_organ'
HERE = Path(__file__).resolve().parent
AGE_GROUPS = ('Amniote_conserved', 'Eutherian_conserved',
              'Primate_restricted', 'Human_specific')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_reference(path):
    require(path.is_file(), f'Reference script not found: {path}')
    spec = importlib.util.spec_from_file_location('_human_age_reference', path)
    require(spec is not None and spec.loader is not None, 'Cannot load reference')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    for name in ('read_tsv', 'annotate_rows', 'validate_groups', 'summarize',
                 'sha256', 'write_tsv', 'scientific_p', 'DEFAULT_AGES', 'BASIC',
                 'COLORS', 'AGE_GROUPS', 'AGE_RING_COLORS', 'METHOD'):
        require(hasattr(ref, name), f'Reference API missing: {name}')
    require(tuple(ref.AGE_GROUPS) == AGE_GROUPS, 'Reference age definitions changed')
    return ref


def discover_inputs(root, requested=None):
    require(root.is_dir(), f'Input directory not found: {root}')
    found = {}
    for path in sorted(root.glob('*/*_major_minor_promoters.tsv')):
        if not path.name.startswith('Human_'):
            print(f'SKIP non-Human input: {path.name}', flush=True)
            continue
        match = re.fullmatch(r'Human_([A-Za-z][A-Za-z0-9_-]*)_major_minor_promoters\.tsv', path.name)
        require(match is not None, f'Unexpected Human filename: {path.name}')
        tissue = match.group(1)
        require(path.parent.name == tissue, f'Tissue folder/filename conflict: {path}')
        require(tissue not in found, f'Duplicate tissue input: {tissue}')
        found[tissue] = path
    require(found, f'No Human tissue combined tables found in {root}')
    names = sorted(found) if requested is None else list(requested)
    require(names and len(names) == len(set(names)), 'Empty/duplicate tissue selection')
    require(set(names) <= set(found), f'Unknown tissues: {sorted(set(names) - set(found))}')
    selected = {name: found[name] for name in names}
    require(len({p.resolve() for p in selected.values()}) == len(selected),
            'Different tissues resolve to the same input file')
    return selected


def read_upstream_summary(ref, root, inputs, all_tissues):
    path = root / 'summary.tsv'
    _, rows = ref.read_tsv(path, ('organ', 'major_promoters', 'minor_promoters',
                                     'singleton_genes', 'multi_promoter_genes'))
    index = {}
    for row in rows:
        require(row['organ'] not in index, f'Duplicate organ in summary: {row["organ"]}')
        require(row.get('species', 'Human') == 'Human', 'Summary is not Human-only')
        index[row['organ']] = row
    require(set(inputs) <= set(index), 'Selected tissues missing from upstream summary')
    if all_tissues:
        require(set(inputs) == set(index), 'Summary and discovered Human tissues disagree')
    return path, index


def select_mp(ref, annotated, upstream):
    """Return only whole MP genes; verify labels and the upstream flowchart."""
    genes, before = ref.validate_groups(annotated)
    totals_before = Counter(r['promoter_class'] for r in annotated)
    for label, key in (('Major', 'major_promoters'), ('Minor', 'minor_promoters')):
        require(totals_before[label] == int(upstream[key]), f'Upstream {label} count mismatch')
    sp = before['single_promoter_genes']
    require(sp == int(upstream['singleton_genes']), 'Upstream SP gene count mismatch')
    mp = {gene: group for gene, group in genes.items() if len(group) >= 2}
    require(len(mp) == int(upstream['multi_promoter_genes']), 'Upstream MP gene count mismatch')
    require(mp, 'No MP genes in this tissue; cannot define age percentages')
    selected = [r for r in annotated if r['geneId'] in mp]
    totals = Counter(r['promoter_class'] for r in selected)
    require(totals['Major'] == len(mp), 'MP genes must retain exactly one Major each')
    require(totals['Minor'] == totals_before['Minor'], 'Minor promoters were unexpectedly removed')
    require(len(selected) == len(annotated)-sp, 'MP filtering count mismatch')
    _, after = ref.validate_groups(selected)
    require(after['single_promoter_genes'] == 0, 'SP gene survived filtering')
    audit = dict(input_promoters=len(annotated), input_genes=len(genes),
                 excluded_SP_genes=sp, excluded_SP_major_promoters=sp,
                 retained_MP_genes=len(mp), retained_major=totals['Major'],
                 retained_minor=totals['Minor'], retained_promoters=len(selected),
                 genes_with_tied_maxima_labels_preserved=after['genes_with_tied_maxima_labels_preserved'])
    return selected, mp, totals, audit


def plot_tissue(ref, tissue, rows, totals, path, *, statistics=True, peak=None, png=False):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                        'pdf.fonttype': 42, 'svg.fonttype': 'none',
                        'axes.linewidth': .8, 'axes.spines.top': False,
                        'axes.spines.right': False, 'legend.frameon': False})
    scale = max(1, peak if peak is not None else max(r[k] for r in rows for k in ('major_n', 'minor_n')))
    fig = plt.figure(figsize=(5.5, 5.4))
    ax = fig.add_axes([.19, (4.5*.27+.9)/5.4, .77, (4.5*.58)/5.4])
    x = np.arange(4)
    width = .29
    for label, offset, key in (('Major', -width*.65, 'major_n'), ('Minor', width*.65, 'minor_n')):
        bars = ax.bar(x+offset, [r[key] for r in rows], width, color=ref.COLORS[label],
                      edgecolor='#555555', linewidth=.55,
                      label=f'{label} (N={totals[label]:,})')
        for bar, row in zip(bars, rows):
            ax.text(bar.get_x()+width/2, bar.get_height()+scale*.016,
                    f'{row[key]:,}', ha='center', va='bottom', fontsize=8)
    ax.set(xlim=(-.55, 3.55), ylim=(0, scale*1.30))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_ylabel('Number of Promoters', fontsize=14, labelpad=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['TSS_age_label']}\n(n={r['total_n']:,})" for r in rows],
                       rotation=25, ha='right', rotation_mode='anchor', fontsize=11)
    ax.tick_params(axis='x', length=0, pad=7)
    legend = fig.legend(*ax.get_legend_handles_labels(), loc='upper right',
                        bbox_to_anchor=(.975, .99), fontsize=10.5,
                        handlelength=1.15, labelspacing=.25, borderpad=.2)
    # Keep the tissue title above the bars and below the legend, so long organ
    # names cannot overlap the Major/Minor sample sizes.
    title = fig.text(.19, .89, f'Human | {tissue} (MP genes)', fontsize=11, va='center')
    if statistics:
        for i, row in enumerate(rows):
            if row['log_p_BH'] < math.log(.05):
                y = max(row['major_n'], row['minor_n'])+scale*.085
                ax.plot([i-.22, i-.22, i+.22, i+.22],
                        [y, y+scale*.018, y+scale*.018, y], color='black', lw=.7)
                ax.text(i, y+scale*.031, '$P_{BH}$='+ref.scientific_p(row['log_p_BH'], 2),
                        ha='center', va='bottom', fontsize=7.5)
    total = sum(totals.values())
    require(total == sum(r['total_n'] for r in rows), 'Donut denominator mismatch')
    rings = []
    for i, (row, color) in enumerate(zip(rows, ref.AGE_RING_COLORS)):
        fraction = row['total_n']/total
        require(row['all_promoters_n'] == total and math.isclose(100*fraction, row['age_percent']),
                'Inconsistent tissue age percentage')
        cx = .19+.77*((i+.55)/4.10)
        ring = fig.add_axes([cx-.05, .115, .10, .55/5.4])
        ring.pie([fraction, 1-fraction], startangle=90, counterclock=False,
                 colors=[color, '#EEEEEE'],
                 wedgeprops={'width': .38, 'edgecolor': 'white', 'linewidth': .7})
        ring.text(0, 0, f'{100*fraction:.1f}%', ha='center', va='center', fontsize=8)
        ring.set_aspect('equal')
        rings.append(ring)
    fig.text(.19, .085, f'Age share of MP-gene promoters (Major + Minor; N={total:,}).', fontsize=6.6)
    fig.text(.19, .050, 'MP genes only; within-gene label test; BH across 4 ages within tissue.'
             if statistics else 'Descriptive counts; statistical tests omitted.', fontsize=6.6)
    fig.text(.19, .018, "'Human': no mapping detected in five target species.", fontsize=7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        if artist in ax.get_yticklabels() and not 0 <= artist.get_position()[1] <= ax.get_ylim()[1]:
            continue
        b = artist.get_window_extent(renderer)
        require(b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1
                and b.y1 <= fig.bbox.height+1, f'Text outside figure: {artist.get_text()}')
    legend_box, title_box = legend.get_window_extent(renderer), title.get_window_extent(renderer)
    require(legend_box.y0 > ax.bbox.y1 and title_box.y0 > ax.bbox.y1
            and not legend_box.overlaps(title_box), 'Header overlap')
    for ring in rings:
        box = ring.get_window_extent(renderer)
        require(all(not box.overlaps(t.get_window_extent(renderer))
                    for t in list(ax.get_xticklabels())+fig.texts), 'Donut/text overlap')
    fig.savefig(path, facecolor='white', metadata={
        'Title': f'Human {tissue}: MP-gene Major and Minor promoter counts by mapping-based age',
        'Subject': 'MP genes only; tissue-specific labels retained; SP genes excluded. '
                   'Rings: age count / all MP-gene promoters in this tissue. '+
                   (mp_method(ref) if statistics else 'Descriptive only.')})
    if png:
        fig.savefig(path.with_suffix('.png'), dpi=220, facecolor='white')
    plt.close(fig)


def mp_method(ref):
    return ref.METHOD.replace('SP labels fixed', 'MP genes only; SP genes excluded')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--reference-script', type=Path, default=HERE.parent/'classify_and_plot_human_promoter_age.py')
    parser.add_argument('--age-table', type=Path)
    parser.add_argument('--outdir', type=Path, default=HERE)
    parser.add_argument('--tissues', nargs='+')
    parser.add_argument('--no-statistics', action='store_true')
    parser.add_argument('--free-y', action='store_true', help='Independent y scales; default shared across selected tissues')
    parser.add_argument('--png', action='store_true', help='Also save one PNG preview per tissue')
    parser.add_argument('--check-only', action='store_true', help='Validate and count; no tests, plots or output files')
    parser.add_argument('--overwrite', action='store_true', help='Replace only this run\'s named output files')
    args = parser.parse_args()
    ref = load_reference(args.reference_script)
    inputs = discover_inputs(args.input_dir, args.tissues)
    summary_path, upstream = read_upstream_summary(ref, args.input_dir, inputs, args.tissues is None)
    age_table = args.age_table or ref.DEFAULT_AGES
    sources = [args.reference_script, age_table, summary_path, *inputs.values()]
    hashes = {str(p.resolve()): ref.sha256(p) for p in sources}
    _, age_rows = ref.read_tsv(age_table, ref.BASIC+('TSS_age_group',))
    processed, all_counts = {}, []
    for tissue, path in inputs.items():
        fields, rows = ref.read_tsv(path, ref.BASIC+('mean_promoter_activity', 'promoter_class'))
        require(not {'TSS_age_group', 'TSS_age_label'} & set(fields), 'Input already contains age columns')
        annotated = ref.annotate_rows(rows, age_rows)
        annotated, genes, totals, gene_summary = select_mp(ref, annotated, upstream[tissue])
        counts = ref.summarize(annotated, genes, do_tests=not (args.no_statistics or args.check_only))
        for count in counts:
            if 'method' in count:
                count['method'] = mp_method(ref)
            all_counts.append(dict(species='Human', tissue=tissue, scope='MP genes only',
                                   conditions=upstream[tissue].get('conditions', ''),
                                   MP_genes=len(genes), excluded_SP_genes=gene_summary['excluded_SP_genes'],
                                   **count))
        processed[tissue] = (fields, annotated, totals, counts, gene_summary)
        print(f'VALIDATED {tissue}: {len(genes):,} MP genes; {len(annotated):,} promoters; '
              f'Major={totals["Major"]:,}; Minor={totals["Minor"]:,}', flush=True)
    require(all(ref.sha256(Path(p)) == digest for p, digest in hashes.items()), 'Input changed during processing')
    if args.check_only:
        print('CHECK PASSED. All MP selections, age joins and upstream counts validated; no files written.')
        return

    relative = [Path('all_tissues_MP_promoter_age_counts.tsv'), Path('run_manifest.json')]
    for tissue in processed:
        stem = f'Human_{tissue}_MP_major_minor'
        relative += [Path(tissue)/(stem+'_promoters_with_age.tsv'),
                     Path(tissue)/(stem+'_promoter_age_counts.pdf')]
        if args.png:
            relative.append(Path(tissue)/(stem+'_promoter_age_counts.png'))
    targets = [args.outdir/p for p in relative]
    protected = {p.resolve() for p in sources} | {Path(__file__).resolve()}
    require(all(p.resolve() not in protected for p in targets), 'Output would overwrite a source')
    require(args.overwrite or not any(p.exists() for p in targets), 'Output exists; choose --outdir or explicitly --overwrite')
    peak = max(r[k] for r in all_counts for k in ('major_n', 'minor_n'))
    args.outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.age_staging_', dir=args.outdir) as temporary:
        stage = Path(temporary)
        for tissue, (fields, annotated, totals, counts, _) in processed.items():
            folder = stage/tissue
            folder.mkdir()
            stem = f'Human_{tissue}_MP_major_minor'
            ref.write_tsv(folder/(stem+'_promoters_with_age.tsv'),
                          fields+['TSS_age_group', 'TSS_age_label'], annotated)
            plot_tissue(ref, tissue, counts, totals, folder/(stem+'_promoter_age_counts.pdf'),
                        statistics=not args.no_statistics, peak=None if args.free_y else peak, png=args.png)
        ref.write_tsv(stage/relative[0], list(all_counts[0]), all_counts)
        manifest = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
                        species='Human', tissues=list(processed), source_sha256=hashes,
                        script_sha256=ref.sha256(Path(__file__)), command=sys.argv,
                        scope='Only genes with >=2 retained promoters within each tissue; preserve original labels and values; remove whole SP genes',
                        age_method='Reuse existing human sequence-mapping age annotation; no new liftover',
                        age_caveat='Mapping-based categories, not established functional promoter age; Human means no mapping detected',
                        statistics=None if args.no_statistics else mp_method(ref),
                        bh_family='Four ages separately within each tissue; NOT across all tissues',
                        donut_denominator='All Major + Minor promoters in MP genes within the same tissue',
                        y_scales='independent' if args.free_y else 'shared across selected tissues',
                        selection={t: item[4] for t, item in processed.items()},
                        qa=dict(duplicate_ids=0, missing_ages=0, annotation_conflicts=0,
                                MP_rows_and_labels_preserved=True, all_minor_promoters_preserved=True,
                                SP_genes_excluded=True, upstream_counts_match=True,
                                one_major_per_gene=True, text_clipping_checked=True,
                                header_and_donut_overlap_checked=True),
                        output_sha256={str(p.relative_to(stage)): ref.sha256(p)
                                       for p in stage.rglob('*') if p.is_file()})
        (stage/'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False,
                                                        allow_nan=False)+'\n', encoding='utf-8')
        require(all(ref.sha256(Path(p)) == digest for p, digest in hashes.items()), 'Input changed before publication')
        for relative_path in [p for p in relative if p.name != 'run_manifest.json']+[Path('run_manifest.json')]:
            destination = args.outdir/relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.overwrite:
                os.replace(stage/relative_path, destination)
            else:
                os.link(stage/relative_path, destination)
    print(f'COMPLETE: {len(processed)} tissue figures written to {args.outdir}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
