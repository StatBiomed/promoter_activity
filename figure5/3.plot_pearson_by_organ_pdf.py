#!/usr/bin/env python3
"""One multi-organ PDF per species and metric scale (Python/matplotlib).

Python >=3.9 and matplotlib. Defaults: ./runs/<species>, both metric scales.
Each original organ-stage task stays one bar: no averaging, pooling, training,
inference, or changes to the input result files. Use --check-only to validate.
Human x ticks use the exact source stage strings (e.g. 4w, 0dpb, 6mpb, 13ypb).
Wholebrain is excluded from figures, never deleted from source results.
Default layout: four columns, a shared y axis per row, two-column legend.
Only PDF images are exported; TSV/JSON files record source data and checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parent
SPECIES = ('Human', 'Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken')
METRICS = {'original_activity': 'test_metrics_per_task.tsv',
           'log2p1': 'test_metrics_per_task_log2p1.tsv'}
SCALE_NAMES = {'original_activity': 'Original activity', 'log2p1': 'log2(activity + 1)'}
ORGAN_COLORS = {'Brain': '#3399CC', 'Cerebellum': '#33CCFF', 'Heart': '#CC0000',
                'Kidney': '#CC9900', 'Liver': '#339900', 'Ovary': '#CC3399', 'Testis': '#FF6600'}
# Colour aliases only. These do NOT merge result tables, stages or organs.
COLOR_ALIASES = {'WholeBrain': 'Brain', 'Wholebrain': 'Brain', 'Forebrain': 'Brain',
                 'Hindbrain': 'Cerebellum', 'HindBrain': 'Cerebellum'}
ORGAN_ORDER = ('Brain', 'Forebrain', 'Cerebellum', 'Hindbrain', 'HindBrain',
               'Heart', 'Kidney', 'Liver', 'Testis', 'Ovary')
DAYS = {'d': 1., 'w': 7., 'm': 365.25 / 12, 'y': 365.25}
YLABEL = 'Pearson\u2019s Correlation'
FIELDS = ('species', 'organ', 'color_organ', 'color_hex', 'stage', 'display_label',
          'stage_order', 'phase', 'sort_age_days', 'label_note', 'activity_scale',
          'pearson_r', 'pearson_status', 'n_valid', 'output_index_0based', 'target_column',
          'best_epoch', 'source_file')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    return format(value, 'g')


def is_wholebrain(organ):
    return re.sub(r'[^a-z]', '', organ.casefold()) == 'wholebrain'


def organ_key(organ):
    return (ORGAN_ORDER.index(organ) if organ in ORGAN_ORDER else len(ORGAN_ORDER), organ)


def parse_age(stage, species):
    """Return sortable phase, mean age in days, and parsed endpoints.

    Month/year-to-day conversion is ONLY for ordering, never a claim of an
    exact collection day. Range endpoints remain a single original task.
    """
    if stage.lower() == 'adult':
        return 2, 0., [], 'Adult: exact age unavailable in the metric label.'
    pieces = re.split(r'[-\u2013\u2014]', stage.strip().lower())
    require(len(pieces) in (1, 2), f'Unknown stage: {species}/{stage}')
    endpoints = []
    for piece in pieces:
        m = re.fullmatch(r'(\d+(?:\.\d+)?)([dwmy]?)(pb|ph)?', piece)
        require(m is not None, f'Unknown stage: {species}/{stage}')
        endpoints.append([float(m[1]), m[2], m[3] or ''])
    note = ''
    explicit_units = {e[1] for e in endpoints if e[1]}
    for e in endpoints:
        if e[2] and not e[1]:
            require(len(endpoints) == 2 and len(explicit_units) == 1, f'Ambiguous unit: {stage}')
            e[1] = next(iter(explicit_units))
            note = 'Range unit inherited from explicit endpoint; source label preserved.'
        if e[2]:
            require(e[2] == ('ph' if species == 'Chicken' else 'pb'), f'Wrong birth/hatch suffix: {species}/{stage}')
        else:
            require(e[1] in ('', 'd', 'w'), f'Ambiguous prenatal unit: {species}/{stage}')
            e[1] = e[1] or 'd'
    phases = {1 if e[2] else 0 for e in endpoints}
    require(len(phases) == 1, f'Range crosses birth/hatching: {stage}')
    days = [e[0] * DAYS[e[1]] for e in endpoints]
    require(len(days) == 1 or days[0] <= days[1], f'Reversed age range: {stage}')
    return phases.pop(), sum(days) / len(days), endpoints, note


def stage_label(stage, species, human_labels='raw'):
    require(human_labels == 'raw', 'Human labels must be raw source stages; category labels are no longer supported.')
    phase, age_days, endpoints, note = parse_age(stage, species)
    if species == 'Human':
        if phase == 0:
            require({e[1] for e in endpoints} == {'w'}, f'Human prenatal age needs weeks: {stage}')
        # Preserve exact source labels for ALL Human ages, including adult ages.
        # Age parsing only controls chronological sorting, never the label text.
        return stage, (phase, age_days, stage), note
    if phase == 2:
        # Do not silently turn Chicken Adult into P155 without sample metadata.
        return 'Adult', (phase, age_days, stage), note
    units = {e[1] for e in endpoints}
    if phase == 0:
        label = 'E' + '-'.join(number(e[0] * DAYS[e[1]]) for e in endpoints)
    elif species == 'Macaque':
        # P6m, P1y, P14-15y match the primate notation without losing age units.
        if units <= {'d', 'w'}:
            label = 'P' + '-'.join(number(e[0] * DAYS[e[1]]) for e in endpoints)
        elif len(units) == 1:
            label = 'P' + '-'.join(number(e[0]) for e in endpoints) + endpoints[0][1]
        else:
            label = 'P' + '-'.join(number(e[0])+e[1] for e in endpoints)
    else:
        if units <= {'d', 'w'}:
            label = 'P' + '-'.join(number(e[0] * DAYS[e[1]]) for e in endpoints)
        else:
            label = 'P' + '-'.join(number(e[0])+e[1] for e in endpoints)
            note = (note + ' Exact day not inferred from a month/year label.').strip()
    return label, (phase, age_days, stage), note


def read_metrics(run_root, species, scale, human_labels='raw'):
    directory = run_root / species
    status = json.loads((directory / 'run_status.json').read_text())
    require(status.get('status') == 'complete', f'Training incomplete: {directory}')
    summary = json.loads((directory / 'test_metrics_summary.json').read_text())
    require(summary['species'] == species, f'Species mismatch: {directory}')
    path = directory / METRICS[scale]
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t')
    required = {'organ', 'stage', 'activity_scale', 'pearson_r', 'pearson_status',
                'n_valid', 'output_index_0based', 'target_column'}
    require(reader.fieldnames and required <= set(reader.fieldnames), f'Missing columns: {path}')
    require(len(reader.fieldnames) == len(set(reader.fieldnames)), f'Duplicate headers: {path}')
    rows, seen, indices = [], set(), set()
    for raw in reader:
        require(None not in raw and all(raw.get(k) is not None for k in required), f'Ragged row: {path}')
        organ, stage = raw['organ'].strip(), raw['stage'].strip()
        color_organ = 'Brain' if is_wholebrain(organ) else COLOR_ALIASES.get(organ, organ)
        require(color_organ in ORGAN_COLORS, f'No reference colour for organ: {organ}')
        require(raw['activity_scale'] == scale, f'Mixed metric scales: {path}')
        require(raw['target_column'] == f'{species}_{organ}_{stage}.mean', f'Target identity mismatch: {path}')
        idx, n = int(raw['output_index_0based']), int(raw['n_valid'])
        require((organ, stage) not in seen and idx not in indices and idx >= 0, f'Duplicate task: {path}')
        require(0 <= n <= summary['n_test_promoters'], f'Invalid sample count: {path}')
        seen.add((organ, stage)); indices.add(idx)
        token = raw['pearson_r'].strip()
        value = None if token.lower() in ('', 'na', 'nan', 'none', 'null') else float(token)
        defined = raw['pearson_status'] in ('ok', 'defined_but_only_two_observations')
        require((value is not None) == defined and bool(raw['pearson_status']), f'Pearson status mismatch: {path}')
        require(value is None or (math.isfinite(value) and -1 <= value <= 1 and n >= 2), f'Invalid Pearson: {path}')
        label, order, note = stage_label(stage, species, human_labels)
        rows.append(dict(species=species, organ=organ, color_organ=color_organ,
            color_hex=ORGAN_COLORS[color_organ], stage=stage, display_label=label,
            phase=order[0], sort_age_days=order[1], label_note=note, activity_scale=scale,
            pearson_r=value, pearson_status=raw['pearson_status'], n_valid=n,
            output_index_0based=idx, target_column=raw['target_column'],
            best_epoch=summary['best_epoch'], source_file=str(path), _sort=order))
    require(rows and indices == set(range(len(rows))) and len(rows) == summary['n_outputs'], f'Task count/index mismatch: {path}')
    return rows, {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'n_tasks': len(rows)}


def write_tsv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: 'NA' if row.get(key) is None else row[key] for key in fields})


def make_figure(rows, limits, args):
    """Quantitative grid: one original organ-stage estimate per bar.

    One selected checkpoint, no invented error bars or significance tests.
    The y range is identical across all panels; only column one shows y axes.
    Returns the figure and data axes (the legend-only cell is not a data axis).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    grouped = defaultdict(list)
    for row in rows:
        if not is_wholebrain(row['organ']):
            grouped[row['organ']].append(row)
    require(grouped, 'No organs remain after excluding Wholebrain')
    organs = sorted(grouped, key=organ_key)
    ncols = min(args.columns, len(organs))
    nrows = math.ceil(len(organs) / ncols)
    spare_cell = len(organs) < nrows * ncols
    # Widen dense species panels instead of dropping or merging time labels.
    max_stages = max(map(len, grouped.values()))
    width = args.width or max(3.05, .10625 * max_stages) * ncols + .55
    height = args.height or 2.60 * nrows + .65 + (0 if spare_cell else .70)
    fig, grid = plt.subplots(nrows, ncols, figsize=(width, height),
                             sharey='row', squeeze=False)
    fig.subplots_adjust(left=.065, right=.99, bottom=.14 if not spare_cell else .13,
                        top=.86, wspace=.13, hspace=.68)
    species, scale = rows[0]['species'], rows[0]['activity_scale']
    fig.suptitle(species, fontsize=14, fontweight='normal', y=.98)
    fig.text(.5, .925, SCALE_NAMES[scale], ha='center', va='center',
             fontsize=8.5, color='#555555')
    axes = []
    for index, organ in enumerate(organs):
        ax = grid.flat[index]
        organ_rows = sorted(grouped[organ], key=lambda r:r['_sort'])
        n = len(organ_rows)
        for i, row in enumerate(organ_rows):
            r = row['pearson_r']
            if r is None:
                ax.text(i, .025, 'NA', ha='center', va='bottom', fontsize=6)
            else:
                ax.bar(i, r, width=args.bar_width, color=row['color_hex'],
                       edgecolor='white', linewidth=.20)
                if args.show_values:
                    inside = abs(r) > .93
                    direction = 1 if r >= 0 else -1
                    ax.text(i, r + direction * .02 * (-1 if inside else 1),
                            f'{r:.2f}', ha='center', fontsize=5, rotation=90,
                            va=('top' if r >= 0 else 'bottom') if inside else
                               ('bottom' if r >= 0 else 'top'),
                            color='white' if inside else '#222222')
        rotation = args.rotation if args.rotation is not None else (90 if n > 18 else 60)
        ax.set_xticks(range(n), [r['display_label'] for r in organ_rows],
                      rotation=rotation, ha='center' if rotation == 90 else 'right',
                      va='top', rotation_mode='default', fontsize=6 if n > 26 else 7)
        ax.set_xlim(-.55, n-.45)
        ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        first = index % ncols == 0
        ax.set_ylabel(YLABEL if first else '', fontsize=9, labelpad=6)
        ax.tick_params(axis='y', left=first, labelleft=first, width=.6, length=3,
                       labelsize=7)
        ax.tick_params(axis='x', width=.5, length=2, pad=2)
        ax.set_xlabel('')
        ax.set_title(organ, fontsize=10, pad=7)
        ax.grid(False, which='both', axis='both')
        ax.minorticks_off()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(first)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        if limits[0] < 0:
            ax.axhline(0, color='#666666', linewidth=.6)
        axes.append(ax)
    # Matplotlib fills legend columns first. Reorder handles so reading each
    # legend row left-to-right follows the organ panel order, two entries/row.
    handles = [Patch(facecolor=grouped[o][0]['color_hex'], label=o) for o in organs]
    handles = handles[::2] + handles[1::2]
    legend_kw = dict(handles=handles, ncol=2, frameon=False, fontsize=8.5,
                     handlelength=1.1, handleheight=.9, handletextpad=.5,
                     columnspacing=1.2, labelspacing=.9)
    for ax in list(grid.flat)[len(organs):]:
        ax.set_axis_off()
    if spare_cell:
        grid.flat[len(organs)].legend(loc='center', **legend_kw)
    else:
        fig.legend(loc='lower center', bbox_to_anchor=(.5, .005), **legend_kw)
    return fig, axes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=BASE/'runs')
    parser.add_argument('--out-dir', type=Path, default=BASE/'pearson_species_pdf')
    parser.add_argument('--species', nargs='+', choices=SPECIES, default=list(SPECIES))
    parser.add_argument('--scale', choices=['both', *METRICS], default='both')
    parser.add_argument('--human-labels', choices=['raw'], default='raw',
                        help='Human labels are always the exact source stage strings, with no age categories.')
    parser.add_argument('--width', type=float, default=None)
    parser.add_argument('--height', type=float, default=None, help='Full figure height in inches.')
    parser.add_argument('--columns', type=int, default=4, help='Organ panels per row (default: 4).')
    parser.add_argument('--bar-width', type=float, default=.96, help='Bar width / centre spacing (default: 0.96).')
    parser.add_argument('--rotation', type=float, default=None, help='X tick angle; default 90 for >18 stages, otherwise 60.')
    parser.add_argument('--show-values', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Only rewrite explicitly planned plot/source files.')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    require(len(set(args.species)) == len(args.species), 'Duplicate species')
    require((args.height is None or (math.isfinite(args.height) and args.height > 0)) and
            (args.width is None or (math.isfinite(args.width) and args.width > 0)) and
            (args.rotation is None or 0 <= args.rotation <= 90), 'Invalid figure dimensions/rotation')
    require(1 <= args.columns <= 8 and 0 < args.bar_width <= 1, 'Invalid columns/bar width')
    root, out = args.run_dir.resolve(), args.out_dir.resolve()
    require(out != root and root not in out.parents and out not in root.parents, 'Output must be separate from input results')
    scales = list(METRICS) if args.scale == 'both' else [args.scale]
    groups, receipts, excluded = defaultdict(list), [], []
    for species in args.species:
        identities = []
        for scale in scales:
            rows, receipt = read_metrics(root, species, scale, args.human_labels)
            receipts.append(receipt)
            identities.append({(r['target_column'], r['output_index_0based'], r['n_valid']) for r in rows})
            for row in rows:
                if is_wholebrain(row['organ']):
                    excluded.append(row)
                else:
                    groups[scale, species].append(row)
            require(groups[scale, species], f'No organs remain for {species}/{scale}')
        require(len(identities) == 1 or identities[0] == identities[1], f'Scale task mismatch: {species}')
    groups = dict(sorted(groups.items(), key=lambda item: (scales.index(item[0][0]), args.species.index(item[0][1]))))
    for rows in groups.values():
        rows.sort(key=lambda r:(organ_key(r['organ']), r['_sort']))
        counts = Counter()
        for row in rows:
            counts[row['organ']] += 1
            row['stage_order'] = counts[row['organ']]
    source = [r for rows in groups.values() for r in rows]
    limits = {s: [-1., 1.] if any(r['pearson_r'] is not None and r['pearson_r'] < 0 for r in source if r['activity_scale'] == s) else [0., 1.] for s in scales}
    planned = {}
    for key in groups:
        scale, species = key
        planned[key] = out/scale/f'{species}_all_organs_pearson.pdf'
    extras = [out/name for name in ('pearson_plot_data.tsv', 'excluded_targets.tsv', 'plot_manifest.tsv', 'plot_run.json')]
    for path in [*planned.values(), *extras]:
        require(path.resolve().is_relative_to(out), f'Output escapes requested folder: {path}')
        require(not path.is_symlink(), f'Refusing symlink output: {path}')
        require(not path.exists() or args.overwrite, f'Output exists; choose another --out-dir or use --overwrite: {path}')
    print(json.dumps({'source_run':str(root), 'output':str(out), 'species':args.species,
                      'n_figures':len(groups), 'task_scale_rows':len(source), 'formats':['pdf'],
                      'excluded_task_scale_rows':len(excluded), 'columns':args.columns,
                      'legend_columns':2, 'bar_width':args.bar_width, 'share_y':'row',
                      'ylabel':YLABEL, 'xlabel':'', 'grid':False,
                      'organ_colors':ORGAN_COLORS, 'human_labels':args.human_labels}, ensure_ascii=False, indent=2), flush=True)
    for (scale,species), rows in groups.items():
        print(f'{scale}/{species}: {dict(Counter(r["organ"] for r in rows))}', flush=True)
    if args.check_only:
        print('PREFLIGHT_OK: no files written.'); return 0
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                               'axes.grid':False,'axes.spines.top':False,'axes.spines.right':False,
                               'axes.linewidth':.7,'savefig.facecolor':'white'})
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for key, rows in groups.items():
        dest = planned[key]
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = make_figure(rows, limits[key[0]], args)
        try:
            for i, ax in enumerate(axes):
                first = i % min(args.columns, len(axes)) == 0
                require(ax.get_xlabel() == '' and ax.get_ylabel() == (YLABEL if first else ''), 'Axis label regression')
                require(not any(x.get_visible() for x in [*ax.get_xgridlines(),*ax.get_ygridlines()]), 'Grid regression')
            fig.savefig(dest, format='pdf', bbox_inches='tight', metadata={
                'Title':f'{key[1]} | All organs (excluding Wholebrain) | {key[0]}',
                'Subject':'Pearson across held-out promoters per original developmental timepoint; one checkpoint; no uncertainty bars.'})
        finally:
            plt.close(fig)
        manifest.append(dict(species=key[1], activity_scale=key[0], n_bars=len(rows),
            n_organs=len(axes), organs=';'.join(dict.fromkeys(r['organ'] for r in rows)), best_epoch=rows[0]['best_epoch'],
            pdf_file=str(dest.relative_to(out)), sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    write_tsv(out/'pearson_plot_data.tsv', source, FIELDS)
    write_tsv(out/'excluded_targets.tsv', excluded, tuple(f for f in FIELDS if f != 'stage_order'))
    write_tsv(out/'plot_manifest.tsv', manifest, tuple(manifest[0]))
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_run=str(root),
        species=args.species, scales=scales, n_figures=len(groups), figure_formats=['pdf'],
        n_panels=sum(len({r['organ'] for r in rows}) for rows in groups.values()),
        columns=args.columns, legend_columns=2, bar_width=args.bar_width, share_y='row',
        excluded_organ='Wholebrain (case-insensitive; separators ignored)',
        excluded_task_scale_rows=len(excluded),
        sources=receipts, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__, organ_colors=ORGAN_COLORS, color_aliases=COLOR_ALIASES,
        ylabel=YLABEL, xlabel='', grid=False, human_labels=args.human_labels, y_limits=limits,
        aggregation='None; one bar per original organ-stage target. Colour aliases never merge organs.',
        uncertainty='Single-run point estimates; no invented error bars or significance tests.',
        metric='Pearson across held-out promoters within one organ-stage target, not across stages.',
        ordering='Prenatal before postnatal; numeric ages; midpoint for ranges; Adult last.',
        human_label_policy='Exact source stage strings for every Human target; no age categories or bins.',
        label_caveats=sorted({r['label_note'] for r in source if r['label_note']}),
        status_counts=dict(Counter(r['pearson_status'] for r in source)))
    (out/'plot_run.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'DONE: {len(groups)} species-level PDF figures -> {out}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1)
