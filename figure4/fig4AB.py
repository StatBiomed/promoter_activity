#!/usr/bin/env python3
"""Create exactly two final-size Mouse Liver promoter-usage boxplot PDFs.

Figure contract: describe the stage-level distributions for the existing 183
Group A / Group B promoter pairs, without assuming a directional conclusion.
One single quantitative panel per group; Python/matplotlib only.

Input pairs: 1.results/Mouse_Liver/Mouse_Liver_representative_promoter_pairs.tsv.
Activity source: 1.proactiv_result/Mouse/Mouse_Liver_<stage>_rowData.tsv.
usage = source promoter .mean / source gene .gene.mean, a ratio of condition
means, NOT replicate-level ratios, and NOT A/(A+B). No new promoter selection,
membership threshold, logarithm, pseudo-count, imputation or normalization.
0/positive stays 0; 0/0 stays NA, omitted only at that stage. A/B promoters of
one gene must have the same denominator and therefore the same missing stages.

Box: Q1-Q3, line: median, whiskers: observations within 1.5 IQR, dots: outliers.
n counts promoters with defined usage, not biological replicates. The same
promoters recur across stages. Stages are equally spaced chronological
categories. No significance tests are performed.

Final-figure-small: exact 5x4 inch canvas, 12 pt ticks, 14 pt y-axis label,
12 pt legend if present (no redundant legend here), editable TrueType text,
PDF compression off. No x-axis title. Stage names are retained and vertical
to avoid overlap on the fixed small canvas. A boxes: #E71D36; B: #2EC4B6.
No PNG/SVG/TSV/metrics sidecars are written; only the two PDFs on the server.
Source checks, sample counts and descriptive statistics are printed to stdout.

Run using a Python environment with numpy and matplotlib:
    python3 2.plot_mouse_liver_AB_usage_boxplots.py --check-only
    python3 2.plot_mouse_liver_AB_usage_boxplots.py
Outputs: 2.results/Mouse_Liver_group_A_usage_boxplot.pdf and Group B equivalent.
An existing output directory is never overwritten; use --output-dir to rerun.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo')
COLORS = {'A': '#E71D36', 'B': '#2EC4B6'}
CLUSTERS = {'A': {2, 4, 5}, 'B': {1, 3, 6, 7, 8, 9, 10}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tsv_rows(path, required):
    """Handle the extra unnamed leading index in R write.table exports."""
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader, [])
        require(header and len(set(header)) == len(header), f'{path}: invalid header')
        require(set(required) <= set(header), f'{path}: missing fields {set(required)-set(header)}')
        offset = None
        for line, row in enumerate(reader, 2):
            if not row:
                continue
            if offset is None:
                offset = len(row)-len(header)
                require(offset in (0, 1), f'{path}:{line}: invalid row width')
            require(len(row) == len(header)+offset, f'{path}:{line}: inconsistent row width')
            yield dict(zip(header, row[offset:]))


def stage_of(path):
    return path.name[len('Mouse_Liver_'):-len('_rowData.tsv')]


def stage_key(stage):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)(dpb|wpb)?', stage)
    require(match is not None, f'Unrecognized stage: {stage}')
    value, unit = match.groups()
    return (0, float(value)) if unit is None else (1, float(value)*(7 if unit == 'wpb' else 1))


def activity(raw, context):
    value = float(raw)
    require(math.isfinite(value) and value >= 0, f'{context}: invalid activity {raw!r}')
    return value


def load_pairs(path):
    required = ['geneId', 'promoterId_A', 'promoterId_B', 'cluster_A', 'cluster_B']
    pairs = sorted(tsv_rows(path, required), key=lambda r: r['geneId'])
    require(len(pairs) == 183 and len({r['geneId'] for r in pairs}) == 183,
            'Expected the existing 183 representative gene pairs')
    selected = {}
    for pair in pairs:
        require(pair['geneId'].startswith('ENSMUSG'), 'Expected Mouse Ensembl gene IDs')
        for group in ('A', 'B'):
            pid = pair[f'promoterId_{group}']
            require(pid and pid not in selected, f'Duplicate or empty promoter: {pid!r}')
            require(int(pair[f'cluster_{group}']) in CLUSTERS[group], 'Promoter in an unexpected cluster group')
            selected[pid] = pair['geneId']
    return pairs, selected


def calculate(pairs_path, activity_dir):
    paths = sorted(activity_dir.glob('Mouse_Liver_*_rowData.tsv'), key=lambda p: stage_key(stage_of(p)))
    require(len(paths) == 14, f'Expected 14 stages; found {len(paths)}')
    require(len({stage_key(stage_of(p)) for p in paths}) == len(paths), 'Duplicate stage time')
    sources = [pairs_path, *paths]
    stamps = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in sources}
    pairs, selected = load_pairs(pairs_path)
    stages = [stage_of(p) for p in paths]
    matrices = {g: np.full((len(pairs), len(stages)), np.nan) for g in ('A', 'B')}
    for j, path in enumerate(paths):
        stage = stages[j]
        pcol, gcol = f'Mouse_Liver_{stage}.mean', f'Mouse_Liver_{stage}.gene.mean'
        found = {}
        for row in tsv_rows(path, ['promoterId', 'geneId', pcol, gcol]):
            pid = row['promoterId']
            if pid not in selected:
                continue
            require(pid not in found and row['geneId'] == selected[pid], f'{path}: promoter/gene inconsistency: {pid}')
            found[pid] = (activity(row[pcol], f'{path}:{pid}'), activity(row[gcol], f'{path}:{pid}'))
        require(found.keys() == selected.keys(), f'{path}: missing selected promoters')
        for i, pair in enumerate(pairs):
            ga = found[pair['promoterId_A']][1]
            gb = found[pair['promoterId_B']][1]
            require(math.isclose(ga, gb, rel_tol=1e-12, abs_tol=0), f'{stage}: A/B gene denominator differs')
            for group in ('A', 'B'):
                p, g = found[pair[f'promoterId_{group}']]
                require(g > 0 or p == 0, f'{stage}: positive promoter activity / zero gene activity')
                if g > 0:
                    matrices[group][i, j] = p/g
    require(np.array_equal(np.isnan(matrices['A']), np.isnan(matrices['B'])), 'A/B missingness differs')
    provenance = [{'path': str(p), 'sha256': sha256(p)} for p in sources]
    require(all((p.stat().st_size, p.stat().st_mtime_ns) == stamps[p] for p in sources), 'Input changed while reading')
    require(set(paths) == set(activity_dir.glob('Mouse_Liver_*_rowData.tsv')), 'Stage inventory changed')
    summary = {}
    for group, matrix in matrices.items():
        summary[group] = []
        for j, stage in enumerate(stages):
            values = matrix[np.isfinite(matrix[:, j]), j]
            require(len(values) > 0, f'{group}:{stage}: no defined usage')
            q1, median, q3 = np.quantile(values, [.25, .5, .75])
            summary[group].append(dict(stage=stage, n=len(values), missing=183-len(values),
                zeros=int(np.sum(values == 0)), above_one=int(np.sum(values > 1)),
                q1=float(q1), median=float(median), q3=float(q3)))
    return pairs, stages, matrices, summary, provenance


def apply_style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Arial', 'Helvetica', 'DejaVu Sans') if f in installed), 'DejaVu Sans')
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [font],
        'font.size': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'axes.labelsize': 14, 'legend.fontsize': 12, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'savefig.facecolor': 'white'})
    return font


def make_figure(group, stages, matrix, summary, largest):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=.18, right=.985, bottom=.245, top=.91)
    values = [matrix[np.isfinite(matrix[:, j]), j] for j in range(len(stages))]
    color = COLORS[group]
    edge = tuple(np.array(mpl.colors.to_rgb(color))*.65)
    boxes = ax.boxplot(values, positions=range(1, len(stages)+1), widths=.58,
        patch_artist=True, whis=1.5, showfliers=True, manage_ticks=False,
        boxprops=dict(facecolor=color, edgecolor=edge, linewidth=.8),
        medianprops=dict(color='#263238', linewidth=1.0),
        whiskerprops=dict(color=edge, linewidth=.7), capprops=dict(color=edge, linewidth=.7),
        flierprops=dict(marker='o', markersize=1.5, markerfacecolor='#6E8085',
                        markeredgecolor='none', alpha=.5))
    for box, median_line, row in zip(boxes['boxes'], boxes['medians'], summary):
        require(math.isclose(float(median_line.get_ydata()[0]), row['median'], rel_tol=1e-12, abs_tol=1e-15),
                'Rendered median disagrees with input values')
        ys = box.get_path().vertices[:, 1]
        require(math.isclose(float(ys.min()), row['q1'], rel_tol=1e-12, abs_tol=1e-15) and
                math.isclose(float(ys.max()), row['q3'], rel_tol=1e-12, abs_tol=1e-15),
                'Rendered quartiles disagree with input values')
    ax.set_xlim(.4, len(stages)+.6)
    ax.set_ylim(-.025*largest, 1.13*largest)
    if largest == 1:
        ax.set_yticks([0, .25, .5, .75, 1], ['0', '0.25', '0.50', '0.75', '1.00'])
    labels = ['E'+stage if stage_key(stage)[0] == 0 else stage for stage in stages]
    ax.set_xticks(range(1, len(stages)+1), labels, rotation=90, fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Promoter usage', fontsize=14, labelpad=5)
    ax.set_title(f'Group {group}', fontsize=12, pad=9)
    for i, row in enumerate(summary, 1):
        ax.text(i, 1.075*largest, f'n=\n{row["n"]}', ha='center', va='center', fontsize=6.2,
                linespacing=1.0, color='#586D73')
    ax.yaxis.grid(True, color='#E8ECEE', linewidth=.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=12, length=3, width=.75)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.yaxis.label, ax.title, *ax.texts]
    for text in texts:
        bounds = text.get_window_extent(renderer)
        require(bounds.x0 >= -1 and bounds.y0 >= -1 and bounds.x1 <= fig.bbox.x1+1
                and bounds.y1 <= fig.bbox.y1+1, f'Clipped text on fixed canvas: {text.get_text()!r}')
    sample_boxes = [text.get_window_extent(renderer) for text in ax.texts]
    require(all(a.x1 < b.x0 for a, b in zip(sample_boxes, sample_boxes[1:])),
            'Adjacent sample-count annotations overlap on the fixed small canvas')
    return fig


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pairs', type=Path, default=folder/'1.results/Mouse_Liver/Mouse_Liver_representative_promoter_pairs.tsv')
    parser.add_argument('--activity-dir', type=Path, default=BASE/'1.proactiv_result/Mouse')
    parser.add_argument('--output-dir', type=Path, default=folder/'2.results')
    parser.add_argument('--check-only', action='store_true', help='Validate data without creating any output')
    args = parser.parse_args()
    if not args.check_only:
        require(not args.output_dir.exists(), f'Output exists; choose a new --output-dir: {args.output_dir}')
    pairs, stages, matrices, summary, sources = calculate(args.pairs, args.activity_dir)
    print(json.dumps(dict(gene_pairs=len(pairs), promoters_per_group=183, stages=stages,
        usage='Source promoter .mean / source gene .gene.mean; 0/0 -> NA; valid zeros retained',
        stage_summary=summary, source_files=sources), indent=2), flush=True)
    if args.check_only:
        return
    font = apply_style()
    largest = max(1.0, *(float(np.nanmax(matrix)) for matrix in matrices.values()))
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.AB_boxplots_', dir=args.output_dir.parent) as temporary:
        ready = Path(temporary)/'ready'
        ready.mkdir()
        for group in ('A', 'B'):
            fig = make_figure(group, stages, matrices[group], summary[group], largest)
            try:
                with mpl.rc_context({'pdf.compression': 0}):
                    fig.savefig(ready/f'Mouse_Liver_group_{group}_usage_boxplot.pdf', format='pdf')
            finally:
                plt.close(fig)
        require(not args.output_dir.exists(), 'Output appeared during plotting')
        os.rename(ready, args.output_dir)
    print(f'Saved exactly two PDFs to {args.output_dir}; font={font}; canvas=5x4 inches', flush=True)


if __name__ == '__main__':
    main()
