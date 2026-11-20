#!/usr/bin/env python3
"""Mouse liver HNF4A/E2F4 developmental gene-expression rainclouds (PDF only).

Run with Python >=3.9, numpy, scipy and matplotlib. Defaults read the ORIGINAL
Mouse RPKM matrix, not human-ortholog-converted expression or bigWig coverage.
Each dot is one observed liver RNA-seq library; no sample means are treated as
replicates. Stage sample counts are read from the matrix, not inferred from BAMs.
Current source has 57 samples: E13.5 n=5; all other stages n=4.

Figure contract: descriptive temporal mRNA-expression distributions, one TF per
5 x 4 inch page, common y-axis across TFs, chronological equally spaced stages.
These cross-sectional samples are NOT connected as individual trajectories.
Cloud = Scott-bandwidth Gaussian KDE, clipped to observed min/max and normalized
to equal maximum width per stage. With n=4-5, density shape is descriptive only.
Box = median and IQR; whiskers = observed min/max (all dots are displayed).
No significance tests, protein-activity claims or causal claims are made.
HNF4A #F16B6F; E2F4 #AACD6E. Ticks/legend 12 pt, axis labels 14 pt.
Only two PDF files are saved, with editable TrueType text and uncompressed PDF
streams. Full sample values and source hash are embedded in PDF metadata and
printed to stdout for traceability; no data/metrics sidecar files are generated.

Usage:
  python plot_hnf4a_e2f4_development_raincloud.py
  python plot_hnf4a_e2f4_development_raincloud.py --check-only
  python plot_hnf4a_e2f4_development_raincloud.py --scale log2 --output-dir NEW_DIR
  # Existing PDFs are preserved unless --overwrite is explicitly supplied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

BASE = Path('/mnt/ruiyanhou/nfs_share2')
DEFAULT_EXPRESSION = BASE / 'RNA_seq_organ_species/gene_expression/Mouse_rpkm.txt'
DEFAULT_GTF = BASE / 'RNA_seq_organ_species/ref_annotation/Mus_musculus.GRCm38.77.gtf'
STAGES = tuple([f'e{i}.5' for i in range(10, 19)] + ['P0', 'P3', 'P14', 'P28', 'P63'])
LABELS = tuple([f'E{i}.5' for i in range(10, 19)] + ['P0', 'P3', '2 wpb', '4 wpb', '9 wpb'])
TARGETS = {
    'HNF4A': ('ENSMUSG00000017950', 'Hnf4a', '#F16B6F'),
    'E2F4': ('ENSMUSG00000014859', 'E2f4', '#AACD6E'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_gene_ids(gtf):
    """Verify both stable IDs against the mouse gene annotation, without aliases."""
    wanted = {v[0]: v[1] for v in TARGETS.values()}
    found = {}
    with gtf.open() as handle:
        for line in handle:
            if line.startswith('#') or '\tgene\t' not in line:
                continue
            fields = line.rstrip('\n').split('\t')
            attrs = dict(re.findall(r'(\w+) "([^"]+)";', fields[8]))
            gid = attrs.get('gene_id', '').split('.')[0]
            if gid not in wanted:
                continue
            require(gid not in found, f'Duplicate GTF gene: {gid}')
            require(attrs.get('gene_name') == wanted[gid], f'Gene-symbol mismatch: {gid}')
            require(attrs.get('gene_biotype', attrs.get('gene_type')) == 'protein_coding',
                    f'Unexpected biotype: {gid}')
            found[gid] = attrs['gene_name']
    require(found == wanted, 'GTF does not uniquely validate both mouse TF genes')
    return found


def load_expression(path):
    """Read quoted space-separated original data; validate samples and targets."""
    targets = {v[0]: k for k, v in TARGETS.items()}
    observed = {}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.reader(handle, delimiter=' ', skipinitialspace=True)
        header = next(reader)
        require(header[0] == 'Names', 'Expected original Mouse RPKM matrix: Names header')
        require(len(header) == len(set(header)), 'Duplicate expression sample columns')
        indices = [i for i, name in enumerate(header) if name.startswith('Liver.')]
        samples = []
        for i in indices:
            match = re.fullmatch(r'Liver\.(e\d+\.5|P\d+)\.(\d+)', header[i])
            require(match is not None, f'Unrecognized liver sample: {header[i]}')
            stage, replicate = match.groups()
            require(stage in STAGES, f'Unexpected liver stage: {stage}')
            samples.append(dict(sample=header[i], stage=stage, replicate=int(replicate)))
        require(set(s['stage'] for s in samples) == set(STAGES), 'Missing liver developmental stages')
        for row in reader:
            if not row:
                continue
            gid = re.sub(r'\.\d+$', '', row[0])
            if gid not in targets:
                continue
            require(len(row) == len(header), f'Invalid row width: {gid}')
            require(gid not in observed, f'Duplicate expression gene after version stripping: {gid}')
            values = np.array([float(row[i]) for i in indices], dtype=float)
            require(np.isfinite(values).all() and (values >= 0).all(), f'Invalid/missing RPKM: {gid}')
            observed[gid] = values
    require(set(observed) == set(targets), 'One or both target genes are missing from expression matrix')
    order = sorted(range(len(samples)), key=lambda i: (STAGES.index(samples[i]['stage']), samples[i]['replicate']))
    samples = [samples[i] for i in order]
    grouped, records = {}, []
    for tf, (gid, symbol, _) in TARGETS.items():
        values = observed[gid][order]
        grouped[tf] = []
        for stage in STAGES:
            stage_values = values[[s['stage'] == stage for s in samples]]
            require(len(stage_values) >= 2, f'{tf} {stage}: need >=2 real observations')
            grouped[tf].append(stage_values)
        records += [dict(TF=tf, gene_id=gid, gene_symbol=symbol, **sample, RPKM=float(value))
                    for sample, value in zip(samples, values)]
    return samples, grouped, records


def source_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def style():
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 14,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'pdf.fonttype': 42, 'pdf.compression': 0,
    })


def raincloud(ax, values, x, color, rng):
    """One bounded half-KDE, thin box and non-overlapping-side raw sample dots."""
    low, q1, median, q3, high = np.quantile(values, [0, .25, .5, .75, 1])
    if len(values) >= 3 and high - low > max(1., abs(high)) * 1e-10:
        grid = np.linspace(low, high, 160)
        density = gaussian_kde(values, bw_method='scott')(grid)
        widths = .32 * density / density.max()
        ax.fill_betweenx(grid, x - .075 - widths, x - .075,
                         facecolor=color, edgecolor=color, alpha=.55, linewidth=.55, zorder=2)
    else:
        # Constant values have no estimable KDE; retain the real box and dots.
        ax.plot([x - .36, x - .075], [median, median], color=color, lw=1, zorder=2)
    ax.plot([x, x], [low, high], color='#404040', linewidth=.65, zorder=3)
    ax.hlines([low, high], x - .04, x + .04, color='#404040', linewidth=.65, zorder=3)
    ax.add_patch(Rectangle((x - .05, q1), .10, q3 - q1, facecolor='white',
                           edgecolor='#404040', linewidth=.7, zorder=4))
    ax.plot([x - .05, x + .05], [median, median], color='#222222', linewidth=1.0, zorder=5)
    xs = x + .14 + rng.permutation(np.linspace(0, .16, len(values)))
    ax.scatter(xs, values, s=12, color=color, edgecolors='#444444', linewidths=.25,
               alpha=1, zorder=6, clip_on=True)


def make_figure(tf, groups, ylim, scale='raw'):
    style()
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=.17, right=.98, bottom=.27, top=.84)
    rng = np.random.default_rng(20261030)
    color = TARGETS[tf][2]
    for x, values in enumerate(groups):
        transformed = np.log2(values + 1) if scale == 'log2' else values
        raincloud(ax, transformed, x, color, rng)
    ax.set_xlim(-.55, len(STAGES) - .48)
    ax.set_ylim(0, ylim)
    ax.set_xticks(np.arange(len(STAGES)), labels=LABELS, rotation=90)
    ax.tick_params(axis='both', labelsize=12, width=.8, length=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel('Developmental stage', fontsize=14, labelpad=9)
    ax.set_ylabel('Expression (RPKM)' if scale == 'raw' else r'Expression [log$_2$(RPKM + 1)]',
                  fontsize=14, labelpad=6)
    ax.set_title(tf + ' | Mouse liver', fontsize=14, pad=17)
    counts = [len(v) for v in groups]
    note = 'n = 4 per stage; E13.5: n = 5' if counts == [5 if s == 'e13.5' else 4 for s in STAGES] else 'n by stage: ' + ', '.join(map(str, counts))
    fig.text(.575, .865, note, ha='center', va='center', fontsize=9, color='#555555')
    fig.text(.17, .025, 'Dots: samples; box: median / IQR; cloud: density', fontsize=8, color='#555555')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--expression', type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument('--gtf', type=Path, default=DEFAULT_GTF)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--scale', choices=['raw', 'log2'], default='raw')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    mapping = verify_gene_ids(args.gtf)
    samples, grouped, records = load_expression(args.expression)
    trace = dict(source=str(args.expression.resolve()), source_sha256=source_sha256(args.expression),
                 annotation=str(args.gtf.resolve()), mapping=mapping, unit='RPKM', scale=args.scale,
                 sample_count=len(samples), stage_counts={s: sum(r['stage'] == s for r in samples) for s in STAGES},
                 observations=records, cloud='Scott KDE clipped to sample range; equal maximum width',
                 box='median; IQR; min-max whiskers', tests='none', time_axis='ordered categories, not equal elapsed time')
    maxima = [max(v) for groups in grouped.values() for v in groups]
    maximum = max(maxima)
    transformed_max = float(np.log2(maximum + 1)) if args.scale == 'log2' else float(maximum)
    ylim = max(transformed_max * 1.08, 1.)
    trace['shared_ylim'] = [0, ylim]
    destinations = {tf: args.output_dir / f'Mouse_Liver_{tf}_development_raincloud_{args.scale}.pdf' for tf in TARGETS}
    if not args.check_only:
        for path in destinations.values():
            require(args.overwrite or not path.exists(), f'Preserving existing PDF: {path}; use --overwrite intentionally')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for tf, path in destinations.items():
            fig = make_figure(tf, grouped[tf], ylim, args.scale)
            metadata = {'Title': f'Mouse liver {tf} developmental expression', 'Author': 'Analysis script',
                        'Subject': json.dumps(trace, separators=(',', ':')),
                        'Keywords': 'raincloud;Mouse;Liver;RPKM;14 stages;observed samples',
                        'Creator': Path(__file__).name}
            with matplotlib.rc_context({'pdf.compression': 0}):
                fig.savefig(path, format='pdf', metadata=metadata)
            plt.close(fig)
        trace['outputs'] = {tf: str(p.resolve()) for tf, p in destinations.items()}
    print(json.dumps(trace, indent=2), flush=True)


if __name__ == '__main__':
    main()
