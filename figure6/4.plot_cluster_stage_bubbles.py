#!/usr/bin/env python3
"""Plot existing stage-by-cluster motif counts without rerunning clustering.

Each row = one Leiden cluster. Bubble AREA = raw positive parent motif count, NOT
TF activity or enrichment. Stages are categorical and chronologically ordered;
equal spacing is not elapsed developmental time. Blank cells mean zero.
Area scales linearly with count; no normalization, filtering or smoothing.
Only one 5 x 4 inch vector PDF is exported; provenance is embedded in the PDF.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/revise/train_model/redo/'
            'explain_development/mouse_liver_rerun')
REPORT = ROOT / ('4.motif_clustering/positive_stage_motifs_20261106T035245054681Z/'
                 'cluster_patterns_20261106T055337241457Z')
STAGES = ('E10.5', 'E11.5', 'E12.5', 'E13.5', 'E14.5', 'E15.5', 'E16.5',
          'E17.5', 'E18.5', 'P0', 'P3', '2wpb', '4wpb', '9wpb')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_counts(path):
    """Read the existing wide count table; never derive values from a screenshot."""
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require(reader.fieldnames is not None and len(reader.fieldnames) == len(STAGES)+1
                and set(reader.fieldnames) == {'leiden_cluster', *STAGES},
                'Expected leiden_cluster and exactly the 14 developmental-stage columns')
        records = {}
        for row in reader:
            label = row['leiden_cluster']
            require(re.fullmatch(r'[1-9]\d*', label or '') is not None, 'Invalid cluster ID')
            cluster = int(label)
            require(cluster not in records, f'Duplicate cluster {cluster}')
            values = [row[stage] for stage in STAGES]
            require(all(re.fullmatch(r'\d+', value or '') for value in values),
                    f'Cluster {cluster}: counts must be nonnegative integers, not missing values')
            records[cluster] = [int(value) for value in values]
    require(bool(records), 'Empty count table')
    clusters = sorted(records)
    return clusters, np.asarray([records[c] for c in clusters], dtype=np.int64)


def audit_against_patterns(path, clusters, counts):
    """Independently count the retained motif records to verify all table cells."""
    observed = np.zeros_like(counts)
    stage_index = {stage: i for i, stage in enumerate(STAGES)}
    cluster_index = {cluster: i for i, cluster in enumerate(clusters)}
    seen = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require({'motif_id', 'leiden_cluster', 'stage_label'}.issubset(reader.fieldnames or []),
                'Pattern table lacks required identity/stage/cluster columns')
        for row in reader:
            motif = row['motif_id']
            require(motif and motif not in seen, f'Missing or duplicate motif ID: {motif}')
            seen.add(motif)
            stage, cluster = row['stage_label'], int(row['leiden_cluster'])
            require(stage in stage_index and cluster in cluster_index, 'Unknown stage/cluster in pattern records')
            observed[cluster_index[cluster], stage_index[stage]] += 1
    require(np.array_equal(observed, counts), 'Count table disagrees with underlying motif records')
    return len(seen)


def read_cluster_colors(script, clusters):
    """Read the previous scatter palette as a literal, without executing its analysis."""
    tree = ast.parse(Path(script).read_text(encoding='utf-8'))
    matches = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'CLUSTER_COLORS' for t in node.targets)]
    require(len(matches) == 1, 'Expected one CLUSTER_COLORS assignment in the clustering script')
    palette = ast.literal_eval(matches[0])
    require(isinstance(palette, (tuple, list)) and len(palette) >= len(clusters), 'Insufficient cluster colors')
    colors = {cluster: palette[i] for i, cluster in enumerate(clusters)}
    require(all(mpl.colors.is_color_like(color) for color in colors.values()), 'Invalid cluster color')
    return colors


AREA_PER_MOTIF = 16.0  # scatter size in points^2; AREA, not radius, scales with count.
SIZE_LEGEND_COUNTS = (1, 5, 10, 13)


def build_figure(clusters, counts, colors):
    from matplotlib.lines import Line2D
    require(counts.shape == (len(clusters), len(STAGES)), 'Unexpected count matrix shape')
    require(np.isfinite(counts).all() and np.all(counts >= 0), 'Invalid counts')
    require(np.all(counts == np.floor(counts)), 'Expected integer motif counts')
    style = {'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
             'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 12,
             'ytick.labelsize': 12, 'legend.fontsize': 12, 'pdf.fonttype': 42,
             'pdf.compression': 0, 'savefig.bbox': None, 'axes.spines.top': False,
             'axes.spines.right': False, 'axes.linewidth': .8, 'legend.frameon': False}
    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.subplots_adjust(left=.13, right=.98, bottom=.28, top=.75)
        x = np.arange(len(STAGES))
        for row, (cluster, values) in enumerate(zip(clusters, counts)):
            mask = values > 0
            ax.scatter(x[mask], np.full(mask.sum(), row), s=AREA_PER_MOTIF*values[mask],
                       c=colors[cluster], edgecolors='none', linewidths=0, alpha=1,
                       rasterized=False, zorder=3, label=f'Cluster {cluster}')
        ax.set_xticks(x, STAGES, rotation=90)
        ax.set_yticks(np.arange(len(clusters)), [str(c) for c in clusters])
        ax.tick_params(axis='both', labelsize=12)
        for tick, cluster in zip(ax.get_yticklabels(), clusters):
            tick.set_color(colors[cluster])
        ax.set_xlim(-.5, len(STAGES)-.5)
        ax.set_ylim(len(clusters)-.5, -.5)  # Cluster 1 at the top; no row reordering.
        ax.set_xlabel('Developmental stage', fontsize=14)
        ax.set_ylabel('Cluster', fontsize=14)
        ax.set_axisbelow(True)
        ax.grid(axis='x', linewidth=.35, color='#eeeeee')
        ax.grid(axis='y', linewidth=.5, color='#e5e5e5')
        fig.suptitle('Mouse liver: positive motifs', fontsize=14, y=.98)
        handles = [Line2D([], [], linestyle='none', marker='o',
                          markersize=np.sqrt(AREA_PER_MOTIF*n),
                          markerfacecolor='#777777', markeredgewidth=0, label=str(n))
                   for n in SIZE_LEGEND_COUNTS]
        fig.legend(handles=handles, title='Motif count (blank = 0)', fontsize=12,
                   title_fontsize=12, ncol=len(handles), loc='upper center',
                   bbox_to_anchor=(.55, .91), frameon=False, borderaxespad=0,
                   handlelength=1.25, handletextpad=.4, columnspacing=.95, labelspacing=.35,
                   markerscale=1)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(),
                       *ax.get_yticklabels(), *fig.texts, *fig.legends]:
            bounds = artist.get_window_extent(renderer)
            require(bounds.x0 >= -1 and bounds.y0 >= -1 and bounds.x1 <= fig.bbox.x1+1
                    and bounds.y1 <= fig.bbox.y1+1, 'A label is outside the fixed 5x4 inch page')
        # Largest diameter must remain smaller than one categorical column spacing.
        spacing_pt = ax.get_window_extent(renderer).width / fig.dpi * 72 / len(STAGES)
        require(np.sqrt(AREA_PER_MOTIF*counts.max()) < spacing_pt,
                'Bubbles would overlap adjacent columns; reduce AREA_PER_MOTIF')
    return fig




def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', type=Path, default=REPORT/'cluster_stage_counts.tsv')
    parser.add_argument('--patterns', type=Path, default=REPORT/'all_cluster_patterns.tsv')
    parser.add_argument('--cluster-script', type=Path, default=ROOT/'4.cluster_stage_tfmodisco_motifs.py')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'4.motif_clustering/final_figure_small/cluster_stage_motif_bubbles.pdf')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require(args.output.suffix.lower() == '.pdf', 'Only PDF export is supported')
    require(not args.output.exists(), f'Refusing to overwrite existing output: {args.output}')
    clusters, counts = read_counts(args.counts)
    n = audit_against_patterns(args.patterns, clusters, counts)
    colors = read_cluster_colors(args.cluster_script, clusters)
    fig = build_figure(clusters, counts, colors)
    provenance = dict(stage_order=STAGES, cluster_ids=clusters, counts=counts.tolist(),
                      cluster_colors=colors, n_motifs=n, figsize_inches=[5,4],
                      bubble_area_per_motif_pt2=AREA_PER_MOTIF, zero_counts="blank",
                      size_legend_counts=SIZE_LEGEND_COUNTS,
                      source_files={str(p): sha256(p) for p in (args.counts, args.patterns, args.cluster_script)},
                      script_sha256=sha256(__file__), interpretation=__doc__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open('xb') as handle:
            with mpl.rc_context({'pdf.fonttype':42, 'pdf.compression':0, 'savefig.bbox':None}):
                with PdfPages(handle, metadata={'Title':'Mouse liver motif counts: bubble matrix',
                                               'Subject':'Raw positive motif counts, not TF activity or enrichment'}) as pdf:
                    pdf.attach_note(json.dumps(provenance, allow_nan=False))
                    pdf.savefig(fig, bbox_inches=None)
    finally:
        plt.close(fig)
    raw = args.output.read_bytes()
    require(re.search(rb'/MediaBox\s*\[\s*0\s+0\s+360\s+288\s*\]',raw) is not None, 'Wrong PDF size')
    require(b'/FontFile2' in raw and b'/Subtype /Image' not in raw, 'Expected editable vector PDF')
    print(json.dumps(dict(status='complete', output=str(args.output), n_motifs=n,
                          cluster_totals=dict(zip(clusters, counts.sum(axis=1).tolist())),
                          cluster_colors=colors, pdf_sha256=sha256(args.output))))


if __name__ == '__main__':
    main()

