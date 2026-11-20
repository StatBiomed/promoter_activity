#!/usr/bin/env python3
"""Promoter counts above the redo8 8.4 GO heatmaps, with exact column alignment.

Python/matplotlib only. Default: seven Mouse organs + Human/Rat Liver, C10.
Run: python plot_cluster_promoter_counts.py
Read-only preflight: python plot_cluster_promoter_counts.py --check-only
Offline numerical tests: python plot_cluster_promoter_counts.py --self-test

COUNT: ALL hard-assigned dynamic MP promoters in each Mfuzz cluster, not unique
genes and not only high-membership/core promoters. Counts are descriptive totals,
so there are no error bars, inferential tests, or percentage labels.

ALIGNMENT: reuse the 8.4 display order, full PDF page width and matrix left edge.
The plotting box is exactly 2.65 inches wide, xlim=(0,10), and bar centers are
0.5,1.5,...9.5, matching the GO heatmap's cell centers. Keep PDF page boundaries
and use the SAME scaling factor when placing the two PDFs above/below each other.
Do not crop each figure independently. No bbox_inches='tight' on export.

STYLE: light-gray #C9CBCA bars, dark-gray #232325 outlines, 6.5 pt tick labels
(inherited from 8.4 by default), y label 2 pt larger, four-sided axes box.
Each target gets only one vector PDF and one display-ordered count TSV.
A root manifest records source hashes, exact page/axes/column positions in points.
No upstream files are modified; existing output directories are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parent
REDO = ROOT.parent
TARGETS = ['Mouse_' + x for x in ['Brain', 'Cerebellum', 'Heart', 'Kidney', 'Liver',
           'Ovary', 'Testis']] + ['Human_Liver', 'Rat_Liver']
# Explicit geometry contract for the existing 8.4 writer; fail if it changes.
SUPPORTED_GO_SHA256 = 'f8b1b83040d64469a9c93ebdd98fb5952c82423bf8cec294cd1b1f6cd88711e4'
MATRIX_WIDTH_IN = 2.65


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def record(path, sources, expected=None):
    require(path.is_file(), f'Missing input: {path}')
    digest = sha256(path)
    require(expected is None or digest == expected, f'Changed/stale source: {path}')
    sources[str(path.resolve())] = digest
    return digest


def checked_output(root, path, manifest, sources):
    key = str(path.relative_to(root))
    require(key in manifest['output_sha256'], f'File absent from upstream manifest: {path}')
    return record(path, sources, manifest['output_sha256'][key])


def count_clusters(annotations, order):
    require({'promoterId', 'cluster', 'geneId', 'membership'} <= set(annotations), 'Mfuzz annotation schema mismatch')
    require(annotations.promoterId.notna().all() and annotations.promoterId.is_unique, 'Duplicate/missing promoter IDs')
    cluster = pd.to_numeric(annotations.cluster, errors='raise')
    require(np.isfinite(cluster).all() and (cluster == cluster.astype(int)).all(), 'Invalid cluster labels')
    require(len(order) == len(set(order)) and set(cluster) == set(order), 'Missing/duplicate/mismatched clusters')
    require(annotations.membership.between(0, 1).all(), 'Invalid fuzzy membership')
    require(annotations.geneId.notna().all(), 'Missing gene IDs')
    frame = annotations.assign(cluster=cluster.astype(int))
    counts = frame.groupby('cluster').agg(promoter_n=('promoterId', 'size'),
                                         gene_n=('geneId', 'nunique')).reindex(order).reset_index()
    counts.insert(0, 'display_position', np.arange(1, len(order) + 1))
    require(int(counts.promoter_n.sum()) == len(annotations), 'Promoter count conservation failed')
    return counts


def heatmap_geometry(terms, go_manifest, group):
    """Replay the VERIFIED 8.4 layout without importing/running the enrichment code."""
    require('label' in terms and terms.label.notna().all(), 'Missing GO display labels')
    require(len(terms) > 0, '8.4 has no heatmap rows; cannot align to a nonexistent matrix')
    params = go_manifest['parameters']
    fontsize = float(params['font_size'])
    fig = plt.figure(figsize=(float(params['figure_width']), float(params['figure_height'])), dpi=100)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    font = FontProperties(family='DejaVu Sans', size=fontsize)
    label_width = max(renderer.get_text_width_height_descent(str(s), font, False)[0] / 100 for s in terms.label)
    plt.close(fig)
    left = max(1.95, label_width + .12)
    width = max(float(params['figure_width']), left + MATRIX_WIDTH_IN + .70)
    height = max(float(params['figure_height']), .42 + len(terms) * .13)
    require(np.allclose([width, height], group['figure']['inches'], rtol=0, atol=1e-10),
            'Cannot reproduce 8.4 page geometry; font/runtime/layout changed. Do not align approximately.')
    require(np.isclose(fontsize, group['figure']['tick_font_pt']), '8.4 recorded font mismatch')
    return dict(page_width_in=width, heatmap_page_height_in=height, axes_left_in=left,
                axes_width_in=MATRIX_WIDTH_IN, tick_font_pt=fontsize,
                centers_page_pt=[72 * (left + MATRIX_WIDTH_IN * (i + .5) / 10) for i in range(10)],
                boundaries_page_pt=[72 * (left + MATRIX_WIDTH_IN * i / 10) for i in range(11)])


def prepare(a):
    sources = {str(Path(__file__).resolve()): sha256(__file__)}
    mf_manifest_path = a.mfuzz / 'output/run_manifest.json'
    go_root = a.go / 'output'
    go_manifest_path = go_root / 'run_manifest.json'
    for p in [mf_manifest_path, go_manifest_path]:
        record(p, sources)
    mf = json.loads(mf_manifest_path.read_text())
    go = json.loads(go_manifest_path.read_text())
    require(mf['status'] == go['status'] == 'complete', 'Run 8.2 and 8.4 successfully first')
    require(mf['clusters'] == 10, 'This script expects the redo8 C10 results')
    go_script = a.go / 'run_cluster_gene_enrichment.py'
    record(go_script, sources, SUPPORTED_GO_SHA256)
    require(go['source_sha256'].get(str(go_script.resolve())) == SUPPORTED_GO_SHA256,
            '8.4 results were made by a different layout writer')
    require(go['source_sha256'].get(str(mf_manifest_path.resolve())) == sources[str(mf_manifest_path.resolve())],
            '8.4 refers to a different Mfuzz run')
    require(go['runtime']['matplotlib'] == matplotlib.__version__,
            f"Use the 8.4 matplotlib runtime ({go['runtime']['matplotlib']}) to reproduce exact label metrics")
    targets = a.targets if a.targets else list(go['targets'])
    require(len(targets) == len(set(targets)) and set(targets) <= set(TARGETS), 'Invalid/duplicate target list')
    groups = {g['target']: g for g in go['groups']}
    result = []
    for target in targets:
        require(target in mf['targets'] and target in groups, f'Target absent from upstream results: {target}')
        organ = target.split('_', 1)[1]
        ann_path = a.mfuzz / 'output' / target / f'{organ}_promoter_cluster_annotations.tsv'
        mf_count_path = a.mfuzz / 'output' / target / f'{organ}_cluster_summary.tsv'
        checked_output(a.mfuzz, ann_path, mf, sources)
        checked_output(a.mfuzz, mf_count_path, mf, sources)
        require(go['source_sha256'].get(str(ann_path.resolve())) == sources[str(ann_path.resolve())],
                f'{target}: GO and current Mfuzz assignments differ')
        paths = {name: go_root / target / name for name in ['heatmap_scores.tsv', 'heatmap_terms.tsv',
                  'cluster_summary.tsv', f'{target}_C10_gene_enrichment_heatmap.pdf']}
        for p in paths.values():
            checked_output(go_root, p, go, sources)
        matrix = pd.read_csv(paths['heatmap_scores.tsv'], sep='\t', index_col=0)
        order = [int(x) for x in matrix.columns]
        require(len(order) == 10 and set(order) == set(range(1, 11)), f'{target}: invalid heatmap column order')
        require(order == groups[target]['cluster_order'], f'{target}: GO manifest/order mismatch')
        # Also tie the chosen order back to the stage-activity heatmap in 8.3.
        stage_matrix_path = a.heatmap / 'output' / target / f'{target}_C10_heatmap_matrix.tsv'
        require(str(stage_matrix_path.resolve()) in go['source_sha256'], f'{target}: 8.3 file not recorded by 8.4')
        record(stage_matrix_path, sources, go['source_sha256'][str(stage_matrix_path.resolve())])
        old_order = [int(x) for x in pd.read_csv(stage_matrix_path, sep='\t', nrows=0).columns[1:]]
        require(order == old_order, f'{target}: 8.3 and 8.4 column orders differ')
        terms = pd.read_csv(paths['heatmap_terms.tsv'], sep='\t')
        require(len(matrix) == len(terms) == groups[target]['displayed_terms'], f'{target}: missing GO rows')
        geometry = heatmap_geometry(terms, go, groups[target])
        ann = pd.read_csv(ann_path, sep='\t', dtype={'promoterId': str, 'geneId': str})
        require(ann.padj_BH.notna().all() and ann.padj_BH.between(0, .01, inclusive='left').all(),
                f'{target}: contains non-dynamic promoters')
        counts = count_clusters(ann, order)
        mf_counts = pd.read_csv(mf_count_path, sep='\t').set_index('cluster')
        go_counts = pd.read_csv(paths['cluster_summary.tsv'], sep='\t')
        require(go_counts.groupby('cluster').assigned_promoters.nunique().eq(1).all(), f'{target}: inconsistent GO counts')
        mapped_go = go_counts.groupby('cluster').assigned_promoters.first()
        np.testing.assert_array_equal(counts.promoter_n, mf_counts.loc[order, 'assigned_n'])
        np.testing.assert_array_equal(counts.gene_n, mf_counts.loc[order, 'gene_n'])
        np.testing.assert_array_equal(counts.promoter_n, mapped_go.loc[order])
        require(int(counts.promoter_n.sum()) == groups[target]['assigned_promoters'], f'{target}: GO total differs')
        result.append(dict(target=target, counts=counts, geometry=geometry))
    return result, sources


def draw(group, path, a):
    g, counts = group['geometry'], group['counts']
    fontsize = g['tick_font_pt']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'pdf.compression': 0, 'text.usetex': False})
    bottom, top = .30, .09
    height = bottom + a.axes_height + top
    fig = plt.figure(figsize=(g['page_width_in'], height), dpi=100)
    ax = fig.add_axes([g['axes_left_in'] / g['page_width_in'], bottom / height,
                       g['axes_width_in'] / g['page_width_in'], a.axes_height / height])
    centers = np.arange(10) + .5
    ax.bar(centers, counts.promoter_n.to_numpy(), width=.80,
           color='#C9CBCA', edgecolor='#232325', linewidth=.45, align='center', rasterized=False)
    ax.set_xlim(0, 10)
    maximum = int(counts.promoter_n.max())
    y_ticks = MaxNLocator(nbins=5, integer=True, steps=[1, 2, 2.5, 5, 10]).tick_values(0, max(1, maximum) * 1.05)
    y_ticks = y_ticks[(y_ticks >= 0) & np.isclose(y_ticks, np.round(y_ticks))]
    upper = float(y_ticks[-1]) if len(y_ticks) and y_ticks[-1] > maximum else float(maximum + 1)
    ax.set_ylim(0, upper)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.set_xticks(centers, labels=counts.cluster.astype(str).tolist())
    ax.tick_params(axis='both', labelsize=fontsize, width=.6, length=2, pad=2)
    ax.tick_params(axis='x', top=False, bottom=True, labeltop=False, labelbottom=True)
    ax.set_ylabel('Number of Promoter', fontsize=fontsize + 2, labelpad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#232325')
        spine.set_linewidth(.6)
    ax.grid(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        bounds = text.get_window_extent(renderer)
        require(bounds.x0 >= 0 and bounds.y0 >= 0 and bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height,
                f"Clipped bar-plot label: {text.get_text()}")
    actual_left, _, actual_width, _ = ax.get_position().bounds
    actual_centers = [float(ax.transData.transform((x, 0))[0] / fig.dpi * 72) for x in centers]
    np.testing.assert_allclose(actual_width * fig.get_figwidth(), g['axes_width_in'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual_left * fig.get_figwidth(), g['axes_left_in'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual_centers, g['centers_page_pt'], rtol=0, atol=1e-10)
    alignment = dict(target=group['target'], page_width_in=g['page_width_in'], page_height_in=height,
                     axes_left_in=g['axes_left_in'], axes_width_in=g['axes_width_in'], axes_height_in=a.axes_height,
                     cluster_order=counts.cluster.astype(int).tolist(), centers_page_pt=actual_centers,
                     heatmap_boundaries_page_pt=g['boundaries_page_pt'], tick_font_pt=fontsize,
                     ylabel_font_pt=fontsize + 2, fill='#C9CBCA', outline='#232325', bar_width_cell=.8,
                     total_promoters=int(counts.promoter_n.sum()), y_axis_max=upper,
                     vector_bars=True, tight_bbox=False)
    # Preserve the page box. Independently tight-cropping destroys the alignment.
    fig.savefig(path, format='pdf', metadata=dict(Title=f"{group['target']} C10 promoter counts",
                Subject='All assigned dynamic MP promoters, aligned to 8.4 GO heatmap columns',
                Keywords=json.dumps(alignment)))
    plt.close(fig)
    return alignment


def self_test():
    a = pd.DataFrame(dict(promoterId=['P1', 'P2', 'P3', 'P4'], cluster=[2, 2, 1, 2],
                         geneId=['G1', 'G1', 'G1', 'G2'], membership=[.99, .15, .4, .6]))
    c = count_clusters(a, [2, 1])
    assert c.cluster.tolist() == [2, 1] and c.promoter_n.tolist() == [3, 1]
    assert c.gene_n.tolist() == [2, 1] and c.promoter_n.sum() == 4
    duplicate = pd.concat([a, a.iloc[[0]]], ignore_index=True)
    try:
        count_clusters(duplicate, [2, 1])
        raise AssertionError('Duplicate promoter accepted')
    except ValueError:
        pass
    try:
        count_clusters(a, [1, 3])
        raise AssertionError('Mismatched order accepted')
    except ValueError:
        pass
    left = 2.46375
    centers = np.array([72 * (left + MATRIX_WIDTH_IN * (i + .5) / 10) for i in range(10)])
    bounds = np.array([72 * (left + MATRIX_WIDTH_IN * i / 10) for i in range(11)])
    np.testing.assert_allclose(centers, (bounds[:-1] + bounds[1:]) / 2)
    print('SELF-TEST PASS: promoter not gene/core count, conservation, order, duplicate rejection, exact cell centers')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--targets', nargs='+', choices=TARGETS, help='Default: every target in completed 8.4 manifest')
    p.add_argument('--mfuzz', type=Path, default=REDO / '8.2.run_mfuzz')
    p.add_argument('--heatmap', type=Path, default=REDO / '8.3.plot_heatmap')
    p.add_argument('--go', type=Path, default=REDO / '8.4.GO_analysis')
    p.add_argument('--output', type=Path, default=ROOT / 'output')
    p.add_argument('--axes-height', type=float, default=2.18, help='Bar plotting-box height in inches; width always matches 8.4')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--self-test', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    require(a.axes_height >= 1., 'Plotting-box height must be at least 1 inch')
    groups, sources = prepare(a)
    for group in groups:
        g, c = group['geometry'], group['counts']
        print(f"CHECK {group['target']}: {int(c.promoter_n.sum())} promoters, "
              f"order={c.cluster.tolist()}, axes width={g['axes_width_in']} in, "
              f"left={g['axes_left_in']} in, PDF width={g['page_width_in']} in", flush=True)
    if a.check_only:
        print('CHECK-ONLY PASS: no files written.')
        return 0
    require(not a.output.exists(), f'Output exists; choose a NEW --output directory, never overwritten: {a.output}')
    for upstream in [a.mfuzz, a.heatmap, a.go]:
        require(not a.output.resolve().is_relative_to(upstream.resolve()), 'Cannot write inside upstream directories')
    for p, digest in sources.items():
        require(sha256(p) == digest, f'Source changed during preflight: {p}')
    a.output.mkdir(parents=True)
    manifest = dict(status='running', started_utc=now(), targets=[g['target'] for g in groups],
                    count_scope='ALL hard-assigned dynamic MP promoters; no gene deduplication or membership filtering',
                    interpretation='descriptive counts, no error bars or statistical tests',
                    alignment='same PDF page width, matrix left edge, matrix width and column centers as 8.4; no tight cropping',
                    parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                    runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__),
                    source_sha256=sources, groups=[], output_sha256={})
    mp = a.output / 'run_manifest.json'
    write_json(mp, manifest)
    try:
        for group in groups:
            target = group['target']
            out = a.output / target
            out.mkdir()
            group['counts'].to_csv(out / f'{target}_C10_promoter_counts.tsv', sep='\t', index=False)
            layout = draw(group, out / f'{target}_C10_promoter_counts_barplot.pdf', a)
            manifest['groups'].append(layout)
            print('DONE', target, flush=True)
        for p, digest in sources.items():
            require(sha256(p) == digest, f'Source changed during export: {p}')
        manifest['output_sha256'] = {str(p.relative_to(a.output)): sha256(p) for p in sorted(a.output.rglob('*'))
                                     if p.is_file() and p != mp}
        manifest.update(status='complete', finished_utc=now(), upstream_unchanged=True)
    except Exception as error:
        manifest.update(status='failed', finished_utc=now(), error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(mp, manifest)
    print('COMPLETE:', a.output)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(2)
