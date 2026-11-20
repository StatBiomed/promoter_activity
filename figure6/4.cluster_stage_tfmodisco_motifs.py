#!/usr/bin/env python3
"""Jointly cluster positive TF-MoDISco motifs from 14 Mouse Liver stages.

Reference-compatible defaults: PPM IC trim >0.2 bits (A/T=.27, C/G=.23),
all-vs-all memelite.tomtom with reverse complements, E-value row profiles,
PCA <=50 dimensions, t-SNE perplexity 15, and PCA-space weighted 15-NN
Leiden (RBConfiguration, resolution 1, seed 1234). t-SNE is NOT the clustering
input. Each parent positive pattern is one point; no subpatterns, negative
patterns, seqlet expansion, motif deduplication, or TF-name assignment.

Only complete, audited attempts selected by stage/status.json are used.
Inputs are read-only. Each execution creates a NEW result directory containing
ONLY two 5 x 4 inch vector PDFs (cluster-colored and stage-colored).
Reproducibility metadata are embedded in each PDF as a nonvisible text note.
Dependencies: memelite, numpy, pandas, h5py, sklearn, igraph, leidenalg,
matplotlib (already installed in the existing PY310 environment).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/revise/train_model/redo/"
            "explain_development/mouse_liver_rerun")
DEFAULT_BATCH = ROOT / "3.TFMoDISco_results/stage_IG_TFMoDISco_full_20261105T103236372075Z"
STAGES = dict(zip(
    ["10.5", "11.5", "12.5", "13.5", "14.5", "15.5", "16.5", "17.5", "18.5",
     "0dpb", "3dpb", "2wpb", "4wpb", "9wpb"],
    ["E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5", "E17.5", "E18.5",
     "P0", "P3", "2wpb", "4wpb", "9wpb"]))
VERSION = "mouse_liver_joint_positive_motif_clustering_final_small_v2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Motif:
    motif_id: str
    stage: str
    stage_label: str
    pattern: str
    source_hdf5: str
    ppm: np.ndarray
    original_length: int
    trim_start: int
    trim_end: int
    n_seqlets: int
    n_informative_positions: int


def normalize_ppm(ppm, motif_id):
    ppm = np.asarray(ppm, dtype=np.float64)
    require(ppm.ndim == 2 and ppm.shape[1] == 4 and len(ppm) > 0,
            f"{motif_id}: expected nonempty (L,4) sequence PPM, got {ppm.shape}")
    require(np.isfinite(ppm).all() and np.all(ppm >= 0), f"{motif_id}: invalid PPM")
    sums = ppm.sum(axis=1, keepdims=True)
    require(np.all(sums > 0), f"{motif_id}: zero-sum PPM row")
    return ppm / sums


def information_content(ppm, background):
    # Same stabilizer and expression as the reference implementation.
    eps = 1e-6
    return np.sum(ppm * np.log2((ppm + eps) / (background[None, :] + eps)), axis=1)


def trim_ppm(ppm, background, threshold, flank):
    informative = np.flatnonzero(information_content(ppm, background) > threshold)
    if informative.size == 0:
        # Preserve reference behavior; flag these motifs in metadata, never silently drop.
        return ppm, 0, len(ppm)
    start = max(0, int(informative[0]) - flank)
    end = min(len(ppm), int(informative[-1]) + 1 + flank)
    return ppm[start:end], start, end


def load_sources(batch):
    """Resolve authoritative attempts and audit input provenance before clustering."""
    batch = Path(batch).resolve(strict=True)
    manifest_path = batch / "batch_manifest.json"
    signature = read_json(manifest_path)["signature"]
    require(signature["version"] == "mouse_liver_top1500_stage_ig_tfmodisco_v1"
            and signature["mode"] == "full", "Expected a full stage-IG TF-MoDISco batch")
    plan = signature["tasks"]
    require([t["stage"] for t in plan] == list(STAGES), "Expected all 14 stages in chronological order")
    batch_status = read_json(batch / "status.json")
    require(batch_status.get("status") == "complete" and
            batch_status.get("completed_stages") == batch_status.get("total_stages") == 14,
            "All 14 upstream stages must be complete; incomplete batches are not silently skipped")
    sources = []
    for target in plan:
        stage, label = target["stage"], target["stage_label"]
        require(label == STAGES[stage] and target["target_column"] == f"Mouse_Liver_{stage}.mean",
                "Stage/target mismatch in batch plan")
        folder = batch / label
        require(not folder.is_symlink(), f"{label}: unexpected symlink stage folder")
        status_path = folder / "status.json"
        status = read_json(status_path)
        require(status.get("status") == "complete" and status["stage"] == stage
                and status["target_column"] == target["target_column"], f"{label}: incomplete/mismatched stage")
        attempt = Path(status["attempt"]).resolve(strict=True)
        require(attempt.parent == (folder / "attempts").resolve(strict=True)
                and re.fullmatch(r"attempt_\d{4,}", attempt.name), f"{label}: unsafe attempt path")
        worker = read_json(attempt / "worker_status.json")
        audit = read_json(attempt / "result_audit.json")
        require(worker.get("status") == "complete" and worker["stage"] == stage,
                f"{label}: worker is not complete for this stage")
        require(audit.get("status") == "passed" and audit == worker["audit"] == status["audit"]
                and audit["species"] == "Mouse" and audit["organ"] == "Liver",
                f"{label}: failed or inconsistent discovery audit")
        require(worker["output_sha256"] == status["output_sha256"], f"{label}: output hash records disagree")
        h5 = attempt / "modisco_results.h5"
        require(h5.resolve(strict=True).parent == attempt, f"{label}: HDF5 escapes its attempt")
        digest = sha256(h5)
        require(digest == status["output_sha256"][h5.name] == audit["hdf5_sha256"],
                f"{label}: HDF5 changed since discovery audit")
        preparation_path = folder / "preparation_manifest.json"
        prep, run = read_json(preparation_path), read_json(folder / "run_manifest.json")
        require(sha256(preparation_path) == run["preparation_manifest_sha256"],
                f"{label}: preparation metadata changed")
        require(run["mode"] == "full" and prep["preparation_status"] == "complete"
                and prep["target_type"] == "single_stage_raw_activity"
                and prep["species"] == "Mouse" and prep["organ"] == "Liver"
                and not prep["is_pilot_subset"] and prep["n_test_promoters"] == 1500
                and prep["discovery_promoters"] + prep["excluded_promoters"] == 1500,
                f"{label}: unexpected source cohort/target")
        for key in ("stage", "stage_label", "target_column", "output_index_0based"):
            require(prep[key] == target[key], f"{label}: preparation {key} mismatch")
        sources.append(dict(**target, hdf5=str(h5), hdf5_sha256=digest,
                            attempt=str(attempt), status_path=str(status_path),
                            status_sha256=sha256(status_path),
                            positive_motifs=int(audit["positive_motifs"]),
                            negative_motifs=int(audit["negative_motifs"]),
                            discovery_promoters=int(prep["discovery_promoters"])))
    return sources


def read_motifs(sources, background, threshold, flank):
    import h5py
    motifs = []
    for source in sources:
        count = 0
        with h5py.File(source["hdf5"], "r") as handle:
            for group, expected in (("pos_patterns", source["positive_motifs"]),
                                    ("neg_patterns", source["negative_motifs"])):
                names = list(handle[group]) if group in handle else []
                require(all(re.fullmatch(r"pattern_\d+", n) for n in names),
                        f"Unexpected parent pattern names: {source['stage_label']}/{group}")
                require(len(names) == expected, f"{source['stage_label']}: motif count differs from audit")
            if "pos_patterns" not in handle:
                continue
            for name in sorted(handle["pos_patterns"], key=lambda n: int(n.split('_')[-1])):
                pattern = handle["pos_patterns"][name]
                motif_id = f"{source['stage_label']}__pos__{name}"
                ppm = normalize_ppm(pattern["sequence"][:], motif_id)
                trimmed, start, end = trim_ppm(ppm, background, threshold, flank)
                values = np.asarray(pattern["seqlets/n_seqlets"][()]).reshape(-1)
                require(values.size == 1 and values[0] > 0 and int(values[0]) == values[0],
                        f"{motif_id}: invalid n_seqlets")
                nseqlets = int(values[0])
                require(len(pattern["seqlets/start"]) == nseqlets, f"{motif_id}: inconsistent seqlets")
                motifs.append(Motif(motif_id, source["stage"], source["stage_label"], name,
                                    source["hdf5"], trimmed, len(ppm), start, end, nseqlets,
                                    int(np.sum(information_content(ppm, background) > threshold))))
                count += 1
        require(count == source["positive_motifs"], "Loaded positive motif count mismatch")
    require(len({m.motif_id for m in motifs}) == len(motifs), "Duplicate motif IDs")
    return motifs


def calculate_tomtom_matrices(motifs, threads):
    from memelite import tomtom
    matrices = [np.ascontiguousarray(m.ppm.T) for m in motifs]
    # Explicitly freeze the installed reference defaults; only resource usage differs.
    result = tomtom(Qs=matrices, Ts=matrices, n_nearest=None, n_score_bins=100,
                    n_median_bins=1000, n_target_bins=100, n_cache=100,
                    reverse_complement=True, n_jobs=threads)
    raw_pvalues = np.asarray(result[0], dtype=np.float64)
    require(raw_pvalues.shape == (len(motifs), len(motifs)) and np.isfinite(raw_pvalues).all(),
            "Tomtom returned a nonfinite or incorrectly shaped matrix")
    # Keep raw values for audit. Reproduce the reference floor rather than dropping comparisons.
    pvalues = np.clip(raw_pvalues, 1e-15, 1.0)
    evalues = pvalues * len(motifs)
    similarity = -np.log10(np.clip(evalues, np.finfo(np.float64).tiny, 1.0))
    np.fill_diagonal(similarity, np.max(similarity))
    arrays = dict(tomtom_pvalues_raw=raw_pvalues, tomtom_pvalues=pvalues,
                  tomtom_evalues=evalues, tomtom_similarity_neglog10_evalue=similarity)
    for name, value in zip(("scores", "offsets", "overlaps", "strands"), result[1:]):
        arrays[f"tomtom_{name}"] = np.asarray(value)
    audit = dict(raw_pvalues_below_floor=int(np.sum(raw_pvalues < 1e-15)),
                 raw_pvalues_above_one=int(np.sum(raw_pvalues > 1)),
                 floor=1e-15, evalue_target_count=len(motifs), reverse_complement=True)
    return arrays, audit


def pca_and_tsne(matrix, perplexity, seed):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    require(matrix.ndim == 2 and np.isfinite(matrix).all(), "Invalid embedding matrix")
    require(len(matrix) >= 3 and 0 < perplexity < len(matrix),
            f"Need >=3 motifs and 0 < perplexity < {len(matrix)}; no silent parameter change")
    require(np.any(np.ptp(matrix, axis=0) > 0), "All motif profiles are identical; no meaningful embedding")
    pca = PCA(n_components=min(50, len(matrix) - 1, matrix.shape[1]), svd_solver="full")
    coordinates = pca.fit_transform(matrix)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                random_state=seed, max_iter=1000, method="barnes_hut", angle=0.5)
    embedding = tsne.fit_transform(coordinates)
    require(np.isfinite(coordinates).all() and np.isfinite(embedding).all(), "Nonfinite PCA/t-SNE")
    return coordinates, embedding, pca.explained_variance_ratio_, float(tsne.kl_divergence_)


def build_knn_graph(pca_coordinates, n_neighbors):
    """PCA Euclidean kNN; explicitly exclude SELF even when identical rows tie."""
    from sklearn.neighbors import NearestNeighbors
    n = len(pca_coordinates)
    require(0 < n_neighbors < n, f"Require 0 < n_neighbors < {n}")
    distances, indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(
        pca_coordinates).kneighbors(pca_coordinates)
    directed = []
    for node, (neighbors, dists) in enumerate(zip(indices, distances)):
        # Sorting by index for returned equal-distance neighbors stabilizes edge ordering.
        others = sorted(((float(d), int(j)) for d, j in zip(dists, neighbors)
                         if int(j) != node), key=lambda pair: (pair[0], pair[1]))[:n_neighbors]
        require(len(others) == n_neighbors, "Wrong number of non-self neighbors")
        directed.extend((node, neighbor, distance) for distance, neighbor in others)
    positive = [distance for _, _, distance in directed if distance > 0]
    scale = max(float(np.median(positive)) if positive else 1.0, np.finfo(float).eps)
    edges = {}
    for node, neighbor, distance in directed:
        edge = tuple(sorted((node, neighbor)))
        weight = float(np.exp(-((distance / scale) ** 2)))
        edges[edge] = max(edges.get(edge, 0.0), weight)
    ordered = sorted(edges)
    return ordered, [edges[e] for e in ordered], scale, directed


def leiden_clusters(coordinates, n_neighbors, resolution, seed):
    import igraph as ig
    import leidenalg
    edges, weights, scale, directed = build_knn_graph(coordinates, n_neighbors)
    graph = ig.Graph(n=len(coordinates), edges=edges, directed=False)
    graph.es["weight"] = weights
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=resolution, seed=seed)
    labels = np.asarray(partition.membership, dtype=int) + 1
    return labels, edges, weights, scale, directed


def make_metadata(motifs, labels, embedding):
    import pandas as pd
    return pd.DataFrame([dict(
        matrix_index_0based=i, motif_id=m.motif_id, stage=m.stage, stage_label=m.stage_label,
        source_hdf5=m.source_hdf5, h5_group=f"pos_patterns/{m.pattern}",
        n_seqlets=m.n_seqlets, original_length=m.original_length,
        trim_start_0based=m.trim_start, trim_end_0based_exclusive=m.trim_end,
        trimmed_length=len(m.ppm), n_informative_positions=m.n_informative_positions,
        no_position_above_ic_threshold=m.n_informative_positions == 0,
        consensus_argmax="".join("ACGT"[j] for j in np.argmax(m.ppm, axis=1)),
        leiden_cluster=int(labels[i]), tsne_1=float(embedding[i, 0]), tsne_2=float(embedding[i, 1]))
        for i, m in enumerate(motifs)])


CLUSTER_COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00',
                  '#56B4E9', '#999999', '#332288', '#44AA99', '#AA4499']
STAGE_COLORS = dict(zip(STAGES.values(), [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
    '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#c5b0d5']))
PDF_NAMES = ('positive_motif_clusters.pdf', 'positive_motifs_by_stage.pdf')


def build_figure(metadata, color_by, annotate=False):
    """Same points, drawing order and axes in both views; only color/legend change."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    require(color_by in ('cluster', 'stage'), 'Unknown coloring mode')
    require(len(metadata) > 0 and metadata.motif_id.is_unique, 'Invalid motif identities')
    require(set(metadata.stage_label).issubset(STAGES.values()), 'Unknown stage label')
    xy = metadata[['tsne_1', 'tsne_2']].to_numpy(dtype=float)
    require(np.isfinite(xy).all(), 'Nonfinite t-SNE coordinates')
    clusters = sorted(int(c) for c in metadata.leiden_cluster.unique())
    cluster_colors = {c: CLUSTER_COLORS[i] if i < len(CLUSTER_COLORS)
                      else mpl.colors.to_hex(plt.get_cmap('tab20')(i % 20))
                      for i, c in enumerate(clusters)}
    if color_by == 'cluster':
        keys, palette = clusters, cluster_colors
        point_colors = [palette[int(c)] for c in metadata.leiden_cluster]
        legend_title = 'Cluster'
    else:
        keys, palette = [s for s in STAGES.values() if s in set(metadata.stage_label)], STAGE_COLORS
        point_colors = [palette[s] for s in metadata.stage_label]
        legend_title = 'Stage'
    style = {
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 12,
        'ytick.labelsize': 12, 'legend.fontsize': 12, 'pdf.fonttype': 42,
        'pdf.compression': 0, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'legend.frameon': False, 'savefig.bbox': None,
    }
    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.subplots_adjust(left=.16, right=.72, bottom=.17, top=.83)
        ax.scatter(xy[:, 0], xy[:, 1], c=point_colors, s=21, alpha=.92,
                   edgecolors='white', linewidths=.25, rasterized=False)
        for dimension, setter in [(0, ax.set_xlim), (1, ax.set_ylim)]:
            low, high = xy[:, dimension].min(), xy[:, dimension].max()
            pad = max(float(high-low)*.07, .5)
            setter(low-pad, high+pad)
        ax.set_xlabel('t-SNE 1', fontsize=14)
        ax.set_ylabel('t-SNE 2', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if annotate:
            for row in metadata.itertuples():
                ax.annotate(f'P{row.matrix_index_0based+1:03d}', (row.tsne_1, row.tsne_2),
                            xytext=(2, 2), textcoords='offset points', fontsize=6)
        handles = [Line2D([], [], marker='o', linestyle='none', label=str(k),
                          markerfacecolor=palette[k], markeredgecolor='none', markersize=5)
                   for k in keys]
        fig.legend(handles=handles, title=legend_title, title_fontsize=12, fontsize=12,
                   loc='center left', bbox_to_anchor=(.735, .505), ncol=1,
                   borderaxespad=0, borderpad=0, labelspacing=.24,
                   handlelength=.9, handletextpad=.35, frameon=False)
        fig.suptitle('Mouse liver: positive motifs', fontsize=14, y=.965)
        fig.text(.44, .885, f'{len(metadata)} motifs | {len(keys) if color_by == "stage" else metadata.stage_label.nunique()} stages | '
                 f'{len(clusters)} clusters', ha='center', fontsize=10)
        # Check the exact fixed-size layout instead of silently enlarging the PDF.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        artists = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(),
                   *fig.texts, *fig.legends]
        bounds = fig.bbox
        for artist in artists:
            if not artist.get_visible():
                continue
            box = artist.get_window_extent(renderer)
            require(box.x0 >= bounds.x0-1 and box.x1 <= bounds.x1+1
                    and box.y0 >= bounds.y0-1 and box.y1 <= bounds.y1+1,
                    f'Figure text outside fixed 5x4 inch canvas: {artist}')
    return fig


def plot_results(metadata, output, annotate=False, provenance=None):
    """Only two final PDFs; reproducibility data embedded as a nonvisible PDF note."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output = Path(output)
    require(output.is_dir(), f'Output directory does not exist: {output}')
    require(all(not (output/name).exists() for name in PDF_NAMES),
            'Refusing to overwrite an existing PDF; choose a new output directory')
    figures = []
    payload = dict(provenance or {}, points=metadata.to_dict(orient='records'),
                   stage_palette=STAGE_COLORS, cluster_palette=CLUSTER_COLORS,
                   figure_size_inches=[5, 4], tick_fontsize=12, axis_label_fontsize=14,
                   legend_fontsize=12, same_coordinates_in_both_figures=True)
    try:
        for mode in ('cluster', 'stage'):
            figures.append(build_figure(metadata, mode, annotate))
        for fig, name, mode in zip(figures, PDF_NAMES, ('cluster', 'stage')):
            path = output / name
            require(not path.exists(), f'Refusing to overwrite {path}')
            with mpl.rc_context({'pdf.compression': 0, 'pdf.fonttype': 42, 'savefig.bbox': None}):
                with PdfPages(path, metadata={
                    'Title': f'Mouse liver positive motifs colored by {mode}',
                    'Subject': 'Joint motif similarity map; stage colors identify motif origin, not activity.',
                    'Keywords': 'TF-MoDISco, Tomtom, PCA, Leiden, t-SNE',
                    'Creator': VERSION,
                }) as pdf:
                    pdf.attach_note(json.dumps(dict(payload, color_by=mode), allow_nan=False))
                    pdf.savefig(fig, bbox_inches=None)
            raw = path.read_bytes()
            require(re.search(rb'/MediaBox\s*\[\s*0\s+0\s+360\s+288\s*\]', raw) is not None,
                    f'Unexpected PDF page dimensions: {path}')
            require(b'/FontFile2' in raw and b'/Subtype /Image' not in raw,
                    'Expected embedded TrueType fonts and vector points')
    finally:
        for fig in figures:
            plt.close(fig)





def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--output-dir", type=Path, help="Must not exist; default: 4.motif_clustering/final_figure_small")
    parser.add_argument("--check-only", action="store_true", help="Audit inputs/dependencies without creating results")
    parser.add_argument("--background", type=float, nargs=4, default=(.27, .23, .23, .27), metavar=("A", "C", "G", "T"))
    parser.add_argument("--trim-ic-threshold", type=float, default=.2)
    parser.add_argument("--trim-flank", type=int, default=0)
    parser.add_argument("--embedding-input", choices=("evalue", "similarity"), default="evalue",
                        help="evalue reproduces reference; similarity uses clipped -log10(E), a different analysis")
    parser.add_argument("--perplexity", type=float, default=15.)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--annotate-patterns", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    background = np.asarray(args.background, dtype=float)
    require(np.isfinite(background).all() and np.all(background > 0) and np.isclose(background.sum(), 1),
            'Background must be finite, positive and sum to one in A,C,G,T order')
    require(np.isfinite(args.trim_ic_threshold) and args.trim_ic_threshold >= 0 and args.trim_flank >= 0,
            'Invalid IC threshold/flank')
    require(np.isfinite(args.resolution) and args.resolution > 0 and args.threads > 0 and args.n_neighbors > 0,
            'Resolution, threads and neighbors must be positive')
    require(0 <= args.seed < 2**31, 'Seed must be in [0,2**31)')
    packages = ['memelite', 'numpy', 'pandas', 'h5py', 'scikit-learn', 'igraph', 'leidenalg', 'matplotlib', 'numba']
    versions = {p: package_metadata.version(p) for p in packages}
    output = (args.output_dir or (ROOT / '4.motif_clustering/final_figure_small')).resolve()
    if not args.check_only:
        require(not output.exists(), f'Output directory already exists: {output}. Choose a new --output-dir.')
    print('Auditing all 14 stage inputs ...', flush=True)
    sources = load_sources(args.batch_dir)
    motifs = read_motifs(sources, background, args.trim_ic_threshold, args.trim_flank)
    require(len(motifs) >= 3 and 0 < args.perplexity < len(motifs) and args.n_neighbors < len(motifs),
            'Perplexity and neighbor count must be smaller than the number of motifs')
    counts = {s['stage_label']: s['positive_motifs'] for s in sources}
    print(json.dumps(dict(n_positive_motifs=len(motifs), per_stage=counts, versions=versions)), flush=True)
    if args.check_only:
        print('CHECK_ONLY_PASSED (no results written)', flush=True)
        return
    started = now()
    print('Running all-vs-all Tomtom ...', flush=True)
    arrays, audit = calculate_tomtom_matrices(motifs, args.threads)
    matrix = arrays['tomtom_evalues'] if args.embedding_input == 'evalue' else arrays['tomtom_similarity_neglog10_evalue']
    print('PCA / t-SNE / weighted-kNN Leiden ...', flush=True)
    pca, embedding, variance, kl = pca_and_tsne(matrix, args.perplexity, args.seed)
    labels, edges, weights, scale, directed = leiden_clusters(pca, args.n_neighbors, args.resolution, args.seed)
    metadata = make_metadata(motifs, labels, embedding)
    for source in sources:
        require(sha256(source['status_path']) == source['status_sha256']
                and sha256(source['hdf5']) == source['hdf5_sha256'],
                'An upstream status/result changed during clustering')
    provenance = dict(version=VERSION, started_utc=started, computed_utc=now(),
        script_sha256=sha256(__file__), python=sys.version, software=versions,
        parameters={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        sources=sources, method=dict(
            pattern_group='pos_patterns', parent_only=True, motif_source='sequence PPM',
            joint_across_stages=True, reverse_complement=True, pvalue_floor=1e-15,
            tomtom_parameters=dict(n_score_bins=100, n_median_bins=1000, n_target_bins=100, n_cache=100),
            evalue='clipped_pvalue * total_motifs', symmetrize_tomtom=False, standardize_matrix=False,
            pca='centered full-SVD; min(50,N-1,N)',
            graph='PCA Euclidean kNN; undirected union; max directed weight; exclude self',
            graph_weight='exp(-(distance/median_positive_directed_knn_distance)^2)',
            leiden='RBConfigurationVertexPartition', tsne_for_visualization_only=True),
        n_motifs=len(motifs), n_clusters=int(metadata.leiden_cluster.nunique()), per_stage=counts,
        n_pca_components=pca.shape[1], pca_variance_explained=float(variance.sum()),
        tsne_kl_divergence=kl, gaussian_distance_scale=scale, tomtom_audit=audit,
        graph_edges=len(edges), zero_weight_edges=int(np.sum(np.asarray(weights)==0)),
        interpretation='Exploratory motif similarity, not TF identity, binding or stage activity. '
                       'Stage color indicates the stage from which a motif was discovered. '
                       'IG targets single-stage raw activity; promoter cohorts differ and may overlap.')
    # No PNG, SVG or tabular/metric sidecars: all point identities, source hashes,
    # stage labels, cluster labels and coordinates are embedded in each PDF note.
    output.mkdir(parents=True, exist_ok=False)
    plot_results(metadata, output, args.annotate_patterns, provenance)
    print(json.dumps(dict(status='complete', n_motifs=len(motifs),
        n_clusters=int(metadata.leiden_cluster.nunique()),
        cluster_counts={int(k):int(v) for k,v in metadata.leiden_cluster.value_counts().sort_index().items()},
        pdfs=[str(output/name) for name in PDF_NAMES])), flush=True)
    return metadata


if __name__ == '__main__':
    main()

