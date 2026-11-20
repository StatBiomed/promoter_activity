#!/usr/bin/env python3
"""Rebuild the six-by-six proActiv/FANTOM5 Pearson-rank heatmap.

This is a standalone script: it reads the original human proActiv rowData TSVs
and the original FANTOM5 hg19 TPM OSC, not another script or a derived matrix.
Each RNA-seq column is the equal-weight mean of the listed stage means, taken
per promoter before correlation. Each CAGE row is one FANTOM5 sample. Promoters
are matched to same-strand CAGE peaks within 100 bp of their proActiv TSS;
the largest nearby peak TPM is used. Zeros remain in the common denominator.

Only non-internal promoters are included. Promoters inactive in all six RNA
groups are removed globally, never separately for individual cells. Pearson
correlations are calculated across this same promoter universe, then ranked
within each CAGE row. Colors encode rank, not correlation magnitude.
"""

from __future__ import annotations

import argparse
import gzip
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata


ROOT = Path("/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species")
HUMAN = Path("proactiv_result/Human")
TPM = Path(
    "proactiv_result/prove_proactiv_reliable/dataset/"
    "hg19.cage_peak_phase1and2combined_tpm_ann.osc.txt.gz"
)
OUTPUT = Path("proactiv_result/prove_proactiv_reliable/cage_tpm_age_window_pearson_small")
IDENTITY = ("promoterId", "geneId", "seqnames", "start", "strand", "internalPromoter")
PEAK_RE = re.compile(r"^(chr[^:]+):(\d+)\.\.(\d+),([+-])$")
COLORS = ("#e72c2c", "#f59fa6", "#e7f2f5", "#afd7e3")


@dataclass(frozen=True)
class Pair:
    tissue: str
    cage_id: str
    cage_age: str
    rna_stages: tuple[str, ...]
    row_label: str
    column_label: str


PAIRS = (
    Pair("Brain", "CNhs11797", "20–33w", ("20w",),
         "Brain · 20–33w · CNhs11797", "Brain · 20w"),
    Pair("Cerebellum", "CNhs14075", "0d", ("0dpb",),
         "Cerebellum · 0d · CNhs14075", "Cerebellum · 0d"),
    Pair("Cerebellum", "CNhs11795", "22–68y",
         ("29ypb", "32ypb", "39ypb", "50ypb", "53ypb", "58ypb"),
         "Cerebellum · 22–68y · CNhs11795", "Cerebellum · 29–58y"),
    Pair("Heart", "CNhs10653", "12–31w",
         ("12w", "13w", "16w", "18w", "19w"),
         "Heart · 12–31w · CNhs10653", "Heart · 12–19w"),
    Pair("Kidney", "CNhs10652", "12–31w",
         ("12w", "13w", "16w", "18w", "19w", "20w"),
         "Kidney · 12–31w · CNhs10652", "Kidney · 12–20w"),
    Pair("Testis", "CNhs12998", "14–64y",
         ("14ypb", "16ypb", "17ypb", "28ypb", "29ypb", "39ypb", "46ypb", "55ypb"),
         "Testis · 14–64y · CNhs12998", "Testis · 14–55y"),
)
IDS = tuple(pair.cage_id for pair in PAIRS)


def stage_files(root: Path) -> dict[tuple[str, str], Path]:
    files = {}
    for pair in PAIRS:
        for stage in pair.rna_stages:
            key = (pair.tissue, stage)
            path = root / HUMAN / f"Human_{pair.tissue}_{stage}_rowData.tsv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing proActiv rowData: {path}")
            files[key] = path
    return files


def load_rna_means(files: dict[tuple[str, str], Path]) -> tuple[pd.DataFrame, np.ndarray, dict]:
    reference = None
    noninternal = None
    stage_values = {}
    for (tissue, stage), path in files.items():
        column = f"Human_{tissue}_{stage}.mean"
        data = pd.read_csv(path, sep="\t", usecols=[*IDENTITY, column])
        if data.promoterId.duplicated().any():
            raise ValueError(f"Duplicate promoterId in {path}")
        identity = data.loc[:, list(IDENTITY)].reset_index(drop=True)
        if reference is None:
            reference = identity
            flag = identity.internalPromoter.astype(str).str.upper()
            if not flag.isin(("TRUE", "FALSE")).all():
                raise ValueError(f"Unexpected internalPromoter values in {path}")
            noninternal = flag.eq("FALSE").to_numpy()
        elif not identity.equals(reference):
            raise ValueError(f"Promoter IDs or coordinates differ in {path}")
        values = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid proActiv activity values in {path}")
        stage_values[(tissue, stage)] = values[noninternal]
        print(f"Loaded {tissue} {stage}", flush=True)

    assert reference is not None and noninternal is not None
    activity = np.column_stack([
        np.mean([stage_values[(pair.tissue, stage)] for stage in pair.rna_stages], axis=0)
        for pair in PAIRS
    ])
    keep = (activity > 0).any(axis=1)
    promoters = reference.loc[
        noninternal, ["promoterId", "geneId", "seqnames", "start", "strand"]
    ].reset_index(drop=True)
    audit = {
        "n_reference_promoters": len(reference),
        "n_noninternal_promoters": int(noninternal.sum()),
        "n_globally_inactive_excluded": int((~keep).sum()),
        "n_shared_promoters": int(keep.sum()),
        "n_unique_rna_stages": len(files),
    }
    return promoters.loc[keep].reset_index(drop=True), activity[keep], audit


def cage_column_indices(header: list[str]) -> list[int]:
    indices = []
    for cage_id in IDS:
        matches = [i for i, name in enumerate(header)
                   if name.startswith("tpm.") and f".{cage_id}." in name]
        if len(matches) != 1:
            raise ValueError(f"Expected one TPM column for {cage_id}; found {matches}")
        indices.append(matches[0])
    return indices


def load_cage(path: Path, promoters: pd.DataFrame, window_bp: int) -> tuple[np.ndarray, dict]:
    if window_bp < 0:
        raise ValueError("--window-bp must be nonnegative")
    if not path.is_file():
        raise FileNotFoundError(f"Missing FANTOM5 TPM file: {path}")
    loci = {}
    for key, frame in promoters.groupby(["seqnames", "strand"], sort=False):
        ordered = frame.sort_values("start")
        loci[key] = (ordered.start.astype(int).tolist(), ordered.index.to_numpy(dtype=np.int64))
    cage = np.zeros((len(promoters), len(PAIRS)), dtype=np.float64)
    near_count = np.zeros(len(promoters), dtype=np.uint32)
    n_peaks = n_near = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("00Annotation\t"):
                indices = cage_column_indices(line.rstrip("\n").split("\t"))
                break
        else:
            raise ValueError("FANTOM5 TPM header not found")
        for line in handle:
            match = PEAK_RE.fullmatch(line.split("\t", 1)[0])
            if match is None:
                continue
            n_peaks += 1
            chrom, start, end, strand = match.groups()
            locus = loci.get((chrom, strand))
            if locus is None:
                continue
            positions, rows = locus
            lo = bisect_left(positions, int(start) - window_bp)
            hi = bisect_right(positions, int(end) + window_bp)
            if lo == hi:
                continue
            fields = line.rstrip("\n").split("\t")
            values = np.asarray([float(fields[i]) for i in indices], dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid CAGE TPM at peak {fields[0]}")
            found = rows[lo:hi]
            cage[found] = np.maximum(cage[found], values)
            near_count[found] += 1
            n_near += 1
            if n_peaks % 50000 == 0:
                print(f"Scanned {n_peaks:,} CAGE peaks", flush=True)
    return cage, {
        "n_cage_peaks_scanned": n_peaks,
        "n_peaks_near_any_promoter": n_near,
        "n_promoters_near_any_peak": int((near_count > 0).sum()),
        "window_bp": window_bp,
        "peak_aggregation": "max TPM of same-strand peaks within TSS window",
    }


def pearson_matrix(cage: np.ndarray, activity: np.ndarray) -> np.ndarray:
    if cage.shape != activity.shape or cage.shape[1] != len(PAIRS):
        raise ValueError("CAGE and RNA matrices must share promoter rows and six columns")
    c = cage - cage.mean(axis=0)
    a = activity - activity.mean(axis=0)
    c_norm = np.linalg.norm(c, axis=0)
    a_norm = np.linalg.norm(a, axis=0)
    if (c_norm == 0).any() or (a_norm == 0).any():
        raise ValueError("Pearson correlation undefined for a zero-variance sample")
    correlation = (c.T @ a) / np.outer(c_norm, a_norm)
    correlation = np.clip(correlation, -1, 1)
    if correlation.shape != (len(PAIRS), len(PAIRS)) or not np.isfinite(correlation).all():
        raise ValueError("Pearson correlation undefined for at least one sample")
    for i, j in ((0, 0), (1, 1), (2, 1), (5, 5)):
        direct = pearsonr(cage[:, i], activity[:, j])[0]
        if not np.isclose(correlation[i, j], direct, atol=1e-10, rtol=0):
            raise AssertionError(f"Pearson verification failed at ({i}, {j})")
    return correlation


def draw_heatmap(ranks: np.ndarray, out_dir: Path, dpi: int) -> None:
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm((0.5, 1.5, 2.5, 3.5, 4.5), cmap.N)
    fig, ax = plt.subplots(figsize=(8.0, 6.0), facecolor="white")
    ax.imshow(np.minimum(ranks, 4), cmap=cmap, norm=norm,
              interpolation="nearest", aspect="equal")
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels([p.column_label for p in PAIRS], rotation=43, ha="right",
                       rotation_mode="anchor", fontsize=8.5)
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels([p.row_label for p in PAIRS], fontsize=8.5)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(PAIRS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PAIRS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    # No plot title or axis titles; the labels specify the six comparisons.
    handles = [Patch(facecolor=COLORS[i], edgecolor="#26333a", linewidth=0.8,
                     label=label)
               for i, label in enumerate(("1st", "2nd", "3rd", "≥4th"))]
    fig.legend(handles=handles, title="Rank of correlation\n(Pearson)",
               ncol=2, loc="lower center", bbox_to_anchor=(0.66, 0.015),
               frameon=False, fontsize=9, title_fontsize=9,
               columnspacing=1.0, handlelength=1.1, handletextpad=0.35)
    fig.subplots_adjust(left=0.36, right=0.985, top=0.98, bottom=0.36)
    fig.savefig(out_dir / "cage_tpm_age_window_pearson_rank_small.png",
                dpi=dpi, facecolor="white")
    fig.savefig(out_dir / "cage_tpm_age_window_pearson_rank_small.pdf",
                facecolor="white")
    plt.close(fig)


def run(root: Path, cage_tpm: Path, out_dir: Path, window_bp: int, dpi: int) -> None:
    files = stage_files(root)
    promoters, activity, rna_audit = load_rna_means(files)
    cage, cage_audit = load_cage(cage_tpm, promoters, window_bp)
    correlation = pearson_matrix(cage, activity)
    ranks = rankdata(-correlation, axis=1, method="min").astype(int)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(correlation, index=IDS, columns=IDS).to_csv(
        out_dir / "pearson_coefficients_for_audit.tsv", sep="\t", float_format="%.12g")
    pd.DataFrame(ranks, index=IDS, columns=IDS).to_csv(
        out_dir / "pearson_ranks.tsv", sep="\t")
    pd.DataFrame([{
        "order": i + 1,
        "tissue": p.tissue,
        "cage_id": p.cage_id,
        "cage_age": p.cage_age,
        "rna_stages": ",".join(p.rna_stages),
        "n_rna_stages": len(p.rna_stages),
        "aggregation": "equal-weight mean of proActiv stage means per promoter",
        "cage_zero_tpm_fraction": float(np.mean(cage[:, i] == 0)),
    } for i, p in enumerate(PAIRS)]).to_csv(
        out_dir / "sample_mapping.tsv", sep="\t", index=False)
    pd.DataFrame([{
        **rna_audit,
        **cage_audit,
        "n_cage_samples": len(PAIRS),
        "n_rna_age_window_groups": len(PAIRS),
        "correlation": "Pearson across shared non-internal promoters with zeros retained",
        "tie_handling": "not applicable to Pearson coefficients",
        "heatmap_encoding": "descending correlation rank within each CAGE row",
        "source_proactiv_dir": str(root / HUMAN),
        "source_cage_tpm": str(cage_tpm),
    }]).to_csv(out_dir / "methods_summary.tsv", sep="\t", index=False)
    draw_heatmap(ranks, out_dir, dpi)
    print(f"Saved independent Pearson figure and audit tables in {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="RNA_seq_organ_species root holding proactiv_result/Human")
    parser.add_argument("--cage-tpm", type=Path,
                        help="Original gzipped FANTOM5 hg19 TPM OSC; defaults to ROOT/dataset path")
    parser.add_argument("--out-dir", type=Path,
                        help="Output directory; defaults to ROOT/cage_tpm_age_window_pearson_small")
    parser.add_argument("--window-bp", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    root = args.root.resolve()
    run(root, (args.cage_tpm or root / TPM).resolve(),
        (args.out_dir or root / OUTPUT).resolve(), args.window_bp, args.dpi)


if __name__ == "__main__":
    main()
