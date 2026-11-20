#!/usr/bin/env python3
"""Plot tissue-coloured proActiv/CAGE support bars for human and mouse.

Inputs are FANTOM5 raw CAGE count matrices and proActiv rowData. A promoter is
supported when at least one same-strand CAGE peak with count > 0 overlaps its
TSS expanded by --window-bp on each side. For the three mouse cerebellum P0
replicates, positive mean count is equivalent to support in any replicate.
Only non-internal promoters with positive mean activity in that comparison are
ranked. Tied activity values are fractionally distributed over 20 equal-sized
5-percentile bins, so ties cannot produce unequal bin denominators.

Mouse proActiv is GRCm38/mm10: mouse CAGE peak IDs are joined to the matching
mm10 BED coordinates; the legacy mm9 positions embedded in IDs are not used.
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
import numpy as np
import pandas as pd


ROOT = Path("/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species")
DATASET = Path("proactiv_result/prove_proactiv_reliable/dataset")
DEFAULT_OUT = Path("proactiv_result/prove_proactiv_reliable/barplot")
HUMAN_COUNTS = "hg19.cage_peak_phase1and2combined_counts_ann.osc.txt.gz"
MOUSE_COUNTS = "mm10_fair+new_CAGE_peaks_phase1and2_counts_ann.osc.txt.gz"
MOUSE_BED = "mm10_fair+new_CAGE_peaks_phase1and2.bed.gz"
IDENTITY = ("promoterId", "geneId", "seqnames", "start", "strand", "internalPromoter")
HUMAN_PEAK = re.compile(r"^(chr[^:]+):(\d+)\.\.(\d+),([+-])$")
N_BINS = 20

# Colors follow the organ key supplied with the example figure.
ORGAN_COLORS = {
    "Brain": "#349BCB",
    "Cerebellum": "#37C4E7",
    "Heart": "#CF111B",
    "Kidney": "#D5A500",
    "Liver": "#2FA900",
    "Ovary": "#BF319A",
    "Testis": "#FF7500",
}


@dataclass(frozen=True)
class Comparison:
    species: str
    tissue: str
    stage_label: str
    stages: tuple[str, ...]
    cage_ids: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.species.lower()}_{self.tissue.lower()}_{self.stage_label.lower()}"


COMPARISONS = (
    Comparison("Human", "Brain", "Adult", ("28ypb", "39ypb", "50ypb", "53ypb", "58ypb"), ("CNhs10617",)),
    Comparison("Human", "Cerebellum", "Newborn", ("0dpb",), ("CNhs14075",)),
    Comparison("Human", "Cerebellum", "Adult", ("29ypb", "32ypb", "39ypb", "50ypb", "53ypb", "58ypb"), ("CNhs11795",)),
    Comparison("Human", "Heart", "Fetal", ("12w", "13w", "16w", "18w", "19w"), ("CNhs10653",)),
    Comparison("Human", "Heart", "Adult", ("25ypb", "54ypb"), ("CNhs10621",)),
    Comparison("Human", "Kidney", "Fetal", ("12w", "13w", "16w", "18w", "19w", "20w"), ("CNhs10652",)),
    Comparison("Human", "Liver", "Adult", ("29ypb", "39ypb", "55ypb", "58ypb"), ("CNhs10624",)),
    Comparison("Mouse", "Cerebellum", "P0", ("0dpb",), ("CNhs12963", "CNhs13010", "CNhs13022")),
    Comparison("Mouse", "Heart", "P0", ("0dpb",), ("CNhs11213",)),
    Comparison("Mouse", "Kidney", "P0", ("0dpb",), ("CNhs11214",)),
    Comparison("Mouse", "Liver", "P0", ("0dpb",), ("CNhs11117",)),
    Comparison("Mouse", "Ovary", "P0", ("0dpb",), ("CNhs11217",)),
    Comparison("Mouse", "Testis", "P0", ("0dpb",), ("CNhs11189",)),
)


def load_comparison(root: Path, spec: Comparison, cache: dict) -> dict:
    reference = None
    values = []
    for stage in spec.stages:
        key = (spec.species, spec.tissue, stage)
        mean_col = f"{spec.species}_{spec.tissue}_{stage}.mean"
        if key not in cache:
            path = root / "proactiv_result" / spec.species / f"{spec.species}_{spec.tissue}_{stage}_rowData.tsv"
            data = pd.read_csv(path, sep="\t", usecols=[*IDENTITY, mean_col])
            if data["promoterId"].duplicated().any():
                raise ValueError(f"Duplicate promoterId: {path}")
            flag = data["internalPromoter"].astype(str).str.strip().str.upper()
            if not flag.isin(("TRUE", "FALSE")).all():
                raise ValueError(f"Invalid internalPromoter value: {path}")
            data["internalPromoter"] = flag.eq("TRUE")
            cache[key] = data
        data = cache[key]
        identity = data.loc[:, list(IDENTITY)].reset_index(drop=True)
        if reference is None:
            reference = identity.copy()
        elif not identity.equals(reference):
            raise ValueError(f"Promoter identity/coordinate mismatch: {spec.name}, {stage}")
        value = pd.to_numeric(data[mean_col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"Invalid proActiv activity: {spec.name}, {stage}")
        values.append(value)
    assert reference is not None
    reference["activity_mean"] = np.mean(values, axis=0)
    noninternal = reference.loc[~reference["internalPromoter"]]
    n_zero_excluded = int((noninternal["activity_mean"] == 0).sum())
    promoters = noninternal.loc[noninternal["activity_mean"] > 0].reset_index(drop=True)
    if promoters.empty:
        raise ValueError(f"No positive non-internal promoters: {spec.name}")
    loci = {}
    for key, group in promoters.groupby(["seqnames", "strand"], sort=False):
        ordered = group.sort_values("start")
        loci[key] = (ordered["start"].astype(int).tolist(), ordered.index.to_numpy(dtype=int))
    return {
        "spec": spec,
        "promoters": promoters,
        "loci": loci,
        "supported": np.zeros(len(promoters), dtype=bool),
        "n_noninternal": len(noninternal),
        "n_zero_excluded": n_zero_excluded,
        "cage_columns": (),
    }


def load_mouse_bed(path: Path) -> dict[str, tuple[str, int, int, str]]:
    if "mm10" not in path.name:
        raise ValueError("Mouse BED must use mm10/GRCm38 coordinates")
    peaks = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise ValueError("Malformed mm10 BED row")
            chrom, start0, end0, peak_id, _, strand = fields[:6]
            start1, end1 = int(start0) + 1, int(end0)
            if peak_id in peaks or start1 > end1 or strand not in ("+", "-"):
                raise ValueError(f"Invalid/duplicate mm10 BED peak: {peak_id}")
            peaks[peak_id] = (chrom, start1, end1, strand)
    if not peaks:
        raise ValueError("No mm10 CAGE coordinates found")
    return peaks


def scan_cage_counts(path: Path, contexts: list[dict], window_bp: int,
                     mouse_bed: dict | None = None) -> dict:
    species = contexts[0]["spec"].species
    if any(x["spec"].species != species for x in contexts):
        raise ValueError("One CAGE matrix cannot contain both species")
    n_records = n_joined = 0
    with gzip.open(path, "rt") as handle:
        assembly = None
        for line in handle:
            if "genome_assembly" in line:
                assembly = line.rstrip("\n").rsplit("=", 1)[-1]
            if line.startswith("00Annotation\t"):
                header = line.rstrip("\n").split("\t")
                break
        else:
            raise ValueError(f"FANTOM5 matrix header not found: {path}")
        if species == "Mouse" and (assembly != "mm10" or mouse_bed is None):
            raise ValueError("Mouse count matrix and BED must both be mm10")
        if species == "Human" and assembly not in (None, "hg19"):
            raise ValueError(f"Unexpected human CAGE assembly: {assembly}")
        col_by_id = {}
        for context in contexts:
            spec = context["spec"]
            selected = []
            for cage_id in spec.cage_ids:
                matches = [i for i, name in enumerate(header)
                           if name.startswith("counts.") and f".{cage_id}." in name]
                if len(matches) != 1:
                    raise ValueError(f"Expected one count column for {cage_id}; got {matches}")
                index = matches[0]
                col_by_id[cage_id] = index
                selected.append(header[index])
            context["cage_columns"] = tuple(selected)
        max_col = max(col_by_id.values())
        for line in handle:
            peak_id = line.split("\t", 1)[0]
            if peak_id.startswith(("01STAT:", "02STAT:")):
                continue
            if species == "Human":
                match = HUMAN_PEAK.fullmatch(peak_id)
                if match is None:
                    continue
                chrom, start, end, strand = match.groups()
                coordinates = (chrom, int(start), int(end), strand)
                n_joined += 1
            else:
                coordinates = mouse_bed.get(peak_id)
                if coordinates is None:
                    n_records += 1
                    continue
                n_joined += 1
            n_records += 1
            chrom, start, end, strand = coordinates
            nearby = []
            for context in contexts:
                locus = context["loci"].get((chrom, strand))
                if locus is None:
                    continue
                positions, rows = locus
                left = bisect_left(positions, start - window_bp)
                right = bisect_right(positions, end + window_bp)
                if left < right:
                    nearby.append((context, rows[left:right]))
            if not nearby:
                continue
            fields = line.rstrip("\n").split("\t", max_col + 1)
            if len(fields) <= max_col:
                raise ValueError(f"Truncated CAGE count row: {peak_id}")
            counts = {cage_id: float(fields[index]) for cage_id, index in col_by_id.items()}
            if any(not np.isfinite(x) or x < 0 for x in counts.values()):
                raise ValueError(f"Invalid CAGE count at {peak_id}")
            for context, rows in nearby:
                # Mean count > 0 across a replicate group iff at least one
                # replicate has a positive raw count for this peak.
                if any(counts[cage_id] > 0 for cage_id in context["spec"].cage_ids):
                    context["supported"][rows] = True
    if species == "Mouse" and n_joined < 0.95 * min(n_records, len(mouse_bed)):
        raise ValueError("Most CAGE IDs failed to join to mm10 BED")
    return {"n_cage_peak_records": n_records, "n_peak_ids_joined_to_coordinates": n_joined}


def percentile_summary(context: dict) -> pd.DataFrame:
    promoters = context["promoters"]
    n = len(promoters)
    grouped = pd.DataFrame({
        "activity": promoters["activity_mean"].to_numpy(),
        "supported": context["supported"],
    }).groupby("activity", sort=True)["supported"].agg(["size", "sum"])
    effective_n = np.zeros(N_BINS, dtype=float)
    effective_supported = np.zeros(N_BINS, dtype=float)
    rank = 0
    for row in grouped.itertuples(index=False):
        count, supported = int(row.size), int(row.sum)
        start, end = rank, rank + count
        first = N_BINS * start // n
        last = min(N_BINS - 1, (N_BINS * end - 1) // n)
        for b in range(first, last + 1):
            overlap = max(0, min(N_BINS * end, (b + 1) * n)
                          - max(N_BINS * start, b * n)) / N_BINS
            effective_n[b] += overlap
            effective_supported[b] += overlap * supported / count
        rank = end
    if not np.allclose(effective_n, n / N_BINS, atol=1e-7, rtol=0):
        raise AssertionError("Percentile-bin denominators differ")
    if not np.isclose(effective_supported.sum(), context["supported"].sum(), atol=1e-7):
        raise AssertionError("CAGE support does not reconcile")
    return pd.DataFrame({
        "species": context["spec"].species,
        "comparison": context["spec"].name,
        "percentile_low": np.arange(N_BINS) * 5,
        "percentile_high": (np.arange(N_BINS) + 1) * 5,
        "effective_n": effective_n,
        "effective_supported": effective_supported,
        "percent_cage_support": 100 * effective_supported / effective_n,
    })


def draw_panel(ax, spec: Comparison, bins: pd.DataFrame, n: int) -> None:
    centers = (bins["percentile_low"] + bins["percentile_high"]) / 2
    ax.bar(centers, bins["percent_cage_support"], width=4.4,
           color=ORGAN_COLORS[spec.tissue], edgecolor="#243039", linewidth=0.55)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.set_xticks((0, 20, 40, 60, 80, 100))
    ax.set_yticks((0, 20, 40, 60, 80, 100))
    ids = "/".join(spec.cage_ids)
    ax.set_title(f"{spec.tissue} · {spec.stage_label}\nCAGE: {ids}",
                 loc="left", fontsize=9.0, pad=7)
    ax.text(0.98, 0.96, f"n = {n:,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#475569")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, length=3)


def plot_species(species: str, contexts: list[dict], bins: pd.DataFrame,
                 outdir: Path, dpi: int) -> None:
    ncols = 2
    nrows = (len(contexts) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.8, nrows * 2.8 + 0.85),
                             sharex=True, sharey=True, squeeze=False)
    for ax, context in zip(axes.flat, contexts):
        spec = context["spec"]
        data = bins.loc[bins["comparison"] == spec.name]
        draw_panel(ax, spec, data, len(context["promoters"]))
        ax.tick_params(labelbottom=True, labelleft=True)
        one, one_ax = plt.subplots(figsize=(3.65, 3.0))
        draw_panel(one_ax, spec, data, len(context["promoters"]))
        one_ax.set_xlabel("Promoter activity (percentiles)", fontsize=9)
        one_ax.set_ylabel("CAGE-supported promoters (%)", fontsize=9)
        one.tight_layout()
        one.savefig(outdir / f"{spec.name}.png", dpi=dpi)
        one.savefig(outdir / f"{spec.name}.pdf")
        plt.close(one)
    for ax in list(axes.flat)[len(contexts):]:
        fig.delaxes(ax)
    fig.supxlabel("Promoter activity (percentiles)", fontsize=10)
    fig.supylabel("CAGE-supported promoters (%)", fontsize=10)
    fig.tight_layout(rect=(0.04, 0.04, 1, 1), h_pad=1.3, w_pad=1.3)
    fig.savefig(outdir / f"{species.lower()}_cage_support_bars.png", dpi=dpi)
    fig.savefig(outdir / f"{species.lower()}_cage_support_bars.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--window-bp", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.window_bp < 0 or args.dpi < 72:
        parser.error("--window-bp must be >= 0 and --dpi must be >= 72")
    outdir = args.outdir or args.root / DEFAULT_OUT
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.75,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    cache = {}
    bins_all = []
    manifest = []
    for species in ("Human", "Mouse"):
        contexts = [load_comparison(args.root, spec, cache)
                    for spec in COMPARISONS if spec.species == species]
        print(f"Loaded {len(contexts)} {species} comparisons", flush=True)
        cage_path = args.root / DATASET / (HUMAN_COUNTS if species == "Human" else MOUSE_COUNTS)
        bed = load_mouse_bed(args.root / DATASET / MOUSE_BED) if species == "Mouse" else None
        scan = scan_cage_counts(cage_path, contexts, args.window_bp, bed)
        print(f"Scanned {scan['n_cage_peak_records']:,} {species} CAGE peaks", flush=True)
        species_bins = pd.concat([percentile_summary(x) for x in contexts], ignore_index=True)
        bins_all.append(species_bins)
        for context in contexts:
            spec = context["spec"]
            n = len(context["promoters"])
            manifest.append({
                "species": species, "comparison": spec.name, "tissue": spec.tissue,
                "stage_label": spec.stage_label, "proactiv_stages": ",".join(spec.stages),
                "cage_ids": ",".join(spec.cage_ids),
                "cage_columns": " | ".join(context["cage_columns"]),
                "cage_count_file": str(cage_path),
                "cage_assembly": "hg19" if species == "Human" else "mm10",
                "n_noninternal": context["n_noninternal"],
                "n_zero_excluded": context["n_zero_excluded"],
                "n_positive_promoters": n,
                "n_supported_promoters": int(context["supported"].sum()),
                "percent_supported": 100 * context["supported"].mean(),
                "n_cage_peak_records": scan["n_cage_peak_records"],
                "n_peak_ids_joined_to_coordinates": scan["n_peak_ids_joined_to_coordinates"],
                "window_bp": args.window_bp,
                "mouse_cerebellum_rule": "any of three replicates positive" if species == "Mouse" and spec.tissue == "Cerebellum" else "one CAGE library",
            })
        plot_species(species, contexts, species_bins, outdir, args.dpi)
    pd.concat(bins_all, ignore_index=True).to_csv(outdir / "cage_support_percentile_bins.tsv", sep="\t", index=False)
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(outdir / "comparison_manifest.tsv", sep="\t", index=False)
    print(manifest_df[["comparison", "cage_ids", "n_positive_promoters",
                       "percent_supported"]].to_string(index=False), flush=True)
    print(f"Saved figures and audit tables to {outdir}", flush=True)


if __name__ == "__main__":
    main()
