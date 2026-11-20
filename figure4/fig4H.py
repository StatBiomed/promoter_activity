#!/usr/bin/env python3
"""E2F4 overlap with Background, Group A and Group B merged BEDs.

Python >=3.9; matplotlib and scipy required. Default inputs are mm10, BED 0-based,
half-open. No additional peak filtering, window expansion or merging is done.
Adapted from the supplied NFIL3 three-comparison overlap workflow. The default export is ONE
5 x 4 inch PDF on the server; numerical results/provenance go to stdout only.
Use --check-only for a read-only audit, or --preview-stdout for an in-memory
PNG preview (no image file). Existing PDF files are never overwritten.

Interpretation: each bar is the fraction of unique merged regions that touch
at least one E2F4 peak by >=1 bp. This is descriptive binding association,
not evidence of causal developmental promoter switching. No error bars are
inferred from regions: these are not biological replicates. At the user's
request, two-sided Fisher tests compare each group with the FULL background,
using [[group overlap, group nonoverlap], [background overlap, background
nonoverlap]], and also compare Group A with Group B. Background is NOT reduced
by the tested group. Displayed P values are nominal, unadjusted values. Groups
and background share genomic regions; A/B also derive from paired genes and
can overlap. Independence is therefore not assured; these requested comparisons
are exploratory, not calibrated evidence of enrichment or depletion.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

BASE = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/5.chipseq_overlap')
DEFAULT_PEAKS = BASE / 'bed_overlap/E2F4/2.converted_peaks/GSM427091_E2F4_liver.mm10.bed'
DEFAULT_QUERIES = BASE / 'chip_data/E2F4'
DEFAULT_OUTPUT = BASE / 'bed_overlap/E2F4/E2F4_overlap_percentage.pdf'
DEFAULT_SIZES = BASE / 'bed_overlap/E2F4/1.assembly_evidence/mm10.chrom.sizes'


@dataclass(frozen=True)
class Interval:
    chrom: str
    start: int
    end: int


def normalize_chrom(value):
    """Normalize standard mouse aliases only; this is NOT liftOver."""
    match = re.fullmatch(r'(?:chr)?([1-9]|1[0-9]|X|Y|M|MT)', value)
    return 'chr' + ('M' if match[1] == 'MT' else match[1]) if match else value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_bed(path):
    """Count genomic regions, not peak IDs or strand-specific promoter IDs."""
    path = Path(path)
    records, seen, headers = [], set(), []
    raw_n = 0
    aliases = Counter()
    opener = gzip.open if path.suffix.lower() == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(('#', 'track ', 'browser ')) or line in ('track', 'browser'):
                headers.append(line)
                declared = re.findall(r'\b(mm8|mm9|mm39|GRCm39)\b', line, re.I)
                if declared:
                    raise ValueError(f'{path}:{lineno}: header conflicts with mm10: {declared}')
                continue
            fields = line.split()
            if len(fields) < 3 or len(fields) >= 12:
                raise ValueError(f'{path}:{lineno}: expected BED3 to BED11, not BED12 blocks')
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError:
                raise ValueError(f'{path}:{lineno}: coordinates must be integers') from None
            if start < 0 or end <= start:
                raise ValueError(f'{path}:{lineno}: invalid half-open interval')
            chrom = normalize_chrom(fields[0])
            if chrom != fields[0]:
                aliases[f'{fields[0]} -> {chrom}'] += 1
            record = Interval(chrom, start, end)
            raw_n += 1
            if record not in seen:
                records.append(record)
                seen.add(record)
    if not records:
        raise ValueError(f'No valid records: {path}')
    return records, dict(path=str(path.resolve()), sha256=sha256_file(path),
                         raw_records=raw_n, unique_regions=len(records),
                         duplicate_coordinates_removed=raw_n-len(records),
                         chromosome_aliases=dict(aliases), headers=headers)


def read_sizes(path):
    sizes = {}
    with Path(path).open() as handle:
        for line in handle:
            if line.strip() and not line.startswith('#'):
                chrom, size = line.split()[:2]
                sizes[normalize_chrom(chrom)] = int(size)
    if not sizes:
        raise ValueError('Empty mm10 chromosome-size file')
    return sizes


def validate_bounds(records, sizes, label):
    invalid = [r for r in records if r.chrom not in sizes or r.end > sizes[r.chrom]]
    if invalid:
        raise ValueError(f'{label}: {len(invalid)} regions outside mm10 bounds: {invalid[:3]}')


def peak_index(peaks):
    starts, ends = defaultdict(list), defaultdict(list)
    for peak in peaks:
        starts[peak.chrom].append(peak.start)
        ends[peak.chrom].append(peak.end)
    return {chrom: (sorted(values), sorted(ends[chrom])) for chrom, values in starts.items()}


def overlap_counts(records, index):
    """Equivalent to bedtools intersect -c, unstranded and with >=1 bp overlap."""
    counts = []
    for record in records:
        starts, ends = index.get(record.chrom, ([], []))
        counts.append(bisect_left(starts, record.end) - bisect_right(ends, record.start))
    return counts


def fisher_pairwise(rows):
    """Three requested two-sided comparisons; do not subtract group from BG."""
    import math
    from scipy.stats import fisher_exact
    by_label = {row['group']: row for row in rows}
    tests = []
    for label, comparator in [('Group A', 'Background'), ('Group B', 'Background'),
                              ('Group A', 'Group B')]:
        group = by_label[label]
        reference = by_label[comparator]
        table = [[group['overlap_n'], group['nonoverlap_n']],
                 [reference['overlap_n'], reference['nonoverlap_n']]]
        if any(n < 0 or int(n) != n for row in table for n in row):
            raise ValueError('Fisher test requires non-negative integer counts')
        odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
        tests.append(dict(group=label, comparator=comparator,
                          row_labels=[label, comparator],
                          column_labels=['Overlapping', 'Not overlapping'],
                          contingency_table=table, test='two-sided Fisher exact test',
                          odds_ratio=float(odds_ratio) if math.isfinite(odds_ratio) else str(odds_ratio),
                          p_value=float(p_value), multiple_testing_adjustment='None',
                          inference='Exploratory only: shared genomic regions and paired genes can violate independence'))
    return tests


def analyse(peaks_path, query_dir, sizes_path):
    peaks, peak_audit = read_bed(peaks_path)
    sizes = read_sizes(sizes_path)
    validate_bounds(peaks, sizes, 'E2F4 peaks')
    index = peak_index(peaks)
    groups, audits, rows = {}, {}, []
    # Explicit inputs prevent the unrelated E2F4 peak BED being read as a group.
    for label, filename in [('Background', 'background.merged.bed'),
                            ('Group A', 'A.merged.bed'), ('Group B', 'B.merged.bed')]:
        path = query_dir / filename
        if path.resolve() == peaks_path.resolve():
            raise ValueError('Peak and query BED cannot be the same file')
        records, audits[label] = read_bed(path)
        validate_bounds(records, sizes, label)
        if not ({r.chrom for r in records} & set(index)):
            raise ValueError(f'{label}: no chromosomes shared with peaks')
        counts = overlap_counts(records, index)
        yes = sum(n > 0 for n in counts)
        rows.append(dict(group=label, total_n=len(records), overlap_n=yes,
                         nonoverlap_n=len(records)-yes, overlap_percentage=100*yes/len(records),
                         chromosomes_without_peaks=sorted({r.chrom for r in records} - set(index))))
        groups[label] = set(records)
    subsets = {label: group.issubset(groups['Background'])
               for label, group in groups.items() if label != 'Background'}
    return dict(assembly='mm10/GRCm38', peaks=peak_audit, queries=audits, summary=rows,
                mm10_sizes_sha256=sha256_file(sizes_path),
                exact_background_subset=subsets,
                statistics='Two-sided Fisher exact tests: A vs FULL background, B vs FULL background, A vs B; nominal P values; no error bars.',
                fisher_tests=fisher_pairwise(rows),
                statistical_limitation='Shared group/background regions and paired A/B genes can violate independence; interpret these requested tests as exploratory, not calibrated enrichment evidence.',
                unit='Unique merged genomic region (chromosome, start, end), ignoring strand',
                overlap_rule='At least 1 bp; each query region counted once even if it hits multiple peaks',
                assembly_note='E2F4 peaks are the previously verified mm9-to-mm10 liftOver output (8821 mapped peaks; see 2.converted_peaks/conversion_manifest.json); promoter inputs are mm10. Original mm9 peaks are not used.')


def make_plot(rows, tests):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    # Equivalent to the final-figure-small bundled helper; kept self-contained.
    mpl.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'font.size': 12, 'axes.labelsize': 14,
                         'axes.titlesize': 14, 'xtick.labelsize': 12,
                         'ytick.labelsize': 12, 'legend.fontsize': 12,
                         'axes.linewidth': .8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False})
    fig, ax = plt.subplots(figsize=(5, 4))
    values = [row['overlap_percentage'] for row in rows]
    ax.bar(range(3), values, color=['#F1B656', '#397FC7', '#6F6F6F'], width=.75)
    maximum = max(5, *values)
    for i, row in enumerate(rows):
        ax.text(i, row['overlap_percentage'] + maximum*.025,
                f"{row['overlap_n']}/{row['total_n']}\n{row['overlap_percentage']:.1f}%",
                ha='center', va='bottom', fontsize=11, linespacing=1.15)
    ax.set_xticks(range(3), [r['group'] for r in rows])
    ax.set_ylabel('Regions overlapping with\nE2F4 ChIP-seq peaks (%)', fontsize=14)
    ax.set_xlabel('', fontsize=14)
    ax.set_title('E2F4', fontsize=14, pad=10)
    # Preserve bars/count labels and add reference-style comparison lines.
    # P values are raw (not BH-adjusted); the full-background caveat is in
    # stdout, this script's methods, and the PDF Subject metadata.
    positions = {row['group']: i for i, row in enumerate(rows)}
    for test, height in zip(tests, (1.30, 1.52, 1.74)):
        left, right = sorted((positions[test['group']], positions[test['comparator']]))
        y = maximum * height
        ax.plot([left, right], [y, y], color='#303030', linewidth=.9)
        ax.text((left+right)/2, y + maximum*.025, f"P = {test['p_value']:.4g}",
                ha='center', va='bottom', fontsize=11, fontstyle='italic')
    ax.set_ylim(0, maximum*1.98)
    # Suppress the unused tick beyond ylim (e.g. 125%) before checking page bounds.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
    ax.tick_params(direction='out', labelsize=12, width=.8)
    fig.tight_layout(pad=1.0)
    # Ensure all labels stay on the fixed 5 x 4 inch page.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = [ax.title, ax.yaxis.label, *ax.get_xticklabels(),
               *ax.get_yticklabels(), *ax.texts]
    for artist in artists:
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
                raise ValueError(f'Figure text falls outside fixed page: {artist.get_text()}')
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--peaks', type=Path, default=DEFAULT_PEAKS)
    parser.add_argument('--query-dir', type=Path, default=DEFAULT_QUERIES)
    parser.add_argument('--chrom-sizes', type=Path, default=DEFAULT_SIZES)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--preview-stdout', action='store_true', help='In-memory visual QA only; never saves a PNG')
    args = parser.parse_args(argv)
    report = analyse(args.peaks, args.query_dir, args.chrom_sizes)
    report['script_sha256'] = sha256_file(Path(__file__))
    if args.check_only:
        print(json.dumps(report, indent=2))
        return 0
    if args.output.suffix.lower() != '.pdf':
        parser.error('PDF is the only file export format')
    if not args.preview_stdout and args.output.exists():
        parser.error('Output exists; choose a new --output path to preserve earlier results')
    fig = make_plot(report['summary'], report['fisher_tests'])
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if args.preview_stdout:
        import base64
        import io
        stream = io.BytesIO()
        fig.savefig(stream, format='png', dpi=140)
        print(json.dumps({'png': base64.b64encode(stream.getvalue()).decode()}))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as handle, mpl.rc_context({'pdf.compression': 0}):
            fig.savefig(handle, format='pdf', metadata={
                'Title': 'E2F4: promoter-region overlap',
                'Subject': 'Nominal two-sided Fisher tests: A vs full background, B vs full background, A vs B. Shared regions/paired genes can violate independence; exploratory only. No multiple-testing adjustment.'})
            # No bbox_inches='tight': preserve 360 x 288 pt.
        report.update(output_pdf=str(args.output), figure_inches=[5, 4],
                      matplotlib_version=mpl.__version__, python_version=sys.version)
        print(json.dumps(report, indent=2))
    plt.close(fig)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError) as exc:
        raise SystemExit(f'ERROR: {exc}')
