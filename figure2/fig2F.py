#!/usr/bin/env python3
"""Classify Human promoter-associated RNAs and make separate Major/Minor PDFs.

Reference: Li et al., Genome Res. 2018, doi:10.1101/gr.231449.117,
Supplemental Methods, 'Definition of transcription types'. FANTOM CAT robust
(2017-08-09, hg19/GRCh37) geneClass is mapped to four plotted RNA types;
priority: mRNA > proximal lncRNA > intergenic lncRNA > pseudogene > other RNA.
This follows the paper's RNA classification, not its complete TSS filtering.

Adaptation for ProActiv input: use the promoter's own txId associations to
FANTOM transcripts first (ignore Ensembl version suffixes). If none match,
use same-strand CAGE peaks containing the 1-based TSS. Never use the nearest
gene, the whole gene body, or a distance window. No match stays Unassigned.
FANTOM geneClass is an annotation of the associated RNA locus, NOT evidence
that an individual transcript is translated or that its CDS is unchanged.

Keep existing Major/Minor and age labels, including SP genes. 'Human' age
means no mapping detected in five other species, not proven de novo origin.
Each promoterId contributes once. Percentages use assigned non-pseudogene
promoters within each age x Major/Minor group; excluded counts are retained.
These are descriptive compositions, without significance tests/error bars.

Figure contract: descriptive RNA composition by age, separate Major/Minor
panels (quantitative small multiples), without causal/statistical claims.
Final size: 5 x 4 inches; ticks 12 pt, labels 14 pt, legend 12 pt.
Outputs: one row-level TSV, one count TSV, TWO PDFs; no combined figure.
No PNG/SVG, extra logs, backups, or automatic local copies are generated.
The plotted subset and exclusions are reported in the TSV and PDF metadata.
Three required compressed reference files are cached in reference/.
Dependencies: Python >=3.9, pandas, numpy, matplotlib.

Run: python3 plot_rna_types_final_small.py
     python3 plot_rna_types_final_small.py --overwrite
     python3 plot_rna_types_final_small.py --match-mode cage-only --check-only
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
from pathlib import Path
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo')
DEFAULT_INPUT = ROOT / '3.get_promoter_age/Human_major_minor_promoters_with_age.tsv'
DEFAULT_OUT = ROOT / '4.get_character/4.2.mRNA_lncRNA'
BASE_URL = 'https://fantom.gsc.riken.jp/5/suppl/Hon_et_al_2016/data/assembly/lv3_robust/'
PREFIX = 'FANTOM_CAT.lv3_robust.'
AGES = ['Amniote', 'Eutherian', 'Primate', 'Human']
ORDER = ['mRNA', 'proximal lncRNA', 'intergenic lncRNA', 'pseudogene', 'other RNA']
RNA_TYPES = {
    'coding_mRNA': 'mRNA',
    'lncRNA_antisense': 'proximal lncRNA',
    'lncRNA_divergent': 'proximal lncRNA',
    'lncRNA_sense_intronic': 'proximal lncRNA',
    'lncRNA_intergenic': 'intergenic lncRNA',
    'pseudogene': 'pseudogene',
    **{x: 'other RNA' for x in ['sense_overlap_RNA', 'short_ncRNA',
                              'small_RNA', 'structural_RNA', 'uncertain_coding']},
}
COLORS = {'mRNA': '#E90F44', 'proximal lncRNA': '#63ADEE',
          'intergenic lncRNA': '#FFC839', 'other RNA': '#60966D'}


def reference_file(directory, suffix):
    """Reuse existing references; otherwise download and validate gzip in memory."""
    path = directory / (PREFIX + suffix)
    if not path.exists():
        url = BASE_URL + path.name
        print('Downloading', url, flush=True)
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        gzip.decompress(data)  # Reject incomplete downloads before saving.
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def normalize_chrom(chrom):
    return 'chrM' if chrom in ('M', 'MT', 'chrM', 'chrMT') else 'chr' + chrom.removeprefix('chr')


def load_reference(directory):
    genes = pd.read_csv(reference_file(directory, 'info_table.gene.tsv.gz'),
                        sep='\t', dtype=str, usecols=['geneID', 'geneClass', 'coding_status'])
    links = pd.read_csv(reference_file(directory, 'info_table.ID_mapping.tsv.gz'),
                        sep='\t', dtype=str)
    peaks = pd.read_csv(reference_file(directory, 'CAGE_cluster.bed.gz'),
                        sep='\t', header=None, usecols=[0, 1, 2, 3, 5],
                        names=['chrom', 'lo', 'hi', 'peak', 'strand'])
    if genes.geneID.duplicated().any() or genes.isna().any().any():
        raise ValueError('Invalid FANTOM gene annotation.')
    unknown = set(genes.geneClass) - set(RNA_TYPES)
    if unknown or not set(links.geneID) <= set(genes.geneID):
        raise ValueError(f'Unrecognized classes or missing gene IDs: {unknown}')
    by_tx, by_peak = defaultdict(set), defaultdict(set)
    for row in links.itertuples(index=False):
        by_tx[row.transcriptID.split('.')[0]].add(row.geneID)
        by_peak[row.CAGEClusterID].add(row.geneID)
    peaks['chrom'] = peaks.chrom.map(normalize_chrom)
    intervals = {}
    for key, frame in peaks.groupby(['chrom', 'strand']):
        frame = frame.sort_values('lo')
        intervals[key] = (frame.lo.to_numpy(), frame.hi.to_numpy(),
                          frame.peak.to_numpy(), np.maximum.accumulate(frame.hi.to_numpy()))
    return genes.set_index('geneID').to_dict('index'), by_tx, by_peak, intervals


def overlapping_peaks(index, chrom, strand, pos):
    """BED is 0-based half-open; pos is a single 0-based genomic position."""
    if (chrom, strand) not in index:
        return []
    starts, ends, ids, max_end = index[chrom, strand]
    j = bisect_right(starts, pos) - 1
    found = []
    while j >= 0 and max_end[j] > pos:
        if ends[j] > pos:
            found.append(ids[j])
        j -= 1
    return sorted(found)


def assign_types(promoters, reference, mode):
    genes, by_tx, by_peak, intervals = reference
    records = []
    for row in promoters.itertuples(index=False):
        tids = {t.strip().split('.')[0] for t in row.txId.split(',') if t.strip()}
        matched_tids = {t for t in tids if t in by_tx}
        peaks = overlapping_peaks(intervals, normalize_chrom(row.seqnames),
                                  row.strand, int(row.start) - 1)
        if mode == 'transcript-first' and matched_tids:
            gids = set().union(*(by_tx[t] for t in matched_tids))
            method = 'transcript_id'
        else:
            gids = set().union(*(by_peak[p] for p in peaks)) if peaks else set()
            method = 'same_strand_CAGE_overlap' if gids else 'unassigned'
        raw = {genes[g]['geneClass'] for g in gids}
        types = {RNA_TYPES[c] for c in raw}
        rna_type = next((x for x in ORDER if x in types), 'Unassigned')
        records.append({
            'RNA_type': rna_type, 'annotation_method': method,
            'matched_transcript_ids': ';'.join(sorted(matched_tids)),
            'n_input_transcripts': len(tids), 'n_matched_transcripts': len(matched_tids),
            'overlapping_CAGE_peaks': ';'.join(peaks),
            'FANTOM_gene_ids': ';'.join(sorted(gids)),
            'FANTOM_gene_classes': ';'.join(sorted(raw)),
            'FANTOM_coding_status': ';'.join(sorted({genes[g]['coding_status'] for g in gids})),
            'included_in_plot': rna_type in COLORS,
        })
    return pd.concat([promoters.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def summarize(annotation):
    rows = []
    for label in ['Major', 'Minor']:
        for age in AGES:
            group = annotation.loc[(annotation.promoter_class == label) &
                                   (annotation.TSS_age_label == age)]
            counts = group.RNA_type.value_counts()
            n_plot = int(group.included_in_plot.sum())
            if not n_plot:
                raise ValueError(f'No classified non-pseudogene TSSs: {label}, {age}')
            for category in ORDER + ['Unassigned']:
                n = int(counts.get(category, 0))
                rows.append(dict(promoter_class=label, TSS_age_label=age,
                                 RNA_type=category, count=n, n_input=len(group),
                                 n_plotted=n_plot, percent_of_input=100*n/len(group),
                                 percent_in_plot=100*n/n_plot if category in COLORS else np.nan))
    return pd.DataFrame(rows)


def plot_class(counts, label, out):
    """Final small figure: unchanged data, requested colors, PDF-only export."""
    data = counts.loc[counts.promoter_class == label]
    pct = data.pivot(index='TSS_age_label', columns='RNA_type', values='percent_in_plot').reindex(AGES)
    totals = data.drop_duplicates('TSS_age_label').set_index('TSS_age_label').reindex(AGES)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=.15, right=.985, bottom=.22, top=.76)
    bottom = np.zeros(len(AGES))
    for category in ['other RNA', 'intergenic lncRNA', 'proximal lncRNA', 'mRNA']:
        ax.bar(range(4), pct[category], bottom=bottom, width=.82,
               color=COLORS[category], edgecolor='#555555', linewidth=.55)
        bottom += pct[category].to_numpy()
    np.testing.assert_allclose(bottom, 100)
    for i, n in enumerate(totals.n_plotted):
        ax.text(i, 103, f'n={n:,}', ha='center', va='bottom', fontsize=10)
    ax.set(ylim=(0, 114), xlim=(-.58, 3.58), yticks=range(0, 101, 20), xticks=range(4))
    ax.set_ylabel('% of TSSs', fontsize=14)
    ax.set_xticklabels(AGES, rotation=30, ha='right', rotation_mode='anchor')
    ax.spines['left'].set_bounds(0, 100)
    ax.tick_params(axis='both', labelsize=12)
    ax.tick_params(axis='x', length=0)
    handles = [Patch(facecolor=COLORS[c], edgecolor='#555555', linewidth=.55, label=c) for c in COLORS]
    fig.legend(handles=handles, ncol=2, loc='upper center', bbox_to_anchor=(.56, .985),
               fontsize=12, handlelength=.85, handletextpad=.45, columnspacing=1.0)
    unknown = int(data.loc[data.RNA_type == 'Unassigned', 'count'].sum())
    pseudo = int(data.loc[data.RNA_type == 'pseudogene', 'count'].sum())
    note = (f'Within each age group, denominator = assigned non-pseudogene promoters. '
            f'{label}: plotted {totals.n_plotted.sum():,}/{totals.n_input.sum():,}; '
            f'excluded {unknown:,} unassigned and {pseudo:,} pseudogene. '
            'n counts promoter IDs, not unique genomic coordinates or replicates.')
    fig.savefig(out, format='pdf', metadata={
        'Title': f'Human {label} promoter RNA types by age',
        'Subject': note,
        'Keywords': 'FANTOM CAT; GRCh37; Li et al. 2018; final-figure-small',
    })
    plt.close(fig)
    print(note)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--match-mode', choices=['transcript-first', 'cage-only'], default='transcript-first')
    parser.add_argument('--check-only', action='store_true', help='Compute/check counts without writing result files.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    outputs = [args.outdir / name for name in ['Human_major_minor_RNA_annotation.tsv',
               'Human_major_minor_RNA_counts.tsv', 'Human_major_RNA_types_by_age.pdf',
               'Human_minor_RNA_types_by_age.pdf']]
    if not args.check_only and not args.overwrite and any(p.exists() for p in outputs):
        parser.error('Results already exist; use --overwrite to replace these four outputs.')
    p = pd.read_csv(args.input, sep='\t', dtype=str, keep_default_na=False)
    required = ['promoterId', 'geneId', 'seqnames', 'start', 'strand', 'txId',
                'internalPromoter', 'promoter_class', 'TSS_age_label']
    if not set(required) <= set(p) or p.empty:
        raise ValueError('Missing required columns or empty input.')
    if p.promoterId.duplicated().any() or not p.promoter_class.isin(['Major', 'Minor']).all():
        raise ValueError('Duplicate promoter IDs or invalid Major/Minor labels.')
    if not p.TSS_age_label.isin(AGES).all() or not p.strand.isin(['+', '-']).all():
        raise ValueError('Unexpected age or strand.')
    if not p.internalPromoter.str.lower().isin(['false', '0']).all():
        raise ValueError('Input contains internal promoters.')
    if not p.start.str.fullmatch(r'[1-9][0-9]*').all():
        raise ValueError('TSS must be a positive, 1-based integer.')
    reference = load_reference(args.reference_dir or args.outdir / 'reference')
    annotation = assign_types(p, reference, args.match_mode)
    counts = summarize(annotation)
    assert len(annotation) == len(p) == counts['count'].sum()
    assert annotation[required].equals(p[required])
    print('Method:', args.match_mode)
    print(annotation.groupby(['promoter_class', 'annotation_method']).size().to_string())
    print(counts.pivot_table(index=['promoter_class', 'TSS_age_label'], columns='RNA_type',
                            values='count', fill_value=0).to_string())
    if args.check_only:
        return
    args.outdir.mkdir(parents=True, exist_ok=True)
    annotation.to_csv(outputs[0], sep='\t', index=False)
    counts.to_csv(outputs[1], sep='\t', index=False, float_format='%.8f', na_rep='NA')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 12, 'axes.labelsize': 14, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .8, 'legend.frameon': False})
    for label, path in zip(['Major', 'Minor'], outputs[2:]):
        plot_class(counts, label, path)
    for path in outputs:
        print('Saved:', path)


if __name__ == '__main__':
    main()

