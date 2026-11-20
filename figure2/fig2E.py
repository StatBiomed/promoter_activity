#!/usr/bin/env python3
"""Human MP promoter cross-species TSS contexts: 5UTR/3UTR/Intron/Exon.

Python >=3.9; pandas, numpy, matplotlib. Extend the existing all_tissue_MP
workflow without changing its inputs, mapping rules, tissue-specific labels,
age strata, direct ortholog restriction, paired denominators or exclusions.
The original scripts and input tables are read-only dependencies.

Region definitions match step 4.1 exactly: 1-based inclusive TSS on BOTH
strands; ALL transcripts of the same gene/chromosome/strand; exonic UTRs
outside CDS + start/stop codon bounds (5'/3' reversed on minus); noncoding
transcripts have no inferred UTRs; introns are gaps within each transcript.
Pool feature hits, then assign 5UTR > 3UTR > Intron > Exon > Unassigned.
Exon is residual, NOT strictly CDS. Unassigned is not relabelled Intron.
Fresh Human GTF annotations MUST equal the existing step-4.1 annotations.
The same rules are applied to each target's own assembly-matched GTF.

One Human promoter contributes once per target species, including when
multiple direct orthologs/loci match. Pool matches over those loci/genes and
apply the same priority; retain detailed ambiguous evidence in the TSV.
Major/Minor denote HUMAN tissue labels, NOT target activity or classification.
These are mapped coordinate contexts, not proof of evolutionary conversion.

Figure contract: descriptive quantitative grid, Human -> five targets;
separate Major/Minor, All + four mapping ages; 12 x 4.8 inch editable PDFs.
Each panel is normalized to its own eligible paired n; coverage = n/N.
N includes all input promoters of that tissue/age/class. Missing pairs are
excluded from flows but fully retained in detail/coverage/exclusion tables.
There are 16 possible transitions (4 x 4), including zero-count transitions.

Default: 10 PDFs + 1 detail TSV.gz per tissue, three shared TSVs and manifest.
--pdf-layout multipage: two five-page PDFs per tissue instead.
--check-only: full annotation and consistency validation, no result files.
--self-test: synthetic annotation/boundary/strand tests only.
--plot-only: replot validated saved tables, no GTF annotation. It never deletes
superseded PDFs; use a new outdir for fresh analyses. No overwrite by default.

Examples:
  python plot_MP_5UTR_3UTR_intron_exon_by_tissue_age.py --self-test
  python plot_MP_5UTR_3UTR_intron_exon_by_tissue_age.py --check-only
  python plot_MP_5UTR_3UTR_intron_exon_by_tissue_age.py --tissues Testis
  python plot_MP_5UTR_3UTR_intron_exon_by_tissue_age.py
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo')
HERE = Path(__file__).resolve().parent
REGIONS = ['5UTR', '3UTR', 'Intron', 'Exon']
PRIORITY = '5UTR > 3UTR > Intron > Exon > Unassigned'
COLORS = {'5UTR': '#FBB463', '3UTR': '#80B1D3', 'Intron': '#F47F72', 'Exon': '#BDBADB'}
LABELS = {'5UTR': "5'UTR", '3UTR': "3'UTR", 'Intron': 'Intron', 'Exon': 'Exon'}
STEM = 'all_tissue_MP_5UTR_3UTR_intron_exon'
EXTRA = ['assigned_region', 'matched_regions', 'n_matched_region_types',
         'unassigned_reason', 'n_reference_transcripts']
EXPECTED_AGES = ['All', 'Amniote', 'Eutherian', 'Primate', 'Human']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(path, name):
    require(path.is_file(), f'Missing reference script: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def norm_gene(value):
    return value.strip().split('.')[0]


def norm_chrom(value):
    value = value.strip()
    value = value[3:] if value.startswith('chr') else value
    return 'MT' if value == 'M' else value


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(set(intervals)):
        require(1 <= start <= end, f'Invalid 1-based closed interval: {start}, {end}')
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def contains(intervals, point):
    i = bisect.bisect_right(intervals, (point, float('inf'))) - 1
    return i >= 0 and intervals[i][0] <= point <= intervals[i][1]


def transcript_regions(exons, coding, strand):
    """Identical closed-interval/codon-inclusive structural rule to step 4.1."""
    require(strand in {'+', '-'}, 'Invalid transcript strand')
    exons = merge_intervals(exons)
    require(exons, 'Transcript has no exons')
    features = {'Exon': exons, '5UTR': [], '3UTR': [],
                'Intron': [(a[1] + 1, b[0] - 1) for a, b in zip(exons, exons[1:])]}
    if coding:
        for start, end in coding:
            i = bisect.bisect_right(exons, (start, float('inf'))) - 1
            require(1 <= start <= end and i >= 0 and exons[i][0] <= start <= end <= exons[i][1],
                    'Coding feature is not contained in an exon')
        low, high = min(s for s, _ in coding), max(e for _, e in coding)
        for start, end in exons:
            if start < low:
                features['5UTR' if strand == '+' else '3UTR'].append((start, min(end, low - 1)))
            if end > high:
                features['3UTR' if strand == '+' else '5UTR'].append((max(start, high + 1), end))
    return features


def parse_gtf(lines, wanted):
    transcripts, headers = {}, []
    for number, line in enumerate(lines, 1):
        if line.startswith('#'):
            headers.append(line.strip())
            continue
        if not line.strip():
            continue
        f = line.rstrip('\n').split('\t')
        require(len(f) == 9, f'Malformed GTF line {number}')
        if f[2] not in {'exon', 'CDS', 'start_codon', 'stop_codon'}:
            continue
        attrs = dict(re.findall(r'(\w+) "([^"]*)"', f[8]))
        gene = norm_gene(attrs.get('gene_id', ''))
        if gene not in wanted:
            continue
        require(bool(attrs.get('transcript_id')), f'Missing transcript ID at line {number}')
        key = (gene, norm_chrom(f[0]), f[6], attrs['transcript_id'])
        record = transcripts.setdefault(key, {'exons': [], 'coding': []})
        record['exons' if f[2] == 'exon' else 'coding'].append((int(f[3]), int(f[4])))
    index, tx_counts = defaultdict(lambda: defaultdict(list)), Counter()
    for (gene, chrom, strand, tx), record in transcripts.items():
        require(record['exons'], f'Coding transcript without exons: {tx}')
        key = (gene, chrom, strand)
        tx_counts[key] += 1
        for name, intervals in transcript_regions(record['exons'], record['coding'], strand).items():
            index[key][name].extend(intervals)
    for features in index.values():
        for name in features:
            features[name] = merge_intervals(features[name])
    return dict(index), tx_counts, headers


def read_gtf(path, build, wanted):
    print(f'Reading {path.name}: {len(wanted):,} requested genes', flush=True)
    with path.open() as stream:
        index, counts, headers = parse_gtf(stream, wanted)
    require(any(build in h for h in headers), f'GTF assembly mismatch: {path}')
    print(f'  {sum(counts.values()):,} transcripts; {len(index):,} gene loci', flush=True)
    return index, counts


def query(index, gene, chrom, position, strand):
    require(strand in {'+', '-'} and position > 0, 'Invalid query TSS')
    features = index.get((norm_gene(gene), norm_chrom(chrom), strand), {})
    return [region for region in REGIONS if contains(features.get(region, []), position)]


def human_annotations(ref, promoters, gtf_dir):
    index, counts = read_gtf(gtf_dir / ref.SPECIES['Human'][1], ref.SPECIES['Human'][2],
                             set(promoters.geneId.map(norm_gene)))
    genes, output = {k[0] for k in index}, {}
    for row in promoters.itertuples(index=False):
        key = (norm_gene(row.geneId), norm_chrom(row.seqnames), row.strand)
        matches = query(index, row.geneId, row.seqnames, int(row.start), row.strand)
        reason = '' if matches else ('gene_not_in_gtf' if key[0] not in genes else
                  'chromosome_or_strand_mismatch' if key not in index else 'outside_all_transcript_regions')
        output[row.promoterId] = dict(assigned_region=matches[0] if matches else 'Unassigned',
            matched_regions=';'.join(matches), n_matched_region_types=len(matches),
            unassigned_reason=reason, n_reference_transcripts=counts.get(key, 0))
    return output


def compare_human_to_step4(ref, pipeline, frames, human, region_dir):
    """Exact per-promoter annotation AND unchanged tissue-input-column check."""
    audit = {}
    for tissue, frame in frames.items():
        path = region_dir / tissue / f'Human_{tissue}_MP_TSS_region_annotation.tsv'
        previous = ref.pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
        require(set(frame.columns) | set(EXTRA) <= set(previous), f'Invalid step-4.1 schema: {tissue}')
        previous.index = previous.promoterId.map(pipeline.promoter_key).to_numpy()
        require(not previous.index.duplicated().any() and set(previous.index) == set(frame.index),
                f'Step-4.1 promoter population mismatch: {tissue}')
        previous = previous.loc[frame.index]
        for name in frame.columns:
            require(previous[name].tolist() == frame[name].tolist(), f'Step-4.1 input conflict: {tissue}/{name}')
        for pid in frame.index:
            for name in EXTRA:
                require(str(human[str(pid)][name]) == str(previous.at[pid, name]),
                        f'Fresh annotation differs from step 4.1: {tissue}/promoter {pid}/{name}')
        audit[tissue] = dict(promoters_checked=len(frame), annotation_mismatches=0,
                             all_input_fields_identical=True, annotation_fields_checked=EXTRA)
        print(f'STEP 4.1 MATCH: {tissue}, {len(frame):,} promoters, all five annotation fields identical.', flush=True)
    return audit


def annotate_targets(ref, mappings, human, gtf_dir):
    output = []
    for species in ref.TARGETS:
        group = mappings.loc[mappings.target_species == species]
        wanted = {norm_gene(g) for text in group.linked_target_gene_ids for g in text.split(';') if g}
        index, _ = read_gtf(gtf_dir / ref.SPECIES[species][1], ref.SPECIES[species][2], wanted)
        available = {key[0] for key in index}
        for row in group.itertuples(index=False):
            h = human[row.human_promoter_id]
            genes = [g for g in row.linked_target_gene_ids.split(';') if g]
            loci = [text.rsplit(':', 2) for text in row.target_loci_1based.split(';') if text]
            require(len(loci) == int(row.n_coherent_loci), 'Coherent locus count mismatch')
            evidence, gene_hits = [], {r: set() for r in REGIONS}
            for chrom, pos, strand in loci:
                for gene in genes:
                    hits = query(index, gene, chrom, int(pos), strand)
                    for region in hits:
                        gene_hits[region].add(gene)
                    if hits:
                        evidence.append(dict(gene=gene, locus=f'{chrom}:{pos}:{strand}',
                                             assigned_region=hits[0], matched_regions=hits))
            target_hits = [region for region in REGIONS if gene_hits[region]]
            hr, tr = h['assigned_region'], target_hits[0] if target_hits else 'Unassigned'
            reason = ('Human_unassigned' if hr == 'Unassigned' else
                      'No_coherent_TSS_mapping' if not loci else
                      'No_direct_ortholog' if not genes else
                      'Target_gene_not_in_GTF' if not any(norm_gene(g) in available for g in genes) else
                      'Target_unassigned' if tr == 'Unassigned' else 'Included')
            record = row._asdict()
            record.update(human_region=hr, target_region=tr,
                human_matched_regions=h['matched_regions'], target_matched_regions=';'.join(target_hits),
                human_n_matched_region_types=h['n_matched_region_types'], target_n_matched_region_types=len(target_hits),
                human_unassigned_reason=h['unassigned_reason'], human_n_reference_transcripts=h['n_reference_transcripts'],
                target_locus_gene_evidence=json.dumps(evidence, separators=(',', ':')),
                target_distinct_locus_gene_regions=';'.join(r for r in REGIONS if any(e['assigned_region'] == r for e in evidence)),
                annotation_priority=PRIORITY, included_in_plot=int(reason == 'Included'), exclusion_reason=reason)
            for region in REGIONS:
                record[f'target_{region}_gene_ids'] = ';'.join(sorted(gene_hits[region]))
            output.append(record)
        print(f'{species}: {len(group):,} promoter/species annotations completed', flush=True)
    return ref.pd.DataFrame(output)


def summarize(ref, pipeline, details, tissue, previous):
    flows, coverage, exclusions = [], [], []
    age_rows = previous.loc[previous.tissue == tissue]
    require(len(age_rows) == 4 and set(age_rows.TSS_age_label) == set(pipeline.AGES), 'Incomplete upstream ages')
    expected = age_rows.set_index('TSS_age_label')
    for age in pipeline.STRATA:
        subset = details if age == 'All' else details.loc[details.TSS_age_label == age]
        for kind in ['Major', 'Minor']:
            for species in ref.TARGETS:
                group = subset.loc[(subset.promoter_class == kind) & (subset.target_species == species)]
                total = len(group)
                column = 'major_n' if kind == 'Major' else 'minor_n'
                wanted = int(expected[column].sum()) if age == 'All' else int(expected.loc[age, column])
                require(total == wanted, f'Input population changed: {tissue}/{age}/{kind}/{species}')
                included = group.loc[group.included_in_plot == 1]
                n, tally = len(included), included.groupby(['human_region', 'target_region']).size()
                common = dict(tissue=tissue, age_stratum=age, promoter_class=kind, target_species=species)
                for source in REGIONS:
                    for target in REGIONS:
                        count = int(tally.get((source, target), 0))
                        flows.append(dict(**common, human_region=source, target_region=target,
                            n_promoters=count, plotted_pairs=n, percent_of_plotted_pairs=100 * count / n if n else float('nan')))
                require(int(tally.sum()) == n and included.human_region.isin(REGIONS).all()
                        and included.target_region.isin(REGIONS).all(), 'Unknown included region')
                reason_counts = group.exclusion_reason.value_counts()
                require(int(reason_counts.sum()) == total and int(reason_counts.get('Included', 0)) == n,
                        'Exclusion/coverage mismatch')
                for reason in pipeline.REASONS:
                    exclusions.append(dict(**common, exclusion_reason=reason, n_promoters=int(reason_counts.get(reason, 0))))
                coverage.append(dict(**common, total_human_promoters=total, plotted_pairs=n, excluded_promoters=total - n,
                    percent_of_input=100 * n / total if total else float('nan'),
                    human_multiple_region_types=int(group.human_n_matched_region_types.gt(1).sum()),
                    target_multiple_region_types=int(group.target_n_matched_region_types.gt(1).sum()),
                    multiple_coherent_loci=int(group.n_coherent_loci.astype(int).gt(1).sum()), annotation_priority=PRIORITY))
    return tuple(ref.pd.DataFrame(rows) for rows in [flows, coverage, exclusions])


def draw_page(ref, flows, coverage, tissue, kind, age):
    plt, np = ref.plt, ref.np
    cov = coverage.loc[(coverage.age_stratum == age) & (coverage.promoter_class == kind)].set_index('target_species')
    require(cov.total_human_promoters.nunique() == 1, 'Inconsistent N')
    total = int(cov.total_human_promoters.iloc[0])
    fig, axes = plt.subplots(1, 5, figsize=(12, 4.8))
    fig.subplots_adjust(left=.025, right=.975, bottom=.28, top=.81, wspace=.37)
    fig.suptitle(f'Human | {tissue} | {kind} promoters | {"All ages" if age == "All" else age}', y=.98, fontsize=16)
    fig.text(.5, .91, f'MP genes only; Human TSS contexts mapped to five species (N = {total:,})', ha='center', fontsize=11)
    for ax, species in zip(axes, ref.TARGETS):
        group = flows.loc[(flows.age_stratum == age) & (flows.promoter_class == kind) & (flows.target_species == species)]
        require(len(group) == len(REGIONS) ** 2, 'Expected 16 four-region transitions')
        matrix = group.pivot(index='human_region', columns='target_region', values='n_promoters').reindex(
            index=REGIONS, columns=REGIONS).to_numpy(dtype=int)
        n = int(cov.loc[species, 'plotted_pairs'])
        require(matrix.sum() == n and (matrix >= 0).all(), 'Flow count mismatch')
        ax.set(xlim=(-.05, 1.05), ylim=(0, 1))
        ax.axis('off')
        for x, label in [(.075, 'Human'), (.925, species)]:
            ax.text(x, 1.045, label, transform=ax.transAxes, ha='center', fontsize=10)
        if n:
            fractions = matrix / n
            left, right = fractions.sum(axis=1), fractions.sum(axis=0)
            bases = [np.r_[0, np.cumsum(v)[:-1]] for v in [left, right]]
            used_left, used_right = np.zeros(len(REGIONS)), np.zeros(len(REGIONS))
            for i in range(len(REGIONS)):
                for j in range(len(REGIONS)):
                    height = fractions[i, j]
                    if not height:
                        continue
                    a, b = bases[0][i] + used_left[i], bases[1][j] + used_right[j]
                    vertices = [(.16, a), (.43, a), (.57, b), (.84, b), (.84, b + height),
                                (.57, b + height), (.43, a + height), (.16, a + height), (.16, a)]
                    codes = [ref.MplPath.MOVETO] + [ref.MplPath.CURVE4] * 3 + [ref.MplPath.LINETO] + [ref.MplPath.CURVE4] * 3 + [ref.MplPath.CLOSEPOLY]
                    ax.add_patch(ref.PathPatch(ref.MplPath(vertices, codes), facecolor=COLORS[REGIONS[j]], alpha=.6, edgecolor='none'))
                    used_left[i] += height
                    used_right[j] += height
            require(np.allclose(used_left, left) and np.allclose(used_right, right), 'Ribbon imbalance')
            for x, heights, bottom in [(0, left, bases[0]), (.84, right, bases[1])]:
                for region, height, start in zip(REGIONS, heights, bottom):
                    if height:
                        ax.add_patch(ref.Rectangle((x, start), .16, height, facecolor=COLORS[region], edgecolor='white', linewidth=.5, zorder=2))
                        if height >= .08:
                            ax.text(x + .08, start + height / 2, f'{100 * height:.1f}%', ha='center', va='center', rotation=90, fontsize=9, zorder=3)
                ax.add_patch(ref.Rectangle((x, 0), .16, 1, fill=False, edgecolor='#333333', linewidth=.8, zorder=4))
        else:
            ax.text(.5, .5, 'No eligible pairs' if total else 'No input promoters', ha='center', fontsize=9)
        ax.text(.5, -.075, f'n = {n:,}', transform=ax.transAxes, ha='center', fontsize=11)
        ax.text(.5, -.15, f'{100 * n / total:.1f}% of this group' if total else 'Coverage: NA', transform=ax.transAxes, ha='center', fontsize=9)
    fig.legend(handles=[ref.Patch(facecolor=COLORS[r], label=LABELS[r]) for r in REGIONS],
               ncol=4, loc='lower center', bbox_to_anchor=(.5, .12), frameon=False, fontsize=11)
    notes = ["Own-species GTF; 5'UTR > 3'UTR > Intron > Exon. Exon is residual, not strictly CDS.",
             'Each panel uses its own paired n; N includes all MP promoters in this tissue/age/class.',
             'Human labels retained. Mapped context is not target promoter activity or proof of evolutionary conversion.']
    for y, note in zip([.08, .05, .02], notes):
        fig.text(.5, y, note, ha='center', fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(ref.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1,
                    'Clipped text: ' + text.get_text())
    return fig


def export_pdfs(ref, flows, coverage, folder, tissue, kind, layout):
    from matplotlib.backends.backend_pdf import PdfPages
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'Human_{tissue}_MP_{kind.lower()}_5UTR_3UTR_intron_exon_'
    meta = {'Title': f'Human {tissue} MP {kind}: four-region mapped contexts',
            'Subject': PRIORITY + '; paired eligible promoters; descriptive, not target activity or evolutionary conversion.',
            'CreationDate': None, 'ModDate': None}
    paths = []
    if layout == 'multipage':
        path = folder / (stem + 'by_age.pdf')
        with PdfPages(path, metadata=meta) as pdf:
            for age in EXPECTED_AGES:
                fig = draw_page(ref, flows, coverage, tissue, kind, age)
                pdf.savefig(fig)
                ref.plt.close(fig)
        paths.append(path)
    else:
        for age in EXPECTED_AGES:
            path = folder / (stem + age + '.pdf')
            fig = draw_page(ref, flows, coverage, tissue, kind, age)
            fig.savefig(path, metadata=meta)
            ref.plt.close(fig)
            paths.append(path)
    return paths


def self_test():
    exons, coding = [(100, 199), (300, 399)], [(150, 199), (300, 350)]
    plus = transcript_regions(exons, coding, '+')
    minus = transcript_regions(exons, coding, '-')
    require(plus['5UTR'] == [(100, 149)] and plus['3UTR'] == [(351, 399)], 'Plus UTR rule')
    require(minus['5UTR'] == plus['3UTR'] and minus['3UTR'] == plus['5UTR'], 'Minus UTR rule')
    require(plus['Intron'] == [(200, 299)], 'Intron boundaries')
    require(not contains(plus['Intron'], 199) and contains(plus['Intron'], 200) and contains(plus['Intron'], 299)
            and not contains(plus['Intron'], 300), 'Closed-interval boundaries')
    require(not transcript_regions(exons, [], '+')['5UTR'], 'Noncoding UTR incorrectly inferred')
    require(not transcript_regions([(1, 10), (11, 20)], [], '+')['Intron'], 'Adjacent exons create empty intron')
    require(not transcript_regions([(1, 20)], [], '+')['Intron'], 'Single-exon intron')
    lines = []
    def feature(tx, kind, start, end, strand='+'):
        lines.append(f'1\ttest\t{kind}\t{start}\t{end}\t.\t{strand}\t.\tgene_id "G.1"; transcript_id "{tx}";\n')
    for start, end in exons:
        feature('A', 'exon', start, end)
    for start, end in coding:
        feature('A', 'CDS', start, end)
    feature('A', 'stop_codon', 351, 353)
    feature('B', 'exon', 200, 250)
    feature('B', 'CDS', 220, 250)
    feature('N', 'exon', 500, 600, '-')
    index, counts, _ = parse_gtf(lines, {'G'})
    require(query(index, 'G.2', 'chr1', 200, '+') == ['5UTR', 'Intron', 'Exon'], 'All-isoform priority')
    require(query(index, 'G', '1', 230, '+') == ['Intron', 'Exon'], 'Intron before residual Exon')
    require(query(index, 'G', '1', 351, '+') == ['Exon'] and query(index, 'G', '1', 354, '+') == ['3UTR', 'Exon'], 'Stop codon included')
    require(query(index, 'G', 'chr1', 550, '-') == ['Exon'], 'Noncoding minus exon')
    require(not query(index, 'G', 'chr1', 550, '+') and not query(index, 'G', 'chr2', 200, '+')
            and not query(index, 'OTHER', 'chr1', 200, '+'), 'Gene/chromosome/strand restriction')
    require(counts['G', '1', '+'] == 2 and not query(index, 'G', '1', 99, '+'), 'Transcript count/outside')
    require(norm_chrom('chrM') == 'MT', 'Mitochondrial chromosome normalization')
    print('SELF-TEST PASSED: strands, codons, noncoding transcripts, isoform overlap, boundaries, restrictions.', flush=True)


def publish(stage, outdir, files, overwrite, protected):
    for rel in files:
        destination = outdir / rel
        require(not destination.is_symlink() and destination.resolve().is_relative_to(outdir.resolve())
                and str(destination.resolve()) not in protected, f'Unsafe output: {destination}')
        require(overwrite or not destination.exists(), f'Output exists: {destination}; choose another --outdir or --overwrite')
    for rel in files:
        destination = outdir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if overwrite:
            os.replace(stage / rel, destination)
        else:
            os.link(stage / rel, destination)


def replot(ref, args):
    path = args.outdir / 'run_manifest.json'
    old = json.loads(path.read_text())
    require(old['status'] == 'complete' and old['region_priority'] == REGIONS + ['Unassigned'], 'Incompatible saved run')
    require(args.tissues is None or args.tissues == old['tissues'], '--plot-only must cover the saved tissue set')
    for source, digest in old['source_sha256'].items():
        require(checksum(source) == digest, f'Source changed: {source}')
    for name, digest in old['output_sha256'].items():
        require(checksum(args.outdir / name) == digest, f'Saved output changed: {name}')
    tables = [ref.pd.read_csv(args.outdir / f'{STEM}_{s}.tsv', sep='\t') for s in ['flows', 'coverage']]
    with tempfile.TemporaryDirectory(prefix='.four_region_replot_', dir=args.outdir) as temporary:
        stage, paths = Path(temporary), []
        for tissue in old['tissues']:
            flows, cov = [t.loc[t.tissue == tissue] for t in tables]
            for kind in ['Major', 'Minor']:
                paths.extend(export_pdfs(ref, flows, cov, stage / tissue, tissue, kind, args.pdf_layout))
        new = {**old, 'replotted_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': checksum(__file__),
               'pdf_layout': args.pdf_layout, 'command': sys.argv,
               'output_sha256': {**old['output_sha256'], **{p.relative_to(stage).as_posix(): checksum(p) for p in paths}}}
        (stage / 'run_manifest.json').write_text(json.dumps(new, indent=2) + '\n')
        protected = set(old['source_sha256']) | {str(Path(__file__).resolve())}
        publish(stage, args.outdir, [p.relative_to(stage) for p in paths] + [Path('run_manifest.json')], args.overwrite, protected)
    print(f'REPLOT COMPLETE: {len(paths)} PDFs; no old files deleted.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parent = ROOT / '5.get_promoter_exon_intron_change'
    region_parent = ROOT / '4.get_character/4.1.CDS_5UTR_3UTR'
    parser.add_argument('--input-dir', type=Path, default=ROOT / '3.get_promoter_age/all_tissues_MP')
    parser.add_argument('--original-script', type=Path, default=parent / 'all_tissue_MP/plot_MP_exon_intron_by_tissue_age.py')
    parser.add_argument('--reference-script', type=Path, default=parent / 'try/plot_major_minor_exon_intron.py')
    parser.add_argument('--region-dir', type=Path, default=region_parent / 'all_tissues_MP')
    parser.add_argument('--mappings', type=Path, default=parent / 'results/Human_major_minor_mapping_details.tsv')
    parser.add_argument('--gtf-dir', type=Path, default=ROOT.parent / 'RNA_seq_organ_species/ref_annotation')
    parser.add_argument('--outdir', type=Path, default=HERE)
    parser.add_argument('--tissues', nargs='+')
    parser.add_argument('--pdf-layout', choices=['separate', 'multipage'], default='separate')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--plot-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    pipeline = load_module(args.original_script, '_original_MP')
    require(pipeline.STRATA == EXPECTED_AGES, 'Original age strata changed')
    ref = pipeline.load_reference(args.reference_script)
    region_script = args.region_dir / 'plot_human_MP_tss_regions_by_age_by_tissue.py'
    region = load_module(region_script, '_step4_MP')
    region_ref_path = args.region_dir.parent / 'plot_human_tss_regions_by_age.py'
    region_ref = region.load_reference(region_ref_path)
    require(list(region.REGIONS) == REGIONS + ['Unassigned'] and
            all(region_ref.COLORS[r] == COLORS[r] for r in REGIONS), 'Step-4.1 region rule/palette changed')
    ref.plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'axes.linewidth': .8})
    if args.plot_only:
        replot(ref, args)
        return
    files, found = region.discover(args.input_dir, args.tissues)
    tissues = list(files)
    upstream_path, ages_path = args.input_dir / 'run_manifest.json', args.input_dir / 'all_tissues_MP_promoter_age_counts.tsv'
    upstream = json.loads(upstream_path.read_text())
    require(upstream['status'] == 'complete' and upstream['species'] == 'Human' and set(upstream['tissues']) == found,
            'Invalid upstream MP age manifest')
    region_manifest_path = args.region_dir / 'run_manifest.json'
    region_manifest = json.loads(region_manifest_path.read_text())
    require(region_manifest['status'] == 'complete' and region_manifest['region_priority'] == REGIONS + ['Unassigned'],
            'Invalid step-4.1 manifest')
    region_tables = [args.region_dir / t / f'Human_{t}_MP_TSS_region_annotation.tsv' for t in tissues]
    sources = [args.original_script, args.reference_script, region_script, region_ref_path, region_manifest_path,
               args.mappings, upstream_path, ages_path, *files.values(), *region_tables,
               *[args.gtf_dir / v[1] for v in ref.SPECIES.values()]]
    hashes = {str(p.resolve()): checksum(p) for p in sources}
    for path in [ages_path, *files.values()]:
        require(checksum(path) == upstream['output_sha256'][path.relative_to(args.input_dir).as_posix()], f'Changed MP input: {path}')
    for path in region_tables:
        require(checksum(path) == region_manifest['output_sha256'][path.relative_to(args.region_dir).as_posix()], f'Changed region reference: {path}')
    require(args.outdir.resolve() not in {args.input_dir.resolve(), args.region_dir.resolve(), args.original_script.parent.resolve()},
            'Do not write into a reference directory')
    relative = [Path(f'{STEM}_{s}.tsv') for s in ['flows', 'coverage', 'exclusions']]
    for tissue in tissues:
        relative.append(Path(tissue) / f'Human_{tissue}_MP_5UTR_3UTR_intron_exon_details.tsv.gz')
        for kind in ['major', 'minor']:
            for suffix in (['by_age'] if args.pdf_layout == 'multipage' else EXPECTED_AGES):
                relative.append(Path(tissue) / f'Human_{tissue}_MP_{kind}_5UTR_3UTR_intron_exon_{suffix}.pdf')
    relative.append(Path('run_manifest.json'))
    protected = set(hashes) | {str(Path(__file__).resolve())}
    for rel in relative:
        target = args.outdir / rel
        require(not target.is_symlink() and target.resolve().is_relative_to(args.outdir.resolve())
                and str(target.resolve()) not in protected, f'Unsafe output: {target}')
        require(args.check_only or args.overwrite or not target.exists(), f'Results exist: {target}; choose --outdir or --overwrite')
    frames = {t: pipeline.load_tissue(ref, files[t], upstream['selection'][t]) for t in tissues}
    union = pipeline.combine_tissues(ref, frames)
    mappings = pipeline.load_mappings(ref, args.mappings, union)
    promoters = union.copy()
    promoters.promoterId = promoters.index.astype(str)
    print(f'{len(union):,} unique Human MP promoters across {len(tissues)} tissues.', flush=True)
    human = human_annotations(ref, promoters, args.gtf_dir)
    concordance = compare_human_to_step4(ref, pipeline, frames, human, args.region_dir)
    annotated = annotate_targets(ref, mappings, human, args.gtf_dir)
    require(len(annotated) == len(union) * len(ref.TARGETS), 'Wrong annotation row count')
    previous = ref.pd.read_csv(ages_path, sep='\t')
    audit, summaries = {}, [[], [], []]
    if not args.check_only:
        args.outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.four_region_', dir=args.outdir if not args.check_only else None) as temporary:
        stage = Path(temporary)
        for tissue, frame in frames.items():
            details = pipeline.join_tissue(ref, frame, annotated)
            tables = summarize(ref, pipeline, details, tissue, previous)
            for parts, table in zip(summaries, tables):
                parts.append(table)
            audit[tissue] = dict(MP_genes=int(frame.geneId.nunique()), input_promoters=len(frame), detail_rows=len(details),
                all_input_fields_preserved=True, step4_annotation_concordance=concordance[tissue],
                paired_n=tables[1].loc[tables[1].age_stratum == 'All', ['promoter_class', 'target_species', 'plotted_pairs']].to_dict('records'))
            print(f'VALIDATED {tissue}: {len(frame):,} promoters; {len(details):,} mapped-context records.', flush=True)
            if args.check_only:
                continue
            folder = stage / tissue
            folder.mkdir()
            details.to_csv(folder / f'Human_{tissue}_MP_5UTR_3UTR_intron_exon_details.tsv.gz', sep='\t', index=False,
                           compression={'method': 'gzip', 'mtime': 0})
            for kind in ['Major', 'Minor']:
                export_pdfs(ref, tables[0], tables[1], folder, tissue, kind, args.pdf_layout)
        require(all(checksum(p) == digest for p, digest in hashes.items()), 'Source changed during run')
        if args.check_only:
            print('CHECK PASSED: fresh Human annotations match step 4.1; all tissue fields, mappings and counts validated. No result files written.', flush=True)
            return
        for suffix, parts in zip(['flows', 'coverage', 'exclusions'], summaries):
            ref.pd.concat(parts, ignore_index=True).to_csv(stage / f'{STEM}_{suffix}.tsv', sep='\t', index=False,
                                                         float_format='%.8f', na_rep='NA')
        manifest = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(), species='Human', tissues=tissues,
            source_sha256=hashes, script_sha256=checksum(__file__), command=sys.argv, unique_promoters=len(union),
            region_priority=REGIONS + ['Unassigned'], colors=COLORS, age_strata=EXPECTED_AGES, pdf_layout=args.pdf_layout,
            annotation='Own-species GTF; all same-gene/chromosome/strand transcripts; codon-inclusive coding bounds; transcript-specific introns',
            coordinate_convention='1-based inclusive TSS; no strand-dependent coordinate shift',
            mappings='Inherited coherent exact TSS/windows and direct linked orthologs; no new liftOver or nearest-gene assignment',
            multi_hit_rule='Pool feature hits across linked genes and coherent loci, apply region priority once per Human promoter/target species',
            labels='Unchanged HUMAN tissue-specific Major/Minor/activity/age; mapping_global_promoter_class is audit-only',
            denominator='Eligible paired n per tissue/age/class/target; coverage n/N, where N includes all source MP promoters',
            empty_groups='n=0 gives NA percentages and no inferred flow; Unassigned remains in details and exclusions',
            figure_size_inches=[12, 4.8], statistics='Descriptive only', validation=audit,
            caveats=['Residual Exon is not strictly CDS', 'Mapped target context is not target promoter activity',
                     'GTF annotation is not tissue-specific', 'Mapping ages/region differences do not demonstrate evolutionary conversion'],
            output_sha256={p.relative_to(stage).as_posix(): checksum(p) for p in stage.rglob('*') if p.is_file()})
        (stage / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        publish(stage, args.outdir, relative, args.overwrite, protected)
    print(f'COMPLETE: {len(tissues)} tissues; {len([p for p in relative if p.suffix == ".pdf"])} PDFs; {args.outdir}', flush=True)


if __name__ == '__main__':
    main()
