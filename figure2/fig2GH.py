#!/usr/bin/env python3
"""Annotate human promoter TSSs with UCSC CpG islands and plot proportions.

Default inputs are the existing GRCh37.75 human promoter age table and hg19
cpgIslandExt dump. TSS start is 1-based on BOTH strands; convert to BED
[start-1, start). Overlap requires at least one shared base, without a strand
filter. No promoter window expansion. Preserve every input row and inherited
Major/Minor and mapping-based age labels. Denominator = ALL promoters in each
class x age group, including the existing SP and MP gene populations.

Requires Python >=3.9 and matplotlib. No bedtools dependency for calculation;
the emitted BED files support independent bedtools intersect -c verification.
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

BASE = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo')
AGES = ('Amniote_conserved', 'Eutherian_conserved', 'Primate_restricted', 'Human_specific')
LABELS = ('Amniote', 'Eutherian', 'Primate', 'Human')
KINDS = ('Major', 'Minor')
COLORS = {'CGI-overlap': '#FBB463', 'No CGI overlap': '#BDBADB'}
EXTRA = ['TSS_bed_start', 'TSS_bed_end', 'CGI_overlap', 'CGI_overlap_count', 'CGI_IDs']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha.hexdigest()}


def read_promoters(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fields, rows = reader.fieldnames or [], list(reader)
    required = {'promoterId', 'geneId', 'seqnames', 'start', 'strand',
                'promoter_class', 'TSS_age_group', 'TSS_age_label'}
    require(rows and required <= set(fields), 'Empty promoter table or missing required fields')
    require(len(fields) == len(set(fields)), 'Duplicate column names')
    require(not set(EXTRA) & set(fields), 'Input already contains CGI output columns')
    seen = set()
    for n, r in enumerate(rows, 2):
        require(None not in r and all(v is not None for v in r.values()), f'Malformed row {n}')
        key = int(r['promoterId'])
        require(key not in seen, f'Duplicate promoterId: {key}')
        seen.add(key)
        require(int(r['start']) >= 1 and r['strand'] in ('+', '-'), f'Invalid TSS/strand at row {n}')
        require(r['seqnames'] and r['geneId'], f'Missing chromosome/gene at row {n}')
        require(r['promoter_class'] in KINDS, f'Unknown class at row {n}')
        require(r['TSS_age_group'] in AGES, f'Unknown age at row {n}')
        require(r['TSS_age_label'] == LABELS[AGES.index(r['TSS_age_group'])], f'Age/label mismatch at row {n}')
    return fields, rows


def read_cgi(path):
    opener = gzip.open if path.suffix == '.gz' else open
    by_chrom, seen = defaultdict(list), set()
    with opener(path, 'rt', encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or line.startswith('#'):
                continue
            r = line.rstrip('\r\n').split('\t')
            require(len(r) == 11, f'Expected UCSC 11-column cpgIslandExt dump at line {n}')
            chrom, start, end = r[1], int(r[2]), int(r[3])
            require(0 <= start < end and end-start == int(r[5]), f'Invalid CGI coordinates at line {n}')
            key = (chrom, start, end)
            require(key not in seen, f'Duplicate CGI interval at line {n}')
            seen.add(key)
            by_chrom[chrom].append((start, end, f'{chrom}:{start}-{end}'))
    require(seen, 'No CGI intervals found')
    index = {}
    for chrom, intervals in by_chrom.items():
        intervals.sort()
        prefix_max_end, greatest = [], 0
        for start, end, name in intervals:
            greatest = max(greatest, end)
            prefix_max_end.append(greatest)
        index[chrom] = (intervals, [r[0] for r in intervals], prefix_max_end)
    return index, len(seen)


def annotate(rows, index):
    unknown = sorted({r['seqnames'] for r in rows} - set(index))
    require(not unknown, f'Promoter chromosomes missing from CGI input: {unknown}')
    annotated = []
    for r in rows:
        start, end = int(r['start'])-1, int(r['start'])
        intervals, starts, max_end = index[r['seqnames']]
        j = bisect_left(starts, end)-1
        hits = []
        while j >= 0 and max_end[j] > start:
            a, b, name = intervals[j]
            if a < end and b > start:
                hits.append(name)
            j -= 1
        annotated.append(dict(r, TSS_bed_start=start, TSS_bed_end=end,
                              CGI_overlap=int(bool(hits)), CGI_overlap_count=len(hits),
                              CGI_IDs=';'.join(reversed(hits))))
    return annotated


def summarize(rows):
    total, overlap = Counter(), Counter()
    for r in rows:
        key = (r['promoter_class'], r['TSS_age_group'])
        total[key] += 1
        overlap[key] += r['CGI_overlap']
    result = []
    for kind in KINDS:
        for age, label in zip(AGES, LABELS):
            n, c = total[kind, age], overlap[kind, age]
            require(n > 0 and 0 <= c <= n, f'Empty/invalid group: {kind} {label}')
            p, q = 100*c/n, 100*(n-c)/n
            require(math.isclose(p+q, 100, abs_tol=1e-10), 'Percentages do not sum to 100')
            result.append(dict(promoter_class=kind, TSS_age_group=age, TSS_age_label=label,
                               total_n=n, CGI_overlap_n=c, no_CGI_overlap_n=n-c,
                               CGI_overlap_percent=p, no_CGI_overlap_percent=q))
    require(sum(r['total_n'] for r in result) == len(rows), 'Denominator reconciliation failed')
    return result


def write_tsv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def make_figure(summary, kind):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': 1,
                         'legend.frameon': False, 'xtick.labelsize': 13,
                         'ytick.labelsize': 13, 'axes.labelsize': 16})
    selected = [r for r in summary if r['promoter_class'] == kind]
    fig, ax = plt.subplots(figsize=(6.5, 4.9))
    fig.subplots_adjust(left=.14, right=.985, bottom=.19, top=.745)
    x = list(range(4))
    p = [r['CGI_overlap_percent'] for r in selected]
    q = [r['no_CGI_overlap_percent'] for r in selected]
    ax.bar(x, p, width=.68, color=COLORS['CGI-overlap'], edgecolor='white', linewidth=.8, label='CGI-overlap')
    ax.bar(x, q, bottom=p, width=.68, color=COLORS['No CGI overlap'], edgecolor='white', linewidth=.8, label='No CGI overlap')
    ax.set(xlim=(-.6, 3.6), ylim=(0, 100), ylabel='Proportion (%)')
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x, LABELS)
    ax.tick_params(axis='x', length=0, pad=8)
    for j, r in enumerate(selected):
        ax.annotate(f"n={r['total_n']:,}", (j, 100), xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, annotation_clip=False)
        for bottom, percent in ((0, p[j]), (p[j], q[j])):
            if percent >= 7:
                ax.text(j, bottom+percent/2, f'{percent:.1f}%', ha='center', va='center', fontsize=12, color='#222222')
    title = fig.text(.56, .98, f'{kind} promoters', ha='center', va='top', fontsize=20)
    legend = fig.legend(*ax.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.56, .925),
                        ncol=2, fontsize=13, columnspacing=1.3, handlelength=1.3, handletextpad=.5)
    fig.text(.14, .065, 'CGI-overlap: TSS falls within a UCSC hg19 CpG island.', fontsize=8)
    fig.text(.14, .027, "'Human': no mapping detected in five target species.", fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            b = text.get_window_extent(renderer)
            require(b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1,
                    f'Text clipped: {text.get_text()}')
    box = legend.get_window_extent(renderer)
    require(not box.overlaps(title.get_window_extent(renderer)), 'Title/legend collision')
    require(all(not box.overlaps(t.get_window_extent(renderer)) for t in ax.texts), 'Legend/bar-label collision')
    ticks = ax.get_xticklabels()
    require(all(not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer))
                for a,b in zip(ticks, ticks[1:])), 'Age-label collision')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--age-table', type=Path, default=BASE/'3.get_promoter_age/Human_major_minor_promoters_with_age.tsv')
    parser.add_argument('--cgi-table', type=Path, default=BASE/'7.CGI/hg19_cpgIslandExt.txt.gz')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'results_cgi_by_age_tss')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    require(not args.output_dir.exists() or args.overwrite, 'Output directory exists; use --overwrite')
    fields, rows = read_promoters(args.age_table)
    index, cgi_n = read_cgi(args.cgi_table)
    annotated = annotate(rows, index)
    summary = summarize(annotated)
    figures = {kind: make_figure(summary, kind) for kind in KINDS}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir/'Human_CGI_by_age_summary.tsv', list(summary[0]), summary)
    write_tsv(args.output_dir/'Human_promoters_CGI_status.tsv', fields+EXTRA, annotated)
    with (args.output_dir/'Human_promoter_TSS.bed').open('w') as f:
        for r in annotated:
            f.write(f"{r['seqnames']}\t{r['TSS_bed_start']}\t{r['TSS_bed_end']}\t{r['promoterId']}\t0\t{r['strand']}\n")
    with (args.output_dir/'hg19_CGI.bed').open('w') as f:
        for chrom in sorted(index):
            for start, end, name in index[chrom][0]:
                f.write(f'{chrom}\t{start}\t{end}\t{name}\n')
    import matplotlib
    import matplotlib.pyplot as plt
    for kind, fig in figures.items():
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(args.output_dir/f'Human_{kind.lower()}_CGI_by_age.{ext}', dpi=300, facecolor='white')
        plt.close(fig)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    script=fingerprint(Path(__file__)), age_table=fingerprint(args.age_table),
                    cgi_table=fingerprint(args.cgi_table), CGI_count=cgi_n, promoter_count=len(rows),
                    genome_build='GRCh37 / hg19',
                    genome_evidence='Existing promoter annotation pipeline uses GRCh37.75; CGI dump is hg19',
                    CGI_source='https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/cpgIslandExt.txt.gz',
                    schema='bin,chrom,chromStart,chromEnd,name,length,cpgNum,gcNum,perCpg,perGc,obsExp',
                    overlap_definition='1-based TSS start -> BED [start-1,start); any shared base with CGI; no strand filter',
                    denominator='All original promoters in each Major/Minor x mapping-based age category',
                    age_definition='Inherited mapping-based groups; Human means no mapping detected in five species',
                    class_definition='Inherited GLOBAL step-3 Major/Minor, including the existing SP and MP genes',
                    preserved_input_rows=True, unmatched_chromosomes=[],
                    CGI_ID_definition='chrom:start-end, 0-based half-open; CpG: N names are not unique IDs',
                    plot=dict(size_inches=[6.5,4.9], colors=COLORS, dpi=300, matplotlib=matplotlib.__version__,
                              text_bounds_and_collision_checks='Passed', error_bars='None; descriptive proportions'),
                    summary=summary)
    (args.output_dir/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    for r in summary:
        print(f"{r['promoter_class']:5} {r['TSS_age_label']:9} n={r['total_n']:5,} CGI={r['CGI_overlap_n']:5,} ({r['CGI_overlap_percent']:.4f}%)")
    print(f'CGIs={cgi_n:,}; promoter rows preserved={len(rows):,}; output={args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
