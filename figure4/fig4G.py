#!/usr/bin/env python3
"""Compare each mouse liver TF ChIP-Atlas BED with A, B and background BEDs.

Python >=3.9, matplotlib and scipy. Inputs are mm10, BED 0-based half-open.
Each unique query region is counted once if it overlaps >=1 peak by >=1 bp,
ignoring strand. Exact duplicate coordinates are removed, but regions are
NOT merged or extended and no new peak-score filtering is applied. Different
experiments/replicates of one TF are kept separate; bigWig is not used here.

Default outputs: one fixed 5x4-inch PDF per experiment, a summary TSV,
region-level overlap TSV, Fisher-test TSV and reproducibility JSON. Existing
output directories are never reused/overwritten. --check-only computes and
checks all statistics/layouts without writing results. --self-test checks
the interval implementation against a brute-force implementation.

As in the supplied single-experiment workflow, nominal two-sided Fisher
tests compare A vs FULL background, B vs FULL background, and A vs B. The
tested group is NOT removed from background. Shared/overlapping genomic
regions and paired A/B genes violate independence; these nominal P values
are exploratory and are NOT calibrated enrichment or causal evidence.
No error bars or biological replicate counts are inferred from regions.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

BASE = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/5.chipseq_overlap')
DEFAULT_PEAK_DIR = BASE / 'chip_atlas'
DEFAULT_QUERIES = BASE / 'groupAB_bed'
DEFAULT_SIZES = BASE / 'bed_overlap/E2F4/1.assembly_evidence/mm10.chrom.sizes'
DEFAULT_OUTPUT = DEFAULT_PEAK_DIR / 'overlap_results'
GROUP_FILES = [('Background', 'background.merged.bed'),
               ('Group A', 'A.merged.bed'), ('Group B', 'B.merged.bed')]
NAME_RE = re.compile(r'^((?:SRX|DRX|ERX)\d+)_([A-Z0-9]+)_Mouse_Liver\.bed(?:\.gz)?$')
LIMITATION = ('Nominal unadjusted two-sided Fisher tests against FULL background; '
              'shared regions/paired genes can violate independence. Exploratory '
              'association only, not calibrated enrichment or causal evidence.')


@dataclass(frozen=True, order=True)
class Interval:
    chrom: str
    start: int
    end: int


def normalize_chrom(value):
    match = re.fullmatch(r'(?:chr)?([1-9]|1[0-9]|X|Y|M|MT)', value)
    return 'chr' + ('M' if match[1] == 'MT' else match[1]) if match else value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_bed(path):
    """BED3..11, including narrowPeak names in column 4; not BED12 blocks."""
    path = Path(path)
    records, seen, headers = [], set(), []
    raw_n, aliases, columns = 0, Counter(), Counter()
    opener = gzip.open if path.suffix.lower() == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig') as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(('#', 'track ', 'browser ')) or line in ('track', 'browser'):
                headers.append(line)
                if re.search(r'\b(mm8|mm9|mm39|GRCm39)\b', line, re.I):
                    raise ValueError(f'{path}:{line_no}: header conflicts with mm10')
                continue
            fields = line.split()
            if len(fields) < 3 or len(fields) >= 12:
                raise ValueError(f'{path}:{line_no}: expected BED3..11, not BED12 blocks')
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError:
                raise ValueError(f'{path}:{line_no}: coordinates must be integers') from None
            if start < 0 or end <= start:
                raise ValueError(f'{path}:{line_no}: invalid half-open interval')
            chrom = normalize_chrom(fields[0])
            if chrom != fields[0]:
                aliases[f'{fields[0]} -> {chrom}'] += 1
            record = Interval(chrom, start, end)
            columns[len(fields)] += 1
            raw_n += 1
            if record not in seen:
                records.append(record)
                seen.add(record)
    if not records:
        raise ValueError(f'No valid BED records: {path}')
    return records, dict(path=str(path.resolve()), sha256=sha256_file(path),
        raw_records=raw_n, unique_regions=len(records),
        duplicate_coordinates_removed=raw_n-len(records), chromosome_aliases=dict(aliases),
        field_counts=dict(columns), headers=headers)


def read_sizes(path):
    sizes = {}
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip() and not line.startswith('#'):
                chrom, length = line.split()[:2]
                chrom, length = normalize_chrom(chrom), int(length)
                if length <= 0 or (chrom in sizes and sizes[chrom] != length):
                    raise ValueError(f'Conflicting/invalid chromosome size: {chrom}')
                sizes[chrom] = length
    # Anchors guard against accidentally supplying mm9/mm39 sizes.
    if sizes.get('chr1') != 195471971 or sizes.get('chrX') != 171031299:
        raise ValueError('Chromosome sizes do not match mm10/GRCm38 chr1 and chrX')
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
    """Same rule as bedtools intersect -c: unstranded, >=1 bp overlap."""
    counts = []
    for record in records:
        starts, ends = index.get(record.chrom, ([], []))
        counts.append(bisect_left(starts, record.end)-bisect_right(ends, record.start))
    return counts


def merge_for_audit(records):
    """Only audit geometric relationships; NEVER replace analysis denominators."""
    merged = []
    for record in sorted(records):
        if merged and record.chrom == merged[-1].chrom and record.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = Interval(last.chrom, last.start, max(last.end, record.end))
        else:
            merged.append(record)
    return merged


def audit_query_relationships(groups):
    background = set(groups['Background'])
    bg_merged = merge_for_audit(groups['Background'])
    return dict(
        exact_coordinate_subset_of_background={label:set(records).issubset(background)
            for label, records in groups.items() if label!='Background'},
        background_equals_merged_union_A_B=(bg_merged == merge_for_audit(groups['Group A']+groups['Group B'])),
        A_regions_touching_B=sum(n>0 for n in overlap_counts(groups['Group A'],peak_index(groups['Group B']))),
        B_regions_touching_A=sum(n>0 for n in overlap_counts(groups['Group B'],peak_index(groups['Group A']))),
        merged_regions_if_remerged_for_audit={label:len(merge_for_audit(records)) for label,records in groups.items()},
        denominator_note='Unique coordinates in each supplied file; audit merging is not used for statistics.')


def fisher_pairwise(rows):
    from scipy.stats import fisher_exact
    by_label = {row['group']:row for row in rows}
    tests = []
    for label, comparator in [('Group A','Background'), ('Group B','Background'), ('Group A','Group B')]:
        a, b = by_label[label], by_label[comparator]
        table = [[a['overlap_n'],a['nonoverlap_n']], [b['overlap_n'],b['nonoverlap_n']]]
        odds, p = fisher_exact(table, alternative='two-sided')
        tests.append(dict(group=label, comparator=comparator, contingency_table=table,
            odds_ratio=float(odds) if math.isfinite(odds) else str(odds), p_value=float(p),
            test='two-sided Fisher exact test', multiple_testing_adjustment='None',
            inference=LIMITATION))
    return tests


def discover_peaks(peak_dir, selected):
    metadata = {}
    records_path = peak_dir/'chipatlas_source_records.json'
    if records_path.exists():
        for row in json.loads(records_path.read_text(encoding='utf-8')):
            metadata[row['experiment_id']] = row
    downloads = {}
    manifest_path = peak_dir/'download_manifest.tsv'
    if manifest_path.exists():
        with manifest_path.open(encoding='utf-8-sig',newline='') as handle:
            downloads = {row['filename']:row for row in csv.DictReader(handle,delimiter='\t')}
    candidates = sorted(list(peak_dir.glob('*.bed'))+list(peak_dir.glob('*.bed.gz')))
    found = []
    for path in candidates:
        match = NAME_RE.fullmatch(path.name)
        if not match:
            continue
        expid, tf = match.groups()
        if selected and expid not in selected:
            continue
        source = metadata.get(expid,{})
        if source and (source['assembly']!='mm10' or source['tf']!=tf):
            raise ValueError(f'Source metadata conflicts with filename: {path.name}')
        download = downloads.get(path.name,{})
        if download and (download.get('status')!='complete' or download.get('assembly')!='mm10'):
            raise ValueError(f'Download manifest is incomplete or not mm10: {path.name}')
        found.append(dict(path=path,experiment_id=expid,tf=tf,source=source,download=download))
    ids = [item['experiment_id'] for item in found]
    if len(ids)!=len(set(ids)):
        raise ValueError('Multiple BEDs for one experiment; select a directory with one peak BED per experiment')
    if selected and set(selected)-set(ids):
        raise ValueError(f'Experiments not found: {sorted(set(selected)-set(ids))}')
    if not found:
        raise ValueError(f'No EXPERIMENT_TF_Mouse_Liver.bed[.gz] files in {peak_dir}')
    return found


def analyse(args):
    sizes = read_sizes(args.chrom_sizes)
    groups, audits = {}, {}
    for label, filename in GROUP_FILES:
        groups[label], audits[label] = read_bed(args.query_dir/filename)
        validate_bounds(groups[label],sizes,label)
    found = discover_peaks(args.peak_dir,set(args.experiment_id or []))
    experiments, details = [], []
    for item in found:
        peaks, audit = read_bed(item['path'])
        if item['download'].get('sha256') and audit['sha256']!=item['download']['sha256']:
            raise ValueError(f'Peak file changed since download validation: {item["path"]}')
        validate_bounds(peaks,sizes,item['path'].name)
        index, rows = peak_index(peaks), []
        for label, records in groups.items():
            if not ({r.chrom for r in records} & set(index)):
                raise ValueError(f'{item["path"].name}: no chromosomes shared with {label}')
            counts = overlap_counts(records,index)
            yes = sum(n>0 for n in counts)
            rows.append(dict(group=label,total_n=len(records),overlap_n=yes,
                nonoverlap_n=len(records)-yes,overlap_percentage=100*yes/len(records),
                chromosomes_without_peaks=sorted({r.chrom for r in records}-set(index))))
            for region,count in zip(records,counts):
                details.append(dict(experiment_id=item['experiment_id'],tf=item['tf'],group=label,
                    chrom=region.chrom,start=region.start,end=region.end,
                    overlapping_peak_n=count,overlaps_any_peak=int(count>0)))
        source = item['source']
        experiments.append(dict(experiment_id=item['experiment_id'],tf=item['tf'],
            peak_file=str(item['path'].resolve()),peak_audit=audit,
            tf_source_group=source.get('group',''),condition_note=source.get('condition_note',''),
            actual_sample_type=source.get('tissue',''),
            experiment_page=source.get('experiment_page',f'https://chip-atlas.org/view?id={item["experiment_id"]}'),
            summary=rows,fisher_tests=fisher_pairwise(rows)))
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),assembly='mm10/GRCm38',
        script_sha256=sha256_file(Path(__file__)),query_audits=audits,
        chrom_sizes=dict(path=str(args.chrom_sizes.resolve()),sha256=sha256_file(args.chrom_sizes)),
        query_relationships=audit_query_relationships(groups),
        unit='Unique genomic region (chromosome,start,end), ignoring strand; not promoter ID or gene count',
        overlap_rule='At least 1 bp; each query region counted once even with multiple peak hits',
        preprocessing='Exact coordinate deduplication and standard mouse chromosome aliases only; no window expansion/merging/filtering',
        statistical_limitation=LIMITATION,
        assembly_note='Assembly follows source provenance and supplied mm10 promoter inputs; chromosome bounds alone do not establish assembly.',
        experiments=experiments)
    return report, details


def make_plot(experiment):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'pdf.fonttype':42,'font.size':12,'axes.labelsize':14,'axes.titlesize':14,
        'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,
        'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    rows, tests = experiment['summary'], experiment['fisher_tests']
    fig, ax = plt.subplots(figsize=(5,4))
    values = [r['overlap_percentage'] for r in rows]
    ax.bar(range(3),values,color=['#F1B656','#397FC7','#6F6F6F'],width=.75)
    maximum = max(5,*values)
    for i,row in enumerate(rows):
        ax.text(i,row['overlap_percentage']+maximum*.025,
            f"{row['overlap_n']}/{row['total_n']}\n{row['overlap_percentage']:.1f}%",
            ha='center',va='bottom',fontsize=11,linespacing=1.15)
    ax.set_xticks(range(3),[r['group'] for r in rows])
    ax.set_ylabel(f"Regions overlapping with\n{experiment['tf']} ChIP-seq peaks (%)",fontsize=14)
    ax.set_xlabel('')
    ax.set_title(f"{experiment['tf']} | {experiment['experiment_id']}",fontsize=14,pad=10)
    positions = {row['group']:i for i,row in enumerate(rows)}
    for test,height in zip(tests,(1.30,1.52,1.74)):
        left,right = sorted((positions[test['group']],positions[test['comparator']]))
        y = maximum*height
        ax.plot([left,right],[y,y],color='#303030',linewidth=.9)
        ax.text((left+right)/2,y+maximum*.025,f"P = {test['p_value']:.4g}",
            ha='center',va='bottom',fontsize=11,fontstyle='italic')
    upper = maximum*1.98
    ax.set_ylim(0,upper)
    # MaxNLocator can return a tick above ylim. Exclude that invisible tick
    # before checking all text bounding boxes against the fixed PDF page.
    ticks = MaxNLocator(nbins=5).tick_values(0,upper)
    ax.set_yticks([tick for tick in ticks if 0 <= tick <= upper])
    ax.tick_params(direction='out',labelsize=12,width=.8)
    fig.tight_layout(pad=1.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in [ax.title,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels(),*ax.texts]:
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
                raise ValueError(f'Figure text outside fixed 5x4-inch page: {artist.get_text()}')
    return fig


def write_tsv(path,rows,columns):
    with path.open('x',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=columns,delimiter='\t',extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(report,details,output_dir):
    # Strictly new directory: no unrelated result or data can be overwritten.
    output_dir.mkdir(parents=True,exist_ok=False)
    pdf_dir = output_dir/'pdf'
    pdf_dir.mkdir()
    summary,tests = [],[]
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    for exp in report['experiments']:
        common = {k:exp[k] for k in ['experiment_id','tf','peak_file','tf_source_group',
            'condition_note','actual_sample_type','experiment_page']}
        common['unique_peak_n'] = exp['peak_audit']['unique_regions']
        summary.extend(dict(common,**row) for row in exp['summary'])
        for test in exp['fisher_tests']:
            row = dict(common,**test)
            row['contingency_table'] = json.dumps(row['contingency_table'])
            tests.append(row)
        fig = make_plot(exp)
        destination = pdf_dir/f"{exp['experiment_id']}_{exp['tf']}_Mouse_Liver_overlap_percentage.pdf"
        with destination.open('xb') as handle,mpl.rc_context({'pdf.compression':0}):
            fig.savefig(handle,format='pdf',metadata={'Title':f"{exp['tf']} | {exp['experiment_id']}: region overlap",
                'Subject':LIMITATION})
        plt.close(fig)
        exp['output_pdf'] = str(destination.resolve())
    write_tsv(output_dir/'overlap_summary.tsv',summary,
        ['experiment_id','tf','tf_source_group','group','total_n','overlap_n','nonoverlap_n',
         'overlap_percentage','unique_peak_n','condition_note','actual_sample_type','peak_file','experiment_page'])
    write_tsv(output_dir/'fisher_tests.tsv',tests,
        ['experiment_id','tf','group','comparator','contingency_table','odds_ratio','p_value',
         'test','multiple_testing_adjustment','inference'])
    write_tsv(output_dir/'region_overlap_details.tsv',details,
        ['experiment_id','tf','group','chrom','start','end','overlapping_peak_n','overlaps_any_peak'])
    report.update(output_dir=str(output_dir.resolve()),figure_inches=[5,4],
        matplotlib_version=mpl.__version__,python_version=sys.version)
    with (output_dir/'analysis_audit.json').open('x',encoding='utf-8') as handle:
        json.dump(report,handle,indent=2,ensure_ascii=False)
        handle.write('\n')


def compare_with_reference(report,args):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('_provided_overlap_reference',args.reference_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for exp in report['experiments']:
        other = module.analyse(Path(exp['peak_file']),args.query_dir,args.chrom_sizes)
        for new,old in zip(exp['summary'],other['summary']):
            for key in ['group','total_n','overlap_n','nonoverlap_n','overlap_percentage']:
                if new[key]!=old[key]:
                    raise AssertionError(f'{exp["experiment_id"]}: reference mismatch in {key}')
        for new,old in zip(exp['fisher_tests'],other['fisher_tests']):
            if not math.isclose(new['p_value'],old['p_value'],rel_tol=1e-12,abs_tol=1e-15):
                raise AssertionError('Fisher P-value differs from reference')
    report['reference_verification'] = dict(experiments_checked=len(report['experiments']),
        reference_sha256=sha256_file(args.reference_script),result='All counts, percentages and nominal Fisher P values match')


def self_test():
    import random
    import tempfile
    rng = random.Random(42)
    for _ in range(300):
        peaks = [Interval(rng.choice(['chr1','chr2']),s,s+rng.randint(1,25))
                 for s in [rng.randint(0,100) for _ in range(30)]]
        queries = [Interval(rng.choice(['chr1','chr2','chr3']),s,s+rng.randint(1,25))
                   for s in [rng.randint(0,100) for _ in range(20)]]
        brute = [sum(p.chrom==q.chrom and p.start<q.end and p.end>q.start for p in peaks) for q in queries]
        assert overlap_counts(queries,peak_index(peaks)) == brute
    assert overlap_counts([Interval('chr1',10,20),Interval('chr1',9,11)],
                          peak_index([Interval('chr1',0,10)])) == [0,1]
    with tempfile.TemporaryDirectory(prefix='chipatlas_overlap_selftest_') as temp:
        path = Path(temp)/'narrowpeak.bed'
        path.write_text('chr1\t0\t10\tpeak_1\t300\t.\t5\t20\t10\t4\n'
                        '1\t0\t10\tpeak_duplicate\t400\t.\t5\t20\t10\t4\n',encoding='utf-8')
        regions,audit = read_bed(path)
        assert len(regions)==1 and audit['duplicate_coordinates_removed']==1
    rows = [dict(group=label,total_n=n,overlap_n=h,nonoverlap_n=n-h)
            for label,n,h in [('Background',100,40),('Group A',20,12),('Group B',20,3)]]
    tests = fisher_pairwise(rows)
    assert len(tests)==3 and tests[0]['contingency_table']==[[12,8],[40,60]]
    print('PASS: 300 random interval tests, touching-boundary test, narrowPeak/dedup test, full-background Fisher table')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--peak-dir',type=Path,default=DEFAULT_PEAK_DIR)
    parser.add_argument('--query-dir',type=Path,default=DEFAULT_QUERIES)
    parser.add_argument('--chrom-sizes',type=Path,default=DEFAULT_SIZES)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--experiment-id',action='append',help='Optional accession filter; repeat to select several experiments')
    parser.add_argument('--reference-script',type=Path,help='Optional read-only cross-check against the supplied reference implementation')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true',help='Compute all overlaps and validate PDF layouts without creating results')
    modes.add_argument('--self-test',action='store_true')
    modes.add_argument('--preview-stdout',metavar='EXPERIMENT_ID',help='Return one in-memory PNG as JSON for QA; no image files')
    args=parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.check_only and not args.preview_stdout and args.output_dir.exists():
        parser.error('Output directory exists; choose a new --output-dir to preserve prior results')
    report,details=analyse(args)
    if args.reference_script:
        compare_with_reference(report,args)
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    if args.preview_stdout:
        import base64
        import io
        selected=[exp for exp in report['experiments'] if exp['experiment_id']==args.preview_stdout]
        if len(selected)!=1:
            parser.error('Preview experiment not found in selected inputs')
        fig=make_plot(selected[0])
        stream=io.BytesIO()
        fig.savefig(stream,format='png',dpi=140)
        plt.close(fig)
        print(json.dumps({'png':base64.b64encode(stream.getvalue()).decode('ascii')}))
        return 0
    if args.check_only:
        for exp in report['experiments']:
            fig=make_plot(exp)
            plt.close(fig)
        report['layout_validation']='All figures drawn in memory; all labels inside fixed 5x4-inch pages; no result files written'
    else:
        write_outputs(report,details,args.output_dir)
    print('experiment_id\tTF\tquery_group\ttotal_regions\toverlap_regions\toverlap_percentage')
    for exp in report['experiments']:
        for row in exp['summary']:
            print(f"{exp['experiment_id']}\t{exp['tf']}\t{row['group']}\t{row['total_n']}\t{row['overlap_n']}\t{row['overlap_percentage']:.6f}")
    print(json.dumps({key:report[key] for key in ['query_relationships','statistical_limitation']},ensure_ascii=False))
    if args.reference_script:
        print(json.dumps(report['reference_verification'],ensure_ascii=False))
    print(report.get('layout_validation',f"Results: {args.output_dir}"))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,ImportError,AssertionError) as exc:
        raise SystemExit(f'ERROR: {exc}')
