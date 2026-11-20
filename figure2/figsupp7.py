#!/usr/bin/env python3
"""Human Testis promoter conservation: final-figure-small violin plots.

Preserves the original evolutionary age groups, Major/Minor labels, covered-base
hg19 phastCons100way means, exact 2,000-bp TSS windows, exclusions, labels and
three-page order (All/Major/Minor). No developmental-age reclassification.

Python-only: 5 x 4 inches per page; x/y ticks 12 pt; axis labels 14 pt;
legend default 12 pt; editable PDF text; PDF compression disabled. Every violin
uses #ee2560. Scott KDE is evaluated on 256 positions, restricted to the observed
data range. Each violin has the same maximum width (width does not encode n).
The narrow inner box retains median, IQR, 1.5-IQR whiskers and all outliers.
This is descriptive; same-gene/overlapping promoters are not independent samples.

Writes ONLY one three-page PDF on the server, never PNG/SVG/TSV sidecars.
Sources and validation are embedded in PDF metadata/notes and printed to stdout.
Requires numpy, matplotlib and pyBigWig in the existing PY310 environment.

Usage:
  python plot_promoter_conservation_violin.py --self-test
  python plot_promoter_conservation_violin.py --check-only
  python plot_promoter_conservation_violin.py
  # Optional one-to-one verification against the archived previous score table:
  python plot_promoter_conservation_violin.py --reference-scores /path/old_scores.tsv
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo/17.conservation_score')
DEFAULT_INPUT = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo/3.get_promoter_age/all_tissues_MP/Testis/Human_Testis_MP_major_minor_promoters_with_age.tsv')
AGES = ('Amniote_conserved','Eutherian_conserved','Primate_restricted','Human_specific')
LABELS = ('Amniote','Eutherian','Primate','Human')
KINDS = ('Major','Minor')
ASSEMBLIES = {
    'hg19':dict(chr1=249250621,chr2=243199373,
                url='https://hgdownload.soe.ucsc.edu/goldenPath/hg19/phastCons100way/hg19.100way.phastCons.bw'),
    'hg38':dict(chr1=248956422,chr2=242193529,
                url='https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw')}
ADDED = ('input_row','tss_1based','tss_0based','bigwig_chrom','requested_start0','requested_end0',
         'window_start0','window_end0','requested_window_bp','window_bp','clipped',
         'covered_bp','coverage_fraction','phastCons_mean','primary_plot_included','score_status')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda:handle.read(4*1024*1024),b''):
            digest.update(data)
    return digest.hexdigest()


def safe_json(value):
    if isinstance(value,dict):
        return {k:safe_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):
        return None
    return value


def write_tsv(path,fields,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t',lineterminator='\n',extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:'NA' if v is None or isinstance(v,(float,np.floating)) and not math.isfinite(v)
                             else v for k,v in row.items() if k in fields})


def integer(value,name):
    try:
        number=Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError('Invalid '+name+': '+str(value)) from error
    require(number.is_finite() and number==number.to_integral_value(),'Non-integer '+name+': '+str(value))
    return int(number)


def load_input(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle,delimiter='\t')
        fields=reader.fieldnames or []
        require(len(fields)==len(set(fields)),'Duplicate input columns')
        needed={'promoterId','geneId','seqnames','start','strand','promoter_class','TSS_age_group'}
        require(needed<=set(fields),'Missing input columns: '+str(needed-set(fields)))
        require(not set(fields)&set(ADDED),'Input already contains conservation output columns')
        rows=list(reader)
    require(rows and all(None not in r and all(v is not None for v in r.values()) for r in rows),'Empty/malformed input')
    seen=set()
    by_gene=defaultdict(list)
    for line,row in enumerate(rows,2):
        match=re.fullmatch(r'(?:promoter_)?(\d+)',row['promoterId'].strip())
        require(match is not None,'Unexpected promoterId on line '+str(line))
        key=int(match[1])
        require(key not in seen,'Duplicate normalized promoter ID: '+str(key))
        seen.add(key)
        require(row['promoter_class'] in KINDS and row['TSS_age_group'] in AGES,'Unknown age/class on line '+str(line))
        if 'TSS_age_label' in fields:
            require(row['TSS_age_label']==LABELS[AGES.index(row['TSS_age_group'])],'Age label disagrees with age group')
        require(row['strand'] in ('+','-') and integer(row['start'],'TSS')>0,'Invalid TSS/strand')
        require(row['geneId'] not in ('','NA','.') and row['seqnames'] not in ('','NA','.'),'Missing gene/chromosome')
        if 'internalPromoter' in fields:
            require(row['internalPromoter'].strip().lower() in ('false','f','0'),'Internal promoter in filtered MP table')
        by_gene[row['geneId']].append(row)
    for gene,items in by_gene.items():
        require(len(items)>=2 and sum(r['promoter_class']=='Major' for r in items)==1,
                'Not an MP gene with one preserved Major label: '+gene)
    return fields,rows


def resolve_chrom(name,chroms):
    if name in chroms:
        return name
    bare=name[3:] if name.startswith('chr') else name
    candidates=['chr'+bare,bare]
    if bare in ('M','MT'):
        candidates+=['chrM','MT','M']
    found=sorted(set(c for c in candidates if c in chroms))
    require(len(found)<=1,'Ambiguous chromosome alias for '+name)
    return found[0] if found else None


def verify_assembly(chroms,assembly):
    for canonical in ('chr1','chr2'):
        name=resolve_chrom(canonical,chroms)
        require(name is not None and chroms[name]==ASSEMBLIES[assembly][canonical],
                f'bigWig does not match declared {assembly}: {canonical} expected {ASSEMBLIES[assembly][canonical]}')


def make_window(tss1,flank,chrom_length):
    require(tss1>0 and flank>0 and chrom_length>0,'Invalid window inputs')
    anchor=tss1-1
    requested=(anchor-flank,anchor+flank)
    if anchor>=chrom_length:
        return requested,None,None
    start=max(0,requested[0])
    end=min(chrom_length,requested[1])
    return requested,(start,end),(start,end)!=requested


def exact_score(bw,chrom,start,end):
    mean=bw.stats(chrom,start,end,type='mean',exact=True)[0]
    coverage=bw.stats(chrom,start,end,type='coverage',exact=True)[0]
    if mean is None:
        require(coverage is None or coverage==0.,'Mean missing despite positive coverage')
        return np.nan,0,0.
    require(math.isfinite(mean) and 0.<=mean<=1.,'Value outside phastCons range [0,1]')
    require(coverage is not None and math.isfinite(coverage) and 0.<coverage<=1.,'Invalid covered fraction')
    covered=integer(round(coverage*(end-start)),'covered bp')
    require(1<=covered<=end-start and np.isclose(covered/(end-start),coverage,rtol=0,atol=1e-8),
            'Exact coverage is inconsistent with an integer covered-base count')
    return float(mean),covered,float(coverage)


def extract(rows,bw,args):
    chroms=bw.chroms()
    verify_assembly(chroms,args.assembly)
    result=[]
    cache={}
    verified=0
    for number,row in enumerate(rows,1):
        tss1=integer(row['start'],'TSS')
        chrom=resolve_chrom(row['seqnames'],chroms)
        anchor=tss1-1
        out=dict(row,input_row=number+1,tss_1based=tss1,tss_0based=anchor,bigwig_chrom=chrom,
                 requested_start0=anchor-args.flank,requested_end0=anchor+args.flank,
                 requested_window_bp=2*args.flank,window_start0=None,window_end0=None,window_bp=None,
                 clipped=None,covered_bp=None,coverage_fraction=np.nan,phastCons_mean=np.nan,
                 primary_plot_included=False,score_status='chromosome_absent_from_bigwig')
        if chrom is not None:
            _,window,clipped=make_window(tss1,args.flank,chroms[chrom])
            if window is None:
                out['score_status']='TSS_outside_chromosome'
            else:
                start,end=window
                key=chrom,start,end
                if key not in cache:
                    cache[key]=exact_score(bw,*key)
                    if verified<args.verify_windows:
                        values=np.asarray(bw.values(*key,numpy=True),dtype=float)
                        require(len(values)==end-start and not np.isinf(values).any(),'Malformed per-base bigWig values')
                        valid=np.isfinite(values)
                        expected=float(values[valid].mean()) if valid.any() else np.nan
                        require(np.isclose(expected,cache[key][0],rtol=1e-7,atol=1e-9,equal_nan=True)
                                and valid.sum()==cache[key][1],'stats(exact=True) disagrees with per-base extraction')
                        verified+=1
                score,covered,coverage=cache[key]
                status=('no_scored_bases' if covered==0 else
                        'clipped_window_excluded' if clipped and not args.allow_clipped_windows else
                        'below_min_coverage' if coverage<args.min_coverage else 'ok')
                out.update(window_start0=start,window_end0=end,window_bp=end-start,clipped=clipped,
                           covered_bp=covered,coverage_fraction=coverage,phastCons_mean=score,
                           primary_plot_included=status=='ok',score_status=status)
        result.append(out)
        if number%1000==0:
            print(f'EXTRACTED {number}/{len(rows)} promoters; unique queried windows={len(cache)}',flush=True)
    require(any(r['primary_plot_included'] for r in result),'No promoter has a usable conservation score')
    return result,dict(unique_windows=len(cache),per_base_verified_windows=verified,
                       status_counts=dict(Counter(r['score_status'] for r in result)))


def summarize(rows):
    output=[]
    for kind in ('All',*KINDS):
        for age,label in zip(AGES,LABELS):
            selected=[r for r in rows if r['TSS_age_group']==age and (kind=='All' or r['promoter_class']==kind)]
            used=[r for r in selected if r['primary_plot_included']]
            values=np.asarray([r['phastCons_mean'] for r in used])
            quantiles=np.quantile(values,[.25,.5,.75]) if len(values) else [np.nan]*3
            finite_coverage=[r['coverage_fraction'] for r in selected if math.isfinite(r['coverage_fraction'])]
            output.append(dict(promoter_class=kind,TSS_age_group=age,TSS_age_label=label,
                               input_promoter_n=len(selected),included_promoter_n=len(used),
                               excluded_promoter_n=len(selected)-len(used),
                               mean_score=float(values.mean()) if len(values) else np.nan,
                               q25=float(quantiles[0]),median=float(quantiles[1]),q75=float(quantiles[2]),
                               median_coverage_all_resolved_windows=float(np.median(finite_coverage)) if finite_coverage else np.nan,
                               zero_coverage_n=sum(r['score_status']=='no_scored_bases' for r in selected),
                               clipped_window_n=sum(r['clipped'] is True for r in selected),
                               chromosome_absent_n=sum(r['score_status']=='chromosome_absent_from_bigwig' for r in selected)))
    return output


def build_figure(rows,kind,args):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif',
                         'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'pdf.fonttype':42,'ps.fonttype':42,'font.size':12,
                         'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,
                         'legend.fontsize':12,'legend.frameon':False,
                         'axes.linewidth':.8,'axes.spines.top':False,'axes.spines.right':False})
    data=[]
    for age in AGES:
        data.append([r['phastCons_mean'] for r in rows if r['primary_plot_included']
                     and r['TSS_age_group']==age and (kind=='All' or r['promoter_class']==kind)])
    fig,ax=plt.subplots(figsize=(5,4))
    fig.subplots_adjust(left=.19,right=.97,bottom=.24,top=.80)
    # Gaussian KDE is undefined for a singleton/constant group: show its actual
    # horizontal value rather than inventing a distribution or adding jitter.
    for position,values in enumerate(data,1):
        if len(values)>1 and np.ptp(values)>0:
            violin=ax.violinplot([values],positions=[position],widths=.78,points=256,
                                 bw_method='scott',showmeans=False,showmedians=False,
                                 showextrema=False)
            for body in violin['bodies']:
                body.set_facecolor('#ee2560')
                body.set_edgecolor('#ee2560')
                body.set_linewidth(.9)
                body.set_alpha(1.)
                vertices=body.get_paths()[0].vertices
                require(vertices[:,1].min()>=min(values)-1e-12 and
                        vertices[:,1].max()<=max(values)+1e-12,
                        'Violin density extends beyond observed scores')
        elif values:
            ax.hlines(values[0],position-.2,position+.2,color='#ee2560',linewidth=2)
    positions=[i+1 for i,v in enumerate(data) if v]
    if positions:
        ax.boxplot([v for v in data if v],positions=positions,widths=.12,patch_artist=True,
                   showfliers=True,whis=1.5,zorder=3,
                   boxprops=dict(facecolor='white',edgecolor='#3A3A3A',linewidth=.8),
                   whiskerprops=dict(color='#555555',linewidth=.8),
                   capprops=dict(color='#555555',linewidth=.8),
                   medianprops=dict(color='#202020',linewidth=1.2),
                   flierprops=dict(marker='.',markersize=2,markerfacecolor='#555555',
                                   markeredgecolor='none',alpha=.20,rasterized=False))
    ax.set(xlim=(.4,4.6),ylim=(-.025,1.10))
    ax.set_xticks(range(1,5),LABELS,rotation=25,ha='right',rotation_mode='anchor')
    ax.set_yticks([0.,.25,.5,.75,1.])
    ax.set_ylabel('Mean phastCons score',fontsize=14)
    ax.tick_params(axis='both',labelsize=12)
    for i,values in enumerate(data,1):
        ax.text(i,1.045,f'n={len(values):,}',ha='center',va='center',fontsize=9.5)
        if not values:
            ax.text(i,.5,'No data',ha='center',va='center',fontsize=10,color='#666666')
    fig.text(.58,.96,f'Human Testis | {kind} promoters',ha='center',va='top',fontsize=13)
    fig.text(.58,.89,f'{args.assembly} phastCons100way | TSS +/- {args.flank/1000:g} kb',
             ha='center',va='top',fontsize=10.5)
    fig.text(.58,.02,'One value per promoter; missing bases excluded',
             ha='center',va='bottom',fontsize=8.5)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    width,height=fig.canvas.get_width_height()
    for artist in [*fig.texts,ax.yaxis.label,*ax.texts,*ax.get_xticklabels(),*ax.get_yticklabels()]:
        box=artist.get_window_extent(renderer)
        require(box.x0>=-1 and box.y0>=-1 and box.x1<=width+1 and box.y1<=height+1,
                'Clipped figure text: '+artist.get_text())
    require(tuple(fig.get_size_inches())==(5.,4.),'Final physical size changed')
    require(all(t.get_fontsize()==12 for t in [*ax.get_xticklabels(),*ax.get_yticklabels()]),
            'Tick fontsize differs from final-figure-small')
    require(ax.yaxis.label.get_fontsize()==14,'Axis label fontsize differs')
    return fig


def compare_reference(scores,path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        prior=list(csv.DictReader(handle,delimiter='\t'))
    require(len(prior)==len(scores),'Promoter count changed from previous result')
    previous={r['promoterId']:r for r in prior}
    require(len(previous)==len(prior),'Duplicate promoter in reference')
    for row in scores:
        require(row['promoterId'] in previous,'Promoter membership changed')
        old=previous[row['promoterId']]
        for key in ('geneId','seqnames','start','strand','promoter_class','TSS_age_group',
                    'score_status'):
            require(str(row[key])==old[key],'Previous-result mismatch: '+row['promoterId']+' '+key)
        require(str(row['primary_plot_included'])==old['primary_plot_included'],
                'Inclusion/exclusion rule changed')
        for key in ('phastCons_mean','coverage_fraction','covered_bp','tss_0based',
                    'window_start0','window_end0','window_bp'):
            actual=row[key]
            saved=old[key]
            missing=actual is None or isinstance(actual,(float,np.floating)) and not np.isfinite(actual)
            if missing:
                require(saved in ('NA','nan','None',''),'Missing score/window changed')
            else:
                require(saved not in ('NA','nan','None','') and
                        np.isclose(float(actual),float(saved),rtol=0,atol=1e-15),
                        'Previous-result value mismatch: '+row['promoterId']+' '+key)
    return dict(status='passed',promoters_compared=len(scores),reference=str(path),
                reference_sha256=sha256(path),tolerance_absolute=1e-15)


def self_test():
    require(integer('1.5e3','coordinate')==1500,'Scientific integer parsing')
    try:
        integer('1.5','coordinate')
    except ValueError:
        pass
    else:
        raise ValueError('Fractional genomic coordinate accepted')
    require(make_window(1001,1000,10000)==((0,2000),(0,2000),False),'TSS/BED conversion')
    require(make_window(1,1000,10000)==((-1000,1000),(0,1000),True),'Left clipping')
    require(make_window(10000,1000,10000)==((8999,10999),(8999,10000),True),'Right clipping')
    require(make_window(10001,1000,10000)[1] is None,'Outside-chromosome TSS')
    require(resolve_chrom('1',{'chr1':100})=='chr1' and resolve_chrom('MT',{'chrM':10})=='chrM','Chromosome alias')
    class FakeBW:
        def stats(self,chrom,start,end,type,exact):
            require(exact is True,'Accidental zoom approximation')
            values=np.array([0.,np.nan,.5,1.])[start:end]
            valid=np.isfinite(values)
            return [(float(values[valid].mean()) if valid.any() else None) if type=='mean'
                    else (float(valid.mean()) if valid.any() else None)]
    score,covered,coverage=exact_score(FakeBW(),'chr1',0,4)
    require(score==.5 and covered==3 and coverage==.75,'Covered-base mean/coverage')
    require(exact_score(FakeBW(),'chr1',0,1)==(0.,1,1.),'Observed zero was removed')
    missing=exact_score(FakeBW(),'chr1',1,2)
    require(np.isnan(missing[0]) and missing[1:]==(0,0.),'Missing treated as zero')
    verify_assembly({'chr1':249250621,'chr2':243199373},'hg19')
    try:
        verify_assembly({'chr1':249250621,'chr2':243199373},'hg38')
    except ValueError:
        pass
    else:
        raise ValueError('hg19 track accepted for hg38')
    print('SELF_TEST_OK: coordinates, clipping, aliases, exact covered-base means, observed zero/NA, assembly guard')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--assembly',choices=tuple(ASSEMBLIES),default='hg19')
    parser.add_argument('--bigwig',type=Path,default=ROOT/'resources/hg19.100way.phastCons.bw')
    parser.add_argument('--output',type=Path,default=ROOT/'results/Human_Testis_phastCons100way/Human_Testis_promoter_phastCons100way_by_age_violinplots.pdf')
    parser.add_argument('--flank',type=int,default=1000)
    parser.add_argument('--min-coverage',type=float,default=0.)
    parser.add_argument('--allow-clipped-windows',action='store_true')
    parser.add_argument('--verify-windows',type=int,default=100)
    parser.add_argument('--reference-scores',type=Path,help='Optional archived prior score TSV; strict one-to-one check')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.assembly=='hg19' or args.input.resolve()!=DEFAULT_INPUT.resolve(),
            'Original input is hg19; hg38 needs an explicitly different converted input')
    require(args.flank>0 and 0.<=args.min_coverage<=1. and args.verify_windows>=0,'Invalid parameters')
    require(args.output.suffix.lower()=='.pdf','Only PDF export is supported')
    require(args.check_only or not args.output.exists(),'Output already exists; choose a NEW --output PDF')
    require(args.bigwig.is_file(),'Local conservation bigWig is missing')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pyBigWig
    table_stamp=(args.input.stat().st_size,args.input.stat().st_mtime_ns)
    track_stamp=(args.bigwig.stat().st_size,args.bigwig.stat().st_mtime_ns)
    input_hash=sha256(args.input)
    if args.input.resolve()==DEFAULT_INPUT.resolve():
        require(input_hash=='dd5ffaf3d5210a6f2ddef25c97e30c8340881279604cd2954cfa20b1767807dc',
                'Original promoter input changed; refusing an unannounced change of content')
    fields,rows=load_input(args.input)
    with pyBigWig.open(str(args.bigwig)) as bw:
        require(bw.isBigWig(),'Input is not a bigWig')
        verify_assembly(bw.chroms(),args.assembly)
        header=bw.header()
        if args.check_only:
            print(json.dumps(dict(status='check_only_passed',input_promoter_n=len(rows),
                                  input_sha256=input_hash,assembly=args.assembly,
                                  bigwig=str(args.bigwig),track_header=header)),flush=True)
            return
        scores,audit=extract(rows,bw,args)
    require(table_stamp==(args.input.stat().st_size,args.input.stat().st_mtime_ns),'Input table changed during extraction')
    require(track_stamp==(args.bigwig.stat().st_size,args.bigwig.stat().st_mtime_ns),'Conservation bigWig changed during extraction')
    reference_check=compare_reference(scores,args.reference_scores) if args.reference_scores else None
    summary=summarize(scores)
    provenance=dict(input=str(args.input),input_sha256=input_hash,bigwig=str(args.bigwig),
                    track_size_bytes=track_stamp[0],track_mtime_ns=track_stamp[1],header=header,
                    script_sha256=sha256(__file__),assembly=args.assembly,
                    parameters=dict(flank=args.flank,min_coverage=args.min_coverage,
                                    allow_clipped_windows=args.allow_clipped_windows),
                    extraction_audit=audit,summary=summary,reference_check=reference_check,
                    display=dict(size_inches=[5,4],color='#ee2560',ticks_pt=12,axis_labels_pt=14,
                                 legend_pt=12,kde='Scott',kde_grid=256,kde_support='observed min/max',
                                 violin_width='equal maximum; not proportional to n'),
                    versions=dict(python=sys.version.split()[0],numpy=np.__version__,
                                  matplotlib=matplotlib.__version__,pyBigWig=pyBigWig.__version__))
    figures=[build_figure(scores,kind,args) for kind in ('All',*KINDS)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    from matplotlib.backends.backend_pdf import PdfPages
    metadata={'Title':'Human Testis promoter conservation by evolutionary age',
              'Subject':'hg19 phastCons100way exact covered-base means; TSS +/- 1 kb; All/Major/Minor violin plots',
              'Keywords':json.dumps(safe_json(provenance),allow_nan=False)}
    # File mode x prevents replacement if a concurrent run created the same PDF.
    with args.output.open('xb') as file, matplotlib.rc_context({'pdf.compression':0}):
        with PdfPages(file,metadata=metadata) as pages:
            for kind,fig in zip(('All',*KINDS),figures):
                pages.attach_note('One window mean per promoter; observed zero retained; missing not zero. '
                                  'Scott KDE (256 positions), support limited to observed min/max. '
                                  'Each violin equal maximum width; n labels give sample sizes. '
                                  'Inner box: median/IQR, 1.5-IQR whiskers and all outliers. '
                                  'No inferential tests; same-gene/overlapping-window dependence. '
                                  'Age alignability and conservation are not independent functional validation. Page: '+kind)
                pages.savefig(fig)
                plt.close(fig)
    print(json.dumps(safe_json(dict(status='complete',pdf=str(args.output),
                                   plotted_promoter_n=sum(r['primary_plot_included'] for r in scores),
                                   source_table_rows=len(scores),reference_check=reference_check,
                                   extraction_audit=audit,summary=summary)),indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    main()

