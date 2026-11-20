#!/usr/bin/env python3
"""Human Testis locus: promoter activity, transcript structure and alignment.

Python >=3.9, numpy and matplotlib. Descriptive three-track locus example:
existing tissue-mean ProActiv activity; promoter-associated Ensembl transcript
models; five selected pairwise-chain alignments. No sequencing-coverage or
repeat-annotation input, dependency, track, or output is needed.

Promoter/age classes are inherited, not inferred from this plot. Chain blocks
are alignable intervals, NOT sequence identity, MULTIZ, or proof of presence/
absence. One vector PDF with editable text; original coordinates and activity
values retained. GTF/TSS input is 1-based; internal/chain is 0-based half-open.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

BASE=Path('/mnt/ruiyanhou/nfs_share2')
ROOT=BASE/'RNA_seq_organ_species'
HERE=Path(__file__).resolve().parent
DEFAULT_GENE='ENSG00000112167'
DEFAULT_PROMOTERS=BASE/'proactiv_redo/2.define_major_minor_promoter/try_human_liver/results_by_organ/Testis/Human_Testis_major_minor_promoters.tsv'
DEFAULT_AGE=ROOT/'proactiv_result/1.get_TSS_age/1.1get_all_promoter/allpromoter_basic_info_with_TSS_age.tsv'
DEFAULT_GTF=ROOT/'ref_annotation/Homo_sapiens.GRCh37.75.gtf'
DEFAULT_CHAINS=ROOT/'revise/1.conservation_gene_promoter/chains'
CHAIN_FILES=[('Macaque','rheMac2','hg19ToRheMac2.over.chain'),
             ('Mouse','mm10','hg19ToMm10.over.chain'),
             ('Rat','rn5','hg19ToRn5.over.chain'),
             ('Rabbit','oryCun2','hg19ToOryCun2.over.chain'),
             ('Chicken','galGal4','hg19ToGalGal4.over.chain')]
OLD={'Amniote_conserved','Eutherian_conserved'}
YOUNG={'Primate_restricted','Human_specific'}
AGE_NAMES=dict(Amniote_conserved='Amniote',Eutherian_conserved='Eutherian',
               Primate_restricted='Primate',Human_specific='Human')
TRACKS=['ProActiv mean activity','Ensembl transcripts','Pairwise chain-aligned blocks']
METHOD=dict(activity='Existing Testis raw absolute promoter means, equal weight per condition; labels unchanged',
            transcript='Only original Ensembl transcripts assigned to these promoters; exon and CDS coordinates unchanged',
            alignment='Five pairwise chains; highest-scoring qualifying chain per species; not MULTIZ or sequence identity',
            uncovered='No block in selected chain; not proof of biological absence',
            age='Existing dual-window liftOver classes, not re-inferred from displayed chain',
            coordinates='Internal 0-based half-open; genomic ticks and promoter start column are 1-based')

def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def snapshot(path, full_hash=False):
    path = Path(path)
    state = path.stat()
    record = dict(path=str(path.resolve()), bytes=state.st_size, mtime_ns=state.st_mtime_ns)
    if full_hash:
        record['sha256'] = sha256(path)
    return record


def unchanged(record):
    return snapshot(record['path']) == {k: v for k, v in record.items() if k != 'sha256'}


def text_open(path):
    return gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8-sig')


def promoter_id(value):
    value = str(value).removeprefix('promoter_')
    require(value.isdigit(), f'Invalid promoter ID: {value}')
    return str(int(value))


def load_promoters(table, ages, gene):
    with text_open(table) as handle:
        selected = [r for r in csv.DictReader(handle, delimiter='\t') if r['geneId'] == gene]
    require(len(selected) >= 2, f'{gene}: fewer than two retained Testis promoters')
    age_index = {}
    with text_open(ages) as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            key = promoter_id(row['promoterId'])
            require(key not in age_index, f'Duplicate age annotation ID {key}')
            age_index[key] = row
    seen = set()
    for row in selected:
        key = promoter_id(row['promoterId'])
        require(key not in seen and key in age_index, f'Duplicate/missing promoter {key}')
        seen.add(key)
        age = age_index[key]
        for column in ['geneId', 'seqnames', 'start', 'strand', 'promoterPosition', 'txId']:
            require(row[column] == age[column], f'Annotation conflict: promoter {key}, {column}')
        require(row['internalPromoter'].lower() == age['internalPromoter'].lower() == 'false', 'Internal promoter survived upstream selection')
        row['TSS_age_group'] = age['TSS_age_group']
        require(row['TSS_age_group'] in OLD | YOUNG, 'Unknown mapping-based age')
        row['activity'] = float(row['mean_promoter_activity'])
        require(math.isfinite(row['activity']) and row['activity'] > 0, 'Invalid mean activity')
        row['tss0'] = int(row['start'])-1
        require(row['tss0'] >= 0, 'Invalid TSS coordinate')
        row['transcript_ids'] = [t.strip() for t in row['txId'].split(',')]
    require(sum(r['promoter_class'] == 'Major' for r in selected) == 1, 'Expected exactly one upstream Major')
    require(all(r['promoter_class'] in ['Major', 'Minor'] for r in selected), 'Unknown promoter class')
    require(len({r['seqnames'] for r in selected}) == len({r['strand'] for r in selected}) == 1, 'Conflicting locus chromosomes/strands')
    require(any(r['TSS_age_group'] in OLD for r in selected) and any(r['TSS_age_group'] in YOUNG for r in selected),
            'Choose a gene containing both older and younger promoter classes')
    return sorted(selected, key=lambda row: row['tss0'])


def read_transcripts(path, gene, chrom, strand, required_ids):
    transcripts, gene_name, build = {}, None, False
    needle = f'gene_id "{gene}"'
    with text_open(path) as handle:
        for line in handle:
            if line.startswith('#'):
                build |= 'GRCh37' in line
                continue
            if needle not in line:
                continue
            fields = line.rstrip().split('\t')
            require(len(fields) == 9, 'Malformed GTF row')
            chromosome = fields[0] if fields[0].startswith('chr') else 'chr'+fields[0]
            require(chromosome == chrom and fields[6] == strand, 'GTF coordinate/strand conflict')
            attrs = dict(re.findall(r'(\S+) "([^"]*)"', fields[8]))
            if 'gene_name' in attrs:
                require(gene_name in [None, attrs['gene_name']], 'Inconsistent GTF gene name')
                gene_name = attrs['gene_name']
            tx = attrs.get('transcript_id')
            if tx not in required_ids or fields[2] not in ['exon', 'CDS']:
                continue
            record = transcripts.setdefault(tx, dict(id=tx, strand=strand, exons=[], cds=[]))
            record['exons' if fields[2] == 'exon' else 'cds'].append([int(fields[3])-1, int(fields[4])])
    require(build, 'GTF does not declare GRCh37; do not mix hg38 and hg19')
    require(set(transcripts) == set(required_ids), f'GTF missing transcripts: {sorted(set(required_ids)-set(transcripts))}')
    for record in transcripts.values():
        record['exons'] = sorted(set(map(tuple, record['exons'])))
        record['cds'] = sorted(set(map(tuple, record['cds'])))
        require(record['exons'] and all(a < b for a, b in record['exons']), 'Invalid exon intervals')
    return gene_name or gene, [transcripts[key] for key in sorted(transcripts)]


def best_chain(path, chrom, start, end, minimum_score=10000):
    """One highest-scoring overlapping chain; never union unrelated chains.

    Blocks indicate alignability, NOT nucleotide identity or expression.
    Uncovered reference bases mean no block in this selected chain, not
    demonstrated sequence absence from the query species.
    """
    best, active, candidates = None, None, 0
    def finish():
        nonlocal best, candidates
        if active is None:
            return
        require(active['t_cursor'] == active['t_end'] and active['q_cursor'] == active['q_end'], 'Truncated/inconsistent chain blocks')
        if not active['blocks']:
            return
        candidates += 1
        active['aligned_bp'] = sum(b['end0']-b['start0'] for b in active['blocks'])
        if best is None or (active['score'], active['aligned_bp'], -active['id']) > (best['score'], best['aligned_bp'], -best['id']):
            best = active
    with text_open(path) as handle:
        for line in handle:
            if line.startswith('chain '):
                finish()
                f = line.split()
                require(len(f) == 13, 'Malformed chain header')
                active = None
                if f[2] != chrom or int(f[5]) >= end or int(f[6]) <= start or float(f[1]) < minimum_score:
                    continue
                require(f[4] == '+', 'Only plus reference-strand chains are supported')
                active = dict(score=float(f[1]), id=int(f[12]), t_size=int(f[3]),
                              t_start=int(f[5]), t_end=int(f[6]), t_cursor=int(f[5]),
                              q_chrom=f[7], q_size=int(f[8]), q_strand=f[9],
                              q_start=int(f[10]), q_end=int(f[11]), q_cursor=int(f[10]), blocks=[])
                require(active['q_strand'] in ['+', '-'], 'Invalid query strand')
            elif active is not None and line.strip() and not line.startswith('#'):
                values = list(map(int, line.split()))
                require(len(values) in [1, 3] and values[0] > 0 and all(v >= 0 for v in values), 'Malformed chain block')
                size = values[0]
                t, q = active['t_cursor'], active['q_cursor']
                left, right = max(start, t), min(end, t+size)
                if left < right:
                    qleft, qright = q+left-t, q+right-t
                    if active['q_strand'] == '-':
                        qleft, qright = active['q_size']-qright, active['q_size']-qleft
                    active['blocks'].append(dict(start0=left, end0=right, query_start0=qleft, query_end0=qright))
                active['t_cursor'] += size+(values[1] if len(values) == 3 else 0)
                active['q_cursor'] += size+(values[2] if len(values) == 3 else 0)
                require(active['t_cursor'] <= active['t_end'] and active['q_cursor'] <= active['q_end'], 'Chain overrun')
    finish()
    if best is None:
        return dict(status='no qualifying chain', blocks=[], candidates=candidates)
    return dict(status='selected chain', candidates=candidates,
                **{k: v for k, v in best.items() if k not in ['t_cursor', 'q_cursor']})



def validate_data(data):
    require(data.get('schema_version')==2 and data.get('organ')=='Testis','Unsupported prepared data')
    require(data.get('assembly')=='GRCh37/hg19','Genome assembly mismatch')
    start,end=data['start0'],data['end0']
    require(isinstance(start,int) and isinstance(end,int) and 0<=start<end and end-start<=200000,'Invalid locus bounds')
    require(data['chrom'].startswith('chr') and data['strand'] in ['+','-'],'Invalid chromosome/strand')
    promoters=data['promoters']
    require(len(promoters)>=2 and len({p['promoterId'] for p in promoters})==len(promoters),'Missing or duplicate promoters')
    require([p['tss0'] for p in promoters]==sorted(p['tss0'] for p in promoters),'Promoters not in genomic order')
    require(sum(p['promoter_class']=='Major' for p in promoters)==1,'Expected one inherited Major')
    require(any(p['TSS_age_group'] in OLD for p in promoters) and any(p['TSS_age_group'] in YOUNG for p in promoters),'Missing age comparison')
    tx_ids=set()
    for p in promoters:
        require(p['geneId']==data['gene_id'] and p['seqnames']==data['chrom'] and p['strand']==data['strand'],'Promoter locus mismatch')
        require(p['promoter_class'] in ['Major','Minor'] and p['TSS_age_group'] in OLD|YOUNG,'Invalid promoter label')
        require(p['tss0']==int(p['start'])-1 and start<=p['tss0']<end,'Invalid TSS coordinate')
        require(math.isfinite(p['activity']) and p['activity']>0 and p['activity']==float(p['mean_promoter_activity']),'Activity changed/nonfinite')
        require(p['internalPromoter'].lower()=='false','Internal promoter found')
        tx_ids.update(p['transcript_ids'])
    transcripts=data['transcripts']
    require(len({t['id'] for t in transcripts})==len(transcripts) and {t['id'] for t in transcripts}==tx_ids,'Transcript association changed')
    for tx in transcripts:
        require(tx['strand']==data['strand'] and tx['exons'],'Invalid transcript model')
        for a,b in tx['exons']+tx['cds']:
            require(isinstance(a,int) and isinstance(b,int) and 0<=a<b,'Invalid transcript interval')
    require([(r['species'],r['genome']) for r in data['alignments']]==[(s,g) for s,g,_ in CHAIN_FILES],'Unexpected alignment species/order')
    for row in data['alignments']:
        blocks=row['blocks']
        require(all(start<=b['start0']<b['end0']<=end and 0<=b['query_start0']<b['query_end0'] for b in blocks),'Invalid chain block')
        require(all(a['end0']<=b['start0'] for a,b in zip(blocks,blocks[1:])),'Overlapping/reordered chain blocks')
        if blocks:
            require(row['t_size']>=end and sum(b['end0']-b['start0'] for b in blocks)==row['aligned_bp'],'Alignment length mismatch')
    return data


def compact_prepared_data(raw, source):
    """Whitelist the three-track fields; discard unrelated legacy track data."""
    require(raw.get('schema_version') in [1,2],'Unsupported prepared-data schema')
    keys=['gene_id','gene_name','organ','assembly','chrom','strand','start0','end0',
          'promoters','transcripts','alignments','reference']
    data={key:raw[key] for key in keys}
    data.update(schema_version=2,created_utc=datetime.now(timezone.utc).isoformat(),
                method=dict(METHOD),plotted_tracks=list(TRACKS),
                input_snapshots=[s for s in raw['input_snapshots'] if Path(s['path']).suffix not in ['.bw','.bigWig']],
                prepared_data_source=snapshot(source,True))
    return validate_data(data)


def prepare(args):
    inputs=[snapshot(args.promoters,True),snapshot(args.ages,True),snapshot(args.gtf)]
    promoters=load_promoters(args.promoters,args.ages,args.gene)
    chrom,strand=promoters[0]['seqnames'],promoters[0]['strand']
    require(chrom.startswith('chr') and strand in ['+','-'],'Expected canonical Human chromosome/strand')
    start,end=max(0,promoters[0]['tss0']-args.flank),promoters[-1]['tss0']+args.flank+1
    require(end-start<=200000,'Locus exceeds 200 kb')
    tx_ids={tx for p in promoters for tx in p['transcript_ids']}
    gene_name,transcripts=read_transcripts(args.gtf,args.gene,chrom,strand,tx_ids)
    print(f'GENE {gene_name} {args.gene}: {len(promoters)} promoters; {len(transcripts)} transcript models',flush=True)
    alignments=[]
    reference_sizes=set()
    for species,genome,name in CHAIN_FILES:
        path=args.chain_dir/name
        state=snapshot(path)
        print(f'READING CHAIN {species}',flush=True)
        row=best_chain(path,chrom,start,end)
        if row['blocks']:
            require(end<=row['t_size'],'Locus exceeds chain reference chromosome')
            reference_sizes.add(row['t_size'])
        row.update(species=species,genome=genome,source=state)
        alignments.append(row);inputs.append(state)
    require(len(reference_sizes)<=1,'Inconsistent chain reference chromosome lengths')
    require(all(unchanged(r) for r in inputs),'An input changed during extraction')
    return validate_data(dict(schema_version=2,created_utc=datetime.now(timezone.utc).isoformat(),
        gene_id=args.gene,gene_name=gene_name,organ='Testis',assembly='GRCh37/hg19',
        chrom=chrom,strand=strand,start0=start,end0=end,promoters=promoters,
        transcripts=transcripts,alignments=alignments,input_snapshots=inputs,
        reference=dict(doi='10.1101/gr.231449.117',figure='1B',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC5932608/'),
        method=dict(METHOD),plotted_tracks=list(TRACKS)))


def build_figure(data,highlight_bp=70):
    validate_data(data)
    require(highlight_bp>=1,'Highlight width must be positive')
    start,end=data['start0'],data['end0']
    span=end-start;n_tx=len(data['transcripts'])
    height=max(4.4,3.5+n_tx*.26)
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'pdf.compression':0,'axes.grid':False,'axes.linewidth':.7}):
        fig=plt.figure(figsize=(11,height))
        grid=fig.add_gridspec(3,1,height_ratios=[.85,max(.8,n_tx*.23),1.55],
                             left=.22,right=.975,bottom=.17,top=.78,hspace=.52)
        axes=[fig.add_subplot(grid[i]) for i in range(3)]
        for axis in axes:
            axis.set_xlim(start,end)
            axis.tick_params(axis='x',bottom=False,labelbottom=False)
            for side in ['top','right','bottom']:axis.spines[side].set_visible(False)
            axis.grid(False)
            for p in data['promoters']:
                color='#F4A3A8' if p['TSS_age_group'] in OLD else '#43CED6'
                axis.axvspan(p['tss0']-highlight_bp,p['tss0']+highlight_bp+1,
                             color=color,alpha=.35,lw=0,zorder=1.5,rasterized=False)
        activity_axis,tx_axis,chain_axis=axes
        for p in data['promoters']:
            color='#E4AA38' if p['promoter_class']=='Major' else '#397FC7'
            activity_axis.vlines(p['tss0'],0,p['activity'],color=color,lw=2.5)
            activity_axis.plot(p['tss0'],p['activity'],'o',color=color,ms=4)
            activity_axis.annotate(f"P{p['promoterId']} {p['promoter_class']}\n{p['activity']:.2f}",
                (p['tss0'],p['activity']),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7)
        activity_axis.set_ylim(0,max(p['activity'] for p in data['promoters'])*1.6)
        activity_axis.yaxis.set_major_locator(MaxNLocator(3))
        activity_axis.set_ylabel('ProActiv\nmean activity',rotation=0,ha='right',va='center',labelpad=24)
        for i,tx in enumerate(data['transcripts']):
            y=n_tx-1-i
            left,right=max(start,tx['exons'][0][0]),min(end,tx['exons'][-1][1])
            if left>=right:continue
            color='#174FCA' if tx['strand']=='-' else '#C62A36'
            tx_axis.plot([left,right],[y,y],color=color,lw=.65)
            for x in np.arange(left+span*.025,right,span*.055):
                tx_axis.plot(x,y,marker='<' if tx['strand']=='-' else '>',ms=2.4,color=color,mew=0)
            for a,b in tx['exons']:
                a,b=max(a,start),min(b,end)
                if a<b:tx_axis.add_patch(Rectangle((a,y-.15),b-a,.30,color=color,lw=0))
            for a,b in tx['cds']:
                a,b=max(a,start),min(b,end)
                if a<b:tx_axis.add_patch(Rectangle((a,y-.23),b-a,.46,color=color,lw=0))
            tx_axis.text(-.012,y,tx['id'],transform=tx_axis.get_yaxis_transform(),ha='right',va='center',fontsize=7)
        tx_axis.set_ylim(-.6,n_tx-.4);tx_axis.set_yticks([])
        tx_axis.spines['left'].set_visible(False)
        tx_axis.set_title('Ensembl transcripts (displayed region only)',loc='left',fontsize=8,pad=3)
        rows=[dict(species='Human',genome='hg19',blocks=[dict(start0=start,end0=end)])]+data['alignments']
        for i,row in enumerate(rows):
            y=len(rows)-1-i
            chain_axis.add_patch(Rectangle((start,y-.32),span,.64,facecolor='#F4F4F4',edgecolor='#DDDDDD',lw=.25))
            for block in row['blocks']:
                chain_axis.add_patch(Rectangle((block['start0'],y-.32),block['end0']-block['start0'],.64,facecolor='#626262',lw=0))
            chain_axis.text(-.012,y,f"{row['species']} ({row['genome']})",
                           transform=chain_axis.get_yaxis_transform(),ha='right',va='center',fontsize=7.5)
            if not row['blocks']:
                chain_axis.text((start+end)/2,y,'no qualifying chain',ha='center',va='center',fontsize=6.5,color='#666666')
        chain_axis.set_ylim(-.6,len(rows)-.4);chain_axis.set_yticks([])
        chain_axis.spines['left'].set_visible(False)
        chain_axis.set_title('Pairwise chain-aligned blocks (not MULTIZ; not sequence identity)',loc='left',fontsize=8,pad=3)
        chain_axis.spines['bottom'].set_visible(True)
        chain_axis.tick_params(axis='x',bottom=True,labelbottom=True,labelsize=7,length=3)
        chain_axis.xaxis.set_major_locator(MaxNLocator(5,integer=True))
        chain_axis.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f'{int(round(value))+1:,}'))
        chain_axis.set_xlabel(f"{data['chrom']} genomic position (hg19; 1-based display)",fontsize=8)
        fig.text(.22,.975,f"{data['gene_name']} locus | Human Testis",ha='left',va='top',fontsize=13,fontweight='bold')
        fig.text(.22,.915,f"{data['gene_id']}  |  {data['chrom']}:{start+1:,}-{end:,}  |  strand {data['strand']}",
                 ha='left',va='top',fontsize=8)
        for i,p in enumerate(data['promoters']):
            kind='Older' if p['TSS_age_group'] in OLD else 'Younger'
            color='#C35A65' if kind=='Older' else '#008E9C'
            fig.text(.22+i*.755/max(2,len(data['promoters'])),.855,
                     f"{kind}: P{p['promoterId']} ({AGE_NAMES[p['TSS_age_group']]}; {p['promoter_class']})",
                     color=color,fontsize=8,va='top')
        fig.text(.22,.043,'Promoter ages and Major/Minor labels are inherited; uncovered chain positions do not prove sequence absence.',fontsize=6.5)
        fig.text(.22,.019,'Activity: Testis condition mean. Pairwise chain blocks show alignability, not sequence identity or expression.',fontsize=6.5)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for text in fig.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text():continue
            if text in chain_axis.get_xticklabels() and not start<=text.get_position()[0]<=end:continue
            if any(text in ax.get_yticklabels() and not ax.get_ylim()[0]<=text.get_position()[1]<=ax.get_ylim()[1] for ax in axes):continue
            box=text.get_window_extent(renderer)
            require(box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1,
                    f'Text outside figure: {text.get_text()}')
        return fig


def plot(data,destination,highlight_bp=70):
    fig=build_figure(data,highlight_bp)
    try:
        with plt.rc_context({'pdf.fonttype':42,'pdf.compression':0}):
            fig.savefig(destination,format='pdf',facecolor='white',metadata={
                'Title':f"Human Testis {data['gene_name']} promoter locus",
                'Subject':'Three tracks only: tissue-mean ProActiv activity, Ensembl models, selected pairwise-chain blocks; no new age inference.'})
    finally:plt.close(fig)


def write_table(path,rows,columns):
    with path.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,delimiter='\t',extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gene',default=DEFAULT_GENE,help='Ensembl gene ID; default SAYSD1')
    parser.add_argument('--promoters',type=Path,default=DEFAULT_PROMOTERS)
    parser.add_argument('--ages',type=Path,default=DEFAULT_AGE)
    parser.add_argument('--gtf',type=Path,default=DEFAULT_GTF)
    parser.add_argument('--chain-dir',type=Path,default=DEFAULT_CHAINS)
    parser.add_argument('--flank',type=int,default=1200)
    parser.add_argument('--highlight-bp',type=int,default=70)
    parser.add_argument('--prepared-data',type=Path,help='Offline redraw from an exported locus_data.json (schema 1 or 2)')
    parser.add_argument('--outdir',type=Path,default=HERE/'results/Human_Testis_SAYSD1')
    args=parser.parse_args(argv)
    require(args.flank>=1 and args.highlight_bp>=1,'Flank/highlight widths must be positive')
    require(not args.outdir.is_symlink() and (not args.outdir.exists() or not any(args.outdir.iterdir())),
            'Output directory is not empty; choose a new --outdir')
    data=compact_prepared_data(json.loads(args.prepared_data.read_text()),args.prepared_data) if args.prepared_data else prepare(args)
    args.outdir.mkdir(parents=True,exist_ok=True)
    pdf=args.outdir/f"Human_Testis_{data['gene_name']}_promoter_locus.pdf"
    plot(data,pdf,args.highlight_bp)
    (args.outdir/'locus_data.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    write_table(args.outdir/'selected_promoters.tsv',data['promoters'],
                ['promoterId','geneId','seqnames','start','tss0','strand','promoter_class','mean_promoter_activity','TSS_age_group','txId'])
    blocks=[dict(species=r['species'],genome=r['genome'],chain_id=r.get('id',''),query_chrom=r.get('q_chrom',''),
                 query_strand=r.get('q_strand',''),**b) for r in data['alignments'] for b in r['blocks']]
    write_table(args.outdir/'chain_blocks.tsv',blocks,
                ['species','genome','chain_id','start0','end0','query_chrom','query_strand','query_start0','query_end0'])
    manifest=dict(status='complete',schema_version=2,created_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha256(Path(__file__)),python=sys.version,matplotlib=matplotlib.__version__,
                  plotted_tracks=TRACKS,method=data['method'],inputs=data['input_snapshots'],
                  outputs={p.name:sha256(p) for p in args.outdir.iterdir() if p.is_file()})
    (args.outdir/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'SAVED {pdf}',flush=True)


if __name__=='__main__':main()
