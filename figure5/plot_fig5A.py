#!/usr/bin/env python3
"""Fig. 4A-style cross-species hierarchy of measured promoter activity.

One column = species x organ x developmental stage; one feature = a common
single-copy orthogroup. In each species/condition, take the maximum stored
proActiv .mean among the gene's non-internal promoters. This is a condition-
specific major-promoter activity, not gene expression, PSI, or the activity
of a fixed orthologous promoter. All plotted conditions use the same features.

The references use scipy's Jensen-Shannon distance (square root of the
divergence, natural log), then seaborn's default average/Euclidean clustering
of distance-profile vectors. That is the default here; --cluster-on direct-jsd
instead uses the JSD distances as the linkage input, a different analysis.
No activity log/threshold/re-scaling other than the unit-sum normalization
intrinsic to JSD. Genuine zero activity is kept; missing is never zero-filled.

Outputs: activity matrix, JSD matrix, leaf order/condition metadata, linkage,
run manifest, and one PDF plus PNG. Inputs and existing outputs are read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import jensenshannon, pdist, squareform

BASE = Path('/mnt/ruiyanhou/nfs_share2')
INPUT = BASE / 'proactiv_redo/1.proactiv_result'
ORTHOLOGS = BASE / ('proactiv_redo/branch/3.get_homologous_genes/'
                    '3.get_singlecopy_orthologue_gene/singlecopy_orthologue_genes.tsv')
ROOT = Path(__file__).resolve().parent
SPECIES = {'Human':'red', 'Macaque':'orange', 'Mouse':'gray',
           'Rat':'black', 'Rabbit':'brown', 'Chicken':'violet'}
ORGANS = {'Brain':'#3399CC', 'Cerebellum':'#33CCFF', 'Heart':'#CC0000',
          'Kidney':'#CC9900', 'Liver':'#339900', 'Ovary':'#CC3399', 'Testis':'#FF6600'}
ALIASES = {'WholeBrain':'Brain','Forebrain':'Brain','Hindbrain':'Cerebellum'}
REFS = {
    'get_JSD_matrix.py': '9b4c59ff7f8f5d4d4383f89f09cb1a87a3cdbd53b4910a575662f30cffae200d',
    'plot_fig4A.py': '1bb481d0843def3e586e8d6b5e995ffb225fcbf37805338693e528df5ce8b3e4',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib,'file_digest') else _sha(f)


def _sha(f):
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1024*1024), b''):
        h.update(chunk)
    return h.hexdigest()


def stamp(p):
    s = p.stat()
    return [s.st_size, s.st_mtime_ns]


def inventory(root, species, organs):
    records, ignored = [], []
    for sp in species:
        directory = root / sp
        require(directory.is_dir(), f'Missing species directory: {directory}')
        seen = set()
        for path in sorted(directory.glob('*_rowData.tsv')):
            label = path.name.removesuffix('_rowData.tsv')
            parts = label.split('_', 2)
            if len(parts) != 3 or parts[0] != sp or parts[1].lower() == 'all':
                ignored.append(str(path))
                continue
            _, raw_organ, stage = parts
            organ = ALIASES.get(raw_organ, raw_organ)
            require(organ in ORGANS and bool(stage), f'Unexpected organ/stage filename: {path}')
            if organ not in organs:
                continue
            with path.open(encoding='utf-8-sig', newline='') as h:
                head = next(csv.reader(h, delimiter='\t'))
            cols = [c for c in head if c.endswith('.mean') and not c.endswith('.gene.mean')]
            require(len(cols) == 1 and {'promoterId','geneId','internalPromoter'} <= set(head),
                    f'Expected one promoter .mean column and promoter metadata: {path}')
            partner = path.with_name(label + '_activity.tsv')
            replicate_n = None
            if partner.is_file():
                with partner.open(encoding='utf-8-sig', newline='') as h:
                    rd = csv.reader(h, delimiter='\t'); hh, rr = next(rd), next(rd)
                require(len(rr) in (len(hh), len(hh)+1), f'Unexpected sample activity header: {partner}')
                replicate_n = len(rr)-1
                require(replicate_n > 0, f'No sample columns: {partner}')
            require(label not in seen, f'Duplicate condition: {label}')
            seen.add(label)
            records.append(dict(condition=label, species=sp, organ=organ, organ_source=raw_organ,
                stage=stage, source_file=str(path), activity_column=cols[0],
                replicate_columns=replicate_n, source_stat=stamp(path)))
        require(bool(seen), f'No conditions for {sp} matching the organ filter')
    require(len(records) > 1, 'At least two conditions are needed for clustering')
    return records, ignored


def load_map(path, species):
    m = pd.read_csv(path, sep='\t', dtype=str)
    keys = ['Orthogroup', *(sp.lower() for sp in species)]
    require(set(keys) <= set(m.columns), f'Missing mapping columns: {set(keys)-set(m.columns)}')
    m = m[keys].copy()
    for c in keys:
        m[c] = m[c].str.strip()
        require(m[c].notna().all() and m[c].str.fullmatch(r'[^\s,;]+').all() and m[c].is_unique,
                f'Expected one distinct, non-missing ID per row of {c}')
    return m.set_index('Orthogroup')


def group_max(frame, col, gene_to_group):
    require(frame.promoterId.notna().all() and frame.promoterId.is_unique, 'Nonunique/missing promoterId')
    flags = frame.internalPromoter.str.strip().str.lower()
    known = flags.isin(['false','true','0','1'])
    require((known | flags.isna()).all(), 'Unrecognized internalPromoter flag')
    sub = frame.loc[flags.isin(['false','0']) & frame.geneId.isin(gene_to_group), ['geneId',col]].copy()
    sub[col] = pd.to_numeric(sub[col], errors='raise')
    observed = sub[col].dropna()
    require(np.isfinite(observed).all() and observed.ge(0).all(), 'Negative or nonfinite activity')
    values = sub.dropna(subset=[col]).groupby('geneId',sort=False)[col].max()
    values.index = values.index.map(gene_to_group)
    return values, dict(eligible_rows=len(sub), observed_rows=len(observed),
                        missing_activity_rows=int(sub[col].isna().sum()), unknown_internal_rows=int(flags.isna().sum()),
                        observed_groups=len(values))


def collect(records, m, workers):
    keys = {sp: dict(zip(m[sp.lower()], m.index)) for sp in {r['species'] for r in records}}

    def work(rec):
        p, col = Path(rec['source_file']), rec['activity_column']
        require(stamp(p) == rec['source_stat'], f'Input changed before read: {p}')
        frame = pd.read_csv(p, sep='\t', usecols=['promoterId','geneId','internalPromoter',col],
                            dtype={'promoterId':str,'geneId':str,'internalPromoter':str})
        vals, qc = group_max(frame, col, keys[rec['species']])
        require(stamp(p) == rec['source_stat'], f'Input changed during read: {p}')
        return vals.reindex(m.index).rename(rec['condition']), qc

    values = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (rec, (series,qc)) in enumerate(zip(records,pool.map(work, records)),1):
            rec.update(qc)
            values.append(series)
            if i % 75 == 0 or i == len(records):
                print(f'LOAD {i}/{len(records)} {rec["condition"]}', flush=True)
    a = pd.concat(values, axis=1)
    complete = a.notna().all(axis=1)
    a = a.loc[complete]
    nonzero = a.gt(0).any(axis=1)
    a = a.loc[nonzero]
    require(len(a) > 1, 'Fewer than two observed/nonzero common orthogroups; JSD is not informative')
    require(a.sum(axis=0).gt(0).all(), 'All-zero condition after intersection; JSD undefined')
    return a, dict(mapped=len(m), absent_or_missing=int((~complete).sum()),
                   all_zero=int((~nonzero).sum()), included=len(a))


def cluster(a, cluster_on):
    p = np.ascontiguousarray(a.to_numpy(dtype=float).T)
    condensed = pdist(p, 'jensenshannon')
    require(np.isfinite(condensed).all() and (condensed >= 0).all() and
            (condensed <= np.sqrt(np.log(2))+1e-12).all(), 'Invalid Jensen-Shannon distances')
    d = squareform(condensed)
    n = len(d)
    for i,j in [(0,n-1),(0,1),(n//2,n-1)]:
        np.testing.assert_allclose(d[i,j], jensenshannon(p[i],p[j]), atol=1e-10, rtol=1e-8)
    z = linkage(pdist(d, 'euclidean') if cluster_on == 'distance-profiles' else condensed, method='average')
    order = leaves_list(z)
    require(np.array_equal(np.sort(order), np.arange(n)), 'Leaf order is not a permutation')
    return d,z,order


def plot(d,z,order,records,feature_n,output):
    n = len(order)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                        'pdf.compression':0,'axes.linewidth':.55})
    fig = plt.figure(figsize=(7.2,7.45), facecolor='white')
    left, bottom, size = .15,.125,.755
    ax = fig.add_axes([left,bottom,size,size*7.2/7.45])
    h = ax.get_position().height
    t = fig.add_axes([.026,bottom,.092,h])
    plot_order = dendrogram(z, orientation='left', no_labels=True, ax=t,
                            link_color_func=lambda _: '#454545')['leaves']
    require(np.array_equal(order, plot_order), 'Dendrogram order differs from matrix order')
    t.set_ylim(10*n,0);t.set_axis_off()
    for lines in t.collections:
        lines.set_linewidth(.5)
    colors = np.array([[to_rgba(SPECIES[records[i]['species']]),to_rgba(ORGANS[records[i]['organ']])] for i in order])
    annot = fig.add_axes([.122,bottom,.024,h]);annot.imshow(colors,aspect='auto',interpolation='none');annot.set_axis_off()
    # Native-resolution RGB image for hundreds of thousands of cells; tree and text remain vectors.
    vmax = float(d.max()) or 1.
    im = ax.imshow(d[np.ix_(order,order)], interpolation='none', cmap='magma',vmin=0,vmax=vmax,aspect='equal')
    ax.set_axis_off()
    cax = fig.add_axes([.921,bottom+.065,.014,h-.13]); c = fig.colorbar(im,cax=cax)
    if c.solids is not None:
        c.solids.set_rasterized(False)
    c.ax.tick_params(labelsize=7,length=2,pad=2)
    c.set_label('Jensen–Shannon distance',fontsize=8,labelpad=4)
    fig.text(.15,.968,'Promoter activity across development',fontsize=12,weight='semibold',va='top')
    fig.text(.15,.939,f'{n:,} organ-stage conditions  |  {feature_n:,} shared single-copy orthogroups',fontsize=8,va='top')
    for y,title,mapping,field in [(.072,'Species',SPECIES,'species'),(.037,'Organ',ORGANS,'organ')]:
        shown={r[field] for r in records}
        patches=[Patch(facecolor=c,edgecolor='none',label=k) for k,c in mapping.items() if k in shown]
        fig.text(.15,y,title+':',fontsize=8,ha='left',va='center',weight='semibold')
        fig.legend(handles=patches,ncol=len(patches),frameon=False,loc='center left',
                   bbox_to_anchor=(.235,y),borderaxespad=0,handlelength=.9,handletextpad=.4,columnspacing=.9,fontsize=7.5)
    fig.canvas.draw()
    pdf = output/'promoter_activity_JSD_hierarchy.pdf'
    fig.savefig(pdf,metadata={'Title':'Promoter activity across development',
        'Subject':'Jensen-Shannon distances of condition-specific major promoter activities across shared single-copy orthogroups.'})
    fig.savefig(output/'promoter_activity_JSD_hierarchy.png',dpi=180)
    plt.close(fig)
    raw = pdf.read_bytes()
    require(b'/FontFile2' in raw and b'/Indexed' not in raw, 'Expected embedded TrueType fonts and non-indexed color')


def self_test():
    f = pd.DataFrame({'promoterId':['1','2','3','4','5'], 'geneId':['a','a','a','b','c'],
         'internalPromoter':['false','false','true','FALSE','FALSE'],'m':[1.,4.,99.,0.,np.nan]})
    x,qc = group_max(f,'m',{'a':'A','b':'B','c':'C'})
    assert x.to_dict() == {'A':4.,'B':0.} and qc['missing_activity_rows'] == 1
    a = pd.DataFrame({'a':[1,0,2],'b':[2,0,4],'c':[0,1,0]})
    for mode in ('distance-profiles','direct-jsd'):
        d,z,order = cluster(a,mode)
        np.testing.assert_allclose([d[0,1],d[0,2]], [0,np.sqrt(np.log(2))],atol=1e-12)
        assert set(order[:2]) == {0,1} or set(order[1:]) == {0,1}
    print('SELF-TEST PASS: true zero, missing, internal exclusion, normalization, JSD and clustering')


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-root',type=Path,default=INPUT)
    parser.add_argument('--ortholog-map',type=Path,default=ORTHOLOGS)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--species',nargs='+',choices=SPECIES,default=list(SPECIES))
    parser.add_argument('--organs',nargs='+',choices=ORGANS,default=list(ORGANS))
    parser.add_argument('--cluster-on',choices=('distance-profiles','direct-jsd'),default='distance-profiles')
    parser.add_argument('--workers',type=int,choices=range(1,9),default=4)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test();return
    require(len(set(args.species)) == len(args.species), 'Duplicate species')
    out = args.output_dir.resolve()
    require(not out.is_relative_to(args.input_root.resolve()), 'Do not write into the activity input directory')
    if not args.check_only:
        require(not out.exists(), f'Output directory already exists: {out}; use a fresh --output-dir')
    map_hash = sha(args.ortholog_map)
    mapping = load_map(args.ortholog_map,args.species)
    records, ignored = inventory(args.input_root,args.species,args.organs)
    sizes = dict(Counter(r['species'] for r in records))
    print('INVENTORY',json.dumps(dict(conditions=sizes,orthogroups=len(mapping),ignored=ignored)),flush=True)
    if args.check_only:
        return
    a, groups = collect(records,mapping,args.workers)
    d,z,order = cluster(a,args.cluster_on)
    require(sha(args.ortholog_map) == map_hash, 'Ortholog map changed during run')
    for rec in records:
        require(stamp(Path(rec['source_file'])) == rec['source_stat'], 'Upstream activity changed during run')
    out.mkdir(parents=True, exist_ok=False)
    a.to_csv(out/'major_promoter_activity_by_orthogroup.tsv.gz',sep='\t',float_format='%.17g')
    mapping.loc[a.index].to_csv(out/'included_orthogroups.tsv',sep='\t')
    pd.DataFrame(d,index=a.columns,columns=a.columns).to_csv(out/'promoter_activity_JSD_matrix.tsv.gz',sep='\t',float_format='%.17g')
    meta = pd.DataFrame([records[i] for i in order]);meta.insert(0,'display_position',range(1,len(meta)+1))
    meta.to_csv(out/'condition_cluster_order.tsv',sep='\t',index=False)
    pd.DataFrame(z,columns=['left_node','right_node','height','leaf_n']).to_csv(out/'average_linkage.tsv',sep='\t',index=False)
    plot(d,z,order,records,len(a),out)
    summary = dict(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                   cluster_on=args.cluster_on, linkage='average',
                   metric='sqrt(Jensen-Shannon divergence), natural log; each condition normalized to sum 1',
                   feature_definition='Within each ortholog gene and condition: max observed noninternal promoter .mean; not PSI',
                   no_inference_tests=True, replicate_aggregation='stored condition-level proActiv .mean',
                   group_counts=groups, conditions_by_species=sizes, condition_n=len(records),
                   jsd_max=float(d.max()), mapping_path=str(args.ortholog_map), mapping_sha256=map_hash,
                   input_root=str(args.input_root), ignored_tables=ignored, source_metadata=records,
                   script_sha256=sha(__file__), references={name:{'url':'https://github.com/StatBiomed/promoter_activity/blob/main/figure4/'+name,'sha256':digest} for name,digest in REFS.items()},
                   versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),
                   output_sha256={p.name:sha(p) for p in sorted(out.iterdir())})
    (out/'run_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('COMPLETE',out,json.dumps(dict(groups=groups,conditions=sizes,jsd_max=summary['jsd_max'])),flush=True)


if __name__ == '__main__':
    main()
