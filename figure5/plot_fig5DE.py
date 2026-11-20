#!/usr/bin/env python3
"""Joint positive-parent TF-MoDISco clustering across species and organs.

Follows the USER-SUPPLIED reference workflow, not a claim about current CREsted
defaults: PPM IC trim (>0.2 bits, A/T=.27,C/G=.23), all-vs-all memelite.tomtom
(both strands), E=N*p profiles -> PCA<=50 -> t-SNE(perplexity15). Leiden uses a
weighted 15-nearest-neighbour graph in PCA space, resolution1, NOT t-SNE space.

Only each task/status.json-selected, completed, audited attempt is read. Old
attempts, report copies, negative patterns and nested subpatterns are NOT extra
motifs. WholeBrain remains distinct from Brain. PPM sequence similarities do not
identify TFs or prove activation/conservation. pos_patterns is an algorithmic
metacluster label; net CWM signs are retained as QC, not used to filter motifs.

Outputs in a NEW run directory: motif_tsne.pdf (cluster/species/organ pages),
cluster_members.tsv (numeric cluster order, one motif per row),
cluster_summary.tsv, cluster_members_lists.tsv, motif_index.tsv, motifs.meme,
matrices.npz, input_sources.tsv, graph_edges.tsv, and run_manifest.json.
No original source is changed and no package installation is performed.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata as package_metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback
import warnings

# Bound shared-server CPU usage. The Tomtom thread count is passed explicitly.
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np

BASE=Path('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/revise/train_model/redo/organ_species_model_explanation')
DEFAULT_INPUT=BASE/'tfmodisco_results/organ_mean_TFMoDISco_full_20261105T063935249044Z'
SPECIES=('Human','Mouse','Rat','Rabbit','Macaque','Chicken')
VERSION='organ_species_joint_motif_cluster_v1'


@dataclass
class Source:
    species: str
    organ: str
    h5_path: Path
    attempt: Path
    status_path: Path
    source_sha256: str
    expected_positive: int


@dataclass
class Motif:
    motif_id: str
    species: str
    organ: str
    pattern_name: str
    source_h5: str
    source_attempt: str
    source_sha256: str
    ppm: np.ndarray
    original_length: int
    trim_start: int
    trim_end: int
    n_seqlets: int
    trim_status: str
    n_informative_positions: int
    ic_mean: float
    ic_total: float
    cwm_net_sum: float


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def write_tsv(path,rows,fieldnames=None):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        fields=fieldnames or list(rows[0])
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t')
        writer.writeheader();writer.writerows(rows)


def require(condition,message):
    if not condition:raise ValueError(message)


def natural_pattern_key(name):
    match=re.fullmatch(r'pattern_(\d+)',name)
    return (int(match.group(1)),name) if match else (10**9,name)


def normalize_ppm(ppm,motif_id):
    ppm=np.asarray(ppm,dtype=np.float64)
    require(ppm.ndim==2 and ppm.shape[1]==4 and len(ppm)>0,f'{motif_id}: expected nonempty (L,4) PPM')
    require(np.isfinite(ppm).all() and np.all(ppm>=0),f'{motif_id}: invalid/nonfinite/negative PPM')
    sums=ppm.sum(1,keepdims=True)
    require(np.all(sums>0),f'{motif_id}: zero-total PPM row')
    return ppm/sums


def information_content(ppm,background):
    ppm=np.asarray(ppm,dtype=float);background=np.asarray(background,dtype=float)
    require(background.shape==(4,) and np.isfinite(background).all() and np.all(background>0)
            and np.isclose(background.sum(),1), 'Background must have four positive frequencies summing to1')
    # Preserve the supplied script's epsilon-stabilized IC formula exactly.
    eps=1e-6
    return np.sum(ppm*np.log2((ppm+eps)/(background[None,:]+eps)),axis=1)


def trim_ppm(ppm,background,threshold,flank):
    require(math.isfinite(threshold) and threshold>=0,'IC threshold must be finite and nonnegative')
    require(isinstance(flank,(int,np.integer)) and flank>=0,'Trim flank must be a nonnegative integer')
    ic=information_content(ppm,background);keep=np.flatnonzero(ic>threshold)
    if not len(keep):return ppm,0,len(ppm)
    a=max(0,int(keep[0])-flank);b=min(len(ppm),int(keep[-1])+1+flank)
    return ppm[a:b],a,b


def discover_sources(batch,species=None,organs=None,allow_partial=False,verify_hashes=True):
    batch=Path(batch).resolve(strict=True)
    manifest=read_json(batch/'batch_manifest.json');signature=manifest['signature']
    require(signature.get('version')=='organ_mean_ig_tfmodisco_v1' and signature.get('mode')=='full',
            'Expected full organ-mean TF-MoDISco batch manifest')
    plan=signature['tasks'];keys=[(t['species'],t['organ']) for t in plan]
    require(bool(keys) and len(keys)==len(set(keys)),'Empty or duplicate task plan')
    requested_species=set(species or []);requested_organs=set(organs or [])
    require(not requested_species-set(s for s,o in keys),'Unknown --species selection')
    require(not requested_organs-set(o for s,o in keys),'Unknown --organs selection')
    sources=[];audits=[];seen_paths=set()
    for target in plan:
        sp,organ=target['species'],target['organ']
        require(sp in SPECIES and re.fullmatch(r'[A-Za-z0-9_.-]+',organ) is not None,'Unsafe/unknown task name')
        if requested_species and sp not in requested_species:continue
        if requested_organs and organ not in requested_organs:continue
        folder=batch/sp/organ;status_path=folder/'status.json'
        status=read_json(status_path) if status_path.is_file() else {}
        audit=dict(species=sp,organ=organ,status=status.get('status','missing'),included=False)
        if status.get('status')!='complete':
            audits.append(audit)
            if allow_partial:continue
            raise ValueError(f'{sp}/{organ} is not complete; use --allow-partial only intentionally')
        attempt=Path(status['attempt']).resolve(strict=True)
        require(attempt.parent==(folder/'attempts').resolve(strict=True)
                and re.fullmatch(r'attempt_\d{4,}',attempt.name) is not None,'Selected attempt escapes its task')
        h5=(attempt/'modisco_results.h5').resolve(strict=True)
        require(h5.parent==attempt and h5 not in seen_paths,'Duplicate/escaped source HDF5')
        seen_paths.add(h5)
        worker=read_json(attempt/'worker_status.json');result=read_json(attempt/'result_audit.json')
        expected=status['output_sha256']['modisco_results.h5']
        require(worker.get('status')=='complete','Selected worker is not complete')
        require(worker['output_sha256']==status['output_sha256'],'Task/worker fingerprints disagree')
        require(result.get('status')=='passed' and status['audit']['status']=='passed','Source audit has not passed')
        require(result['hdf5_sha256']==expected==worker['audit']['hdf5_sha256'],'HDF5 fingerprints disagree')
        preparation_path=folder/'preparation_manifest.json';prep=read_json(preparation_path);run=read_json(folder/'run_manifest.json')
        require(sha256(preparation_path)==run['preparation_manifest_sha256'],'Preparation metadata changed')
        require(prep['species']==sp and prep['organ']==organ
                and prep['target_type']=='mean_raw_activity_all_organ_stages','Species/organ/model target mismatch')
        if verify_hashes:
            require(sha256(h5)==expected,f'HDF5 changed: {sp}/{organ}')
            require(sha256(attempt/'result_audit.json')==status['output_sha256']['result_audit.json'],'Result audit changed')
        sources.append(Source(sp,organ,h5,attempt,status_path,expected,int(result['positive_motifs'])))
        audit.update(included=True,attempt=str(attempt),h5_path=str(h5),h5_sha256=expected,
                     hash_verified=verify_hashes,positive_patterns=int(result['positive_motifs']))
        audits.append(audit)
    require(bool(audits) and bool(sources),'Selection has no complete input tasks')
    return sources,audits


def read_positive_patterns(source,background,threshold,flank):
    import h5py
    motifs=[]
    with h5py.File(source.h5_path,'r') as handle:
        group=handle.get('pos_patterns')
        names=[] if group is None else sorted((n for n in group if re.fullmatch(r'pattern_\d+',n)),key=natural_pattern_key)
        require(len(names)==source.expected_positive,f'{source.species}/{source.organ}: positive count differs from audited count')
        for name in names:
            pattern=group[name];ident=f'{source.species}__{source.organ}__pos_patterns__{name}'
            ppm=normalize_ppm(pattern['sequence'][:],ident)
            trimmed,a,b=trim_ppm(ppm,background,threshold,flank)
            ic=information_content(ppm,background);trimmed_ic=information_content(trimmed,background)
            count=np.asarray(pattern['seqlets/n_seqlets'][()]).reshape(-1)
            require(len(count)==1 and int(count[0])==count[0] and count[0]>0,f'{ident}: invalid seqlet count')
            if 'example_idx' in pattern['seqlets']:
                require(len(pattern['seqlets/example_idx'])==int(count[0]),f'{ident}: inconsistent seqlet memberships')
            cwm=np.asarray(pattern['contrib_scores'][:],dtype=float)
            require(cwm.shape==ppm.shape and np.isfinite(cwm).all(),f'{ident}: invalid CWM')
            motifs.append(Motif(ident,source.species,source.organ,name,str(source.h5_path),str(source.attempt),
                                source.source_sha256,trimmed,len(ppm),a,b,int(count[0]),
                                'trimmed_to_informative_span' if np.any(ic>threshold) else 'no_position_above_threshold_kept_full',
                                int((ic>threshold).sum()),float(trimmed_ic.mean()),float(trimmed_ic.sum()),float(cwm.sum())))
    return motifs


def transform_pvalues(pvalues,n_motifs):
    p=np.asarray(pvalues,dtype=np.float64).copy()
    require(p.shape==(n_motifs,n_motifs),'Tomtom returned an unexpected matrix shape')
    require(np.isfinite(p).all(),'Tomtom returned NaN/Inf p-values')
    # memelite0.2.0 computes survival probabilities by1-CDF; accumulated
    # float64 tail error can be a few1e-11. Bound, record and preserve raw data.
    tolerance=1e-10
    require(np.all(p>=-tolerance) and np.all(p<=1+tolerance),'Tomtom returned invalid p-values outside[0,1] beyond roundoff tolerance')
    audit=dict(raw_min=float(p.min()),raw_max=float(p.max()),numerical_out_of_range_tolerance=tolerance,
               tiny_negative_roundoff_count=int((p<0).sum()),negative_diagonal_count=int((np.diag(p)<0).sum()),
               nonpositive_floor_count=int((p<=0).sum()),above_one_roundoff_count=int((p>1).sum()),p_floor=1e-15)
    p[p<=0]=1e-15;p=np.clip(p,1e-15,1.)
    e=p*n_motifs
    similarity=-np.log10(np.clip(e,np.finfo(float).tiny,1.))
    # Compatibility with the supplied optional similarity branch only.
    np.fill_diagonal(similarity,float(similarity.max()))
    audit['max_directional_p_asymmetry']=float(np.max(np.abs(p-p.T)))
    return p,e,similarity,audit


def calculate_tomtom_matrices(motifs,n_jobs=2,raw_output=None):
    from memelite import tomtom
    params=inspect.signature(tomtom).parameters
    require({'Qs','Ts','reverse_complement','n_nearest','n_jobs'}<=set(params),'Unsupported installed memelite.tomtom API')
    matrices=[np.ascontiguousarray(m.ppm.T) for m in motifs]
    # Reverse complements are internal; do not double N when converting p to E.
    result=tomtom(Qs=matrices,Ts=matrices,n_nearest=None,reverse_complement=True,n_jobs=n_jobs)
    require(len(result)==5,'Unexpected full-matrix Tomtom return structure')
    raw_p=np.asarray(result[0]);scores=np.asarray(result[1]);overlaps=np.asarray(result[3])
    if raw_output is not None:
        np.savez_compressed(raw_output,motif_ids=np.asarray([m.motif_id for m in motifs]),
                            pvalues=raw_p,alignment_scores=scores,alignment_overlaps=overlaps)
    p,e,similarity,audit=transform_pvalues(raw_p,len(motifs))
    negative=raw_p<0
    lengths=np.asarray([m.ppm.shape[0] for m in motifs])
    max_overlap=np.minimum(lengths[:,None],lengths[None,:])
    require(np.isfinite(scores).all() and np.isfinite(overlaps).all(),'Tomtom returned invalid alignment metadata')
    require(np.all(scores[negative]>0),'Negative Tomtom P with nonpositive alignment score; cannot treat as tail roundoff')
    require(np.all((overlaps[negative]>=1)&(overlaps[negative]<=max_overlap[negative])&
                   (overlaps[negative]==np.floor(overlaps[negative]))),'Negative Tomtom P with invalid overlap')
    zero_score_bad=int(((scores<=0)&(raw_p<1-1e-10)).sum())
    require(zero_score_bad==0,'Nonpositive alignment score with P below1: possible old memelite score-zero boundary issue')
    audit.update(negative_p_score_min=float(scores[negative].min()) if negative.any() else None,
                 negative_p_score_max=float(scores[negative].max()) if negative.any() else None,
                 negative_p_overlap_min=float(overlaps[negative].min()) if negative.any() else None,
                 negative_p_overlap_max=float(overlaps[negative].max()) if negative.any() else None,
                 nonpositive_score_with_p_below_one_count=zero_score_bad)
    return p,e,similarity,audit


def pca_and_tsne(matrix,perplexity,seed,max_components=50):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    matrix=np.asarray(matrix,dtype=float);n=matrix.shape[0]
    require(matrix.ndim==2 and n>=3 and np.isfinite(matrix).all(),'At least3 finite motif profiles required')
    require(math.isfinite(perplexity) and 0<perplexity<n,f'perplexity must be>0 and<N={n}; set it explicitly for small selections')
    require(np.any(np.ptp(matrix,axis=0)>0),'All motif similarity profiles are identical; no distinguishable embedding')
    n_components=min(max_components,n-1,matrix.shape[1])
    require(n_components>=2,'At least2 PCA dimensions required')
    model=PCA(n_components=n_components,svd_solver='full')
    pca=model.fit_transform(matrix)
    require(np.isfinite(pca).all() and np.any(np.ptp(pca,axis=0)>0),'Degenerate PCA result')
    tsne=TSNE(n_components=2,perplexity=perplexity,init='pca',learning_rate='auto',random_state=seed).fit_transform(pca)
    require(np.isfinite(tsne).all(),'Nonfinite t-SNE result')
    return pca,tsne,dict(n_components=n_components,explained_variance_ratio=model.explained_variance_ratio_.tolist())


def build_knn_edges(pca_coordinates,n_neighbors):
    from sklearn.metrics import pairwise_distances
    x=np.asarray(pca_coordinates,dtype=float);n=len(x)
    require(x.ndim==2 and n>=2 and np.isfinite(x).all(),'Invalid PCA coordinates for kNN')
    require(isinstance(n_neighbors,(int,np.integer)) and n_neighbors>=1,'n_neighbors must be a positive integer')
    k=min(n_neighbors,n-1)
    # Full distance matrix gives deterministic tie-breaking by input index.
    # Do not drop the first neighbour: identical profiles may put self elsewhere.
    distances=pairwise_distances(x,metric='euclidean',n_jobs=1)
    np.fill_diagonal(distances,np.inf)
    neighbour_ids=np.argsort(distances,axis=1,kind='stable')[:,:k]
    near=distances[np.arange(n)[:,None],neighbour_ids]
    positive=near[near>0]
    scale=float(np.median(positive)) if len(positive) else 1.
    scale=max(scale,np.finfo(float).eps)
    pairs={}
    for node in range(n):
        for neighbour,distance in zip(neighbour_ids[node],near[node]):
            neighbour=int(neighbour);require(neighbour!=node,'Unexpected self-neighbour')
            # Preserve Gaussian weighting; clip only numerical underflow.
            exponent=min((float(distance)/scale)**2,700.)
            weight=float(np.exp(-exponent));edge=tuple(sorted((node,neighbour)))
            pairs[edge]=max(pairs.get(edge,0.),weight)
    edges=np.asarray(sorted(pairs),dtype=int).reshape(-1,2)
    weights=np.asarray([pairs[tuple(edge)] for edge in edges],dtype=float)
    require(np.all(edges[:,0]!=edges[:,1]) and np.all(weights>0),'Invalid kNN graph')
    return edges,weights,dict(requested_n_neighbors=n_neighbors,effective_n_neighbors=k,
                              gaussian_scale_median_positive_knn_distance=scale,n_undirected_edges=len(edges),
                              tie_policy='stable distance order then input index; self excluded by index',
                              directed_neighbours_per_node=k)


def canonical_cluster_labels(membership):
    membership=np.asarray(membership,dtype=int)
    groups=sorted(set(membership.tolist()),key=lambda group:int(np.flatnonzero(membership==group)[0]))
    mapping={old:new+1 for new,old in enumerate(groups)}
    return np.asarray([mapping[int(value)] for value in membership],dtype=int)


def leiden_clusters(pca_coordinates,n_neighbors,resolution,seed):
    import igraph as ig
    import leidenalg
    require(math.isfinite(resolution) and resolution>0,'Resolution must be finite and positive')
    edges,weights,details=build_knn_edges(pca_coordinates,n_neighbors)
    graph=ig.Graph(n=len(pca_coordinates),edges=edges.tolist(),directed=False)
    graph.es['weight']=weights.tolist()
    partition=leidenalg.find_partition(graph,leidenalg.RBConfigurationVertexPartition,
                                      weights='weight',resolution_parameter=resolution,seed=seed,n_iterations=2)
    labels=canonical_cluster_labels(partition.membership)
    details.update(n_clusters=len(set(labels.tolist())),partition_type='RBConfigurationVertexPartition',
                   n_iterations=2,graph_space='PCA',undirected_edge_merge='union,max_weight')
    return labels,edges,weights,details


def sort_membership_rows(rows):
    return sorted(rows,key=lambda r:(int(r['cluster_id']),r['species'],r['organ'],natural_pattern_key(r['pattern_name'])))


def motif_rows(motifs):
    return [dict(matrix_row_0based=i,motif_id=m.motif_id,species=m.species,organ=m.organ,
                 pattern_name=m.pattern_name,h5_group='pos_patterns/'+m.pattern_name,
                 n_seqlets=m.n_seqlets,original_length=m.original_length,
                 trim_start_0based=m.trim_start,trim_end_0based_exclusive=m.trim_end,
                 trimmed_length=len(m.ppm),trim_status=m.trim_status,
                 n_positions_IC_above_threshold=m.n_informative_positions,
                 trimmed_IC_mean_bits=m.ic_mean,trimmed_IC_total_bits=m.ic_total,
                 consensus=''.join('ACGT'[j] for j in m.ppm.argmax(1)),
                 full_CWM_net_sum=m.cwm_net_sum,source_hdf5=m.source_h5,
                 source_attempt=m.source_attempt,source_hdf5_sha256=m.source_sha256)
            for i,m in enumerate(motifs)]


def plot_tsne(rows,path,annotate=False,preview=False):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    coords=np.array([[r['tsne_1'],r['tsne_2']] for r in rows])
    with PdfPages(path) as pdf:
        for field,title in [('cluster_id','Leiden cluster'),('species','Species'),('organ','Organ')]:
            categories=sorted({r[field] for r in rows});count=len(categories)
            cmap=plt.get_cmap('tab20' if count>10 else 'tab10')
            palette=([cmap(i) for i in range(count)] if count<=20 else
                     [colorsys.hls_to_rgb((i*.618033988749895)%1.,.43+.075*(i%3),.62) for i in range(count)])
            require(len({tuple(c) for c in palette})==count,'Plot palette contains duplicate colors')
            colors=dict(zip(categories,palette))
            fig,ax=plt.subplots(figsize=(5,4))
            try:
                fig.subplots_adjust(left=.16,right=.96 if field=='cluster_id' else .65,bottom=.16,top=.82)
                ax.scatter(coords[:,0],coords[:,1],c=[colors[r[field]] for r in rows],s=18,alpha=.82,
                           edgecolors='white',linewidths=.2)
                if field=='cluster_id':
                    for category in categories:
                        selected=coords[[r[field]==category for r in rows]];center=np.median(selected,axis=0)
                        ax.text(*center,str(category),ha='center',va='center',fontsize=10,
                                bbox=dict(boxstyle='round,pad=.15',facecolor='white',edgecolor='#555555',alpha=.85))
                else:
                    handles=[Line2D([0],[0],marker='o',linestyle='none',color=colors[c],label=str(c),markersize=5) for c in categories]
                    fig.legend(handles=handles,title=title,loc='upper left',bbox_to_anchor=(.66,.83),
                               fontsize=10,title_fontsize=11,handletextpad=.35)
                if annotate:
                    for r in rows:ax.annotate(r['motif_id'],(r['tsne_1'],r['tsne_2']),fontsize=4,alpha=.65)
                ax.set_xlabel('t-SNE1',fontsize=14);ax.set_ylabel('t-SNE2',fontsize=14)
                ax.tick_params(labelsize=12)
                fig.suptitle('Joint positive TF-MoDISco motifs\n'+title,fontsize=12,y=.98)
                pdf.savefig(fig)
                if preview:fig.savefig(path.with_name('preview_'+field+'.png'),dpi=160)
            finally:plt.close(fig)


def dependencies():
    versions={};missing=[]
    for module,dist in [('numpy','numpy'),('h5py','h5py'),('scipy','scipy'),('sklearn','scikit-learn'),
                        ('matplotlib','matplotlib'),('igraph','igraph'),('leidenalg','leidenalg'),('memelite','memelite')]:
        try:
            importlib.import_module(module)
            versions[dist]=package_metadata.version(dist)
        except Exception as exc:missing.append(f'{module}: {exc}')
    require(not missing,'Missing/unusable dependencies; no installation attempted:\n'+'\n'.join(missing))
    return versions


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-root',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--output-dir',type=Path,help='A NEW directory; existing directories are never overwritten')
    p.add_argument('--species',nargs='+',choices=SPECIES)
    p.add_argument('--organs',nargs='+')
    p.add_argument('--background',nargs=4,type=float,default=(.27,.23,.23,.27),metavar=('A','C','G','T'))
    p.add_argument('--trim-ic-threshold',type=float,default=.2)
    p.add_argument('--trim-flank',type=int,default=0)
    p.add_argument('--embedding-input',choices=('evalue','similarity'),default='evalue',help='Default rawE profiles follow supplied script; similarity is optional clipped -log10(E)')
    p.add_argument('--perplexity',type=float,default=15.)
    p.add_argument('--n-neighbors',type=int,default=15)
    p.add_argument('--resolution',type=float,default=1.)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--skip-hash-verification',action='store_true',help='Explicitly skip HDF5/audit file content hashes; metadata checks remain')
    p.add_argument('--check-only',action='store_true',help='Validate inputs and dependencies, without creating outputs or running Tomtom')
    p.add_argument('--annotate-patterns',action='store_true')
    p.add_argument('--preview',action='store_true',help='Write small PNGs for figure QA in addition to the three-page PDF')
    return p.parse_args(argv)


def main(argv=None):
    args=parse_args(argv);background=np.asarray(args.background,dtype=float)
    information_content(np.array([[.25]*4]),background)
    require(math.isfinite(args.trim_ic_threshold) and args.trim_ic_threshold>=0,'Invalid IC threshold')
    require(args.trim_flank>=0 and args.n_neighbors>0 and args.threads>0,'Flank>=0 and neighbours/threads>0 required')
    require(math.isfinite(args.resolution) and args.resolution>0,'Invalid Leiden resolution')
    versions=dependencies();sources,audits=discover_sources(args.input_root,args.species,args.organs,args.allow_partial,not args.skip_hash_verification)
    motifs=[]
    for source in sources:
        found=read_positive_patterns(source,background,args.trim_ic_threshold,args.trim_flank);motifs.extend(found)
        print(f'INPUT {source.species}/{source.organ}: {len(found)} positive parent patterns',flush=True)
    require(len({m.motif_id for m in motifs})==len(motifs),'Duplicate motif IDs')
    require(len(motifs)>=3 and math.isfinite(args.perplexity) and 0<args.perplexity<len(motifs),
            f'Need>=3motifs and0<perplexity<N; N={len(motifs)}')
    summary=dict(status='check_only_passed',sources=len(sources),motifs=len(motifs),software=versions,
                 low_information_full_length_kept=sum(m.n_informative_positions==0 for m in motifs),
                 nonpositive_net_CWM_in_pos_metacluster=sum(m.cwm_net_sum<=0 for m in motifs),
                 input_hashes_verified=not args.skip_hash_verification)
    if args.check_only:print(json.dumps(summary,indent=2),flush=True);return
    out=args.output_dir or BASE/'do_cluster/results'/('joint_positive_motifs_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out=out.resolve();require(not out.exists(),'Output already exists; choose a new run directory')
    # Never create outputs inside the read-only source batch.
    require(not out.is_relative_to(args.input_root.resolve()),'Output must be outside the input batch')
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(version=VERSION,status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                  script_sha256=sha256(__file__),parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  software=versions,input_summary=summary,source_manifest_sha256=sha256(args.input_root/'batch_manifest.json'),
                  method=dict(IC='sum P*log2((P+1e-6)/(background+1e-6)),strict>threshold,first-to-last informative span',
                              no_informative_positions='keep original full length,flag',
                              similarity='memelite.tomtom,all-vs-all,reverse_complement=True,n_nearest=None',
                              embedding='raw E=N*p profiles' if args.embedding_input=='evalue' else 'clipped -log10(E),self diagonal maximum',
                              graph='PCA Euclidean kNN union,Gaussian weights,Leiden RBConfiguration,resolution parameter',
                              tsne='visualization only,does not determine Leiden membership'),
                  interpretation='Sequence-pattern communities,not TF identification/validated activation or phylogenetic conservation')
    write_json(out/'run_manifest.json',manifest)
    try:
        rows=motif_rows(motifs);write_tsv(out/'motif_index.tsv',rows)
        write_tsv(out/'input_sources.tsv',audits,fieldnames=['species','organ','status','included','attempt','h5_path','h5_sha256','hash_verified','positive_patterns'])
        with (out/'motifs.meme').open('w',encoding='utf-8') as f:
            f.write('MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\n'+' '.join(f'{b} {v}' for b,v in zip('ACGT',background))+'\n\n')
            for m in motifs:
                f.write(f'MOTIF {m.motif_id}\nletter-probability matrix: alength= 4 w= {len(m.ppm)} nsites= {m.n_seqlets} E= 0\n')
                for row in m.ppm:f.write(' '.join(f'{v:.12g}' for v in row)+'\n')
                f.write('\n')
        print(f'TOMTOM all-vs-all N={len(motifs)},threads={args.threads}',flush=True)
        p,e,similarity,tomtom_audit=calculate_tomtom_matrices(motifs,args.threads,out/'tomtom_raw_pvalues.npz')
        np.savez_compressed(out/'tomtom_checkpoint.npz',motif_ids=np.asarray([m.motif_id for m in motifs]),tomtom_pvalues=p,tomtom_evalues=e,tomtom_similarity_neglog10_evalue=similarity)
        print('PCA and t-SNE',flush=True)
        pca,tsne,pca_details=pca_and_tsne(e if args.embedding_input=='evalue' else similarity,args.perplexity,args.seed)
        labels,edges,weights,graph_details=leiden_clusters(pca,args.n_neighbors,args.resolution,args.seed)
        for i,r in enumerate(rows):r.update(cluster_id=int(labels[i]),tsne_1=float(tsne[i,0]),tsne_2=float(tsne[i,1]))
        write_tsv(out/'cluster_members.tsv',sort_membership_rows(rows))
        write_tsv(out/'motif_coordinates.tsv',rows)
        summaries=[];lists=[]
        for cluster in sorted(set(labels.tolist())):
            members=sort_membership_rows([r for r in rows if r['cluster_id']==cluster])
            representative=max(members,key=lambda r:(r['n_seqlets'],r['trimmed_IC_total_bits'],-r['matrix_row_0based']))
            summaries.append(dict(cluster_id=cluster,n_motifs=len(members),n_species=len({r['species'] for r in members}),
                                  n_organs=len({r['organ'] for r in members}),total_seqlet_memberships=sum(r['n_seqlets'] for r in members),
                                  representative_motif=representative['motif_id'],representative_policy='max n_seqlets,tie totalIC then source order',
                                  species=';'.join(sorted({r['species'] for r in members})),organs=';'.join(sorted({r['organ'] for r in members}))))
            lists.append(dict(cluster_id=cluster,n_motifs=len(members),motif_ids=';'.join(r['motif_id'] for r in members)))
        write_tsv(out/'cluster_summary.tsv',summaries);write_tsv(out/'cluster_members_lists.tsv',lists)
        write_tsv(out/'graph_edges.tsv',[dict(node1_row_0based=int(a),node2_row_0based=int(b),weight=float(w)) for (a,b),w in zip(edges,weights)])
        np.savez_compressed(out/'matrices.npz',motif_ids=np.asarray([m.motif_id for m in motifs]),tomtom_pvalues=p,tomtom_evalues=e,
                            tomtom_similarity_neglog10_evalue=similarity,embedding_input=np.asarray(args.embedding_input),
                            pca_coordinates=pca,tsne_coordinates=tsne,leiden_clusters=labels,graph_edges=edges,graph_weights=weights)
        plot_tsne(rows,out/'motif_tsne.pdf',args.annotate_patterns,args.preview)
        require(sum(r['n_motifs'] for r in summaries)==len(motifs),'Cluster membership totals do not reconcile')
        require(sorted(r['matrix_row_0based'] for r in rows)==list(range(len(motifs))),'Matrix row identity mismatch')
        manifest.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),n_clusters=len(summaries),
                        tomtom_audit=tomtom_audit,pca=pca_details,graph=graph_details,
                        output_sha256={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='run_manifest.json'})
        write_json(out/'run_manifest.json',manifest)
        print(json.dumps(dict(status='complete',output_directory=str(out),motifs=len(motifs),clusters=len(summaries)),indent=2),flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(out/'run_manifest.json',manifest);raise


if __name__=='__main__':main()
