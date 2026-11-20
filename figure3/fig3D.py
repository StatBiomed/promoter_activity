#!/usr/bin/env python3
"""Small cluster-profile heatmaps for completed redo8 Mfuzz results.

One self-contained Python script; no R calls and no Mfuzz refitting. Each target
is one figure: chronological observed stages are rows, the ten original Mfuzz
cluster IDs are columns. Cluster columns, NOT time rows, are reordered by
Euclidean average-linkage hierarchical clustering (optimal_ordering=False).
These are ten cluster summaries, not ten individual-promoter heatmap panels.

Cell definition follows the preceding heatmap workflow: use promoters whose
membership in a cluster is STRICTLY above the upstream core threshold (currently
0.8). Standardise each promoter's stage-mean curve with its temporal sample SD
(ddof=1), then average the z-scores within the core. Read the existing exported
core profiles unchanged and recompute them independently to verify correctness.
This post-fit display standardisation never changes the original Mfuzz fit.
Constant curves have undefined z-scores and are excluded as in the upstream
export. Empty core profiles cause a clear error; they are never replaced by zero
or silently dropped. All ten clusters must be shown.

Output is deliberately minimal: exactly two files per target, a vector RGB PDF
and the exact plotted matrix in its displayed column order (TSV). No extra PNG,
SVG, barplot, linkage table, README, log, or manifest is written. Upstream files
are read-only and checked against their completed run's SHA256 checksums.

The supplied screenshot is 281x310 pixels without DPI. At nominal 100 dpi its
chosen canvas is 2.81x3.10 inches (PDF 202.32x223.20 pt), not a claim about its
unknown original physical size. Stage/cluster ticks are 5.5 pt, title 7 pt and
colourbar ticks 5 pt. No panel letter is added. All targets share symmetric,
data-derived coolwarm limits, rounded upward to 0.5, with no value clipping.

Run on the server:
  /mnt/ruiyanhou/nfs_share2/conda_env/envs/PY310/bin/python -B plot_mfuzz_cluster_heatmaps.py
  ... plot_mfuzz_cluster_heatmaps.py --check-only  # no output
  ... plot_mfuzz_cluster_heatmaps.py --self-test   # in-memory numerical tests
Outputs: output/<Species_Organ>/<Species_Organ>_C10_heatmap.pdf and
         output/<Species_Organ>/<Species_Organ>_C10_heatmap_matrix.tsv
TSV values are mean temporal z-scores, NOT PSI or absolute activity. Existing
nonempty destinations are never overwritten. No package installation occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

ROOT = Path(__file__).resolve().parent
MOUSE_ORGANS = ('Brain','Cerebellum','Heart','Kidney','Liver','Ovary','Testis')
TARGETS = tuple('Mouse_' + organ for organ in MOUSE_ORGANS) + ('Human_Liver','Rat_Liver')
STYLE = {'figsize':[2.81,3.10], 'tick_fontsize':5.5, 'title_fontsize':7.,
         'colorbar_fontsize':5., 'stat_label_fontsize':5.5,
         'heatmap_axes':[32/281,23/310,242/281,217/310],
         'dendrogram_axes':[32/281,241/310,242/281,30/310],
         'colorbar_axes':[177/281,282/310,90/281,11/310]}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read_indexed(path, first, sep=','):
    with path.open(newline='') as stream:
        header = next(csv.reader(stream,delimiter=sep))
    require(header[0]==first and len(set(header))==len(header),f'Unexpected/duplicate header: {path}')
    frame = pd.read_csv(path,sep=sep,dtype={first:str,'geneId':str},keep_default_na=False)
    require(frame[first].ne('').all() and frame[first].is_unique,f'Invalid/duplicate IDs: {path}')
    return frame.set_index(first)


def boolean(values, name):
    lower = values.astype(str).str.lower()
    require(lower.isin(['true','false']).all(),f'Invalid boolean: {name}')
    return lower.eq('true')


def verify_core_profiles(means, membership, assignments, profiles, summary, threshold, k):
    require(means.index.is_unique and means.index.equals(membership.index)
            and means.index.equals(assignments.index),'Promoter IDs/order differ')
    require(membership.columns.tolist()==[f'Cluster_{i}' for i in range(1,k+1)],'Membership cluster columns differ')
    require(profiles.index.tolist()==[f'Cluster_{i}' for i in range(1,k+1)],'Core profile cluster IDs/order differ')
    require(profiles.columns.tolist()==means.columns.tolist(),'Core profile stage columns differ')
    require(summary.index.tolist()==list(range(1,k+1)),'Cluster summary IDs/order differ')
    x = means.to_numpy(dtype=float)
    u = membership.to_numpy(dtype=float)
    require(np.isfinite(x).all() and (x>=0).all(),'Non-finite/negative input activity')
    require(u.shape==(len(means),k) and np.isfinite(u).all() and (u>=0).all() and (u<=1).all(),'Invalid memberships')
    np.testing.assert_allclose(u.sum(axis=1),1,atol=1e-8)
    np.testing.assert_array_equal(u.argmax(axis=1)+1,assignments.cluster)
    np.testing.assert_allclose(u.max(axis=1),assignments.membership,atol=1e-12)
    np.testing.assert_array_equal(u.max(axis=1)>threshold,boolean(assignments.is_core,'is_core'))
    sd = x.std(axis=1,ddof=1)
    valid = np.isfinite(sd) & (sd>0)
    z = np.full_like(x,np.nan)
    z[valid] = (x[valid]-x[valid].mean(axis=1,keepdims=True))/sd[valid,None]
    for cluster in range(1,k+1):
        core = u[:,cluster-1]>threshold
        usable = core & valid
        row = summary.loc[cluster]
        require(int(row.assigned_n)==int(assignments.cluster.eq(cluster).sum()),'Assigned promoter count differs')
        require(int(row.core_n)==int(core.sum()) and int(row.core_profile_n)==int(usable.sum()),'Core promoter count differs')
        require(int(row.core_constant_excluded)==int((core & ~valid).sum()),'Constant-profile count differs')
        require(usable.any(),f'Cluster {cluster} has no nonconstant core promoters; refusing a fabricated or omitted column')
        actual = pd.to_numeric(profiles.loc[f'Cluster_{cluster}'],errors='raise').to_numpy()
        require(np.isfinite(actual).all(),f'Cluster {cluster}: invalid exported core profile')
        np.testing.assert_allclose(actual,z[usable].mean(axis=0),rtol=1e-10,atol=1e-10,
                                   err_msg=f'Cluster {cluster}: core mean z-score differs')
    matrix = profiles.astype(float).T
    matrix.columns = list(range(1,k+1))
    matrix.index.name = 'stage'
    return matrix


def column_hierarchy(matrix):
    require(matrix.shape[1]>=2 and np.isfinite(matrix.to_numpy()).all(),'Need at least two finite cluster profiles')
    tree = linkage(matrix.T.to_numpy(),method='average',metric='euclidean',optimal_ordering=False)
    order = matrix.columns[leaves_list(tree)].tolist()
    require(sorted(order)==sorted(matrix.columns.tolist()),'Cluster lost/duplicated in hierarchy')
    return tree,order


def load_target(target, args, upstream, hashes):
    organ = target.split('_',1)[1]
    out,inp = args.mfuzz_output/target,args.mfuzz_input/target
    paths = {
        'profiles':out/f'{organ}_core_profiles_zscore.csv',
        'summary':out/f'{organ}_cluster_summary.tsv',
        'stages':out/'stage_index_to_time.tsv',
        'membership':out/f'{organ}_membership.csv',
        'assignments':out/f'{organ}_assignments.tsv',
        'annotations':out/f'{organ}_promoter_cluster_annotations.tsv',
        'means':inp/f'{organ}_stage_mean_activity.csv',
    }
    known = upstream.get('output_sha256',{})
    base = args.mfuzz_output.parent.resolve()
    for path in paths.values():
        require(path.is_file(),f'Missing upstream result: {path}')
        resolved = path.resolve()
        require(resolved.is_relative_to(base),'Inputs must belong to the same upstream run')
        relative = resolved.relative_to(base).as_posix()
        digest = sha256(path)
        require(known.get(relative)==digest,f'Upstream file changed/checksum missing: {relative}')
        hashes[str(resolved)] = digest
    means = read_indexed(paths['means'],'promoterId').apply(pd.to_numeric,errors='raise')
    membership = read_indexed(paths['membership'],'promoterId').apply(pd.to_numeric,errors='raise')
    assignments = read_indexed(paths['assignments'],'promoterId','\t')
    profiles = read_indexed(paths['profiles'],'cluster')
    summary = read_indexed(paths['summary'],'cluster','\t')
    summary.index = pd.to_numeric(summary.index,errors='raise').astype(int)
    stages = pd.read_csv(paths['stages'],sep='\t',dtype={'stage':str})
    require(stages.stage_index.tolist()==list(range(1,len(stages)+1)),'Unexpected stage order/index')
    require(stages.stage.is_unique and stages.Time.is_unique and stages.Time.is_monotonic_increasing,
            'Stage labels/times are duplicated or not chronological')
    require(np.isfinite(stages.Time).all(),'Invalid stage times')
    require(means.columns.tolist()==stages.stage.tolist(),'Stage names/order differ from chronological key')
    annotations = read_indexed(paths['annotations'],'promoterId','\t')
    require(annotations.index.equals(means.index),'Promoter annotation order differs')
    require(annotations.retained_promoters_in_gene.ge(2).all()
            and not boolean(annotations.internalPromoter,'internalPromoter').any(),'Non-MP/internal promoter in upstream fit')
    require(pd.to_numeric(annotations.padj_BH,errors='raise').lt(.01).all(),'Non-dynamic promoter (FDR>=0.01) in upstream fit')
    require(annotations.cluster.equals(assignments.cluster),'Annotated cluster assignments differ')
    matrix = verify_core_profiles(means,membership,assignments,profiles,summary,
                                  float(upstream['core_threshold']),10)
    tree,order = column_hierarchy(matrix)
    record = {'target':target,'stages':len(stages),'promoters':len(means),
              'core_profile_promoters':int(summary.core_profile_n.sum()),
              'cluster_order':order,'core_promoter_n':summary.core_profile_n.to_dict()}
    return matrix,tree,order,record


def shared_limit(matrices):
    peak = max(float(np.abs(matrix.to_numpy()).max()) for matrix in matrices)
    require(np.isfinite(peak),'Invalid heatmap colour range')
    return max(.5,float(np.ceil(peak*2)/2))


def plot_heatmap(matrix, tree, order, target, limit, output_pdf, metadata):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':5.5,'pdf.fonttype':42,
                         'pdf.compression':0,'axes.linewidth':.35,'axes.grid':False})
    display = matrix.loc[:,order]
    require(np.abs(display.to_numpy()).max()<=limit,'Colour range would clip data')
    fig = plt.figure(figsize=STYLE['figsize'],dpi=100)
    ax = fig.add_axes(STYLE['heatmap_axes'])
    den = fig.add_axes(STYLE['dendrogram_axes'])
    cb = fig.add_axes(STYLE['colorbar_axes'])
    dend = dendrogram(tree,ax=den,no_labels=True,color_threshold=0,
                      above_threshold_color='#999999',link_color_func=lambda _: '#999999')
    require(display.columns.tolist()==matrix.columns[np.asarray(dend['leaves'])].tolist(),'Tree/heatmap order mismatch')
    for collection in den.collections:
        collection.set_linewidth(.35)
    den.set_axis_off()
    n,k = display.shape
    mesh = ax.pcolormesh(np.arange(k+1),np.arange(n+1),display.to_numpy(),
                         cmap='coolwarm',norm=Normalize(-limit,limit),shading='flat',
                         edgecolors='face',linewidth=0,antialiased=False,rasterized=False)
    ax.set_xlim(0,k);ax.set_ylim(n,0)
    ax.set_xticks(np.arange(k)+.5,labels=[str(i) for i in order])
    ax.set_yticks(np.arange(n)+.5,labels=display.index.tolist())
    ax.tick_params(axis='both',length=0,labelsize=STYLE['tick_fontsize'],pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(mesh,cax=cb,orientation='horizontal',ticks=[-limit,0,limit])
    # Matplotlib colourbars otherwise rasterise long colour ramps automatically.
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
    bar.ax.tick_params(labelsize=STYLE['colorbar_fontsize'],length=1.2,width=.35,pad=1)
    bar.outline.set_visible(False)
    species,organ = target.split('_',1)
    fig.text(222/281,1-3/310,f'{organ} in {species}',ha='center',va='top',fontsize=STYLE['title_fontsize'])
    fig.text(32/281,1-21/310,'Mean z-score',ha='left',va='center',fontsize=STYLE['stat_label_fontsize'])
    # No bbox_inches='tight': preserve the selected reference canvas exactly.
    fig.savefig(output_pdf,metadata={'Title':f'{organ} in {species}',
        'Subject':'Chronological stages x Mfuzz C10; mean temporal z-scores of nonconstant core promoters; not PSI or absolute activity.',
        'Keywords':json.dumps(metadata,separators=(',',':'))})
    plt.close(fig)


def self_test():
    class Tests(unittest.TestCase):
        def fixture(self):
            ids = pd.Index(['p1','p2','p3','p4'],name='promoterId')
            means = pd.DataFrame([[1.,2.,3.],[2.,4.,6.],[3.,2.,1.],[4.,4.,4.]],index=ids,columns=['a','b','c'])
            u = pd.DataFrame([[.9,.1],[.8,.2],[.05,.95],[.02,.98]],index=ids,columns=['Cluster_1','Cluster_2'])
            a = pd.DataFrame({'cluster':[1,1,2,2],'membership':[.9,.8,.95,.98],
                              'is_core':[True,False,True,True]},index=ids)
            p = pd.DataFrame([[-1.,0.,1.],[1.,0.,-1.]],index=u.columns,columns=means.columns)
            summary = pd.DataFrame({'assigned_n':[2,2],'core_n':[1,2],'core_profile_n':[1,1],
                                     'core_constant_excluded':[0,1]},index=[1,2])
            return means,u,a,p,summary

        def test_strict_core_threshold_sample_SD_and_constants(self):
            means,u,a,p,s = self.fixture()
            m = verify_core_profiles(means,u,a,p,s,.8,2)
            np.testing.assert_allclose(m.to_numpy(),[[-1.,1.],[0.,0.],[1.,-1.]])

        def test_empty_core_is_not_filled_or_omitted(self):
            means,u,a,p,s = self.fixture()
            a['is_core']=False
            s[['core_n','core_profile_n','core_constant_excluded']]=0
            with self.assertRaisesRegex(ValueError,'no nonconstant core'):
                verify_core_profiles(means,u,a,p,s,.99,2)

        def test_tampered_profile_rejected(self):
            means,u,a,p,s = self.fixture();p.iloc[0,0]=.2
            with self.assertRaises(AssertionError):
                verify_core_profiles(means,u,a,p,s,.8,2)

        def test_hierarchy_preserves_ids_and_time_order(self):
            matrix = pd.DataFrame({1:[-1,0,1],2:[1,0,-1],3:[-.9,0,.9]},index=['a','b','c'])
            tree,order = column_hierarchy(matrix)
            self.assertEqual(sorted(order),[1,2,3])
            self.assertEqual(matrix.loc[:,order].index.tolist(),['a','b','c'])
            self.assertEqual(tree.shape,(2,4))

        def test_shared_scale_includes_all_values(self):
            self.assertEqual(shared_limit([pd.DataFrame([[2.21,-1.]])]),2.5)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    require(result.wasSuccessful(),'Heatmap self-tests failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mfuzz-output',type=Path,default=ROOT.parent/'8.2.run_mfuzz/output')
    parser.add_argument('--mfuzz-input',type=Path,default=ROOT.parent/'8.2.run_mfuzz/input')
    parser.add_argument('--targets',nargs='+',choices=TARGETS,default=list(TARGETS))
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args(argv)
    require(len(set(args.targets))==len(args.targets),'Duplicate requested targets')
    if args.self_test:
        self_test();print('SELF_TEST_PASSED: no files written');return 0
    path = args.mfuzz_output/'run_manifest.json'
    require(path.is_file(),'Missing upstream run_manifest.json')
    upstream = json.loads(path.read_text())
    require(upstream.get('status')=='complete' and upstream.get('clusters')==10,'Require a complete C10 Mfuzz run')
    require(.5<float(upstream['core_threshold'])<1,'Invalid upstream core threshold')
    hashes = {str(path.resolve()):sha256(path)}
    prepared = {}
    for target in args.targets:
        prepared[target] = load_target(target,args,upstream,hashes)
        print('PREFLIGHT',json.dumps(prepared[target][3]),flush=True)
    limit = shared_limit(item[0] for item in prepared.values())
    print(f'SHARED_COLOUR_LIMITS {-limit:g} to {limit:g}',flush=True)
    if args.check_only:
        for file,expected in hashes.items():
            require(sha256(Path(file))==expected,'Upstream changed during checks: '+file)
        print('CHECK_ONLY_PASSED: all core profiles/cluster counts/stages/checksums verified; no files written')
        return 0
    destination = args.output_dir.resolve()
    require(destination.is_relative_to(ROOT) and destination!=ROOT,'Output must be a subdirectory of this script directory')
    require(not destination.exists() or (destination.is_dir() and not any(destination.iterdir())),
            'Output directory is nonempty; nothing overwritten. Choose a new --output-dir.')
    destination.mkdir(parents=True,exist_ok=True)
    for target,(matrix,tree,order,record) in prepared.items():
        folder = destination/target;folder.mkdir()
        stem = folder/f'{target}_C10_heatmap'
        matrix.loc[:,order].to_csv(stem.with_name(stem.name+'_matrix.tsv'),sep='\t',float_format='%.17g')
        audit = {**record,'core_threshold_strict_greater_than':upstream['core_threshold'],
                 'row_zscore_SD_ddof':1,'linkage':'average','distance':'euclidean',
                 'optimal_ordering':False,'row_clustering':False,'colour_limits':[-limit,limit],
                 'style':STYLE,'script_sha256':sha256(Path(__file__)),
                 'source_sha256':{Path(p).relative_to(args.mfuzz_output.parent.resolve()).as_posix():d for p,d in hashes.items()}}
        plot_heatmap(matrix,tree,order,target,limit,stem.with_suffix('.pdf'),audit)
        print('COMPLETE',target,'PDF + plotted matrix TSV; cluster order',order,flush=True)
    for file,expected in hashes.items():
        require(sha256(Path(file))==expected,'Upstream changed during plotting: '+file)
    print(f'COMPLETE: {len(prepared)} targets, exactly {2*len(prepared)} output files; no Mfuzz refit',flush=True)
    return 0


if __name__=='__main__':
    sys.exit(main())
