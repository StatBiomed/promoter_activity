#!/usr/bin/env python3
"""Cluster redo8 dynamic MP promoters with genuine R Mfuzz and small Python plots.

Only this entry-point script is required. The numerical R backend is embedded;
it calls installed Mfuzz, not a Python approximation, and opens no R graphics.
Default targets: Mouse's seven organs plus Human Liver and Rat Liver, separately.

Read padj_BH from 8.1.run_magSigPro/output/<target> and select STRICTLY <0.01.
Use its exported sample-level activity and matching input metadata. Average only
replicates within each actual stage, then fit on the unchanged activity scale:
no PSI conversion, log transform, z-score, pooling organs, or pooling species.
MP membership is inherited and rechecked against the complete upstream MP cohort,
NOT redefined by counting the subset of dynamic promoters.

Defaults match the reference numerical procedure: C=10, seed=20261015, iter.max
1000, m reestimated by Mfuzz::mestimate for each target, core membership >0.8.
Blue curves: all hard-assigned promoters, 0.30 pt, alpha=0.38, drawn first.
Red band: promoter 25th-75th percentiles, alpha=0.82, drawn second.
Red line: arithmetic mean, not fuzzy fitted center. Band is not a confidence
interval. Original activity, not PSI. Empty hard-assignment clusters are labelled.
Figure canvas 1.41x1.32 inches, PNG 141x132 at 100 dpi; PDF/SVG are vector RGB.
Y labels 0,4,8; x values are ordinal chronological stage indices. For >14 stages,
only five x labels are printed to avoid collisions; minor ticks show all stages.

Usage on the server:
  /mnt/ruiyanhou/nfs_share2/conda_env/envs/PY310/bin/python -B run_dynamic_mp_mfuzz.py
  ... run_dynamic_mp_mfuzz.py --check-only   # all source/runtime checks, no output
  ... run_dynamic_mp_mfuzz.py --self-test    # synthetic numerical tests in temporary storage

No installation, heatmap or GO analysis. Upstream and old-reference files are
read-only. Existing nonempty input/output destinations are never overwritten.
Use --rscript/--r-library if running on a node with different R installation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MOUSE_ORGANS = ('Brain','Cerebellum','Heart','Kidney','Liver','Ovary','Testis')
TARGETS = tuple('Mouse_' + organ for organ in MOUSE_ORGANS) + ('Human_Liver','Rat_Liver')
PLOT_STYLE = {'blue_color':'#469BCC', 'blue_linewidth_pt':.30, 'blue_alpha':.38,
              'blue_zorder':1, 'red_band_color':'#F34D70', 'red_band_alpha':.82,
              'red_band_zorder':2, 'red_mean_color':'#D91D4E',
              'red_mean_linewidth_pt':.8, 'red_mean_zorder':3}

R_BACKEND = r'''
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=8L) stop('Expected INPUT OUTPUT C SEED CORE ITER_MAX R_LIBRARY ORGAN')
input <- args[1]; output <- args[2]; k <- as.integer(args[3]); seed <- as.integer(args[4])
threshold <- as.numeric(args[5]); iter_max <- as.integer(args[6]); organ <- args[8]
if (nzchar(args[7]) && dir.exists(args[7])) .libPaths(c(args[7],.libPaths()))
suppressPackageStartupMessages(library(Mfuzz))
raw <- read.csv(input,check.names=FALSE,colClasses='character')
ids <- raw[[1]]
stopifnot(!anyDuplicated(ids),all(nzchar(ids)))
x <- as.matrix(data.frame(lapply(raw[-1],as.numeric),check.names=FALSE))
rownames(x) <- ids; colnames(x) <- names(raw)[-1]
stopifnot(all(is.finite(x)),all(x>=0),nrow(x)>=k,ncol(x)>=2,k>=2)
eset <- Biobase::ExpressionSet(assayData=x)
m <- Mfuzz::mestimate(eset)
stopifnot(is.finite(m),m>1)
RNGkind('Mersenne-Twister','Inversion','Rejection'); set.seed(seed)
fit_warnings <- character()
cl <- withCallingHandlers(Mfuzz::mfuzz(eset,centers=k,m=m,iter.max=iter_max),
  warning=function(w) {fit_warnings <<- c(fit_warnings,conditionMessage(w));invokeRestart('muffleWarning')})
if (cl$iter>=iter_max) stop('Mfuzz reached iteration limit')
membership <- cl$membership[ids,,drop=FALSE]
colnames(membership) <- paste0('Cluster_',seq_len(k))
stopifnot(max(abs(rowSums(membership)-1))<1e-8,all(membership>=0),all(membership<=1))
assignment <- max.col(membership,ties.method='first')
stopifnot(all(as.integer(cl$cluster[ids])==assignment))
name <- function(suffix) file.path(output,paste0(organ,suffix))
write.matrix <- function(mat,path,first='promoterId') {
  d <- data.frame(rownames(mat),mat,check.names=FALSE);names(d)[1] <- first
  write.csv(d,path,row.names=FALSE,na='NA')
}
write.matrix(membership,name('_membership.csv'))
centers <- cl$centers;rownames(centers) <- paste0('Cluster_',seq_len(k));colnames(centers) <- colnames(x)
write.matrix(centers,name('_cluster_centers.csv'),'cluster')
best <- membership[cbind(seq_along(ids),assignment)]
assigned <- data.frame(promoterId=ids,cluster=assignment,membership=best,is_core=best>threshold)
write.table(assigned,name('_assignments.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
sd_rows <- apply(x,1,sd); valid_z <- is.finite(sd_rows) & sd_rows>0
z <- matrix(NA_real_,nrow(x),ncol(x),dimnames=dimnames(x))
z[valid_z,] <- sweep(sweep(x[valid_z,,drop=FALSE],1,rowMeans(x[valid_z,,drop=FALSE]),'-'),1,sd_rows[valid_z],'/')
core_profiles <- matrix(NA_real_,k,ncol(x),dimnames=dimnames(centers))
dir.create(file.path(output,'clusters'),showWarnings=FALSE)
summary_rows <- list()
for (j in seq_len(k)) {
  selected <- which(assignment==j); core <- which(membership[,j]>threshold); good <- core[valid_z[core]]
  if (length(good)) core_profiles[j,] <- colMeans(z[good,,drop=FALSE])
  write.csv(data.frame(promoterId=ids[selected],Membership=membership[selected,j]),
    file.path(output,'clusters',sprintf('%s_mfuzz_%02d.csv',organ,j)),row.names=FALSE)
  write.csv(data.frame(promoterId=ids[core],Membership=membership[core,j]),
    file.path(output,'clusters',sprintf('%s_core_%02d.csv',organ,j)),row.names=FALSE)
  summary_rows[[j]] <- data.frame(cluster=j,assigned_n=length(selected),core_n=length(core),
    core_profile_n=length(good),core_constant_excluded=length(core)-length(good))
}
write.table(do.call(rbind,summary_rows),name('_cluster_summary.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
# Compatibility summary only: z-scores never enter the fitted ExpressionSet.
write.matrix(core_profiles,name('_core_profiles_zscore.csv'),'cluster')
saveRDS(list(eset=eset,cl=cl,m=m,seed=seed,core_threshold=threshold,
  standardised_before_clustering=FALSE,fit_warnings=fit_warnings),name('_mfuzz_fit.rds'))
diag <- data.frame(key=c('Mfuzz_version','e1071_version','R_version','m','clusters','seed',
  'core_threshold','iterations','iteration_limit','withinerror','fit_warning_count'),
  value=c(as.character(packageVersion('Mfuzz')),as.character(packageVersion('e1071')),R.version.string,
    format(m,digits=17),k,seed,threshold,cl$iter,iter_max,format(cl$withinerror,digits=17),length(fit_warnings)))
write.table(diag,file.path(output,'fit_diagnostics.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
writeLines(fit_warnings,file.path(output,'fit_warnings.txt'))
capture.output(sessionInfo(),file=file.path(output,'sessionInfo.txt'))
cat(sprintf('COMPLETE: promoters=%d stages=%d C=%d m=%.8f iterations=%d\n',nrow(x),ncol(x),k,m,cl$iter))
'''


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_indexed(path, sep=',', first='promoterId'):
    import csv
    with path.open(newline='') as stream:
        header = next(csv.reader(stream, delimiter=sep))
    require(header[0] == first and len(header) == len(set(header)), f'Invalid/duplicate header: {path}')
    data = pd.read_csv(path, sep=sep, dtype={first:str,'geneId':str,'stage':str}, keep_default_na=False)
    require(data[first].ne('').all() and data[first].is_unique, f'Invalid/duplicate IDs: {path}')
    return data.set_index(first)


def boolean(series, name):
    values = series.astype(str).str.lower()
    require(values.isin(['true','false','0','1']).all(), 'Invalid boolean: ' + name)
    return values.isin(['true','1'])


def select_fdr(stats, threshold):
    require('padj_BH' in stats, 'Missing padj_BH')
    q = pd.to_numeric(stats.padj_BH, errors='raise')
    require(np.isfinite(q).all() and q.between(0,1).all(), 'Invalid upstream FDR')
    return stats.loc[q.lt(threshold)].copy()


def average_stages(data, metadata, stages):
    require(data.columns.tolist() == metadata.index.tolist(), 'Matrix/sample metadata order differs')
    require(metadata.index.is_unique and stages.stage.is_unique, 'Duplicate sample/stage')
    require(stages.Time.is_monotonic_increasing and stages.Time.is_unique, 'Ambiguous chronological stages')
    require(set(metadata.stage) == set(stages.stage), 'Sample/stage-key membership differs')
    means = pd.DataFrame(index=data.index)
    for row in stages.itertuples(index=False):
        sub = metadata.loc[metadata.stage.eq(row.stage)]
        require(len(sub) == int(row.sample_n) and sub.Time.eq(row.Time).all(), f'Wrong samples/time: {row.stage}')
        require(sub.Replicates.eq(row.stage_index).all(), f'Wrong replicate block: {row.stage}')
        means[row.stage] = data.loc[:,sub.index].mean(axis=1)
    require(np.isfinite(means).to_numpy().all() and means.ge(0).to_numpy().all(), 'Invalid stage means')
    require(means.gt(0).any(axis=1).all(), 'All-zero promoter after stage means')
    return means


def prepare(target, args, upstream_manifest):
    species, organ = target.split('_',1)
    source = args.masigpro_output/target
    inp = args.masigpro_input/target
    paths = {
        'all_stats':source/f'{organ}_all_promoters_statistics.tsv',
        'sig_stats':source/f'{organ}_dynamic_promoters_statistics.tsv',
        'activity':source/f'{organ}_dynamic_promoters_fdr0_01.csv',
        'metadata':inp/'sample_metadata.tsv', 'stages':inp/'stage_index_to_time.tsv',
        'design':inp/f'{organ}_design_matrix.csv', 'annotation':inp/'MP_promoter_annotations.tsv',
    }
    require(all(p.is_file() for p in paths.values()), f'{target}: missing upstream files: {[str(p) for p in paths.values() if not p.is_file()]}')
    hashes = {str(p.resolve()):sha256(p) for p in paths.values()}
    known = upstream_manifest.get('output_sha256',{})
    for path, expected in hashes.items():
        require(known.get(path) == expected, 'Source changed/missing upstream checksum: ' + path)
    stats = read_indexed(paths['all_stats'], '\t')
    required = {'geneId','internalPromoter','retained_promoters_in_gene','padj_BH','dynamic'}
    require(required.issubset(stats.columns), f'{target}: upstream annotation/statistics missing')
    cohort = read_indexed(paths['annotation'], '\t')
    require(stats.index.equals(cohort.index), f'{target}: full MP cohort differs from statistics')
    require(stats.geneId.equals(cohort.geneId) and not boolean(stats.internalPromoter,'internalPromoter').any(), 'Invalid MP gene/flag')
    counts = cohort.groupby('geneId').size()
    require(counts.ge(2).all() and stats.geneId.map(counts).eq(stats.retained_promoters_in_gene).all(), 'Upstream MP definition/count mismatch')
    chosen = select_fdr(stats, args.fdr)
    require(len(chosen) >= args.clusters, f'{target}: only {len(chosen)} promoters with FDR<{args.fdr}; cannot fit C={args.clusters}')
    data = read_indexed(paths['activity']).apply(pd.to_numeric, errors='raise').astype(float)
    require(np.isfinite(data).to_numpy().all() and data.ge(0).to_numpy().all(), 'Invalid source activity')
    expected_export = stats.index[boolean(stats.dynamic,'dynamic')]
    require(data.index.equals(expected_export), f'{target}: exported dynamic matrix differs from statistics')
    significant = read_indexed(paths['sig_stats'], '\t')
    require(significant.index.equals(data.index), f'{target}: exported dynamic statistics differ')
    require(chosen.index.isin(data.index).all(), 'FDR-selected IDs absent from exported activity')
    data = data.loc[chosen.index]
    metadata = read_indexed(paths['metadata'], '\t', first='sample')
    design = read_indexed(paths['design'], first='sample').apply(pd.to_numeric, errors='raise')
    require(metadata.index.equals(design.index) and design.columns.tolist()==['Time','Replicates','Group'], 'Design/metadata row or column schema differs')
    np.testing.assert_allclose(metadata[['Time','Replicates','Group']].to_numpy(dtype=float),
                               design.to_numpy(dtype=float),rtol=0,atol=0,
                               err_msg='Design/metadata numeric values differ')
    require(metadata.species.eq(species).all() and metadata.organ.eq(organ).all(), 'Wrong species/organ metadata')
    stages = pd.read_csv(paths['stages'],sep='\t',dtype={'stage':str})
    require(stages.stage_index.tolist() == list(range(1,len(stages)+1)), 'Stage order/index differs')
    means = average_stages(data, metadata, stages)
    require(len(means.drop_duplicates()) >= args.clusters, f'{target}: fewer than C distinct stage profiles')
    for path, expected in hashes.items():
        require(sha256(Path(path)) == expected, 'Source changed during read: ' + path)
    summary = {'target':target, 'species':species, 'organ':organ, 'promoters':len(means),
               'genes':int(chosen.geneId.nunique()), 'samples':len(metadata), 'stages':len(stages),
               'FDR_selection':f'padj_BH < {args.fdr}', 'source_MP_genes':len(counts)}
    return means, chosen, metadata, stages, summary, hashes


def runtime_check(args):
    import matplotlib
    code = r'''
library.path <- commandArgs(TRUE)[1]
if (nzchar(library.path) && dir.exists(library.path)) .libPaths(c(library.path,.libPaths()))
if (!requireNamespace('Mfuzz',quietly=TRUE)) stop('Mfuzz unavailable on ',Sys.info()[['nodename']],
  '; R=',R.home(),'; searched: ',paste(.libPaths(),collapse=' | '),'; set --rscript/--r-library to the installed environment.')
cat(as.character(packageVersion('Mfuzz')),'\n',find.package('Mfuzz'),'\n',R.version.string,sep='')
'''
    result = subprocess.run([args.rscript,'--vanilla','-e',code,args.r_library], text=True,capture_output=True)
    require(result.returncode == 0, 'Mfuzz runtime check failed: ' + result.stderr[-2000:])
    return {'hostname':socket.gethostname(), 'Rscript':args.rscript, 'R_library':args.r_library,
            'Mfuzz':result.stdout.splitlines(), 'matplotlib':matplotlib.__version__}


def run_fit(matrix_path, out, organ, args):
    result = subprocess.run([args.rscript,'--vanilla','-e',R_BACKEND,str(matrix_path),str(out),
                             str(args.clusters),str(args.seed),str(args.core_threshold),
                             str(args.iter_max),args.r_library,organ],capture_output=True,text=True,
                            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    (out/'mfuzz_run.log').write_text(result.stdout + '\n' + result.stderr)
    require(result.returncode == 0, f'Mfuzz failed; see {out/"mfuzz_run.log"}: {result.stderr[-1800:]}')


def validate_fit(means, out, organ, args):
    membership = read_indexed(out/f'{organ}_membership.csv')
    assignments = read_indexed(out/f'{organ}_assignments.tsv','\t')
    assignments['is_core'] = boolean(assignments.is_core,'is_core')
    centers = read_indexed(out/f'{organ}_cluster_centers.csv', first='cluster')
    summary = pd.read_csv(out/f'{organ}_cluster_summary.tsv',sep='\t')
    diagnostics = pd.read_csv(out/'fit_diagnostics.tsv',sep='\t',dtype=str).set_index('key').value
    u = membership.to_numpy(dtype=float)
    require(membership.index.equals(means.index) and assignments.index.equals(means.index), 'Fit IDs/order differs')
    require(u.shape == (len(means),args.clusters) and np.isfinite(u).all() and (u>=0).all() and (u<=1).all(), 'Invalid memberships')
    np.testing.assert_allclose(u.sum(axis=1),1,atol=1e-8)
    np.testing.assert_array_equal(u.argmax(axis=1)+1,assignments.cluster)
    np.testing.assert_allclose(u.max(axis=1),assignments.membership,atol=1e-12)
    np.testing.assert_array_equal(u.max(axis=1)>args.core_threshold,assignments.is_core)
    require(centers.shape == (args.clusters,means.shape[1]) and centers.columns.tolist() == means.columns.tolist(), 'Center shape/stage mismatch')
    require(summary.cluster.tolist() == list(range(1,args.clusters+1)) and summary.assigned_n.sum()==len(means), 'Cluster count mismatch')
    require(int(diagnostics['iterations']) < args.iter_max and int(diagnostics['fit_warning_count']) == 0, 'Unconverged/warned Mfuzz fit')
    weights = u ** float(diagnostics['m'])
    np.testing.assert_allclose(centers,weights.T @ means.to_numpy()/weights.sum(axis=0)[:,None],rtol=1e-3,atol=1e-3)
    for cluster in range(1,args.clusters+1):
        row = summary.loc[summary.cluster.eq(cluster)].iloc[0]
        hard = assignments.cluster.eq(cluster).to_numpy()
        core = u[:,cluster-1]>args.core_threshold
        require(row.assigned_n == hard.sum() and row.core_n == core.sum(), 'Cluster/core counts differ')
        for mode,mask in [('mfuzz',hard),('core',core)]:
            exported = pd.read_csv(out/'clusters'/f'{organ}_{mode}_{cluster:02d}.csv',dtype={'promoterId':str})
            require(exported.promoterId.tolist() == means.index[mask].tolist(), 'Exported cluster promoter IDs differ')
            np.testing.assert_allclose(exported.Membership.to_numpy(dtype=float),u[mask,cluster-1],atol=1e-12)
    return assignments, summary, diagnostics


def numbered_ticks(n):
    if n > 14:
        return np.rint(np.linspace(1,n,5)).astype(int).tolist()
    ticks = list(range(1,n+1,3))
    if ticks[-1] != n:
        if n-ticks[-1] < 2:
            ticks[-1] = n
        else:
            ticks.append(n)
    return ticks


def profile_statistics(means, assignments, k):
    rows = []
    for cluster in range(1,k+1):
        values = means.loc[assignments.cluster.eq(cluster)].to_numpy()
        if len(values):
            mean = values.mean(axis=0)
            lo,hi = np.quantile(values,[.25,.75],axis=0)
            upper = max(16.,float(values.max())*1.04)
        else:
            mean = lo = hi = np.full(means.shape[1],np.nan)
            upper = 16.
        for i,stage in enumerate(means.columns):
            rows.append({'cluster':cluster,'stage_index':i+1,'stage':stage,'assigned_n':len(values),
                         'mean_activity':mean[i],'q25_activity':lo[i],'q75_activity':hi[i],
                         'y_axis_upper':upper,'expanded_y_upper':upper>16.})
    return pd.DataFrame(rows)


def plot_clusters(means, assignments, profiles, out, target, k):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'pdf.fonttype':42,
                         'svg.fonttype':'none','pdf.compression':0,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.linewidth':.65,'axes.grid':False})
    figures = out/'figures'; figures.mkdir()
    x = np.arange(1,means.shape[1]+1); n = len(x)
    pdf_path = out/f'{target}_MP_dynamic_mfuzz_C{k}.pdf'
    with PdfPages(pdf_path,metadata={'Title':f'{target} dynamic MP promoters; Mfuzz C={k}',
                  'Subject':'Original stage-mean activity; blue promoters, red mean and interquartile range; not PSI or confidence interval.'}) as pdf:
        for cluster in range(1,k+1):
            fig = plt.figure(figsize=(1.41,1.32),dpi=100)
            ax = fig.add_axes((28/141,34/132,94/141,68/132))
            values = means.loc[assignments.cluster.eq(cluster)].to_numpy()
            table = profiles.loc[profiles.cluster.eq(cluster)]
            if len(values):
                ax.add_collection(LineCollection([np.column_stack((x,y)) for y in values],
                    colors=PLOT_STYLE['blue_color'],linewidths=PLOT_STYLE['blue_linewidth_pt'],
                    alpha=PLOT_STYLE['blue_alpha'],zorder=PLOT_STYLE['blue_zorder']))
                ax.fill_between(x,table.q25_activity.to_numpy(),table.q75_activity.to_numpy(),
                    color=PLOT_STYLE['red_band_color'],alpha=PLOT_STYLE['red_band_alpha'],
                    linewidth=0,zorder=PLOT_STYLE['red_band_zorder'])
                ax.plot(x,table.mean_activity.to_numpy(),color=PLOT_STYLE['red_mean_color'],
                        lw=PLOT_STYLE['red_mean_linewidth_pt'],zorder=PLOT_STYLE['red_mean_zorder'])
            else:
                ax.text(.5,.5,'No assigned\npromoters',ha='center',va='center',fontsize=5.5,transform=ax.transAxes)
            ax.set_xlim(.5,n+.1); ax.set_ylim(0,float(table.y_axis_upper.iloc[0]))
            ax.set_xticks(numbered_ticks(n)); ax.set_xticks(x,minor=True)
            ax.set_yticks([0,4,8]); ax.set_yticks([2,6,10,12],minor=True)
            ax.spines['left'].set_bounds(0,12)
            ax.tick_params(axis='x',which='both',length=5.5,width=.65)
            ax.tick_params(axis='x',which='major',labelsize=7.5,pad=4)
            ax.tick_params(axis='y',which='both',length=4.5,width=.65)
            ax.tick_params(axis='y',which='major',labelsize=7.5,labelrotation=90,pad=3)
            fig.text(.55,.94,f'Cluster {cluster}',ha='center',va='top',fontsize=8.5,weight='bold')
            pdf.savefig(fig)
            stem = figures/f'{target}_MP_cluster_{cluster:02d}'
            fig.savefig(stem.with_suffix('.svg'))
            fig.savefig(stem.with_suffix('.png'),dpi=100)
            plt.close(fig)
    return pdf_path


def self_test(args):
    class Tests(unittest.TestCase):
        def test_strict_fdr_boundary(self):
            stats = pd.DataFrame({'padj_BH':[0,.009999,.01,.010001]},index=['a','b','c','d'])
            self.assertEqual(select_fdr(stats,.01).index.tolist(),['a','b'])

        def test_stage_means_keep_chronology(self):
            data = pd.DataFrame([[2.,4.,9.]],index=['p'],columns=['s1','s2','s3'])
            meta = pd.DataFrame({'stage':['10.5','10.5','0dpb'],'Time':[10.5,10.5,20.],
                                 'Replicates':[1,1,2]},index=data.columns)
            stages = pd.DataFrame({'stage_index':[1,2],'stage':['10.5','0dpb'],'Time':[10.5,20.],'sample_n':[2,1]})
            means = average_stages(data,meta,stages)
            self.assertEqual(means.columns.tolist(),['10.5','0dpb'])
            np.testing.assert_allclose(means,[[3.,9.]])

        def test_profile_summary_and_ticks(self):
            y = pd.DataFrame([[0.,4.],[4.,20.]],columns=['a','b'])
            a = pd.DataFrame({'cluster':[1,1]})
            stats = profile_statistics(y,a,2)
            c = stats.loc[stats.cluster.eq(1)]
            np.testing.assert_allclose(c.mean_activity,[2.,12.])
            np.testing.assert_allclose(c.q25_activity,[1.,8.])
            np.testing.assert_allclose(c.q75_activity,[3.,16.])
            self.assertTrue(c.y_axis_upper.gt(20.).all())
            self.assertTrue(stats.loc[stats.cluster.eq(2),'mean_activity'].isna().all())
            self.assertEqual(numbered_ticks(14),[1,4,7,10,14])
            self.assertEqual(numbered_ticks(24),[1,7,12,18,24])
            self.assertEqual(numbered_ticks(15),[1,4,8,12,15])

        def test_actual_R_fit(self):
            rng = np.random.default_rng(125)
            n = 14
            x = np.linspace(0,1,n)
            array = np.vstack([2+4*x+rng.normal(0,.08,n) for _ in range(15)] +
                              [6-4*x+rng.normal(0,.08,n) for _ in range(15)])
            means = pd.DataFrame(array,index=[f'p{i}' for i in range(30)],columns=[str(i) for i in range(n)])
            with tempfile.TemporaryDirectory(prefix='Mfuzz_selftest_') as temp:
                out = Path(temp)
                matrix = out/'matrix.csv'; means.to_csv(matrix,index_label='promoterId',float_format='%.17g')
                test_args = argparse.Namespace(**vars(args)); test_args.clusters=2
                run_fit(matrix,out,'Liver',test_args)
                assignments, summary, diagnostics = validate_fit(means,out,'Liver',test_args)
                self.assertEqual(summary.assigned_n.sum(),30)
                self.assertTrue(summary.assigned_n.gt(0).all())
                self.assertGreater(float(diagnostics['m']),1.)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    require(result.wasSuccessful(),'Self-tests failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--masigpro-output',type=Path,default=ROOT.parent/'8.1.run_magSigPro/output')
    parser.add_argument('--masigpro-input',type=Path,default=ROOT.parent/'8.1.run_magSigPro/input')
    parser.add_argument('--targets',nargs='+',choices=TARGETS,default=list(TARGETS))
    parser.add_argument('--input-dir',type=Path,default=ROOT/'input')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output')
    parser.add_argument('--fdr',type=float,default=.01)
    parser.add_argument('--clusters',type=int,default=10)
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--core-threshold',type=float,default=.8)
    parser.add_argument('--iter-max',type=int,default=1000)
    parser.add_argument('--rscript',default='/usr/bin/Rscript')
    parser.add_argument('--r-library',default='/home/ruiyanhou/R/x86_64-pc-linux-gnu-library/4.4')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args(argv)
    require(0<args.fdr<=.01 and args.clusters>=2 and .5<args.core_threshold<1 and args.iter_max>1,'Invalid parameters')
    require(len(set(args.targets))==len(args.targets),'Duplicate targets')
    runtime = runtime_check(args)
    if args.self_test:
        self_test(args)
        print('SELF_TEST_PASSED: synthetic fit only; no production inputs/results written')
        return 0
    upstream_path = args.masigpro_output/'run_manifest.json'
    require(upstream_path.is_file(),'Missing complete upstream run_manifest.json')
    upstream = json.loads(upstream_path.read_text())
    require(upstream['status']=='complete','maSigPro batch is not complete')
    hashes = {str(upstream_path.resolve()):sha256(upstream_path)}
    cohorts = {}
    for target in args.targets:
        prepared = prepare(target,args,upstream)
        cohorts[target] = prepared
        hashes.update(prepared[-1])
        print('PREFLIGHT',json.dumps(prepared[-2]),flush=True)
    if args.check_only:
        print('CHECK_ONLY_PASSED: all sources, MP definitions, strict FDR, stage averaging and runtimes checked; no output written')
        return 0
    input_root,output_root = args.input_dir.resolve(),args.output_dir.resolve()
    require(input_root!=output_root and not input_root.is_relative_to(output_root)
            and not output_root.is_relative_to(input_root),'Input/output directories must be separate, not nested')
    for folder in [input_root,output_root]:
        require(folder.is_relative_to(ROOT),'Outputs must remain within this script directory')
        require(not folder.exists() or (folder.is_dir() and not any(folder.iterdir())),f'Refusing nonempty output: {folder}')
    input_root.mkdir(parents=True,exist_ok=True); output_root.mkdir(parents=True,exist_ok=True)
    manifest = {'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
                'targets':args.targets,'clusters':args.clusters,'seed':args.seed,'core_threshold':args.core_threshold,
                'iter_max':args.iter_max,'FDR_selection':f'padj_BH < {args.fdr}',
                'pooling':'One independent fit per species/organ; no major/minor split',
                'fit_transform':'none; arithmetic within-stage means of source sample activities',
                'PSI_conversion':False,'m':'Mfuzz::mestimate per target',
                'plot_mean':'Arithmetic mean across all hard-assigned promoters, not fitted fuzzy center',
                'plot_band':'25th-75th percentiles across promoters, not confidence interval',
                'plot_style':PLOT_STYLE,'plot_inches':[1.41,1.32],'plot_pixels':[141,132],
                'plot_y_major_ticks':[0,4,8],'plot_y_minor_ticks':[2,6,10,12],
                'x_tick_rule':'Reference ordinal labels up to 14 stages; five spaced labels for >14; all stages have minor ticks',
                'runtime':runtime,'source_sha256':hashes,'script_sha256':sha256(Path(__file__)),'groups':[]}
    manifest_path = output_root/'run_manifest.json'; save_json(manifest_path,manifest)
    try:
        for target,(means,labels,metadata,stages,record,_) in cohorts.items():
            organ = record['organ']; inp,out = input_root/target,output_root/target
            inp.mkdir(); out.mkdir()
            matrix = inp/f'{organ}_stage_mean_activity.csv'
            means.to_csv(matrix,index_label='promoterId',float_format='%.17g')
            labels.to_csv(inp/f'{organ}_selected_promoter_annotations.tsv',sep='\t',index_label='promoterId')
            metadata.to_csv(inp/'sample_to_stage.tsv',sep='\t',index_label='sample')
            for folder in [inp,out]:
                stages.to_csv(folder/'stage_index_to_time.tsv',sep='\t',index=False)
            run_fit(matrix,out,organ,args)
            assignments,summary,diagnostics = validate_fit(means,out,organ,args)
            classified = assignments.join(labels,validate='one_to_one')
            classified.to_csv(out/f'{organ}_promoter_cluster_annotations.tsv',sep='\t',index_label='promoterId')
            summary['gene_n'] = summary.cluster.map(classified.groupby('cluster').geneId.nunique()).fillna(0).astype(int)
            summary.to_csv(out/f'{organ}_cluster_summary.tsv',sep='\t',index=False)
            profiles = profile_statistics(means,assignments,args.clusters)
            profiles.to_csv(out/f'{organ}_plot_profile_statistics.tsv',sep='\t',index=False)
            plot_clusters(means,assignments,profiles,out,target,args.clusters)
            record.update(clusters=args.clusters,m=float(diagnostics['m']),iterations=int(diagnostics['iterations']),
                          fit_warnings=int(diagnostics['fit_warning_count']),
                          nonempty_clusters=int(summary.assigned_n.gt(0).sum()),
                          expanded_y_clusters=int(profiles.loc[profiles.expanded_y_upper,'cluster'].nunique()))
            manifest['groups'].append(record); save_json(manifest_path,manifest)
            print('COMPLETE',json.dumps(record),flush=True)
        for path,expected in hashes.items():
            require(sha256(Path(path))==expected,'Upstream input/result changed: '+path)
        pd.DataFrame(manifest['groups']).to_csv(output_root/'all_targets_run_summary.tsv',sep='\t',index=False)
        manifest['output_sha256'] = {str(p.relative_to(ROOT)):sha256(p) for folder in [input_root,output_root]
                                     for p in folder.rglob('*') if p.is_file() and p!=manifest_path}
        manifest.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),upstream_unchanged=True)
        save_json(manifest_path,manifest)
    except Exception as error:
        manifest.update(status='failed',error=str(error));save_json(manifest_path,manifest)
        raise
    return 0


if __name__=='__main__':
    sys.exit(main())
