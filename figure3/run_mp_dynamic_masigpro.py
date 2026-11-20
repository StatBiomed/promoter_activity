#!/usr/bin/env python3
"""Find dynamic promoters of MP genes in Mouse organs and Human/Rat Liver.

One self-contained Python entry point; the embedded numerical backend calls the
installed R maSigPro package without creating an R script. No plotting is done.

MP definition, separately in each species/organ:
  1. Remove internal promoters.
  2. Remove promoters whose activity is zero in EVERY sample at EVERY stage.
  3. Keep genes with >=2 distinct promoters remaining after steps 1 and 2.
All remaining promoters of those genes are tested together, without splitting
major/minor, choosing only two promoters, or requiring two dynamic promoters.

Run with the server Python environment:
  /mnt/ruiyanhou/nfs_share2/conda_env/envs/PY310/bin/python -B run_mp_dynamic_masigpro.py
  ... run_mp_dynamic_masigpro.py --check-only   # full input checks; no output
  ... run_mp_dynamic_masigpro.py --self-test    # synthetic tests in temporary storage

Default fit: degree=2, Gaussian counts=FALSE, min.obs=6, BH Q<=0.01 separately
within each species/organ's entire MP-promoter cohort. Sample-level input activity
and original numerical developmental times are used unchanged, with no additional
log, PSI conversion, standardisation, or averaging of biological replicates.
For an all-zero sample column, keep the sample: inactivity is a PROMOTER filter.
Existing nonempty input/output destinations are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo')
TIME_ROOT = Path('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/time_transfer')
MOUSE_ORGANS = ('Brain', 'Cerebellum', 'Heart', 'Kidney', 'Liver', 'Ovary', 'Testis')
TARGETS = tuple('Mouse_' + organ for organ in MOUSE_ORGANS) + ('Human_Liver', 'Rat_Liver')
ANNOTATION = ['geneId', 'seqnames', 'start', 'strand', 'internalPromoter', 'promoterPosition', 'txId']

R_BACKEND = r'''
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==6L)
matrix.path <- args[1]; design.path <- args[2]; out <- args[3]
degree <- as.integer(args[4]); Q <- as.numeric(args[5])
.libPaths(c(args[6], .libPaths()))
suppressPackageStartupMessages(library(maSigPro))
options(digits=17)
read.numeric <- function(path) {
  z <- read.csv(path, check.names=FALSE, colClasses='character')
  ids <- z[[1]]
  if (anyNA(ids) || any(!nzchar(ids)) || anyDuplicated(ids)) stop('Invalid row IDs')
  x <- as.matrix(z[-1]); storage.mode(x) <- 'double'; rownames(x) <- ids
  if (any(!is.finite(x))) stop('Non-finite numeric input')
  x
}
y <- read.numeric(matrix.path)
d <- as.data.frame(read.numeric(design.path))
if (!identical(colnames(y), rownames(d))) stop('Sample order differs')
if (!identical(colnames(d), c('Time','Replicates','Group'))) stop('Wrong design columns')
if (nrow(y)<2L || ncol(y)<6L || any(y<0) || any(rowSums(y)==0)) stop('Invalid MP matrix')
if (is.unsorted(d$Time) || any(d$Group!=1)) stop('Invalid single-organ design')
if (length(unique(d$Time))<=degree || ncol(y)<=degree+1L) stop('Insufficient time/residual degrees of freedom')
design <- make.design.matrix(d, degree=degree)
x <- cbind(Intercept=1, as.matrix(design$dis))
probe <- glm(as.numeric(y[1,]) ~ ., data=as.data.frame(design$dis), family=gaussian(), epsilon=1e-5)
if (probe$rank!=ncol(x) || !probe$converged) stop('Rank-deficient or unconverged design')
warnings <- character()
fit <- withCallingHandlers(p.vector(y, design, Q=Q, MT.adjust='BH', counts=FALSE,
                                   min.obs=6, item='MP promoter'),
  warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart('muffleWarning')})
if (length(warnings)) stop('maSigPro warnings; results withheld: ', paste(unique(warnings),collapse=' | '))
if (fit$G!=nrow(y) || fit$g!=nrow(y) || !identical(rownames(fit$p.vector),rownames(y)))
  stop('maSigPro dropped or reordered promoters')
p <- as.numeric(fit$p.vector[,1]); q <- as.numeric(fit$p.adjusted)
if (length(p)!=nrow(y) || any(!is.finite(p)) || any(p<0 | p>1) || any(!is.finite(q))) stop('Invalid P/FDR values')
if (!isTRUE(all.equal(q,p.adjust(p,'BH'),tolerance=1e-14))) stop('BH mismatch')
keep <- q<=Q
selected <- rownames(fit$SELEC); if (is.null(selected)) selected <- character()
if (!identical(selected,rownames(y)[keep]) || fit$i!=sum(keep)) stop('SELEC/FDR mismatch')
write.table(data.frame(promoterId=rownames(y),p_value=p,padj_BH=q,dynamic=keep),
            file.path(out,'fit_statistics.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
write.table(data.frame(degree=degree,samples=ncol(y),stages=length(unique(d$Time)),
  parameters=ncol(x),glm_rank=probe$rank,residual_df=probe$df.residual,
  condition_number=kappa(x),maSigPro_version=as.character(packageVersion('maSigPro'))),
  file.path(out,'design_diagnostics.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
write.table(data.frame(sample=rownames(design$dis),design$dis,check.names=FALSE),
  file.path(out,'regression_design_matrix.csv'),sep=',',row.names=FALSE,quote=TRUE)
saveRDS(list(fit=fit,design=design,degree=degree,Q=Q,counts=FALSE,
             BH_scope='all retained MP promoters within this species/organ'),
        file.path(out,'maSigPro_fit.rds'))
capture.output(sessionInfo(),file=file.path(out,'sessionInfo.txt'))
cat('COMPLETE:',sum(keep),'/',nrow(y),'dynamic MP promoters\n')
'''


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stamp(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns


def header_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, delimiter='\t')
        header, first = next(reader), next(reader)
    require(len(header) == len(set(header)), f'Duplicate header: {path}')
    return header, first


def read_annotation(path, condition):
    header, first = header_rows(path)
    if len(first) == len(header) + 1:
        names = ['__R_rownames__'] + header
    else:
        require(len(first) == len(header), f'Invalid annotation widths: {path}')
        names = header
    mean_column = condition + '.mean'
    columns = ['promoterId'] + ANNOTATION + [mean_column]
    require(set(columns).issubset(names), f'Missing annotation fields: {path}')
    frame = pd.read_csv(path, sep='\t', skiprows=1, header=None, names=names,
                        usecols=columns, dtype=str, keep_default_na=False)
    require(frame.promoterId.ne('').all() and frame.promoterId.is_unique, f'Invalid promoter IDs: {path}')
    require(frame.geneId.ne('').all() and not frame.geneId.isin(['NA','NaN']).any(), f'Missing gene IDs: {path}')
    flags = frame.internalPromoter.str.lower()
    require(flags.isin(['true','false','1','0']).all(), f'Unknown internal flag: {path}')
    frame['internalPromoter'] = flags.isin(['true','1'])
    frame[mean_column] = pd.to_numeric(frame[mean_column], errors='raise')
    require(np.isfinite(frame[mean_column]).all() and frame[mean_column].ge(0).all(), f'Invalid condition means: {path}')
    return frame.set_index('promoterId')


def read_activity(path):
    header, first = header_rows(path)
    if len(first) == len(header) + 1:
        samples = header
    else:
        require(len(first) == len(header) and header[0] in ['', 'promoterId', 'Unnamed: 0'],
                f'Invalid activity header: {path}')
        samples = header[1:]
    require(samples and len(set(samples)) == len(samples), f'Empty/duplicate samples: {path}')
    frame = pd.read_csv(path, sep='\t', skiprows=1, header=None,
                        names=['promoterId'] + samples, dtype={'promoterId': str}, keep_default_na=False)
    require(frame.promoterId.ne('').all() and frame.promoterId.is_unique, f'Invalid activity IDs: {path}')
    frame = frame.set_index('promoterId').apply(pd.to_numeric, errors='raise').astype(float)
    require(np.isfinite(frame.to_numpy()).all() and frame.ge(0).to_numpy().all(), f'Invalid activity: {path}')
    return frame


def read_times(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    key = '0' if '0' in frame else 'time'
    require(key in frame and 'new_time_days' in frame, f'Invalid time table: {path}')
    require(frame[key].is_unique and frame[key].ne('').all(), f'Ambiguous stage names: {path}')
    times = pd.to_numeric(frame.new_time_days, errors='raise')
    require(np.isfinite(times).all(), f'Non-finite times: {path}')
    return dict(zip(frame[key], times.astype(float)))


def select_mp(annotation, matrix, stage_means):
    require(annotation.index.equals(matrix.index), 'Annotation/activity row alignment differs')
    noninternal = ~annotation.internalPromoter
    active = matrix.gt(0).any(axis=1)
    eligible = noninternal & active
    counts = annotation.loc[eligible].groupby('geneId').size()
    mp_genes = counts.index[counts.ge(2)]
    selected = eligible & annotation.geneId.isin(mp_genes)
    labels = annotation.loc[selected].copy()
    labels['retained_promoters_in_gene'] = labels.geneId.map(counts).astype(int)
    labels['active_stage_n'] = stage_means.loc[selected].gt(0).sum(axis=1)
    labels['mean_activity_across_stages'] = stage_means.loc[selected].mean(axis=1)
    labels['mean_activity_across_samples'] = matrix.loc[selected].mean(axis=1)
    summary = {
        'initial_genes': int(annotation.geneId.nunique()), 'initial_promoters': len(annotation),
        'internal_promoters_removed': int((~noninternal).sum()),
        'noninternal_genes': int(annotation.loc[noninternal, 'geneId'].nunique()),
        'noninternal_promoters': int(noninternal.sum()),
        'all_stage_zero_promoters_removed_after_internal': int((noninternal & ~active).sum()),
        'active_noninternal_genes': len(counts), 'active_noninternal_promoters': int(eligible.sum()),
        'single_promoter_genes_excluded': int(counts.eq(1).sum()),
        'MP_genes': len(mp_genes), 'MP_promoters': int(selected.sum()),
    }
    require(summary['active_noninternal_genes'] == summary['single_promoter_genes_excluded'] + summary['MP_genes'],
            'Gene count reconciliation failed')
    require(summary['active_noninternal_promoters'] == summary['single_promoter_genes_excluded'] + summary['MP_promoters'],
            'Promoter count reconciliation failed')
    return matrix.loc[selected].copy(), labels, summary


def prepare(target, args):
    species, organ = target.split('_', 1)
    folder = args.activity_root / species
    time_path = args.time_root / f'{species}.csv'
    sources = {time_path: stamp(time_path)}
    times = read_times(time_path)
    prefix = f'{species}_{organ}_'
    paths = list(folder.glob(prefix + '*_activity.tsv'))
    require(len(paths) >= 3, f'{target}: fewer than three condition files')
    expected_annotations = {p.name.replace('_activity.tsv', '_rowData.tsv') for p in paths}
    require(expected_annotations == {p.name for p in folder.glob(prefix + '*_rowData.tsv')},
            f'{target}: activity/annotation condition inventories differ')
    records = []
    for path in paths:
        stage = path.name[len(prefix):-len('_activity.tsv')]
        require(stage in times, f'{target}: stage absent from time map: {stage}')
        annotation_path = path.with_name(path.name.replace('_activity.tsv', '_rowData.tsv'))
        require(annotation_path.is_file(), f'Missing annotation: {annotation_path}')
        records.append((times[stage], stage, path, annotation_path))
    records.sort(key=lambda r: (r[0], r[1]))
    require(len({r[0] for r in records}) == len(records), f'{target}: distinct stages share numerical times')
    matrices, means, sample_rows, stage_rows = [], [], [], []
    reference = None
    for block, (time, stage, activity_path, annotation_path) in enumerate(records, 1):
        sources.update({p: stamp(p) for p in [activity_path, annotation_path]})
        condition = prefix + stage
        anno = read_annotation(annotation_path, condition)
        activity = read_activity(activity_path)
        require(set(anno.index) == set(activity.index), f'Annotation/activity promoter IDs differ: {condition}')
        if reference is None:
            reference = anno[ANNOTATION].copy()
        require(set(anno.index) == set(reference.index), f'Promoter universe changes: {condition}')
        anno = anno.loc[reference.index]
        require(anno[ANNOTATION].equals(reference), f'Static annotation/internal flags change: {condition}')
        activity = activity.loc[reference.index]
        actual_mean = activity.mean(axis=1)
        np.testing.assert_allclose(actual_mean, anno[condition + '.mean'], rtol=1e-10, atol=1e-12,
                                   err_msg=f'Sample means disagree with rowData: {condition}')
        for sample in activity.columns:
            require(f'.{species}.{organ}.{stage}.' in sample and sample.endswith('.sorted'),
                    f'Sample name/condition mismatch: {sample}')
            sample_rows.append({'sample': sample, 'species': species, 'organ': organ, 'stage': stage,
                                'Time': time, 'Replicates': block, 'Group': 1,
                                'source_activity_file': str(activity_path)})
        matrices.append(activity.loc[~reference.internalPromoter])
        means.append(actual_mean.loc[~reference.internalPromoter].rename(stage))
        stage_rows.append({'stage_index': block, 'stage': stage, 'Time': time, 'sample_n': activity.shape[1]})
    metadata = pd.DataFrame(sample_rows).set_index('sample')
    require(metadata.index.is_unique, f'{target}: duplicate sample across conditions')
    combined = pd.concat(matrices, axis=1)
    stage_activity = pd.concat(means, axis=1)
    require(combined.columns.tolist() == metadata.index.tolist(), 'Sample order differs')
    # Restore internal rows as zero placeholders ONLY for count reconciliation.
    # Their activities are not used: noninternal is applied before inactivity/MP.
    full = combined.reindex(reference.index, fill_value=0.)
    all_stage = stage_activity.reindex(reference.index, fill_value=0.)
    selected, labels, summary = select_mp(reference, full, all_stage)
    require(len(selected) >= 2, f'{target}: no eligible MP-promoter cohort')
    design = metadata[['Time','Replicates','Group']].copy()
    require(selected.shape[1] >= 6 and len(records) > args.degree, f'{target}: too few samples/stages')
    x = np.vander(design.Time.to_numpy(), N=args.degree + 1, increasing=True)
    require(np.isfinite(x).all() and np.linalg.matrix_rank(x) == args.degree + 1, f'{target}: rank-deficient time design')
    require(selected.shape[1] > args.degree + 1, f'{target}: no residual degrees of freedom')
    source_hashes = {}
    for path, before in sources.items():
        require(stamp(path) == before, f'Source changed while reading: {path}')
        source_hashes[str(path)] = sha256(path)
        require(stamp(path) == before, f'Source changed while hashing: {path}')
    summary.update(species=species, organ=organ, samples=len(metadata), stages=len(records),
                   degree=args.degree, Q=args.q, all_zero_sample_columns_retained=int(selected.eq(0).all(axis=0).sum()),
                   positive_constant_promoters_retained=int(selected.nunique(axis=1).eq(1).sum()))
    return selected, labels, design, metadata, pd.DataFrame(stage_rows), summary, source_hashes


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind='stable')
    scaled = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum(1., np.minimum.accumulate(scaled[::-1])[::-1])
    result = np.empty_like(q)
    result[order] = q
    return result


def run_backend(matrix_path, design_path, out, args):
    result = subprocess.run([args.rscript, '--vanilla', '-e', R_BACKEND, str(matrix_path),
                             str(design_path), str(out), str(args.degree), str(args.q), args.r_library],
                            capture_output=True, text=True)
    (out / 'maSigPro_run.log').write_text(result.stdout + '\n' + result.stderr)
    require(result.returncode == 0, f'maSigPro failed; inspect {out / "maSigPro_run.log"}: {result.stderr[-1800:]}')
    stats = pd.read_csv(out / 'fit_statistics.tsv', sep='\t', dtype={'promoterId': str}).set_index('promoterId')
    require(np.isfinite(stats[['p_value','padj_BH']]).to_numpy().all(), 'Invalid regression P/FDR')
    require(stats[['p_value','padj_BH']].ge(0).to_numpy().all() and
            stats[['p_value','padj_BH']].le(1).to_numpy().all(), 'P/FDR outside [0,1]')
    np.testing.assert_allclose(stats.padj_BH, bh_adjust(stats.p_value), atol=1e-13, rtol=1e-12)
    flags = stats.dynamic.astype(str).str.lower()
    require(flags.isin(['true','false']).all(), 'Invalid dynamic flags')
    stats['dynamic'] = flags.eq('true')
    require(stats.dynamic.equals(stats.padj_BH.le(args.q).rename('dynamic')), 'Dynamic/FDR mismatch')
    return stats


def runtime_check(args):
    code = '.libPaths(c(commandArgs(TRUE)[1],.libPaths()));suppressPackageStartupMessages(library(maSigPro));cat(as.character(packageVersion("maSigPro")))'
    result = subprocess.run([args.rscript, '--vanilla', '-e', code, args.r_library], capture_output=True, text=True)
    require(result.returncode == 0, 'Installed maSigPro unavailable: ' + result.stderr[-1200:])
    return result.stdout.strip()


def self_test(args):
    class Tests(unittest.TestCase):
        def test_definition_and_filter_order(self):
            ids = pd.Index(['a','b','c','d','e','f','g','h','i'], name='promoterId')
            anno = pd.DataFrame({'geneId':['g1','g1','g2','g2','g3','g3','g4','g4','g5'],
                                 'internalPromoter':[False,False,False,False,False,True,False,False,False]}, index=ids)
            activity = pd.DataFrame([[0,2],[1,0],[0,0],[1,1],[2,3],[8,8],[2,2],[3,3],[4,4]], index=ids)
            selected, labels, summary = select_mp(anno, activity, activity)
            self.assertEqual(selected.index.tolist(), ['a','b','g','h'])
            self.assertEqual(summary['MP_genes'], 2)
            self.assertEqual(summary['single_promoter_genes_excluded'], 3)
            self.assertTrue(labels.retained_promoters_in_gene.eq(2).all())

        def test_bh_known_example(self):
            np.testing.assert_allclose(bh_adjust([.001,.01,.04,.2]), [.004,.02,.05333333333333334,.2])

        def test_all_three_promoters_are_retained(self):
            ids = pd.Index(['a','b','c'], name='promoterId')
            anno = pd.DataFrame({'geneId':['g1']*3, 'internalPromoter':[False]*3}, index=ids)
            activity = pd.DataFrame([[0,2],[1,0],[1,1]], index=ids)
            y, labels, summary = select_mp(anno, activity, activity)
            self.assertEqual(y.index.tolist(), ['a','b','c'])
            self.assertTrue(labels.retained_promoters_in_gene.eq(3).all())
            self.assertEqual(summary['MP_genes'], 1)

        def test_misaligned_annotation_rejected(self):
            anno = pd.DataFrame({'geneId':['g1','g1'], 'internalPromoter':[False,False]}, index=['a','b'])
            activity = pd.DataFrame([[1,2],[2,3]], index=['b','a'])
            with self.assertRaisesRegex(ValueError, 'alignment'):
                select_mp(anno, activity, activity)

        def test_actual_backend_and_R_style_tables(self):
            with tempfile.TemporaryDirectory(prefix='maSigPro_selftest_') as temp:
                folder = Path(temp)
                t = np.repeat(np.arange(4), 2)
                columns = [f's{i}' for i in range(8)]
                y = pd.DataFrame([1+3*t+np.tile([-.1,.1],4), np.full(8,2.),
                                  4-t+np.tile([-.15,.15],4)], index=['p1','p2','p3'], columns=columns)
                d = pd.DataFrame({'Time':t, 'Replicates':t+1, 'Group':1}, index=columns)
                y.to_csv(folder/'matrix.csv', index_label='promoterId')
                d.to_csv(folder/'design.csv', index_label='sample')
                test_args = argparse.Namespace(**vars(args)); test_args.degree=2; test_args.q=.01
                stats = run_backend(folder/'matrix.csv', folder/'design.csv', folder, test_args)
                self.assertEqual(stats.index.tolist(), ['p1','p2','p3'])
                self.assertTrue(stats.loc['p1','dynamic'])
                self.assertFalse(stats.loc['p2','dynamic'])
                self.assertEqual(stats.loc['p2','p_value'], 1.)
                (folder/'activity.tsv').write_text('s1\ts2\n1\t0\t2\n2\t1\t0\n')
                self.assertEqual(read_activity(folder/'activity.tsv').index.tolist(), ['1','2'])
                header = ['promoterId'] + ANNOTATION + ['Mouse_Liver_0dpb.mean']
                values = ['1','1','ENSMUSG1','chr1','100','+','FALSE','1','tx1','1']
                (folder/'rowData.tsv').write_text('\t'.join(header)+'\n'+'\t'.join(values)+'\n')
                self.assertEqual(read_annotation(folder/'rowData.tsv','Mouse_Liver_0dpb').index.tolist(), ['1'])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    require(result.wasSuccessful(), 'Self-tests failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--activity-root', type=Path, default=BASE/'1.proactiv_result')
    parser.add_argument('--time-root', type=Path, default=TIME_ROOT)
    parser.add_argument('--targets', nargs='+', choices=TARGETS, default=list(TARGETS))
    parser.add_argument('--input-dir', type=Path, default=ROOT/'input')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'output')
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--q', type=float, default=.01)
    parser.add_argument('--rscript', default='/usr/bin/Rscript')
    parser.add_argument('--r-library', default='/home/ruiyanhou/R/x86_64-pc-linux-gnu-library/4.4')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    require(args.degree >= 1 and 0 < args.q < 1, 'Invalid degree/FDR threshold')
    require(len(set(args.targets)) == len(args.targets), 'Duplicate requested targets')
    version = runtime_check(args)
    if args.self_test:
        self_test(args)
        print('SELF_TEST_PASSED; no production inputs/results written')
        return 0
    if not args.check_only:
        require(args.input_dir.resolve() != args.output_dir.resolve(), 'Input/output roots must differ')
        for folder in [args.input_dir, args.output_dir]:
            require(not folder.exists() or (folder.is_dir() and not any(folder.iterdir())),
                    f'Refusing nonempty destination: {folder}')
        for folder in [args.input_dir, args.output_dir]:
            folder.mkdir(parents=True, exist_ok=True)
    manifest = {'status':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
                'targets':args.targets, 'degree':args.degree, 'Q':args.q, 'counts':False,
                'maSigPro_version':version, 'min_obs':6, 'T_fit_run':False,
                'MP_definition':'Remove internal; remove promoters zero in all samples at all organ stages; retain genes with >=2 remaining distinct promoters',
                'BH_scope':'All MP promoters together, separately within each species/organ; no major/minor split',
                'activity_transform':'none; exact source sample-level values', 'replicate_averaging':False,
                'Time':'original new_time_days from species time map', 'script_sha256':sha256(Path(__file__)),
                'groups':[], 'source_sha256':{}}
    manifest_path = args.output_dir/'run_manifest.json'
    if not args.check_only:
        save_json(manifest_path, manifest)
    try:
        for target in args.targets:
            y, labels, design, metadata, stages, summary, hashes = prepare(target, args)
            print('PREFLIGHT', target, json.dumps(summary, ensure_ascii=False), flush=True)
            manifest['source_sha256'].update(hashes)
            if args.check_only:
                continue
            organ = summary['organ']
            inp, out = args.input_dir/target, args.output_dir/target
            inp.mkdir(); out.mkdir()
            matrix_path, design_path = inp/f'{organ}_dynamic_matrix.csv', inp/f'{organ}_design_matrix.csv'
            y.to_csv(matrix_path, index_label='promoterId', float_format='%.17g')
            design.to_csv(design_path, index_label='sample', float_format='%.17g')
            labels.to_csv(inp/'MP_promoter_annotations.tsv', sep='\t', index_label='promoterId')
            metadata.to_csv(inp/'sample_metadata.tsv', sep='\t', index_label='sample')
            stages.to_csv(inp/'stage_index_to_time.tsv', sep='\t', index=False)
            save_json(inp/'filter_summary.json', summary)
            stats = run_backend(matrix_path, design_path, out, args)
            require(stats.index.equals(y.index), 'maSigPro row identity/order differs')
            stats = labels.join(stats, how='left', validate='one_to_one')
            stats['degree'], stats['Q'], stats['counts'] = args.degree, args.q, False
            stats['BH_scope'] = 'all_MP_promoters_within_' + target
            stats.to_csv(out/f'{organ}_all_promoters_statistics.tsv', sep='\t', index_label='promoterId')
            dynamic = stats.loc[stats.dynamic]
            dynamic.to_csv(out/f'{organ}_dynamic_promoters_statistics.tsv', sep='\t', index_label='promoterId')
            suffix = format(args.q, '.12g').replace('.', '_')
            y.loc[dynamic.index].to_csv(out/f'{organ}_dynamic_promoters_fdr{suffix}.csv',
                                       index_label='promoterId', float_format='%.17g')
            (out/f'{organ}_dynamic_promoter_ids_fdr{suffix}.txt').write_text(''.join(str(p)+'\n' for p in dynamic.index))
            summary.update(dynamic_promoters=len(dynamic), nondynamic_promoters=len(stats)-len(dynamic),
                           genes_with_dynamic_promoter=int(dynamic.geneId.nunique()))
            manifest['groups'].append(summary)
            save_json(manifest_path, manifest)
            print('COMPLETE', target, len(dynamic), '/', len(stats), 'dynamic MP promoters', flush=True)
        for path, expected in manifest['source_sha256'].items():
            require(sha256(Path(path)) == expected, f'Source changed during workflow: {path}')
        if args.check_only:
            print('CHECK_ONLY_PASSED: all requested source cohorts validated; no files/directories created')
            return 0
        pd.DataFrame(manifest['groups']).to_csv(args.output_dir/'all_targets_run_summary.tsv', sep='\t', index=False)
        manifest['output_sha256'] = {str(p.resolve()):sha256(p) for folder in [args.input_dir,args.output_dir]
                                     for p in folder.rglob('*') if p.is_file() and p != manifest_path}
        manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(manifest_path, manifest)
    except Exception as error:
        if not args.check_only:
            manifest.update(status='failed', error=str(error)); save_json(manifest_path, manifest)
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
