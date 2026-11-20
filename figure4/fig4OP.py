#!/usr/bin/env python3
"""Compare WT/HNF4A-KO PROMOTER USAGE for fixed mouse-liver A/B sets.

Python 3.10+; numpy, matplotlib, scipy (for the default Wilcoxon test).
  python -B plot_HNF4A_KO_groupAB_promoter_usage.py --self-test
  python -B plot_HNF4A_KO_groupAB_promoter_usage.py --check-only
  python -u -B plot_HNF4A_KO_groupAB_promoter_usage.py

Project-consistent usage = condition promoter .mean / condition gene .gene.mean.
This is a RATIO OF CONDITION MEANS, not mean(sample-level ratios), NOT A/(A+B).
The denominator includes ALL promoters of that gene in the provided ProActiv
assay, not merely the two selected A/B promoters. It is checked by reconstructing
every gene's sample activity sums and matching all condition .gene.mean values.
ProActiv geneExpression sums the supplied absolute-promoter activity values;
these may already be logged. Do not invert/log them or reinterpret the result as
a linear read-count fraction. Per-sample usage in the audit table is not averaged
to form box observations. Official definitions:
https://raw.githubusercontent.com/GoekeLab/proActiv/master/R/estimate-promoter-activity.R
https://raw.githubusercontent.com/GoekeLab/proActiv/master/R/proActiv.R

0/positive remains 0, 0/0 remains undefined. For matched boxes and tests, include
only promoters with defined usage in BOTH WT and KO. Original selected promoters
and all undefined values are retained in the audit tables; no imputation occurs.
Both conditions must match the selected promoterId/gene/chr/TSS/strand exactly.

One quantitative distribution comparison per PDF: Group A and Group B. Same
45 x 68 mm canvas, shared y limits, WT green / HNF4A_KO coral; median, IQR,
1.5-IQR whiskers and hollow-circle outliers, all values inside the axes. The
scientific question is the WT-KO usage distribution difference, not a prior
assumption about its direction. Usage ticks span 0 to 1; space above 1 is reserved
for the bracket. Original files and previous absolute-activity plots are preserved.
Cross-condition joint normalization is not established by these separately
exported WT/KO activity files. These plots describe the supplied values; they
alone cannot establish a causal response to the knockout.

Default --test wilcoxon performs an exploratory paired signed-rank test across
promoter usage values, with BH correction across the A/B comparisons. Use
--test none to omit it. These are NOT independent biological replicates:
there are three samples per condition. No mouse/sample pairing is assumed.
Two plots plus per-sample activities, per-promoter usage, gene denominators,
missingness, statistics and provenance go to results_usage_wilcoxon/; existing output is
refused. --check-only and --self-test never create final outputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

BASE = Path('/mnt/ruiyanhou/nfs_share2')
ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE/'RNA_seq_organ_species/unibind_result/HNF4A_KO/proactiv_result'
DEFAULT_SELECTION = BASE/'proactiv_redo/13.promoter_usage/1.results/Mouse_Liver/Mouse_Liver_selected_promoters.tsv'
CONDITIONS = ('WT', 'HNF4A_KO')
COLORS = {'WT': '#A9C995', 'HNF4A_KO': '#E47766'}
IDENTITY = ('geneId', 'seqnames', 'start', 'strand')
FIGURE_MM = (45., 68.)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_r_tsv(path):
    """Accept R write.table's optional, unlabelled leading row-name field."""
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader)
        first = next(reader)
        require(len(first) in (len(header), len(header)+1), 'Invalid TSV width: '+str(path))
        if len(first) == len(header)+1:
            header = ['_R_rowname', *header]
        elif header[0] in ('', 'X', 'row.names'):
            header[0] = '_R_rowname'
        require(len(set(header)) == len(header), 'Duplicate TSV headers: '+str(path))
        require(len(first) == len(header), 'Invalid first row: '+str(path))
        rows = [dict(zip(header, first))]
        for line_no, fields in enumerate(reader, 3):
            require(len(fields) == len(header), '{}:{}: inconsistent TSV width'.format(path, line_no))
            rows.append(dict(zip(header, fields)))
    return header, rows


def load_activity(path):
    header, rows = read_r_tsv(path)
    require(header[0] in ('_R_rowname', 'promoterId'), 'Activity requires an explicit leading promoter ID')
    samples = header[1:]
    require(len(samples) == 3 and len(set(samples)) == 3, 'Expected three unique sample columns: '+str(path))
    values = {}
    for row in rows:
        pid = row[header[0]]
        require(pid.isdigit() and pid not in values, 'Invalid/duplicate activity promoter ID: '+pid)
        vector = np.asarray([float(row[s]) for s in samples], dtype=float)
        require(np.isfinite(vector).all() and (vector >= 0.).all(), 'Missing/invalid/negative activity for '+pid)
        values[pid] = vector
    require(values, 'Empty activity table')
    return samples, values


def load_annotation(path, condition, values):
    header, rows = read_r_tsv(path)
    required = {'promoterId', *IDENTITY, condition+'.mean', condition+'.gene.mean'}
    require(required <= set(header), 'Missing rowData columns: '+str(path))
    annotation = {}
    for row in rows:
        pid = row['promoterId']
        require(pid not in annotation, 'Duplicate annotation promoter ID: '+pid)
        annotation[pid] = row
    require(set(annotation) == set(values), 'rowData/activity ID sets differ for '+condition)
    for pid, row in annotation.items():
        require(math.isclose(float(np.mean(values[pid])), float(row[condition+'.mean']),
                             rel_tol=1e-10, abs_tol=1e-12),
                'Activity is not the promoter .mean for {} / {}'.format(condition, pid))
    return annotation


def identity(row):
    return tuple(row[k] for k in IDENTITY)


def usage_ratio(promoter_mean, gene_mean):
    require(math.isfinite(promoter_mean) and math.isfinite(gene_mean) and
            promoter_mean >= 0. and gene_mean >= 0., 'Invalid usage numerator/denominator')
    if gene_mean == 0.:
        require(promoter_mean == 0., 'Positive promoter activity / zero gene activity')
        return None
    ratio = promoter_mean/gene_mean
    require(0. <= ratio <= 1.+1e-10, 'Promoter usage is outside [0,1]')
    return ratio


def reconstruct_gene_activity(values, annotation, condition):
    """All supplied gene promoters contribute, including nonselected promoters."""
    totals = {}
    for pid, vector in values.items():
        gene = annotation[pid]['geneId']
        totals.setdefault(gene, np.zeros(len(vector), dtype=float))
        totals[gene] += vector
    maximum_error = 0.
    for pid, row in annotation.items():
        expected = float(np.mean(totals[row['geneId']]))
        supplied = float(row[condition+'.gene.mean'])
        maximum_error = max(maximum_error, abs(expected-supplied))
        require(math.isclose(expected, supplied, rel_tol=1e-10, abs_tol=1e-12),
                'Gene denominator does not equal all-promoter activity sum: '+pid)
    return totals, {'gene_n': len(totals), 'all_gene_mean_rows_checked': len(annotation),
                    'maximum_absolute_gene_mean_error': maximum_error}


def prepare(args):
    files = [args.selection] + [args.input_dir/(c+suffix) for c in CONDITIONS
                                for suffix in ('_activity.tsv', '_rowData.tsv')]
    require(all(p.is_file() for p in files), 'One or more input files are missing')
    before = {p: sha(p) for p in files}
    header, selected = read_r_tsv(args.selection)
    require({'promoterId', 'group', *IDENTITY} <= set(header), 'Invalid selected-promoter table')
    require(len(selected) == 366 and len({r['promoterId'] for r in selected}) == 366,
            'Expected 366 unique selected promoters')
    require({r['group'] for r in selected} == {'A', 'B'}, 'Expected groups A and B')
    genes = {}
    for row in selected:
        group = row['group']
        gene_groups = genes.setdefault(row['geneId'], {})
        require(group not in gene_groups, 'Repeated selected gene/group')
        gene_groups[group] = row['promoterId']
    require(len(genes) == 183 and all(set(v) == {'A', 'B'} for v in genes.values()),
            'Expected one A and one B promoter per each of 183 genes')
    samples, activity, annotation, gene_activity, gene_checks = {}, {}, {}, {}, {}
    for condition in CONDITIONS:
        samples[condition], activity[condition] = load_activity(args.input_dir/(condition+'_activity.tsv'))
        annotation[condition] = load_annotation(args.input_dir/(condition+'_rowData.tsv'), condition, activity[condition])
        gene_activity[condition], gene_checks[condition] = reconstruct_gene_activity(
            activity[condition], annotation[condition], condition)
    require(not set(samples['WT']) & set(samples['HNF4A_KO']), 'Samples are reused across conditions')
    require(set(activity['WT']) == set(activity['HNF4A_KO']), 'WT/KO activity ID sets differ')
    for pid in activity['WT']:
        require(identity(annotation['WT'][pid]) == identity(annotation['HNF4A_KO'][pid]),
                'WT/KO promoter annotations disagree: '+pid)
    long_rows, means, mappings = [], [], []
    gene_denominators = {}
    arrays = {g: {c: [] for c in CONDITIONS} for g in ('A', 'B')}
    for row in selected:
        pid, group = row['promoterId'], row['group']
        record = {'group': group, 'promoterId': pid, **{k: row[k] for k in IDENTITY}}
        for condition in CONDITIONS:
            require(pid in annotation[condition], 'Selected promoter missing: {} / {}'.format(condition, pid))
            require(identity(annotation[condition][pid]) == identity(row),
                    'Selected promoter ID belongs to different coordinates/gene: '+pid)
            vector = activity[condition][pid]
            promoter_mean = float(annotation[condition][pid][condition+'.mean'])
            denominator = float(annotation[condition][pid][condition+'.gene.mean'])
            value = usage_ratio(promoter_mean, denominator)
            record[condition+'_mean_activity'] = promoter_mean
            record[condition+'_gene_mean_activity'] = denominator
            record[condition+'_usage'] = value
            record[condition+'_usage_defined'] = value is not None
            record[condition+'_sample_n'] = len(vector)
            gene_vector = gene_activity[condition][row['geneId']]
            key = (condition, row['geneId'])
            if key in gene_denominators:
                require(gene_denominators[key] == denominator, 'A/B gene denominators differ')
            gene_denominators[key] = denominator
            for sample, raw, gene_raw in zip(samples[condition], vector, gene_vector):
                long_rows.append({'group': group, 'promoterId': pid, 'geneId': row['geneId'],
                                  'condition': condition, 'sample': sample,
                                  'absolute_promoter_activity': float(raw),
                                  'all_gene_promoter_activity_sum': float(gene_raw),
                                  'sample_usage_audit_only': usage_ratio(float(raw), float(gene_raw)),
                                  'sample_usage_defined': bool(gene_raw > 0.)})
            mappings.append({'group': group, 'condition': condition, 'selected_promoterId': pid,
                             'matched_promoterId': pid, **{k: row[k] for k in IDENTITY},
                             'identity_verified': True})
        included = all(record[c+'_usage_defined'] for c in CONDITIONS)
        record['included_in_paired_boxplot_and_test'] = included
        undefined = [c for c in CONDITIONS if not record[c+'_usage_defined']]
        record['exclusion_reason'] = '' if included else 'Undefined 0/0 in '+','.join(undefined)
        record['KO_minus_WT_usage'] = record['HNF4A_KO_usage']-record['WT_usage'] if included else None
        if included:
            for condition in CONDITIONS:
                arrays[group][condition].append(record[condition+'_usage'])
        means.append(record)
    arrays = {g: {c: np.asarray(v) for c, v in values.items()} for g, values in arrays.items()}
    require(all(len(values['WT']) == len(values['HNF4A_KO']) > 0 for values in arrays.values()), 'No complete promoter pairs')
    masks_by_gene = {}
    for record in means:
        mask = tuple(record[c+'_usage_defined'] for c in CONDITIONS)
        gene = record['geneId']
        require(gene not in masks_by_gene or masks_by_gene[gene] == mask, 'A/B missingness differs for one gene')
        masks_by_gene[gene] = mask
    require(all(sha(p) == digest for p, digest in before.items()), 'Input changed during loading')
    validation = {'selected_promoters': len(selected), 'paired_genes': len(genes),
                  'promoters_per_group': {'A': 183, 'B': 183}, 'samples': samples,
                  'all_annotation_rows_checked_per_condition': {c: len(annotation[c]) for c in CONDITIONS},
                  'all_activity_rows_reproduce_promoter_mean': True,
                  'selected_ID_gene_TSS_strand_matches': True, 'missing_promoters': 0,
                  'zeros_retained': True, 'extra_log_transform': False,
                  'cross_condition_joint_normalization_verified': False,
                  'usage_formula': 'condition promoter .mean / condition gene .gene.mean',
                  'gene_denominator_includes': 'All promoters of that gene in supplied ProActiv assay',
                  'aggregation': 'Ratio of condition means, NOT average of sample ratios, NOT A/(A+B)',
                  'all_gene_denominator_checks': gene_checks,
                  'box_observation': 'One defined condition usage per promoter, WT/KO complete pairs only',
                  'defined_promoters_by_condition': {g: {c: sum(r['group']==g and r[c+'_usage_defined'] for r in means)
                                                         for c in CONDITIONS} for g in ('A','B')},
                  'complete_pairs_by_group': {g: len(arrays[g]['WT']) for g in ('A','B')},
                  'excluded_pairs_by_group': {g: 183-len(arrays[g]['WT']) for g in ('A','B')},
                  'undefined_zero_over_zero_not_imputed': True,
                  'source_hashes': {str(p.resolve()): v for p, v in before.items()}}
    return arrays, long_rows, means, mappings, validation


def statistics(arrays, method):
    summary, tests = [], []
    for group in ('A', 'B'):
        for condition in CONDITIONS:
            values = arrays[group][condition]
            q1, median, q3 = np.quantile(values, [.25, .5, .75])
            summary.append({'group': group, 'condition': condition, 'paired_promoter_n': len(values),
                            'samples_per_condition': 3, 'zero_usage_n': int(np.count_nonzero(values == 0.)),
                            'mean': float(np.mean(values)), 'median': float(median), 'Q1': float(q1), 'Q3': float(q3),
                            'minimum': float(np.min(values)), 'maximum': float(np.max(values))})
        if method == 'wilcoxon':
            from scipy.stats import wilcoxon
            differences = np.round(arrays[group]['HNF4A_KO']-arrays[group]['WT'], 12)
            if np.count_nonzero(differences) == 0:
                stat, p = 0., 1.
            else:
                result = wilcoxon(differences, alternative='two-sided', zero_method='wilcox',
                                  correction=False, method='approx')
                stat, p = float(result.statistic), float(result.pvalue)
            tests.append({'group': group, 'method': 'paired Wilcoxon signed-rank, two-sided, normal approximation',
                          'pair_unit': 'Same promoter condition usage ratios; NOT biological sample pairs',
                          'paired_promoter_n': len(differences), 'nonzero_difference_n': int(np.count_nonzero(differences)),
                          'difference_rounding_decimals': 12, 'statistic': stat, 'P': p,
                          'interpretation': 'Exploratory across-promoter shift, not genotype inference over independent mice'})
    if tests:
        order = np.argsort([t['P'] for t in tests])
        adjusted, running = [1.]*len(tests), 1.
        for rank in range(len(tests)-1, -1, -1):
            idx = int(order[rank])
            running = min(running, tests[idx]['P']*len(tests)/(rank+1))
            adjusted[idx] = min(running, 1.)
        for test, q in zip(tests, adjusted):
            test['BH_adjusted_P'] = q
    return summary, tests


def draw(arrays, tests, output_dir):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
        font = 'Arial'
    except ValueError:
        font = 'DejaVu Sans'
    mpl.rcParams.update({'font.family': font, 'font.size': 7., 'pdf.fonttype': 42,
                         'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
                         'text.usetex': False})
    ymax = 1.25 if tests else 1.05
    exported = []
    for group in ('A', 'B'):
        fig = plt.figure(figsize=tuple(mm/25.4 for mm in FIGURE_MM), dpi=300)
        ax = fig.add_axes([.30, .16, .67, .72])
        boxes = ax.boxplot([arrays[group][c] for c in CONDITIONS], positions=[1, 2], widths=.32,
                           patch_artist=True, whis=1.5, showfliers=True,
                           boxprops={'linewidth': .6, 'edgecolor': '#666666'},
                           medianprops={'linewidth': .9, 'color': '#444444'},
                           whiskerprops={'linewidth': .6, 'color': '#999999'},
                           capprops={'linewidth': .6, 'color': '#999999'},
                           flierprops={'marker': 'o', 'markersize': 2.6, 'markerfacecolor': 'none',
                                       'markeredgecolor': '#666666', 'markeredgewidth': .6})
        for patch, condition in zip(boxes['boxes'], CONDITIONS):
            patch.set_facecolor(COLORS[condition])
        ax.set_xlim(.5, 2.5)
        ax.set_ylim(-.02, ymax)
        ax.set_yticks([0., .25, .5, .75, 1.], ['0', '0.25', '0.50', '0.75', '1.00'])
        ax.set_xticks([1, 2], ['WT', 'HNF4A_KO'])
        ax.set_ylabel('Promoter usage', fontsize=8., labelpad=3.)
        ax.tick_params(axis='both', labelsize=7., length=2., width=.6, pad=2.)
        ax.set_xlabel('')
        title = fig.text(.635, .965, 'Group {} promoters'.format(group), ha='center', va='top', fontsize=8.)
        note = ax.text(.5, 1.02, 'n = {} paired promoters'.format(len(arrays[group]['WT'])),
                       transform=ax.transAxes, ha='center', fontsize=6.0)
        extra = [fig.text(.635, .085, 'Ratio of 3-sample means', ha='center', fontsize=5.1)]
        if tests:
            test = next(t for t in tests if t['group'] == group)
            y = 1.10
            ax.plot([1, 1, 2, 2], [y-.02*ymax, y, y, y-.02*ymax], color='#444444', linewidth=.7)
            extra.append(ax.text(1.5, y+.02*ymax, 'q = {:.2g}'.format(test['BH_adjusted_P']),
                                 ha='center', va='bottom', fontsize=6.5))
            extra.append(fig.text(.635, .045, 'Paired-promoter Wilcoxon (BH)', ha='center', fontsize=5.1))
        fig.canvas.draw()
        for text in [title, note, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *extra]:
            if not text.get_visible():
                continue
            box = text.get_window_extent(fig.canvas.get_renderer())
            require(box.x0 >= -.5 and box.y0 >= -.5 and box.x1 <= fig.bbox.width+.5 and box.y1 <= fig.bbox.height+.5,
                    'Text clipping in Group {}: {}'.format(group, text.get_text()))
        path = output_dir/'Group_{}_WT_vs_HNF4A_KO_usage_boxplot.pdf'.format(group)
        with path.open('xb') as handle:
            fig.savefig(handle, format='pdf', bbox_inches=None)
        plt.close(fig)
        exported.append({'group': group, 'path': str(path.resolve()), 'sha256': sha(path),
                         'size_mm': list(FIGURE_MM), 'shared_y_limits': [-.02, ymax],
                         'font': font, 'text_bounds_verified': True})
    return exported


def write_tsv(path, rows, fields=None):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields if fields is not None else list(rows[0]),
                                delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    class Checks(unittest.TestCase):
        def test_missing_R_header(self):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'x.tsv'
                path.write_text('S1\tS2\tS3\n45\t1\t2\t3\n100\t0\t0\t0\n')
                samples, values = load_activity(path)
                self.assertEqual(samples, ['S1', 'S2', 'S3'])
                self.assertEqual(set(values), {'45', '100'})
                self.assertTrue(np.array_equal(values['100'], [0., 0., 0.]))
        def test_invalid_value_rejected(self):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'x.tsv'
                path.write_text('S1\tS2\tS3\n45\t1\tnan\t3\n')
                with self.assertRaises(ValueError):
                    load_activity(path)
        def test_duplicate_ID_rejected(self):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'x.tsv'
                path.write_text('S1\tS2\tS3\n45\t1\t2\t3\n45\t0\t0\t0\n')
                with self.assertRaises(ValueError):
                    load_activity(path)
        def test_annotation_extra_index(self):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'x.tsv'
                path.write_text('promoterId\tgeneId\tseqnames\tstart\tstrand\tWT.mean\tWT.gene.mean\n'
                                '1\t45\tG\tchr1\t100\t+\t2\t100\n')
                rows = load_annotation(path, 'WT', {'45': np.array([1., 2., 3.])})
                self.assertEqual(rows['45']['geneId'], 'G')
                self.assertEqual(rows['45']['WT.gene.mean'], '100')
        def test_usage_zero_policy(self):
            self.assertIsNone(usage_ratio(0., 0.))
            self.assertEqual(usage_ratio(0., 10.), 0.)
            with self.assertRaises(ValueError):
                usage_ratio(1., 0.)
        def test_ratio_of_means_not_mean_of_ratios(self):
            promoter, gene = np.array([1., 0., 9.]), np.array([1., 1., 10.])
            ratio = usage_ratio(float(np.mean(promoter)), float(np.mean(gene)))
            self.assertAlmostEqual(ratio, 10./12.)
            self.assertNotAlmostEqual(ratio, float(np.mean(promoter/gene)))
        def test_all_gene_promoters_in_denominator(self):
            values = {'1': np.array([1.,2.,3.]), '2': np.array([3.,2.,1.]), '3': np.array([2.,2.,2.])}
            annotation = {'1': {'geneId':'G', 'WT.gene.mean':'6'},
                          '2': {'geneId':'G', 'WT.gene.mean':'6'},
                          '3': {'geneId':'G', 'WT.gene.mean':'6'}}
            total, check = reconstruct_gene_activity(values, annotation, 'WT')
            self.assertTrue(np.array_equal(total['G'], [6.,6.,6.]))
            self.assertEqual(check['gene_n'], 1)
            self.assertAlmostEqual(usage_ratio(2., 6.), 1./3.)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    require(result.wasSuccessful(), 'Self-tests failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results_usage_wilcoxon')
    parser.add_argument('--test', choices=('none', 'wilcoxon'), default='wilcoxon')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    arrays, raw, means, mappings, validation = prepare(args)
    summary, tests = statistics(arrays, args.test)
    print(json.dumps({'validation': validation, 'summary': summary, 'tests': tests}, indent=2), flush=True)
    if args.check_only:
        print('CHECK_ONLY_PASSED: no output written', flush=True)
        return
    require(not args.output_dir.exists(), 'Output exists; select a new --output-dir: '+str(args.output_dir))
    args.output_dir.mkdir(parents=True)
    write_tsv(args.output_dir/'selected_promoter_sample_usage_audit.tsv', raw)
    write_tsv(args.output_dir/'selected_promoter_condition_usage.tsv', means)
    write_tsv(args.output_dir/'excluded_undefined_usage_promoters.tsv',
              [r for r in means if not r['included_in_paired_boxplot_and_test']], fields=list(means[0]))
    write_tsv(args.output_dir/'promoter_identity_mapping.tsv', mappings)
    write_tsv(args.output_dir/'boxplot_summary.tsv', summary)
    if tests:
        write_tsv(args.output_dir/'exploratory_paired_promoter_tests.tsv', tests)
    figures = draw(arrays, tests, args.output_dir)
    require(all(sha(Path(p)) == value for p, value in validation['source_hashes'].items()), 'An input file changed')
    manifest = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'script_sha256': sha(Path(__file__)), 'validation': validation,
                'figure_contract': {'question': 'WT vs HNF4A-KO usage of fixed Group A/B promoters',
                                    'archetype': 'two matched quantitative distribution plots',
                                    'backend': 'Python/matplotlib', 'format': 'PDF',
                                    'transform': 'condition promoter .mean / condition gene .gene.mean',
                                    'observation': 'One defined usage ratio per promoter per condition, complete WT/KO pairs only',
                                    'box': 'Median and IQR', 'whiskers': '1.5 IQR',
                                    'outliers': 'Hollow circles; all observations retained'},
                'test_option': args.test, 'tests': tests, 'summary': summary, 'figures': figures,
                'limitations': ['Promoter means are features, not 183 independent mice.',
                                'Three sample columns per condition; no sample/mouse pairing assumed.',
                                'Cross-condition joint normalization is not established by separately exported WT/KO activity files; plots are exploratory.',
                                'Usage is the established ProActiv activity ratio, not a linear read-count fraction.',
                                'Promoters with undefined 0/0 usage in either condition are excluded from BOTH boxes and the paired test; all exclusions are reported.',
                                'Source values are not imputed or additionally logged; source files and earlier results remain unchanged.']}
    with (args.output_dir/'analysis_manifest.json').open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print('COMPLETE: '+str(args.output_dir.resolve()), flush=True)


if __name__ == '__main__':
    main()
