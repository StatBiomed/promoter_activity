#!/usr/bin/env python3
"""Mouse Liver: HNF4A/Group A and E2F4/Group B mean promoter USAGE per sample.

Sample matching follows the USER-SPECIFIED convention, not an independently
verified sample-ID crosswalk: read the supplied SDRF in physical row order;
within each organ/developmental stage number UNIQUE Source Names .1, .2, ... .
Do not renumber after removing a missing expression/activity sample. In
particular, if Liver.e11.5.1 is absent, exclude that library, not shift .2 to .1.

Figure contract (Python; quantitative association / discovery):
  - Each point is ONE matched Mouse Liver sequencing library.
  - Start with the CURRENT 183 selected promoters in each group; keep zeros.
  - Per-sample usage = promoter activity / all-promoter gene activity sum.
    Numerator: *_activity.tsv. Denominator: *_exp.csv (geneExpression), checked
    against sums from the full promoter assay and against rowData .gene.mean.
    No unlogging, extra log, pseudocount, or A/(A+B) normalization is applied.
  - x = arithmetic mean of individual promoter usages in that ONE sample.
    It is not ratio(sum numerator/sum denominator), not a stage-mean ratio
    repeated for each sample, and not log2(mean usage + 1).
  - 0/positive=0; 0/0=NA. The primary cohort is fixed across all matched samples
    (USER CONFIRMED): only promoters whose gene is active in every such sample.
    All 183 originals stay in the audit; available-case results are sensitivity
    analysis only, not a hidden replacement for the primary fixed cohort.
  - y = log2(TF gene RPKM + 1), from original Mouse_rpkm.txt, not ortholog CSV.
  - Exactly two requested pairs: HNF4A--Group A and E2F4--Group B. TF gene IDs
    reuse the validated mouse mapping table from the preceding analysis.
  - Two-sided Spearman rho/P on complete sample pairs; include valid zeros.
    BH correction across these TWO planned comparisons, separately per cohort.
  - Blue OLS line and grey pointwise 95% mean-response CI are descriptive
    on raw-usage x and log2(RPKM+1) y, NOT a Spearman CI or proof of regulation.
  - Across-stage association can reflect development/cell composition; sample
    numbering is an explicit assumption and is recorded in all provenance.
  - 5 x 4 inch vector PDF only, editable TrueType text, labels14/ticks12.
    Missing expression rows remain NA; no artificial zero-filled figures.

Dependencies: Python >=3.10, numpy, scipy, pandas, matplotlib.
Default output: 5.results/Mouse_Liver_TF_usage_correlations, NEW only.
Run with PY310/bin/python; --self-test and --check-only create no outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path('/mnt/ruiyanhou/nfs_share2')
ROOT = BASE / 'proactiv_redo/13.promoter_usage'
STAGES = [f'{n}.5' for n in range(10, 19)] + ['0dpb', '3dpb', '2wpb', '4wpb', '9wpb']
POST = {'0dpb': 'P0', '3dpb': 'P3', '2wpb': 'P14', '4wpb': 'P28', '9wpb': 'P63'}
REQUESTED_TFS = {'A': ['HNF4A'], 'B': ['E2F4']}
MATCH_RULE = ('USER-SPECIFIED: first occurrence of each unique Source Name in SDRF; '
              'counter resets within organ and developmental stage; no renumbering '
              'after unmatched samples are excluded; NOT independently verified')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        rows = list(reader)
    require(rows and all(None not in row for row in rows), f'Invalid TSV: {path}')
    return rows


def write_tsv(path, rows, fields=None):
    require(bool(rows), f'Empty output: {path}')
    fields = fields or list(rows[0])
    with io.StringIO(newline='') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', lineterminator='\n', fieldnames=fields,
                                extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ('NA' if value is None or isinstance(value, (float, np.floating))
                                   and not np.isfinite(value) else value)
                             for key, value in row.items() if key in fields})
        content = handle.getvalue().encode('utf-8')
    if Path(path).exists():
        require(Path(path).read_bytes() == content, f'Existing table differs; nothing overwritten: {path}')
    else:
        with Path(path).open('xb') as handle:
            handle.write(content)


def expression_alias(stage, number):
    return f"Liver.{POST.get(stage, 'e' + stage)}.{number}"


def source_from_activity(name):
    value = re.sub(r'^X(?=\d)', '', name)
    value = re.sub(r'\.sorted(?:\.bam)?$', '', value)
    # Five activity headers retain an extra processing suffix (sTSm/sTSmc).
    # Preserve library number and ALL species/organ/stage/sex fields; demand
    # unique normalized IDs and exact membership in the SDRF below.
    return re.sub(r'^(\d+)sTS(?:mc|m)(?=\.Mouse\.)', r'\1sTS', value)


def number_sdrf(rows, header):
    """Number unique samples before filtering to expression overlap."""
    cols = ['Source Name', 'Characteristics[organism]', 'Characteristics[organism part]',
            'Characteristics[sex]', 'Comment[ENA_RUN]']
    require(all(header.count(col) == 1 for col in cols), 'Missing/duplicate required SDRF column')
    indices = {col: header.index(col) for col in cols}
    seen, counts, result = {}, Counter(), []
    for line_number, row in enumerate(rows, 2):
        require(len(row) == len(header), f'SDRF width mismatch on line {line_number}')
        source = row[indices['Source Name']].strip()
        if row[indices['Characteristics[organism part]']].strip().lower() != 'liver':
            continue
        require(row[indices['Characteristics[organism]']].strip() == 'Mus musculus',
                f'SDRF not Mouse: {source}')
        match = re.fullmatch(r'\d+sTS\.Mouse\.Liver\.(.+)\.(Male|Female)', source)
        require(match is not None and match[1] in STAGES, f'Unexpected Mouse Liver source: {source}')
        stage, filename_sex = match.groups()
        sex = row[indices['Characteristics[sex]']].strip().capitalize()
        require(sex in ['Male', 'Female'], f'Unexpected SDRF sex: {source}')
        run = row[indices['Comment[ENA_RUN]']].strip()
        if source in seen:
            require(seen[source]['stage'] == stage and seen[source]['sex'] == sex
                    and seen[source]['ENA_RUN'] == run, f'Conflicting duplicate Source Name: {source}')
            seen[source]['sdrf_duplicate_rows'] += 1
            continue
        counts[('Liver', stage)] += 1
        number = counts[('Liver', stage)]
        item = dict(source_name=source, sdrf_line=line_number, organ='Liver', stage=stage,
                    stage_label='E' + stage if stage not in POST else stage,
                    sex=sex, filename_sex=filename_sex, sex_name_mismatch=(sex != filename_sex),
                    ENA_RUN=run, within_stage_order=number,
                    expression_sample=expression_alias(stage, number), sdrf_duplicate_rows=0)
        seen[source] = item
        result.append(item)
    require(set(item['stage'] for item in result) == set(STAGES), 'Incomplete SDRF Liver stages')
    require(len({item['expression_sample'] for item in result}) == len(result), 'Nonunique SDRF aliases')
    return result


def load_sdrf(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        return number_sdrf(reader, next(reader))


def load_selection(path):
    rows = read_tsv(path)
    require(len(rows) == 366 and len({row['promoterId'] for row in rows}) == 366,
            'Expected CURRENT 366 unique promoters')
    groups = {group: [row['promoterId'] for row in rows if row['group'] == group] for group in ['A', 'B']}
    genes = {group: {row['geneId'] for row in rows if row['group'] == group} for group in groups}
    require(all(len(ids) == 183 for ids in groups.values())
            and genes['A'] == genes['B'] and len(genes['A']) == 183,
            'Expected one A and one B promoter for each of 183 genes')
    require(all(row['internalPromoter'].lower() == 'false' for row in rows), 'Internal promoter selected')
    return groups


def load_expression(path, mapping):
    wanted = {item['gene_id'] for item in mapping.values() if item['gene_id']}
    values = {}
    with path.open(newline='') as handle:
        reader = csv.reader(handle, delimiter=' ', skipinitialspace=True)
        header = next(reader)
        require(header[0] == 'Names' and len(header) == len(set(header)), 'Unexpected RPKM header')
        indices = [i for i, name in enumerate(header) if name.startswith('Liver.')]
        samples = [header[i] for i in indices]
        for row in reader:
            if not row:
                continue
            require(len(row) == len(header), f'RPKM row width: {row[0]}')
            gid = re.sub(r'\.\d+$', '', row[0])
            if gid not in wanted:
                continue
            require(gid not in values, f'Duplicate expression gene: {gid}')
            data = np.array([float(row[i]) for i in indices])
            require(np.isfinite(data).all() and (data >= 0).all(), f'Invalid RPKM values: {gid}')
            values[gid] = dict(zip(samples, data))
    for item in mapping.values():
        item['expression_status'] = ('observed' if item['gene_id'] in values else 'missing_expression_row') \
            if item['gene_id'] else item['mapping_status']
    return samples, values


def load_requested_mapping(path):
    rows = read_tsv(path)
    result = {}
    for tf in ('HNF4A', 'E2F4'):
        found = [r for r in rows if r['TF'] == tf]
        require(len(found) == 1, 'Expected one validated mapping for '+tf)
        item = dict(found[0])
        require(item['mapping_status'] == 'mapped' and item['gene_id'].startswith('ENSMUSG')
                and item['gene_symbol'].upper() == tf, 'Invalid mouse mapping for '+tf)
        result[tf] = item
    return result


def sample_usage(numerator, denominator):
    a, g = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    require(a.shape == g.shape and np.isfinite(a).all() and np.isfinite(g).all()
            and (a >= 0).all() and (g >= 0).all(), 'Invalid usage arrays')
    require(not ((g == 0) & (a != 0)).any(), 'Positive numerator / zero denominator')
    result = np.divide(a,g,out=np.full(a.shape,np.nan),where=g>0)
    finite = result[np.isfinite(result)]
    require(((finite >= 0) & (finite <= 1.+1e-10)).all(), 'Usage outside [0,1]')
    return result


def load_usage_inputs(folder, groups, selection):
    wanted = groups['A'] + groups['B']
    selected = {r['promoterId']: r for r in read_tsv(selection)}
    output, inputs, selected_rows, checks = {}, [], [], []
    for stage in STAGES:
        prefix = folder / f'Mouse_Liver_{stage}'
        apath, gpath, rpath = [Path(str(prefix)+s) for s in ('_activity.tsv','_exp.csv','_rowData.tsv')]
        actual = pd.read_csv(apath, sep='\t', index_col=0)
        genes = pd.read_csv(gpath, index_col=0)
        annotation = pd.read_csv(rpath, sep='\t', index_col=0, dtype=str).set_index('promoterId', verify_integrity=True)
        for frame in (actual, genes, annotation):
            frame.index = frame.index.astype(str)
            require(frame.index.is_unique and frame.columns.is_unique, f'Duplicate IDs: {prefix}')
        require(set(actual.index) == set(genes.index) == set(annotation.index), 'Assay/annotation ID mismatch')
        require(list(actual.columns) == list(genes.columns), 'Promoter/gene sample columns differ')
        genes, annotation = genes.loc[actual.index], annotation.loc[actual.index]
        a, g = actual.to_numpy(dtype=float), genes.to_numpy(dtype=float)
        require(np.isfinite(a).all() and np.isfinite(g).all() and (a >= 0).all() and (g >= 0).all(),
                'Missing/invalid promoter or gene assay value')
        gene_totals = actual.groupby(annotation.geneId, sort=False).sum()
        reconstructed = gene_totals.loc[annotation.geneId].to_numpy(dtype=float)
        require(np.allclose(reconstructed, g, rtol=1e-10, atol=1e-12),
                f'{stage}: *_exp.csv is not the all-promoter gene-expression denominator')
        require(np.allclose(a.mean(axis=1), pd.to_numeric(annotation[f'Mouse_Liver_{stage}.mean']),
                            rtol=1e-10, atol=1e-12), 'Promoter assay does not reproduce .mean')
        require(np.allclose(g.mean(axis=1), pd.to_numeric(annotation[f'Mouse_Liver_{stage}.gene.mean']),
                            rtol=1e-10, atol=1e-12), 'Gene assay does not reproduce .gene.mean')
        for pid in wanted:
            require(pid in actual.index, 'Selected promoter missing: '+pid)
            require(all(annotation.loc[pid, key] == selected[pid][key]
                        for key in ('geneId','seqnames','start','strand')), 'Selected promoter identity changed: '+pid)
        numerator = actual.loc[wanted]
        denominator = genes.loc[wanted]
        usage = pd.DataFrame(sample_usage(numerator.to_numpy(), denominator.to_numpy()),
                             index=numerator.index,columns=numerator.columns)
        u = usage.to_numpy(dtype=float)
        require(not np.isinf(u).any() and np.all((u[np.isfinite(u)] >= 0.) & (u[np.isfinite(u)] <= 1.+1e-10)),
                'Usage outside [0,1]')
        for column in actual.columns:
            source = source_from_activity(column)
            require(source not in output, f'Duplicate usage sample: {column}')
            require(f'.Mouse.Liver.{stage}.' in source, f'Activity name/stage mismatch: {column}')
            item = dict(activity_sample=column, source_name=source, stage=stage)
            for group in ['A', 'B']:
                vector = usage.loc[groups[group], column].to_numpy(dtype=float)
                item[f'{group}_usage_vector'] = vector
                selected_rows += [dict(group=group, promoterId=pid, stage=stage, source_name=source,
                                       geneId=selected[pid]['geneId'], activity_sample=column,
                                       promoter_activity=float(numerator.loc[pid,column]),
                                       gene_activity=float(denominator.loc[pid,column]),
                                       promoter_usage=float(value), usage_defined=bool(np.isfinite(value)))
                                  for pid, value in zip(groups[group], vector)]
            output[source] = item
        inputs.extend([apath,gpath,rpath])
        checks.append(dict(stage=stage, promoter_rows=len(actual), gene_n=len(gene_totals), sample_n=len(actual.columns),
                           all_gene_denominators_match_all_promoter_sum=True,
                           max_abs_gene_denominator_error=float(np.max(np.abs(g-reconstructed))),
                           promoter_assay_matches_rowData_mean=True, gene_assay_matches_rowData_gene_mean=True))
        print(f'USAGE_INPUT: {stage}, {len(actual.columns)} samples; all-gene denominator verified', flush=True)
    return output, inputs, selected_rows, checks


def aggregate_usage(groups, matched, usage, selected, mode):
    by_id = {r['promoterId']:r for r in read_tsv(selected)}
    cohort, profiles = [], []
    masks = {}
    for group in ('A','B'):
        array = np.column_stack([usage[r['source_name']][group+'_usage_vector'] for r in matched])
        mask = np.isfinite(array).all(axis=1)
        require(mask.any(), f'{group}: no promoters defined in all matched samples')
        masks[group] = mask
        for i,pid in enumerate(groups[group]):
            cohort.append(dict(group=group,promoterId=pid,geneId=by_id[pid]['geneId'],
                               fixed_cohort_included=bool(mask[i]),
                               undefined_usage_matched_sample_n=int((~np.isfinite(array[i])).sum()),
                               exclusion_reason='' if mask[i] else '0/0 gene activity in at least one matched sample'))
    common_genes = [{by_id[pid]['geneId'] for pid, keep in zip(groups[g],masks[g]) if keep} for g in ('A','B')]
    require(common_genes[0] == common_genes[1], 'A/B common gene cohorts differ')
    for source,item in usage.items():
        for group in ('A','B'):
            vector = item[group+'_usage_vector']
            valid = np.isfinite(vector)
            fixed_vector = vector[masks[group]]
            available = float(np.mean(vector[valid])) if valid.any() else np.nan
            fixed = float(np.mean(fixed_vector)) if np.isfinite(fixed_vector).all() else np.nan
            item[group+'_available_mean_usage'] = available
            item[group+'_fixed_mean_usage'] = fixed
            item[group+'_valid_promoter_n'] = int(valid.sum())
            item[group+'_fixed_promoter_n'] = int(masks[group].sum())
            item[group+'_mean_usage'] = fixed if mode == 'fixed' else available
            profiles.append(dict(group=group,source_name=source,stage=item['stage'],activity_sample=item['activity_sample'],
                                 selected_promoter_n=len(groups[group]),valid_promoter_n=int(valid.sum()),
                                 fixed_cohort_promoter_n=int(masks[group].sum()),fixed_cohort_mean_usage=fixed,
                                 available_case_mean_usage=available,primary_cohort_mode=mode,
                                 primary_mean_usage=item[group+'_mean_usage']))
    return cohort, profiles


def match_samples(metadata, expression_samples, activity):
    expression_set = set(expression_samples)
    meta_sources = {item['source_name'] for item in metadata}
    meta_aliases = {item['expression_sample'] for item in metadata}
    require(not (set(activity) - meta_sources), 'Activity libraries absent from SDRF')
    audit, matched = [], []
    for record in metadata:
        source, alias = record['source_name'], record['expression_sample']
        activity_present, expression_present = source in activity, alias in expression_set
        if activity_present:
            require(activity[source]['stage'] == record['stage'], f'Stage mismatch for {source}')
        status = ('matched' if activity_present and expression_present else
                  'no_expression_sample' if activity_present else
                  'no_activity_sample' if expression_present else 'no_activity_or_expression_sample')
        row = dict(**record, activity_sample=activity[source]['activity_sample'] if activity_present else None,
                   expression_sample_present=expression_present, activity_sample_present=activity_present,
                   matching_status=status)
        audit.append(row)
        if status == 'matched':
            matched.append(row)
    for alias in sorted(expression_set - meta_aliases):
        audit.append(dict(source_name=None, expression_sample=alias, matching_status='no_SDRF_alias',
                          expression_sample_present=True, activity_sample_present=False))
    require(len(matched) >= 10, 'Fewer than 10 matched sample pairs')
    return audit, matched


def bh_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid], kind='stable')]
    if len(order):
        adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
        result[order] = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1)
    return result


def calculate(groups, tfs, mapping, expression, matched, activity, cohort_mode='fixed'):
    details, tests, plot_data = [], [], {}
    for group in ['A', 'B']:
        for rank, tf in enumerate(tfs[group], 1):
            gid = mapping[tf]['gene_id']
            sample_values = expression.get(gid, {})
            x, y = [], []
            for sample in matched:
                item = activity[sample['source_name']]
                raw_x = item[f'{group}_{cohort_mode}_mean_usage']
                raw_y = sample_values.get(sample['expression_sample'], np.nan)
                valid = math.isfinite(raw_x) and math.isfinite(raw_y)
                lx, ly = raw_x, np.log2(raw_y + 1)
                reason = ('included' if valid else mapping[tf]['expression_status']
                          if not math.isfinite(raw_y) else 'undefined_group_usage')
                details.append(dict(group=group, TF=tf, TF_rank=rank, gene_id=gid,
                                    source_name=sample['source_name'], expression_sample=sample['expression_sample'],
                                    stage=sample['stage'], sex=sample['sex'],
                                    promoter_n=len(groups[group]), valid_promoter_n=item[f'{group}_valid_promoter_n'],
                                    used_promoter_n=item[f'{group}_fixed_promoter_n'] if cohort_mode=='fixed'
                                                    else item[f'{group}_valid_promoter_n'],
                                    cohort_mode=cohort_mode, mean_promoter_usage=raw_x, TF_RPKM=raw_y,
                                    x_usage_fraction=float(lx), log2_TF_RPKM_plus_1=float(ly),
                                    included=valid, status=reason))
                if valid:
                    x.append(lx)
                    y.append(ly)
            x, y = np.array(x), np.array(y)
            status = ('ok' if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0 else
                      mapping[tf]['expression_status'] if len(x) == 0 else 'constant_or_insufficient_pairs')
            rho, p = stats.spearmanr(x, y) if status == 'ok' else (np.nan, np.nan)
            tests.append(dict(group=group, TF=tf, TF_rank=rank, gene_id=gid,
                              gene_symbol=mapping[tf]['gene_symbol'], n=len(x),
                              Spearman_rho=float(rho), P_value=float(p), status=status,
                              matched_sample_n=len(matched), selected_group_promoter_n=len(groups[group]),
                              fixed_cohort_promoter_n=activity[matched[0]['source_name']][group+'_fixed_promoter_n'],
                              cohort_mode=cohort_mode))
            plot_data[(group, tf)] = (x, y)
    adjusted = bh_adjust([row['P_value'] for row in tests])
    for row, q in zip(tests, adjusted):
        row['BH_q_across_two_requested_comparisons'] = float(q)
    return details, tests, plot_data


def build_figure(group, tf, row, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'pdf.fonttype': 42,
                         'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                         'legend.fontsize': 12, 'axes.linewidth': .8, 'pdf.compression': 0})
    x, y = data
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=.20, right=.94, bottom=.24, top=.81)
    fit = stats.linregress(x, y)
    grid = np.linspace(float(x.min()), float(x.max()), 180)
    fitted = fit.intercept + fit.slope * grid
    residual_sd = np.sqrt(np.sum((y - fit.intercept - fit.slope * x) ** 2) / (len(x) - 2))
    ci = stats.t.ppf(.975, len(x) - 2) * residual_sd * np.sqrt(
        1 / len(x) + (grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    ax.fill_between(grid, fitted - ci, fitted + ci, color='#B7BBC0', alpha=.55, linewidth=0, zorder=1)
    ax.plot(grid, fitted, color='#4168AD', linewidth=2, zorder=2)
    ax.scatter(x, y, color='#202020', s=16, alpha=.85, edgecolors='none', zorder=3)
    fig.text(.57,.96,f'Mouse Liver | Group {group} | {tf}',ha='center',va='top',fontsize=13)
    p = row['P_value']
    p_label = f'{p:.3g}' if p >= .001 else f'{p:.2e}'
    q = row['BH_q_across_two_requested_comparisons']
    q_label = f'{q:.3g}' if q >= .001 else f'{q:.2e}'
    ax.text(.5,1.025,f"Spearman ρ = {row['Spearman_rho']:.2f}   n = {row['n']}\nP = {p_label}   BH q = {q_label}",
            transform=ax.transAxes,ha='center',va='bottom',fontsize=10.5)
    ax.set_xlabel(f'Group {group} mean promoter usage\n(fraction)',fontsize=14)
    ax.set_ylabel(f'{tf} expression\n' + r'$\log_2$(RPKM + 1)', fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.tick_params(axis='both', labelsize=12, width=.8, length=3.5)
    for spine in ax.spines.values():
        spine.set_color('#555555')
        spine.set_linewidth(.8)
    ax.margins(x=.07, y=.13)
    fig.text(.57,.02,f"Fixed common cohort: {row['fixed_cohort_promoter_n']} promoters" if row['cohort_mode']=='fixed'
             else 'Available-case sensitivity cohort',ha='center',va='bottom',fontsize=9.)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    # MaxNLocator also creates out-of-view ticks that Matplotlib does NOT draw.
    # Check only rendered ticks, not those outside the displayed axis limits.
    xlabels = [label for value, label in zip(ax.get_xticks(), ax.get_xticklabels())
               if ax.get_xlim()[0] <= value <= ax.get_xlim()[1]]
    ylabels = [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels())
               if ax.get_ylim()[0] <= value <= ax.get_ylim()[1]]
    for artist in [*fig.texts, ax.xaxis.label, ax.yaxis.label, *ax.texts, *xlabels, *ylabels]:
        box = artist.get_window_extent(renderer)
        require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= width + 1 and box.y1 <= height + 1,
                f'Clipped text: group {group}, TF {tf}, {artist.get_text()}')
    return fig


def self_test():
    # Check duplicate handling and stage-local counters, regardless of row order.
    header = ['Source Name', 'Characteristics[organism]', 'Characteristics[organism part]',
              'Characteristics[sex]', 'Comment[ENA_RUN]']
    rows = []
    for stage in STAGES:
        rows.append([f'100sTS.Mouse.Liver.{stage}.Male', 'Mus musculus', 'liver', 'male', 'ERR1'])
    rows += [rows[0].copy(), ['101sTS.Mouse.Liver.10.5.Female', 'Mus musculus', 'liver', 'female', 'ERR2']]
    numbered = number_sdrf(rows, header)
    require(numbered[0]['expression_sample'] == 'Liver.e10.5.1', 'Wrong first alias')
    require(numbered[-1]['expression_sample'] == 'Liver.e10.5.2', 'Duplicate incorrectly counted')
    require(numbered[0]['sdrf_duplicate_rows'] == 1, 'Duplicate audit missing')
    require(numbered[9]['expression_sample'] == 'Liver.P0.1', 'Wrong postnatal alias')
    activity = {r['source_name']: dict(stage=r['stage'], activity_sample='X' + r['source_name'] + '.sorted')
                for r in numbered}
    expressions = [r['expression_sample'] for r in numbered if r['expression_sample'] != 'Liver.e10.5.1']
    audit, matched = match_samples(numbered, expressions, activity)
    require(audit[0]['matching_status'] == 'no_expression_sample'
            and matched[-1]['expression_sample'] == 'Liver.e10.5.2', 'Missing .1 shifted .2')
    np.testing.assert_allclose(bh_adjust([.01, .04, .03, np.nan])[:3], [.03, .04, .04])
    require(np.isnan(bh_adjust([.01, np.nan])[1]), 'NA P-value was imputed')
    require(source_from_activity('X2496sTS.Mouse.Liver.10.5.Male.sorted') == '2496sTS.Mouse.Liver.10.5.Male',
            'R sample-name normalization failed')
    require(source_from_activity('X2225sTSmc.Mouse.Liver.14.5.Male.sorted') == '2225sTS.Mouse.Liver.14.5.Male'
            and source_from_activity('X1958sTSm.Mouse.Liver.9wpb.Male.sorted') == '1958sTS.Mouse.Liver.9wpb.Male',
            'Explicit processing-suffix normalization failed')
    u = sample_usage([0.,0.,1.],[1.,0.,2.])
    require(u[0]==0 and np.isnan(u[1]) and u[2]==.5,'Usage zero/NA policy failed')
    require(np.mean(sample_usage([1.,0.],[1.,9.]))==.5,'Mean usage is not mean of individual ratios')
    require(not np.isclose(.5,1./10.),'Ratio of sums confused with mean ratios')
    try:
        sample_usage([1.],[0.])
    except ValueError:
        pass
    else:
        raise ValueError('Positive numerator/zero denominator not rejected')
    with tempfile.TemporaryDirectory() as folder:
        selected = Path(folder)/'selected.tsv'
        selected.write_text('promoterId\tgeneId\na1\tG1\na2\tG2\nb1\tG1\nb2\tG2\n')
        groups = {'A':['a1','a2'],'B':['b1','b2']}
        usage = {s:dict(activity_sample=s,stage='10.5',
                        A_usage_vector=np.array(a),B_usage_vector=np.array(b)) for s,a,b in
                 [('S1',[.1,np.nan],[.9,np.nan]),('S2',[.2,.3],[.8,.7]),('S3',[np.nan,.3],[np.nan,.7])]}
        cohort,_ = aggregate_usage(groups,[{'source_name':'S1'},{'source_name':'S2'}],usage,selected,'fixed')
        require(sum(r['fixed_cohort_included'] for r in cohort)==2,'Fixed cohort ignores only unmatched samples')
        require(usage['S2']['A_fixed_promoter_n']==1 and usage['S2']['A_mean_usage']==.2,'Fixed mean masking failed')
        require(usage['S2']['A_available_mean_usage']==.25,'Available-case sensitivity mean failed')
    print('SELF_TEST_OK: SDRF order, duplicates, missing aliases, naming, BH, usage zero policy, mean of ratios')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sdrf', type=Path, default=BASE / 'RNA_seq_organ_species/bam_file/mouse/E-MTAB-6798.sdrf.txt')
    parser.add_argument('--selected', type=Path, default=ROOT / '1.results/Mouse_Liver/Mouse_Liver_selected_promoters.tsv')
    parser.add_argument('--activity-dir', type=Path, default=BASE / 'proactiv_redo/1.proactiv_result/Mouse')
    parser.add_argument('--expression', type=Path, default=BASE / 'RNA_seq_organ_species/gene_expression/Mouse_rpkm.txt')
    parser.add_argument('--tf-mapping',type=Path,default=ROOT/'4.results/Mouse_Liver_TF_activity_correlations/TF_gene_mapping.tsv')
    parser.add_argument('--reference-script',type=Path,default=ROOT/'4.plot_mouse_liver_TF_activity_correlations.py')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'5.results/Mouse_Liver_TF_usage_correlations')
    parser.add_argument('--cohort',choices=['fixed','available'],default='fixed',
                        help='User-confirmed default: fixed common cohort over matched samples; available is sensitivity only')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--resume-incomplete',action='store_true',
                        help='Resume only an incomplete table-only export; tables must match byte-for-byte, no PDF/manifest may exist')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    # Check all plotting dependencies BEFORE creating any output.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy
    if args.output_dir.exists() and not args.check_only:
        require(args.resume_incomplete,'Output exists; use a NEW --output-dir or --resume-incomplete for a table-only interrupted export')
        allowed = {'sample_matching.tsv','TF_gene_mapping.tsv','selected_promoter_usage_by_sample.tsv',
                   'fixed_common_promoter_cohort.tsv','gene_denominator_validation.tsv',
                   'group_mean_promoter_usage.tsv','TF_usage_sample_pairs.tsv',
                   'TF_usage_correlations.tsv','TF_usage_cohort_sensitivity.tsv'}
        for path in args.output_dir.rglob('*'):
            require(not path.is_symlink(),'Resume refuses symlinks')
            relative = path.relative_to(args.output_dir).as_posix()
            require(relative in allowed if path.is_file() else relative in {'figures','figures/Group_A','figures/Group_B'},
                    'Resume refuses unknown/final output: '+str(path))
    inputs = [args.selected,args.sdrf,args.expression,args.tf_mapping,args.reference_script]
    inputs += [args.activity_dir/f'Mouse_Liver_{stage}{suffix}' for stage in STAGES
               for suffix in ('_activity.tsv','_exp.csv','_rowData.tsv')]
    require(all(p.is_file() for p in inputs),'One or more source files are missing')
    stamps = {p:(p.stat().st_size,p.stat().st_mtime_ns) for p in inputs}
    groups = load_selection(args.selected)
    tfs = REQUESTED_TFS
    mapping = load_requested_mapping(args.tf_mapping)
    expression_samples, expression = load_expression(args.expression, mapping)
    require(all(m['expression_status']=='observed' for m in mapping.values()),'A requested TF expression row is missing')
    metadata = load_sdrf(args.sdrf)
    activity, activity_inputs, promoter_rows, denominator_checks = load_usage_inputs(args.activity_dir,groups,args.selected)
    audit, matched = match_samples(metadata, expression_samples, activity)
    cohort,group_profiles = aggregate_usage(groups,matched,activity,args.selected,args.cohort)
    details,tests,plot_data = calculate(groups,tfs,mapping,expression,matched,activity,args.cohort)
    require(len(tests)==2 and all(t['status']=='ok' for t in tests),'Both requested correlations must be testable')
    sensitivity=[]
    for mode in ('fixed','available'):
        _,mode_tests,_ = calculate(groups,tfs,mapping,expression,matched,activity,mode)
        sensitivity.extend(mode_tests)
    require(all((p.stat().st_size,p.stat().st_mtime_ns)==stamp for p,stamp in stamps.items()),'Source changed during reading')
    profile=dict(matched_sample_n=len(matched),cohort=args.cohort,
                 fixed_cohort_promoter_n={g:activity[matched[0]['source_name']][g+'_fixed_promoter_n'] for g in groups},
                 primary_tests=tests,sensitivity_tests=sensitivity)
    print(json.dumps(json_safe(profile),indent=2),flush=True)
    if args.check_only:
        print('CHECK_ONLY_PASSED: no output written',flush=True)
        return
    args.output_dir.mkdir(parents=True,exist_ok=args.resume_incomplete)
    write_tsv(args.output_dir / 'sample_matching.tsv', audit, list(audit[0]))
    write_tsv(args.output_dir / 'TF_gene_mapping.tsv', list(mapping.values()))
    write_tsv(args.output_dir/'selected_promoter_usage_by_sample.tsv',promoter_rows)
    write_tsv(args.output_dir/'fixed_common_promoter_cohort.tsv',cohort)
    write_tsv(args.output_dir/'gene_denominator_validation.tsv',denominator_checks)
    write_tsv(args.output_dir/'group_mean_promoter_usage.tsv',group_profiles)
    write_tsv(args.output_dir/'TF_usage_sample_pairs.tsv',details)
    write_tsv(args.output_dir/'TF_usage_correlations.tsv',tests)
    write_tsv(args.output_dir/'TF_usage_cohort_sensitivity.tsv',sensitivity)
    pdfs = []
    for row in tests:
        if row['status'] != 'ok':
            continue
        group, tf = row['group'], row['TF']
        folder = args.output_dir / 'figures' / f'Group_{group}'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'Mouse_Liver_{group}_{tf}_expression_vs_mean_promoter_usage.pdf'
        require(not path.exists(), f'Existing PDF not overwritten: {path}')
        fig = build_figure(group, tf, row, plot_data[(group, tf)])
        with plt.rc_context({'pdf.compression': 0, 'pdf.fonttype': 42}):
            fig.savefig(path, format='pdf', metadata={'Title': f'Mouse Liver Group {group}: {tf}',
                                                     'Subject': MATCH_RULE})
        plt.close(fig)
        pdfs.append(path)
    require(all((p.stat().st_size,p.stat().st_mtime_ns)==stamp for p,stamp in stamps.items()),'Input changed during run')
    manifest = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
                    matching_rule=MATCH_RULE, sample_unit='matched Mouse Liver library',
                    activity_name_normalization='Remove R leading X, trailing .sorted[.bam], and sTSm/sTSmc processing suffix only; require unique exact SDRF matches.',
                    selected_promoter_n={g:len(p) for g,p in groups.items()},requested_TFs=tfs,
                    primary_cohort=args.cohort,fixed_cohort_promoter_n=profile['fixed_cohort_promoter_n'],
                    usage_formula='Per sample: promoter activity / sum of all same-gene promoter activities',
                    group_aggregation='Arithmetic mean of individual promoter usage fractions within one sample',
                    undefined_policy='0/0=NA, 0/positive=0; never imputed',
                    fixed_cohort_scope='All matched library pairs, not all SDRF libraries; TF values do not select promoters',
                    denominator_checks=denominator_checks,primary_tests=tests,
                    sensitivity_tests=sensitivity,
                    sdrf_Liver_unique_sample_n=len(metadata), expression_Liver_sample_n=len(expression_samples),
                    activity_Liver_sample_n=len(activity), matched_sample_n=len(matched),
                    unmatched_samples=[r for r in audit if r['matching_status'] != 'matched'],
                    SDRF_filename_sex_discrepancies=[r['source_name'] for r in metadata if r['sex_name_mismatch']],
                    testable_comparison_n=sum(r['status'] == 'ok' for r in tests), planned_comparison_n=len(tests),
                    missing_or_untestable=[r for r in tests if r['status'] != 'ok'],
                    BH_family='Exactly two requested HNF4A/A and E2F4/B comparisons; separate family for each sensitivity cohort',
                    tests='Two-sided scipy.stats.spearmanr; asymptotic P, ties use average ranks',
                    plot_fit='OLS: raw mean usage x, log2(RPKM+1) y; descriptive pointwise 95% Student-t mean-response CI',
                    caveats=['SDRF order matching is the user-specified convention, not an independently confirmed biological crosswalk.',
                             'Across-stage correlation may reflect shared development and cell composition, not TF binding or causality.',
                             'Promoters and TFs were selected using the same developmental dataset: exploratory, not independent validation.',
                             'Fixed cohort excludes genes with undefined usage in any matched sample; it is not the full 183-gene cohort.',
                             'Usage follows ProActiv activity-scale ratios, not linear RNA/read fractions.',
                             'Available-case sensitivity uses sample-varying promoter sets and is not the primary plot.'],
                    versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                                  scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                    inputs=[dict(path=str(p), sha256=sha256(p)) for p in inputs],
                    script_sha256=sha256(Path(__file__)), pdfs=[str(p.relative_to(args.output_dir)) for p in pdfs])
    with (args.output_dir / 'analysis_manifest.json').open('x') as handle:
        json.dump(json_safe(manifest), handle, indent=2, allow_nan=False)
    print(json.dumps(dict(output_dir=str(args.output_dir), matched_sample_n=len(matched), pdf_n=len(pdfs),
                          testable_n=manifest['testable_comparison_n'],
                          missing_TFs=[r['TF'] for r in tests if r['status'] != 'ok'],
                          unmatched=[r['expression_sample'] for r in audit if r['matching_status'] != 'matched']), indent=2))


if __name__ == '__main__':
    main()
