#!/usr/bin/env python3
"""Rank TEST Mouse Liver promoters by PREDICTED, raw-scale stage specificity.

score(p, s) = predicted_activity(p, s) - mean(predicted_activity(p, t) for t != s)

Run with Python >=3.9; standard library only. No GPU, model inference or IG.
Saved pred:: columns are already inverse-z-scored and inverse-log2p1 transformed.
They are checked against the saved pred_log2p1:: values; never transformed twice
or clipped to zero. Observed values are output for comparison, NOT selection.
Inputs are verified against the trained Mouse model's input manifest. No input
is modified. Output directories must not already exist. No extra activity,
multiple-promoter-gene, cross-stage-exclusive, or gene-deduplication filter.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys

DEFAULT_ROOT = Path('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/revise/train_model/redo')
STAGES = tuple((f'{n}.5', f'E{n}.5') for n in range(10, 19)) + (
    ('0dpb', 'P0'), ('3dpb', 'P3'), ('2wpb', '2 wpb'), ('4wpb', '4 wpb'), ('9wpb', '9 wpb'))
VERSION = 'mouse_liver_predicted_stage_minus_others_v1'
REFERENCE_URL = 'https://www.biorxiv.org/content/10.1101/2024.08.27.609971v1.full'
META_REQUIRED = {'sample_id', 'species', 'split', 'promoterId', 'geneId',
                 'seqnames', 'start', 'strand', 'assembly', 'internalPromoter'}
RANK_FIELDS = ['stage_order_1based', 'stage', 'stage_label', 'rank',
               'source_test_row_0based', 'sample_id', 'promoterId', 'geneId',
               'predicted_stage_activity_raw', 'predicted_other_stages_mean_raw', 'specificity_score',
               'score_positive', 'stage_is_activity_max', 'stage_is_unique_activity_max',
               'peak_stages', 'equal_score_count']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        fields = reader.fieldnames or []
        require(fields and len(fields) == len(set(fields)), f'Invalid header: {path}')
        records = []
        for line, record in enumerate(reader, 2):
            require(None not in record and all(v is not None for v in record.values()),
                    f'Ragged row at {path}:{line}')
            records.append(record)
    require(records, f'Empty table: {path}')
    return fields, records


def write_table(path, fields, records):
    """Write exclusive files; gzip has a reproducible header."""
    if str(path).endswith('.gz'):
        with Path(path).open('xb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t',
                                            lineterminator='\n', extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(records)
    else:
        with Path(path).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t',
                                    lineterminator='\n', extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)


def load_inputs(root):
    data = root / 'datasets/Mouse'
    paths = dict(test=data / 'test.tsv.gz', targets=data / 'target_columns.tsv',
                 training_manifest=root / 'runs/Mouse/input_manifest.json',
                 training_status=root / 'runs/Mouse/run_status.json',
                 predictions=root / 'runs/Mouse/test_predictions.tsv.gz')
    for name, filename in [('checkpoint', 'best_model.pt'), ('run_config', 'run_config.json')]:
        if (root / 'runs/Mouse' / filename).is_file():
            paths[name] = root / 'runs/Mouse' / filename
    for path in paths.values():
        require(path.is_file(), f'Missing input: {path}')
    provenance = {key: dict(path=str(path.resolve()), bytes=path.stat().st_size,
                           sha256=sha256(path)) for key, path in paths.items()}
    status = json.loads(paths['training_status'].read_text())
    require(status.get('status') == 'complete', 'Mouse model training is not complete')
    recorded = json.loads(paths['training_manifest'].read_text())
    require(isinstance(recorded, list), 'Unexpected training manifest format')
    for key, expected_split in [('test', 'test'), ('targets', 'task_definition')]:
        matches = [r for r in recorded if Path(r['path']).name == paths[key].name]
        require(len(matches) == 1, f'Ambiguous/missing trained input record: {paths[key]}')
        record = matches[0]
        require(record.get('split') == expected_split and
                record['sha256'] == provenance[key]['sha256'] and
                record['bytes'] == provenance[key]['bytes'],
                f'Input differs from the trained model manifest: {paths[key]}')

    target_fields, targets = read_table(paths['targets'])
    require({'output_index_0based', 'species', 'organ', 'stage', 'target_column'} <= set(target_fields),
            'Missing target metadata fields')
    for i, target in enumerate(targets):
        require(target['species'] == 'Mouse' and int(target['output_index_0based']) == i,
                'Target species/index mismatch')
        require(target['target_column'] == f"Mouse_{target['organ']}_{target['stage']}.mean",
                'Target name does not match organ/stage metadata')
    columns = [r['target_column'] for r in targets]
    require(len(columns) == len(set(columns)), 'Duplicate output target')
    liver = [r for r in targets if r['organ'] == 'Liver']
    liver_map = {r['stage']: r for r in liver}
    require(len(liver_map) == len(liver) == len(STAGES) and
            set(liver_map) == {s for s, _ in STAGES}, 'Expected exactly the 14 recorded Mouse Liver stages')
    ordered = [dict(liver_map[stage], stage_label=label, stage_order_1based=i + 1)
               for i, (stage, label) in enumerate(STAGES)]
    fields, rows = read_table(paths['test'])
    require(META_REQUIRED <= set(fields), 'TEST metadata columns are missing')
    require([f for f in fields if f.endswith('.mean')] == columns,
            'TEST activity columns do not match trained target order')
    metadata_fields = [f for f in fields if not f.endswith('.mean')]
    require(not (set(metadata_fields) - {'sample_id', 'promoterId', 'geneId'}) & set(RANK_FIELDS),
            'Metadata field conflicts with an output ranking field')
    pred_fields, pred_rows = read_table(paths['predictions'])
    pred_meta = {'sample_id', 'promoterId', 'geneId', 'seqnames', 'start', 'strand'}
    require(pred_meta <= set(pred_fields), 'Missing saved prediction metadata')
    for prefix in ['pred::', 'true::', 'pred_log2p1::']:
        require([f for f in pred_fields if f.startswith(prefix)] == [prefix + c for c in columns],
                f'Saved prediction target columns mismatch: {prefix}')
    pred_map = {}
    for record in pred_rows:
        require(record['sample_id'] not in pred_map, f'Duplicate prediction: {record["sample_id"]}')
        pred_map[record['sample_id']] = record
    require(len(pred_map) == len(rows) and set(pred_map) == {r['sample_id'] for r in rows},
            'Prediction cohort differs from the original TEST sample IDs')
    seen_samples, seen_promoters, activities = set(), set(), []
    for i, row in enumerate(rows):
        sid, pid = row['sample_id'], row['promoterId']
        require(row['species'] == 'Mouse' and row['split'] == 'test', f'Not Mouse TEST: {sid}')
        require(pid and sid == f'Mouse:{pid}' and sid not in seen_samples and pid not in seen_promoters,
                f'Duplicate/invalid promoter identity: {sid}')
        require(row['geneId'].strip().upper() not in {'', 'NA', 'NAN'}, f'Missing gene: {sid}')
        require(row['internalPromoter'].upper() == 'FALSE',
                f'Training dataset invariant violated (internal promoter): {sid}')
        require(row['assembly'] == 'GRCm38' and row['strand'] in {'+', '-'} and int(row['start']) > 0,
                f'Invalid Mouse assembly/TSS metadata: {sid}')
        seen_samples.add(sid)
        seen_promoters.add(pid)
        observed = tuple(float(row[t['target_column']]) for t in ordered)
        require(all(math.isfinite(v) and v >= 0 for v in observed),
                f'Missing/nonfinite/negative raw Liver activity: {sid}')
        prediction = pred_map[sid]
        require(all(prediction[k] == row[k] for k in pred_meta), f'Saved prediction identity mismatch: {sid}')
        values = tuple(float(prediction['pred::' + t['target_column']]) for t in ordered)
        log_values = tuple(float(prediction['pred_log2p1::' + t['target_column']]) for t in ordered)
        require(all(math.isfinite(v) for v in values + log_values), f'Nonfinite model prediction: {sid}')
        for t, observed_value, pred_value, log_value in zip(ordered, observed, values, log_values):
            col = t['target_column']
            require(math.isclose(float(prediction['true::' + col]), observed_value, rel_tol=1e-10, abs_tol=1e-10),
                    f'Prediction true:: column differs from TEST observation: {sid}/{col}')
            require(math.isclose(pred_value, math.expm1(log_value * math.log(2)), rel_tol=1e-10, abs_tol=1e-10),
                    f'Raw prediction is not the unclipped inverse-log2p1 prediction: {sid}/{col}')
            row['pred::' + col] = pred_value
            row['pred_log2p1::' + col] = log_value
            row['observed::' + col] = observed_value
        activities.append(values)
    # Confirm the exact small source files have not changed while being read.
    for key, path in paths.items():
        require(sha256(path) == provenance[key]['sha256'], f'Input changed during read: {path}')
    return rows, activities, ordered, metadata_fields, provenance


def rank_promoters(rows, activities, stages, top_k):
    require(len(rows) == len(activities) and len(rows) >= top_k > 0, 'Insufficient rows or invalid top-k')
    n_stages = len(stages)
    require(n_stages > 1 and all(len(v) == n_stages for v in activities), 'Activity/stage dimensions mismatch')
    require(all(math.isfinite(x) for v in activities for x in v), 'Invalid predicted activity')
    full, selected, summary = [], [], []
    peaks = [';'.join(stages[j]['stage'] for j, value in enumerate(values) if value == max(values))
             for values in activities]
    for j, stage in enumerate(stages):
        scored = []
        for i, (row, values) in enumerate(zip(rows, activities)):
            others = values[:j] + values[j + 1:]
            other_mean = math.fsum(others) / (n_stages - 1)
            score = values[j] - other_mean
            require(math.isfinite(score), f'Nonfinite score: {row["sample_id"]}/{stage["stage"]}')
            scored.append(dict(stage_order_1based=stage['stage_order_1based'], stage=stage['stage'],
                               stage_label=stage['stage_label'], source_test_row_0based=i,
                               sample_id=row['sample_id'], promoterId=row['promoterId'], geneId=row['geneId'],
                               predicted_stage_activity_raw=values[j], predicted_other_stages_mean_raw=other_mean,
                               specificity_score=score, score_positive=score > 0,
                               stage_is_activity_max=values[j] >= max(others),
                               stage_is_unique_activity_max=values[j] > max(others), peak_stages=peaks[i]))
        # ID is only a deterministic tie breaker, not an additional biological filter.
        scored.sort(key=lambda r: (-r['specificity_score'], r['sample_id']))
        counts = Counter(r['specificity_score'] for r in scored)
        for rank, record in enumerate(scored, 1):
            record.update(rank=rank, equal_score_count=counts[record['specificity_score']])
        full.extend(scored)
        selected.extend(scored[:top_k])
        cutoff = scored[top_k - 1]['specificity_score']
        summary.append(dict(stage_order_1based=stage['stage_order_1based'], stage=stage['stage'],
                            stage_label=stage['stage_label'], n_test_promoters=len(rows),
                            n_positive_score=sum(r['score_positive'] for r in scored), n_selected=top_k,
                            n_selected_positive=sum(r['score_positive'] for r in scored[:top_k]),
                            n_selected_unique_activity_max=sum(r['stage_is_unique_activity_max'] for r in scored[:top_k]),
                            top_score=scored[0]['specificity_score'], cutoff_score=cutoff,
                            n_at_cutoff=counts[cutoff],
                            tie_crosses_cutoff=top_k < len(rows) and scored[top_k]['specificity_score'] == cutoff))
    return full, selected, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--redo-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--outdir', type=Path, help='New output directory; default: timestamped results beside this script')
    parser.add_argument('--top-k', type=int, default=1500)
    parser.add_argument('--check-only', action='store_true', help='Validate and calculate counts without writing output')
    args = parser.parse_args(argv)
    require(args.top_k > 0, '--top-k must be positive')
    rows, activities, stages, metadata, sources = load_inputs(args.redo_root.resolve())
    full, selected, summary = rank_promoters(rows, activities, stages, args.top_k)
    selected_by_id = defaultdict(list)
    for record in selected:
        selected_by_id[record['sample_id']].append(record)
    report = dict(n_test_promoters=len(rows), n_stages=len(stages), top_k=args.top_k,
                  n_selected_stage_promoter_pairs=len(selected), n_unique_selected_promoters=len(selected_by_id),
                  n_all_zero_predicted_liver_promoters=sum(all(x == 0 for x in values) for values in activities),
                  n_negative_predicted_liver_values=sum(x < 0 for values in activities for x in values),
                  all_selected_scores_positive=all(r['score_positive'] for r in selected),
                  n_selected_pairs_not_unique_activity_max=sum(not r['stage_is_unique_activity_max'] for r in selected))
    if args.check_only:
        print(json.dumps(dict(status='check_only_passed', **report, stages=summary), indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.outdir or Path(__file__).resolve().parent / 'results' / f'top{args.top_k}_stage_specific_{stamp}').resolve()
    require(not output.exists(), f'Refusing existing output directory: {output}')
    output.mkdir(parents=True, exist_ok=False)
    activity_columns = [prefix + t['target_column'] for prefix in ['pred::', 'observed::'] for t in stages]
    observed_fields = ['observed_stage_activity_raw', 'observed_other_stages_mean_raw', 'observed_specificity_score']
    top_fields = RANK_FIELDS + observed_fields + [m for m in metadata if m not in RANK_FIELDS] + activity_columns
    top_rows = []
    for r in selected:
        row = rows[r['source_test_row_0based']]
        j = r['stage_order_1based'] - 1
        obs = tuple(float(row[t['target_column']]) for t in stages)
        obs_others = math.fsum(obs[:j] + obs[j + 1:]) / (len(stages) - 1)
        top_rows.append(dict(row, **r, observed_stage_activity_raw=obs[j],
                             observed_other_stages_mean_raw=obs_others,
                             observed_specificity_score=obs[j] - obs_others))
    top_name = f'top{args.top_k}_stage_specific_promoters.tsv'
    write_table(output / top_name, top_fields, top_rows)
    write_table(output / 'all_stage_rankings.tsv.gz', RANK_FIELDS, full)
    write_table(output / 'stage_summary.tsv', list(summary[0]), summary)
    ledger = [dict(stage_order_1based=s['stage_order_1based'], stage=s['stage'], stage_label=s['stage_label'],
                   target_column=s['target_column'], output_index_0based=s['output_index_0based'],
                   focal_stage_weight=1, each_other_liver_stage_weight=-1 / (len(stages) - 1),
                   n_other_liver_stages=len(stages) - 1) for s in stages]
    write_table(output / 'stage_definition.tsv', list(ledger[0]), ledger)
    unique_rows = []
    for i, row in enumerate(rows):
        picked = selected_by_id.get(row['sample_id'])
        if picked:
            unique_rows.append(dict(row, source_test_row_0based=i,
                                    selected_stages=';'.join(r['stage'] for r in picked),
                                    selected_stage_ranks=';'.join(f"{r['stage']}:{r['rank']}" for r in picked),
                                    n_selected_stages=len(picked)))
    unique_fields = ['source_test_row_0based', *metadata, 'selected_stages', 'selected_stage_ranks',
                     'n_selected_stages', *activity_columns]
    write_table(output / 'selected_unique_promoters.tsv', unique_fields, unique_rows)
    notes = f'''Mouse Liver stage-specific TEST promoter ranking

Activity source: saved model predictions from runs/Mouse/test_predictions.tsv.gz, pred:: columns.
Formula: predicted_activity(p, stage) - mean predicted_activity of the SAME promoter at the OTHER 13 Liver stages.
The pred:: columns are already on the raw activity scale, after inverse z-score and inverse
log2p1. Their consistency with pred_log2p1:: columns is verified. Negative raw predictions
are preserved, not clipped or filtered. Rank on RAW predicted activity, not log2p1 or z scores.
Observed labels from datasets/Mouse/test.tsv.gz are comparison-only and never affect selection.
No IG attribution values, baseline predictions or other-organ measurements enter the score.
This script reuses saved TEST predictions; it does NOT run inference or IG.

Selection: exactly top {args.top_k} scores per stage, descending; exact numeric ties are resolved
by sample_id ascending. All valid TEST promoters are ranked, including all-zero Liver rows.
No positive-score cutoff, activity threshold, gene deduplication, or cross-stage exclusivity
is added. Observed all-zero promoters also remain eligible. The existing training dataset
already excludes internal promoters; this is verified.
Thus a promoter/gene can appear in several stage lists. This score is an absolute difference,
not a P value or proof of expression exclusively in one stage; raw activity magnitude affects it.
stage_is_unique_activity_max reports strict stage maxima but does NOT filter the selection.

Outputs:
  {top_name}: stage top-{args.top_k}, predicted/observed contrasts, metadata, 14 predictions and 14 observations.
  all_stage_rankings.tsv.gz: full per-stage ranking including zero and negative scores.
  selected_unique_promoters.tsv: deduplicated union for downstream lookup (no re-selection).
  stage_summary.tsv: positive counts, top/cutoff scores, ties and selected-stage-max flags.
  stage_definition.tsv: original target names/output indices and contrast weights.
  manifest.json: formula, input/script/output checksums and counts; final status is complete.

Stages are DISPLAYED chronologically, not in checkpoint output order. Original labels are kept:
10.5..18.5 = E10.5..E18.5; 0dpb=P0; 3dpb=P3. All 14 stages receive equal comparison weight.
source_test_row_0based refers to the original test.tsv.gz row, NOT a quality.tsv physical row.
Coordinates: GRCm38; start is 1-based TSS; window_start0/window_end0 are 0-based half-open.
No train/validation promoters are ranked. Selection on TEST model predictions is exploratory,
not an independent performance evaluation cohort for any later model-tuning decision.

Reference checked via PMC11383055, Methods: Deep learning analysis / De novo motif discovery.
The paper uses predicted log10CPM(target state) minus the mean prediction of other states,
and top 10,000 peaks. Here the user requested predicted promoter activity and top {args.top_k}; we adapt
the contrast idea to RAW promoter activity, 14 Liver stages, and only the Mouse TEST cohort.
The paper's error filter/threshold and log10CPM scale are not imposed on this different assay.
{REFERENCE_URL}
'''
    with (output / 'README.txt').open('x', encoding='utf-8') as stream:
        stream.write(notes)
    # Verify selected values by an independent sum-minus-focal identity and input membership.
    for record in selected:
        i = record['source_test_row_0based']
        j = record['stage_order_1based'] - 1
        expected = activities[i][j] - (math.fsum(activities[i]) - activities[i][j]) / (len(stages) - 1)
        require(math.isclose(record['specificity_score'], expected, rel_tol=1e-12, abs_tol=1e-10),
                'Selected score verification failed')
        require(rows[i]['sample_id'] == record['sample_id'], 'Selected TEST identity mismatch')
    for source in sources.values():
        require(sha256(source['path']) == source['sha256'], 'Input source changed during ranking')
    manifest = dict(status='complete', version=VERSION, completed_utc=datetime.now(timezone.utc).isoformat(),
                    species='Mouse', organ='Liver', split='test', activity_source='model_predicted_raw',
                    formula='predicted_activity[p,s] - mean(predicted_activity[p,t] for t in Liver_stages if t != s)',
                    prediction_columns='pred::Mouse_Liver_<stage>.mean',
                    raw_scale_validation='pred == expm1(pred_log2p1 * ln(2)); saved predictions already inverse z-scored',
                    negative_prediction_handling='preserved; no clipping',
                    scale='raw, untransformed', other_stage_weighting='equal over the other 13 stages',
                    ranking='score descending; sample_id ascending for exact numeric ties',
                    additional_filters='none; inherited non-internal TEST cohort verified',
                    cross_stage_exclusivity=False, gene_deduplication=False, inputs=sources,
                    script=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)),
                    output=str(output), counts=report, reference_url=REFERENCE_URL,
                    output_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='complete', output=str(output), **report), indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, OverflowError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
