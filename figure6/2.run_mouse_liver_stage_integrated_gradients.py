#!/usr/bin/env python3
"""Per-stage Captum IG for the previously selected Mouse Liver TEST Top 1500.

For each stage, explain its SINGLE predicted RAW promoter activity, not the
stage-minus-rest selection score, not a loss, and not the organ-wide mean.
Default scope is all 14 stages x 1500 sequences. --check-only writes nothing;
--mode pilot explicitly runs only evenly spaced ranks. No retraining occurs.

Saved Top1500 predictions are provenance/ranking values only: do not require
fresh FP32 predictions to match the original TF32 inference. Keep the saved
value and absolute difference in quality.tsv; IG explains the CURRENT model
prediction. Finite predictions, source hashes and IG numerical QA remain required.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import traceback

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/revise/train_model/redo')
STAGES = tuple((f'{n}.5', f'E{n}.5') for n in range(10, 19)) + (
    ('0dpb', 'P0'), ('3dpb', 'P3'), ('2wpb', '2 wpb'), ('4wpb', '4 wpb'), ('9wpb', '9 wpb'))
TOP_K, LENGTH = 1500, 10000
VERSION = 'mouse_liver_single_stage_raw_captum_ig_v1'
CORE_HASHES = {
    'ig_core.py': 'dd02927f3798ae5a79c48decba114887f9cfc8cbff2537dda70466600f642301',
    'ig_inputs.py': 'abe221f3de6bab29899d6f231653944607801103ef46e4ff51c8e5a38195753e',
    'run_organ_species_integrated_gradients.py': 'c4bcfa261a87901cdc40e7574e1d46a8fff1780c132bc3b88f97d1a87a4df4b7',
}
QUALITY_FIELDS = [
    'attribution_row_0based', 'rank', 'stage', 'target_column', 'output_index_0based',
    'source_array_index_0based', 'sample_id', 'promoterId', 'geneId',
    'predicted_stage_activity_raw', 'saved_predicted_stage_activity_raw',
    'prediction_absolute_error', 'baseline_stage_activity_raw', 'explained_difference',
    'observed_stage_activity_raw', 'selection_specificity_score_NOT_IG_TARGET',
    'n_steps', 'converged', 'accepted', 'completeness_pass', 'map_pass',
    'attribution_sum', 'stored_attribution_sum', 'stored_completeness_pass',
    'completeness_signed_error', 'completeness_absolute_error', 'completeness_tolerance',
    'captum_delta', 'map_l1_change', 'map_l1_mass', 'map_relative_l1_change',
    'position_map_l1_change', 'position_map_l1_mass', 'position_map_relative_l1_change',
    'map_cosine', 'near_zero_map', 'n_canonical_positions', 'n_unknown_positions',
    'elapsed_seconds', 'ig_row_sha256', 'position_row_sha256',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_tsv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), f'Invalid header: {path}')
        rows = list(reader)
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows), f'Ragged table: {path}')
    return rows


def write_tsv(path, rows):
    require(rows, 'Cannot write empty metadata table')
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def truth(value):
    return str(value).lower() == 'true'


def discover_selection(task_dir):
    candidates = [Path(task_dir) / parent / 'predicted_raw_top1500' for parent in ['1.results', 'results']]
    found = [p for p in candidates if (p / 'manifest.json').is_file()]
    require(len(found) == 1, 'Expected one Top1500 directory; supply --selection-dir explicitly')
    return found[0].resolve()


def read_selection(folder, bundle, sources, predictions_path):
    """Verify the selection and join by sample ID, NEVER top-table row number."""
    folder = Path(folder).resolve(strict=True)
    manifest = read_json(folder / 'manifest.json')
    require(manifest.get('status') == 'complete' and manifest.get('species') == 'Mouse' and
            manifest.get('organ') == 'Liver' and manifest.get('split') == 'test' and
            manifest.get('activity_source') == 'model_predicted_raw', 'Wrong/incomplete selection source')
    counts = manifest['counts']
    require(counts['top_k'] == TOP_K and counts['n_stages'] == len(STAGES) and
            counts['n_selected_stage_promoter_pairs'] == TOP_K * len(STAGES) and
            counts['n_test_promoters'] == bundle['n_promoters'], 'Selection scope is not Top1500 per stage')
    outputs = manifest['output_sha256']
    require({'top1500_stage_specific_promoters.tsv', 'stage_definition.tsv'} <= set(outputs), 'Missing selection outputs')
    for name, digest in outputs.items():
        require(Path(name).name == name and (folder / name).is_file(), 'Unsafe/missing selection file')
        require(sha(folder / name) == digest, f'Selection output changed: {name}')
    for name, original in [('test', 'test.tsv.gz'), ('targets', 'target_columns.tsv'), ('checkpoint', 'best_model.pt')]:
        require(manifest['inputs'][name]['sha256'] == sources[original]['sha256'], f'Selection uses a different {name}')
    prediction_hash = sha(predictions_path)
    require(prediction_hash == manifest['inputs']['predictions']['sha256'], 'Saved predictions changed since selection')
    by_id = {p['sample_id']: i for i, p in enumerate(bundle['promoters'])}
    require(len(by_id) == len(bundle['promoters']) == bundle['n_promoters'], 'Duplicate/incomplete cache IDs')
    targets = {t['stage']: t for t in bundle['targets'] if t['organ'] == 'Liver' and t['species'] == 'Mouse'}
    require(set(targets) == {s for s, _ in STAGES}, 'Expected the original 14 Liver targets')
    ledger = read_tsv(folder / 'stage_definition.tsv')
    require(len(ledger) == len(STAGES) and {r['stage'] for r in ledger} == set(targets), 'Invalid stage ledger')
    for r in ledger:
        t = targets[r['stage']]
        require(r['target_column'] == t['target_column'] and int(r['output_index_0based']) == int(t['output_index_0based']),
                'Selection and checkpoint target order differ')
    selected = read_tsv(folder / 'top1500_stage_specific_promoters.tsv')
    require(len(selected) == TOP_K * len(STAGES) and {r['stage'] for r in selected} == set(targets), 'Invalid selected row/stage count')
    plans = []
    for stage_order, (stage, label) in enumerate(STAGES, 1):
        rows = [r for r in selected if r['stage'] == stage]
        rows.sort(key=lambda r: int(r['rank']))
        require(len(rows) == TOP_K and [int(r['rank']) for r in rows] == list(range(1, TOP_K + 1)), 'Missing/duplicate stage ranks')
        require(len({r['sample_id'] for r in rows}) == TOP_K, 'Duplicate promoter within a stage')
        for r in rows:
            require(r['sample_id'] in by_id, f'Not in TEST DNA cache: {r["sample_id"]}')
            i = by_id[r['sample_id']]
            p = bundle['promoters'][i]
            require(all(r[k] == p[k] for k in ['promoterId', 'geneId', 'seqnames', 'start', 'strand']), 'Selected promoter identity mismatch')
            require(int(r['source_test_row_0based']) == int(p['source_row_index_0based']) and
                    int(p['array_index_0based']) == i, 'Selected/cache source indices disagree')
            require(int(r['stage_order_1based']) == stage_order, 'Stage display order mismatch')
            raw = float(r['predicted_stage_activity_raw'])
            other = float(r['predicted_other_stages_mean_raw'])
            score = float(r['specificity_score'])
            require(all(math.isfinite(v) for v in [raw, other, score]), 'Nonfinite selected prediction/score')
            require(math.isclose(raw - other, score, rel_tol=1e-10, abs_tol=1e-10), 'Invalid stage-specific score')
            require(math.isclose(raw, float(r['pred::' + targets[stage]['target_column']]), rel_tol=1e-10, abs_tol=1e-10),
                    'Selected target prediction does not match its stage column')
            r['cache_index_0based'] = i
        require(rows == sorted(rows, key=lambda r: (-float(r['specificity_score']), r['sample_id'])), 'Top1500 ranking order changed')
        plans.append(dict(stage=stage, stage_label=label, stage_order_1based=stage_order,
                          target_column=targets[stage]['target_column'],
                          output_index_0based=int(targets[stage]['output_index_0based']), rows=rows))
    require(len({r['sample_id'] for r in selected}) == counts['n_unique_selected_promoters'], 'Selection union count mismatch')
    provenance = dict(directory=str(folder), manifest_sha256=sha(folder / 'manifest.json'),
                      output_sha256=outputs, prediction_file=str(Path(predictions_path).resolve()),
                      prediction_sha256=prediction_hash)
    return plans, provenance


def choose_rows(plan, mode, pilot_count):
    rows = plan['rows']
    if mode == 'full':
        return list(rows)
    indices = np.unique(np.linspace(0, len(rows) - 1, min(pilot_count, len(rows)), dtype=int))
    return [rows[i] for i in indices]


def load_backend(code_dir):
    """Reuse audited source/model loading and Captum numerical core unchanged."""
    code_dir = Path(code_dir).resolve(strict=True)
    for name, expected in CORE_HASHES.items():
        require(sha(code_dir / name) == expected, f'Existing IG dependency version changed: {name}')
    sys.path.insert(0, str(code_dir))
    core = importlib.import_module('ig_core')
    runner = importlib.import_module('run_organ_species_integrated_gradients')
    require(Path(core.__file__).resolve() == code_dir / 'ig_core.py' and
            Path(runner.__file__).resolve() == code_dir / 'run_organ_species_integrated_gradients.py', 'Wrong dependency import')
    return core, runner


def run_stage(item, plan, chosen, folder, args, core, torch):
    """Commit arrays before their row ledger; partial/unaccepted rows stay explicit."""
    bundle = item['bundle']
    stage, idx = plan['stage'], plan['output_index_0based']
    n = len(chosen)
    layout = {'ig_attributions.npy': (n, 4, LENGTH), 'ig_position_sum.npy': (n, LENGTH)}
    current = dict(stage=stage, stage_label=plan['stage_label'], target_column=plan['target_column'],
                   output_index_0based=idx, total=n, processed=0, failed_quality_rows=0, status='starting')
    resuming = folder.exists()
    if resuming:
        require(args.resume and (folder / 'manifest.json').is_file(), 'Incomplete stage initialization; use a NEW output directory')
        manifest = read_json(folder / 'manifest.json')
        require(manifest['stage'] == stage and manifest['output_index_0based'] == idx and manifest['total'] == n, 'Stage identity changed')
        for name, digest in manifest['input_sha256'].items():
            require(sha(folder / name) == digest, f'Stage metadata changed: {name}')
        previous = read_json(folder / 'status.json') if (folder / 'status.json').is_file() else {}
        if previous.get('status') in ['complete', 'pilot_complete', 'finished_with_quality_failures']:
            require('output_sha256' in manifest, 'Missing completed output checksums')
            for name, digest in manifest['output_sha256'].items():
                require(sha(folder / name) == digest, f'Completed stage output changed: {name}')
            print('SKIP_FINISHED', stage, flush=True)
            return previous
    else:
        folder.mkdir()
        indices = [r['cache_index_0based'] for r in chosen]
        np.save(folder / 'sequence_codes.npy', bundle['codes'][indices], allow_pickle=False)
        np.save(folder / 'canonical_mask.npy', bundle['codes'][indices] < 4, allow_pickle=False)
        write_tsv(folder / 'promoter_index.tsv', [dict(bundle['promoters'][r['cache_index_0based']],
            attribution_row_0based=j, selected_stage=stage, selection_rank=int(r['rank']),
            selection_specificity_score_NOT_IG_TARGET=float(r['specificity_score']),
            saved_predicted_stage_activity_raw=float(r['predicted_stage_activity_raw'])) for j, r in enumerate(chosen)])
        write_tsv(folder / 'target.tsv', [dict(stage=stage, stage_label=plan['stage_label'], target_column=plan['target_column'],
            output_index_0based=idx, train_log2p1_mean=float(bundle['mean'][idx]), train_log2p1_std=float(bundle['std'][idx]),
            target_weight=1, other_stage_weight=0)])
        for name, shape in layout.items():
            a = np.lib.format.open_memmap(folder / name, mode='w+', dtype=np.float32, shape=shape)
            a[:] = np.nan
            a.flush()
            del a
        manifest = dict(**current, version=VERSION, started_utc=now(),
            target='Single-stage predicted raw activity; NOT stage-minus-rest specificity',
            formula='expm1(ln(2)*(model_z[output_index]*TRAIN_std[output_index]+TRAIN_mean[output_index]))',
            explained_quantity='F_stage(sequence)-F_stage(baseline)',
            baseline='../baseline_mean_train_onehot.npy (same DNA baseline, stage-specific prediction)',
            channels=['A', 'C', 'G', 'T'], position_score='SUM of all 4 actual IG channels',
            pending_rows='NaN/uncommitted; accept only quality.tsv rows with accepted=True',
            shape={name:list(shape) for name, shape in layout.items()},
            input_sha256={name:sha(folder / name) for name in ['sequence_codes.npy', 'canonical_mask.npy', 'promoter_index.tsv', 'target.tsv']})
        save_json(folder / 'manifest.json', manifest)
    arrays = {name:np.load(folder / name, mmap_mode='r+', allow_pickle=False) for name in layout}
    for name, a in arrays.items():
        require(a.shape == layout[name] and a.dtype == np.float32, 'Wrong attribution shape/dtype')
    quality_path = folder / 'quality.tsv'
    committed = read_tsv(quality_path) if quality_path.is_file() else []
    require(len(committed) <= n, 'Too many committed rows')
    for j, r in enumerate(committed):
        expected = chosen[j]
        require(int(r['attribution_row_0based']) == j and r['sample_id'] == expected['sample_id'] and
                int(r['rank']) == int(expected['rank']) and r['stage'] == stage and
                int(r['output_index_0based']) == idx and int(r['source_array_index_0based']) == expected['cache_index_0based'],
                'Resume ledger row identity mismatch')
        for name, key in [('ig_attributions.npy', 'ig_row_sha256'), ('ig_position_sum.npy', 'position_row_sha256')]:
            require(array_sha(arrays[name][j]) == r[key], 'Resume committed array checksum mismatch')
    # This one-element RawOrganMean is exactly ONE stage, with a scalar per input.
    wrapper = core.RawOrganMean(item['model'], [idx], bundle['mean'][[idx]], bundle['std'][[idx]]).eval().to(args.device)
    baseline = torch.from_numpy(bundle['baseline']).to(args.device)
    with torch.no_grad():
        fb = float(wrapper(baseline).item())
    require(math.isfinite(fb), 'Nonfinite baseline target prediction')
    failed = sum(not truth(r['accepted']) for r in committed)
    current.update(processed=len(committed), failed_quality_rows=failed, baseline_stage_activity_raw=fb)
    save_json(folder / 'status.json', current)
    for j in range(len(committed), n):
        r = chosen[j]
        i = r['cache_index_0based']
        x = torch.from_numpy(core.onehot(bundle['codes'][i:i+1])).to(args.device)
        with torch.no_grad():
            fx = float(wrapper(x).item())
        saved = float(r['predicted_stage_activity_raw'])
        # Original saved predictions used a different TF32 inference path.
        # Record the difference below, but do not reject a sequence for it.
        # IG completeness must use current fx, never substitute the saved value.
        require(math.isfinite(fx), 'Nonfinite current target prediction')
        # Do NOT place the IG call in no_grad(). The core invokes Captum's
        # IntegratedGradients(multiply_by_inputs=True), target=None for scalar F.
        result = core.adaptive_ig_one(wrapper, x, baseline, steps=args.steps, max_steps=args.max_steps,
            internal_batch_size=args.internal_batch_size, atol=args.atol, rtol=args.rtol,
            map_atol=args.map_atol, map_rtol=args.map_rtol, fx=fx, fb=fb)
        q = result['quality']
        a = np.asarray(result['attributions'], dtype=np.float32)
        p = a.sum(axis=0, dtype=np.float64).astype(np.float32)
        require(a.shape == (4, LENGTH) and np.isfinite(a).all() and np.isfinite(p).all(), 'Nonfinite stored attribution')
        stored_sum = float(a.sum(dtype=np.float64))
        stored_ok = abs(stored_sum - (fx - fb)) <= q['completeness_tolerance']
        accepted = bool(q['converged'] and stored_ok)
        arrays['ig_attributions.npy'][j] = a
        arrays['ig_position_sum.npy'][j] = p
        for value in arrays.values():
            value.flush()
        record = dict(attribution_row_0based=j, rank=int(r['rank']), stage=stage,
            target_column=plan['target_column'], output_index_0based=idx, source_array_index_0based=i,
            sample_id=r['sample_id'], promoterId=r['promoterId'], geneId=r['geneId'],
            predicted_stage_activity_raw=fx, saved_predicted_stage_activity_raw=saved,
            prediction_absolute_error=abs(fx-saved), baseline_stage_activity_raw=fb, explained_difference=fx-fb,
            observed_stage_activity_raw=float(bundle['observed'][i, idx]),
            selection_specificity_score_NOT_IG_TARGET=float(r['specificity_score']),
            **{k:v for k,v in q.items() if k != 'trials'})
        record.update(stored_attribution_sum=stored_sum, stored_completeness_pass=bool(stored_ok), accepted=accepted,
                      ig_row_sha256=array_sha(a), position_row_sha256=array_sha(p))
        with (folder / 'quadrature_trials.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(attribution_row_0based=j, sample_id=r['sample_id'], stage=stage, trials=q['trials']), allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        with quality_path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=QUALITY_FIELDS, delimiter='\t', lineterminator='\n')
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow({key:record.get(key, '') for key in QUALITY_FIELDS})
            stream.flush()
            os.fsync(stream.fileno())
        failed += int(not accepted)
        current.update(status='running', processed=j+1, failed_quality_rows=failed, updated_utc=now(),
                       last_rank=int(r['rank']), last_n_steps=q['n_steps'], last_accepted=accepted)
        save_json(folder / 'status.json', current)
        print(json.dumps(dict(stage=stage, processed=j+1, total=n, rank=int(r['rank']), n_steps=q['n_steps'],
                              accepted=accepted, seconds=q['elapsed_seconds'])), flush=True)
    for a in arrays.values():
        a.flush()
    manifest.update(finished_utc=now(), baseline_stage_activity_raw=fb,
        output_sha256={name:sha(folder / name) for name in [*layout, 'quality.tsv', 'quadrature_trials.jsonl']})
    save_json(folder / 'manifest.json', manifest)
    current.update(status=('finished_with_quality_failures' if failed else
                          ('complete' if args.mode == 'full' else 'pilot_complete')),
                   all_rows_passed=failed == 0, finished_utc=now())
    save_json(folder / 'status.json', current)
    del arrays, wrapper
    return current


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--redo-root', type=Path, default=ROOT)
    p.add_argument('--selection-dir', type=Path, help='Auto-detect 1.results/predicted_raw_top1500 or results/predicted_raw_top1500')
    p.add_argument('--inputs-root', type=Path, help='Existing prepared all-TEST sequence cache parent')
    p.add_argument('--code-dir', type=Path, help='Existing audited organ_species_model_explanation modules')
    p.add_argument('--stages', nargs='+', choices=[s for s, _ in STAGES], help='Original stage labels; default all 14 chronologically')
    p.add_argument('--mode', choices=['full', 'pilot'], default='full')
    p.add_argument('--pilot-count', type=int, default=3)
    p.add_argument('--device', default='cpu', help='cpu or explicit cuda:N; check GPU occupancy before a long run')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--max-steps', type=int, default=2048)
    p.add_argument('--internal-batch-size', type=int, default=4)
    p.add_argument('--atol', type=float, default=1e-3)
    p.add_argument('--rtol', type=float, default=.01)
    p.add_argument('--map-atol', type=float, default=1e-3)
    p.add_argument('--map-rtol', type=float, default=.10)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-only', action='store_true', help='Read-only input/model validation; no IG or output files')
    args = p.parse_args(argv)
    require(args.pilot_count > 0 and args.threads > 0 and args.internal_batch_size > 0, 'Counts must be positive')
    require(0 < args.steps < args.max_steps, 'At least two quadrature levels required: 0 < steps < max-steps')
    require(all(math.isfinite(v) and v >= 0 for v in [args.atol, args.rtol, args.map_atol, args.map_rtol]), 'Invalid tolerances')
    require(not args.stages or len(args.stages) == len(set(args.stages)), 'Duplicate requested stage')
    require(not args.resume or args.output_dir is not None, '--resume requires --output-dir')
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.redo_root.resolve(strict=True)
    task = root / 'explain_development/mouse_liver_rerun'
    selection = (args.selection_dir or discover_selection(task)).resolve(strict=True)
    inputs = (args.inputs_root or root / 'explain_development/run_deeplift/inputs').resolve(strict=True)
    code = (args.code_dir or root / 'organ_species_model_explanation').resolve(strict=True)
    core, runner = load_backend(code)
    import torch
    import captum
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device != 'cpu':
        require(args.device.startswith('cuda:') and torch.cuda.is_available(), 'Use cpu or an available explicit cuda:N')
        torch.cuda.set_device(args.device)
    print('VERIFY_INPUTS_AND_CHECKPOINT', flush=True)
    item = runner.load_species(root, inputs, 'Mouse', ['Liver'])
    plans, provenance = read_selection(selection, item['bundle'], item['sources'], root / 'runs/Mouse/test_predictions.tsv.gz')
    if args.stages:
        plans = [plan for plan in plans if plan['stage'] in args.stages]
    chosen = {p['stage']:choose_rows(p, args.mode, args.pilot_count) for p in plans}
    plan_summary = [{k:p[k] for k in ['stage', 'stage_label', 'target_column', 'output_index_0based']} |
                    dict(n_selected=TOP_K, n_to_run=len(chosen[p['stage']])) for p in plans]
    estimate = sum(len(r) for r in chosen.values()) * LENGTH * 22  # actual4+sum1 float32, DNA uint8, mask bool
    signature = dict(version=VERSION, script_sha256=sha(__file__), dependencies=CORE_HASHES,
        selection=provenance, prepared=item['bundle']['verified_hashes'], sources=item['sources'],
        config={k:getattr(args,k) for k in ['mode','pilot_count','device','threads','steps','max_steps',
                'internal_batch_size','atol','rtol','map_atol','map_rtol']}, plan=plan_summary,
        target='single_stage_raw_activity', torch=str(torch.__version__), captum=captum.__version__, numpy=np.__version__)
    print(json.dumps(dict(plan=plan_summary, total_attributions=sum(len(r) for r in chosen.values()),
                         estimated_array_GiB=estimate / 1024**3, baseline='same 128 TRAIN mean-one-hot'), indent=2), flush=True)
    if args.check_only:
        print('CHECK_ONLY_PASSED: no IG and no output writes.', flush=True)
        return 0
    output = (args.output_dir or task / '2.IG_results' / (
        'stage_activity_IG_' + args.mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    require(not any(output.is_relative_to(p) for p in [inputs, selection, root/'runs', root/'datasets', code]),
            'Output may not be inside any input/model/dependency directory')
    if args.resume:
        require(output.is_dir() and read_json(output/'batch_manifest.json')['signature'] == signature,
                'Resume requires identical code, input files, scope, runtime and numerical settings')
    else:
        require(not output.exists(), f'Refusing existing output directory: {output}')
        parent = output.parent
        while not parent.exists():
            parent = parent.parent
        require(shutil.disk_usage(parent).free > estimate * 1.10 + 100_000_000, 'Insufficient free disk space')
        output.mkdir(parents=True, exist_ok=False)
    import fcntl
    lock = (output/'.writer.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not args.resume:
            np.save(output/'baseline_mean_train_onehot.npy', item['bundle']['baseline'], allow_pickle=False)
            write_tsv(output/'baseline_reference_promoters.tsv', item['bundle']['reference_promoters'])
            (output/'source').mkdir()
            shutil.copyfile(__file__, output/'source'/Path(__file__).name)
            for name in CORE_HASHES:
                shutil.copyfile(code/name, output/'source'/name)
            save_json(output/'batch_manifest.json', dict(signature=signature, started_utc=now(),
                baseline='ONE positional mean of 128 TRAIN one-hot sequences, NOT mean IG over 128 baselines',
                baseline_sha256=sha(output/'baseline_mean_train_onehot.npy'),
                reference_metadata_sha256=sha(output/'baseline_reference_promoters.tsv'),
                target='Each stage has its own scalar predicted raw activity; no other stages in target',
                caveats=['Not TF binding or causal evidence', 'Selection specificity is not the IG target',
                         'Positive/negative IG is relative to the mean-TRAIN-DNA baseline',
                         'Unknown bases remain zero one-hot; mask for categorical motif analyses',
                         'Only quality.tsv accepted=True rows passed numerical QA']))
        batch = read_json(output/'batch_manifest.json')
        require(sha(output/'baseline_mean_train_onehot.npy') == batch['baseline_sha256'] and
                sha(output/'baseline_reference_promoters.tsv') == batch['reference_metadata_sha256'] and
                np.array_equal(np.load(output/'baseline_mean_train_onehot.npy', allow_pickle=False), item['bundle']['baseline']),
                'Shared baseline/reference changed')
        state = dict(status='running', output=str(output), pid=os.getpid(), host=socket.gethostname(),
                     started_or_resumed_utc=now(), total_stages=len(plans), stages=[])
        save_json(output/'status.json', state)
        try:
            for plan in plans:
                result = run_stage(item, plan, chosen[plan['stage']], output/plan['stage_label'].replace(' ', ''), args, core, torch)
                state['stages'].append(result)
                state.update(completed_stages=len(state['stages']), updated_utc=now())
                save_json(output/'status.json', state)
            # Detect sources changing during the run before accepting it as complete.
            require(sha(__file__) == signature['script_sha256'], 'Runner changed during attribution')
            for name, digest in CORE_HASHES.items():
                require(sha(code/name) == digest, 'Dependency changed during attribution')
            require(sha(selection/'manifest.json') == provenance['manifest_sha256'], 'Selection manifest changed')
            for name, digest in provenance['output_sha256'].items():
                require(sha(selection/name) == digest, 'Selection output changed during attribution')
            require(sha(provenance['prediction_file']) == provenance['prediction_sha256'], 'Saved prediction source changed')
            for record in item['sources'].values():
                require(sha(record['path']) == record['sha256'], 'Original model/data source changed')
            for name, digest in item['bundle']['verified_hashes'].items():
                path = root/'datasets/Mouse/target_columns.tsv' if name == 'target_columns.tsv' else inputs/'Mouse'/name
                require(sha(path) == digest, 'Prepared sequence/scaler source changed')
            failed = sum(s['failed_quality_rows'] for s in state['stages'])
            state.update(status=('finished_with_quality_failures' if failed else ('complete' if args.mode == 'full' else 'pilot_complete')),
                         all_rows_passed=failed == 0, failed_quality_rows=failed, finished_utc=now())
            save_json(output/'status.json', state)
            print('FINISHED', json.dumps(state), flush=True)
            return 2 if failed else 0
        except BaseException as exc:
            state.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}',
                         updated_utc=now(), traceback=traceback.format_exc())
            save_json(output/'status.json', state)
            raise
    finally:
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
