#!/usr/bin/env python3
"""Train six independent 10 kb species multi-organ/stage residual dilated CNNs.

One promoter -> all organ/stage .mean targets listed in target_columns.tsv.
Head stability revision: Linear(2560,256) -> LayerNorm(256) -> GELU.
Records per-epoch validation metrics, activation probes and training gradients.
This architecture must be trained from scratch; old checkpoints are incompatible.
Y is log2(activity + 1), then z-scored with TRAIN-only per-task statistics.
Main test metrics remain in original activity units for comparison with older
runs; additional *_log2p1.tsv metrics evaluate log2(activity + 1) units.
Uses existing chromosome splits and requires non-internal, complete-label data.
No further homology/activity filtering. Test labels are loaded
only after validation-based model selection. Requires Python >=3.10, numpy and
PyTorch >=2.2. This file does nothing unless explicitly executed.

Default: queue all six species on cuda:0. Use --devices cuda:0 cuda:1 to run
one independent species per GPU. The backbone, head (except output dimension),
loss, train-only label transform and optimizer defaults match the retained
train_human_multitask_dilated_cnn.py. No weight sharing between species.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit('Required packages: numpy and PyTorch >=2.2. Activate a suitable environment. '
                     f'Original error: {exc}') from exc

LENGTH = 10000
SPECIES = ('Human', 'Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken')
REFERENCE_SHA256 = 'f46179ac826f5b3a3fafab97829ac2a87a81846c286857d09968f10efec9af8e'
TARGET_TRANSFORM = 'log2(activity + 1)'
MODEL_VERSION = 'head_layernorm_v1'
CHECKPOINT_VERSION = 3
# Diagnostic policy, not a task filter: Pearson is unreliable for essentially
# constant predictions. Keep those tasks in the loss and mark their r as NA.
NEAR_CONSTANT_STD_RATIO = 1e-6
NEAR_CONSTANT_ABS_STD = 1e-12
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
META = ('sample_id', 'promoterId', 'geneId', 'seqnames', 'start', 'strand',
        'sequence_sha256', 'non_acgt_bp')
ARCHITECTURE = {
    'input': [4, LENGTH], 'stem_channels': [64, 128], 'stem_kernels': [15, 5],
    'stem_pool_factors': [5, 2], 'normalization': 'GroupNorm', 'groups': 8,
    'activation': 'GELU', 'residual_channels': 128, 'dilations': list(DILATIONS),
    'convolutions_per_block': 2, 'residual_kernel': 3, 'residual_dropout': 0.1,
    'pool_bins': 20, 'hidden_units': 256, 'head_dropout': 0.2,
    'head_normalization': 'LayerNorm(256) before GELU', 'head_norm_eps': 1e-5,
    'model_version': MODEL_VERSION,
    'conv_bias': False, 'output_activation': 'linear',
    'unknown_bases': 'all-zero A/C/G/T encoding', 'homology_filter': False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    # Only replaces the script's own file inside the newly created run directory.
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    os.replace(temp, path)


def write_tsv(path, fields, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'NA' if row.get(k) is None else row.get(k) for k in fields})


def read_targets(data_dir, species):
    """The builder's manifest, not hard-coded stage counts, defines all tasks."""
    path = data_dir/'target_columns.tsv'
    with path.open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        required = {'output_index_0based', 'species', 'organ', 'stage', 'target_column'}
        require(required <= set(reader.fieldnames or []), f'Invalid target manifest: {path}')
        targets = []
        for i, row in enumerate(reader):
            require(int(row['output_index_0based']) == i, 'Manifest indices must be consecutive and ordered')
            require(row['species'] == species and row['organ'] and row['stage'], f'Invalid {species} target metadata')
            require(row['target_column'] == f'{species}_{row["organ"]}_{row["stage"]}.mean',
                    f'Target name differs from organ/stage metadata: {row}')
            targets.append({k: row[k] for k in ('organ', 'stage', 'target_column')})
    require(targets and len({t['target_column'] for t in targets}) == len(targets), 'Empty/duplicate targets')
    return targets


@dataclass
class Table:
    split: str
    columns: list[str]
    rows: list[dict]
    labels: np.ndarray
    table_path: Path
    fasta_path: Path


def read_table(data_dir, split, expected_columns, species):
    path = data_dir / f'{split}.tsv.gz'
    rows, labels, seen = [], [], set()
    with gzip.open(path, 'rt', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)), f'Duplicate table columns: {path}')
        required = {*META, 'species', 'split', 'sequence_length', 'internalPromoter'}
        require(required <= set(fields), f'Missing metadata columns: {required-set(fields)}')
        columns = [x for x in fields if x.endswith('.mean')]
        require(columns == expected_columns, f'Target column/order differs from manifest in {split}')
        for line, row in enumerate(reader, 2):
            require(None not in row and all(v is not None for v in row.values()),
                    f'Ragged TSV row {path}:{line}')
            sid = row['sample_id']
            require(sid not in seen and sid != '', f'Duplicate/empty sample ID: {sid}')
            seen.add(sid)
            require(row['species'] == species and row['split'] == split and sid.startswith(species+':'),
                    f'Wrong split/species: {sid}')
            require(row['internalPromoter'].strip().upper() == 'FALSE',
                    f'Internal/unknown promoter found: {sid}; rebuild the filtered dataset')
            require(int(row['sequence_length']) == LENGTH, f'Wrong length: {sid}')
            require(row['strand'] in ('+', '-'), f'Invalid strand: {sid}')
            require(row['geneId'] not in ('', 'NA'), f'Missing gene ID: {sid}')
            require(re.fullmatch('[0-9a-f]{64}', row['sequence_sha256']), f'Invalid hash: {sid}')
            values = []
            for column in columns:
                raw = row[column].strip()
                require(raw.lower() not in ('', 'na', 'nan'),
                        f'Missing activity {sid}/{column}; rebuild the complete-case dataset')
                value = float(raw)
                require(math.isfinite(value) and value >= 0,
                        f'Invalid activity {sid}/{column}: {raw}')
                values.append(value)
            labels.append(values)
            rows.append({key: row[key] for key in META})
    require(len(rows) > 0, f'Empty input: {path}')
    result = Table(split, columns, rows, np.asarray(labels, dtype=np.float64),
                   path, data_dir / f'{split}.fa.gz')
    logging.info('%s: %d promoters, %d targets, %d missing labels; zeros retained',
                 split, len(rows), len(columns), int(np.isnan(result.labels).sum()))
    return result


def split_check(left, right):
    report = {'left': left.split, 'right': right.split}
    for field in ('sample_id', 'seqnames', 'geneId'):
        overlap = {r[field] for r in left.rows} & {r[field] for r in right.rows}
        require(not overlap, f'{field} leakage between {left.split}/{right.split}: {sorted(overlap)[:5]}')
        report[field + '_overlap'] = 0
    hashes = {r['sequence_sha256'] for r in left.rows} & {r['sequence_sha256'] for r in right.rows}
    report['identical_sequence_hash_overlap'] = len(hashes)
    if hashes:
        logging.warning('%s/%s share %d sequence hashes; retained (no homology/sequence filter)',
                        left.split, right.split, len(hashes))
    return report


def fasta_records(path):
    name, chunks = None, []
    with gzip.open(path, 'rb') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(b'>'):
                if name is not None:
                    yield name, b''.join(chunks).upper()
                name = line[1:].split()[0].decode('ascii')
                chunks = []
            else:
                require(name is not None, f'FASTA sequence without header: {path}')
                chunks.append(line)
        if name is not None:
            yield name, b''.join(chunks).upper()


def load_sequences(table):
    """Keep uint8 nucleotide codes (one byte per base), not full float one-hot."""
    codes = np.empty((len(table.rows), LENGTH), dtype=np.uint8)
    lookup = np.full(256, 4, dtype=np.uint8)
    for i, base in enumerate(b'ACGT'):
        lookup[base] = i
    allowed = np.zeros(256, dtype=bool)
    allowed[list(b'ACGTRYSWKMBDHVN')] = True
    count = 0
    for i, (name, seq) in enumerate(fasta_records(table.fasta_path)):
        require(i < len(table.rows), f'Too many FASTA records: {table.split}')
        row = table.rows[i]
        require(name == row['sample_id'], f'FASTA/TSV order mismatch: {name}/{row["sample_id"]}')
        require(len(seq) == LENGTH, f'Incorrect sequence length: {name}')
        require(hashlib.sha256(seq).hexdigest() == row['sequence_sha256'], f'Sequence hash mismatch: {name}')
        values = np.frombuffer(seq, dtype=np.uint8)
        require(bool(allowed[values].all()), f'Invalid DNA character: {name}')
        codes[i] = lookup[values]
        require(int((codes[i] == 4).sum()) == int(row['non_acgt_bp']), f'Unknown-base count differs: {name}')
        count += 1
    require(count == len(table.rows), f'Too few FASTA records: {table.split}')
    logging.info('%s: all %d sequences verified; uint8 storage %.1f MB',
                 table.split, count, codes.nbytes / 1e6)
    return codes


def transform_activity(values):
    """Numerically stable log2(1 + activity); measured zero remains zero."""
    values = np.asarray(values, dtype=np.float64)
    require(bool(np.isfinite(values).all() and (values >= 0).all()),
            'Log2p1 requires finite, nonnegative activity')
    return np.log1p(values) / np.log(2.0)


def inverse_activity(log_values):
    """Undo log2p1 without clipping; a negative log prediction maps to (-1, 0)."""
    with np.errstate(over='ignore', invalid='ignore'):
        values = np.expm1(np.asarray(log_values, dtype=np.float64) * np.log(2.0))
    require(bool(np.isfinite(values).all()), 'Inverse log2p1 prediction overflow/nonfinite')
    return values


def train_statistics(table):
    count = np.isfinite(table.labels).sum(axis=0)
    require(bool((count >= 2).all()), 'Each target needs at least 2 valid TRAIN labels')
    require(bool(np.isfinite(table.labels).all()), 'Missing/nonfinite TRAIN labels')
    transformed = transform_activity(table.labels)
    mean = np.mean(transformed, axis=0)
    raw_std = np.std(transformed, axis=0, ddof=0)
    require(bool(np.isfinite(mean).all() and np.isfinite(raw_std).all()), 'TRAIN scaler overflow')
    constant = raw_std <= 1e-12
    scale = np.where(constant, 1.0, raw_std)
    if constant.any():
        logging.warning('Constant/near-constant TRAIN targets retained with scale=1: %s',
                        np.asarray(table.columns)[constant].tolist())
    return mean, scale, count, raw_std, constant


class PromoterDataset(Dataset):
    def __init__(self, table, codes, mean, std):
        self.codes = codes
        self.targets = ((transform_activity(table.labels) - mean) / std).astype(np.float32)
        require(bool(np.isfinite(self.targets).all()), 'Nonfinite standardized labels')

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return torch.from_numpy(self.codes[index]), torch.from_numpy(self.targets[index]), index


def one_hot(codes, device):
    codes = codes.to(device=device, dtype=torch.long, non_blocking=True)
    # Unknown code=4 maps to all-zero A/C/G/T; no random bases, no strand augmentation.
    return F.one_hot(codes, num_classes=5)[..., :4].permute(0, 2, 1).contiguous().float()


class ResidualDilatedBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv1d(128, 128, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128, 128, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, 128))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.branch(x))


class PromoterDilatedCNN(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 64, 15, padding=7, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AvgPool1d(5, stride=5),
            nn.Conv1d(64, 128, 5, padding=2, bias=False), nn.GroupNorm(8, 128), nn.GELU(),
            nn.AvgPool1d(2, stride=2))
        self.backbone = nn.Sequential(*(ResidualDilatedBlock(d) for d in DILATIONS))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(20), nn.Flatten(),
                                  nn.Linear(2560, 256), nn.LayerNorm(256, eps=1e-5),
                                  nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(256, n_outputs))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (4, LENGTH):
            raise ValueError(f'Expected B x 4 x {LENGTH}, got {tuple(x.shape)}')
        return self.head(self.backbone(self.stem(x)))


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, args, device, shuffle):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                      drop_last=False, worker_init_fn=seed_worker, generator=generator,
                      persistent_workers=False)


def epoch_loss(model, loader, device, optimizer=None, log_every=200, grad_clip=1.0,
               prediction_buffer=None, gradient_diagnostics=None):
    training = optimizer is not None
    require(prediction_buffer is None or not training, 'Prediction collection is validation-only')
    seen = np.zeros(len(loader.dataset), dtype=bool) if prediction_buffer is not None else None
    model.train(training)
    squared_error, observations = 0.0, 0
    with torch.set_grad_enabled(training):
        for step, (codes, target, indices) in enumerate(loader, 1):
            target = target.to(device, non_blocking=True)
            require(bool(torch.isfinite(target).all().item()), 'Nonfinite target; no masking is used')
            n = target.numel()
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model(one_hot(codes, device))
            require(pred.shape == target.shape, 'Prediction/target shape mismatch; broadcasting is forbidden')
            require(bool(torch.isfinite(pred).all().item()), 'Non-finite model prediction')
            if prediction_buffer is not None:
                index = indices.numpy()
                require(not bool(seen[index].any()), 'Repeated validation prediction index')
                prediction_buffer[index] = pred.detach().cpu().numpy()
                seen[index] = True
            errors = (pred - target).square()
            loss = errors.mean()
            require(bool(torch.isfinite(loss).item()), 'Non-finite MSE')
            if training:
                loss.backward()
                if step == 1 and gradient_diagnostics is not None:
                    # Actual training MSE gradients, BEFORE clipping/optimizer.step.
                    parameters = dict(model.named_parameters())
                    for name in ('stem.0.weight', 'backbone.0.branch.0.weight',
                                 'head.2.weight', 'head.3.weight', 'head.6.weight', 'head.6.bias'):
                        grad = parameters[name].grad
                        value = None if grad is None else float(grad.detach().double().norm().item())
                        require(value is None or math.isfinite(value), f'Nonfinite gradient norm: {name}')
                        gradient_diagnostics[name] = value
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            squared_error += float(errors.detach().double().sum().item())
            observations += n
            if training and step % log_every == 0:
                logging.info('  train batch %d/%d, standardized MSE %.6f',
                             step, len(loader), squared_error / observations)
    require(observations > 0, 'No valid labels in this entire split')
    if prediction_buffer is not None:
        require(bool(seen.all()) and bool(np.isfinite(prediction_buffer).all()),
                'Incomplete validation predictions')
    return squared_error / observations, observations


@torch.no_grad()
def activation_probe(model, codes, device):
    """Fixed validation subset, eval mode; no loader/RNG changes or weight updates."""
    was_training = model.training
    model.eval()
    try:
        x = model.backbone(model.stem(one_hot(codes, device)))
        flattened = model.head[1](model.head[0](x))
        linear = model.head[2](flattened)
        normalized = model.head[3](linear)
        activated = model.head[4](normalized)
        output = model.head[6](model.head[5](activated))
        result = {'n_probe_promoters': len(codes)}
        for name, value in (('hidden_linear', linear), ('pre_gelu_normalized', normalized),
                            ('post_gelu', activated), ('standardized_output', output)):
            require(bool(torch.isfinite(value).all().item()), f'Nonfinite activation: {name}')
            value = value.double()
            result[name] = {'min': float(value.min().item()), 'max': float(value.max().item()),
                            'mean': float(value.mean().item()),
                            'zero_fraction': float((value == 0).double().mean().item()),
                            'across_promoter_std_mean': float(value.std(dim=0, unbiased=False).mean().item())}
        return result
    finally:
        model.train(was_training)


def append_diagnostics(path, record):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def validate_checkpoint(checkpoint, columns):
    require(checkpoint.get('target_columns') == columns, 'Checkpoint target columns differ')
    require(checkpoint.get('target_transform') == TARGET_TRANSFORM and
            checkpoint.get('format_version') == CHECKPOINT_VERSION and
            checkpoint.get('model_version') == MODEL_VERSION,
            'Incompatible checkpoint: the LayerNorm head requires a NEW training run, not an old checkpoint')


def save_checkpoint(path, model, columns, mean, std, epoch, validation_mse, config):
    checkpoint = {
        'format_version': CHECKPOINT_VERSION, 'model_version': MODEL_VERSION,
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'target_transform': TARGET_TRANSFORM, 'scaler_space': TARGET_TRANSFORM,
        'architecture': {**ARCHITECTURE, 'n_outputs': len(columns)}, 'target_columns': columns,
        'label_mean': torch.tensor(mean, dtype=torch.float64),
        'label_std': torch.tensor(std, dtype=torch.float64),
        'epoch': int(epoch), 'validation_mse_standardized': float(validation_mse),
        'config': config,
    }
    temp = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temp)
    os.replace(temp, path)


@torch.no_grad()
def predict(model, loader, device, mean, std):
    """Return log2p1-scale predictions after undoing only the z-score."""
    model.eval()
    result = np.full((len(loader.dataset), len(mean)), np.nan, dtype=np.float64)
    seen = np.zeros(len(result), dtype=bool)
    for codes, _, indices in loader:
        index = indices.numpy()
        require(not bool(seen[index].any()), 'Repeated prediction index')
        standardized = model(one_hot(codes, device)).cpu().numpy().astype(np.float64)
        require(bool(np.isfinite(standardized).all()), 'Non-finite test predictions')
        result[index] = standardized * std + mean
        seen[index] = True
    require(bool(seen.all()) and bool(np.isfinite(result).all()), 'Incomplete test predictions')
    return result


def task_metrics(table, prediction, train_mean, targets, scale='original_activity'):
    require(scale in ('original_activity', 'log2p1'), 'Unknown metric scale')
    truth = transform_activity(table.labels) if scale == 'log2p1' else table.labels
    result = []
    for i, target in enumerate(targets):
        valid = np.isfinite(truth[:, i])
        y, pred = truth[valid, i], prediction[valid, i]
        n = len(y)
        r, r2, rmse, mae, baseline = None, None, None, None, None
        true_std, pred_std, std_ratio = None, None, None
        status = 'insufficient_n'
        if n:
            mse = float(np.mean((pred-y)**2))
            rmse, mae = math.sqrt(mse), float(np.mean(np.abs(pred-y)))
            baseline = math.sqrt(float(np.mean((train_mean[i]-y)**2)))
            yc, pc = y-y.mean(), pred-pred.mean()
            yy, pp = float(yc @ yc), float(pc @ pc)
            true_std, pred_std = math.sqrt(yy / n), math.sqrt(pp / n)
            std_ratio = pred_std / true_std if true_std > 0 else None
            if yy > 0:
                r2 = 1.0 - float(np.sum((pred-y)**2)) / yy
            if n >= 2:
                if yy == 0:
                    status = 'constant_true_activity'
                elif pp == 0:
                    status = 'constant_prediction'
                elif pred_std <= max(NEAR_CONSTANT_ABS_STD, NEAR_CONSTANT_STD_RATIO * true_std):
                    status = 'near_constant_prediction'
                else:
                    r = float(np.clip(float(yc @ pc)/(math.sqrt(yy)*math.sqrt(pp)), -1, 1))
                    status = 'ok' if n > 2 else 'defined_but_only_two_observations'
        result.append({'output_index_0based': i, 'organ': target['organ'], 'stage': target['stage'],
                       'activity_scale': scale,
                       'target_column': table.columns[i], 'n_valid': n, 'pearson_r': r,
                       'pearson_status': status, 'rmse': rmse, 'mae': mae, 'r2': r2,
                       'true_std': true_std, 'prediction_std': pred_std,
                       'prediction_std_over_true_std': std_ratio,
                       'train_mean_baseline_rmse': baseline})
    return result


def save_predictions(path, table, prediction, log_prediction):
    meta = ('sample_id', 'promoterId', 'geneId', 'seqnames', 'start', 'strand', 'non_acgt_bp')
    fields = [*meta, *(f'true::{c}' for c in table.columns), *(f'pred::{c}' for c in table.columns),
              *(f'true_log2p1::{c}' for c in table.columns), *(f'pred_log2p1::{c}' for c in table.columns)]
    log_truth = transform_activity(table.labels)
    with gzip.open(path, 'wt', newline='', compresslevel=1) as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(fields)
        for i, row in enumerate(table.rows):
            truth = ['NA' if not math.isfinite(v) else format(v, '.17g') for v in table.labels[i]]
            pred = [format(v, '.17g') for v in prediction[i]]
            writer.writerow([*(row[k] for k in meta), *truth, *pred,
                             *(format(v, '.17g') for v in log_truth[i]),
                             *(format(v, '.17g') for v in log_prediction[i])])


def record_sources(path, table, manifest):
    for p in (table.table_path, table.fasta_path):
        manifest.append({'split': table.split, 'path': str(p), 'bytes': p.stat().st_size,
                         'sha256': sha256_file(p)})
    save_json(path, manifest)


def summarize_organs(metrics):
    rows = []
    for organ in sorted({m['organ'] for m in metrics}):
        subset = [m for m in metrics if m['organ'] == organ]
        valid = [m['pearson_r'] for m in subset if m['pearson_r'] is not None]
        rows.append({'organ': organ, 'n_tasks': len(subset), 'n_defined_pearson': len(valid),
                     'pearson_macro_mean': float(np.mean(valid)) if valid else None,
                     'pearson_macro_median': float(np.median(valid)) if valid else None})
    return rows


def train_and_evaluate(args, device, out, config):
    # Neither test labels nor test FASTA are opened during model selection.
    targets = read_targets(args.data_dir, args.species)
    columns = [t['target_column'] for t in targets]
    train = read_table(args.data_dir, 'train', columns, args.species)
    validation = read_table(args.data_dir, 'validation', train.columns, args.species)
    checks = [split_check(train, validation)]
    mean, std, counts, raw_std, constant = train_statistics(train)
    original_train_mean = np.mean(train.labels, axis=0)
    require(bool((np.isfinite(validation.labels).sum(axis=0) > 0).all()),
            'Each target needs at least one valid VALIDATION label')
    target_rows = [{'output_index_0based': i, **targets[i],
                    'scaler_space': TARGET_TRANSFORM, 'original_activity_train_mean': float(original_train_mean[i]),
                    'n_train_valid': int(counts[i]), 'train_mean': float(mean[i]),
                    'train_std_ddof0': float(raw_std[i]), 'scaling_std': float(std[i]),
                    'constant_or_near_constant_train': bool(constant[i])}
                   for i, c in enumerate(train.columns)]
    write_tsv(out/'target_columns_and_scaler.tsv', list(target_rows[0]), target_rows)
    manifest_path = args.data_dir/'target_columns.tsv'
    manifest = [{'split': 'task_definition', 'path': str(manifest_path),
                 'bytes': manifest_path.stat().st_size, 'sha256': sha256_file(manifest_path)}]
    record_sources(out/'input_manifest.json', train, manifest)
    record_sources(out/'input_manifest.json', validation, manifest)
    train_set = PromoterDataset(train, load_sequences(train), mean, std)
    val_set = PromoterDataset(validation, load_sequences(validation), mean, std)
    train_loader = make_loader(train_set, args, device, True)
    val_loader = make_loader(val_set, args, device, False)
    model = PromoterDilatedCNN(len(train.columns)).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    require(parameters == 1491840 + 257*len(columns), f'Architecture parameter-count mismatch: {parameters}')
    logging.info('Architecture verified: %s parameters. Training starts ONLY in this explicit invocation.', parameters)
    # Fixed, evenly spaced VALIDATION probes; no TEST access or random draws.
    probe_indices = np.linspace(0, len(val_set)-1, min(16, args.batch_size, len(val_set)), dtype=int)
    probe_codes = torch.from_numpy(val_set.codes[probe_indices].copy())
    probe = activation_probe(model, probe_codes, device)
    append_diagnostics(out/'training_diagnostics.jsonl', {
        'epoch': 0, 'phase': 'initialization', 'validation_probe_indices': probe_indices.tolist(),
        'activation_probe': probe})
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, best_epoch, bad_epochs = float('inf'), 0, 0
    checkpoint_path = out/'best_model.pt'
    with (out/'training_history.tsv').open('x', newline='') as history, \
         (out/'validation_metrics_per_task_log2p1.tsv').open('x', newline='') as val_history:
        writer = csv.writer(history, delimiter='\t', lineterminator='\n')
        writer.writerow(['epoch', 'train_mse_standardized', 'validation_mse_standardized',
                         'train_valid_labels', 'validation_valid_labels', 'seconds', 'best_epoch',
                         'validation_pearson_macro_log2p1', 'validation_defined_pearson_tasks',
                         'validation_constant_prediction_tasks', 'probe_post_gelu_zero_fraction',
                         'first_batch_stem_gradient_norm', 'first_batch_hidden_gradient_norm'])
        for epoch in range(1, args.epochs+1):
            started = time.monotonic()
            gradients = {}
            tr, tr_n = epoch_loss(model, train_loader, device, optimizer, args.log_every, args.grad_clip,
                                  gradient_diagnostics=gradients)
            standardized_val = np.full(validation.labels.shape, np.nan, dtype=np.float64)
            va, va_n = epoch_loss(model, val_loader, device, prediction_buffer=standardized_val)
            val_metrics = task_metrics(validation, standardized_val * std + mean,
                                       mean, targets, scale='log2p1')
            if epoch == 1:
                val_writer = csv.DictWriter(val_history, fieldnames=['epoch', *val_metrics[0]],
                                            delimiter='\t', lineterminator='\n')
                val_writer.writeheader()
            for row in val_metrics:
                val_writer.writerow({'epoch': epoch, **{k: 'NA' if v is None else v for k, v in row.items()}})
            val_history.flush()
            valid_r = [r['pearson_r'] for r in val_metrics if r['pearson_r'] is not None]
            val_r = float(np.mean(valid_r)) if valid_r else None
            collapsed = [r['target_column'] for r in val_metrics
                         if r['pearson_status'] in ('constant_prediction', 'near_constant_prediction')]
            probe = activation_probe(model, probe_codes, device)
            append_diagnostics(out/'training_diagnostics.jsonl', {
                'epoch': epoch, 'phase': 'epoch_end', 'activation_probe': probe,
                'first_training_batch_gradient_norms_before_clipping': gradients,
                'validation_pearson_macro_log2p1': val_r, 'validation_defined_pearson_tasks': len(valid_r),
                'validation_constant_prediction_tasks': collapsed})
            if collapsed:
                logging.warning('VALIDATION: %d/%d outputs are constant/near-constant (Pearson=NA): %s',
                                len(collapsed), len(columns), collapsed[:8])
            if probe['post_gelu']['zero_fraction'] >= 0.99:
                logging.warning('HEAD SATURATION: at least 99%% of probe GELU activations are zero')
            if any(gradients.get(k) in (None, 0.0) for k in ('stem.0.weight', 'head.2.weight')):
                logging.warning('Missing/zero first-batch gradient in backbone or hidden head: %s', gradients)
            if va < best:
                best, best_epoch, bad_epochs = va, epoch, 0
                save_checkpoint(checkpoint_path, model, train.columns, mean, std, epoch, va, config)
            else:
                bad_epochs += 1
            elapsed = time.monotonic()-started
            writer.writerow([epoch, tr, va, tr_n, va_n, elapsed, best_epoch,
                             'NA' if val_r is None else val_r, len(valid_r), len(collapsed),
                             probe['post_gelu']['zero_fraction'],
                             gradients.get('stem.0.weight'), gradients.get('head.2.weight')])
            history.flush()
            logging.info('Epoch %d: train MSE=%.6f; validation MSE=%.6f; best=%d; %.1fs',
                         epoch, tr, va, best_epoch, elapsed)
            logging.info('Validation log2p1 Pearson=%s (%d/%d defined); constant predictions=%d; '
                         'probe GELU zero fraction=%.4f', val_r, len(valid_r), len(columns),
                         len(collapsed), probe['post_gelu']['zero_fraction'])
            save_json(out/'run_status.json', {'status': 'training', 'epoch': epoch,
                                              'best_epoch': best_epoch, 'best_validation_mse': best})
            if bad_epochs >= args.patience:
                logging.info('Early stopping after %d epochs without validation improvement.', bad_epochs)
                break
    del optimizer, train_loader, val_loader, train_set, val_set
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    validate_checkpoint(checkpoint, train.columns)
    model.load_state_dict(checkpoint['model_state_dict'])
    mean, std = checkpoint['label_mean'].numpy(), checkpoint['label_std'].numpy()
    logging.info('Best epoch fixed at %d. Loading TEST data now for final evaluation.', best_epoch)
    save_json(out/'run_status.json', {'status': 'test_evaluation', 'best_epoch': best_epoch})
    test = read_table(args.data_dir, 'test', train.columns, args.species)
    checks.extend([split_check(train, test), split_check(validation, test)])
    save_json(out/'split_integrity.json', checks)
    record_sources(out/'input_manifest.json', test, manifest)
    test_set = PromoterDataset(test, load_sequences(test), mean, std)
    test_loader = make_loader(test_set, args, device, False)
    log_prediction = predict(model, test_loader, device, mean, std)
    prediction = inverse_activity(log_prediction)
    metrics = task_metrics(test, prediction, original_train_mean, targets)
    log_metrics = task_metrics(test, log_prediction, mean, targets, scale='log2p1')
    for scale_name, results in (('original_activity', metrics), ('log2p1', log_metrics)):
        bad = [r['target_column'] for r in results if r['pearson_status'] != 'ok']
        if bad:
            logging.warning('TEST %s: %d tasks need Pearson-status review: %s', scale_name, len(bad), bad[:8])
    for i, row in enumerate(metrics):
        row['constant_or_near_constant_train'] = bool(constant[i])
    write_tsv(out/'test_metrics_per_task.tsv', list(metrics[0]), metrics)
    organ_metrics = summarize_organs(metrics)
    write_tsv(out/'test_metrics_per_organ.tsv', list(organ_metrics[0]), organ_metrics)
    for i, row in enumerate(log_metrics):
        row['constant_or_near_constant_train'] = bool(constant[i])
    write_tsv(out/'test_metrics_per_task_log2p1.tsv', list(log_metrics[0]), log_metrics)
    log_organ_metrics = summarize_organs(log_metrics)
    write_tsv(out/'test_metrics_per_organ_log2p1.tsv', list(log_organ_metrics[0]), log_organ_metrics)
    save_predictions(out/'test_predictions.tsv.gz', test, prediction, log_prediction)
    valid_r = [m['pearson_r'] for m in metrics if m['pearson_r'] is not None]
    log_valid_r = [m['pearson_r'] for m in log_metrics if m['pearson_r'] is not None]
    summary = {
        'species': args.species, 'organs': [m['organ'] for m in organ_metrics], 'n_test_promoters': len(test.rows),
        'n_outputs': len(test.columns), 'n_tasks_with_defined_pearson': len(valid_r),
        'test_pearson_macro_mean': float(np.mean(valid_r)) if valid_r else None,
        'test_pearson_macro_median': float(np.median(valid_r)) if valid_r else None,
        'macro_definition': 'Unweighted mean/median of defined organ-stage task Pearson r; NOT pooled correlation. Organs with more stages contribute more tasks.',
        'pearson_axis': 'Across held-out promoters within each organ-stage task, not across tasks within one promoter',
        'organ_balanced_macro_pearson': float(np.mean([m['pearson_macro_mean'] for m in organ_metrics
               if m['pearson_macro_mean'] is not None])) if valid_r else None,
        'best_epoch': best_epoch, 'best_validation_mse_standardized': best,
        'n_parameters': parameters, 'seed': args.seed,
        'target_transform': TARGET_TRANSFORM,
        'model_version': MODEL_VERSION,
        'near_constant_prediction_policy': {
            'absolute_std_threshold': NEAR_CONSTANT_ABS_STD,
            'std_ratio_threshold': NEAR_CONSTANT_STD_RATIO,
            'effect': 'Mark Pearson NA; retain task in MSE/RMSE/R2 and report defined-task counts'},
        'primary_metric_scale': 'original_activity',
        'prediction_units': 'Original .mean units after inverse z-score then exp2(log_prediction)-1; not clipped',
        'loss_space': 'TRAIN-standardized log2(activity + 1)',
        'test_pearson_macro_mean_log2p1': float(np.mean(log_valid_r)) if log_valid_r else None,
        'test_pearson_macro_median_log2p1': float(np.median(log_valid_r)) if log_valid_r else None,
        'n_tasks_with_defined_pearson_log2p1': len(log_valid_r),
        'test_log2p1_task_results': log_metrics, 'test_log2p1_organ_results': log_organ_metrics,
        'test_used_for_selection': False, 'homology_filter': False,
        'test_task_results': metrics, 'test_organ_results': organ_metrics,
    }
    save_json(out/'test_metrics_summary.json', summary)
    save_json(out/'run_status.json', {'status': 'complete', 'best_epoch': best_epoch,
                                      'completed_at': datetime.now(timezone.utc).isoformat()})
    for row in metrics:
        logging.info('TEST %s/%s Pearson r=%s (n=%d, %s)', row['organ'], row['stage'],
                     'NA' if row['pearson_r'] is None else f'{row["pearson_r"]:.6f}',
                     row['n_valid'], row['pearson_status'])
    logging.info('TEST macro Pearson r (original activity) = %s', summary['test_pearson_macro_mean'])
    logging.info('TEST macro Pearson r (log2p1 activity) = %s', summary['test_pearson_macro_mean_log2p1'])
    logging.info('Completed. Results: %s', out)


def parse_args():
    parent = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--species', nargs='+', choices=SPECIES, default=list(SPECIES))
    parser.add_argument('--data-root', type=Path, default=parent/'datasets')
    parser.add_argument('--data-dir', type=Path, default=None, help='Optional override for ONE selected species only')
    parser.add_argument('--outdir', type=Path, default=None, help='New batch directory; refuses existing directories')
    parser.add_argument('--device', default='cuda:0', help='Single device if --devices is omitted; CPU requires explicit choice')
    parser.add_argument('--devices', nargs='+', help='One concurrent species per distinct device, e.g. cuda:0 cuda:1')
    parser.add_argument('--check-only', action='store_true', help='Check inputs/devices and print plan; no training or writes')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Maximum gradient L2 norm, recorded in run config')
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--num-workers', type=int, default=0, help='0 avoids copying in-memory DNA arrays across spawned workers')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--strict-determinism', action='store_true', help='Fail on unsupported nondeterministic operations')
    args = parser.parse_args()
    require(args.epochs > 0 and args.patience > 0 and args.batch_size > 0 and args.log_every > 0, 'Invalid positive integer argument')
    require(args.lr > 0 and math.isfinite(args.lr) and args.weight_decay >= 0 and math.isfinite(args.weight_decay), 'Invalid optimizer settings')
    require(args.grad_clip > 0 and math.isfinite(args.grad_clip), 'Invalid gradient clipping norm')
    require(0 <= args.seed < 2**32 and args.num_workers >= 0 and args.cpu_threads > 0, 'Invalid seed/worker/thread setting')
    require(len(set(args.species)) == len(args.species), 'Duplicate species')
    require(args.data_dir is None or len(args.species) == 1, '--data-dir requires exactly one species')
    args.data_root = args.data_root.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()
    if args.outdir is not None:
        args.outdir = args.outdir.resolve()
    if args.worker:
        require(len(args.species) == 1 and not args.devices and not args.check_only and args.outdir is not None,
                'Internal worker requires one species, one device, and an explicit output directory')
        args.species = args.species[0]
        args.data_dir = args.data_dir or args.data_root/args.species
    return args


def run_single(args):
    version = re.match(r'(\d+)\.(\d+)', str(torch.__version__))
    require(version and tuple(map(int, version.groups())) >= (2, 2), 'PyTorch >=2.2 required')
    for split in ('train', 'validation', 'test'):
        for suffix in ('tsv.gz', 'fa.gz'):
            require((args.data_dir/f'{split}.{suffix}').is_file(), f'Missing input: {split}.{suffix}')
    targets = read_targets(args.data_dir, args.species)
    build_status = args.data_dir.parent/'build_status.json'
    if build_status.exists():
        require(json.loads(build_status.read_text())['status'] == 'complete',
                'Dataset build is not complete; wait for the dataset builder to finish')
    if args.strict_determinism:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = torch.device(args.device)
    require(device.type in ('cpu', 'cuda'), 'Supported devices: cpu/cuda')
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA is unavailable. Use a GPU node/CUDA PyTorch environment, or explicitly --device cpu.')
        if device.index is None:
            device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.strict_determinism:
        torch.use_deterministic_algorithms(True)
    out = (args.outdir or Path(__file__).resolve().parent /
           f'{args.species}_log2p1_headln_multitask_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}_seed{args.seed}').resolve()
    require(not out.exists(), f'Refusing to overwrite existing run: {out}')
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out/'training.log')], force=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'outdir': str(out), 'python_version': sys.version, 'torch_version': str(torch.__version__),
                   'numpy_version': np.__version__, 'cuda_build': torch.version.cuda,
                   'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                   'architecture': {**ARCHITECTURE, 'n_outputs': len(targets)},
                   'targets': targets, 'script_sha256': sha256_file(__file__),
                   'reference_human_script_sha256': REFERENCE_SHA256,
                   'precision': 'float32', 'loss': 'MSE over all standardized tasks, equal task weight; no masking',
                   'label_transform': 'log2(activity + 1), then train-only per-task z-score; constant train targets use scale=1',
                   'target_transform': TARGET_TRANSFORM, 'primary_metric_scale': 'original_activity',
                   'model_version': MODEL_VERSION, 'checkpoint_version': CHECKPOINT_VERSION,
                   'diagnostics': 'Full-validation per-task metrics each epoch; fixed validation activation probe; first training batch gradient norms before clipping',
                   'near_constant_std_ratio_threshold': NEAR_CONSTANT_STD_RATIO,
                   'near_constant_abs_std_threshold': NEAR_CONSTANT_ABS_STD,
                   'model_selection': 'minimum validation MSE; early stopping; test only after selection'})
    save_json(out/'run_config.json', config)
    save_json(out/'run_status.json', {'status': 'initializing'})
    logging.info('Device: %s (%s); output: %s', device, config['device_name'], out)
    logging.info('Training target: %s, followed by TRAIN-only z-score. Test metrics saved on both scales.', TARGET_TRANSFORM)
    logging.info('Head: Linear(2560,256) -> LayerNorm(256) -> GELU -> Dropout -> Linear; fresh initialization, version=%s', MODEL_VERSION)
    logging.info('One shared CNN, %d output tasks across %d organs/brain-region labels',
                 len(targets), len({t['organ'] for t in targets}))
    if not build_status.exists():
        logging.warning('Dataset build_status.json absent; relying on strict table/FASTA validation')
    try:
        train_and_evaluate(args, device, out, config)
    except BaseException as exc:
        save_json(out/'run_status.json', {'status': 'failed_or_interrupted', 'error': str(exc)})
        logging.exception('Training/evaluation did not complete. Partial files are NOT final results.')
        raise


def normalize_devices(names):
    result = []
    for name in names:
        device = torch.device(name)
        require(device.type in ('cpu', 'cuda'), 'Supported devices: cpu/cuda')
        if device.type == 'cuda':
            require(torch.cuda.is_available(), 'CUDA unavailable; activate PY310 on a GPU node')
            index = 0 if device.index is None else device.index
            require(0 <= index < torch.cuda.device_count(), f'Invalid GPU index: {index}')
            name = f'cuda:{index}'
        else:
            require(device.index is None, 'Use cpu without an index')
            name = 'cpu'
        require(name not in result, f'Duplicate device would oversubscribe one GPU: {name}')
        result.append(name)
    return result


def preflight(args):
    """Check paths/manifests only; never read held-out TEST labels for selection."""
    version = re.match(r'(\d+)\.(\d+)', str(torch.__version__))
    require(version and tuple(map(int, version.groups())) >= (2, 2), 'PyTorch >=2.2 required')
    devices = normalize_devices(args.devices or [args.device])
    plan = []
    for species in args.species:
        data_dir = args.data_dir or args.data_root/species
        for split in ('train', 'validation', 'test'):
            for suffix in ('fa.gz', 'tsv.gz'):
                path = data_dir/f'{split}.{suffix}'
                require(path.is_file() and path.stat().st_size > 0, f'Missing/empty input: {path}')
        targets = read_targets(data_dir, species)
        status_path = data_dir.parent/'build_status.json'
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            require(status.get('status') == 'complete', f'Dataset not complete: {status_path}')
            require(species in status.get('species', []), f'Species absent from build status: {species}')
        plan.append({'species': species, 'data_dir': str(data_dir), 'n_outputs': len(targets),
                     'n_organs': len({t['organ'] for t in targets}),
                     'target_manifest_sha256': sha256_file(data_dir/'target_columns.tsv')})
    return devices, plan


def worker_command(args, script, item, device, out):
    command = [sys.executable, '-u', str(script), '--worker', '--species', item['species'],
               '--data-dir', item['data_dir'], '--device', device, '--outdir', str(out)]
    for key in ('epochs', 'patience', 'batch_size', 'lr', 'weight_decay', 'grad_clip',
                'seed', 'num_workers', 'cpu_threads', 'log_every'):
        command.extend(['--'+key.replace('_', '-'), str(getattr(args, key))])
    if args.strict_determinism:
        command.append('--strict-determinism')
    return command


def request_stop(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}')


def run_batch(args):
    devices, plan = preflight(args)
    out = args.outdir or Path(__file__).resolve().parent/'runs'/(
        f'six_species_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}_seed{args.seed}')
    out = out.resolve()
    require(not out.exists(), f'Refusing to overwrite existing batch: {out}')
    require(not any(out.is_relative_to(Path(item['data_dir'])) for item in plan),
            'Run output must not be inside a species dataset directory')
    preview = {'devices': devices, 'outdir': str(out), 'species': plan,
               'epochs': args.epochs, 'patience': args.patience, 'batch_size': args.batch_size,
               'lr': args.lr, 'loss': 'MSE of train-standardized log2(activity+1)',
               'seed': args.seed, 'independent_species_models': True}
    print(json.dumps(preview, indent=2), flush=True)
    if args.check_only:
        print('PREFLIGHT_OK: no training and no files written.', flush=True)
        return
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    (out/'source').mkdir()
    snapshot = out/'source'/'train_six_species_snapshot.py'
    # Run a frozen source copy so edits to the entrypoint cannot change queued jobs.
    shutil.copy2(Path(__file__).resolve(), snapshot)
    require(sha256_file(snapshot) == sha256_file(__file__), 'Source snapshot mismatch')
    save_json(out/'batch_config.json', {**preview, 'arguments': {
        k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'script_sha256': sha256_file(snapshot), 'reference_human_script_sha256': REFERENCE_SHA256,
        'python_executable': sys.executable, 'architecture': ARCHITECTURE,
        'target_transform': TARGET_TRANSFORM, 'precision': 'float32'})
    status = {'status': 'running', 'pid': os.getpid(),
              'started_at': datetime.now(timezone.utc).isoformat(),
              'species': {item['species']: {'status': 'queued', 'n_outputs': item['n_outputs'],
                          'outdir': str(out/item['species'])} for item in plan}}
    save_json(out/'batch_status.json', status)
    pending, running, failed = list(plan), {}, False
    previous_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        while pending or running:
            if not failed:
                for device in devices:
                    if not pending or device in running:
                        continue
                    item = pending.pop(0)
                    species = item['species']
                    command = worker_command(args, snapshot, item, device, out/species)
                    logfile = (out/'logs'/f'{species}.log').open('x', encoding='utf-8')
                    try:
                        process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT,
                                                   stdin=subprocess.DEVNULL, start_new_session=os.name != 'nt')
                    except BaseException:
                        logfile.close()
                        raise
                    running[device] = (process, logfile, species)
                    status['species'][species].update(status='running', device=device, pid=process.pid,
                                                     command=command)
                    save_json(out/'batch_status.json', status)
                    print(f'START {species} on {device}, PID={process.pid}, outputs={item["n_outputs"]}', flush=True)
            for device, (process, logfile, species) in list(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                logfile.close()
                del running[device]
                completed = False
                status_file = out/species/'run_status.json'
                if returncode == 0 and status_file.is_file():
                    completed = json.loads(status_file.read_text()).get('status') == 'complete'
                state = 'complete' if completed else 'failed'
                failed |= not completed
                status['species'][species].update(status=state, returncode=returncode,
                                                  finished_at=datetime.now(timezone.utc).isoformat())
                save_json(out/'batch_status.json', status)
                print(f'{state.upper()} {species} on {device}; returncode={returncode}', flush=True)
            if failed and not running:
                break
            if pending or running:
                time.sleep(1)
        if failed:
            raise RuntimeError('One or more species failed. Unstarted jobs remain queued; inspect logs. No automatic overwrite/retry.')
        results = []
        for item in plan:
            species = item['species']
            summary = json.loads((out/species/'test_metrics_summary.json').read_text())
            require(summary['species'] == species and summary['n_outputs'] == item['n_outputs'],
                    f'Wrong completed species/output count: {species}')
            results.append({'species': species, 'n_outputs': summary['n_outputs'],
                            'n_test_promoters': summary['n_test_promoters'], 'best_epoch': summary['best_epoch'],
                            'best_validation_mse_standardized': summary['best_validation_mse_standardized'],
                            'test_condition_pearson_log2p1': summary['test_pearson_macro_mean_log2p1'],
                            'test_defined_conditions_log2p1': summary['n_tasks_with_defined_pearson_log2p1'],
                            'test_condition_pearson_original': summary['test_pearson_macro_mean'],
                            'run_directory': str(out/species)})
        write_tsv(out/'species_results_summary.tsv', list(results[0]), results)
        status.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat())
        save_json(out/'batch_status.json', status)
        print(f'ALL_SELECTED_SPECIES_COMPLETE: {out}', flush=True)
    except BaseException as exc:
        # Stop only children created by this invocation; never pkill other users/jobs.
        for process, _, _ in running.values():
            if process.poll() is None:
                if os.name == 'nt':
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
        for process, logfile, species in running.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            logfile.close()
            status['species'][species].update(status='interrupted', returncode=process.returncode)
        status.update(status='failed_or_interrupted', error=str(exc))
        save_json(out/'batch_status.json', status)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)


def main():
    args = parse_args()
    if args.worker:
        signal.signal(signal.SIGTERM, request_stop)
        run_single(args)
    else:
        run_batch(args)


if __name__ == '__main__':
    main()
