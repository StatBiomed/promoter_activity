#!/usr/bin/env python3
"""Define Human Major/Minor promoters independently for each organ.

Python >=3.9, standard library only. Reuses the validated calculation functions
from ../define_major_minor_promoters.py WITHOUT changing that reference script.

The only biological change is the averaging population: each organ's conditions
are handled independently, instead of pooling all Human organs. Prenatal and
postnatal stages are BOTH included within an organ; one rowData file has one
equal vote, regardless of its replicate count. This is NOT stage-specific or
prenatal/postnatal-specific classification.

Rules inherited from the reference:
  1. Read raw absolute promoter .mean, not .gene.mean or existing .class.
  2. Remove internalPromoter=TRUE.
  3. Average across ALL conditions of this organ, INCLUDING zero activities.
  4. Remove promoters whose mean in this organ is zero.
  5. Per retained gene, the highest mean is Major; all others are Minor.
     A singleton is Major. Exact positive ties use the smallest numeric
     promoterId (then lexical nonnumeric IDs), not a biological dominance claim.
  6. Missing/negative/nonfinite retained activities, duplicate promoter IDs,
     inconsistent annotations/promoter sets, or changing inputs are errors.

One promoter can be Major in Liver and Minor, or absent, in another organ.
There is no added expression cutoff, log transform, or multi-promoter-only filter.

Default input: ../../1.proactiv_result/Human, relative to THIS script.
Default output: ./results_by_organ, relative to THIS script, with:
  ORGAN/Human_ORGAN_major_minor_promoters.tsv  (reference-compatible 10 columns)
  ORGAN/Human_ORGAN_major_promoters.tsv
  ORGAN/Human_ORGAN_minor_promoters.tsv
  summary.tsv, conditions.tsv, run_manifest.json
The output directory must not exist. No source or previous output is overwritten.

Examples (from try_human_liver):
  python3 define_human_major_minor_by_organ.py --dry-run
  python3 define_human_major_minor_by_organ.py --self-test
  python3 -u define_human_major_minor_by_organ.py
  python3 -u define_human_major_minor_by_organ.py --organs Liver --output-dir liver_only
  python3 -u define_human_major_minor_by_organ.py --expect-conditions Liver=24
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
SUFFIX = '_rowData.tsv'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference(path):
    path = Path(path)
    require(path.is_file(), f'Reference script not found: {path}')
    spec = importlib.util.spec_from_file_location('_major_minor_reference', path)
    require(spec is not None and spec.loader is not None, f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('calculate_species', 'file_stamp', 'OUTPUT', 'self_test'):
        require(hasattr(module, name), f'Reference missing required API: {name}')
    return module


def discover(input_dir, organs=None, expected=None):
    require(input_dir.is_dir(), f'Human input directory not found: {input_dir}')
    grouped = defaultdict(list)
    paths = sorted(input_dir.glob('*' + SUFFIX))
    require(paths, f'No *{SUFFIX} files in {input_dir}')
    require(len({p.resolve() for p in paths}) == len(paths), 'Duplicate symlinked input files')
    for path in paths:
        require(path.is_file() and path.name.startswith('Human_'), f'Invalid Human condition file: {path}')
        stem = path.name[len('Human_'):-len(SUFFIX)]
        organ, separator, stage = stem.rpartition('_')
        require(separator and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', organ)
                and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', stage),
                f'Expected Human_ORGAN_STAGE{SUFFIX}, got {path.name}')
        grouped[organ].append(path)
    selected = sorted(grouped) if organs is None else list(organs)
    require(selected and len(selected) == len(set(selected)), 'Empty or duplicate organ selection')
    require(set(selected) <= set(grouped),
            f'Unknown organs {sorted(set(selected) - set(grouped))}; available: {sorted(grouped)}')
    expected = expected or {}
    require(set(expected) <= set(selected), 'Expected-count organ is not selected')
    result = {organ: grouped[organ] for organ in selected}
    for organ, count in expected.items():
        require(len(result[organ]) == count,
                f'{organ}: expected {count} conditions, found {len(result[organ])}')
    return result


def calculate_organ(reference_path, organ, files, stage_dir):
    ref = load_reference(reference_path)
    destination = Path(stage_dir) / organ
    destination.mkdir()
    stats, stamps, filename = ref.calculate_species('Human_' + organ, files, destination)
    combined = destination / filename
    major = destination / f'Human_{organ}_major_promoters.tsv'
    minor = destination / f'Human_{organ}_minor_promoters.tsv'
    counts = {'Major': 0, 'Minor': 0}
    with combined.open(newline='', encoding='utf-8') as source, \
            major.open('x', newline='', encoding='utf-8') as major_handle, \
            minor.open('x', newline='', encoding='utf-8') as minor_handle:
        reader = csv.DictReader(source, delimiter='\t')
        writers = {label: csv.DictWriter(handle, fieldnames=ref.OUTPUT,
                   delimiter='\t', lineterminator='\n')
                   for label, handle in (('Major', major_handle), ('Minor', minor_handle))}
        for writer in writers.values():
            writer.writeheader()
        for row in reader:
            label = row['promoter_class']
            require(label in writers, f'Unexpected class {label}')
            writers[label].writerow(row)
            counts[label] += 1
    require(counts['Major'] == stats['major_promoters']
            and counts['Minor'] == stats['minor_promoters'], f'{organ}: split row-count mismatch')
    # Check every field of both split tables against the already validated source.
    for label, split in (('Major', major), ('Minor', minor)):
        with combined.open(newline='', encoding='utf-8') as original, \
                split.open(newline='', encoding='utf-8') as subset:
            complete = csv.DictReader(original, delimiter='\t')
            filtered = csv.DictReader(subset, delimiter='\t')
            require(tuple(filtered.fieldnames) == ref.OUTPUT, f'{split}: wrong header')
            for row in complete:
                if row['promoter_class'] == label:
                    require(next(filtered, None) == row, f'{split}: split content differs')
            require(next(filtered, None) is None, f'{split}: unexpected extra rows')
    stats['species'] = 'Human'
    stats['organ'] = organ
    return stats, stamps


def run(input_dir, output_dir, reference_path, organs=None, workers=2, expected=None, dry_run=False):
    require(workers >= 1, '--workers must be >= 1')
    reference_path = Path(reference_path).resolve()
    ref = load_reference(reference_path)
    reference_hash = digest(reference_path)
    inputs = discover(Path(input_dir), organs, expected)
    for organ, files in inputs.items():
        print(f'{organ}: {len(files)} conditions', flush=True)
        print('  ' + ', '.join(p.name for p in files), flush=True)
    print(f'TOTAL: {len(inputs)} organs, {sum(map(len, inputs.values()))} conditions', flush=True)
    if dry_run:
        print('DRY RUN: no output created.', flush=True)
        return []
    output_dir = Path(output_dir).absolute()
    require(not output_dir.exists() and not output_dir.is_symlink(),
            f'Output directory already exists: {output_dir}; choose a NEW --output-dir')
    source_resolved = Path(input_dir).resolve()
    require(output_dir.resolve() != source_resolved and source_resolved not in output_dir.resolve().parents,
            'Output must not be inside the Human source directory')
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    results = {}
    # No final tables are published until every selected organ passes validation.
    with tempfile.TemporaryDirectory(prefix='.organ_staging_', dir=output_dir) as stage:
        if workers == 1:
            for organ, files in inputs.items():
                results[organ] = calculate_organ(reference_path, organ, files, stage)
        else:
            with ProcessPoolExecutor(max_workers=min(workers, len(inputs))) as pool:
                jobs = {pool.submit(calculate_organ, reference_path, organ, files, stage): organ
                        for organ, files in inputs.items()}
                for future in as_completed(jobs):
                    organ = jobs[future]
                    results[organ] = future.result()
                    print(f'VALIDATED {organ}: {results[organ][0]["major_promoters"]} Major, '
                          f'{results[organ][0]["minor_promoters"]} Minor', flush=True)
        require(discover(Path(input_dir), list(inputs), expected) == inputs,
                'Selected condition inventory changed during processing')
        require(digest(reference_path) == reference_hash, 'Reference script changed during processing')
        for organ in inputs:
            for path, stamp in results[organ][1].items():
                require(ref.file_stamp(Path(path)) == stamp, f'Input changed before publication: {path}')
        summaries = [results[organ][0] for organ in inputs]
        keys = ['species', 'organ', 'conditions'] + [k for k in summaries[0]
                                                      if k not in ('species', 'organ', 'conditions')]
        with (Path(stage) / 'summary.tsv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(summaries)
        provenance = []
        for organ, files in inputs.items():
            for path in files:
                size, mtime = results[organ][1][str(path)]
                provenance.append(dict(organ=organ, condition=path.name[:-len(SUFFIX)],
                                       path=str(path.resolve()), size_bytes=size, mtime_ns=mtime))
        with (Path(stage) / 'conditions.tsv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(provenance[0]), delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(provenance)
        manifest = dict(
            status='complete', started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
            script=str(Path(__file__).resolve()), script_sha256=digest(__file__),
            reference_script=str(reference_path), reference_sha256=reference_hash,
            input_dir=str(Path(input_dir).resolve()), output_dir=str(output_dir),
            python_version=sys.version, workers=workers,
            classification_scope='Human, independently per organ, all available developmental stages',
            activity='Raw absolute promoter .mean; equal weight per condition; zero values included',
            filtering='Exclude internal promoters, then within-organ zero-mean promoters',
            classification='One Major per retained gene by highest organ mean; remaining promoters Minor',
            ties='Exact positive ties: smallest numeric promoterId, then lexical nonnumeric ID',
            summary=summaries, inputs=provenance)
        with (Path(stage) / 'run_manifest.json').open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        for organ in inputs:
            (output_dir / organ).mkdir(exist_ok=False)
        # Same-filesystem hard links publish complete files without overwriting.
        files = sorted(p for p in Path(stage).rglob('*') if p.is_file() and p.name != 'run_manifest.json')
        for path in files:
            os.link(path, output_dir / path.relative_to(stage))
        # The complete manifest is the last file published.
        os.link(Path(stage) / 'run_manifest.json', output_dir / 'run_manifest.json')
    print('COMPLETE ' + str(output_dir), flush=True)
    print('VALIDATED_SUMMARY ' + json.dumps(summaries), flush=True)
    return summaries


def self_test(reference_path):
    ref = load_reference(reference_path)
    with redirect_stdout(io.StringIO()):
        ref.self_test()
    with tempfile.TemporaryDirectory(prefix='human_organ_test_') as temporary:
        root = Path(temporary)
        source = root / 'Human'
        source.mkdir()
        # Promoter 1 is Liver Major but Brain Minor; promoter 2 is the reverse.
        annotations = [('1', 'g1', 'FALSE'), ('2', 'g1', 'FALSE'),
                       ('3', 'g2', 'FALSE'), ('4', 'g1', 'TRUE'),
                       ('9', 'tie', 'FALSE'), ('10', 'tie', 'FALSE')]
        values = {'Liver_10w': ['10', '0', '0', 'NA', '.1', '.2'],
                  'Liver_1ypb': ['0', '2', '0', 'NA', '.2', '.1'],
                  'Brain_10w': ['1', '10', '3', 'NA', '1', '1']}
        for index, (key, activity) in enumerate(values.items()):
            condition = 'Human_' + key
            with (source / (condition + SUFFIX)).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow((*ref.OUTPUT[:8], condition + '.mean', condition + '.gene.mean', condition + '.class'))
                records = list(zip(annotations, activity))
                for (pid, gene, flag), value in (records if index == 0 else reversed(records)):
                    row = [pid, gene, 'chr1', '100', '+', flag, '1', 'tx', value, '999', 'Minor']
                    writer.writerow(([pid] if index == 0 else []) + row)
        with redirect_stdout(io.StringIO()):
            summaries = run(source, root / 'out', reference_path, workers=1,
                            expected={'Liver': 2, 'Brain': 1})
        def read_organ(organ):
            with (root / 'out' / organ / f'Human_{organ}_major_minor_promoters.tsv').open() as handle:
                return {r['promoterId']: r for r in csv.DictReader(handle, delimiter='\t')}
        liver, brain = read_organ('Liver'), read_organ('Brain')
        require(liver['1']['promoter_class'] == brain['2']['promoter_class'] == 'Major', 'Organ-specific major failed')
        require(liver['2']['promoter_class'] == brain['1']['promoter_class'] == 'Minor', 'Organ-specific minor failed')
        require(Decimal(liver['1']['mean_promoter_activity']) == 5, 'Zeros excluded from denominator')
        require('3' not in liver and brain['3']['promoter_class'] == 'Major', 'Organ-local zero filter failed')
        require('4' not in liver and '4' not in brain, 'Internal promoter retained')
        require(liver['9']['promoter_class'] == 'Major' and liver['10']['promoter_class'] == 'Minor', 'Tie break failed')
        require(Decimal(liver['9']['mean_promoter_activity']) == Decimal('.15'), 'Decimal precision failed')
        require(len(summaries) == 2 and sum(x['conditions'] for x in summaries) == 3, 'Summary failed')
        with redirect_stdout(io.StringIO()):
            run(source, root / 'dry', reference_path, ['Liver'], dry_run=True)
            require(not (root / 'dry').exists(), 'Dry run created output')
            selected = run(source, root / 'liver_only', reference_path, ['Liver'], workers=1)
            require(len(selected) == 1 and selected[0]['organ'] == 'Liver', 'Organ selection failed')
            for kwargs in ({'output_dir': root / 'out'},
                           {'output_dir': root / 'wrong', 'expected': {'Liver': 3}},
                           {'output_dir': root / 'wrong', 'organs': ['Unknown']},
                           {'output_dir': root / 'wrong', 'organs': ['Liver', 'Liver']}):
                try:
                    run(source, reference_path=reference_path, workers=1, **kwargs)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'Invalid run accepted: {kwargs}')
        # Parallel execution must have identical TSV contents to serial execution.
        with redirect_stdout(io.StringIO()):
            run(source, root / 'parallel', reference_path, workers=2)
        for path in (root / 'out').rglob('*.tsv'):
            require(path.read_bytes() == (root / 'parallel' / path.relative_to(root / 'out')).read_bytes(),
                    f'Parallel results differ: {path.name}')
    print('SELF-TEST PASSED: reference tests; independent organs; stage averaging with zeros; '
          'local zero filtering; internal filtering; positive ties; both TSV layouts; row alignment; '
          'split validation; counts; organ selection; no overwrite; dry run; serial/parallel parity.')


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', type=Path, default=here.parent.parent / '1.proactiv_result' / 'Human')
    parser.add_argument('--output-dir', type=Path, default=here / 'results_by_organ')
    parser.add_argument('--reference-script', type=Path, default=here.parent / 'define_major_minor_promoters.py')
    parser.add_argument('--organs', nargs='+', help='Exact organ names, e.g. Liver Brain; default: all discovered organs')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--expect-conditions', action='append', default=[], metavar='ORGAN=N')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(args.reference_script.resolve())
        return
    expected = {}
    for item in args.expect_conditions:
        organ, separator, count = item.partition('=')
        require(separator and organ and organ not in expected and count.isdigit() and int(count) > 0,
                f'Invalid --expect-conditions {item!r}')
        expected[organ] = int(count)
    run(args.input_dir, args.output_dir, args.reference_script, args.organs,
        args.workers, expected, args.dry_run)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, csv.Error) as error:
        print(f'ERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
