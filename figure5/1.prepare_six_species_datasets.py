#!/usr/bin/env python3
"""Build separate six-species, all-organ/all-stage promoter activity datasets.

Only the Python >=3.9 standard library is required. Each promoter is one sample;
each original species_organ_stage.mean column is a separate Y target. First
exclude internal promoters, then exclude entire promoters with any missing Y.
Measured zeros are retained. No mask files are generated. No additional
homology/activity/protein-coding filter is applied. Default chromosome membership
is reused from the retained six-species chromosome map to enable fair comparisons. No model training.
Run --split-mode new only when a new chromosome partition is intentionally wanted.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime, timezone

SPECIES = ('Human', 'Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken')
GENOMES = {
    'Human': ('GRCh37', 'Homo_sapiens.GRCh37.73.dna_sm.toplevel.fa.gz'),
    'Mouse': ('GRCm38', 'Mus_musculus.GRCm38.dna_sm.toplevel.fa.gz'),
    'Rat': ('Rnor_5.0', 'Rattus_norvegicus.Rnor_5.0.dna_sm.toplevel.fa.gz'),
    'Rabbit': ('OryCun2.0', 'Oryctolagus_cuniculus.OryCun2.0.dna_sm.toplevel.fa.gz'),
    'Macaque': ('MMUL_1', 'Macaca_mulatta.MMUL_1.dna_sm.toplevel.fa.gz'),
    'Chicken': ('Galgal4', 'Gallus_gallus.Galgal4.dna_sm.toplevel.fa.gz'),
}
SPLITS = ('train', 'validation', 'test')
META = ('promoterId', 'geneId', 'seqnames', 'start', 'strand', 'internalPromoter', 'promoterPosition', 'txId')
RC = str.maketrans('ACGTRYSWKMBDHVN', 'TGCAYRSWMKVHDBN')

def check(ok, message):
    if not ok:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path, fields, rows):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: 'NA' if row.get(key) is None else row[key] for key in fields})


def discover_targets(root, species):
    paths = sorted((root/species).glob(f'{species}_*_rowData.tsv'))
    check(paths, f'No source rowData files for {species}: {root/species}')
    targets = []
    for index, path in enumerate(paths):
        name = path.name.removesuffix('_rowData.tsv')
        organ_stage = name.removeprefix(species+'_')
        check('_' in organ_stage, f'Cannot parse organ/stage: {path.name}')
        organ, stage = organ_stage.split('_', 1)
        check(organ and stage, f'Empty organ or stage: {path.name}')
        targets.append({'output_index_0based': index, 'species': species, 'organ': organ,
                        'stage': stage, 'target_column': name+'.mean', 'source_file': str(path)})
    check(len({t['target_column'] for t in targets}) == len(targets), 'Duplicate target columns')
    return targets

def read_r_table(path):
    """R write.table may emit unnamed row names without a header placeholder."""
    with path.open(newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader)
        check(len(header) == len(set(header)), f'Duplicate column: {path}')
        for line_no, row in enumerate(reader, 2):
            if len(row) == len(header) + 1:
                row = row[1:]
            check(len(row) == len(header), f'Ragged row {path}:{line_no}')
            yield dict(zip(header, row))

def read_species(root, species):
    targets = discover_targets(root, species)
    promoters, provenance, biological_keys = {}, [], {}
    for index, target_info in enumerate(targets):
        path, label = Path(target_info['source_file']), target_info['target_column']
        source_stat = (path.stat().st_size, path.stat().st_mtime_ns)
        seen = set()
        valid, zero, missing = 0, 0, 0
        for row in read_r_table(path):
            check(all(k in row for k in META) and label in row, f'Missing columns: {path}')
            row['internalPromoter'] = row['internalPromoter'].strip().upper()
            check(row['internalPromoter'] in ('TRUE', 'FALSE'),
                  f'Invalid internalPromoter flag: {path}/{row["promoterId"]}')
            pid = row['promoterId']
            check(pid not in ('', 'NA') and not any(c.isspace() for c in pid), f'Invalid promoter ID: {path}')
            check(row['geneId'] not in ('', 'NA'), f'Missing gene ID: {path}/{pid}')
            check(pid not in seen, f'Duplicate promoter {pid}: {path}')
            seen.add(pid)
            key = (row['geneId'], row['seqnames'], row['start'], row['strand'])
            if key in biological_keys:
                check(biological_keys[key] == pid,
                      f'One gene/TSS/strand has multiple IDs: {species} {key}; '
                      f'{biological_keys[key]} vs {pid}. Resolve annotation identity before merging.')
            biological_keys[key] = pid
            if pid not in promoters:
                promoters[pid] = {k: row[k] for k in META}
                # Retain exact original numeric strings; no float-rounding or transform.
                promoters[pid]['labels'] = [None] * len(targets)
            target = promoters[pid]
            for key in META:
                check(target[key] == row[key],
                      f'Inconsistent {key} across organs/stages: {species} {pid} in {path}; '
                      'do not merge mismatched promoter annotations')
            check(row['strand'] in ('+', '-'), f'Invalid strand: {species} {pid}')
            check(int(row['start']) >= 1, f'Invalid TSS: {species} {pid}')
            value = row[label].strip()
            if value.lower() not in ('', 'na', 'nan'):
                numeric = float(value)
                check(math.isfinite(numeric) and numeric >= 0, f'Invalid activity: {species} {pid}')
                target['labels'][index] = value
                valid += 1
                zero += numeric == 0
            else:
                missing += 1
        check(seen, f'Empty source table: {path}')
        target_info.update({'n_source_promoters': len(seen), 'n_source_valid': valid,
                            'n_source_zero': zero, 'n_source_missing': missing})
        provenance.append({'path': str(path), 'rows': len(seen), 'sha256': file_sha256(path)})
        check((path.stat().st_size, path.stat().st_mtime_ns) == source_stat, f'Source changed while reading: {path}')
        if (index+1) % 20 == 0 or index+1 == len(targets):
            print(f'{species}: read {index+1}/{len(targets)} organ-stage tables', flush=True)
    for t in targets:
        t['n_absent_promoters'] = len(promoters)-t['n_source_promoters']
    return promoters, targets, provenance


def filter_promoters(promoters, targets):
    """Sequential, whole-promoter exclusion; never treat measured zero as NA."""
    stats = {'n_input_promoters': len(promoters), 'n_removed_internal': 0,
             'n_removed_missing_activity_after_internal': 0, 'n_retained_promoters': 0}
    retained = {}
    for pid, row in promoters.items():
        if row['internalPromoter'] == 'TRUE':
            stats['n_removed_internal'] += 1
        elif any(value is None for value in row['labels']):
            stats['n_removed_missing_activity_after_internal'] += 1
        else:
            retained[pid] = row
    stats['n_retained_promoters'] = len(retained)
    check(sum(stats[k] for k in ('n_removed_internal',
              'n_removed_missing_activity_after_internal', 'n_retained_promoters'))
          == stats['n_input_promoters'], 'Filtering counts do not reconcile')
    for j, target in enumerate(targets):
        target['n_retained_valid'] = len(retained)
        target['n_retained_zero'] = sum(float(row['labels'][j]) == 0 for row in retained.values())
    return retained, stats


def assign_chromosomes(args, species, promoters, seed):
    if args.split_mode == 'new':
        return allocate_chromosomes(promoters, args.fractions, seed)
    counts = collections.Counter(row['seqnames'] for row in promoters.values())
    assignments = {}
    with args.chromosome_assignment.open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        check({'species', 'chromosome', 'split'} <= set(reader.fieldnames or []), 'Invalid chromosome assignment file')
        for row in reader:
            if row['species'] != species:
                continue
            chrom = row['chromosome']
            check(chrom not in assignments and row['split'] in SPLITS, 'Duplicate/invalid chromosome assignment')
            assignments[chrom] = row['split']
    missing = set(counts)-set(assignments)
    check(not missing,
          f'{species}: chromosomes absent from supplied split: {sorted(missing)}. '
          'Supply a complete --chromosome-assignment, or explicitly use --split-mode new; no chromosomes are dropped.')
    assigned = {chrom: assignments[chrom] for chrom in counts}
    check(set(assigned.values()) == set(SPLITS), f'{species}: one split has no promoters')
    gene_splits = {}
    for row in promoters.values():
        gene, split = row['geneId'], assigned[row['seqnames']]
        check(gene not in gene_splits or gene_splits[gene] == split,
              f'{species}: gene {gene} crosses fixed chromosome splits; inspect annotations or use --split-mode new')
        gene_splits[gene] = split
    return assigned, counts

def allocate_chromosomes(promoters, fractions, seed):
    """Label-blind randomized search balances promoter counts, not expression.

    Whole chromosomes sharing any geneId are linked before allocation to prevent
    the same gene crossing splits (e.g. duplicated chromosome annotations).
    """
    counts = collections.Counter(row['seqnames'] for row in promoters.values())
    parent = {c: c for c in counts}
    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    gene_chr = {}
    for row in promoters.values():
        g, c = row['geneId'], row['seqnames']
        if g in gene_chr:
            parent[find(c)] = find(gene_chr[g])
        gene_chr[g] = c
    units = collections.defaultdict(list)
    for chrom in sorted(counts):
        units[find(chrom)].append(chrom)
    units = sorted(units.values())
    check(len(units) >= 3, 'Fewer than three independent chromosome groups')
    weights = [sum(counts[c] for c in unit) for unit in units]
    total = sum(weights)
    rng = random.Random(seed)
    best_score, best = float('inf'), None
    # Evaluate contiguous cuts of random chromosome-group permutations. No labels
    # or model performance enter the selection; final membership is reproducible.
    for trial in range(4000):
        order = list(range(len(units)))
        rng.shuffle(order)
        prefix = [0]
        for i in order:
            prefix.append(prefix[-1] + weights[i])
        for a in range(1, len(order)-1):
            for b in range(a+1, len(order)):
                ns = (prefix[a], prefix[b]-prefix[a], total-prefix[b])
                score = sum((n/total-f)**2 for n, f in zip(ns, fractions))
                if score < best_score:
                    best_score = score
                    best = [order[:a], order[a:b], order[b:]]
    assigned = {c: split for split, indices in zip(SPLITS, best) for i in indices for c in units[i]}
    return assigned, counts

def fasta_records(path, wanted):
    """Read verified-length primary records from source gzip, one chromosome at a time."""
    name, chunks, expected = None, [], None
    with gzip.open(path, 'rb') as handle:
        for line in handle:
            if line.startswith(b'>'):
                if name in wanted:
                    seq = b''.join(chunks).decode('ascii').upper()
                    check(expected == len(seq), f'FASTA declared length mismatch: {path} {name}')
                    yield name, seq
                header = line.decode('ascii').strip().split()
                name = header[0][1:]
                chunks = []
                expected = None
                if name in wanted:
                    info = next((t for t in header if t.startswith('chromosome:')), None)
                    check(info is not None, f'No chromosome assembly header: {name}')
                    fields = info.split(':')
                    if info == 'chromosome:GRCh37:Y:2649521:59034049:1':
                        # Ensembl GRCh37 header names the unique Y interval,
                        # but this FASTA record contains the full chromosome
                        # (including PARs). Coordinates must NOT be shifted.
                        # https://grch37.ensembl.org/info/genome/genebuild/assembly.html
                        expected = 59373566
                    else:
                        check(int(fields[-3]) == 1, f'Unexpected chromosome start: {info}')
                        expected = int(fields[-2])
            elif name in wanted:
                chunks.append(line.strip())
        if name in wanted:
            seq = b''.join(chunks).decode('ascii').upper()
            check(expected == len(seq), f'FASTA length mismatch: {name}')
            yield name, seq

def window(sequence, tss1, strand, flank):
    check(1 <= tss1 <= len(sequence), f'TSS out of reference bounds: {tss1}')
    # TSS is always offset flank in the strand-oriented output.
    start = tss1 - 1 - flank if strand == '+' else tss1 - flank
    end = start + 2*flank
    left, right = max(0, -start), max(0, end-len(sequence))
    s = 'N'*left + sequence[max(0, start):min(len(sequence), end)] + 'N'*right
    if strand == '-':
        s = s.translate(RC)[::-1]
    check(len(s) == 2*flank, 'Unexpected window length')
    check(set(s) <= set('ACGTRYSWKMBDHVN'), 'Unexpected reference symbols')
    return s, start, end, left+right

def process(args, species, index):
    promoters, targets, sources = read_species(args.promoter_root, species)
    promoters, filtering = filter_promoters(promoters, targets)
    print(f'{species}: filtering {json.dumps(filtering)}', flush=True)
    check(promoters, f'{species}: no promoters remain after internal/NA filtering')
    columns = [t['target_column'] for t in targets]
    organs = sorted({t['organ'] for t in targets})
    assigned, counts = assign_chromosomes(args, species, promoters, args.seed+index)
    assembly, filename = GENOMES[species]
    reference = args.genome_root / species.lower() / filename
    check(reference.is_file(), f'Missing FASTA: {reference}')
    reference_stat = (reference.stat().st_size, reference.stat().st_mtime_ns)
    grouped = collections.defaultdict(list)
    for pid, row in promoters.items():
        chrom = row['seqnames']
        check(chrom.startswith('chr'), f'Unexpected chromosome name {chrom}')
        grouped[chrom.removeprefix('chr')].append(row)
    dest = args.output/species
    dest.mkdir()
    fields = ['sample_id', 'species', 'split', *META, 'assembly', 'window_start0', 'window_end0',
              'sequence_length', 'padding_bp', 'non_acgt_bp', 'sequence_sha256',
              'n_targets_observed', 'n_organs_observed', *columns]
    handles, writers = {}, {}
    summary = {s: {'promoters': 0, 'genes': set(), 'chromosomes': [], 'padded_promoters': 0,
                   'non_acgt_promoters': 0, 'missing_activity_cells': 0} for s in SPLITS}
    sequence_splits = collections.defaultdict(set)
    coverage = {s: [{'n_valid': 0, 'n_zero': 0, 'min_activity': None, 'max_activity': None}
                    for _ in targets] for s in SPLITS}
    try:
        for split in SPLITS:
            th = gzip.open(dest/f'{split}.tsv.gz', 'wt', newline='', compresslevel=1)
            fh = gzip.open(dest/f'{split}.fa.gz', 'wt', compresslevel=1)
            handles[split] = (th, fh)
            writers[split] = csv.writer(th, delimiter='\t', lineterminator='\n')
            writers[split].writerow(fields)
            summary[split]['chromosomes'] = sorted(c for c in assigned if assigned[c] == split)
        found = set()
        for chrom, seq in fasta_records(reference, set(grouped)):
            check(chrom not in found, f'Duplicate chromosome in FASTA: {chrom}')
            found.add(chrom)
            for row in sorted(grouped[chrom], key=lambda x: (int(x['start']), x['promoterId'])):
                split = assigned[row['seqnames']]
                dna, start, end, padding = window(seq, int(row['start']), row['strand'], args.flank)
                digest = hashlib.sha256(dna.encode()).hexdigest()
                sequence_splits[digest].add(split)
                invalid = len(dna)-sum(dna.count(c) for c in 'ACGT')
                sample_id = species + ':' + row['promoterId']
                labels = row['labels']
                check(row['internalPromoter'] == 'FALSE' and all(v is not None for v in labels),
                      f'Excluded promoter reached export: {sample_id}')
                valid_count = sum(value is not None for value in labels)
                organs_observed = {t['organ'] for t, value in zip(targets, labels) if value is not None}
                writers[split].writerow([sample_id, species, split, *(row[k] for k in META), assembly,
                                        start, end, len(dna), padding, invalid, digest, valid_count,
                                        len(organs_observed), *labels])
                handles[split][1].write(f'>{sample_id}\n{dna}\n')
                stats = summary[split]
                stats['promoters'] += 1
                stats['genes'].add(row['geneId'])
                stats['padded_promoters'] += padding > 0
                stats['non_acgt_promoters'] += invalid > 0
                stats['missing_activity_cells'] += len(targets)-valid_count
                for j, value in enumerate(labels):
                    if value is None:
                        continue
                    numeric = float(value)
                    c = coverage[split][j]
                    c['n_valid'] += 1
                    c['n_zero'] += numeric == 0
                    c['min_activity'] = numeric if c['min_activity'] is None else min(c['min_activity'], numeric)
                    c['max_activity'] = numeric if c['max_activity'] is None else max(c['max_activity'], numeric)
            print(f'{species}: extracted chr{chrom}', flush=True)
        check(found == set(grouped), f'Chromosomes missing in reference: {set(grouped)-found}')
    finally:
        for opened in handles.values():
            for handle in opened:
                handle.close()
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            check(not summary[a]['genes'] & summary[b]['genes'], f'Genes cross splits: {a} {b}')
            check(not set(summary[a]['chromosomes']) & set(summary[b]['chromosomes']), 'Chromosome leakage')
    check(sum(v['promoters'] for v in summary.values()) == len(promoters), 'Promoter loss')
    for split, stats in summary.items():
        stats['genes'] = len(stats['genes'])
        stats['fraction'] = stats['promoters']/len(promoters)
        validate_files(dest, split, fields, targets, args.flank, stats['promoters'], promoters)
    coverage_rows = []
    for j, t in enumerate(targets):
        check(sum(coverage[s][j]['n_valid'] for s in SPLITS) == t['n_retained_valid'], 'Retained label count changed')
        check(sum(coverage[s][j]['n_zero'] for s in SPLITS) == t['n_retained_zero'], 'Retained zero activity count changed')
        for split in SPLITS:
            c = coverage[split][j]
            coverage_rows.append({'split': split, 'output_index_0based': j, 'organ': t['organ'],
                                  'stage': t['stage'], 'target_column': t['target_column'],
                                  'n_promoters': summary[split]['promoters'], **c,
                                  'n_missing': summary[split]['promoters']-c['n_valid']})
    write_tsv(dest/'target_columns.tsv', list(targets[0]), targets)
    write_tsv(dest/'target_coverage.tsv', list(coverage_rows[0]), coverage_rows)
    check((reference.stat().st_size, reference.stat().st_mtime_ns) == reference_stat, f'Reference changed: {reference}')
    result = {'species': species, 'assembly': assembly, 'reference_fasta': str(reference),
              'reference_fasta_bytes': reference_stat[0], 'reference_fasta_mtime_ns': reference_stat[1],
              'n_promoters': len(promoters), 'n_targets': len(targets), 'target_columns': columns,
              'n_organs': len(organs), 'organs': organs,
              'targets_per_organ': dict(collections.Counter(t['organ'] for t in targets)),
              'split_mode': args.split_mode, 'filtering': filtering,
              'input_files': sources, 'splits': summary,
              'identical_sequence_hashes_cross_splits': sum(len(v)>1 for v in sequence_splits.values())}
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_files','target_columns')}, indent=2), flush=True)
    return result, [(species,c,assigned[c],counts[c]) for c in sorted(counts)]

def validate_files(dest, split, fields, targets, flank, expected, promoters):
    """Verify TSV/FASTA alignment, complete labels and exclusion of internal TSS."""
    count, seen = 0, set()
    with gzip.open(dest/f'{split}.tsv.gz', 'rt') as table, gzip.open(dest/f'{split}.fa.gz', 'rt') as fasta:
        reader = csv.DictReader(table, delimiter='\t')
        columns = [t['target_column'] for t in targets]
        check(reader.fieldnames == fields, 'Output header differs')
        for row in reader:
            header, sequence = fasta.readline().strip(), fasta.readline().strip()
            check(header == '>'+row['sample_id'], 'FASTA/TSV ID mismatch')
            check(row['sample_id'] not in seen, 'Duplicate output ID')
            check(len(sequence) == 2*flank, 'Output length mismatch')
            check(hashlib.sha256(sequence.encode()).hexdigest() == row['sequence_sha256'], 'Output hash mismatch')
            check(row['split'] == split, 'Output split mismatch')
            values = promoters[row['promoterId']]['labels']
            check(row['internalPromoter'] == 'FALSE', 'Internal promoter exported')
            check(all(v is not None for v in values), 'Missing activity exported')
            check([row[c] for c in columns] == values, 'Activity changed during export')
            check(int(row['n_targets_observed']) == sum(v is not None for v in values), 'Observed-label count differs')
            check(int(row['n_organs_observed']) == len({t['organ'] for t,v in zip(targets,values) if v is not None}), 'Observed-organ count differs')
            seen.add(row['sample_id']); count += 1
        check(fasta.read() == '', 'Extra FASTA records')
    check(count == expected, 'Output row count mismatch')

def build(args):
    report, assignments = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(args.species))) as pool:
        futures = [pool.submit(process, args, species, SPECIES.index(species)) for species in args.species]
        for future in futures:
            result, chroms = future.result()
            report.append(result); assignments.extend(chroms)
    with (args.output/'chromosome_assignment.tsv').open('x', newline='') as f:
        writer = csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['species','chromosome','split','n_promoters']); writer.writerows(assignments)
    with (args.output/'split_summary.tsv').open('x', newline='') as f:
        writer = csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['species','split','n_promoters','n_genes','n_targets','n_organs','fraction','padded_promoters','non_acgt_promoters','missing_activity_cells','chromosomes'])
        for item in report:
            for split, s in item['splits'].items():
                writer.writerow([item['species'],split,s['promoters'],s['genes'],item['n_targets'],item['n_organs'],s['fraction'],s['padded_promoters'],s['non_acgt_promoters'],s['missing_activity_cells'],','.join(s['chromosomes'])])
    with (args.output/'dataset_metadata.json').open('x') as f:
        json.dump({'seed':args.seed,'target_fractions_new_split_only':args.fractions,'flank':args.flank,
                   'split_mode':args.split_mode,
                   'source_chromosome_assignment':str(args.chromosome_assignment) if args.split_mode=='reuse-reference' else None,
                   'source_chromosome_assignment_sha256':file_sha256(args.chromosome_assignment) if args.split_mode=='reuse-reference' else None,
                   'script_sha256':file_sha256(__file__),
                   'reference_builder_sha256':'9b117bf7ef94b17ed4009819326e98b4011325f29007a4c92877e432e77a372c',
                   'homology_filter':False, 'exclude_internal_promoters':True,
                   'missing_activity_policy':'drop whole promoter if any target is missing',
                   'activity_transform':'none; original stage .mean values',
                   'species':report},f,indent=2)
    filtering_rows = [{'species': item['species'], **item['filtering']} for item in report]
    write_tsv(args.output/'filtering_summary.tsv', list(filtering_rows[0]), filtering_rows)
    (args.output/'README.txt').write_text(
        'One promoter = one sample. Six species remain SEPARATE datasets, not a cross-species pool.\n'
        'Each species has train/validation/test.tsv.gz and matching .fa.gz; join by sample_id. No mask files.\n'
        'Y columns are ALL original species_organ_stage.mean columns, in target_columns.tsv order.\n'
        'Original numeric strings and units preserved; no log transform, scaling or tissue averaging.\n'
        'Target order is lexical, NOT chronological. Raw brain-region names remain distinct.\n'
        'Start from the promoter union across organs/stages. First remove internalPromoter=TRUE.\n'
        'Then remove the entire promoter if ANY target label is NA/NaN/empty or absent from a source.\n'
        'Measured zero, including all-zero promoters, is retained. Activity targets contain no missing values.\n'
        'filtering_summary.tsv records sequential exclusion counts; these are promoters, not genes.\n'
        'Repeated promoter IDs must have identical annotation; duplicate gene/TSS/strand IDs fail instead of merging silently.\n'
        'Entire chromosomes are exclusive to a split; shared gene IDs link chromosomes into indivisible groups.\n'
        'No homolog, orthogroup, or exact-sequence exclusion. Hash overlap diagnostics are in dataset_metadata.json.\n'
        'Same species: genomic windows cannot overlap across splits because chromosomes are disjoint.\n'
        'TSS start is 1-based. window_start0/end0 are 0-based half-open genomic coordinates before clipping.\n'
        'For flank F, output length = 2F, TSS offset = F (0-based) on either strand.\n'
        'This is F upstream bases + TSS + F-1 downstream bases; including both full flanks plus TSS would be 2F+1.\n'
        'Negative-strand windows are reverse-complemented; lower-case soft masking is converted to upper case.\n'
        'Out-of-chromosome bases are N-padded, padding_bp records count. non_acgt_bp includes other ambiguity codes.\n'
        'For one-hot encoding, use A,C,G,T channel order and a documented ambiguity policy; no one-hot files are generated.\n'
        'No activity magnitudes are used to choose chromosomes. Fit any normalization ONLY on train.\n'
        'Use validation for early stopping/tuning; evaluate test only after model selection.\n'
        'Missing activity is never imputed as zero. No protein-coding or homology filter is added.\n'
        'target_coverage.tsv reports per-target/split valid, missing, zero and range statistics.\n'
        'Default: reuse retained six-species chromosome membership; proportions may shift if the promoter union changes.\n'
        'New/unmapped chromosomes cause an error; supply a complete map or explicitly --split-mode new.\n'
        'Metadata records input TSV checksums, genome paths, assemblies, split counts and diagnostics.\n'
        'Training code must read ALL Y columns in target_columns.tsv order and use the species-specific output dimension.\n'
        'This file is written only after all REQUESTED species pass reread verification.\n')
    files = [p for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'build_status.json']
    write_tsv(args.output/'manifest.tsv', ['path','bytes','sha256'],
              ({'path':str(p.relative_to(args.output)), 'bytes':p.stat().st_size,
                'sha256':file_sha256(p)} for p in files))
    print('SUCCESS: all requested datasets verified', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species')
    parser.add_argument('--promoter-root', type=Path, default=base/'proactiv_result')
    parser.add_argument('--genome-root', type=Path, default=base/'ref_annotation/orthologous_promoter/make_chain/genomes')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'datasets')
    parser.add_argument('--split-mode', choices=['reuse-reference', 'new'], default='reuse-reference')
    parser.add_argument('--chromosome-assignment', type=Path,
                        default=base/'revise/train_model/1.build_dataset/datasets/chromosome_assignment.tsv')
    parser.add_argument('--fractions', nargs=3, type=float, default=[0.70,0.15,0.15],
                        help='Used only with --split-mode new; train validation test')
    parser.add_argument('--seed', type=int, default=20260928)
    parser.add_argument('--flank', type=int, default=5000)
    parser.add_argument('--species', nargs='+', choices=SPECIES, default=list(SPECIES))
    parser.add_argument('--workers', type=int, default=2, help='Concurrent species; memory depends on target count')
    parser.add_argument('--check-only', action='store_true', help='Validate paths, targets and split maps; write nothing.')
    args = parser.parse_args()
    for key in ('promoter_root', 'genome_root', 'chromosome_assignment', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    check(args.flank > 0 and all(math.isfinite(x) and x>0 for x in args.fractions)
          and abs(sum(args.fractions)-1)<1e-9, 'Invalid parameters')
    check(args.workers >= 1 and len(args.species) == len(set(args.species)), 'Invalid workers/duplicate species')
    if args.split_mode == 'reuse-reference':
        check(args.chromosome_assignment.is_file(), f'Missing split map: {args.chromosome_assignment}')
    check(not args.output.exists(), f'Output exists; choose a new --output directory: {args.output}')
    preflight = []
    for species in args.species:
        targets = discover_targets(args.promoter_root, species)
        reference = args.genome_root/species.lower()/GENOMES[species][1]
        check(reference.is_file(), f'Missing reference for {species}')
        mapping = {}
        if args.split_mode == 'reuse-reference':
            with args.chromosome_assignment.open(newline='') as handle:
                for row in csv.DictReader(handle, delimiter='\t'):
                    if row['species'] == species:
                        chrom, split = row['chromosome'], row['split']
                        check(chrom not in mapping and split in SPLITS, f'Invalid split map for {species}')
                        mapping[chrom] = split
            check(set(mapping.values()) == set(SPLITS), f'Three splits required for {species}')
        preflight.append({'species': species, 'n_targets': len(targets),
                          'n_organs': len({t['organ'] for t in targets}),
                          'assembly': GENOMES[species][0], 'reference': str(reference),
                          'split_chromosomes': {v: sorted(c for c,s in mapping.items() if s==v) for v in SPLITS}})
    print(json.dumps({'output':str(args.output), 'sequence_length':2*args.flank,
                      'species':preflight}, indent=2), flush=True)
    if args.check_only:
        print('PREFLIGHT_OK: no files written; full source/sequence checks run during build.', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output/'build_status.json'
    status.write_text(json.dumps({'status': 'building', 'species': args.species}, indent=2))
    try:
        build(args)
    except BaseException as error:
        status.write_text(json.dumps({'status': 'failed_or_interrupted', 'error': str(error)}, indent=2))
        raise
    status.write_text(json.dumps({'status': 'complete', 'species': args.species,
                                  'completed_utc': datetime.now(timezone.utc).isoformat()}, indent=2))

if __name__ == '__main__':
    main()
