#!/usr/bin/env python3
"""Select ONE A/B promoter pair per Mouse Liver gene by highest membership.

Python >=3.9; standard library only.
A clusters: 2,4,5. B clusters: 6,10,1,7,9,3,8.
For each gene with at least one final Mfuzz assignment in EACH group, choose
the highest membership to its OWN assigned cluster within A and within B.
Exact membership ties: smallest numeric promoterId, then lexical ID. Decimal
comparison preserves the precision present in the input membership strings.

Gene mapping comes from all Mouse_Liver_*_rowData.tsv in 1.proactiv_result/Mouse;
annotations must agree across stages. The Mfuzz annotation table is checked
when present. No additional membership/core/activity/major-minor filter is
applied. A/B indicate the requested cluster groups, not promoter activity rank.

Outputs beside this script, in results/Mouse_Liver/:
  Mouse_Liver_representative_promoter_pairs.tsv    one row per gene
  Mouse_Liver_selected_promoters.tsv              two rows per gene
  Mouse_Liver_candidate_promoters_ranked.tsv      all candidates with rank/selected
  selection_summary.json                         rules, counts, source hashes

Usage:
  python3 select_mouse_liver_representative_promoter_pairs.py --check-only
  python3 select_mouse_liver_representative_promoter_pairs.py --expect-genes 183
Existing outputs are not overwritten; use a new --output-dir for another run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

BASE = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo')
GROUPS = {'A': {2, 4, 5}, 'B': {6, 10, 1, 7, 9, 3, 8}}
BASIC = ['promoterId', 'geneId', 'seqnames', 'start', 'strand',
         'internalPromoter', 'promoterPosition', 'txId']
DETAIL = ['promoterId', 'cluster', 'membership', 'is_core', 'seqnames',
          'start', 'strand', 'internalPromoter', 'promoterPosition', 'txId']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pid_key(pid):
    return (0, int(pid), pid) if pid.isdigit() else (1, 0, pid)


def canonical_bool(value):
    value = value.strip().upper()
    require(value in {'TRUE', 'FALSE', 'T', 'F', '1', '0'}, f'Invalid boolean {value!r}')
    return 'TRUE' if value in {'TRUE', 'T', '1'} else 'FALSE'


def records(path, columns):
    """Read regular TSV or R write.table TSV with an extra unnamed row index."""
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header = next(reader, [])
        require(header and len(header) == len(set(header)), f'{path}: invalid header')
        require(set(columns) <= set(header), f'{path}: missing {set(columns) - set(header)}')
        offset = None
        for line, row in enumerate(reader, 2):
            if not row:
                continue
            if offset is None:
                offset = len(row) - len(header)
                require(offset in (0, 1), f'{path}:{line}: invalid row width')
            require(len(row) == len(header) + offset, f'{path}:{line}: inconsistent row width')
            yield dict(zip(header, row[offset:]))


def read_assignments(path):
    result = {}
    for row in records(path, ['promoterId', 'cluster', 'membership', 'is_core']):
        pid = row['promoterId']
        require(pid and pid.strip() == pid and pid not in result, f'{path}: invalid/duplicate promoterId {pid!r}')
        cluster = int(row['cluster'])
        membership = Decimal(row['membership'])
        require(1 <= cluster <= 10, f'{pid}: invalid C10 cluster {cluster}')
        require(membership.is_finite() and 0 <= membership <= 1, f'{pid}: invalid membership')
        result[pid] = dict(promoterId=pid, cluster=cluster, membership=row['membership'],
                           is_core=canonical_bool(row['is_core']))
    require(result, f'{path}: no assignments')
    return result


def read_annotations(paths):
    reference = None
    for path in paths:
        current = {}
        for raw in records(path, BASIC):
            row = {field: raw[field] for field in BASIC}
            pid = row['promoterId']
            require(pid and pid.strip() == pid and pid not in current, f'{path}: invalid/duplicate promoterId {pid!r}')
            require(re.fullmatch(r'ENSMUSG\d+(?:\.\d+)?', row['geneId']), f'{path}: invalid Mouse geneId for {pid}')
            require(row['strand'] in {'+', '-'} and row['start'].isdigit() and int(row['start']) >= 1,
                    f'{path}: invalid strand/TSS for {pid}')
            row['internalPromoter'] = canonical_bool(row['internalPromoter'])
            current[pid] = row
        require(current, f'{path}: empty annotation')
        if reference is None:
            reference = current
        else:
            require(current.keys() == reference.keys(), f'{path}: promoter set differs across stages')
            for pid in current:
                require(current[pid] == reference[pid], f'{path}: conflicting annotation for {pid}')
    return reference


def rank_candidates(pids, assignments):
    return sorted(pids, key=lambda pid: (-Decimal(assignments[pid]['membership']), pid_key(pid)))


def select(assignments, annotations):
    missing = assignments.keys() - annotations.keys()
    require(not missing, f'Unmapped promoters: {sorted(missing, key=pid_key)[:10]}')
    by_gene = defaultdict(lambda: {'A': [], 'B': []})
    for pid, row in assignments.items():
        label = next(group for group, clusters in GROUPS.items() if row['cluster'] in clusters)
        by_gene[annotations[pid]['geneId']][label].append(pid)
    pairs, selected, candidates = [], [], []
    tied_groups = 0
    for gene, bins in sorted(by_gene.items()):
        if not bins['A'] or not bins['B']:
            continue
        pair = {'geneId': gene}
        for label in ['A', 'B']:
            ordered = rank_candidates(bins[label], assignments)
            winner = ordered[0]
            top = Decimal(assignments[winner]['membership'])
            ties = sum(Decimal(assignments[pid]['membership']) == top for pid in ordered)
            tied_groups += ties > 1
            pair[f'candidate_n_{label}'] = len(ordered)
            pair[f'top_membership_tie_n_{label}'] = ties
            for rank, pid in enumerate(ordered, 1):
                item = dict(annotations[pid], **{k: v for k, v in assignments[pid].items() if k != 'promoterId'})
                candidates.append(dict(item, group=label, rank_in_gene_group=rank,
                                       selected='TRUE' if rank == 1 else 'FALSE'))
                if rank == 1:
                    selected.append(dict(item, group=label))
                    pair.update({f'{field}_{label}': item[field] for field in DETAIL})
        require(pair['promoterId_A'] != pair['promoterId_B'], f'{gene}: same promoter selected twice')
        pairs.append(pair)
    require(len(selected) == 2 * len(pairs), 'Not exactly two promoters per gene')
    require(len(selected) == len({r['promoterId'] for r in selected}), 'Repeated selected promoter')
    require(len(pairs) == len({r['geneId'] for r in pairs}), 'Repeated gene in pairs')
    return pairs, selected, candidates, tied_groups


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def calculate(args):
    paths = sorted(args.annotation_dir.glob('Mouse_Liver_*_rowData.tsv'))
    require(paths, f'No Mouse Liver rowData files in {args.annotation_dir}')
    assignment_path = args.mfuzz_dir / 'Liver_assignments.tsv'
    check_path = args.mfuzz_dir / 'Liver_promoter_cluster_annotations.tsv'
    sources = [assignment_path, *paths] + ([check_path] if check_path.is_file() else [])
    stamps = {p: stamp(p) for p in sources}
    assignments, annotations = read_assignments(assignment_path), read_annotations(paths)
    pairs, selected, candidates, ties = select(assignments, annotations)
    if args.expect_genes is not None:
        require(len(pairs) == args.expect_genes, f'Expected {args.expect_genes} genes; observed {len(pairs)}')
    if check_path.is_file():
        checked = set()
        for row in records(check_path, ['promoterId', 'geneId', 'cluster', 'membership']):
            pid = row['promoterId']
            require(pid not in checked and pid in assignments, f'{check_path}: invalid/duplicate promoter {pid}')
            checked.add(pid)
            require(row['geneId'] == annotations[pid]['geneId'], f'{pid}: gene mapping mismatch')
            require(int(row['cluster']) == assignments[pid]['cluster'], f'{pid}: cluster mismatch')
            require(Decimal(row['membership']) == Decimal(assignments[pid]['membership']), f'{pid}: membership mismatch')
        require(checked == assignments.keys(), 'Mfuzz annotation and assignment promoter sets differ')
    provenance = [dict(path=str(p.absolute()), resolved_path=str(p.resolve()),
                       bytes=stamps[p][0], sha256=sha256(p)) for p in sources]
    require(all(stamp(p) == stamps[p] for p in sources), 'Source file changed during run')
    require(paths == sorted(args.annotation_dir.glob('Mouse_Liver_*_rowData.tsv')), 'Stage inventory changed')
    summary = dict(species='Mouse', organ='Liver', cluster_groups={k: sorted(v) for k, v in GROUPS.items()},
                   selection='For each qualifying gene, maximum own-cluster membership within A and within B',
                   tie_break='Exact membership tie: smallest numeric promoterId, then lexical ID',
                   membership_threshold=None, core_only=False, annotation_stage_n=len(paths),
                   mfuzz_assigned_promoter_n=len(assignments), matched_gene_n=len(pairs),
                   selected_pair_n=len(pairs), selected_distinct_promoter_n=len(selected),
                   selected_group_counts=dict(Counter(r['group'] for r in selected)),
                   candidate_promoter_n=len(candidates), unselected_candidate_n=len(candidates) - len(selected),
                   gene_groups_with_top_membership_ties=ties, annotated_table_cross_checked=check_path.is_file())
    manifest = dict(status='complete', created_at=datetime.now(timezone.utc).isoformat(), summary=summary,
                    script_sha256=sha256(Path(__file__)), sources=provenance)
    return pairs, selected, candidates, manifest


def write_tsv(path, rows, fields):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def save(output, pairs, selected, candidates, manifest):
    require(not output.exists(), f'Output exists; provide a new --output-dir: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.representative_pairs_', dir=output.parent) as work:
        ready = Path(work) / 'ready'
        ready.mkdir()
        write_tsv(ready / 'Mouse_Liver_representative_promoter_pairs.tsv', pairs,
                  ['geneId'] + [f'{field}_{label}' for label in ['A', 'B']
                                for field in DETAIL + ['candidate_n', 'top_membership_tie_n']])
        write_tsv(ready / 'Mouse_Liver_selected_promoters.tsv', selected,
                  BASIC + ['cluster', 'membership', 'is_core', 'group'])
        write_tsv(ready / 'Mouse_Liver_candidate_promoters_ranked.tsv', candidates,
                  BASIC + ['cluster', 'membership', 'is_core', 'group', 'rank_in_gene_group', 'selected'])
        (ready / 'selection_summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        require(not output.exists(), f'Output appeared during run: {output}')
        os.rename(ready, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mfuzz-dir', type=Path, default=BASE / '8.time_dynamic/8.2.run_mfuzz/output/Mouse_Liver')
    parser.add_argument('--annotation-dir', type=Path, default=BASE / '1.proactiv_result/Mouse')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results/Mouse_Liver')
    parser.add_argument('--expect-genes', type=int, default=None, help='Optional expected gene count for verification')
    parser.add_argument('--check-only', action='store_true', help='Validate/compute without writing outputs')
    args = parser.parse_args()
    if not args.check_only:
        require(not args.output_dir.exists(), f'Output exists: {args.output_dir}')
    pairs, selected, candidates, manifest = calculate(args)
    print(json.dumps(manifest['summary'], indent=2), flush=True)
    if not args.check_only:
        save(args.output_dir, pairs, selected, candidates, manifest)
        print(f'Saved: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
