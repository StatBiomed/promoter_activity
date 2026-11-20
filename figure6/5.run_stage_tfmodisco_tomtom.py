#!/usr/bin/env python3
"""Annotate all audited Mouse Liver stage TF-MoDISco motifs using MEME Tomtom.
Python >=3.10; numpy and h5py; a working MEME Suite Tomtom executable.
No model inference or TF-MoDISco rerun. Source files are never modified.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, io, json, math, os, re, shutil, subprocess, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/revise/train_model/redo/explain_development/mouse_liver_rerun")
DEFAULT_BATCH = ROOT / "3.TFMoDISco_results/stage_IG_TFMoDISco_full_20261105T103236372075Z"
DEFAULT_CLUSTER = ROOT / "4.motif_clustering/positive_stage_motifs_20261106T035245054681Z/cluster_patterns_20261106T055337241457Z/all_cluster_patterns.tsv"
DATABASE_NAME = "JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt"
DATABASE_URL = "https://jaspar.elixir.no/download/data/2026/CORE/" + DATABASE_NAME
DEFAULT_DB = ROOT.parents[1] / "model_explanation/reference_motifs" / DATABASE_NAME
BASES = "ACGT"
STAGES = dict(zip(
    ["10.5","11.5","12.5","13.5","14.5","15.5","16.5","17.5","18.5","0dpb","3dpb","2wpb","4wpb","9wpb"],
    ["E10.5","E11.5","E12.5","E13.5","E14.5","E15.5","E16.5","E17.5","E18.5","P0","P3","2wpb","4wpb","9wpb"]))
VERSION = "mouse_liver_stage_tomtom_v1.0"
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    os.replace(temporary, path)

def read_tsv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                f'Missing/duplicate TSV columns: {path}')
        rows = list(reader)
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            f'Ragged TSV: {path}')
    return rows

def write_tsv(path, rows, columns):
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def read_meme(path):
    """Read canonical DNA MEME PPMs; retain database names and original precision."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    require(any(re.match(r'^MEME version\s+\d+', line) for line in lines), f'Not MEME format: {path}')
    require(any(re.fullmatch(r'ALPHABET\s*=\s*ACGT\s*', line) for line in lines),
            f'Explicit ACGT alphabet required: {path}')
    bg = None
    motifs = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Background letter frequencies'):
            parts = {}
            j = i + 1
            while j < len(lines) and len(parts) < 4:
                for base, value in re.findall(r'\b([ACGT])\s+([0-9.eE+\-]+)', lines[j]):
                    require(base not in parts, f'Duplicate background base in {path}')
                    parts[base] = float(value)
                j += 1
                if j < len(lines) and lines[j].startswith('MOTIF '):
                    break
            require(set(parts) == set(BASES), f'Invalid background in {path}')
            bg = np.array([parts[b] for b in BASES], dtype=np.float64)
            require(np.isfinite(bg).all() and np.all(bg > 0) and np.isclose(bg.sum(), 1, atol=1e-5),
                    f'Invalid background frequencies in {path}')
        if line.startswith('MOTIF '):
            fields = line.split()
            require(len(fields) >= 2, f'Missing motif ID in {path}')
            motif_id = fields[1]
            require(motif_id not in motifs and re.fullmatch(r'[A-Za-z0-9_.:+\-]+', motif_id),
                    f'Duplicate/unsafe motif ID: {motif_id}')
            alternate = ' '.join(fields[2:])
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('letter-probability matrix:'):
                require(not lines[i].strip().startswith('MOTIF '), f'No PPM: {motif_id}')
                i += 1
            require(i < len(lines), f'No matrix for {motif_id}')
            header = lines[i]
            width_match = re.search(r'\bw\s*=\s*(\d+)', header)
            alphabet_match = re.search(r'\balength\s*=\s*(\d+)', header)
            require(width_match and alphabet_match and int(alphabet_match[1]) == 4,
                    f'Bad matrix dimensions: {motif_id}')
            width = int(width_match[1])
            require(width > 0 and i + width < len(lines), f'Truncated PPM: {motif_id}')
            try:
                ppm = np.array([[float(x) for x in row.split()] for row in lines[i+1:i+1+width]], dtype=np.float64)
            except ValueError as exc:
                raise ValueError(f'Invalid matrix numbers: {motif_id}') from exc
            require(ppm.shape == (width, 4) and np.isfinite(ppm).all() and np.all(ppm >= 0)
                    and np.allclose(ppm.sum(axis=1), 1, atol=1e-5, rtol=0), f'Invalid PPM: {motif_id}')
            ns = re.search(r'\bnsites\s*=\s*(\S+)', header)
            nsites = float(ns[1]) if ns else None
            if nsites is not None:
                require(math.isfinite(nsites) and nsites > 0, f'Invalid nsites: {motif_id}')
            motifs[motif_id] = dict(tf_name=alternate, ppm=ppm, nsites=nsites, width=width)
            i += width
        i += 1
    require(motifs, f'No motifs in {path}')
    return motifs, bg

def trim_bounds(cwm, threshold=0.3, flank=0):
    """Official TF-MoDISco Tomtom core (NOT the extra-flank logo display crop)."""
    cwm = np.asarray(cwm, dtype=np.float64)
    require(cwm.ndim == 2 and cwm.shape[1] == 4 and len(cwm) > 0
            and np.isfinite(cwm).all(), 'Invalid CWM')
    require(0 <= threshold <= 1 and flank >= 0, 'Invalid trimming settings')
    scores = np.abs(cwm).sum(axis=1)
    require(scores.max() > 0, 'All-zero CWM cannot define a contribution core')
    keep = np.flatnonzero(scores >= scores.max() * threshold)
    return max(0, int(keep[0]) - flank), min(len(scores), int(keep[-1]) + 1 + flank)

def write_query(path, patterns, background):
    with Path(path).open('x', encoding='ascii') as handle:
        handle.write('MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\n')
        handle.write(' '.join(f'{b} {v:.12g}' for b, v in zip(BASES, background)) + '\n\n')
        for p in patterns:
            handle.write(f"MOTIF {p['pattern_id']}\nletter-probability matrix: alength= 4 w= {p['query_width']} nsites= {p['n_seqlets']}\n")
            for row in p['ppm']:
                handle.write(' '.join(f'{v:.12g}' for v in row) + '\n')
            handle.write('\n')

def parse_tomtom(text, query_ids, database):
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith('#')]
    require(lines, 'Tomtom returned no TSV header')
    reader = csv.DictReader(io.StringIO('\n'.join(lines)), delimiter='\t')
    expected = {'Query_ID','Target_ID','Optimal_offset','p-value','E-value','q-value','Overlap','Orientation'}
    require(expected <= set(reader.fieldnames or [])
            and len(reader.fieldnames) == len(set(reader.fieldnames)), 'Unexpected/duplicate Tomtom TSV columns')
    results, seen = [], set()
    for row in reader:
        require(None not in row and all(v is not None for v in row.values()), 'Ragged Tomtom TSV')
        qid, tid = row['Query_ID'], row['Target_ID']
        require(qid in query_ids and tid in database, f'Unknown query/target: {qid}/{tid}')
        require((qid, tid) not in seen, f'Duplicate query-target result: {qid}/{tid}')
        seen.add((qid, tid))
        p, q, e = [float(row[key]) for key in ['p-value','q-value','E-value']]
        require(all(math.isfinite(x) for x in (p,q,e)) and 0 <= p <= 1 and 0 <= q <= 1 and e >= 0,
                f'Invalid Tomtom significance: {qid}/{tid}')
        overlap, offset = int(row['Overlap']), int(row['Optimal_offset'])
        require(overlap > 0 and row['Orientation'] in ('+','-'), 'Invalid Tomtom alignment')
        results.append(dict(pattern_id=qid, matrix_id=tid, tf_name=database[tid]['tf_name'] or tid,
                            p_value=p, e_value=e, q_value=q, overlap=overlap, offset=offset,
                            orientation=row['Orientation'], query_consensus=row.get('Query_consensus',''),
                            target_consensus=row.get('Target_consensus','')))
    return results

def add_cached_metadata(matches, cache):
    """Optional local provenance only; no network calls or species inference."""
    records, hashes = {}, {}
    for target in sorted({r['matrix_id'] for r in matches}):
        path = cache/(target+'.json')
        if path.is_file():
            value = json.loads(path.read_text())
            require(value.get('matrix_id') == target, f'Incorrect cached metadata: {path}')
            records[target] = value
            hashes[str(path.resolve())] = sha256(path)
    for row in matches:
        value = records.get(row['matrix_id'], {})
        row.update(tf_family='; '.join(value.get('family', [])),
                   tf_class='; '.join(value.get('class', [])),
                   reference_species='; '.join(x['name'] for x in value.get('species', [])),
                   metadata_status='cached' if value else 'not_available_locally')
    return hashes

def load_sources(batch):
    """Resolve authoritative attempts and audit input provenance before clustering."""
    batch = Path(batch).resolve(strict=True)
    manifest_path = batch / "batch_manifest.json"
    signature = read_json(manifest_path)["signature"]
    require(signature["version"] == "mouse_liver_top1500_stage_ig_tfmodisco_v1"
            and signature["mode"] == "full", "Expected a full stage-IG TF-MoDISco batch")
    plan = signature["tasks"]
    require([t["stage"] for t in plan] == list(STAGES), "Expected all 14 stages in chronological order")
    batch_status = read_json(batch / "status.json")
    require(batch_status.get("status") == "complete" and
            batch_status.get("completed_stages") == batch_status.get("total_stages") == 14,
            "All 14 upstream stages must be complete; incomplete batches are not silently skipped")
    sources = []
    for target in plan:
        stage, label = target["stage"], target["stage_label"]
        require(label == STAGES[stage] and target["target_column"] == f"Mouse_Liver_{stage}.mean",
                "Stage/target mismatch in batch plan")
        folder = batch / label
        require(not folder.is_symlink(), f"{label}: unexpected symlink stage folder")
        status_path = folder / "status.json"
        status = read_json(status_path)
        require(status.get("status") == "complete" and status["stage"] == stage
                and status["target_column"] == target["target_column"], f"{label}: incomplete/mismatched stage")
        attempt = Path(status["attempt"]).resolve(strict=True)
        require(attempt.parent == (folder / "attempts").resolve(strict=True)
                and re.fullmatch(r"attempt_\d{4,}", attempt.name), f"{label}: unsafe attempt path")
        worker = read_json(attempt / "worker_status.json")
        audit = read_json(attempt / "result_audit.json")
        require(worker.get("status") == "complete" and worker["stage"] == stage,
                f"{label}: worker is not complete for this stage")
        require(audit.get("status") == "passed" and audit == worker["audit"] == status["audit"]
                and audit["species"] == "Mouse" and audit["organ"] == "Liver",
                f"{label}: failed or inconsistent discovery audit")
        require(worker["output_sha256"] == status["output_sha256"], f"{label}: output hash records disagree")
        h5 = attempt / "modisco_results.h5"
        require(h5.resolve(strict=True).parent == attempt, f"{label}: HDF5 escapes its attempt")
        digest = sha256(h5)
        require(digest == status["output_sha256"][h5.name] == audit["hdf5_sha256"],
                f"{label}: HDF5 changed since discovery audit")
        preparation_path = folder / "preparation_manifest.json"
        prep, run = read_json(preparation_path), read_json(folder / "run_manifest.json")
        require(sha256(preparation_path) == run["preparation_manifest_sha256"],
                f"{label}: preparation metadata changed")
        require(run["mode"] == "full" and prep["preparation_status"] == "complete"
                and prep["target_type"] == "single_stage_raw_activity"
                and prep["species"] == "Mouse" and prep["organ"] == "Liver"
                and not prep["is_pilot_subset"] and prep["n_test_promoters"] == 1500
                and prep["discovery_promoters"] + prep["excluded_promoters"] == 1500,
                f"{label}: unexpected source cohort/target")
        for key in ("stage", "stage_label", "target_column", "output_index_0based"):
            require(prep[key] == target[key], f"{label}: preparation {key} mismatch")
        sources.append(dict(**target, hdf5=str(h5), hdf5_sha256=digest,
                            attempt=str(attempt), status_path=str(status_path),
                            status_sha256=sha256(status_path),
                            positive_motifs=int(audit["positive_motifs"]),
                            negative_motifs=int(audit["negative_motifs"]),
                            discovery_promoters=int(prep["discovery_promoters"])))
    return sources


def read_patterns(sources, threshold, min_overlap, sign_selection, cluster_path):
    import h5py
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    cluster = {}
    if cluster_path is not None:
        for row in read_tsv(cluster_path):
            require(row['motif_id'] not in cluster, 'Duplicate cluster motif ID')
            cluster[row['motif_id']] = row
    patterns, matched_cluster_ids = [], set()
    for source in sources:
        with h5py.File(source['hdf5'], 'r') as handle:
            for group, sign, count_key in [('pos_patterns','positive','positive_motifs'),
                                           ('neg_patterns','negative','negative_motifs')]:
                names = list(handle[group]) if group in handle else []
                require(len(names) == source[count_key], 'HDF5 motif count differs from audited count')
                require(all(re.fullmatch(r'pattern_\d+', n) for n in names), 'Unexpected HDF5 pattern name')
                if sign_selection != 'both' and sign != sign_selection:
                    continue
                for name in sorted(names, key=lambda n: int(n.rsplit('_',1)[1])):
                    p = handle[group][name]
                    ppm = np.asarray(p['sequence'][...], dtype=float)
                    cwm = np.asarray(p['contrib_scores'][...], dtype=float)
                    require(ppm.ndim == 2 and ppm.shape[1] == 4 and len(ppm) > 0
                            and ppm.shape == cwm.shape and np.isfinite(ppm).all()
                            and np.isfinite(cwm).all() and np.all(ppm >= 0)
                            and np.allclose(ppm.sum(1), 1, atol=1e-5, rtol=0), 'Invalid sequence/CWM matrices')
                    ppm = ppm / ppm.sum(1, keepdims=True)
                    count = np.asarray(p['seqlets/n_seqlets'][...]).reshape(-1)
                    require(len(count) == 1 and count[0] > 0 and int(count[0]) == count[0], 'Invalid seqlet count')
                    nseq = int(count[0])
                    require(nseq == len(p['seqlets/start']), 'Seqlet support mismatch')
                    motif_id = source['stage_label'] + ('__pos__' if sign == 'positive' else '__neg__') + name
                    zero = bool(np.abs(cwm).sum() == 0)
                    start, end = (0,len(ppm)) if zero else trim_bounds(cwm, threshold, 0)
                    core = ppm[start:end].copy()
                    logs = np.zeros_like(core)
                    np.log2(core, out=logs, where=core > 0)
                    ic = np.maximum(0., 2. + np.sum(core * logs, axis=1))
                    reason = 'zero_cwm' if zero else ('short_core' if len(core) < min_overlap else '')
                    linked = cluster.get(motif_id)
                    if linked:
                        require(Path(linked['source_hdf5']).resolve() == Path(source['hdf5']).resolve()
                                and linked['h5_group'] == group + '/' + name
                                and int(linked['n_seqlets']) == nseq
                                and int(linked['original_length']) == len(ppm), 'Cluster mapping provenance mismatch')
                        matched_cluster_ids.add(motif_id)
                    patterns.append(dict(pattern_id=motif_id, stage=source['stage'], stage_label=source['stage_label'],
                        target_column=source['target_column'], sign=sign, h5_group=group+'/'+name,
                        source_hdf5=source['hdf5'], point_id=linked['point_id'] if linked else '',
                        leiden_cluster=linked['leiden_cluster'] if linked else '', n_seqlets=nseq,
                        original_width=len(ppm), trim_start0=start, trim_end0=end, query_width=len(core),
                        query_consensus=''.join(BASES[i] for i in core.argmax(1)),
                        total_information_bits=float(ic.sum()), informative_positions_over_0_2=int((ic > .2).sum()),
                        low_information_warning=bool((ic > .2).sum() < min_overlap),
                        not_tested_reason=reason, ppm=core))
    require(len({p['pattern_id'] for p in patterns}) == len(patterns), 'Duplicate query IDs')
    if cluster_path is not None and sign_selection != 'negative':
        require(matched_cluster_ids == set(cluster), 'Cluster mapping contains absent or unselected positive motifs')
    return patterns


def summarize_patterns(patterns, matches, q_threshold, min_overlap):
    grouped = {p['pattern_id']: [] for p in patterns}
    for hit in matches:
        require(hit['pattern_id'] in grouped, 'Unknown motif in summary')
        hit['significant_by_tomtom'] = hit['q_value'] < q_threshold
        hit['passes_overlap_filter'] = hit['overlap'] >= min_overlap
        hit['candidate_supported'] = hit['significant_by_tomtom'] and hit['passes_overlap_filter']
        grouped[hit['pattern_id']].append(hit)
    summary, candidates, significant = [], [], []
    for p in patterns:
        hits = sorted(grouped[p['pattern_id']], key=lambda h: (h['q_value'],h['p_value'],-h['overlap'],h['matrix_id']))
        sig = [h for h in hits if h['significant_by_tomtom']]
        passed = [h for h in hits if h['candidate_supported']]
        reason = p.get('not_tested_reason','')
        if reason:
            require(not hits, 'A non-tested motif unexpectedly has comparisons')
            status = 'not_tested_' + reason
        else:
            status = ('candidate_match' if passed else 'significant_short_alignment' if sig else 'no_significant_match')
        best = hits[0] if hits else None
        row = {k:v for k,v in p.items() if k != 'ppm'}
        row.update(match_status=status, n_reported_matches=len(hits), n_significant_by_tomtom=len(sig),
                   n_candidate_matches=len(passed), candidate_tf_names='; '.join(dict.fromkeys(h['tf_name'] for h in passed)),
                   candidate_matrix_ids='; '.join(h['matrix_id'] for h in passed),
                   best_reference_tf=best['tf_name'] if best else '', best_matrix_id=best['matrix_id'] if best else '',
                   best_q_value=best['q_value'] if best else '', best_p_value=best['p_value'] if best else '',
                   best_e_value=best['e_value'] if best else '', best_overlap=best['overlap'] if best else '',
                   best_orientation=best['orientation'] if best else '',
                   best_candidate_tf=passed[0]['tf_name'] if passed else '',
                   best_candidate_q_value=passed[0]['q_value'] if passed else '')
        summary.append(row); candidates.extend(passed); significant.extend(sig)
    return summary, candidates, significant


MATCH_COLUMNS = ['pattern_id','stage_label','sign','point_id','leiden_cluster','n_seqlets','query_width',
    'low_information_warning','matrix_id','tf_name','tf_family','tf_class','reference_species','metadata_status',
    'p_value','e_value','q_value','overlap','offset','orientation','significant_by_tomtom',
    'passes_overlap_filter','candidate_supported','query_consensus','target_consensus']


def run_stage(source, patterns, database, db, executable, out, args):
    label = source['stage_label']
    folder = out / 'stages' / label
    folder.mkdir(parents=True, exist_ok=False)
    tested = [p for p in patterns if not p['not_tested_reason']]
    command, matches = [], []
    if tested:
        query = folder / 'queries_tested.meme'
        write_query(query, tested, [0.25]*4)
        exported, _ = read_meme(query)
        require(set(exported) == {p['pattern_id'] for p in tested}, 'Query IDs changed on export')
        for p in tested:
            require(np.allclose(exported[p['pattern_id']]['ppm'],p['ppm'],atol=1e-10,rtol=0), 'PPM export mismatch')
        command = [str(executable), '-text', '-dist', 'pearson', '-min-overlap', str(args.min_overlap),
                   '-motif-pseudo', '0', '-no-ssc', '-thresh', '1', str(query), str(db)]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (folder/'tomtom.tsv').open('x') as stdout, (folder/'stderr.log').open('x') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, cwd=folder, env=env, timeout=args.timeout)
        require(result.returncode == 0, f'{label}: Tomtom failed; see {folder}/stderr.log')
        matches = parse_tomtom((folder/'tomtom.tsv').read_text(), set(exported), database)
        counts = Counter(m['pattern_id'] for m in matches)
        require(all(counts[p['pattern_id']] == len(database) for p in tested),
                f'{label}: incomplete query x database comparisons; inspect stderr.log before interpreting')
    metadata_hashes = add_cached_metadata(matches, db.parent)
    by_id = {p['pattern_id']:p for p in patterns}
    for hit in matches:
        for key in ['stage_label','sign','point_id','leiden_cluster','n_seqlets','query_width','low_information_warning']:
            hit[key] = by_id[hit['pattern_id']][key]
    summary, candidates, significant = summarize_patterns(patterns,matches,args.q_threshold,args.min_overlap)
    if summary:
        write_tsv(folder/'motif_tf_summary.tsv',summary,list(summary[0]))
    for name, rows in [('candidate_matches.tsv',candidates),('tomtom_significant_matches.tsv',significant)]:
        write_tsv(folder/name,rows,MATCH_COLUMNS)
    with gzip.open(folder/'all_matches_annotated.tsv.gz','xt',encoding='utf-8',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=MATCH_COLUMNS,delimiter='\t',extrasaction='ignore')
        writer.writeheader(); writer.writerows(matches)
    stats = dict(stage_label=label,motifs=len(patterns),tested_motifs=len(tested),reported_comparisons=len(matches),
                 candidate_motifs=sum(r['match_status']=='candidate_match' for r in summary),
                 no_significant_match=sum(r['match_status']=='no_significant_match' for r in summary),
                 not_tested_short_core=sum(r['match_status']=='not_tested_short_core' for r in summary),
                 not_tested_zero_cwm=sum(r['match_status']=='not_tested_zero_cwm' for r in summary),
                 significant_short_alignment=sum(r['match_status']=='significant_short_alignment' for r in summary),
                 candidate_matches=len(candidates),significant_by_tomtom=len(significant))
    save_json(folder/'status.json',dict(status='complete',completed_utc=now(),command=command,
                                      cached_metadata_sha256=metadata_hashes,**stats))
    return summary, candidates, significant, stats


def find_tomtom(requested):
    options = [str(requested)] if requested else [shutil.which('tomtom'),
        '/mnt/ruiyanhou/nfs_share2/software/meme5.5.8_install/bin/tomtom',
        '/mnt/ruiyanhou/nfs_share2/conda_env/envs/PY37/bin/tomtom']
    errors = []
    for candidate in dict.fromkeys(x for x in options if x):
        if not Path(candidate).is_file():
            errors.append(candidate + ': absent'); continue
        result = subprocess.run([candidate,'-version'],capture_output=True,text=True,timeout=30)
        if result.returncode == 0:
            return Path(candidate).resolve(), result.stdout.strip(), errors
        errors.append(candidate + ': ' + result.stderr.strip())
    raise RuntimeError('No working MEME Suite Tomtom. ' + '\n'.join(errors))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-dir',type=Path,default=DEFAULT_BATCH)
    parser.add_argument('--database',type=Path,default=DEFAULT_DB)
    parser.add_argument('--database-receipt',type=Path)
    parser.add_argument('--cluster-index',type=Path,default=DEFAULT_CLUSTER)
    parser.add_argument('--no-cluster-map',action='store_true')
    parser.add_argument('--tomtom',type=Path)
    parser.add_argument('--sign',choices=['both','positive','negative'],default='both')
    parser.add_argument('--q-threshold',type=float,default=.05)
    parser.add_argument('--min-overlap',type=int,default=5)
    parser.add_argument('--trim-threshold',type=float,default=.3)
    parser.add_argument('--jobs',type=int,default=4,help='Concurrent stages; each Tomtom process uses one CPU')
    parser.add_argument('--timeout',type=int,default=3600,help='Seconds per Tomtom stage')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--check-only',action='store_true',help='Read-only full input validation; no search and no output writes')
    args = parser.parse_args()
    require(0 < args.q_threshold <= 1 and 0 < args.trim_threshold <= 1 and args.min_overlap > 0
            and 1 <= args.jobs <= 14 and args.timeout > 0,'Invalid parameters')
    return args


def main():
    args = parse_args()
    sources = load_sources(args.batch_dir)
    db = args.database.resolve(strict=True)
    receipt_path = (args.database_receipt or args.database.with_suffix('.json')).resolve(strict=True)
    receipt = read_json(receipt_path)
    require(receipt.get('sha256') == sha256(db),'Database differs from provenance receipt')
    if db.name == DATABASE_NAME:
        require(receipt.get('url') == DATABASE_URL and str(receipt.get('release')) == '2026','Wrong JASPAR provenance')
    database, _ = read_meme(db)
    executable, tool_version, tool_errors = find_tomtom(args.tomtom)
    cluster = None if args.no_cluster_map else args.cluster_index.resolve(strict=True)
    patterns = read_patterns(sources,args.trim_threshold,args.min_overlap,args.sign,cluster)
    require(patterns,'No motifs selected')
    out = (args.output_dir or ROOT/'5.Tomtom_results'/('stage_JASPAR_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).absolute()
    require(not out.exists() and not out.is_symlink(),'Refusing to overwrite existing output')
    require(not out.resolve().is_relative_to(args.batch_dir.resolve()),'Output cannot be inside source batch')
    inputs = {str(db):sha256(db),str(receipt_path):sha256(receipt_path)}
    if cluster: inputs[str(cluster)] = sha256(cluster)
    for source in sources:
        inputs[source['hdf5']] = source['hdf5_sha256']
        inputs[source['status_path']] = source['status_sha256']
    plan = dict(stages=len(sources),motifs=len(patterns),positive_motifs=sum(p['sign']=='positive' for p in patterns),
        negative_motifs=sum(p['sign']=='negative' for p in patterns),tested_motifs=sum(not p['not_tested_reason'] for p in patterns),
        not_tested_reasons=dict(Counter(p['not_tested_reason'] for p in patterns if p['not_tested_reason'])),
        cluster_mapped_motifs=sum(bool(p['point_id']) for p in patterns),database=str(db),database_motifs=len(database),
        tomtom=str(executable),tomtom_version=tool_version,unusable_tool_candidates=tool_errors,
        q_threshold=args.q_threshold,min_actual_overlap=args.min_overlap,trim_threshold=args.trim_threshold,
        sign_selection=args.sign,jobs=args.jobs,output=str(out))
    print(json.dumps(plan,indent=2),flush=True)
    if args.check_only:
        print('CHECK_ONLY_OK: no files written and no Tomtom search.',flush=True); return
    out.mkdir(parents=True,exist_ok=False)
    status = dict(status='running',pid=os.getpid(),started_utc=now(),completed_stages=0,total_stages=len(sources))
    save_json(out/'status.json',status)
    manifest = dict(version=VERSION,plan=plan,created_utc=now(),input_sha256=inputs,sources=sources,
        script_sha256=sha256(__file__),database_provenance=receipt,
        matrix='Normalized original HDF5 sequence PPM, ACGT columns; never signed contributions as PPM',
        trim='sum(abs(CWM),axis=1) >= 0.3*maximum by default; first/last passing bases, flank=0',
        difference_from_clustering='Clustering used PPM IC>0.2 with AT=.27/CG=.23, NOT this CWM crop',
        short_motifs='Core shorter than min_overlap is retained in index/summary but not tested',
        information_warning='Fewer than min_overlap positions exceed 0.2 bits versus uniform background; warning only',
        selection='All selected top-level pos/neg patterns from audited final attempts; no child subpatterns',
        statistics='Official Tomtom p/E/q retained; q<threshold plus actual overlap filter for candidates. No pooled global-FDR claim.',
        sign_interpretation='Positive/negative contribution to each stage predicted raw activity relative to baseline; NOT TF activator/repressor',
        interpretation='Candidate motif similarity, not demonstrated binding, occupancy, causality, or unique TF identity',
        database_scope='JASPAR CORE vertebrates; reference names/complexes retained, not mouse-only TF assignments',
        method_sources=['https://meme-suite.org/meme/doc/tomtom.html',
            'https://raw.githubusercontent.com/jmschrei/tfmodisco-lite/main/modiscolite/report.py'])
    save_json(out/'manifest.json',manifest)
    try:
        index = [{k:v for k,v in p.items() if k!='ppm'} for p in patterns]
        write_tsv(out/'query_motif_index.tsv',index,list(index[0]))
        results = {}
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(run_stage,s,[p for p in patterns if p['stage']==s['stage']],database,db,executable,out,args):s['stage_label'] for s in sources}
            for future in as_completed(futures):
                label = futures[future]
                results[label] = future.result()
                status['completed_stages'] = len(results); save_json(out/'status.json',status)
                print('STAGE_COMPLETE',json.dumps(results[label][3]),flush=True)
        summary, candidates, significant, stage_counts = [], [], [], []
        for source in sources:
            a,b,c,d = results[source['stage_label']]
            summary.extend(a);candidates.extend(b);significant.extend(c);stage_counts.append(d)
        require(len(summary)==len(patterns) and {r['pattern_id'] for r in summary}=={p['pattern_id'] for p in patterns},'Summary lost motifs')
        write_tsv(out/'motif_tf_summary.tsv',summary,list(summary[0]))
        for name, rows in [('positive_motif_tf_summary.tsv',[r for r in summary if r['sign']=='positive']),
                           ('negative_motif_tf_summary.tsv',[r for r in summary if r['sign']=='negative']),
                           ('motifs_without_supported_match.tsv',[r for r in summary if r['match_status']!='candidate_match'])]:
            write_tsv(out/name,rows,list(summary[0]))
        write_tsv(out/'candidate_matches.tsv',candidates,MATCH_COLUMNS)
        write_tsv(out/'tomtom_significant_matches.tsv',significant,MATCH_COLUMNS)
        write_tsv(out/'stage_summary.tsv',stage_counts,list(stage_counts[0]))
        require(all(sha256(p)==h for p,h in inputs.items()),'Inputs changed during run')
        totals = dict(motifs=len(summary),tested_motifs=plan['tested_motifs'],positive_motifs=plan['positive_motifs'],
            negative_motifs=plan['negative_motifs'],candidate_motifs=sum(r['match_status']=='candidate_match' for r in summary),
            candidate_matches=len(candidates),tomtom_significant_matches=len(significant),
            status_counts=dict(Counter(r['match_status'] for r in summary)))
        save_json(out/'result_summary.json',totals)
        with (out/'README.txt').open('x',encoding='utf-8') as handle:
            handle.write(RESULT_README)
        manifest.update(completed_utc=now(),results=totals,output_sha256={str(p.relative_to(out)):sha256(p) for p in out.rglob('*')
            if p.is_file() and p.name not in ('manifest.json','status.json')})
        save_json(out/'manifest.json',manifest)
        status.update(status='complete',completed_utc=now(),**totals);save_json(out/'status.json',status)
        print('COMPLETE',json.dumps(dict(output=str(out),**totals)),flush=True)
    except BaseException as exc:
        status.update(status='failed',failed_utc=now(),error=repr(exc));save_json(out/'status.json',status)
        raise


RESULT_README = '''Start with motif_tf_summary.tsv: exactly one row per original selected motif, including no-match and untested motifs.
positive_motif_tf_summary.tsv links positive patterns to the existing Pxxx and cluster labels; negative patterns have no such labels.
candidate_matches.tsv retains every q<0.05 AND actual overlap>=5 candidate by default, not only the best reference.
tomtom_significant_matches.tsv also retains q-significant alignments too short to meet the overlap criterion.
motifs_without_supported_match.tsv separates no_significant_match, significant_short_alignment, not_tested_short_core, not_tested_zero_cwm.
best_reference_tf is the nearest reference even when nonsignificant; it is NOT automatically an assignment. Consult match_status.
Low-information warnings are retained; a significant match is not necessarily a unique or experimentally supported TF identity.
stages/<stage>/ contains raw tomtom.tsv, stderr.log, queries_tested.meme, complete annotated comparisons (gzip TSV), and stage summaries.
query_motif_index.tsv records source HDF5, stage, sign, exact CWM crop, support count, information and mapping to clustering IDs.
CWM abs-sum 0.3 core crop is the TF-MoDISco Tomtom rule; prior clustering used IC cropping. The queries therefore differ.
PPMs, not signed CWMs, are compared. Reference and reverse-complement orientations are tested. Pearson, pseudocount=0.
Tomtom q-values are tool outputs, not claimed to be a pooled global FDR across all stage x pattern x reference tests.
Positive/negative means contribution to that stage's predicted raw promoter activity relative to baseline, not activation/repression by TF.
JASPAR 2026 CORE vertebrates non-redundant is not a mouse-only database. A motif match does not prove TF binding or causality.
Source files were not modified; all input hashes, commands, versions, parameters and output hashes are recorded.
'''


if __name__ == '__main__':
    main()
