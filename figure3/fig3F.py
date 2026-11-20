#!/usr/bin/env python3
"""redo8: all-assigned dynamic MP promoter genes -> cluster enrichment.

Run with PY310/bin/python run_cluster_gene_enrichment.py.
Default: seven Mouse organs plus Human/Rat Liver; GO BP/CC/MF + KEGG + Reactome.
Use --check-only for a read-only preflight; --self-test for offline numerical tests.

Foreground = unique Ensembl genes of ALL hard-assigned promoters in a cluster,
not only the membership > 0.8 subset used for the stage-activity heatmap.
Background = unique genes of ALL tested MP promoters in the same maSigPro target.
Each annotation family's universe is defined BEFORE the term-size filter.
All eligible terms, including zero-hit terms, enter cluster x family BH correction.
This is gene-level over-representation, not a promoter-specific function test.

Heatmap columns follow 8.3's matrix verbatim. No column re-clustering, dendrogram,
branch annotations, or hard-coded paper terms. Colors: -log10(BH FDR), with tested
non-significant/too-few-hit cells set to zero; untested cells are gray/NA.
Output: one vector PDF, full tests, significant tests, plotting matrix, selected
term metadata, gene mapping and cluster summary per target, plus one run manifest.
Existing output directories are NEVER overwritten. No upstream files are changed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import urllib.request

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
REDO = ROOT.parent
PROJECT = Path('/mnt/ruiyanhou/nfs_share2/proactiv_redo')
OLD = PROJECT / '8.alternative_promoter_usage_development/8.4.GO_analysis'
LIBRARY = Path('/mnt/ruiyanhou/nfs_share2/conda_env/envs/R433/lib/R/library')
FAMILIES = ['GO_BP', 'GO_CC', 'GO_MF', 'KEGG', 'Reactome']
TARGETS = ['Mouse_' + x for x in ['Brain', 'Cerebellum', 'Heart', 'Kidney',
           'Liver', 'Ovary', 'Testis']] + ['Human_Liver', 'Rat_Liver']
SPECIES = {
    'Mouse': dict(org='Mm', prefix='ENSMUSG', taxid='10090', kegg='mmu',
                  reactome='R-MMU-', latin='Mus musculus', old_cache='kegg_mouse'),
    'Human': dict(org='Hs', prefix='ENSG', taxid='9606', kegg='hsa',
                  reactome='R-HSA-', latin='Homo sapiens', old_cache='kegg'),
    'Rat': dict(org='Rn', prefix='ENSRNOG', taxid='10116', kegg='rno',
                reactome='R-RNO-', latin='Rattus norvegicus', old_cache='kegg_rat'),
}
GO_ROOTS = {'GO:0008150', 'GO:0003674', 'GO:0005575'}
RESULT_COLUMNS = ['target', 'cluster', 'family', 'ID', 'Description', 'overlap_n',
                  'foreground_n', 'term_background_n', 'background_n', 'GeneRatio',
                  'BgRatio', 'fold_enrichment', 'pvalue', 'p_adjust_BH', 'hit_genes']
TERM_COLUMNS = ['family', 'ID', 'Description', 'label', 'minimum_p_adjust_BH',
                'strongest_cluster', 'significant_clusters']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_db(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def normalize_genes(series, species):
    require(series.notna().all(), f'{species}: missing gene ID')
    clean = series.astype(str).str.replace(r'\.\d+$', '', regex=True)
    require(clean.str.fullmatch(SPECIES[species]['prefix'] + r'\d+').all(),
            f'{species}: unexpected Ensembl gene ID (no cross-species substitution)')
    return clean


def bh_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid P values')
    if not len(p):
        return p.copy()
    order = np.argsort(p, kind='stable')
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(q, 1.)
    return result


def checked_source(path, manifest, relative_root, sources):
    require(path.is_file(), f'Missing input: {path}')
    key = str(path.relative_to(relative_root)) if relative_root else str(path.resolve())
    digest = sha256(path)
    require(manifest['output_sha256'].get(key) == digest, f'Stale upstream file: {path}')
    sources[str(path.resolve())] = digest
    return path


def prepare_target(target, a, m1, m2, sources):
    require(target in TARGETS, f'Unsupported target: {target}')
    species, organ = target.split('_', 1)
    require(target in m2['targets'], f'Mfuzz manifest does not contain {target}')
    ap = a.mfuzz / 'output' / target / f'{organ}_promoter_cluster_annotations.tsv'
    bp = a.masigpro / 'output' / target / f'{organ}_all_promoters_statistics.tsv'
    cp = a.mfuzz / 'output' / target / f'{organ}_core_profiles_zscore.csv'
    hp = a.heatmap / 'output' / target / f'{target}_C10_heatmap_matrix.tsv'
    for path, manifest, root in [(ap, m2, a.mfuzz), (cp, m2, a.mfuzz), (bp, m1, None)]:
        checked_source(path, manifest, root, sources)
    require(hp.is_file(), f'Run 8.3 first; missing cluster-order matrix: {hp}')
    sources[str(hp.resolve())] = sha256(hp)
    ann = pd.read_csv(ap, sep='\t', dtype={'promoterId': str, 'geneId': str})
    bg = pd.read_csv(bp, sep='\t', dtype={'promoterId': str, 'geneId': str})
    for frame, name in [(ann, 'Mfuzz'), (bg, 'maSigPro')]:
        require({'promoterId', 'geneId', 'padj_BH'} <= set(frame), f'{target}: bad {name} schema')
        require(frame.promoterId.notna().all() and frame.promoterId.is_unique,
                f'{target}: duplicate/missing {name} promoter IDs')
        frame['geneId'] = normalize_genes(frame.geneId, species)
    require({'cluster', 'membership'} <= set(ann), f'{target}: missing cluster assignment')
    require(np.isfinite(ann.cluster).all() and (ann.cluster == ann.cluster.astype(int)).all(),
            f'{target}: non-integer cluster assignment')
    ann['cluster'] = ann.cluster.astype(int)
    require(set(ann.cluster) == set(range(1, 11)), f'{target}: expected 10 nonempty clusters')
    require(ann.membership.between(0, 1).all(), f'{target}: invalid membership')
    require(ann.padj_BH.notna().all() and ann.padj_BH.between(0, .01, inclusive='left').all(),
            f'{target}: Mfuzz must contain strictly FDR < 0.01 promoters')
    expected = bg.loc[bg.padj_BH.notna() & (bg.padj_BH < .01), 'promoterId']
    require(set(ann.promoterId) == set(expected), f'{target}: Mfuzz and current maSigPro dynamic set differ')
    gene_map = bg.set_index('promoterId').geneId
    require(ann.geneId.eq(ann.promoterId.map(gene_map)).all(), f'{target}: promoter-gene mapping mismatch')
    require(bg.groupby('geneId').promoterId.nunique().ge(2).all(), f'{target}: background includes non-MP genes')
    if 'internalPromoter' in bg:
        require(bg.internalPromoter.astype(str).str.lower().eq('false').all(), f'{target}: internal promoter retained')
    matrix = pd.read_csv(hp, sep='\t', index_col=0)
    core = pd.read_csv(cp, index_col=0)
    require(core.shape[0] == 10 and core.index.astype(str).tolist() ==
            [f'Cluster_{x}' for x in range(1, 11)], f'{target}: unexpected core-profile layout')
    require(matrix.shape == (core.shape[1], 10), f'{target}: stale heatmap matrix dimensions')
    require(all(re.fullmatch(r'(?:Cluster_)?[1-9]\d*', str(x)) for x in matrix.columns),
            f'{target}: invalid cluster labels in 8.3')
    order = [int(str(x).removeprefix('Cluster_')) for x in matrix.columns]
    require(set(order) == set(range(1, 11)), f'{target}: missing/duplicate heatmap clusters')
    require(matrix.index.astype(str).tolist() == core.columns.astype(str).tolist(),
            f'{target}: heatmap developmental stages differ from Mfuzz')
    require(np.isfinite(core.to_numpy()).all(), f'{target}: nonfinite core profiles')
    require(np.allclose(matrix.to_numpy(), core.loc[[f'Cluster_{x}' for x in order]].T.to_numpy(),
                        rtol=1e-10, atol=1e-12), f'{target}: 8.3 matrix not from current Mfuzz profiles')
    expected_order = (leaves_list(linkage(core.to_numpy(), method='average', metric='euclidean',
                                        optimal_ordering=False)) + 1).tolist()
    require(order == expected_order, f'{target}: order differs from current 8.3 clustering algorithm')
    return dict(target=target, species=species, organ=organ, ann=ann, background=set(bg.geneId),
                order=order, background_promoters=len(bg))


def kegg_cache(species, a):
    cached = a.legacy_annotations / SPECIES[species]['old_cache']
    return cached if (cached / 'download_manifest.json').exists() else a.annotation_cache / ('kegg_' + species.lower())


def kegg_urls(species):
    code = SPECIES[species]['kegg']
    return {'pathway_names.tsv': f'https://rest.kegg.jp/list/pathway/{code}',
            'pathway_gene_links.tsv': f'https://rest.kegg.jp/link/{code}/pathway',
            'release_info.txt': 'https://rest.kegg.jp/info/kegg'}


def load_kegg(species, a, sources, download=False):
    cache = kegg_cache(species, a)
    urls = kegg_urls(species)
    mp = cache / 'download_manifest.json'
    if not mp.exists():
        if not download:
            return None
        require(not a.offline, f'{species}: KEGG cache absent in --offline mode: {cache}')
        require(not cache.exists() or not any(cache.iterdir()), f'Incomplete KEGG cache: {cache}')
        # Download all responses before writing a snapshot; gene lists are NEVER uploaded.
        payloads = {}
        for name, url in urls.items():
            print('Public annotation download:', url, flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'Python reproducible enrichment'})
            with urllib.request.urlopen(request, timeout=60) as response:
                payloads[name] = response.read()
            require(payloads[name] and b'<html' not in payloads[name][:200].lower(), f'Invalid KEGG response: {url}')
        parse_kegg(species, payloads['pathway_names.tsv'].decode(), payloads['pathway_gene_links.tsv'].decode())
        cache.mkdir(parents=True, exist_ok=True)
        meta = dict(downloaded_utc=now(), files={})
        for name, data in payloads.items():
            with (cache / name).open('xb') as stream:
                stream.write(data)
            meta['files'][name] = dict(url=urls[name], sha256=sha256(cache / name), bytes=len(data))
        dump_json(mp, meta)
    meta = json.loads(mp.read_text())
    for name, url in urls.items():
        record = meta['files'][name]
        require(record['url'] == url and sha256(cache / name) == record['sha256'],
                f'{species}: KEGG snapshot URL/checksum mismatch: {name}')
        sources[str((cache / name).resolve())] = record['sha256']
    sources[str(mp.resolve())] = sha256(mp)
    names, pairs = parse_kegg(species, (cache / 'pathway_names.tsv').read_text(),
                              (cache / 'pathway_gene_links.tsv').read_text())
    return names, pairs, meta


def parse_kegg(species, name_text, link_text):
    code, latin = SPECIES[species]['kegg'], SPECIES[species]['latin']
    names, pairs = {}, []
    for line in name_text.splitlines():
        key, text = line.split('\t', 1)
        key = key.removeprefix('path:')
        require(re.fullmatch(code + r'\d{5}', key), f'Wrong-species KEGG pathway: {key}')
        names[key] = text.split(' - ' + latin, 1)[0]
    for line in link_text.splitlines():
        pathway, gene = line.split('\t')
        require(pathway.startswith('path:' + code) and re.fullmatch(code + r':\d+', gene),
                'Wrong species or unexpected KEGG link orientation')
        pathway = pathway.removeprefix('path:')
        require(pathway in names, f'KEGG link missing pathway name: {pathway}')
        pairs.append((pathway, gene.removeprefix(code + ':')))
    require(len(names) > 100 and len(pairs) > 10000, f'{species}: suspiciously incomplete KEGG snapshot')
    return names, pairs


def annotation_paths(species, a):
    org = f"org.{SPECIES[species]['org']}.eg"
    return {'OrgDb': a.r_library / f'{org}.db/extdata/{org}.sqlite',
            'GO': a.r_library / 'GO.db/extdata/GO.sqlite',
            'Reactome': a.r_library / 'reactome.db/extdata/reactome.sqlite'}


def load_annotations(species, background, a, sources, download=False):
    paths = annotation_paths(species, a)
    metadata = {}
    needed = ['OrgDb'] + (['GO'] if any(f.startswith('GO_') for f in a.families) else [])
    needed += ['Reactome'] if 'Reactome' in a.families else []
    for name in needed:
        require(paths[name].is_file(), f'Missing {species} {name}: {paths[name]}')
        sources.setdefault(str(paths[name].resolve()), sha256(paths[name]))
        with read_db(paths[name]) as con:
            metadata[name] = dict(con.execute('SELECT * FROM metadata'))
    require(metadata['OrgDb'].get('TAXID') == SPECIES[species]['taxid'], f'Wrong {species} OrgDb TAXID')
    terms = {}
    info = defaultdict(lambda: dict(entrez_ids=set(), symbols=set()))
    def add(family, term, description, gene):
        item = terms.setdefault((family, term), dict(Description=description, genes=set()))
        require(item['Description'] == description, f'Conflicting term descriptions: {term}')
        item['genes'].add(gene)
    go_names = {}
    if 'GO' in metadata:
        require(metadata['GO']['GOSOURCEDATE'] == metadata['OrgDb']['GOSOURCEDATE'], 'GO/OrgDb dates differ')
        with read_db(paths['GO']) as con:
            go_names = {g: (t, o) for g, t, o in con.execute('SELECT go_id, term, ontology FROM go_term')}
    ids = sorted(background)
    with read_db(paths['OrgDb']) as con:
        for start in range(0, len(ids), 700):
            batch = ids[start:start + 700]
            placeholders = ','.join('?' for _ in batch)
            for gene, entrez, symbol in con.execute(
                'SELECT DISTINCT e.ensembl_id,g.gene_id,i.symbol FROM ensembl e '
                'JOIN genes g ON e._id=g._id JOIN gene_info i ON e._id=i._id '
                f'WHERE e.ensembl_id IN ({placeholders})', batch):
                info[gene]['entrez_ids'].add(str(entrez))
                if symbol:
                    info[gene]['symbols'].add(symbol)
            for family in a.families:
                if not family.startswith('GO_'):
                    continue
                ontology = family.removeprefix('GO_')
                for gene, term in con.execute('SELECT DISTINCT e.ensembl_id,a.go_id FROM ensembl e '
                        f'JOIN go_{ontology.lower()}_all a ON e._id=a._id '
                        f'WHERE e.ensembl_id IN ({placeholders})', batch):
                    if term in go_names and term not in GO_ROOTS:
                        description, actual = go_names[term]
                        require(actual == ontology, 'GO ontology mismatch')
                        add(family, term, description, gene)
    reverse = defaultdict(set)
    for gene, values in info.items():
        for entrez in values['entrez_ids']:
            reverse[entrez].add(gene)
    if 'Reactome' in a.families:
        with read_db(paths['Reactome']) as con:
            for term, entrez, name in con.execute('SELECT DISTINCT p.DB_ID,p.gene_id,n.path_name '
                    'FROM pathway2gene p JOIN pathway2name n ON p.DB_ID=n.DB_ID '
                    'WHERE p.DB_ID LIKE ?', (SPECIES[species]['reactome'] + '%',)):
                for gene in reverse.get(str(entrez), ()):
                    add('Reactome', term, name.removeprefix(SPECIES[species]['latin'] + ': '), gene)
    pending = []
    if 'KEGG' in a.families:
        kegg = load_kegg(species, a, sources, download=download)
        if kegg is None:
            pending.append('KEGG')
        else:
            names, pairs, metadata['KEGG'] = kegg
            for term, entrez in pairs:
                for gene in reverse.get(entrez, ()):
                    add('KEGG', term, names[term], gene)
    for family in a.families:
        require(family in pending or any(k[0] == family for k in terms),
                f'{species}: no background genes mapped to {family}; not silently omitted')
    return terms, info, metadata, pending


def build_family_terms(terms, background, families, min_size, max_size):
    universes, eligible = {}, {}
    for family in families:
        annotated = [(term, item['Description'], item['genes'] & background)
                     for (fam, term), item in terms.items() if fam == family]
        universes[family] = set().union(*(g for _, _, g in annotated))
        eligible[family] = sorted((t, d, g) for t, d, g in annotated if min_size <= len(g) <= max_size)
    return universes, eligible


def enrich(query, term_sets, universe, min_foreground):
    query = query & universe
    if len(query) < min_foreground or not term_sets:
        return None, query
    M, N = len(universe), len(query)
    rows = []
    for term, description, genes in term_sets:
        hits = sorted(query & genes)
        k, n = len(hits), len(genes)
        rows.append(dict(ID=term, Description=description, overlap_n=k, foreground_n=N,
                         term_background_n=n, background_n=M, GeneRatio=f'{k}/{N}',
                         BgRatio=f'{n}/{M}', fold_enrichment=(k / N) / (n / M), hit_genes=';'.join(hits)))
    df = pd.DataFrame(rows)
    df['pvalue'] = hypergeom.sf(df.overlap_n.to_numpy() - 1, M, df.term_background_n.to_numpy(), N)
    df['p_adjust_BH'] = bh_adjust(df.pvalue)
    return df, query


def choose_terms(results, order, a):
    sig = results[(results.p_adjust_BH < a.fdr) & (results.overlap_n >= a.min_hits)].copy()
    ranked = {c: sig[sig.cluster.eq(c)].sort_values(['p_adjust_BH', 'pvalue', 'family', 'ID'],
                 kind='stable').head(a.top_per_cluster) for c in order}
    chosen = []
    # Round robin prevents a single early cluster filling the whole display.
    for rank in range(a.top_per_cluster):
        for cluster in order:
            if len(ranked[cluster]) > rank:
                row = ranked[cluster].iloc[rank]
                key = (row.family, row.ID)
                if key not in chosen and len(chosen) < a.max_terms:
                    chosen.append(key)
    rows = []
    for family, term in chosen:
        sub = sig[sig.family.eq(family) & sig.ID.eq(term)]
        best = sub.sort_values(['p_adjust_BH', 'pvalue', 'cluster'], kind='stable').iloc[0]
        rows.append(dict(family=family, ID=term, Description=best.Description,
                         label=best.Description, minimum_p_adjust_BH=float(best.p_adjust_BH),
                         strongest_cluster=int(best.cluster),
                         significant_clusters=';'.join(str(c) for c in order if c in set(sub.cluster))))
    selected = pd.DataFrame(rows, columns=TERM_COLUMNS)
    if len(selected):
        # Put early-column modules at the bottom, like the supplied reference.
        selected['_position'] = selected.strongest_cluster.map({c: i for i, c in enumerate(order)})
        selected = selected.sort_values(['_position', 'minimum_p_adjust_BH', 'family', 'ID'],
                                        ascending=[False, True, True, True]).drop(columns='_position').reset_index(drop=True)
        duplicated = selected.Description.duplicated(keep=False)
        selected.loc[duplicated, 'label'] = selected.loc[duplicated].apply(
            lambda x: f'[{x.family}] {x.Description} ({x.ID})', axis=1)
    return sig, selected


def heatmap_scores(results, selected, order, a):
    score = pd.DataFrame(np.nan, index=[f'{x.family}|{x.ID}' for x in selected.itertuples()], columns=order)
    score.index.name = 'term_key'
    lookup = results.set_index(['family', 'ID', 'cluster'])
    for item in selected.itertuples():
        for cluster in order:
            key = (item.family, item.ID, cluster)
            if key in lookup.index:
                row = lookup.loc[key]
                value = -np.log10(max(float(row.p_adjust_BH), np.finfo(float).tiny))
                score.loc[f'{item.family}|{item.ID}', cluster] = value if row.p_adjust_BH < a.fdr and row.overlap_n >= a.min_hits else 0.
    return score


def analyze_target(group, terms, info, a):
    ann, background, order = group['ann'], group['background'], group['order']
    universes, eligible = build_family_terms(terms, background, a.families, a.min_term_size, a.max_term_size)
    rows, counts = [], []
    for cluster in order:
        assigned = ann[ann.cluster.eq(cluster)]
        query = set(assigned.geneId)
        for family in a.families:
            df, mapped = enrich(query, eligible[family], universes[family], a.min_foreground)
            counts.append(dict(cluster=cluster, family=family, assigned_promoters=len(assigned),
                               unique_foreground_genes=len(query), annotated_foreground_genes=len(mapped),
                               unmapped_foreground_genes=len(query - universes[family]),
                               background_MP_genes=len(background), annotated_background_genes=len(universes[family]),
                               eligible_terms=len(eligible[family]), tested=df is not None,
                               significant_terms=0 if df is None else int(((df.p_adjust_BH < a.fdr) &
                                                                         (df.overlap_n >= a.min_hits)).sum())))
            if df is not None:
                df.insert(0, 'family', family)
                df.insert(0, 'cluster', cluster)
                df.insert(0, 'target', group['target'])
                rows.append(df)
    results = pd.concat(rows, ignore_index=True)[RESULT_COLUMNS] if rows else pd.DataFrame(columns=RESULT_COLUMNS)
    sig, selected = choose_terms(results, order, a)
    score = heatmap_scores(results, selected, order, a)
    mapping = []
    gene_clusters = ann.groupby('geneId').cluster.apply(lambda x: ';'.join(str(c) for c in order if c in set(x))).to_dict()
    for gene in sorted(background):
        item = info.get(gene, dict(entrez_ids=set(), symbols=set()))
        mapping.append(dict(geneId=gene, entrez_ids=';'.join(sorted(item['entrez_ids'])),
                            symbols=';'.join(sorted(item['symbols'])),
                            dynamic_clusters=gene_clusters.get(gene, ''),
                            **{f'annotated_{f}': gene in universes[f] for f in a.families}))
    promoter_map = ann[['promoterId', 'geneId', 'cluster', 'membership']].copy()
    # Retain the promoter-to-gene audit in the same gene table rather than writing 10 cluster files.
    by_gene = promoter_map.groupby('geneId').apply(
        lambda x: ';'.join(f'{p.promoterId}:C{p.cluster}' for p in x.itertuples()), include_groups=False).to_dict()
    mapping = pd.DataFrame(mapping)
    mapping['dynamic_promoter_cluster_pairs'] = mapping.geneId.map(by_gene).fillna('')
    return dict(results=results, significant=sig, selected=selected, score=score,
                summary=pd.DataFrame(counts), mapping=mapping)


def plot_heatmap(group, analysis, color_max, path, a):
    score, selected = analysis['score'], analysis['selected']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'pdf.compression': 0, 'text.usetex': False})
    # Nominal reference size 5.4 x 3.15 inches; widen only for a long unabridged GO label.
    probe = plt.figure(figsize=(a.figure_width, a.figure_height), dpi=100)
    probe.canvas.draw()
    renderer = probe.canvas.get_renderer()
    font = FontProperties(family='DejaVu Sans', size=a.font_size)
    label_width = max([renderer.get_text_width_height_descent(str(s), font, False)[0] / 100
                       for s in selected.label] or [0.])
    plt.close(probe)
    left = max(1.95, label_width + .12)
    height = max(a.figure_height, .42 + len(score) * .13)
    # Preserve the original heatmap height and row height. Change only cell width.
    heatmap_width = (height - .49) * 10 / len(score) if len(score) else 2.65
    width = max(a.figure_width, left + heatmap_width + .70)
    fig = plt.figure(figsize=(width, height), dpi=100)
    fig.suptitle(f"{group['organ']} in {group['species']}", fontsize=a.font_size + 1.5, y=.995)
    ax = fig.add_axes([left / width, .16 / height, heatmap_width / width, (height - .49) / height])
    if len(score):
        cmap = plt.get_cmap('YlGnBu').with_extremes(bad='#d9d9d9')
        mesh = ax.pcolormesh(np.arange(11), np.arange(len(score) + 1),
                             np.ma.masked_invalid(score.to_numpy()), cmap=cmap, vmin=0,
                             vmax=color_max, edgecolors='white', linewidth=.25,
                             shading='flat', rasterized=False, antialiased=False)
        ax.set_xlim(0, 10)
        ax.set_ylim(len(score), 0)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(np.arange(10) + .5, labels=group['order'])
        ax.set_yticks(np.arange(len(score)) + .5, labels=selected.label)
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='both', length=0, labelsize=a.font_size, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cax = fig.add_axes([(left + heatmap_width + .18) / width, .50 / height, .105 / width, (height - .95) / height])
        cbar = fig.colorbar(mesh, cax=cax)
        # Matplotlib auto-rasterizes colorbar solids unless explicitly disabled.
        if cbar.solids is not None:
            cbar.solids.set_rasterized(False)
            cbar.solids.set_edgecolor('face')
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=a.font_size - .5, width=.35, length=2, pad=2)
        cbar.set_label('-log10(BH FDR)', fontsize=a.font_size, labelpad=4)
        if score.isna().any().any():
            fig.text((left + heatmap_width) / width, .015, 'Gray: not tested', ha='right', fontsize=a.font_size - 1)
    else:
        ax.set_axis_off()
        ax.text(.5, .5, 'No terms meet the enrichment criteria', ha='center', va='center', fontsize=8)
    fig.canvas.draw()
    # Catch clipped scientific labels before export (no tight bounding-box size changes).
    rend = fig.canvas.get_renderer()
    for label in ax.get_yticklabels():
        require(label.get_window_extent(rend).x0 >= 1, f'Clipped heatmap label: {label.get_text()}')
    fig.savefig(path, format='pdf', metadata=dict(Title=f"{group['target']} cluster gene enrichment",
                Subject='All assigned dynamic MP promoter genes; same-organ tested MP background',
                Keywords=json.dumps(dict(cluster_order=group['order'], score='-log10(BH FDR)',
                                         families=a.families, significance=f'FDR < {a.fdr}, hits >= {a.min_hits}',
                                         colors_shared_across_targets=True))))
    plt.close(fig)
    return dict(inches=[width, height], tick_font_pt=a.font_size, title_font_pt=a.font_size + 1.5,
                vector_cells=True, pdf_fonttype=42, pdf_compression=0,
                square_cells=True, heatmap_inches=[heatmap_width, height - .49])


def self_test():
    from types import SimpleNamespace
    np.testing.assert_allclose(bh_adjust([.01, .04, .03, 1.]), [.04, .0533333333333, .0533333333333, 1.])
    # A zero-hit eligible term must remain in the BH denominator.
    df, query = enrich({'a', 'b', 'c'}, [('T1', 'one', {'a', 'b', 'c'}),
                      ('T2', 'two', {'d', 'e'})], set('abcdefghij'), 3)
    assert len(df) == 2 and df.overlap_n.tolist() == [3, 0] and df.pvalue.iloc[1] == 1
    np.testing.assert_allclose(df.p_adjust_BH.iloc[0], 2 * hypergeom.sf(2, 10, 3, 3))
    t = {('GO_BP', 'small'): dict(Description='small', genes={'a', 'b'}),
         ('GO_BP', 'large'): dict(Description='large', genes=set('abcdef'))}
    u, eligible = build_family_terms(t, set('abcdef'), ['GO_BP'], 2, 3)
    assert u['GO_BP'] == set('abcdef') and len(eligible['GO_BP']) == 1
    assert enrich({'a'}, eligible['GO_BP'], u['GO_BP'], 3)[0] is None
    assert normalize_genes(pd.Series(['ENSRNOG0001.2']), 'Rat').iloc[0] == 'ENSRNOG0001'
    try:
        normalize_genes(pd.Series(['ENSG0001']), 'Mouse')
        raise AssertionError('Cross-species gene passed')
    except ValueError:
        pass
    a = SimpleNamespace(fdr=.05, min_hits=3, top_per_cluster=3, max_terms=20)
    df.insert(0, 'cluster', 7)
    df.insert(0, 'family', 'GO_BP')
    sig, chosen = choose_terms(df, [7, 2], a)
    score = heatmap_scores(df, chosen, [7, 2], a)
    assert len(sig) == len(chosen) == 1 and score.columns.tolist() == [7, 2]
    assert score.iloc[0, 0] > 0 and np.isnan(score.iloc[0, 1])
    # Nonsignificant TESTED cell is 0, not NA; empty selection stays valid.
    other = df.copy(); other['cluster'] = 2; other['p_adjust_BH'] = 1.
    score = heatmap_scores(pd.concat([df, other]), chosen, [7, 2], a)
    assert score.iloc[0, 1] == 0
    _, none = choose_terms(other, [7, 2], a)
    assert heatmap_scores(other, none, [7, 2], a).shape == (0, 2)
    print('SELF-TEST PASS: BH, zero-hit tests, universe-before-filter, small foreground, species IDs, display order, NA vs zero, no-significance case')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--targets', nargs='+', choices=TARGETS, default=TARGETS)
    p.add_argument('--families', nargs='+', choices=FAMILIES, default=FAMILIES)
    p.add_argument('--masigpro', type=Path, default=REDO / '8.1.run_magSigPro')
    p.add_argument('--mfuzz', type=Path, default=REDO / '8.2.run_mfuzz')
    p.add_argument('--heatmap', type=Path, default=REDO / '8.3.plot_heatmap')
    p.add_argument('--output', type=Path, default=ROOT / 'output')
    p.add_argument('--r-library', type=Path, default=LIBRARY, help='Read-only SQLite annotation location, NOT an R runtime')
    p.add_argument('--legacy-annotations', type=Path, default=OLD / 'annotations')
    p.add_argument('--annotation-cache', type=Path, default=ROOT / 'annotations')
    p.add_argument('--offline', action='store_true', help='Refuse public KEGG downloads if a cache is missing')
    p.add_argument('--min-term-size', type=int, default=10)
    p.add_argument('--max-term-size', type=int, default=500)
    p.add_argument('--min-foreground', type=int, default=3)
    p.add_argument('--min-hits', type=int, default=3)
    p.add_argument('--fdr', type=float, default=.05)
    p.add_argument('--top-per-cluster', type=int, default=3)
    p.add_argument('--max-terms', type=int, default=20)
    p.add_argument('--figure-width', type=float, default=5.4)
    p.add_argument('--figure-height', type=float, default=3.15)
    p.add_argument('--font-size', type=float, default=6.5)
    p.add_argument('--check-only', action='store_true', help='Read-only checks; no output or annotation downloads')
    p.add_argument('--self-test', action='store_true', help='Offline numerical tests; no input/output access')
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    require(len(a.targets) == len(set(a.targets)) and len(a.families) == len(set(a.families)), 'Duplicate targets/families')
    require(0 < a.fdr < 1 and 1 <= a.min_term_size <= a.max_term_size, 'Invalid FDR/term-size setting')
    require(min(a.min_foreground, a.min_hits, a.top_per_cluster, a.max_terms) >= 1, 'Invalid count setting')
    require(a.figure_width >= 4 and a.figure_height >= 2 and a.font_size >= 4, 'Unreadable figure settings')
    sources = {str(Path(__file__).resolve()): sha256(__file__)}
    manifests = []
    for step in [a.masigpro, a.mfuzz]:
        path = step / 'output/run_manifest.json'
        m = json.loads(path.read_text())
        require(m['status'] == 'complete', f'Upstream run incomplete: {path}')
        sources[str(path.resolve())] = sha256(path)
        manifests.append(m)
    m1, m2 = manifests
    require(m2['clusters'] == 10, 'This script expects the redo8 C10 run')
    groups = [prepare_target(t, a, m1, m2, sources) for t in a.targets]
    for g in groups:
        print(f"INPUT OK {g['target']}: {len(g['ann'])} promoters, {g['ann'].geneId.nunique()} foreground genes, "
              f"{len(g['background'])} MP background genes; order={g['order']}", flush=True)
    if not a.check_only:
        require(not a.output.exists(), f'Output exists; choose a NEW --output directory, never overwritten: {a.output}')
        for upstream in [a.masigpro, a.mfuzz, a.heatmap]:
            require(not a.output.resolve().is_relative_to(upstream.resolve()), 'Output cannot be inside upstream data')
    annotations, annotation_meta, pending = {}, {}, {}
    for species in dict.fromkeys(g['species'] for g in groups):
        union = set().union(*(g['background'] for g in groups if g['species'] == species))
        terms, info, metadata, missing = load_annotations(species, union, a, sources, download=not a.check_only)
        annotations[species] = terms, info
        annotation_meta[species] = metadata
        if missing:
            pending[species] = missing
        print(f'ANNOTATION OK {species}: {len(info)}/{len(union)} Entrez-mapped background genes, '
              f'{len(terms)} terms; missing cached families={missing}', flush=True)
    if a.check_only:
        print('CHECK-ONLY PASS; no files written, enrichment not run.')
        if pending:
            print('First normal run will download public KEGG annotations (unless --offline):', pending)
        return 0
    analyses = {g['target']: analyze_target(g, *annotations[g['species']], a) for g in groups}
    finite_max = max([float(x['score'].max().max()) for x in analyses.values()
                      if x['score'].size and np.isfinite(x['score'].to_numpy()).any()] or [0.])
    color_max = max(5., math.ceil(finite_max / 5.) * 5.)
    # Ensure nothing changed during preflight and numerical calculation.
    for path, digest in sources.items():
        require(sha256(path) == digest, f'Source changed during analysis: {path}')
    a.output.mkdir(parents=True)
    manifest = dict(status='running', started_utc=now(), targets=a.targets,
                    parameters={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                    foreground='unique genes of all hard-assigned dynamic MP promoters; no membership cutoff',
                    background='same-target all-tested MP genes; each family annotated universe before term-size filtering',
                    BH_scope='target x cluster x annotation family; all eligible terms including zero overlaps',
                    significance=f'BH FDR < {a.fdr} AND overlap >= {a.min_hits} unique Ensembl genes',
                    multiple_mapping='all valid Entrez mappings retained; Ensembl gene is statistical unit',
                    GO='ancestor-propagated go_bp/cc/mf_all; three root terms excluded',
                    display='round-robin top terms per cluster, unique family/ID keys; early-column peaks at bottom; not exhaustive',
                    gene_cluster_dependence='same gene may occur in several clusters; clusters are not independent gene sets',
                    heatmap='8.3 column order unchanged; 0=tested non-significant/too-few-hit, NA=not tested',
                    shared_color_range=[0, color_max], annotation_metadata=annotation_meta,
                    source_sha256=sources, runtime=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                                                       scipy=scipy.__version__, matplotlib=matplotlib.__version__),
                    groups=[], output_sha256={})
    mp = a.output / 'run_manifest.json'
    dump_json(mp, manifest)
    try:
        for group in groups:
            target = group['target']
            analysis = analyses[target]
            out = a.output / target
            out.mkdir()
            for key, filename in [('results', 'all_enrichment_results.tsv.gz'),
                                  ('significant', 'significant_enrichment.tsv'),
                                  ('selected', 'heatmap_terms.tsv'), ('summary', 'cluster_summary.tsv'),
                                  ('mapping', 'background_gene_mapping.tsv')]:
                analysis[key].to_csv(out / filename, sep='\t', index=False, float_format='%.17g')
            analysis['score'].to_csv(out / 'heatmap_scores.tsv', sep='\t', float_format='%.17g', na_rep='NA')
            pdf = out / f'{target}_C10_gene_enrichment_heatmap.pdf'
            style = plot_heatmap(group, analysis, color_max, pdf, a)
            manifest['groups'].append(dict(target=target, assigned_promoters=len(group['ann']),
                foreground_unique_genes=group['ann'].geneId.nunique(), background_MP_genes=len(group['background']),
                background_MP_promoters=group['background_promoters'], cluster_order=group['order'],
                tests=len(analysis['results']), significant_tests=len(analysis['significant']),
                displayed_terms=len(analysis['selected']), figure=style))
            print(f"DONE {target}: {len(analysis['significant'])} significant tests; {len(analysis['selected'])} displayed terms", flush=True)
        for path, digest in sources.items():
            require(sha256(path) == digest, f'Source changed during export: {path}')
        manifest['output_sha256'] = {str(p.relative_to(a.output)): sha256(p)
                                     for p in sorted(a.output.rglob('*')) if p.is_file() and p != mp}
        manifest.update(status='complete', finished_utc=now(), upstream_unchanged=True)
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}', finished_utc=now())
        raise
    finally:
        dump_json(mp, manifest)
    print('COMPLETE:', a.output)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as error:
        print('ERROR:', error, file=sys.stderr)
        sys.exit(2)
