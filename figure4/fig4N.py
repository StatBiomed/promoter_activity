#!/usr/bin/env python3
"""FIMO motif-positive region percentages in Background, Group A and Group B.

One Python entry point: validate BED/mm10 coordinates, extract unchanged regions,
scan two fixed JASPAR models, count unique hit-positive regions, and draw PDFs.
Original BED, FASTA and parent analyses are read-only. Existing output is refused.

Percentage = regions with >=1 reported motif hit / ALL BED regions * 100.
Multiple sites/strands on one region do NOT increase its binary hit-positive count.
The input BEDs are merged regions, NOT individual promoters or gene replicates.

Models: unchanged HNF4A MA0114.5 and official E2F4 MA0470.1, both human-derived.
They predict motif occurrences in mouse GRCm38/mm10 sequence, not actual binding.
Use one model per TF, P < 1e-4 (NOT FDR), both strands, --max-strand, --no-pgc.
All three scans share an explicit uniform A/C/G/T=0.25 zero-order scoring model.
This nucleotide scoring background is different from the Background REGION SET.
FIMO --text avoids stored-score truncation and does not calculate q-values.

The Background BED is the merged A/B union; groups overlap and background regions
can be longer. Percentages are descriptive; no naive Fisher P values or CIs are
computed. Region lengths, GC and N content are retained for interpretation.

Figure contract: compare model-specific motif-positive fractions, not actual TF
binding or affinity; quantitative grid, HNF4A left and E2F4 right, shared axis,
boxed TF headers and point-centered percentages. One editable PDF, 70 x 52.6688
mm, matching the supplied panel aspect ratio and approximate manuscript size.
No borrowed P values: overlapping A/B/Background sets are not independent.
The two pinned motif matrices and regression hashes are embedded in this single
script. No dependency on the deleted legacy output folders remains. Both TFs'
summary, per-region results, extracted FASTA and FIMO outputs must reproduce the
previous analysis before the figure can be written. The six circular markers use
display coordinates (18 pt diameter), independent of the x/y data-axis scales.

Run with the project Python (matplotlib):
  python -B run_fimo_HNF4A_E2F4_lollipop.py --self-test
  python -B run_fimo_HNF4A_E2F4_lollipop.py --check-only
  python -u -B run_fimo_HNF4A_E2F4_lollipop.py
Default new output directory: results/ beside this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest

BASE = Path('/mnt/ruiyanhou/nfs_share2')
PROJECT = BASE / 'proactiv_redo'
ROOT = Path(__file__).resolve().parent
DEFAULT_BEDS = PROJECT / '13.promoter_usage/5.chipseq_overlap/groupAB_bed'
DEFAULT_GENOME = BASE / 'RNA_seq_organ_species/ref_annotation/Mus_musculus.GRCm38.dna.primary_assembly.fa'
DEFAULT_FIMO = BASE / 'software/meme5.5.8_install/bin/fimo'
DEFAULT_BEDTOOLS = BASE / 'software/bedtools2/bin/bedtools'
GROUPS = [('Background', 'background.merged.bed', 'BG', 311),
          ('Group A', 'A.merged.bed', 'A', 183), ('Group B', 'B.merged.bed', 'B', 182)]
MODELS = {'HNF4A': {'id': 'MA0114.5', 'width': 9, 'species': 'Homo sapiens'},
          'E2F4': {'id': 'MA0470.1', 'width': 11, 'species': 'Homo sapiens'}}
COLORS = {'Background': '#FFB25F', 'Group A': '#BAB4D0', 'Group B': '#B57AAA'}
FIGURE_MM = (70., 70. * 234. / 311.)
MARKER_DIAMETER_PT = 18.
MOTIF_TEXT = '''MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF MA0114.5 HNF4A
letter-probability matrix: alength= 4 w= 9 nsites= 46504 E= 0
 0.031567  0.891171  0.026600  0.050662
 0.931554  0.027030  0.022815  0.018601
 0.864807  0.029460  0.077735  0.027998
 0.930329  0.011698  0.040792  0.017181
 0.015397  0.019697  0.936006  0.028901
 0.019934  0.014945  0.046964  0.918158
 0.059565  0.812919  0.039265  0.088250
 0.029309  0.910481  0.013827  0.046383
 0.788190  0.059823  0.063134  0.088853
URL http://jaspar.genereg.net/matrix/MA0114.5

MOTIF MA0470.1 E2F4
letter-probability matrix: alength= 4 w= 11 nsites= 1878 E= 0
 0.144835  0.197018  0.451544  0.206603
 0.082535  0.271565  0.645900  0.000000
 0.000000  0.005857  0.994143  0.000000
 0.000000  1.000000  0.000000  0.000000
 0.000000  0.000000  1.000000  0.000000
 0.014377  0.116081  0.869542  0.000000
 0.007987  0.021832  0.970181  0.000000
 0.960064  0.000000  0.039936  0.000000
 0.667199  0.000000  0.332801  0.000000
 0.379127  0.185836  0.383387  0.051651
 0.267838  0.209265  0.328009  0.194888
URL https://jaspar.elixir.no/matrix/MA0470.1
'''
EXPECTED_MOTIF_SHA = 'ff3a77632453bfbcd9f4c1ab61eaa52287b6d7c81df11ea21a4206e1ff838359'
EXPECTED_BED_SHA = {
    'background.merged.bed': '09d4a9fd56d57b02fa97cf612497762c508a89077e4c26ed6bac0a26afe1ee36',
    'A.merged.bed': '44274c4cbd62e33f71db2841e893cf8849947c98cbc562d91fbf28f6277927eb',
    'B.merged.bed': 'ea9e5861c62191cc39c30fa1bda3fade8d2e8ba3773a04cf75ad50ba74bf0a7e'}
EXPECTED_FASTA_SHA = {
    'Background': '94a5d3e43bcc2a6714eae1a5e79384cee8d16dad651f4e5a3a0644823557168d',
    'Group A': '813d40359df26006d899d307ac9741309de9dfac2c622e756372bae248e784d8',
    'Group B': '0e445b6305578a36ebca10bb2e4fbe4db0ca4c34f7cb9654203391f0509f992e'}
EXPECTED_FIMO_SHA = {
    'Background': '3e1574c0dd01643c6da77e3d4bfb1dde272ebc304ee113c0469454dc77c04af4',
    'Group A': '912e13e6d30134633f2feca258ec90a79c201097f909165f2d0261ac85dd9e75',
    'Group B': '5380904e311ba68cb923003c9fe41b1cb29b7ab837ba51cf67652dbf760d485f'}
EXPECTED_SUMMARY_SHA = '6284a1c30721fb35e06749e00ee355fe175e24000cd7f6f6e215ff283685259f'
EXPECTED_DETAIL_SHA = '2ac8b6fc6d025f48bba8cb5a4f368134fe8dd0fef43f3e78db9d72362d591e35'
EXPECTED_COUNTS = {'HNF4A': [(100, 124), (63, 78), (50, 62)],
                   'E2F4': [(120, 178), (58, 93), (78, 106)]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns


def json_new(path, obj):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def table_new(path, rows, fields):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def canonical(chrom):
    name = chrom.removeprefix('chr')
    return 'chrM' if name in ('MT', 'M') else 'chr' + name


def merged(intervals):
    result = []
    for c, a, b in sorted(intervals):
        if result and c == result[-1][0] and a <= result[-1][2]:
            result[-1] = (c, result[-1][1], max(b, result[-1][2]))
        else:
            result.append((c, a, b))
    return result


def parse_bed(path, aliases, sizes, prefix):
    records, coordinates = [], set()
    with path.open(encoding='utf-8-sig') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(('#', 'track', 'browser')):
                continue
            fields = line.rstrip('\r\n').split('\t')
            require(len(fields) >= 3 and fields[1].isdigit() and fields[2].isdigit(),
                    f'{path}:{line_number}: invalid tab-delimited BED')
            c, a, b = canonical(fields[0]), int(fields[1]), int(fields[2])
            require(c in aliases and 0 <= a < b <= sizes[c], f'{path}:{line_number}: out-of-bounds BED')
            require(b - a >= max(m['width'] for m in MODELS.values()), f'{path}:{line_number}: region shorter than selected motifs')
            require((c, a, b) not in coordinates, f'{path}:{line_number}: repeated region')
            coordinates.add((c, a, b))
            records.append({'region_id': f'{prefix}_{len(records)+1:06d}',
                            'chrom': c, 'reference_chrom': aliases[c], 'start': a, 'end': b,
                            'length_bp': b-a, 'source_line': line_number})
    require(records, f'Empty BED: {path}')
    require(len(merged(coordinates)) == len(coordinates), f'{path}: expected already merged, disjoint regions')
    return records


def motif_blocks(source):
    require(source.startswith('MEME version') and 'ALPHABET= ACGT' in source, 'Expected DNA MEME file')
    starts = list(re.finditer(r'^MOTIF\s+(\S+)\s+([^\r\n]*)', source, re.M))
    blocks = {}
    for index, match in enumerate(starts):
        key = match[1]
        require(key not in blocks, f'Duplicate motif ID: {key}')
        blocks[key] = source[match.start():starts[index+1].start() if index+1 < len(starts) else len(source)].strip()
    return blocks


def selected_motifs():
    require(hashlib.sha256(MOTIF_TEXT.encode()).hexdigest() == EXPECTED_MOTIF_SHA,
            'Embedded motif resource differs from the original selected matrices')
    sources = motif_blocks(MOTIF_TEXT)
    chosen = []
    for tf, model in MODELS.items():
        block = sources.get(model['id'])
        require(block is not None, f'Missing motif {model["id"]}')
        require(block.splitlines()[0].split()[2].upper() == tf, f'Unexpected TF for {model["id"]}')
        lines = block.splitlines()
        matrix_line = next(i for i, line in enumerate(lines) if line.startswith('letter-probability matrix:'))
        match = re.search(r'\bw\s*=\s*(\d+)', lines[matrix_line])
        require(match and int(match[1]) == model['width'], 'Unexpected motif width')
        matrix = [list(map(float, line.split())) for line in lines[matrix_line+1:matrix_line+1+model['width']]]
        require(len(matrix) == model['width'] and all(len(r) == 4 and all(0 <= v <= 1 for v in r)
                and math.isclose(sum(r), 1., abs_tol=3e-5) for r in matrix), 'Invalid probability matrix')
        chosen.append(block)
    header = 'MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n'
    return header + '\n\n'.join(chosen) + '\n'


def resolve_tool(path, name):
    result = str(path) if path.is_file() else shutil.which(name)
    require(result is not None and Path(result).is_file(), f'Missing executable: {name}')
    return result


def prepare(args):
    fai = Path(str(args.genome) + '.fai')
    require(args.genome.is_file() and fai.is_file(), 'Reference FASTA and pre-existing FAI are required')
    aliases, sizes = {}, {}
    for line in fai.read_text().splitlines():
        fields = line.split('\t')
        key = canonical(fields[0])
        require(key not in aliases, f'Ambiguous FASTA chromosome alias: {key}')
        aliases[key], sizes[key] = fields[0], int(fields[1])
    require(sizes.get('chr1') == 195471971 and sizes.get('chr19') == 61431566, 'Not GRCm38/mm10 reference')
    records, paths = {}, [fai]
    for label, name, prefix, expected in GROUPS:
        path = args.bed_dir / name
        require(sha(path) == EXPECTED_BED_SHA[name], f'Original BED changed: {path}')
        records[label] = parse_bed(path, aliases, sizes, prefix)
        require(len(records[label]) == expected, f'{label}: expected {expected} original merged regions')
        paths.append(path)
    key = lambda r: (r['chrom'], r['start'], r['end'])
    union = merged([key(r) for label in ('Group A', 'Group B') for r in records[label]])
    require(union == sorted(key(r) for r in records['Background']), 'Background differs from the original merged A/B union')
    motif_text = selected_motifs()
    tools = {'bedtools': resolve_tool(args.bedtools, 'bedtools'), 'fimo': resolve_tool(args.fimo, 'fimo')}
    version = subprocess.run([tools['fimo'], '--version'], capture_output=True, text=True, check=True)
    tools['fimo_version'] = (version.stdout + version.stderr).strip()
    require(tools['fimo_version'] == '5.5.8', 'Exact reproduction requires FIMO 5.5.8')
    sources = [{'path': str(p.resolve()), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in paths]
    stamps = {p: stamp(p) for p in [args.genome, *paths]}
    return records, motif_text, tools, sources, stamps


def read_fasta(path):
    sequences, name, parts = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            if name is not None:
                require(name not in sequences, 'Repeated FASTA identifier')
                sequences[name] = ''.join(parts).upper()
            name, parts = line[1:], []
            require(name and not any(c.isspace() for c in name), 'Unexpected FASTA header/coordinate suffix')
        else:
            parts.append(line.strip())
    if name is not None:
        require(name not in sequences, 'Repeated FASTA identifier')
        sequences[name] = ''.join(parts).upper()
    return sequences


def fasta_from_extracted(extracted, fasta, records):
    """Support old/new bedtools -name header conventions without FIMO ID loss."""
    expected = {r['region_id']: r for r in records}
    sequences = {}
    for line in extracted.read_text().splitlines():
        parts = line.split('\t')
        require(len(parts) == 2, 'Expected two-column bedtools -tab output')
        header, sequence = parts
        name = header.split('::', 1)[0]
        require(name in expected and name not in sequences, 'Unexpected/repeated extracted identifier')
        row = expected[name]
        allowed = {name, f'{name}::{row["reference_chrom"]}:{row["start"]}-{row["end"]}'}
        require(header in allowed, 'Unexpected bedtools coordinate suffix')
        sequences[name] = sequence.upper()
    require(set(sequences) == set(expected), 'Extracted BED sequence identities differ')
    with fasta.open('x', encoding='utf-8') as handle:
        for row in records:
            name = row['region_id']
            handle.write(f'>{name}\n{sequences[name]}\n')
    require(read_fasta(fasta) == sequences, 'Normalized FASTA readback mismatch')
    return sequences


def parse_hits(path, records, threshold):
    lookup = {r['region_id']: r for r in records}
    by_id = {v['id']: k for k, v in MODELS.items()}
    result = {tf: {name: {'site_n': 0, 'best_p': None} for name in lookup} for tf in MODELS}
    with path.open(encoding='utf-8') as handle:
        reader = csv.DictReader((line for line in handle if line.strip() and not line.startswith('#')), delimiter='\t')
        require({'motif_id', 'sequence_name', 'start', 'stop', 'strand', 'p-value'} <= set(reader.fieldnames or []), 'Unexpected FIMO TSV header')
        for row in reader:
            require(row['motif_id'] in by_id and row['sequence_name'] in lookup, 'Unknown motif or sequence ID in FIMO output')
            a, b, p = int(row['start']), int(row['stop']), float(row['p-value'])
            require(1 <= a <= b <= lookup[row['sequence_name']]['length_bp'], 'FIMO coordinate out of sequence bounds')
            require(b-a+1 == MODELS[by_id[row['motif_id']]]['width'], 'FIMO hit width differs from motif width')
            require(row['strand'] in ('+', '-') and math.isfinite(p) and 0 <= p <= threshold * (1+1e-6), 'Invalid FIMO hit/threshold')
            entry = result[by_id[row['motif_id']]][row['sequence_name']]
            entry['site_n'] += 1
            entry['best_p'] = p if entry['best_p'] is None else min(entry['best_p'], p)
    return result


def summarize(label, records, hits, threshold):
    summary, detail = [], []
    for tf, model in MODELS.items():
        positive = sum(hits[tf][r['region_id']]['site_n'] > 0 for r in records)
        summary.append({'TF': tf, 'motif_id': model['id'], 'motif_source_species': model['species'],
                        'group': label, 'hit_positive_regions': positive, 'total_regions': len(records),
                        'percentage': 100. * positive / len(records),
                        'total_motif_sites': sum(v['site_n'] for v in hits[tf].values()),
                        'fimo_p_threshold': threshold, 'threshold_type': 'P-value, not FDR'})
        for record in records:
            hit = hits[tf][record['region_id']]
            detail.append({'TF': tf, 'motif_id': model['id'], 'group': label, **record,
                           'motif_hit': int(hit['site_n'] > 0), 'motif_site_n': hit['site_n'],
                           'best_p_value': '' if hit['best_p'] is None else hit['best_p']})
    return summary, detail


def verify_reproduction(summary, detail, per_group_files, out):
    """Pinned hashes validate actual computed outputs, not hardcoded plot heights."""
    require(len(summary) == 6 and len(detail) == 1352, 'Unexpected regression row count')
    for tf, expected in EXPECTED_COUNTS.items():
        for (label, _, _, n), (positive, sites) in zip(GROUPS, expected):
            row = next(r for r in summary if r['TF'] == tf and r['group'] == label)
            require((row['hit_positive_regions'], row['total_regions'], row['total_motif_sites'])
                    == (positive, n, sites), f'{tf}/{label}: motif counts changed')
            require(math.isclose(row['percentage'], positive/n*100, abs_tol=1e-12), 'Changed percentage')
    require(sha(out / 'FIMO_region_hit_percentages.tsv') == EXPECTED_SUMMARY_SHA,
            'Summary differs from the previous analysis')
    require(sha(out / 'FIMO_region_hit_details.tsv') == EXPECTED_DETAIL_SHA,
            'Per-region motif results differ from the previous analysis')
    for item in per_group_files:
        require(item['fasta_sha256'] == EXPECTED_FASTA_SHA[item['group']], 'Extracted sequences changed')
        require(item['fimo_tsv_sha256'] == EXPECTED_FIMO_SHA[item['group']], 'FIMO site output changed')
    return {'status': 'passed', 'both_TFs_summary_identical': True,
            'per_region_rows_identical': len(detail), 'all_FASTA_hashes_identical': True,
            'all_FIMO_site_output_hashes_identical': True,
            'baseline': 'Embedded checksums of the MA0114.5/MA0470.1 reference analysis'}


def draw(summary, figure_dir):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Affine2D
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
        font = 'Arial'
    except ValueError:
        font = 'DejaVu Sans'
    mpl.rcParams.update({'font.family': font, 'font.size': 6.4, 'pdf.fonttype': 42,
                         'pdf.compression': 0, 'text.usetex': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6})
    fig = plt.figure(figsize=tuple(v/25.4 for v in FIGURE_MM), dpi=240)
    axes, visible, circles = [], [], []
    for index, tf in enumerate(('HNF4A', 'E2F4')):
        ax = fig.add_axes([.195 + index*.395, .285, .377, .61])
        axes.append(ax)
        rows = [next(r for r in summary if r['TF'] == tf and r['group'] == label) for label, *_ in GROUPS]
        for x, row in enumerate(rows):
            y = row['percentage']
            ax.vlines(x, 0, y, color='#666666', linewidth=.8, zorder=1)
            # Scatter sizes use display points squared, not x/y data coordinates.
            # A data-coordinate Circle would still become an ellipse on these axes.
            mark = ax.scatter([x], [y], s=MARKER_DIAMETER_PT**2, marker='o',
                              color=COLORS[row['group']], edgecolors='none', linewidths=0, zorder=3)
            text = ax.text(x, y, f'{y:.2f}', ha='center', va='center', fontsize=5.8, color='black', zorder=4)
            circles.append((tf, row['group'], mark, text))
        ax.axhline(0, color='#666666', linestyle=(0, (5, 4)), linewidth=.7, zorder=0)
        ax.set_xlim(-.55, 2.55)
        ax.set_ylim(-3, 73)
        ax.set_yticks(list(range(0, 71, 10)))
        ax.set_xticks([0, 1, 2], [r['group'] for r in rows])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontweight='bold', color='#555555')
        ax.tick_params(axis='x', labelsize=6.2, length=0, pad=3)
        ax.tick_params(axis='y', labelsize=6.2, length=2, pad=1.5, colors='#555555', width=.5)
        for tick in ax.get_yticklabels():
            tick.set_fontweight('bold')
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        if index:
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', left=False, labelleft=False)
        else:
            ax.set_ylabel('Percentage', fontsize=7, fontweight='bold', labelpad=1.5)
            visible.append(ax.yaxis.label)
        ax.add_patch(Rectangle((0, 1), 1, .09, transform=ax.transAxes,
                               facecolor='white', edgecolor='#666666', linewidth=.6, clip_on=False))
        ax.text(.5, 1.045, tf, transform=ax.transAxes, ha='center', va='center',
                fontsize=7.2, fontweight='bold', color='#222222')
        visible.extend([*ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts])
    visible.append(fig.text(.028, .975, 'G', ha='left', va='top', fontsize=11))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    circle_qa = []
    for tf, group, mark, text in circles:
        display_path = mark.get_paths()[0].transformed(Affine2D(mark.get_transforms()[0]))
        box = display_path.get_extents()
        require(math.isclose(box.width, box.height, rel_tol=1e-12), 'Noncircular display marker')
        text_box = text.get_window_extent(renderer)
        require(math.hypot(text_box.width, text_box.height) <= box.width,
                f'{tf}/{group}: numeric label does not fit inside its circle')
        point = mark.get_offset_transform().transform(mark.get_offsets()[0])
        require(point[0] - box.width/2 >= mark.axes.bbox.x0 and
                point[0] + box.width/2 <= mark.axes.bbox.x1 and
                point[1] - box.height/2 >= mark.axes.bbox.y0 and
                point[1] + box.height/2 <= mark.axes.bbox.y1, 'Marker clipped by axes')
        circle_qa.append({'TF': tf, 'group': group, 'display_width_px': box.width,
                          'display_height_px': box.height, 'width_height_ratio': box.width/box.height,
                          'diameter_pt': MARKER_DIAMETER_PT, 'label_fits': True})
    for item in visible:
        if not item.get_visible():
            continue
        box = item.get_window_extent(renderer)
        require(box.x0 >= -.5 and box.y0 >= -.5 and box.x1 <= fig.bbox.width+.5 and box.y1 <= fig.bbox.height+.5,
                f'Clipped text {item.get_text()!r}')
    path = figure_dir / 'HNF4A_E2F4_FIMO_region_percentage_lollipop.pdf'
    with path.open('xb') as handle:
        fig.savefig(handle, format='pdf', bbox_inches=None)
    plt.close(fig)
    outputs = [{'path': str(path), 'sha256': sha(path), 'size_mm': list(FIGURE_MM)}]
    return {'backend': 'Python/matplotlib', 'matplotlib_version': mpl.__version__, 'font': font,
            'text_bounds_checked': True, 'panel_order': ['HNF4A', 'E2F4'],
            'circular_markers': circle_qa, 'figures': outputs}


def analyze(args):
    records, motif_text, tools, sources, stamps = prepare(args)
    profile = {label: {'regions': len(rows), 'min_length_bp': min(r['length_bp'] for r in rows),
                       'median_length_bp': statistics.median(r['length_bp'] for r in rows),
                       'max_length_bp': max(r['length_bp'] for r in rows)} for label, rows in records.items()}
    print(json.dumps({'input_profile': profile, 'tools': tools, 'models': MODELS}, indent=2), flush=True)
    if args.check_only:
        print('CHECK_ONLY_PASSED: no output written', flush=True)
        return
    require(not args.output_dir.exists(), f'Output already exists: {args.output_dir}; choose a new --output-dir')
    args.output_dir.mkdir(parents=True)
    out = args.output_dir
    for name in ('resources', 'fasta', 'fimo', 'figures', 'logs'):
        (out / name).mkdir()
    motif_path = out / 'resources/HNF4A_E2F4_selected.meme'
    motif_path.write_text(motif_text, encoding='utf-8')
    bg_path = out / 'resources/shared_uniform_background.txt'
    bg_path.write_text('# Shared zero-order scoring background for all groups\nA 0.25\nC 0.25\nG 0.25\nT 0.25\n', encoding='utf-8')
    summary, detail, commands, per_group_files = [], [], [], []
    for label, _, prefix, _ in GROUPS:
        rows = records[label]
        named = out / 'fasta' / f'{prefix}.reference_names.bed'
        fasta = out / 'fasta' / f'{prefix}.fa'
        extracted = out / 'fasta' / f'{prefix}.extracted.tsv'
        with named.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(f'{row["reference_chrom"]}\t{row["start"]}\t{row["end"]}\t{row["region_id"]}\n')
        cmd = [tools['bedtools'], 'getfasta', '-name', '-tab', '-fi', str(args.genome), '-bed', str(named), '-fo', str(extracted)]
        commands.append(cmd)
        with (out / 'logs' / f'{prefix}.getfasta.log').open('x') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=log)
        sequences = fasta_from_extracted(extracted, fasta, rows)
        require(set(sequences) == {r['region_id'] for r in rows}, f'{label}: BED/FASTA identity mismatch')
        for row in rows:
            seq = sequences[row['region_id']]
            require(len(seq) == row['length_bp'] and set(seq) <= set('ACGTN'), f'{label}: wrong/invalid extracted sequence')
            counts = Counter(seq)
            row['N_bp'] = counts['N']
            row['ACGT_bp'] = len(seq) - counts['N']
            row['GC_fraction_ACGT'] = (counts['G']+counts['C']) / row['ACGT_bp'] if row['ACGT_bp'] else ''
        profile[label]['total_N_bp'] = sum(r['N_bp'] for r in rows)
        profile[label]['regions_with_N'] = sum(r['N_bp'] > 0 for r in rows)
        result = out / 'fimo' / f'{prefix}.fimo.tsv'
        cmd = [tools['fimo'], '--text', '--max-strand', '--no-pgc', '--thresh', str(args.threshold),
               '--bfile', str(bg_path), str(motif_path), str(fasta)]
        commands.append(cmd)
        with result.open('x') as output, (out / 'logs' / f'{prefix}.fimo.log').open('x') as log:
            subprocess.run(cmd, check=True, stdout=output, stderr=log)
        hits = parse_hits(result, rows, args.threshold)
        s, d = summarize(label, rows, hits, args.threshold)
        summary.extend(s)
        detail.extend(d)
        per_group_files.append({'group': label, 'fasta_sha256': sha(fasta), 'fimo_tsv_sha256': sha(result)})
        print(f'{label}: {len(rows)} extracted and scanned regions', flush=True)
    require(all(stamp(p) == before for p, before in stamps.items()), 'An original input/reference changed during the run')
    table_new(out / 'FIMO_region_hit_percentages.tsv', summary, list(summary[0]))
    table_new(out / 'FIMO_region_hit_details.tsv', detail, list(detail[0]))
    table_new(out / 'region_set_profiles.tsv', [{'group': label, **v} for label, v in profile.items()], ['group', *next(iter(profile.values()))])
    regression = verify_reproduction(summary, detail, per_group_files, out)
    json_new(out / 'calculation_regression_check.json', regression)
    qa = draw(summary, out / 'figures')
    manifest = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'script_sha256': sha(Path(__file__)), 'sources': sources,
                'reference_fasta': {'path': str(args.genome), 'bytes': args.genome.stat().st_size,
                                    'assembly': 'GRCm38/mm10', 'full_fasta_hash_computed': False},
                'models': MODELS,
                'E2F4_official_source': 'https://jaspar.elixir.no/api/v1/matrix/MA0470.1.meme',
                'embedded_selected_motif_sha256': EXPECTED_MOTIF_SHA,
                'calculation_regression': regression,
                'threshold': {'type': 'site P-value', 'value': args.threshold, 'q_values_computed': False},
                'scoring_background': 'shared uniform zero-order A=C=G=T=0.25',
                'commands': commands, 'executables': tools, 'input_profile': profile,
                'counting_rule': 'Unique BED regions with at least one hit / all regions, including zero-hit regions',
                'background_set': 'Original merged A/B union, not the 167 nonsignificant promoters',
                'inferential_tests': None,
                'limitations': ['Motif occurrence is not experimental TF binding.',
                               'Human-derived vertebrate motifs predict sites in mouse DNA.',
                               'Merged regions differ in length, which affects at-least-one-hit probability.',
                               'Background and A/B are overlapping sets, not independent biological replicates.'],
                'per_group_outputs': per_group_files, 'plot_qa': qa, 'summary': summary}
    json_new(out / 'analysis_manifest.json', manifest)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'COMPLETE: {out}', flush=True)


def self_test():
    class Checks(unittest.TestCase):
        def test_merge(self):
            self.assertEqual(merged([('chr1', 10, 20), ('chr1', 20, 25), ('chr1', 40, 50)]), [('chr1', 10, 25), ('chr1', 40, 50)])

        def test_aliases(self):
            self.assertEqual([canonical(x) for x in ('1', 'chr1', 'MT', 'chrM')], ['chr1', 'chr1', 'chrM', 'chrM'])

        def test_embedded_motifs(self):
            self.assertEqual(selected_motifs(), MOTIF_TEXT)
            self.assertEqual(set(motif_blocks(MOTIF_TEXT)), {'MA0114.5', 'MA0470.1'})

        def test_counts_and_zero_hits(self):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'fimo.tsv'
                header = 'motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tmatched_sequence\n'
                rows = [{'region_id': 'r1', 'length_bp': 40}, {'region_id': 'r2', 'length_bp': 40}]
                path.write_text(header + 'MA0114.5\tHNF4A\tr1\t1\t9\t+\t10\t1e-5\tAAAAAAAAA\n' +
                                'MA0114.5\tHNF4A\tr1\t10\t18\t-\t11\t1e-6\tAAAAAAAAA\n')
                hits = parse_hits(path, rows, 1e-4)
                summary, _ = summarize('test', rows, hits, 1e-4)
                self.assertEqual(summary[0]['percentage'], 50.)
                self.assertEqual(summary[0]['total_motif_sites'], 2)
                self.assertEqual(summary[1]['percentage'], 0.)
                path.write_text(header)
                zero, _ = summarize('test', rows, parse_hits(path, rows, 1e-4), 1e-4)
                self.assertEqual([r['percentage'] for r in zero], [0., 0.])

        def test_fasta(self):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'x.fa'
                path.write_text('>A_000001\nACGTN\n>A_000002\nTT\nAA\n')
                self.assertEqual(read_fasta(path), {'A_000001': 'ACGTN', 'A_000002': 'TTAA'})

        def test_bedtools_header_versions(self):
            for header in ('A_000001', 'A_000001::1:10-15'):
                with tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    (root / 'extract.tsv').write_text(header + '\tACGTN\n')
                    rows = [{'region_id': 'A_000001', 'reference_chrom': '1', 'start': 10, 'end': 15}]
                    result = fasta_from_extracted(root/'extract.tsv', root/'out.fa', rows)
                    self.assertEqual(result, {'A_000001': 'ACGTN'})

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    require(result.wasSuccessful(), 'Self-test failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bed-dir', type=Path, default=DEFAULT_BEDS)
    parser.add_argument('--genome', type=Path, default=DEFAULT_GENOME)
    parser.add_argument('--fimo', type=Path, default=DEFAULT_FIMO)
    parser.add_argument('--bedtools', type=Path, default=DEFAULT_BEDTOOLS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--threshold', type=float, default=1e-4, choices=[1e-4],
                        help='Pinned original site P-value threshold; exact redraw only')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    require(0 < args.threshold < 1, 'Invalid P-value threshold')
    if args.self_test:
        self_test()
    else:
        analyze(args)


if __name__ == '__main__':
    main()
