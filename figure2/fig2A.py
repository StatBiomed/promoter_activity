#!/usr/bin/env python3
"""Draw source-validated Human organ-specific Major/Minor flowcharts (PDF only).

Figure contract: show actual gene/promoter counts through internal-promoter and
organ-local all-zero filtering, then split retained genes into single- and
multi-promoter groups. The multi-promoter Major count EXCLUDES singleton Majors.
Archetype: schematic-led workflow. Python/matplotlib, white boxes, black arrows,
orange singleton cartoon, gold Major and purple Minor cartoons. Gene cartoons
are schematic and NOT actual gene structures or area-scaled counts.
Individual pages: 183 x 198 mm, editable TrueType PDF text. No statistical test.

Uses results_by_organ/run_manifest.json and the existing classification tables.
Recomputes each organ from ALL original rowData files with the SAME reference
classifier and requires byte-identical classification before plotting. Checks
input stamps, reference checksum, per-organ condition lists and summary counts.
Never changes classification or input files. Requires Python >=3.9, matplotlib,
and ../define_major_minor_promoters.py. No pandas, R, Graphviz or GPU needed.

Outputs in ./flowcharts_by_organ (must not already exist):
  Human_ORGAN_major_minor_flowchart.pdf (one for each selected organ)
  Human_all_organs_major_minor_flowcharts.pdf (one organ per page)
  flowchart_counts.tsv and flowchart_audit.json (traceability, not image formats)

Usage:
  python3 -u plot_human_organ_major_minor_flowcharts.py
  python3 -u plot_human_organ_major_minor_flowcharts.py --organs Liver --output-dir liver_flowchart
  python3 plot_human_organ_major_minor_flowcharts.py --self-test
  python3 plot_human_organ_major_minor_flowcharts.py --counts-input flowcharts_by_organ/flowchart_counts.tsv --output-dir redraw

All developmental stages in an organ are pooled, with equal weights per
condition INCLUDING zeros. Prenatal/postnatal classifications are not separate.
--counts-input is explicitly render-only, not a new validation of source files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import filecmp
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrowPatch

SUFFIX = '_rowData.tsv'
ORDER = ('Brain', 'Cerebellum', 'Heart', 'Kidney', 'Liver', 'Ovary', 'Testis')
FIELDS = ('organ', 'conditions', 'input_genes', 'input_promoters',
          'noninternal_genes', 'noninternal_promoters', 'retained_genes',
          'retained_promoters', 'single_promoter_genes', 'multi_promoter_genes',
          'single_major_promoters', 'multi_major_promoters', 'multi_minor_promoters',
          'total_major_promoters', 'total_minor_promoters', 'removed_internal_genes',
          'removed_internal_promoters', 'removed_zero_genes', 'removed_zero_promoters',
          'positive_tied_genes')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def reference(path):
    spec = importlib.util.spec_from_file_location('_major_minor_reference', path)
    require(spec is not None and spec.loader is not None, f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_counts(c):
    require(set(c) == set(FIELDS), 'Unexpected count columns')
    require(c['organ'] in ORDER, f'Unsupported organ: {c["organ"]}')
    require(all(isinstance(c[k], int) and c[k] >= 0 for k in FIELDS[1:]), 'Negative/noninteger counts')
    require(c['conditions'] > 0, 'No conditions')
    require(c['input_genes'] >= c['noninternal_genes'] >= c['retained_genes'], 'Gene filter counts increase')
    require(c['input_promoters'] >= c['noninternal_promoters'] >= c['retained_promoters'], 'Promoter filter counts increase')
    for prefix in ('input', 'noninternal', 'retained'):
        require(c[prefix + '_genes'] <= c[prefix + '_promoters'], 'More genes than promoters')
    require(c['single_promoter_genes'] + c['multi_promoter_genes'] == c['retained_genes'], 'SP+MP gene mismatch')
    require(c['single_major_promoters'] == c['single_promoter_genes'], 'Singleton Major mismatch')
    require(c['multi_major_promoters'] == c['multi_promoter_genes'], 'MP Major mismatch')
    require(c['single_major_promoters'] + c['multi_major_promoters'] == c['total_major_promoters'] == c['retained_genes'], 'Total Major mismatch')
    require(c['total_minor_promoters'] == c['multi_minor_promoters'] >= c['multi_promoter_genes'], 'Minor mismatch')
    require(c['total_major_promoters'] + c['total_minor_promoters'] == c['retained_promoters'], 'Major+Minor mismatch')
    for noun in ('genes', 'promoters'):
        require(c['removed_internal_' + noun] == c['input_' + noun] - c['noninternal_' + noun], 'Internal removal mismatch')
        require(c['removed_zero_' + noun] == c['noninternal_' + noun] - c['retained_' + noun], 'Zero removal mismatch')
    require(c['positive_tied_genes'] <= c['multi_promoter_genes'], 'Tie count mismatch')


def collect(job):
    organ, manifest, result_root, input_dir, reference_path = job
    ref = reference(reference_path)
    require(sha256(reference_path) == manifest['reference_sha256'], 'Reference classifier changed since classification')
    files = sorted(p for p in input_dir.glob('*' + SUFFIX)
                   if p.name.startswith('Human_') and p.name[6:-len(SUFFIX)].rsplit('_', 1)[0] == organ)
    recorded = [r for r in manifest['inputs'] if r['organ'] == organ]
    require(files and {p.name for p in files} == {Path(r['path']).name for r in recorded},
            f'{organ}: original condition inventory changed')
    for r in recorded:
        require(stamp(input_dir / Path(r['path']).name) == (r['size_bytes'], r['mtime_ns']),
                f'{organ}: original input changed: {r["condition"]}')
    saved = result_root / organ / f'Human_{organ}_major_minor_promoters.tsv'
    saved_hash = sha256(saved)
    # Read unique genes at each pre-activity-filter stage from source annotations.
    all_annotations = {}
    for annotation, _ in ref.read_condition(files[0]):
        require(annotation[0] not in all_annotations, 'Duplicate promoterId')
        all_annotations[annotation[0]] = annotation
    eligible = [a for a in all_annotations.values() if not ref.is_internal(a[5], a[0])]
    with tempfile.TemporaryDirectory(prefix='organ_flowchart_verify_') as stage:
        stats, stamps, filename = ref.calculate_species('Human_' + organ, files, stage)
        require(filecmp.cmp(Path(stage) / filename, saved, shallow=False),
                f'{organ}: recomputation differs from saved classification; stop without reclassifying')
    prior = next(r for r in manifest['summary'] if r['organ'] == organ)
    for key, value in stats.items():
        if key != 'species':
            require(prior.get(key) == value, f'{organ}: summary mismatch: {key}')
    c = dict(organ=organ, conditions=len(files), input_genes=len({a[1] for a in all_annotations.values()}),
             input_promoters=len(all_annotations), noninternal_genes=len({a[1] for a in eligible}),
             noninternal_promoters=len(eligible), retained_genes=stats['genes'],
             retained_promoters=stats['output_promoters'], single_promoter_genes=stats['singleton_genes'],
             multi_promoter_genes=stats['multi_promoter_genes'], single_major_promoters=stats['singleton_genes'],
             multi_major_promoters=stats['multi_promoter_genes'], multi_minor_promoters=stats['minor_promoters'],
             total_major_promoters=stats['major_promoters'], total_minor_promoters=stats['minor_promoters'],
             positive_tied_genes=stats['tied_maximum_genes'])
    for noun in ('genes', 'promoters'):
        c['removed_internal_' + noun] = c['input_' + noun] - c['noninternal_' + noun]
        c['removed_zero_' + noun] = c['noninternal_' + noun] - c['retained_' + noun]
    validate_counts(c)
    require(sha256(saved) == saved_hash, 'Classification changed during validation')
    require(all(stamp(p) == value for p, value in stamps.items()), 'Input changed during validation')
    evidence = dict(organ=organ, original_conditions=recorded, classification_sha256=saved_hash,
                    classification_path=str(saved), source_recomputation='byte-identical',
                    annotation_sha256=sha256(files[0]))
    print(f'VALIDATED {organ}: {c["input_genes"]:,}/{c["input_promoters"]:,} -> '
          f'{c["noninternal_genes"]:,}/{c["noninternal_promoters"]:,} -> '
          f'{c["retained_genes"]:,}/{c["retained_promoters"]:,}', flush=True)
    return c, evidence


def draw(c):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'pdf.fonttype': 42, 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'text.usetex': False})
    fig = plt.figure(figsize=(183 / 25.4, 198 / 25.4))
    ax = fig.add_axes([.025, .065, .95, .91])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    boxes, texts = [], []
    ink, gray = '#202020', '#888888'
    orange, gold, purple = '#ff6a00', '#efc400', '#6257cc'

    def text(x, y, label, size=11, ha='center', color=ink, weight='normal'):
        t = ax.text(x, y, label, ha=ha, va='center', fontsize=size, color=color,
                    fontweight=weight, linespacing=1.35, zorder=4)
        texts.append(t)
        return t

    def box(x, y, width, height, label):
        patch = Rectangle((x, y), width, height, facecolor='white', edgecolor=gray, lw=.8)
        ax.add_patch(patch)
        t = text(x + width / 2, y + height / 2, label)
        boxes.append((t, patch))

    def arrow(x0, y0, x1, y1, color=ink, lw=.9, head=8):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                    color=color, lw=lw, mutation_scale=head, shrinkA=0, shrinkB=0))

    def exon(x, y, width, color, label=None):
        patch = Rectangle((x, y - .019), width, .038, color=color, lw=0, zorder=3)
        ax.add_patch(patch)
        if label:
            t = text(x + width / 2, y, label, size=10.5, color='white' if color == purple else ink)
            boxes.append((t, patch))

    def tss(x, y, end, height, color):
        ax.plot([x, x], [y + .020, y + height], color=color, lw=.9)
        arrow(x, y + height, end, y + height, color=color, head=7)

    text(.5, .970, f'Human | {c["organ"]}', size=15, weight='bold')
    text(.5, .932, f'{c["conditions"]} developmental-stage conditions', size=9, color='#555555')
    center = .425
    box(center - .135, .810, .270, .085, f'{c["input_genes"]:,} genes\n{c["input_promoters"]:,} promoters')
    arrow(center, .803, center, .718)
    text(.440, .762, 'Exclude internal promoters\n' + f'{c["removed_internal_promoters"]:,} promoters removed', size=9.2, ha='left')
    box(center - .135, .625, .270, .085, f'{c["noninternal_genes"]:,} genes\n{c["noninternal_promoters"]:,} promoters')
    arrow(center, .618, center, .533)
    text(.440, .576, f'Remove promoters inactive in\nevery {c["organ"]} condition\n'
         f'{c["removed_zero_promoters"]:,} promoters removed', size=9.2, ha='left')
    box(center - .135, .440, .270, .085, f'{c["retained_genes"]:,} genes\n{c["retained_promoters"]:,} promoters')
    ax.plot([center, center], [.433, .394], color=ink, lw=.9)
    ax.plot([.235, .695], [.394, .394], color=ink, lw=.9)
    arrow(.235, .394, .235, .358)
    arrow(.695, .394, .695, .358)
    box(.075, .275, .320, .075, f'{c["single_promoter_genes"]:,}\nsingle-promoter genes')
    box(.535, .275, .320, .075, f'{c["multi_promoter_genes"]:,}\nmulti-promoter genes')
    # Gene cartoons are explanatory symbols, not real coordinates or quantities.
    ax.plot([.035, .425], [.170, .170], color=orange, lw=1)
    for x, width in ((.035, .035), (.110, .050), (.205, .050), (.300, .045), (.375, .050)):
        exon(x, .170, width, orange)
    tss(.035, .170, .095, .065, ink)
    text(.235, .121, f'{c["single_major_promoters"]:,} Major promoters', size=10.6)
    text(.770, .235, f'{c["multi_major_promoters"]:,} Major promoters', size=10.6)
    ax.plot([.505, .960], [.170, .170], color='#a88e00', lw=1)
    exon(.505, .170, .045, gold)
    exon(.750, .170, .210, gold, 'Major')
    tss(.505, .170, .565, .065, '#8c7200')
    for x, y, end in ((.600, .106, .655), (.685, .042, .735)):
        ax.plot([x, .960], [y, y], color='#37a5bf', lw=1)
        exon(x, y, .040, purple)
        exon(.750, y, .210, purple, 'Minor')
        tss(x, y, end, .037, '#2699b5')
    text(.770, .003, f'{c["multi_minor_promoters"]:,} Minor promoters', size=10.6)
    fig.text(.5, .025, 'Means include all conditions equally, including zeros. One Major per retained gene.\n'
             'Single/multi-promoter groups are defined after both filters. Gene cartoons are schematic.',
             ha='center', va='center', fontsize=7.2, color='#555555', linespacing=1.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t, patch in boxes:
        a, b = t.get_window_extent(renderer), patch.get_window_extent(renderer)
        require(a.x0 >= b.x0 + 1 and a.x1 <= b.x1 - 1 and a.y0 >= b.y0 + 1 and a.y1 <= b.y1 - 1,
                f'Text escapes box: {t.get_text()}')
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b = t.get_window_extent(renderer)
            require(b.x0 >= 0 and b.y0 >= 0 and b.x1 <= fig.bbox.width and b.y1 <= fig.bbox.height,
                    f'Clipped label: {t.get_text()}')
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            require(not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)),
                    f'Overlapping labels: {a.get_text()} / {b.get_text()}')
    return fig


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', type=Path, default=here / 'results_by_organ')
    parser.add_argument('--input-dir', type=Path, default=here.parent.parent / '1.proactiv_result' / 'Human')
    parser.add_argument('--reference-script', type=Path, default=here.parent / 'define_major_minor_promoters.py')
    parser.add_argument('--output-dir', type=Path, default=here / 'flowcharts_by_organ')
    parser.add_argument('--organs', nargs='+', choices=ORDER)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--counts-input', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    require(args.workers >= 1, '--workers must be positive')
    if args.self_test:
        # Synthetic arithmetic test; no source data accessed and no figure saved.
        c = dict(zip(FIELDS, ['Liver', 2, 6, 12, 5, 9, 3, 5, 1, 2, 1, 2, 2, 3, 2, 1, 3, 2, 4, 0]))
        validate_counts(c)
        fig = draw(c)
        plt.close(fig)
        bad = dict(c, multi_major_promoters=3)
        try:
            validate_counts(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid MP Major count accepted')
        print('SELF-TEST PASS: count invariants, branch definitions, text bounds and label overlap checks')
        return
    require(not args.output_dir.exists() and not args.output_dir.is_symlink(), 'Choose a NEW --output-dir; existing outputs are never overwritten')
    require(args.organs is None or len(args.organs) == len(set(args.organs)), 'Duplicate organ selection')
    started = datetime.now(timezone.utc).isoformat()
    if args.counts_input:
        with args.counts_input.open(newline='', encoding='utf-8') as handle:
            counts = [{k: v if k == 'organ' else int(v) for k, v in row.items()}
                      for row in csv.DictReader(handle, delimiter='\t')]
        require(len({c['organ'] for c in counts}) == len(counts), 'Duplicate organ counts')
        chosen = args.organs or [organ for organ in ORDER if any(c['organ'] == organ for c in counts)]
        require(set(chosen) <= {c['organ'] for c in counts}, 'Requested organ missing from counts')
        counts = [next(c for c in counts if c['organ'] == organ) for organ in chosen]
        evidence = dict(mode='render-only; source validation NOT repeated', counts_sha256=sha256(args.counts_input))
    else:
        manifest_path = args.results_dir / 'run_manifest.json'
        manifest_hash = sha256(manifest_path)
        manifest = json.loads(manifest_path.read_text())
        require(manifest.get('status') == 'complete', 'Classification run is not complete')
        available = {r['organ'] for r in manifest['summary']}
        chosen = args.organs or [organ for organ in ORDER if organ in available]
        require(chosen and set(chosen) <= available, 'Requested organ missing from classification')
        jobs = [(organ, manifest, args.results_dir.resolve(), args.input_dir.resolve(),
                 args.reference_script.resolve()) for organ in chosen]
        if args.workers == 1:
            results = [collect(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
                results = list(pool.map(collect, jobs))
        require(sha256(manifest_path) == manifest_hash, 'Classification manifest changed')
        counts = [result[0] for result in results]
        evidence = dict(mode='all conditions recomputed; exact classification comparison',
                        manifest_sha256=manifest_hash, reference_sha256=manifest['reference_sha256'],
                        organs=[result[1] for result in results])
    require(counts, 'No organs to draw')
    for c in counts:
        validate_counts(c)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='.flowchart_staging_', dir=args.output_dir) as temporary:
        stage = Path(temporary)
        with PdfPages(stage / 'Human_all_organs_major_minor_flowcharts.pdf') as book:
            for c in counts:
                fig = draw(c)
                fig.savefig(stage / f'Human_{c["organ"]}_major_minor_flowchart.pdf', format='pdf')
                book.savefig(fig)
                plt.close(fig)
        with (stage / 'flowchart_counts.tsv').open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(counts)
        audit = dict(started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                     script_sha256=sha256(__file__), matplotlib_version=matplotlib.__version__,
                     evidence=evidence, counts=counts, formats=['pdf'], dimensions_mm=[183, 198],
                     editable_pdf_text=True, text_bounds_and_overlap_checks='PASS',
                     scope='Per organ, all available stages pooled, zeros included, equal condition weights',
                     gene_definition='Distinct geneId with >=1 remaining promoter at each stage',
                     promoter_definition='Distinct promoterId, not summed across conditions',
                     branch_definition='SP/MP assigned after both filters; MP Majors exclude SP Majors',
                     schematic_note='Gene cartoons are illustrative, not real gene structures or count-scaled marks')
        (stage / 'flowchart_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        for path in sorted(stage.iterdir()):
            os.link(path, args.output_dir / path.name)
    print(f'COMPLETE: {len(counts)} individual PDFs + {len(counts)}-page PDF in {args.output_dir}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, csv.Error) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
