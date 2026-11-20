#!/usr/bin/env python3
"""Developmental promoter-activity Pearson trajectories, inspired by Fig. 1E.

For each species and organ, compare every later sample with every sample at
that organ's earliest observed stage. Average the sample-pair Pearson r values,
not activities. Exclude the baseline stage itself. Retain all input promoters;
do not log, normalize, remove internal/zero promoters or select gene IDs.
Shading is a descriptive pair-bootstrap interval, not an independent biological
replicate confidence interval (pairs can share samples). See README.md.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
SPECIES = ('Human', 'Mouse', 'Rat', 'Rabbit', 'Macaque', 'Chicken')
ORGANS = ('Brain', 'Cerebellum', 'Heart', 'Kidney', 'Liver', 'Ovary', 'Testis')
COLORS = dict(Brain='#3399CC', Cerebellum='#33CCFF', Heart='#CC0000',
              Kidney='#CC9900', Liver='#339900', Ovary='#CC3399', Testis='#FF6600')
STYLES = {organ: '-' for organ in ORGANS}
FIGSIZE = (2.5, 2.5)
MAX_XTICK_LABELS = 18
TIME_TABLE = Path('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/revise/2.minor_major_promoter/gene/six_species_611_sample_mean_developmental_time_days.csv')
REFERENCE_URL = 'https://github.com/StatBiomed/promoter_activity/blob/main/figure1/plot_fig1E.py'
REFERENCE_GIT_BLOB = 'b78c1bbef424eda937eae49642fe546903bad610'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def stamp(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def read_activity(path):
    """Read R write.table's unnamed row index or an explicit promoterId index."""
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.reader(handle, delimiter='\t')
        header, first = next(reader), next(reader)
    require(len(set(header)) == len(header), f'{path}: duplicate headers')
    if len(first) == len(header) + 1:
        samples = header
    elif len(first) == len(header) and header[0] in ('', 'promoterId', 'Unnamed: 0'):
        samples = header[1:]
    else:
        raise ValueError(f'{path}: unexpected TSV row width')
    require(samples and all(samples), f'{path}: empty sample name')
    frame = pd.read_csv(path, sep='\t', header=None, skiprows=1,
                        names=['promoterId'] + samples, dtype={'promoterId': str},
                        low_memory=False).set_index('promoterId')
    require(frame.index.is_unique and frame.index.notna().all(), f'{path}: invalid promoter IDs')
    require((frame.index.str.strip() != '').all(), f'{path}: empty promoter ID')
    values = frame.to_numpy(dtype=np.float64)
    require(len(frame) >= 2 and np.isfinite(values).all() and (values >= 0).all(),
            f'{path}: missing, nonfinite, negative activity or too few promoters')
    return pd.DataFrame(values, index=frame.index, columns=samples)


def metadata_for_condition(condition, times):
    species, original_organ, original_stage = condition.split('_', 2)
    organ = {'WholeBrain': 'Brain', 'Forebrain': 'Brain',
             'Hindbrain': 'Cerebellum'}.get(original_organ, original_organ)
    require(species in SPECIES and organ in ORGANS, f'Unknown condition {condition}')
    stage = original_stage
    if species == 'Macaque' and organ == 'Kidney' and stage == '14ypb-15pb':
        stage = '14ypb-15ypb'
    key = f'{species}_{organ}_{stage}'
    require(key in times, f'Missing developmental time: {condition} -> {key}')
    days = float(times[key])
    require(np.isfinite(days) and days >= 0, f'Invalid developmental days: {condition}')
    return dict(species=species, organ=organ, original_organ=original_organ,
                stage=stage, original_stage=original_stage, condition=condition,
                time_lookup_key=key, developmental_time_days=days)


def discover(input_root, species):
    folder = Path(input_root) / species
    files = sorted(folder.glob(f'{species}_*_activity.tsv'))
    require(files, f'No per-condition inputs for {species}')
    require(len({p.resolve() for p in files}) == len(files), 'Duplicate input symlinks')
    ignored = sorted(str(p) for p in folder.glob('*_activity.tsv') if p not in files)
    return files, ignored


def load_species(input_root, species, times):
    files, ignored = discover(input_root, species)
    snapshots = {str(p): stamp(p) for p in files}
    frames, records, hashes, ids = [], [], {}, None
    for number, path in enumerate(files, 1):
        frame = read_activity(path)
        if ids is None:
            ids = frame.index
        require(len(frame) == len(ids) and set(frame.index) == set(ids),
                f'{path}: changed promoter set')
        frame = frame.loc[ids]
        info = metadata_for_condition(path.name[:-len('_activity.tsv')], times)
        frames.append(frame)
        records.extend(dict(info, sample_id=str(sample), source_file=str(path))
                       for sample in frame.columns)
        hashes[str(path)] = sha256(path)
        require(stamp(path) == snapshots[str(path)], f'Input changed: {path}')
        if number == 1 or number % 25 == 0 or number == len(files):
            print(f'{species}: read {number}/{len(files)} conditions', flush=True)
    activity = pd.concat(frames, axis=1)
    metadata = pd.DataFrame(records)
    require(activity.columns.is_unique and metadata.sample_id.is_unique,
            f'{species}: duplicate sample IDs across conditions')
    require(activity.columns.tolist() == metadata.sample_id.tolist(), 'Sample order mismatch')
    require(metadata.groupby('stage').developmental_time_days.nunique().max() == 1,
            f'{species}: inconsistent age for one stage label')
    require(set(metadata.organ) == set(ORGANS), f'{species}: missing expected organ')
    counts = dict(species=species, conditions=len(files), samples=len(metadata),
                  promoters=len(activity),
                  all_zero_promoters_included=int(activity.eq(0).all(axis=1).sum()),
                  ignored_aggregate_files=ignored)
    return activity, metadata, counts, hashes, snapshots


def sample_correlations(activity):
    values = np.asarray(activity, dtype=np.float64)
    require(values.ndim == 2 and min(values.shape) >= 2, 'Too few rows/columns')
    require(np.isfinite(values).all() and np.std(values, axis=0).min() > 0,
            'Nonfinite or constant sample activity column')
    corr = np.corrcoef(values, rowvar=False)
    require(np.isfinite(corr).all(), 'Nonfinite Pearson correlation')
    require(np.abs(corr).max() <= 1 + 1e-12, 'Out-of-range correlation')
    return np.clip(corr, -1, 1)


def pairwise_trajectories(correlation, metadata, n_promoters):
    """Keep each baseline-target pair explicitly, excluding all baseline samples."""
    metadata = metadata.copy()
    require(correlation.shape == (len(metadata), len(metadata)), 'Correlation/metadata mismatch')
    metadata['is_organ_baseline'] = False
    metadata['organ_baseline_stage'] = ''
    baseline_records, pair_records = [], []
    for organ in ORGANS:
        indices = np.flatnonzero(metadata.organ.eq(organ))
        require(len(indices) > 0, f'Missing organ {organ}')
        first_day = float(metadata.iloc[indices].developmental_time_days.min())
        baseline = indices[metadata.iloc[indices].developmental_time_days.eq(first_day)]
        targets = indices[metadata.iloc[indices].developmental_time_days.gt(first_day)]
        require(len(targets), f'{organ}: no stage after baseline')
        labels = sorted(metadata.iloc[baseline].stage.unique())
        require(len(labels) == 1, f'{organ}: ambiguous earliest-stage labels {labels}')
        label = labels[0]
        metadata.loc[baseline, 'is_organ_baseline'] = True
        metadata.loc[indices, 'organ_baseline_stage'] = label
        baseline_records.append(dict(species=metadata.species.iloc[0], organ=organ,
            baseline_stage=label, baseline_days=first_day, n_baseline_samples=len(baseline),
            n_later_samples=len(targets), n_later_stages=metadata.iloc[targets].stage.nunique(),
            baseline_samples=';'.join(metadata.iloc[baseline].sample_id),
            baseline_conditions=';'.join(sorted(metadata.iloc[baseline].condition.unique()))))
        for target in targets:
            query = metadata.iloc[target]
            for reference in baseline:
                pair_records.append(dict(species=query.species, organ=organ,
                    stage=query.stage, developmental_time_days=query.developmental_time_days,
                    baseline_stage=label, baseline_days=first_day,
                    baseline_sample=metadata.iloc[reference].sample_id,
                    target_sample=query.sample_id, target_condition=query.condition,
                    target_original_organ=query.original_organ,
                    target_original_stage=query.original_stage,
                    n_promoters=n_promoters, pearson_r=float(correlation[target, reference])))
    return pd.DataFrame(pair_records), pd.DataFrame(baseline_records), metadata


def bootstrap_pair_mean(values, iterations, seed):
    values = np.asarray(values, dtype=np.float64)
    require(len(values) > 0 and np.isfinite(values).all(), 'Invalid bootstrap values')
    if len(values) < 2:
        return math.nan, math.nan
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(values), size=(iterations, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def summarize_pairs(pairs, seed, iterations):
    records = []
    for (species, organ, stage), frame in pairs.groupby(['species', 'organ', 'stage'], sort=True):
        group_key = f'{seed}|{species}|{organ}|{stage}'.encode()
        group_seed = int.from_bytes(hashlib.sha256(group_key).digest()[:8], 'little')
        values = frame.pearson_r.to_numpy()
        low, high = bootstrap_pair_mean(values, iterations, group_seed)
        nb, nt = frame.baseline_sample.nunique(), frame.target_sample.nunique()
        require(len(frame) == nb * nt, f'Incomplete cross-product: {species}/{organ}/{stage}')
        require(not frame.duplicated(['baseline_sample', 'target_sample']).any(), 'Duplicate pair')
        records.append(dict(species=species, organ=organ, stage=stage,
            developmental_time_days=frame.developmental_time_days.iloc[0],
            baseline_stage=frame.baseline_stage.iloc[0],
            n_baseline_samples=nb, n_target_samples=nt, n_sample_pairs=len(frame),
            n_promoters=frame.n_promoters.iloc[0], mean_pearson_r=values.mean(),
            pairwise_bootstrap_lower=low, pairwise_bootstrap_upper=high,
            min_pair_r=values.min(), max_pair_r=values.max(),
            sample_pair_sd=values.std(ddof=1) if len(values) > 1 else math.nan))
    result = pd.DataFrame(records)
    organ_order = dict(zip(ORGANS, range(len(ORGANS))))
    return result.assign(_order=result.organ.map(organ_order)).sort_values(
        ['_order', 'developmental_time_days', 'stage']).drop(columns='_order').reset_index(drop=True)


def compact_tick_indices(n_stages, max_labels=MAX_XTICK_LABELS):
    """Thin labels, never points; keep actual chronological stage names."""
    require(n_stages > 0 and max_labels >= 2, 'Invalid tick count')
    return np.unique(np.linspace(0, n_stages - 1, min(n_stages, max_labels), dtype=int))


def draw_species(species, summary, baselines, y_limits, destination):
    stages = summary[['stage', 'developmental_time_days']].drop_duplicates().sort_values(
        ['developmental_time_days', 'stage']).stage.tolist()
    require(len(stages) == len(set(stages)), 'Ambiguous plot stages')
    positions = dict(zip(stages, range(len(stages))))
    style = {'font.family': 'DejaVu Sans', 'font.size': 7,
             'pdf.fonttype': 42, 'pdf.compression': 0, 'axes.grid': False,
             'axes.linewidth': .7, 'axes.edgecolor': '#777777',
             'axes.spines.top': True, 'axes.spines.right': True,
             'axes.facecolor': 'white', 'figure.facecolor': 'white',
             'xtick.direction': 'out', 'ytick.direction': 'out',
             'xtick.major.size': 2, 'ytick.major.size': 2,
             'xtick.major.width': .5, 'ytick.major.width': .5,
             'xtick.major.pad': 1.5, 'ytick.major.pad': 2}
    with mpl.rc_context(style):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for organ in ORGANS:
            frame = summary.loc[summary.organ.eq(organ)].sort_values('developmental_time_days')
            if frame.empty:
                continue
            x = np.array([positions[s] for s in frame.stage])
            ax.plot(x, frame.mean_pearson_r, color=COLORS[organ], linestyle='-',
                    linewidth=1.0, marker=None, rasterized=False, zorder=2)
            light = .82 + .18 * np.array(to_rgb(COLORS[organ]))
            ax.fill_between(x, frame.pairwise_bootstrap_lower, frame.pairwise_bootstrap_upper,
                            color=light, linewidth=0, rasterized=False, zorder=1)
        ticks = compact_tick_indices(len(stages))
        ax.set_xticks(ticks, [stages[i] for i in ticks], rotation=90, fontsize=7)
        ax.set_xlim(-.5, len(stages) - .5)
        ax.set_ylim(*y_limits)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(.05))
        ax.tick_params(axis='y', labelsize=7)
        # Baselines vary by organ, so do not falsely label all curves 'to 10.5'.
        ax.set_ylabel("Pearson's correlation\nto earliest stage", fontsize=8.5, labelpad=2)
        ax.set_xlabel('')
        ax.set_title(species, fontsize=10, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#777777')
            spine.set_linewidth(.7)
        ax.grid(False)
        # Reference-style panel: no large legend or methodological footer.
        # Baseline identities and interval interpretation remain in tables/README.
        fig.tight_layout(pad=.45)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = ax.get_xticklabels() + [ax.yaxis.label, ax.title]
        low, high = ax.get_ylim()
        texts += [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels())
                  if low <= value <= high]
        for text in texts:
            box = text.get_window_extent(renderer)
            require(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width
                    and box.y1 <= fig.bbox.height, f'{species}: clipped compact label')
        require(all(line.get_linestyle() == '-' for line in ax.lines),
                'Every organ must use a solid line')
        with mpl.rc_context({'pdf.compression': 0}):
            fig.savefig(destination, format='pdf', facecolor='white')
        dimensions = [float(x) for x in fig.get_size_inches()]
        plt.close(fig)
    return dimensions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=HERE.parent / '1.proactiv_result')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'output')
    parser.add_argument('--time-table', type=Path, default=TIME_TABLE)
    parser.add_argument('--species', nargs='+', choices=SPECIES, default=list(SPECIES))
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=666)
    args = parser.parse_args(argv)
    require(args.bootstrap >= 100, 'Use at least 100 bootstrap iterations')
    require(len(set(args.species)) == len(args.species), 'Duplicate species argument')
    require(not args.output_dir.is_symlink(), 'Output must not be a symlink')
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
            'Output directory is not empty; select a new --output-dir (no automatic overwrite)')
    times_frame = pd.read_csv(args.time_table)
    require(times_frame.sample_name.is_unique, 'Duplicate curated time keys')
    times = times_frame.set_index('sample_name').developmental_time_days.to_dict()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
        parameters=params, reference_url=REFERENCE_URL, reference_git_blob=REFERENCE_GIT_BLOB,
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      matplotlib=mpl.__version__), species={}, input_sha256={
                      str(args.time_table): sha256(args.time_table), str(Path(__file__)): sha256(__file__)},
        method=dict(population='All input promoters, including internal and all-zero rows',
            transform='None: supplied activity values', baseline='Earliest observed stage within each organ',
            estimator='Arithmetic mean of all later-sample x baseline-sample Pearson r values',
            band='2.5th/97.5th percentile of 1000 pair-resampled means by default; descriptive, not independent biological-replicate CI',
            source_difference='Reference uses an older alternative-promoter matrix intersected with RPKM gene IDs; this run uses the updated all-promoter proactiv_redo activity files without those extra filters',
            aliases='WholeBrain/Forebrain pooled with Brain; Hindbrain with Cerebellum; original labels retained',
            stage_alias='Macaque Kidney 14ypb-15pb -> 14ypb-15ypb; original label retained',
            x_scale='Chronological categories with equal spacing, not elapsed days',
            baseline_excluded=True, independent_species=True))
    manifest_path = args.output_dir / 'run_manifest.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    manifest['style'] = dict(figure_inches=list(FIGSIZE), all_organs_solid=True,
        line_width=1.0, marker=None, title='species name only', title_fontsize=10,
        tick_fontsize=7, ylabel_fontsize=8.5, max_xtick_labels=MAX_XTICK_LABELS,
        all_timepoints_plotted=True, grid=False, frame='four grey spines',
        legend_on_figure=False, footer_on_figure=False, PDF_only=True)
    save_manifest()
    prepared, all_snapshots = {}, {}
    for species in args.species:
        activity, metadata, counts, hashes, snapshots = load_species(args.input_root, species, times)
        corr = sample_correlations(activity)
        pairs, baselines, metadata = pairwise_trajectories(corr, metadata, len(activity))
        # Independent pandas Series.corr checks: one later pair for every organ.
        for organ in ORGANS:
            row = pairs.loc[pairs.organ.eq(organ)].iloc[0]
            expected = activity[row.target_sample].corr(activity[row.baseline_sample])
            require(np.isclose(row.pearson_r, expected, rtol=1e-12, atol=1e-12),
                    f'{species}/{organ}: independent Pearson check failed')
        summary = summarize_pairs(pairs, args.seed, args.bootstrap)
        counts.update(baseline_samples=int(metadata.is_organ_baseline.sum()),
                      later_samples=int((~metadata.is_organ_baseline).sum()),
                      correlation_pairs=len(pairs), organ_stage_summaries=len(summary),
                      independent_pandas_pearson_checks=len(ORGANS))
        folder = args.output_dir / species
        folder.mkdir()
        for name, frame in [('sample_metadata.tsv', metadata), ('organ_baselines.tsv', baselines),
                            ('sample_pair_correlations.tsv', pairs), ('time_correlation_summary.tsv', summary)]:
            frame.to_csv(folder / name, sep='\t', index=False)
        pd.DataFrame(corr, index=metadata.sample_id, columns=metadata.sample_id).to_csv(
            folder / 'sample_pearson_correlation.tsv.gz', sep='\t', compression='gzip')
        prepared[species] = (summary, baselines)
        manifest['species'][species] = counts
        manifest['input_sha256'].update(hashes)
        all_snapshots.update(snapshots)
        print(f'{species}: {len(pairs)} sample pairs; baseline stages ' +
              ', '.join(f'{r.organ}={r.baseline_stage}' for r in baselines.itertuples()), flush=True)
        save_manifest()
        del activity, corr, pairs
    lower = min(float(frame[['mean_pearson_r', 'pairwise_bootstrap_lower']].min().min())
                for frame, _ in prepared.values())
    y_limits = (max(-1., math.floor((lower - .025) / .05) * .05), 1.)
    manifest['shared_y_limits'] = list(y_limits)
    for species, (summary, baselines) in prepared.items():
        pdf = args.output_dir / species / f'{species}_promoter_activity_time_correlation.pdf'
        dimensions = draw_species(species, summary, baselines, y_limits, pdf)
        manifest['species'][species]['figure_inches'] = dimensions
        print('SAVED', pdf, flush=True)
    require(all(stamp(p) == old for p, old in all_snapshots.items()), 'An input changed during this run')
    for species in args.species:
        require(set(map(str, discover(args.input_root, species)[0])) ==
                {p for p in all_snapshots if Path(p).parent.name == species}, 'Input inventory changed')
    pd.DataFrame([{k: v for k, v in manifest['species'][species].items()
                   if k != 'ignored_aggregate_files'} for species in args.species]).to_csv(
        args.output_dir / 'run_summary.tsv', sep='\t', index=False)
    files = sorted(p for p in args.output_dir.rglob('*') if p.is_file() and p != manifest_path)
    pd.DataFrame([dict(path=str(p.relative_to(args.output_dir)), sha256=sha256(p)) for p in files]).to_csv(
        args.output_dir / 'output_checksums.tsv', sep='\t', index=False)
    manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat())
    save_manifest()
    print('ALL COMPLETE', args.output_dir, flush=True)


if __name__ == '__main__':
    main()


