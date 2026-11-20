#!/usr/bin/env python3
"""Captum IG for each species x organ: mean RAW activity over ALL its stages.

Uses existing independent species checkpoints and every TEST promoter, without
activity/major/minor/MP filtering. Each task is inverse-zscored and inverse-
log2p1-transformed BEFORE the organ-stage arithmetic mean (no clipping).
Baseline: ONE positional mean one-hot of the prepared 128 TRAIN references.
This is not mean IG over 128 baselines. Inputs and trained models are unchanged.

Defaults run a small pilot for each selected species/organ, not a full job.
Use --check-only first; --mode full requests all TEST sequences. Resume requires
the same code, inputs, device, target selection, and numerical settings.
Dependencies: existing torch, captum, numpy; no install or plotting performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import importlib.util
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
import torch
import captum

from ig_core import RawOrganMean, adaptive_ig_one, onehot, select_organ_targets
from ig_inputs import load_prepared_bundle

SPECIES = ('Human','Mouse','Rat','Rabbit','Macaque','Chicken')
VERSION = 'organ_raw_mean_captum_ig_v1'
HERE = Path(__file__).resolve().parent
LENGTH = 10000
QUALITY_FIELDS = [
    'attribution_row_0based','source_array_index_0based','sample_id','promoterId','geneId',
    'organ','n_stages','predicted_mean_raw','baseline_mean_raw','explained_difference',
    'observed_mean_raw','n_steps','converged','completeness_pass','map_pass',
    'attribution_sum','stored_attribution_sum','stored_completeness_pass',
    'completeness_signed_error','completeness_absolute_error','completeness_tolerance','captum_delta',
    'map_l1_change','map_l1_mass','map_relative_l1_change','position_map_l1_change',
    'position_map_l1_mass','position_map_relative_l1_change','map_cosine','near_zero_map',
    'n_canonical_positions','n_unknown_positions','elapsed_seconds',
    'hypothetical_projection_max_absolute_error','ig_row_sha256','position_row_sha256','hypothetical_row_sha256']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    os.replace(temporary,path)


def read_tsv(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        require(reader.fieldnames and len(reader.fieldnames)==len(set(reader.fieldnames)),f'Invalid header: {path}')
        rows=list(reader)
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows),f'Ragged TSV: {path}')
    return rows


def write_tsv(path, rows, fields=None):
    require(rows or fields,f'Empty table without columns: {path}')
    with Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def link_file(destination, source):
    destination,source=Path(destination),Path(source).resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        require(destination.is_symlink() and destination.resolve(strict=True)==source,
                f'Refusing to replace existing file/link: {destination}')
    else:
        destination.symlink_to(source)


def truth(value):
    return str(value).lower()=='true'


def selected_indices(n, args):
    require(n>0,'Empty TEST set')
    if args.mode=='pilot':
        count=min(n,args.pilot_count)
        return np.unique(np.linspace(0,n-1,count,dtype=np.int64)).tolist()
    if args.max_promoters:
        require(args.max_promoters<=n,'--max-promoters exceeds TEST set')
        return np.unique(np.linspace(0,n-1,args.max_promoters,dtype=np.int64)).tolist()
    return list(range(n))


def verify_original_sources(root, species, bundle):
    data=root/'datasets'/species
    run=root/'runs'/species
    source_files={'model_definition.py':root/'2.train_six_species_models.py',
                  'best_model.pt':run/'best_model.pt','run_config.json':run/'run_config.json',
                  'run_status.json':run/'run_status.json','input_manifest.json':run/'input_manifest.json',
                  **{name:data/name for name in ['target_columns.tsv','test.tsv.gz','test.fa.gz','train.tsv.gz','train.fa.gz']}}
    require(read_json(run/'run_status.json').get('status')=='complete',f'Training is incomplete: {species}')
    training_manifest=read_json(run/'input_manifest.json')
    expected={Path(r['path']).name:r['sha256'] for r in training_manifest}
    require(len(expected)==len(training_manifest),'Duplicate training input basenames')
    verified={}
    for name,path in source_files.items():
        require(path.is_file(),f'Missing source: {path}')
        digest=sha(path)
        prior=bundle['metadata']['source_files'].get(name,{})
        require(digest==prior.get('sha256') and path.stat().st_size==prior.get('bytes'),
                f'Original source no longer matches prepared inputs: {species}/{name}')
        if name in expected:
            require(digest==expected[name],f'Dataset no longer matches trained checkpoint: {species}/{name}')
        verified[name]={'path':str(path.resolve()),'sha256':digest}
    return verified


def load_species(root, inputs_root, species, organs):
    bundle=load_prepared_bundle(inputs_root/species,species,root)
    sources=verify_original_sources(root,species,bundle)
    checkpoint=torch.load(sources['best_model.pt']['path'],map_location='cpu',weights_only=True)
    require(checkpoint.get('format_version')==3 and checkpoint.get('model_version')=='head_layernorm_v1',
            f'Unsupported checkpoint revision: {species}')
    require(checkpoint.get('target_transform')==checkpoint.get('scaler_space')=='log2(activity + 1)',
            f'Unsupported label scale: {species}')
    require(checkpoint['target_columns']==bundle['target_columns'] and checkpoint['config']['species']==species,
            f'Checkpoint species/output order mismatch: {species}')
    require(checkpoint['config']['script_sha256']==sources['model_definition.py']['sha256'],
            f'Checkpoint and model-definition versions differ: {species}')
    mean=checkpoint['label_mean'].cpu().numpy().astype(np.float64)
    std=checkpoint['label_std'].cpu().numpy().astype(np.float64)
    require(np.array_equal(mean,bundle['mean']) and np.array_equal(std,bundle['std']),
            f'Checkpoint/prepared scalers differ: {species}')
    name='_organ_ig_training_'+sources['model_definition.py']['sha256'][:12]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,sources['model_definition.py']['path'])
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    module=sys.modules[name]
    require(checkpoint['architecture']=={**module.ARCHITECTURE,'n_outputs':bundle['n_outputs']},
            f'Checkpoint architecture differs: {species}')
    model=module.PromoterDilatedCNN(bundle['n_outputs'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return dict(species=species,bundle=bundle,sources=sources,model=model,
                groups=select_organ_targets(bundle['targets'],species,organs))


def prediction_check(item, rows, selected, device):
    """Verify the actual wrapper against stored TEST predictions, by sample ID."""
    species,bundle=item['species'],item['bundle']
    path=Path(item['sources']['best_model.pt']['path']).parent/'test_predictions.tsv.gz'
    wanted={bundle['promoters'][i]['sample_id']:i for i in selected}
    saved={}
    with gzip.open(path,'rt',newline='') as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            if row['sample_id'] in wanted:
                require(row['sample_id'] not in saved,'Duplicate saved TEST prediction')
                saved[row['sample_id']]=row
    require(set(saved)==set(wanted),f'Missing saved TEST predictions: {species}')
    indices=[r['output_index_0based'] for r in rows]
    wrapper=RawOrganMean(item['model'],indices,bundle['mean'][indices],bundle['std'][indices]).eval().to(device)
    records=[]
    for sid,i in wanted.items():
        old=saved[sid]
        require(all(old[k]==bundle['promoters'][i][k] for k in ['promoterId','geneId','seqnames','start','strand']),
                'Stored prediction promoter metadata differs')
        with torch.no_grad():
            actual=wrapper.activities(torch.from_numpy(onehot(bundle['codes'][i:i+1])).to(device)).cpu().numpy()[0]
        expected=np.array([float(old['pred::'+r['target_column']]) for r in rows])
        observed=np.array([float(old['true::'+r['target_column']]) for r in rows])
        require(np.allclose(actual,expected,rtol=2e-4,atol=1e-4),f'Raw prediction mismatch: {species}/{sid}')
        require(np.allclose(observed,bundle['observed'][i,indices],rtol=1e-10,atol=1e-10),
                f'Observed activity/task order mismatch: {species}/{sid}')
        records.append(dict(sample_id=sid,source_array_index_0based=i,max_prediction_absolute_error=float(np.abs(actual-expected).max())))
    return wrapper,dict(status='passed',source=str(path),source_sha256=sha(path),checked=records)


def prepare_species_output(folder,item,selected,resume):
    bundle=item['bundle']
    if not resume:
        folder.mkdir()
        (folder/'inputs').mkdir()
        for name,path in bundle['paths'].items():
            if name.endswith(('.npy','.tsv')) or name=='manifest':
                destination='prepared_input_manifest.json' if name=='manifest' else name
                link_file(folder/'inputs'/destination,path)
        for name,record in item['sources'].items():
            link_file(folder/'inputs'/('source_'+name),record['path'])
        np.save(folder/'baseline_mean_train_onehot.npy',bundle['baseline'],allow_pickle=False)
        mask=np.lib.format.open_memmap(folder/'canonical_mask.npy',mode='w+',dtype=np.bool_,shape=(len(selected),LENGTH))
        for j,i in enumerate(selected):mask[j]=bundle['codes'][i]<4
        mask.flush();del mask
        write_tsv(folder/'promoter_index.tsv',[
            dict(attribution_row_0based=j,source_array_index_0based=i,**bundle['promoters'][i]) for j,i in enumerate(selected)])
        write_tsv(folder/'baseline_reference_promoters.tsv',bundle['reference_promoters'])
        save_json(folder/'shared_inputs.json',dict(species=item['species'],selected_indices=selected,
                  baseline_sha256=sha(folder/'baseline_mean_train_onehot.npy'),
                  mask_sha256=sha(folder/'canonical_mask.npy'),promoter_index_sha256=sha(folder/'promoter_index.tsv')))
    shared=read_json(folder/'shared_inputs.json')
    require(shared['selected_indices']==selected and np.array_equal(np.load(folder/'baseline_mean_train_onehot.npy'),bundle['baseline']),
            'Resume baseline/selection differs')
    for name,key in [('baseline_mean_train_onehot.npy','baseline_sha256'),('canonical_mask.npy','mask_sha256'),('promoter_index.tsv','promoter_index_sha256')]:
        require(sha(folder/name)==shared[key],f'Shared input changed: {name}')


def array_layout(n,hypothetical):
    result={'ig_attributions.npy':(n,4,LENGTH),'ig_position_sum.npy':(n,LENGTH)}
    if hypothetical:result['ig_hypothetical.npy']=(n,4,LENGTH)
    return result


def run_organ(item,organ,rows,selected,folder,args):
    bundle=item['bundle']; species=item['species']
    layout=array_layout(len(selected),args.write_hypothetical)
    if folder.is_dir() and (folder/'status.json').is_file():
        previous=read_json(folder/'status.json')
        if previous['status'] in ('complete','pilot_complete','subset_complete'):
            manifest=read_json(folder/'manifest.json')
            require(all(sha(folder/name)==digest for name,digest in manifest['output_sha256'].items()),
                    f'Completed organ output changed: {species}/{organ}')
            print('SKIP_COMPLETE',species,organ,flush=True)
            return previous
    resuming=folder.exists()
    require(not resuming or args.resume,f'Existing organ output: {folder}')
    if not resuming:folder.mkdir()
    status=dict(status='starting',species=species,organ=organ,selected_promoters=len(selected),updated_utc=now())
    save_json(folder/'status.json',status)
    try:
        indices=[r['output_index_0based'] for r in rows]
        ledger=[dict(species=species,organ=organ,stage=r['stage'],target_column=r['target_column'],
                     output_index_0based=i,weight=1/len(rows),train_log2p1_mean=float(bundle['mean'][i]),
                     train_log2p1_std=float(bundle['std'][i])) for r,i in zip(rows,indices)]
        pilot=[selected[j] for j in np.unique(np.linspace(0,len(selected)-1,min(args.pilot_count,len(selected)),dtype=int))]
        # Compute one reference prediction and match several real TEST predictions.
        wrapper,prediction_audit=prediction_check(item,rows,pilot,args.device)
        baseline=torch.from_numpy(bundle['baseline']).to(args.device)
        with torch.no_grad():fb=float(wrapper(baseline).item())
        require(math.isfinite(fb),'Nonfinite baseline mean activity')
        if not resuming:
            arrays={name:np.lib.format.open_memmap(folder/name,mode='w+',dtype=np.float32,shape=shape) for name,shape in layout.items()}
            for array in arrays.values():array[:]=np.nan;array.flush()
            write_tsv(folder/'target_stage_weights.tsv',ledger)
            link_file(folder/'promoter_index.tsv',folder.parent/'promoter_index.tsv')
            link_file(folder/'canonical_mask.npy',folder.parent/'canonical_mask.npy')
            link_file(folder/'baseline_mean_train_onehot.npy',folder.parent/'baseline_mean_train_onehot.npy')
            manifest=dict(version=VERSION,species=species,organ=organ,started_utc=now(),n_stages=len(rows),
                          n_available_test_promoters=bundle['n_promoters'],n_selected_promoters=len(selected),
                          target='Arithmetic mean of predicted RAW promoter activity across ALL recorded stages of this organ',
                          inverse='expm1(ln(2)*(z_t*TRAIN_std_t+TRAIN_mean_t)) per task BEFORE mean; no clipping',
                          baseline='One positional mean-one-hot of 128 TRAIN references, NOT mean IG over 128 baselines',
                          explained_quantity='F(sequence)-F(baseline)',method='Captum IntegratedGradients; multiply_by_inputs=True; gausslegendre',
                          array_shapes={name:list(shape) for name,shape in layout.items()},channels=['A','C','G','T'],
                          position_score='SUM of all four actual IG channels; do not keep only the observed-base channel',
                          hypothetical='Optional categorical scores from separately integrated multipliers; valid observed projection only at canonical sites',
                          unknown_bases='Input code4 stays all-zero. Actual IG retained; canonical_mask marks unavailable A/C/G/T identities. Do not treat N as a nucleotide.',
                          pending_rows='NaN or uncommitted values; only quality.tsv rows are committed',
                          quality='At least two integration levels; require completeness AND consecutive full/position map L1 stability',
                          prediction_check=prediction_audit,baseline_predicted_mean_raw=fb,
                          caveats=['Not TF binding/causal evidence','Means weight sampled stages equally, not chronological interval lengths',
                                   'All TEST promoters included, no activity/major/minor filtering',
                                   'Numerical tolerances are declared QA choices, not universal Captum defaults'])
            save_json(folder/'manifest.json',manifest)
            committed=[]
        else:
            manifest=read_json(folder/'manifest.json')
            require(manifest['species']==species and manifest['organ']==organ and manifest['n_selected_promoters']==len(selected),
                    'Resume organ/shape mismatch')
            require(read_tsv(folder/'target_stage_weights.tsv')==[{k:str(v) for k,v in row.items()} for row in ledger],
                    'Resume stage target weights changed')
            arrays={name:np.lib.format.open_memmap(folder/name,mode='r+') for name in layout}
            for name,array in arrays.items():
                require(array.shape==layout[name] and array.dtype==np.float32,'Resume array layout/dtype differs')
            committed=read_tsv(folder/'quality.tsv') if (folder/'quality.tsv').is_file() else []
            require(len({int(r['attribution_row_0based']) for r in committed})==len(committed),'Duplicate committed IG row')
            for r in committed:
                j=int(r['attribution_row_0based']);i=int(r['source_array_index_0based'])
                require(0<=j<len(selected) and selected[j]==i and r['sample_id']==bundle['promoters'][i]['sample_id'],
                        'Resume committed row identity differs')
                require(truth(r['converged']) and truth(r['stored_completeness_pass']),
                        'Cannot resume failed numerical QA; start a NEW run with revised integration settings')
                for name,key in [('ig_attributions.npy','ig_row_sha256'),('ig_position_sum.npy','position_row_sha256'),('ig_hypothetical.npy','hypothetical_row_sha256')]:
                    if name in arrays:require(array_sha(arrays[name][j])==r[key],f'Resume row checksum mismatch: {name}/{j}')
        done={int(r['source_array_index_0based']) for r in committed}
        output_row={i:j for j,i in enumerate(selected)}
        order=pilot+[i for i in selected if i not in set(pilot)]
        for i in order:
            if i in done:continue
            x=torch.from_numpy(onehot(bundle['codes'][i:i+1])).to(args.device)
            result=adaptive_ig_one(wrapper,x,baseline,steps=args.steps,max_steps=args.max_steps,
                    internal_batch_size=args.internal_batch_size,atol=args.atol,rtol=args.rtol,
                    map_atol=args.map_atol,map_rtol=args.map_rtol,return_multipliers=args.write_hypothetical,fb=fb)
            q=result['quality'];j=output_row[i]
            actual=np.asarray(result['attributions'],dtype=np.float32)
            position=actual.sum(axis=0,dtype=np.float64).astype(np.float32)
            require(actual.shape==(4,LENGTH) and np.isfinite(actual).all() and np.isfinite(position).all(),'Invalid stored IG map')
            stored_sum=float(actual.sum(dtype=np.float64))
            stored_pass=abs(stored_sum-result['explained_difference'])<=q['completeness_tolerance']
            arrays['ig_attributions.npy'][j]=actual
            arrays['ig_position_sum.npy'][j]=position
            if args.write_hypothetical:
                hypothetical=np.asarray(result['hypothetical'],dtype=np.float32)
                require(np.isfinite(hypothetical).all(),'Float32 hypothetical overflow')
                arrays['ig_hypothetical.npy'][j]=hypothetical
            for array in arrays.values():array.flush()
            row=dict(attribution_row_0based=j,source_array_index_0based=i,
                     sample_id=bundle['promoters'][i]['sample_id'],promoterId=bundle['promoters'][i]['promoterId'],
                     geneId=bundle['promoters'][i]['geneId'],organ=organ,n_stages=len(rows),
                     predicted_mean_raw=result['target_input'],baseline_mean_raw=fb,
                     explained_difference=result['explained_difference'],
                     observed_mean_raw=float(bundle['observed'][i,indices].mean()),
                     **{k:v for k,v in q.items() if k!='trials'})
            row.update(stored_attribution_sum=stored_sum,stored_completeness_pass=bool(stored_pass),
                       ig_row_sha256=array_sha(actual),position_row_sha256=array_sha(position),
                       hypothetical_row_sha256=array_sha(hypothetical) if args.write_hypothetical else '')
            with (folder/'quadrature_trials.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(source_array_index_0based=i,sample_id=row['sample_id'],trials=q['trials']),allow_nan=False)+'\n')
                stream.flush();os.fsync(stream.fileno())
            with (folder/'quality.tsv').open('a',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=QUALITY_FIELDS,delimiter='\t',lineterminator='\n',extrasaction='ignore')
                # A previous interruption may have committed just the header.
                # Do not append a duplicate header when resuming that case.
                if stream.tell()==0:writer.writeheader()
                writer.writerow({key:row.get(key,'') for key in QUALITY_FIELDS})
                stream.flush();os.fsync(stream.fileno())
            committed.append(row);done.add(i)
            status.update(status='running',processed=len(done),total=len(selected),updated_utc=now(),last_source_array_index=i,
                          last_n_steps=q['n_steps'],last_converged=bool(q['converged'] and stored_pass))
            save_json(folder/'status.json',status)
            print(json.dumps(dict(species=species,organ=organ,processed=len(done),total=len(selected),source_array_index=i,
                                 steps=q['n_steps'],converged=bool(q['converged'] and stored_pass),seconds=q['elapsed_seconds'])),flush=True)
            require(q['converged'] and stored_pass,
                    'Numerical IG QA failed; scores are flagged, not accepted. Start a NEW run with larger --max-steps if needed.')
        require(len(done)==len(selected),'Incomplete IG row set')
        for array in arrays.values():array.flush()
        manifest.update(finished_utc=now(),output_sha256={name:sha(folder/name) for name in
                        [*layout,'target_stage_weights.tsv','quality.tsv','quadrature_trials.jsonl']})
        save_json(folder/'manifest.json',manifest)
        status.update(status='complete' if args.mode=='full' and not args.max_promoters else ('pilot_complete' if args.mode=='pilot' else 'subset_complete'),
                      finished_utc=now(),processed=len(done),total=len(selected),all_rows_passed=True)
        save_json(folder/'status.json',status)
        del arrays,wrapper
        return status
    except BaseException as exc:
        status.update(status='failed_or_quality_failed',error=f'{type(exc).__name__}: {exc}',updated_utc=now())
        save_json(folder/'status.json',status)
        raise


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--redo-root',type=Path,default=HERE.parent)
    parser.add_argument('--inputs-root',type=Path,help='Existing prepared inputs parent; automatically finds explain_development/run_deeplift/inputs')
    parser.add_argument('--species',nargs='+',choices=SPECIES,default=list(SPECIES))
    parser.add_argument('--organs',nargs='+',help='Exact organ labels; default every recorded organ, including separate WholeBrain labels')
    parser.add_argument('--mode',choices=['pilot','full'],default='pilot')
    parser.add_argument('--pilot-count',type=int,default=3)
    parser.add_argument('--max-promoters',type=int,default=0,help='0=ALL TEST in full mode; positive values are explicitly marked as a subset')
    parser.add_argument('--device',default='cpu',help='Explicit cpu or cuda:N; default does not claim GPU resources')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--internal-batch-size',type=int,default=4)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--max-steps',type=int,default=2048)
    parser.add_argument('--atol',type=float,default=1e-3)
    parser.add_argument('--rtol',type=float,default=.01)
    parser.add_argument('--map-atol',type=float,default=1e-3)
    parser.add_argument('--map-rtol',type=float,default=.10)
    parser.add_argument('--write-hypothetical',action='store_true',help='Optional extra integral/pass for categorical motif scores; increases compute/storage')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check-only',action='store_true',help='Read/validate all inputs and report targets/disk estimate; no attribution or writes')
    args=parser.parse_args()
    require(len(set(args.species))==len(args.species),'Duplicate requested species')
    require(args.pilot_count>=1 and args.max_promoters>=0 and args.threads>=1 and args.internal_batch_size>=1,'Invalid counts')
    require(0<args.steps<args.max_steps,'Require two or more integration levels: 0<steps<max-steps')
    require(all(math.isfinite(v) and v>=0 for v in [args.atol,args.rtol,args.map_atol,args.map_rtol]),'Invalid tolerances')
    require(not args.resume or args.output_dir is not None,'--resume requires an explicit --output-dir')
    require(not (args.mode=='pilot' and args.max_promoters),'Use --pilot-count in pilot mode, not --max-promoters')
    args.redo_root=args.redo_root.resolve(strict=True)
    if args.inputs_root is None:
        choices=[args.redo_root/'explain_development/run_deeplift/inputs',args.redo_root/'run_deeplift/inputs']
        args.inputs_root=next((p for p in choices if p.is_dir()),choices[0])
    args.inputs_root=args.inputs_root.resolve(strict=True)
    return args


def main():
    args=parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args.device!='cpu':
        require(args.device.startswith('cuda:') and torch.cuda.is_available(),'Select cpu or an available explicit cuda:N')
        torch.cuda.set_device(args.device)
    items=[];plan=[]
    for species in args.species:
        print('VERIFY_SPECIES',species,flush=True)
        item=load_species(args.redo_root,args.inputs_root,species,args.organs)
        selected=selected_indices(item['bundle']['n_promoters'],args)
        item['selected']=selected;items.append(item)
        for organ,rows in item['groups'].items():
            plan.append(dict(species=species,organ=organ,n_stages=len(rows),n_test_promoters=item['bundle']['n_promoters'],
                             selected_promoters=len(selected),stages=[r['stage'] for r in rows],
                             output_indices=[r['output_index_0based'] for r in rows],weight_per_stage=1/len(rows),
                             estimated_array_bytes=len(selected)*LENGTH*(20+(16 if args.write_hypothetical else 0))))
    source_hashes={p.name:sha(p) for p in [Path(__file__),HERE/'ig_core.py',HERE/'ig_inputs.py']}
    signature=dict(version=VERSION,species=args.species,requested_organs=args.organs,mode=args.mode,device=args.device,
                   pilot_count=args.pilot_count,max_promoters=args.max_promoters,threads=args.threads,
                   steps=args.steps,max_steps=args.max_steps,internal_batch_size=args.internal_batch_size,
                   atol=args.atol,rtol=args.rtol,map_atol=args.map_atol,map_rtol=args.map_rtol,
                   write_hypothetical=args.write_hypothetical,script_sha256=source_hashes,plan=plan,
                   inputs={it['species']:dict(prepared=it['bundle']['verified_hashes'],original=it['sources']) for it in items})
    print(json.dumps(dict(target='per-species PER-ORGAN arithmetic mean of raw stage activities',n_targets=len(plan),
                          plan=plan,estimated_array_GiB=sum(p['estimated_array_bytes'] for p in plan)/1024**3),indent=2),flush=True)
    if args.check_only:
        print('CHECK_ONLY_OK: no attributions and no new files.',flush=True)
        return
    output=(args.output_dir or HERE/'results'/('organ_mean_IG_'+args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    require(not output.is_relative_to(args.inputs_root),'Output must not be inside prepared input sources')
    if args.resume:
        require(output.is_dir() and read_json(output/'batch_manifest.json')['signature']==signature,'Resume code/configuration/input fingerprint differs')
    else:
        require(not output.exists(),f'Refusing to overwrite output: {output}')
        existing_parent=output.parent
        while not existing_parent.exists():existing_parent=existing_parent.parent
        estimated=sum(p['estimated_array_bytes'] for p in plan)
        require(shutil.disk_usage(existing_parent).free>estimated*1.05,
                f'Insufficient free space for estimated {estimated/1024**3:.2f} GiB of attribution arrays')
        output.mkdir(parents=True,exist_ok=False)
    # Linux-only advisory lock: do not permit concurrent writers to the same run.
    import fcntl
    lock=(output/'.writer.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.resume:
        (output/'source').mkdir()
        for name in source_hashes:shutil.copyfile(HERE/name,output/'source'/name)
        save_json(output/'batch_manifest.json',dict(signature=signature,started_utc=now(),host=socket.gethostname(),
                  torch_version=str(torch.__version__),captum_version=captum.__version__,numpy_version=np.__version__,
                  baseline='single positional mean one-hot of each species prepared 128 TRAIN references',
                  orientation='already strand-oriented, TSS offset 5000; no new reverse-complement step',
                  no_training_or_source_modification=True))
    state=dict(status='running',pid=os.getpid(),host=socket.gethostname(),updated_utc=now(),output=str(output),organs=[])
    save_json(output/'status.json',state)
    try:
        for item in items:
            species=item['species'];spfolder=output/species
            prepare_species_output(spfolder,item,item['selected'],args.resume and spfolder.exists())
            for organ,rows in item['groups'].items():
                result=run_organ(item,organ,rows,item['selected'],spfolder/organ,args)
                state['organs'].append(result)
                state.update(updated_utc=now(),completed_targets=len(state['organs']),total_targets=len(plan))
                save_json(output/'status.json',state)
            item['model'].cpu();gc.collect()
            if args.device!='cpu':torch.cuda.empty_cache()
        for item in items:
            require(all(sha(record['path'])==record['sha256'] for record in item['sources'].values()),
                    f'Original sources changed during attribution: {item["species"]}')
            bundle=item['bundle']
            for name,digest in bundle['verified_hashes'].items():
                path=(args.redo_root/'datasets'/item['species']/'target_columns.tsv') if name=='target_columns.tsv' else (
                    args.inputs_root/item['species']/name)
                require(sha(path)==digest,f'Prepared input changed during attribution: {item["species"]}/{name}')
        require(all(sha(HERE/name)==digest for name,digest in source_hashes.items()),'Runner/module code changed during computation')
        state.update(status='complete' if args.mode=='full' and not args.max_promoters else ('pilot_complete' if args.mode=='pilot' else 'subset_complete'),
                     completed_utc=now(),all_targets_passed=True)
        save_json(output/'status.json',state)
        print('COMPLETE',json.dumps(state),flush=True)
    except BaseException as exc:
        state.update(status='failed_or_interrupted',error=f'{type(exc).__name__}: {exc}',updated_utc=now(),traceback=traceback.format_exc())
        save_json(output/'status.json',state)
        raise
    finally:
        lock.close()


if __name__=='__main__':
    main()
