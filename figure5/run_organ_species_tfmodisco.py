#!/usr/bin/env python3
"""Independent TF-MoDISco for each completed species x organ mean-activity IG.

Default mode is a marked pilot (128 canonical promoters, <=500 seqlets/sign).
Use --mode full explicitly for all eligible promoters. A pilot NEVER chains
into a full run. No training, new IG, package installation or TF annotation.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
DEFAULT_IG = HERE/'results/organ_mean_IG_full_20261104T081335286937Z'
VERSION = 'organ_mean_ig_tfmodisco_v1'
CODE_FILES = ['run_organ_species_tfmodisco.py','organ_modisco_inputs.py','organ_modisco_audit.py']
PARAMETERS = dict(sliding_window_size=20, flank_size=5, trim_to_window_size=30,
                  initial_flank_to_add=10, final_flank_to_add=0,
                  target_seqlet_fdr=.05, n_leiden_runs=2,
                  min_metacluster_size=100, final_min_cluster_size=20,
                  nearest_neighbors_to_compute=500, subcluster_perplexity=50,
                  merging_max_seqlets_subsample=300)


def require(ok, message):
    if not ok: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def set_threads(count):
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:
        os.environ[key]=str(count)
    os.environ['MPLBACKEND']='Agg'


def software():
    import modiscolite
    from modiscolite import tfmodisco
    versions={}
    for package in ['modisco','modisco-lite','numpy','scipy','numba','scikit-learn','leidenalg','igraph','h5py','hdf5plugin']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    version=versions.get('modisco',versions.get('modisco-lite','0'))
    require(int(version.split('.')[0])>=2,'Requires the installed modern modiscolite >=2 API')
    require(versions.get('leidenalg')=='0.11.0','This workflow was verified with leidenalg 0.11.0')
    signature=inspect.signature(tfmodisco.TFMoDISco)
    require(set(PARAMETERS)<=set(signature.parameters),'Installed TFMoDISco API is incompatible')
    package=Path(modiscolite.__file__).resolve().parent
    return dict(versions=versions,api_signature=str(signature),
                package_source_sha256={str(p):sha(p) for p in sorted(package.glob('*.py'))})


def verify_prepared(folder):
    import csv
    import numpy as np
    manifest=read_json(folder/'preparation_manifest.json')
    run=read_json(folder/'run_manifest.json')
    signature=read_json(folder.parents[1]/'batch_manifest.json')['signature']
    expected=dict(code_sha256=signature['code_sha256'],mode=signature['mode'],seed=signature['seed'],
        effective_max_seqlets=signature['effective_max_seqlets'],parameters=signature['parameters'],
        species=manifest['species'],organ=manifest['organ'])
    require(all(run.get(k)==v for k,v in expected.items()),'Prepared task configuration differs from batch signature')
    require(dict(species=manifest['species'],organ=manifest['organ']) in signature['tasks']
            and folder.name==manifest['organ'] and folder.parent.name==manifest['species'],
            'Prepared target identity differs from directory/batch plan')
    require(run.get('preparation_manifest_sha256')==sha(folder/'preparation_manifest.json'),
            'Preparation manifest changed')
    for name,record in manifest['source_files'].items():
        require(sha(Path(record['path']))==record['sha256'],f'Source IG/prepared input changed: {name}')
    resources=folder/'resources'
    for name,digest in manifest['generated_files'].items():
        require(Path(name).name==name,'Unsafe prepared input filename')
        require(sha(resources/name)==digest,f'Prepared input changed: {name}')
    x=np.load(resources/'onehot.npy',mmap_mode='r',allow_pickle=False)
    h=np.load(resources/'hypothetical_attributions.npy',mmap_mode='r',allow_pickle=False)
    require(list(x.shape)==manifest['input_shape'] and x.shape==h.shape and x.shape[1]==4,
            'Prepared input layout differs; require N x 4 x window')
    for start in range(0,len(x),32):
        a=x[start:start+32]
        require(np.all((a==0)|(a==1)) and np.all(a.sum(axis=1)==1),'Discovery inputs must be fully canonical')
        require(np.isfinite(h[start:start+32]).all(),'Nonfinite hypothetical attribution')
    with (resources/'modisco_input_index.tsv').open(newline='') as f:indices=list(csv.DictReader(f,delimiter='\t'))
    require(len(indices)==len(x),'Prepared sequence/index count differs')
    return manifest


def worker(task_folder, attempt, threads):
    """One CPU worker, one target, one isolated attempt; no full-run chaining."""
    import numpy as np
    import numba
    import fcntl
    from modiscolite import tfmodisco,io
    from organ_modisco_audit import audit_h5
    folder=task_folder.resolve(strict=True);attempt=attempt.resolve(strict=True)
    require(attempt.parent==folder/'attempts','Worker attempt is outside the selected task')
    lock=(folder/'.worker.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(status='running',pid=os.getpid(),host=socket.gethostname(),started_utc=now())
    save_json(attempt/'worker_status.json',status)
    try:
        m=verify_prepared(folder);run=read_json(folder/'run_manifest.json')
        code=run['code_sha256']
        require(all(sha(HERE/name)==digest for name,digest in code.items()),'Worker source revision differs')
        environment=software();numba.set_num_threads(threads)
        np.random.seed(run['seed'])
        x=np.ascontiguousarray(np.load(folder/'resources/onehot.npy',mmap_mode='r').transpose(0,2,1))
        h=np.ascontiguousarray(np.load(folder/'resources/hypothetical_attributions.npy',mmap_mode='r').transpose(0,2,1))
        provenance=dict(**environment,parameters=PARAMETERS,max_seqlets_per_sign=run['effective_max_seqlets'],
                        seed=run['seed'],threads=threads,input_shape_NL4=list(x.shape),mode=run['mode'])
        save_json(attempt/'software_and_parameters.json',provenance)
        print('TFMODISCO_START',json.dumps(dict(species=m['species'],organ=m['organ'],**provenance)),flush=True)
        pos,neg=tfmodisco.TFMoDISco(one_hot=x,hypothetical_contribs=h,
            max_seqlets_per_metacluster=run['effective_max_seqlets'],verbose=True,**PARAMETERS)
        temporary=attempt/'modisco_results.partial.h5'
        io.save_hdf5(temporary,pos,neg,window_size=m['window'])
        del pos,neg,x,h
        result=audit_h5(temporary,folder,attempt,m)
        require(result.get('status')=='passed','Motif integrity audit failed')
        os.replace(temporary,attempt/'modisco_results.h5')
        require(all(sha(HERE/name)==digest for name,digest in code.items()),'Worker code changed during discovery')
        status.update(status='complete',finished_utc=now(),audit=result,
            output_sha256={p.name:sha(p) for p in sorted(attempt.iterdir())
                           if p.is_file() and p.name not in ['worker_status.json','tfmodisco.log']})
        save_json(attempt/'worker_status.json',status)
        print('TFMODISCO_COMPLETE',json.dumps(status),flush=True)
    except BaseException as exc:
        status.update(status='failed',updated_utc=now(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        save_json(attempt/'worker_status.json',status)
        raise
    finally:lock.close()


def run_child(folder,args,status):
    previous_pid=status.get('worker_pid')
    if previous_pid and (Path('/proc')/str(previous_pid)).is_dir():
        cmd=(Path('/proc')/str(previous_pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        require(not ('--worker' in cmd and str(folder) in cmd),'A previous worker is still running; do not launch another')
    attempts=folder/'attempts';attempts.mkdir(exist_ok=True)
    number=max([int(p.name.split('_')[-1]) for p in attempts.iterdir() if p.is_dir() and p.name.startswith('attempt_')]+[0])+1
    attempt=attempts/f'attempt_{number:04d}';attempt.mkdir()
    log=attempt/'tfmodisco.log'
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',str(folder),
             '--attempt',str(attempt),'--threads',str(args.threads)]
    with log.open('xb') as stream:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
        status.update(status='running',worker_pid=child.pid,attempt=str(attempt),log=str(log),updated_utc=now())
        save_json(folder/'status.json',status)
        try:
            while child.poll() is None:
                status['updated_utc']=now();save_json(folder/'status.json',status);time.sleep(5)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
        require(child.returncode==0,f'TFMoDISco worker exited {child.returncode}; inspect {log}')
    result=read_json(attempt/'worker_status.json')
    require(result['status']=='complete','Worker did not finish its audit')
    for name,digest in result['output_sha256'].items():require(sha(attempt/name)==digest,f'Worker output changed: {name}')
    return dict(attempt=str(attempt),audit=result['audit'],output_sha256=result['output_sha256'])


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ig-dir',type=Path,default=DEFAULT_IG)
    p.add_argument('--species',nargs='+',choices=['Human','Mouse','Rat','Rabbit','Macaque','Chicken'])
    p.add_argument('--organs',nargs='+',help='Exact case-sensitive names; default every available organ')
    p.add_argument('--mode',choices=['pilot','full'],default='pilot')
    p.add_argument('--pilot-promoters',type=int,default=128)
    p.add_argument('--pilot-seqlets',type=int,default=500)
    p.add_argument('--max-seqlets',type=int,default=10000,help='Maximum per positive/negative metacluster, not promoter count')
    p.add_argument('--window',type=int,default=10000,help='Centered sequence window; no window cropping by default')
    p.add_argument('--chunk-size',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--resume',action='store_true',help='Same code/configuration; skip completed targets, retry failed workers in a NEW attempt')
    p.add_argument('--check-only',action='store_true',help='Deep read-only validation; no preparation/discovery/output writes')
    p.add_argument('--prepare-only',action='store_true',help='Prepare selected inputs only; later --resume uses the same scope/mode')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--attempt',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    require(a.threads>0 and a.chunk_size>0 and a.pilot_promoters>0,'Counts must be positive')
    require(100<=a.window<=10000 and a.window%2==0,'Window must be even, 100..10000')
    require(a.max_seqlets>=100 and a.pilot_seqlets>=100,'Seqlet limits must be >=100')
    require(0<=a.seed<2**32,'Invalid random seed')
    require(not(a.check_only and a.prepare_only),'Choose only one check/preparation mode')
    require(not a.resume or a.output_dir is not None,'--resume needs --output-dir')
    require(bool(a.worker)==bool(a.attempt),'Worker and attempt must be provided together')
    return a


def main(argv=None):
    args=parse_args(argv);set_threads(args.threads)
    if args.worker:
        worker(args.worker,args.attempt,args.threads);return
    import fcntl
    from organ_modisco_inputs import discover_tasks,validate_task,prepare_task
    ig=args.ig_dir.resolve(strict=True)
    plan=discover_tasks(ig,args.species,args.organs)
    environment=software()
    code={name:sha(HERE/name) for name in CODE_FILES}
    effective=min(args.max_seqlets,args.pilot_seqlets) if args.mode=='pilot' else args.max_seqlets
    signature=dict(version=VERSION,ig_directory=str(ig),ig_batch_manifest_sha256=sha(ig/'batch_manifest.json'),
        code_sha256=code,mode=args.mode,species=args.species,organs=args.organs,
        tasks=[dict(species=p['species'],organ=p['organ']) for p in plan],window=args.window,
        pilot_promoters=args.pilot_promoters,effective_max_seqlets=effective,
        chunk_size=args.chunk_size,threads=args.threads,seed=args.seed,parameters=PARAMETERS,
        software=environment)
    print('PLAN',json.dumps(dict(n_targets=len(plan),mode=args.mode,window=args.window,
        max_seqlets_per_sign=effective,pilot_promoters=args.pilot_promoters if args.mode=='pilot' else None,
        targets=signature['tasks'])),flush=True)
    if args.check_only:
        for p in plan:
            info=validate_task(ig,p['species'],p['organ'],window=args.window,chunk_size=args.chunk_size)
            print('VALIDATED',p['species'],p['organ'],json.dumps({k:info[k] for k in ['n_test_promoters','canonical_promoters','excluded_promoters','excluded_fraction','warnings']}),flush=True)
        print('CHECK_ONLY_PASSED: no output writes and no motif discovery.',flush=True);return
    output=(args.output_dir or HERE/'tfmodisco_results'/('organ_mean_TFMoDISco_'+args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    require(not(output.is_relative_to(ig) or ig.is_relative_to(output)),'Output must be separate from source IG run')
    require(not(output.is_relative_to(HERE/'results')),'Keep TFMoDISco outputs separate from IG results')
    if args.resume:
        require(read_json(output/'batch_manifest.json')['signature']==signature,'Resume code/input/software/configuration fingerprint differs')
    else:
        require(not output.exists(),f'Refusing existing output: {output}')
        output.mkdir(parents=True,exist_ok=False)
    lock=(output/'.writer.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.resume:
        (output/'source').mkdir()
        for name in CODE_FILES:shutil.copyfile(HERE/name,output/'source'/name)
        save_json(output/'batch_manifest.json',dict(signature=signature,started_utc=now(),host=socket.gethostname(),
            target='Per-species PER-ORGAN mean RAW activity over all recorded stages; not stage preference/contrast',
            unknown_policy='Exclude any sequence containing unknown bases in the discovery window; report every exclusion',
            no_tf_annotation=True,no_model_inference_or_new_ig=True))
    batch=dict(status='running',pid=os.getpid(),host=socket.gethostname(),output=str(output),started_or_resumed_utc=now(),
               total_targets=len(plan),completed_targets=0,targets=[])
    save_json(output/'status.json',batch)
    try:
        for p in plan:
            species,organ=p['species'],p['organ'];folder=output/species/organ
            status=read_json(folder/'status.json') if (folder/'status.json').is_file() else {}
            if args.resume and status.get('status') in ['complete','pilot_complete']:
                verify_prepared(folder)
                for name,digest in status['output_sha256'].items():require(sha(Path(status['attempt'])/name)==digest,'Completed target output changed')
                print('SKIP_COMPLETE',species,organ,flush=True)
            else:
                try:
                    if not(folder/'preparation_manifest.json').is_file():
                        require(not folder.exists(),'Partial input initialization; choose a new output directory')
                        info=validate_task(ig,species,organ,window=args.window,chunk_size=args.chunk_size)
                        n=info['n_test_promoters']
                        parent=output;estimated=max(n,1)*args.window*32*1.10+100_000_000
                        require(shutil.disk_usage(parent).free>estimated,'Insufficient free disk for next target inputs')
                        folder.mkdir(parents=True,exist_ok=False)
                        m=prepare_task(info,folder,window=args.window,chunk_size=args.chunk_size,seed=args.seed,
                                       pilot_count=args.pilot_promoters if args.mode=='pilot' else 0)
                        save_json(folder/'run_manifest.json',dict(code_sha256=code,mode=args.mode,seed=args.seed,
                            effective_max_seqlets=effective,parameters=PARAMETERS,species=species,organ=organ,
                            preparation_manifest_sha256=sha(folder/'preparation_manifest.json')))
                        for warning in m['warnings']:print('INPUT_WARNING',species,organ,warning,flush=True)
                    else:
                        require(args.resume,'Existing prepared task requires --resume')
                        m=verify_prepared(folder)
                    status.update(status='prepared',species=species,organ=organ,updated_utc=now(),
                        discovery_promoters=m['input_shape'][0],source_scope='full TEST; canonical-window subset for motifs',mode=args.mode)
                    save_json(folder/'status.json',status)
                    if not args.prepare_only:
                        result=run_child(folder,args,status)
                        status.update(**result,status='complete' if args.mode=='full' else 'pilot_complete',finished_utc=now())
                        save_json(folder/'status.json',status)
                except BaseException as exc:
                    if folder.is_dir():
                        status.update(status='failed',error=f'{type(exc).__name__}: {exc}',updated_utc=now())
                        save_json(folder/'status.json',status)
                    raise
            batch['targets'].append(dict(species=species,organ=organ,**{k:status[k] for k in ['status','discovery_promoters','attempt'] if k in status}))
            batch.update(completed_targets=len(batch['targets']),updated_utc=now());save_json(output/'status.json',batch)
        require(all(sha(HERE/name)==digest for name,digest in code.items()),'Code changed during TFMoDISco')
        require(sha(ig/'batch_manifest.json')==signature['ig_batch_manifest_sha256'],'Source IG manifest changed')
        batch.update(status='prepared' if args.prepare_only else ('complete' if args.mode=='full' else 'pilot_complete'),finished_utc=now())
        save_json(output/'status.json',batch);print('FINISHED',json.dumps(batch),flush=True)
    except BaseException as exc:
        batch.update(status='failed_or_interrupted',updated_utc=now(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        save_json(output/'status.json',batch);raise
    finally:lock.close()


if __name__=='__main__':main()
