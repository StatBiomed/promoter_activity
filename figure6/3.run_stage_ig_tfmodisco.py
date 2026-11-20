#!/usr/bin/env python3
"""Run independent TF-MoDISco discovery on each Mouse Liver Top1500 stage IG.

Defaults: all 14 stages, all quality-passed canonical promoters, full 10 kb,
4 CPU threads, <=10000 seqlets PER SIGN. No GPU, retraining, new IG, installation,
or TF-name annotation. --check-only is read-only; --mode pilot is explicitly a
subset and never chains into a full run. Existing outputs are never overwritten.
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

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
DEFAULT_IG=HERE/'2.IG_results/stage_activity_IG_full_20261105T031547018838Z'
VERSION='mouse_liver_top1500_stage_ig_tfmodisco_v1'
CODE_FILES=[Path(__file__).name,'stage_tfmodisco_inputs.py','stage_tfmodisco_audit.py']
PARAMETERS=dict(sliding_window_size=20,flank_size=5,trim_to_window_size=30,
    initial_flank_to_add=10,final_flank_to_add=0,target_seqlet_fdr=.05,
    n_leiden_runs=2,min_metacluster_size=100,final_min_cluster_size=20,
    nearest_neighbors_to_compute=500,subcluster_perplexity=50,
    merging_max_seqlets_subsample=300)


def require(ok,message):
    if not ok:raise ValueError(message)


def now():return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path,data):
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
    for name in ['modisco','modisco-lite','numpy','scipy','numba','scikit-learn','leidenalg','igraph','h5py','hdf5plugin']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    version=versions.get('modisco',versions.get('modisco-lite','0'))
    require(int(version.split('.')[0])>=2,'Requires TF-MoDISco >=2 / modiscolite API; use the existing PY310 environment')
    signature=inspect.signature(tfmodisco.TFMoDISco)
    require((set(PARAMETERS)|{'one_hot','hypothetical_contribs','max_seqlets_per_metacluster'}) <= set(signature.parameters),
            'Installed TF-MoDISco API is incompatible')
    package=Path(modiscolite.__file__).resolve().parent
    return dict(versions=versions,api_signature=str(signature),
                source_sha256={p.name:sha(p) for p in sorted(package.glob('*.py'))})


def verify_prepared(folder):
    import csv
    import numpy as np
    manifest=read_json(folder/'preparation_manifest.json')
    run=read_json(folder/'run_manifest.json')
    signature=read_json(folder.parent/'batch_manifest.json')['signature']
    expected=dict(code_sha256=signature['code_sha256'],mode=signature['mode'],seed=signature['seed'],
        effective_max_seqlets=signature['effective_max_seqlets'],parameters=signature['parameters'],
        stage=manifest['stage'],target_column=manifest['target_column'])
    require(all(run.get(k)==v for k,v in expected.items()),'Prepared stage configuration differs from batch signature')
    matches=[p for p in signature['tasks'] if p['stage']==manifest['stage']]
    require(len(matches)==1 and matches[0]['stage_label'].replace(' ','')==folder.name and
            matches[0]['target_column']==manifest['target_column'],'Prepared stage identity mismatch')
    require(run['preparation_manifest_sha256']==sha(folder/'preparation_manifest.json'),'Preparation manifest changed')
    for name,record in manifest['source_files'].items():
        require(sha(record['path'])==record['sha256'],f'Original IG input changed: {name}')
    resources=folder/'resources'
    for name,digest in manifest['generated_files'].items():
        require(Path(name).name==name and sha(resources/name)==digest,f'Prepared input changed: {name}')
    x=np.load(resources/'onehot.npy',mmap_mode='r',allow_pickle=False)
    h=np.load(resources/'hypothetical_attributions.npy',mmap_mode='r',allow_pickle=False)
    require(list(x.shape)==manifest['input_shape'] and x.shape==h.shape and x.shape[1]==4,'Wrong prepared input shape')
    for start in range(0,len(x),32):
        a=x[start:start+32]
        require(np.all((a==0)|(a==1)) and np.all(a.sum(1)==1),'Noncanonical prepared sequence')
        require(np.isfinite(h[start:start+32]).all(),'Nonfinite hypothetical scores')
    with (resources/'modisco_input_index.tsv').open(newline='') as f:index=list(csv.DictReader(f,delimiter='\t'))
    require(len(index)==len(x) and [int(r['modisco_row_0based']) for r in index]==list(range(len(x))),
            'Prepared sequence/index row order mismatch')
    return manifest


def worker(folder,attempt,threads):
    """Isolated CPU worker releases memory after each individual stage."""
    import fcntl
    import numpy as np
    import numba
    from modiscolite import tfmodisco,io
    from stage_tfmodisco_audit import audit_h5
    folder=folder.resolve(strict=True);attempt=attempt.resolve(strict=True)
    require(attempt.parent==folder/'attempts','Worker attempt is outside selected stage')
    lock=(folder/'.worker.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(status='running',pid=os.getpid(),host=socket.gethostname(),started_utc=now())
    save_json(attempt/'worker_status.json',status)
    try:
        m=verify_prepared(folder);run=read_json(folder/'run_manifest.json')
        require(all(sha(HERE/name)==digest for name,digest in run['code_sha256'].items()),'Worker code changed')
        env=software();numba.set_num_threads(threads);np.random.seed(run['seed'])
        require(env==read_json(folder.parent/'batch_manifest.json')['signature']['software'],'Runtime differs from batch')
        x=np.ascontiguousarray(np.load(folder/'resources/onehot.npy',mmap_mode='r').transpose(0,2,1))
        h=np.ascontiguousarray(np.load(folder/'resources/hypothetical_attributions.npy',mmap_mode='r').transpose(0,2,1))
        save_json(attempt/'software_and_parameters.json',dict(**env,stage=m['stage'],target=m['target'],
            parameters=PARAMETERS,max_seqlets_per_sign=run['effective_max_seqlets'],seed=run['seed'],
            threads=threads,input_shape_NL4=list(x.shape),mode=run['mode']))
        print('TFMODISCO_START',m['stage'],list(x.shape),flush=True)
        pos,neg=tfmodisco.TFMoDISco(one_hot=x,hypothetical_contribs=h,
            max_seqlets_per_metacluster=run['effective_max_seqlets'],verbose=True,**PARAMETERS)
        temporary=attempt/'modisco_results.partial.h5'
        io.save_hdf5(temporary,pos,neg,window_size=m['window'])
        del pos,neg,x,h
        audit=audit_h5(temporary,folder,attempt,m)
        require(audit['status']=='passed','HDF5 integrity/coordinate audit failed')
        for name,record in m['source_files'].items():
            require(sha(record['path'])==record['sha256'],f'Original IG input changed during discovery: {name}')
        os.replace(temporary,attempt/'modisco_results.h5')
        require(all(sha(HERE/name)==digest for name,digest in run['code_sha256'].items()),'Worker code changed during run')
        status.update(status='complete',finished_utc=now(),stage=m['stage'],audit=audit,
            output_sha256={p.name:sha(p) for p in sorted(attempt.iterdir())
                if p.is_file() and p.name not in ['worker_status.json','tfmodisco.log']})
        save_json(attempt/'worker_status.json',status)
        print('TFMODISCO_COMPLETE',m['stage'],json.dumps(audit),flush=True)
    except BaseException as exc:
        status.update(status='failed',updated_utc=now(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        save_json(attempt/'worker_status.json',status);raise
    finally:lock.close()


def run_child(folder,args,status):
    pid=status.get('worker_pid')
    if pid and (Path('/proc')/str(pid)).is_dir():
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        require(not('--worker' in cmd and str(folder) in cmd),'Previous worker is still running')
    attempts=folder/'attempts';attempts.mkdir(exist_ok=True)
    nums=[int(p.name.split('_')[-1]) for p in attempts.iterdir() if p.is_dir() and p.name.startswith('attempt_')]
    attempt=attempts/f'attempt_{max(nums+[0])+1:04d}';attempt.mkdir()
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',str(folder),
             '--attempt',str(attempt),'--threads',str(args.threads)]
    with (attempt/'tfmodisco.log').open('xb') as log:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status.update(status='running',worker_pid=child.pid,attempt=str(attempt),updated_utc=now())
        save_json(folder/'status.json',status)
        try:
            while child.poll() is None:
                status['updated_utc']=now();save_json(folder/'status.json',status);time.sleep(5)
        except BaseException:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
        require(child.returncode==0,f'TF-MoDISco exited {child.returncode}; inspect {attempt}/tfmodisco.log')
    result=read_json(attempt/'worker_status.json')
    require(result['status']=='complete','Worker did not complete output audit')
    for name,digest in result['output_sha256'].items():require(sha(attempt/name)==digest,f'Worker output changed: {name}')
    return dict(attempt=str(attempt),audit=result['audit'],output_sha256=result['output_sha256'])


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ig-dir',type=Path,default=DEFAULT_IG)
    p.add_argument('--stages',nargs='+',help='Stage labels such as E10.5 P0 9wpb (raw labels also accepted); default all 14')
    p.add_argument('--mode',choices=['full','pilot'],default='full')
    p.add_argument('--window',type=int,default=10000,help='TSS-centered window; default entire existing 10 kb IG input')
    p.add_argument('--max-seqlets',type=int,default=10000,help='Maximum seqlets per positive/negative metacluster; NOT promoter count')
    p.add_argument('--pilot-promoters',type=int,default=128)
    p.add_argument('--pilot-seqlets',type=int,default=500)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--chunk-size',type=int,default=32)
    p.add_argument('--seed',type=int,default=20261105)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--resume',action='store_true',help='Same options and output dir; completed stages skipped, failed stages get NEW attempt')
    p.add_argument('--check-only',action='store_true',help='Validate selected complete stage IG inputs; no preparation, clustering or output writes')
    p.add_argument('--prepare-only',action='store_true',help='Write converted inputs only; continue later with same options plus --resume')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--attempt',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    require(a.threads>0 and a.chunk_size>0 and a.pilot_promoters>0,'Counts must be positive')
    require(100<=a.window<=10000 and a.window%2==0,'Window must be even in [100,10000]')
    require(a.max_seqlets>=100 and a.pilot_seqlets>=100,'Seqlet limits must be >=100')
    require(0<=a.seed<2**32,'Invalid random seed')
    require(not(a.check_only and a.prepare_only),'Choose only one check/preparation mode')
    require(not a.resume or a.output_dir is not None,'--resume requires --output-dir')
    require(bool(a.worker)==bool(a.attempt),'Worker and attempt must be supplied together')
    return a


def main(argv=None):
    args=parse_args(argv);set_threads(args.threads)
    if args.worker:worker(args.worker,args.attempt,args.threads);return 0
    import fcntl
    from stage_tfmodisco_inputs import discover_tasks,validate_task,prepare_task
    ig=args.ig_dir.resolve(strict=True);plan=discover_tasks(ig,args.stages)
    environment=software();code={name:sha(HERE/name) for name in CODE_FILES}
    effective=min(args.max_seqlets,args.pilot_seqlets) if args.mode=='pilot' else args.max_seqlets
    tasks=[{k:p[k] for k in ['stage','stage_label','target_column','output_index_0based']} for p in plan]
    signature=dict(version=VERSION,ig_directory=str(ig),ig_batch_manifest_sha256=sha(ig/'batch_manifest.json'),
        code_sha256=code,mode=args.mode,tasks=tasks,window=args.window,pilot_promoters=args.pilot_promoters,
        effective_max_seqlets=effective,chunk_size=args.chunk_size,threads=args.threads,seed=args.seed,
        parameters=PARAMETERS,software=environment)
    print('PLAN',json.dumps(dict(mode=args.mode,stages=[p['stage_label'] for p in plan],window=args.window,
        max_seqlets_per_sign=effective,pilot_promoters=args.pilot_promoters if args.mode=='pilot' else None)),flush=True)
    if args.check_only:
        for p in plan:
            info=validate_task(ig,p['stage'],window=args.window,chunk_size=args.chunk_size)
            print('VALIDATED',p['stage_label'],json.dumps({k:info[k] for k in
                ['n_test_promoters','canonical_promoters','excluded_promoters','warnings']}),flush=True)
        print('CHECK_ONLY_PASSED: no output writes and no motif discovery.',flush=True);return 0
    output=(args.output_dir or HERE/'3.TFMoDISco_results'/('stage_IG_TFMoDISco_'+args.mode+'_'+
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))).resolve()
    require(not(output.is_relative_to(ig) or ig.is_relative_to(output)),'Outputs must be separate from original IG run')
    require(not(output.is_relative_to(HERE/'1.results') or output.is_relative_to(HERE/'2.IG_results')),
            'Do not write TF-MoDISco outputs into original selection or IG directories')
    if args.resume:
        require(read_json(output/'batch_manifest.json')['signature']==signature,'Resume code, input, runtime or configuration changed')
    else:
        require(not output.exists(),f'Refusing existing output directory: {output}')
        output.mkdir(parents=True,exist_ok=False)
    lock=(output/'.writer.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.resume:
        (output/'source').mkdir()
        for name in CODE_FILES:shutil.copyfile(HERE/name,output/'source'/name)
        save_json(output/'batch_manifest.json',dict(signature=signature,started_utc=now(),host=socket.gethostname(),
            target='One stage RAW activity at a time, NOT stage-minus-rest selection specificity',
            scope='Top1500 TEST promoters selected separately for each stage; input sets can overlap',
            comparison_caveat='Do not interpret motif-count differences across differently selected stage sets as a matched differential TF test',
            unknown_policy='Exclude whole promoters with unknown/padded bases in discovery window; export every exclusion',
            no_tf_annotation=True,no_model_inference_or_new_ig=True))
    batch=dict(status='running',pid=os.getpid(),host=socket.gethostname(),output=str(output),
        started_or_resumed_utc=now(),total_stages=len(plan),completed_stages=0,stages=[])
    save_json(output/'status.json',batch)
    try:
        for p in plan:
            stage=p['stage'];folder=output/p['stage_label'].replace(' ','')
            status=read_json(folder/'status.json') if (folder/'status.json').is_file() else {}
            if args.resume and status.get('status') in ['complete','pilot_complete']:
                verify_prepared(folder)
                for name,digest in status['output_sha256'].items():
                    require(sha(Path(status['attempt'])/name)==digest,'Completed stage output changed')
                print('SKIP_COMPLETE',stage,flush=True)
            else:
                try:
                    if not(folder/'preparation_manifest.json').is_file():
                        require(not folder.exists(),'Partial input initialization; choose a NEW output directory')
                        info=validate_task(ig,stage,window=args.window,chunk_size=args.chunk_size)
                        estimated=info['canonical_promoters']*args.window*32*1.1+100_000_000
                        require(shutil.disk_usage(output).free>estimated,'Insufficient free disk for stage preparation')
                        folder.mkdir()
                        m=prepare_task(info,folder,window=args.window,chunk_size=args.chunk_size,seed=args.seed,
                            pilot_count=args.pilot_promoters if args.mode=='pilot' else 0)
                        save_json(folder/'run_manifest.json',dict(code_sha256=code,mode=args.mode,seed=args.seed,
                            effective_max_seqlets=effective,parameters=PARAMETERS,stage=stage,target_column=p['target_column'],
                            preparation_manifest_sha256=sha(folder/'preparation_manifest.json')))
                        for warning in m['warnings']:print('INPUT_WARNING',stage,warning,flush=True)
                        del info
                    else:
                        require(args.resume,'Existing prepared stage requires --resume');m=verify_prepared(folder)
                    status.update(status='prepared',stage=stage,stage_label=p['stage_label'],target_column=p['target_column'],
                        discovery_promoters=m['input_shape'][0],mode=args.mode,updated_utc=now())
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
            batch['stages'].append({k:status[k] for k in ['stage','stage_label','status','discovery_promoters','attempt'] if k in status})
            batch.update(completed_stages=len(batch['stages']),updated_utc=now());save_json(output/'status.json',batch)
        require(all(sha(HERE/name)==digest for name,digest in code.items()),'Code changed during batch')
        require(sha(ig/'batch_manifest.json')==signature['ig_batch_manifest_sha256'],'Source IG batch manifest changed')
        batch.update(status='prepared' if args.prepare_only else ('complete' if args.mode=='full' else 'pilot_complete'),finished_utc=now())
        save_json(output/'status.json',batch);print('FINISHED',json.dumps(batch),flush=True)
    except BaseException as exc:
        batch.update(status='failed_or_interrupted',updated_utc=now(),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        save_json(output/'status.json',batch);raise
    finally:lock.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
