#!/usr/bin/env python3
"""CPU-only temporary fixture generation and full native batch2/T2 proof."""
import argparse,base64,csv,hashlib,io,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='5zwm_B');args=p.parse_args()
    root=Path(__file__).resolve().parent;base=root/'batched_full_frozen_8d2c83840189'
    remote='/e/project1/e-aif-2026fl01-991/ribo_batched_full_training_20261012/batched_full_frozen_8d2c83840189'
    if args.case=='5zwm_B':
        audit=json.loads((root/'near177_feature_audit/near177_feature_summary.json').read_text())
        selected=next(d for d in audit['candidates'] if d['row']['pdb_name']==args.case and d['metadata_member'].endswith('/RNA3DB_train.csv'))
        feature=Path(selected['local_feature_copy']).read_bytes();assert hashlib.sha256(feature).hexdigest()==selected['feature_sha256']
        source_feature='Feature/train_pkl/5zwm_B/5zwm_B.pkl'
        original=(base/'RNAbpFlow_training/metadata/pilot_ensemble.csv').read_text()
        reader=csv.DictReader(io.StringIO(original));rows=list(reader);fields=reader.fieldnames
        rows.append({key:selected['row'].get(key,'') for key in fields})
        text=io.StringIO();writer=csv.DictWriter(text,fieldnames=fields);writer.writeheader();writer.writerows(rows);metadata=text.getvalue()
        (root/'native_fixed_pilot_ensemble.csv').write_text(metadata)
        (root/'native_fixed_feature.pkl').write_bytes(feature)
        source_identity=selected
        extra=json.loads((root/'near177_feature_audit/5zwm_B_native_extra_feature_manifest.json').read_text())
        for record in extra:assert hashlib.sha256(Path(record['local_path']).read_bytes()).hexdigest()==record['sha256']
        source_identity['native_companion_feature_assets']=extra
    else:
        assert args.case=='1l8v_B';metadata=(base/'RNAbpFlow_training/metadata/pilot_ensemble.csv').read_text();feature=None;source_feature=None;source_identity={'fallback':'Previously qualified157nt1l8v_B'};extra=[]
        (root/'native_fixed_pilot_ensemble.csv').write_text(metadata)
    files=['batched_full_trajectory_driver.py','batched_ensemble_core.py','full_trajectory_core.py',
           'differentiable_rotation.py','ensemble_mean_core.py','ensemble_training_driver.py']
    payload={name:(base/name).read_text() for name in files}
    for name in ['native_fixed_launcher_driver.py','prepare_native_fixed_teacher_fixture.py']:payload[name]=(root/name).read_text()
    encoded=base64.b64encode(feature).decode() if feature is not None else None
    extra_encoded={record['archive_member'].removeprefix('RNAbpFlow_training/'):base64.b64encode(Path(record['local_path']).read_bytes()).decode() for record in extra}
    code=f'''import os,sys,tempfile,pathlib,json,shutil,base64,gzip
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
with tempfile.TemporaryDirectory(prefix='ribo_native_fixed_cpu_') as temporary:
    native=temporary+'/RNAbpFlow_training';shutil.copytree({(remote+'/RNAbpFlow_training')!r},native)
    for name,data in {payload!r}.items():(pathlib.Path(temporary)/name).write_text(data)
    metadata_path=pathlib.Path(native+'/metadata/pilot_ensemble.csv');metadata_path.chmod(0o600);metadata_path.write_text({metadata!r})
    feature_path={source_feature!r};feature_base64={encoded!r}
    if feature_path is not None:
        destination=pathlib.Path(native)/feature_path;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(base64.b64decode(feature_base64))
    for name,data in {extra_encoded!r}.items():
        destination=pathlib.Path(native)/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(base64.b64decode(data))
    sys.path.insert(0,temporary)
    import prepare_native_fixed_teacher_fixture as preparer
    fixture=temporary+'/native_fixed_teacher_fixture.json'
    sys.argv=['prepare_native_fixed_teacher_fixture.py','--source',native,'--case',{args.case!r},'--count','128','--out',fixture]
    preparer.main()
    print('NATIVE_FIXED_FIXTURE_GZIP_BASE64:'+base64.b64encode(gzip.compress(pathlib.Path(fixture).read_bytes())).decode(),flush=True)
    import native_fixed_launcher_driver as launcher
    sys.argv=['native_fixed_launcher_driver.py','--source',native,
      '--checkpoint','/e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/pretrained_checkpoint/RNA3DB.ckpt',
      '--fixture',fixture,'--out',temporary+'/results','--device','cpu','--local-constructs','2','--microbatch','2',
      '--case',{args.case!r},'--ensemble-size','2','--grid-points','2','--cpu-proof']
    launcher.main()
    print('NATIVE_FIXED_CPU_PROOF_JSON:'+json.dumps(json.loads(pathlib.Path(temporary+'/results/cpu_batched_full_test.json').read_text()),allow_nan=False),flush=True)
'''
    command='env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /e/project1/e-aif-2026fl01-991/ribo_geometry_pilot_20261012/env/bin/python -u -'
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','jupiter',command],input=code,text=True,capture_output=True,timeout=300)
    (root/'native_fixed_cpu.stdout.txt').write_text(r.stdout);(root/'native_fixed_cpu.stderr.txt').write_text(r.stderr)
    import gzip
    for line in r.stdout.splitlines():
        if line.startswith('NATIVE_FIXED_FIXTURE_GZIP_BASE64:'):(root/'native_fixed_teacher_fixture.json').write_bytes(gzip.decompress(base64.b64decode(line.split(':',1)[1])))
        if line.startswith('NATIVE_FIXED_CPU_PROOF_JSON:'):
            proof=json.loads(line.split(':',1)[1]);proof['execution_source_sha256']={name:hashlib.sha256(text.encode()).hexdigest() for name,text in payload.items()}
            proof['local_archive_feature_provenance']=source_identity
            (root/'cpu_native_fixed_full_test.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'exit_code':r.returncode,'CPU_proof_exists':(root/'cpu_native_fixed_full_test.json').exists(),'case':args.case,'GPU_allocation':False}))
    print(r.stderr[-3500:])

if __name__=='__main__':main()
