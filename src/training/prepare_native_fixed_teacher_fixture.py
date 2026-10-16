#!/usr/bin/env python3
"""CPU-only actual singleton native teacher corruptions, immutable retained frames."""
import argparse,hashlib,json,os,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=128);p.add_argument('--seed',type=int,default=20261012)
    args=p.parse_args();source=args.source.resolve();out=args.out.resolve()
    os.chdir(source);sys.path.insert(0,str(source))
    import torch
    from omegaconf import OmegaConf
    from src.data.pdb_na_dataset_base import PDBNABaseDataset
    from src.data.interpolant import Interpolant
    from ensemble_mean_core import seed_member
    from batched_ensemble_core import digest_tensor
    torch.set_num_threads(2);device=torch.device('cpu')
    cfg=OmegaConf.load(source/'configs/config.yaml');cfg.data_cfg.csv_path=str(source/'metadata/pilot_ensemble.csv')
    cfg.data_cfg.filtering.max_len=800
    data=PDBNABaseDataset(cfg.data_cfg,is_training=True)
    index=next(i for i in range(len(data)) if str(data.csv.iloc[i]['pdb_name'])==args.case)
    single={k:torch.as_tensor(v).unsqueeze(0) for k,v in data[index].items()}
    single={k:v.float() if v.is_floating_point() else v for k,v in single.items()}
    assert all(bool(torch.isfinite(v).all()) for v in single.values() if v.is_floating_point())
    assert bool((single['res_mask']==1).all())
    interpolant=Interpolant(cfg.interpolant);interpolant.set_device(device);entries=[]
    for logical in range(args.count):
        seed_member(args.seed+10000019+logical*1009,device)
        noisy=interpolant.corrupt_batch(single)
        frames={key:noisy[key].detach().clone() for key in ['t','trans_t','rotmats_t']}
        assert all(bool(torch.isfinite(v).all()) for v in frames.values())
        entries.append({'logical_id':logical,'frames':{k:v.tolist() for k,v in frames.items()},
                        'raw_tensor_sha256':{k:digest_tensor(v) for k,v in frames.items()}})
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    result={'seed':args.seed,'role':'Frozen singleton native teacher corruptions; no inverse-model activations',
            'cases':{args.case:{'length':int(single['res_mask'].shape[1]),'entries':entries}},
            'generation':{'device':'CPU','GPU_allocation':False,'CPU_threads':2,'torch_version':torch.__version__,
                          'case':args.case,'count':args.count,'all_input_float_fields_finite':True,'residue_masks_all_valid':True,
                          'native_source_sha256':{name:sha(source/name) for name in ['src/data/pdb_na_dataset_base.py','src/data/interpolant.py','src/data/so3_utils.py','src/data/rigid_utils.py','configs/config.yaml','metadata/pilot_ensemble.csv']},
                          'native_data_sha256':{name:sha(source/name) for name in ['Feature/train_pkl/'+args.case+'/'+args.case+'.pkl','pkls/'+args.case+'_base/'+args.case+'_feat.pkl']},
                          'generator_sha256':sha(Path(__file__).resolve())}}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'fixture_case':args.case,'length':single['res_mask'].shape[1],'entries':len(entries),'GPU_allocation':False}),flush=True)

if __name__=='__main__':main()
