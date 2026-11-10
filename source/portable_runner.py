"""Cross-platform CPU launcher for the byte-preserved scientific engine.

The adapter permits the CPU wheel suffix (+cpu) of the same public PyTorch release.
Every original freeze/configuration/checksum check is retained. Full installed
version metadata is recorded by the engine. Process count is an operational setting.
"""
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import zipfile
import engine

def cross_platform_freeze(path,cfg,engine_hash):
    status=engine.verify_archive(path)
    assert status['stage']=='development' and status['status']=='complete'
    with zipfile.ZipFile(path) as archive:
        frozen=json.loads(archive.read('freeze_manifest.json'))
        assert frozen['all_technical_gates_passed']
        assert frozen['engine_sha256']==engine_hash
        assert frozen['core_sha256']==engine.digest(engine.Config().core())==engine.digest(cfg.core())
        assert frozen['study_config_sha256']==engine.digest(dataclasses.asdict(engine.Config.study()))
        observed=importlib.metadata.version('torch')
        assert observed.split('+')[0]==frozen['torch_version'].split('+')[0]=='2.8.0'
        assert frozen['data_manifest_sha256']==engine.revision_manifest()[1]
        assert sorted(frozen['completed_units'])==sorted(f'{d}/{m}/seed_11' for d in engine.DATASETS for m in engine.MODELS)
        assert json.loads(archive.read('config.json'))==engine.plain(dataclasses.asdict(engine.Config()))
    import torch
    assert not torch.cuda.is_available(),'Select the CPU runtime for the frozen CPU study.'
    return {'source_zip_name':Path(path).name,'source_zip_sha256':engine.file_hash(path),'freeze':frozen,
            'runtime_adapter':{'file':'portable_runner.py','policy':'same public PyTorch 2.8.0 release, including CPU wheel build suffix; CPU execution',
                               'installed_torch_metadata_version':observed,'torch_runtime_version':torch.__version__,
                               'operational_workers':cfg.workers,'scientific_engine_unchanged':True}}

def main():
    stage=sys.argv[1] if len(sys.argv)>1 else 'confirmation'
    cfg=engine.Config.study() if stage=='confirmation' else engine.Config()
    if stage=='confirmation':
        cfg=dataclasses.replace(cfg,workers=max(1,min(4,(os.cpu_count() or 2)//2)))
        engine.validate_freeze=cross_platform_freeze
    archive,status=engine.run(cfg,base=Path(__file__).parent/'runs',development_zip=sys.argv[2] if len(sys.argv)>2 else None)
    (Path(__file__).parent/'last_result.json').write_text(json.dumps({'archive':str(archive.resolve()),'status':status['status'],'identity':status['identity']},indent=2))
    return 0 if status['status']=='complete' else 2

if __name__=='__main__':
    raise SystemExit(main())
