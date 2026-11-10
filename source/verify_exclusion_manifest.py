"""Verify the fixed Adult exclusion list against both original returned archives."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

def verify(manifest_path,archives):
    frozen=json.loads(Path(manifest_path).read_text());seen=set()
    for path in map(Path,archives):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen['previous_archives'][path.name]
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.startswith('units/adult_numeric/'):
                    continue
                if name.endswith('_rows.csv'):
                    seen.update(pd.read_csv(io.BytesIO(z.read(name)),usecols=['id']).id.astype(str))
                elif name.endswith('source_split.npz'):
                    with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as arrays:
                        for key in arrays.files:
                            if key.endswith('_ids'):
                                seen.update(arrays[key].astype(str))
    assert sorted(seen)==frozen['adult_excluded_deployment_ids']
    return {'status':'passed','historical_archives':frozen['previous_archives'],'adult_excluded_identifiers':len(seen),'exact_identifier_match':True,'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('archives',nargs=2);a=p.parse_args();print(json.dumps(verify(a.manifest,a.archives),indent=2))
