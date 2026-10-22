"""Prepare an isolated computational workspace with fixed cohort manifests."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("workspace",type=Path);a=p.parse_args()
r=Path(__file__).resolve().parents[1];w=a.workspace.resolve();w.mkdir(parents=True,exist_ok=False)
(w/"outputs").mkdir();(w/"data/original_v2").mkdir(parents=True);(w/"tmp").mkdir()
for source,target in [("manifest.json","primary_manifest_v1_2026-10-21.json"),("splits.json","primary_splits_v1_2026-10-21.json"),("cohort_screening.json","cohort_screening.json")]:
 shutil.copyfile(r/"config"/source,w/"outputs"/target)
print(w)
