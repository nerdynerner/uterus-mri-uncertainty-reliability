"""Check fixed partitions and recompute primary policies and bootstrap intervals."""
import hashlib,json,statistics,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/"src"))
from study_manifest import make_splits,validate_splits
from analyse_policy import analyse,freeze
read=lambda n:json.loads((R/n).read_text())
cases=read("config/manifest.json")["cases"];splits=read("config/splits.json")
parts=validate_splits(cases,splits)
assert len(cases)==80
assert {k:len(v) for k,v in parts.items()}==dict(train=27,validation=8,calibration=8,internal_test=11,external_test=26)
assert make_splits(cases)["partitions"]==parts
assert hashlib.sha256((R/"config/manifest.json").read_bytes()).hexdigest()==splits["manifest_sha256"]
policy=read("results/frozen_review_policy.json");cal=read("results/calibration_scores.json")
assert freeze(cal,policy["provenance"])["thresholds"]==policy["thresholds"]
final=read("results/final_analysis.json")
def close(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b)
  for k in a:close(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):close(x,y)
 elif isinstance(a,float):assert abs(a-b)<1e-12,(a,b)
 else:assert a==b,(a,b)
for part in ("internal_test","external_test"):
 rows=read("results/"+part+"_scores.json")
 assert len(rows)==3*len(parts[part])
 for m in ("deterministic","ensemble","dropout"):
  assert {r["participant"] for r in rows if r["method"]==m}==set(parts[part])
 result=analyse(rows,policy,part,cal)
 for k,v in result.items():close(v,final[part][k])
training=read("results/training_summary.json")
assert [r["seed"] for r in training]==[61901,61902,61903,61904,61905,61911]
assert [r["completed_epochs"] for r in training]==[35,34,30,40,33,60]
assert [r["selection"]["path"] for r in training]==[f"best_epoch_{i:03}.pt" for i in [20,19,15,25,18,45]]
print("PASS: 80 cases; fixed partitions; six model records; frozen thresholds; primary policies, risk-coverage and bootstrap results.")
