"""Verify sealed component-cost records and recompute stage summaries."""
import json,hashlib,subprocess,zipfile,sys
from pathlib import Path
import numpy as np
tag=sys.argv[1] if len(sys.argv)>1 else 'operator_timing'
assert tag in ('operator_timing','operator_cached_timing')
r=Path('result/preparation/gsd_hypotheses_20261105');q=json.loads((r/(tag+'_receipt.json')).read_text());p=r/(tag+'.zip')
if not p.exists():subprocess.run(['colab','download','-s','tta-gsd-correction-transfer-a100-20261105-r8',q['package'],str(p)],check=True)
assert p.stat().st_size==q['identity']['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==q['identity']['sha256']
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None;rows=json.loads(z.read('records.json'));assert len(rows)==(40 if tag=='operator_timing' else 16)
 summary={}
 for stage in sorted({v['stage'] for v in rows}):
  subset=[v for v in rows if v['stage']==stage];assert len(subset)==8
  for v in subset:assert abs(np.median(v['seconds'])-v['median_seconds'])<1e-12 and min(v['seconds'])>0
  summary[stage]=dict(cases=len(subset),median_across_case_medians_ms=float(np.median([v['median_seconds'] for v in subset])*1000),range_case_medians_ms=[min(v['median_seconds'] for v in subset)*1000,max(v['median_seconds'] for v in subset)*1000],max_additional_peak_bytes=max(v['additional_peak_bytes'] for v in subset))
 result=dict(archive_integrity='passed',stages=summary,scope=json.loads(z.read('complete.json')))
 (r/(tag+'_summary.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
