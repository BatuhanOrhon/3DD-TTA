"""CPU-only compact tables and boundary evidence from accepted immutable raw files."""
import csv,io,json,zipfile
from pathlib import Path
from collections import Counter
import numpy as np
R=Path(__file__).resolve().parent
m=json.loads((R/'main_local_recount.json').read_text());paths=json.loads((R/'stage_paths_local.json').read_text())
g=m['live0_control_gate'];fails=[v for v in g['cases'] if not v['passed']]
rows=[]
for arm,t in m['timing'].items():
 rows.append(dict(arm=arm,n=480,pm_correct=int(np.asarray(m['outcomes']['pm'][arm]).sum()),terminal_correct=int(np.asarray(m['outcomes']['terminal'][arm]).sum()),
  pm_accuracy_pct=float(np.asarray(m['outcomes']['pm'][arm]).mean()*100),terminal_accuracy_pct=float(np.asarray(m['outcomes']['terminal'][arm]).mean()*100),
  **t,graph_plus_reconstruction_seconds=t['reconstruction_seconds']+t['graph_diagnostic_seconds_sum']))
by={r['arm']:r for r in rows}
ratios={scope:dict(H8_over_full=by['h8'][scope]/by['full'][scope],H8_over_control=by['h8'][scope]/by['control'][scope],full_over_control=by['full'][scope]/by['control'][scope]) for scope in ('reconstruction_seconds','graph_plus_reconstruction_seconds','worker_wall_seconds_sum','peak_gpu_memory_mb_max')}
boundaries=[]
for shard,entries in paths.items():
 with zipfile.ZipFile(entries['control']) as control,zipfile.ZipFile(entries['live0']) as live:
  cr=control.namelist()[0].split('/')[0]+'/';lr=live.namelist()[0].split('/')[0]+'/'
  for corruption in m['corruptions']:
   for batch in range(4):
    name='batches/%s/batch%04d/diagnostics.json'%(corruption,batch)
    a=json.loads(control.read(cr+name));b=json.loads(live.read(lr+name));pa=a['trajectory']['probes'][0];pb=b['trajectory']['probes'][0]
    assert a['trajectory']['initial_hashes']==b['trajectory']['initial_hashes']
    boundaries.append(dict(case=shard+'/'+corruption+'/'+str(batch),first_clean_latent_hash_equal=pa['clean_latent_sha256']==pb['clean_latent_sha256'],first_decoder_style_hash_equal=pa['style_sha256']==pb['style_sha256'],first_decoded_hash_equal=pa['decoded_geometry_sha256']==pb['decoded_geometry_sha256']))
cache=json.loads((R/'downloaded/metadata/cache_preparation.json').read_text())
report=dict(rows=rows,ratios=ratios,zero_gate=dict(passed=False,failed_B2_cases=len(fails),total_B2_cases=len(g['cases']),
 failure_by_corruption=dict(Counter(v['case'].split('/')[1] for v in fails)),pm_prediction_disagreements=sum(not v for r in g['cases'] for v in r['pm_agreement']),
 terminal_prediction_disagreements=sum(not v for r in g['cases'] for v in r['terminal_agreement']),
 max_first_decoded_per_object_rms=max(max(r['first_decoded_per_object_rms']) for r in g['cases']),max_logit=g['logit_max_absolute'],max_logit_rms=g['logit_per_object_checkpoint_rms_max']),
 boundary_summary={k:sum(r[k] for r in boundaries) for k in ('first_clean_latent_hash_equal','first_decoder_style_hash_equal','first_decoded_hash_equal')},boundary_cases=boundaries,
 preparation=dict(sum_prepare_boundary_seconds=sum(v['preparation_seconds'] for v in cache['records']),loop_elapsed_seconds=cache['elapsed_seconds'],outer_seconds=json.loads((R/'downloaded/metadata/cache-preparation-phase_receipt.json').read_text())['outer_seconds']),
 limitations='Boundary hashes detect bit differences, not their magnitude or causal kernel. Timing ratios are single-run descriptive measurements; graph plus reconstruction excludes loading/cache preparation/terminal head/artifact overhead. Gate failures are retained; thresholds unchanged.')
with (R/'compact_summary.json').open('x') as f:json.dump(report,f,indent=2)
with (R/'per_corruption.csv').open('x',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['corruption','arm','n','pm_correct','pm_accuracy_pct','terminal_correct','terminal_accuracy_pct']);writer.writeheader()
 for i,c in enumerate(m['corruptions']):
  for arm in m['timing']:
   pm=int(np.asarray(m['outcomes']['pm'][arm])[i].sum());terminal=int(np.asarray(m['outcomes']['terminal'][arm])[i].sum())
   writer.writerow(dict(corruption=c,arm=arm,n=32,pm_correct=pm,pm_accuracy_pct=pm/32*100,terminal_correct=terminal,terminal_accuracy_pct=terminal/32*100))
print(json.dumps({k:v for k,v in report.items() if k!='boundary_cases'},indent=2))
