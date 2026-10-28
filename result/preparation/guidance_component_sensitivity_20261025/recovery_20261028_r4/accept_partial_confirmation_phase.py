"""Archive genuine partial confirmation without creating complete acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,shutil,sys,zipfile
ROOT=Path('/content/studies/guidance-components')
PLAN_SHA='1403356329a8edfc717a505b2d27e2be683f314a99d310eaf6255af4202c4d42'
PACKAGE_ID=dict(bytes=722266,sha256='dc5b854e27ecd96f1d04a5420332749e0b73a7574f011dadc7e4733fd46d68df')
STATUS='accepted_partial_confirmation_with_numerical_failure'
NAMES=('accepted_partial_sensitivity_confirmation.json','validation.json','unavailable_endpoints.json','final_campaign_ledger.json','sensitivity_confirmation_freeze.json','accepted_sensitivity_development.json','original_failed_receipt.json','remaining_receipt.json','failure_preservation_receipt.json')
def bid(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def validate(blobs):
 a=json.loads(blobs[NAMES[0]]);v=json.loads(blobs['validation.json']);l=json.loads(blobs['final_campaign_ledger.json']);u=json.loads(blobs['unavailable_endpoints.json'])
 assert a['status']==v['status']==STATUS and a['stage']==v['stage']=='confirmation_partial'
 assert a['phase_plan_sha256']==v['phase_plan_sha256']==PLAN_SHA and a['local_validation']==bid(blobs['validation.json'])
 assert a['failed_registered_jobs']==v['failed_registered_jobs']==[20] and a['tuned_chamfer_three_seed_endpoint'] is None
 assert v['registered_jobs']==24 and v['completed_jobs']==23 and v['complete_raw_archives']==92 and v['partial_failed_raw_archives']==4
 assert v['primary_complete_jobs']==18 and v['primary_complete_raw_archives']==72 and v['final_unrun_slots']==[]
 assert a['physical_cache_complete'] is v['physical_cache_complete'] is True and v['physical_cache_files']==v['physically_verified_cache_files']==180
 assert a['n_original_object_clusters']==v['n_original_object_clusters']==280
 assert v['analysis_complete_conditions']==a['analysis_complete_conditions']==['spectral-candidate','matched-chamfer','original3dd']
 assert v['frozen_conditions']==a['frozen_conditions']==['spectral-candidate','matched-chamfer','tuned-chamfer','original3dd']
 assert len(u)==1 and u[0]['condition_id']=='tuned-chamfer' and u[0]['status']=='unavailable_due_numerical_failure' and u[0]['three_seed_balanced_accuracy_pct'] is u[0]['conditional_object_bootstrap_interval_pp'] is None
 assert len(v['primary_contrasts'])==2 and {x['comparator'] for x in v['primary_contrasts']}=={'matched-chamfer','original3dd'}
 for x in v['primary_contrasts']:
  assert x['n_original_object_clusters']==280 and x['n_seeds']==3 and x['n_prediction_events']==50400 and x['conditional_object_bootstrap_low_pp']<=x['conditional_object_bootstrap_high_pp']
  assert abs(x['delta_pp']-100*(x['corrected_events']-x['harmed_events'])/50400)<1e-8
 assert v['source_inventory_recovered_from_snapshot'] is True and v['missing_original_executed_source_inventory_keys']==['assets/source_snapshot/shared_trajectory.py']
 assert v['final_ledger_identity']==bid(blobs['final_campaign_ledger.json']) and l['active'] is None and l['cap_seconds']==19800 and a['campaign_spent_seconds']==v['campaign_spent_seconds']==l['spent_seconds']<=19800
 assert l['attempts'][-1]['stage']=='confirmation_remaining' and l['attempts'][-1]['status']=='complete' and [x for x in l['attempts'] if x['stage']=='confirmation'][0]['status']=='failed'
 assert v['frozen_selection']==bid(blobs['sensitivity_confirmation_freeze.json']) and v['development_acceptance']==bid(blobs['accepted_sensitivity_development.json'])
 return a,v,l

def main():
 package=Path('/content/agent_b_r4_partial_confirmation_acceptance.zip');assert bid(package.read_bytes())==PACKAGE_ID
 cp=ROOT/'sensitivity_phases/sensitivity_campaign_controller.py';assert bid(cp.read_bytes())['sha256']=='0095e5af256f4fb980f2666b089b86c7b126bad3994506054cf2f43b9ed2a3fc'
 spec=importlib.util.spec_from_file_location('partial_archive_control',cp);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
 plan,_,_=c.plan_verified(ROOT/'sensitivity_campaign_plan.json',PLAN_SHA)
 hp=ROOT/'phases/deploy_assets_template.py';assert bid(hp.read_bytes())['sha256']=='89fef5cf42dc27db444c89926bdd3aefac8cb8b14727a747256a515c08ca32d0';sys.path.insert(0,str(hp.parent));from deploy_assets_template import checked_members,copy_checked
 with zipfile.ZipFile(package) as z:
  checked_members(z);blobs={name:z.read(name) for name in NAMES}
 a,v,l=validate(blobs);assert v['session']==c.SESSION and v['manifest_sha256']==plan['manifest_sha256'] and v['source_snapshot_sha256']==plan['source_snapshot_sha256']
 assert c.load_ledger(PLAN_SHA)==l and c.identity(ROOT/c.LEDGER_NAME)==v['final_ledger_identity']
 for local,remote in [('original_failed_receipt.json','sensitivity_confirmation_receipt.json'),('remaining_receipt.json','sensitivity_confirmation_remaining_receipt.json'),('failure_preservation_receipt.json','sensitivity_confirmation_failure_preservation_receipt.json'),('sensitivity_confirmation_freeze.json','sensitivity_confirmation_freeze.json'),('accepted_sensitivity_development.json','accepted_sensitivity_development.json')]:
  assert blobs[local]==(ROOT/remote).read_bytes()
 for row in [v['original_failed_phase_archive'],v['remaining_phase_archive'],v['cache_evidence_archive']]:c.verify(c.owned(row['drive_path'],c.DRIVE),row)
 targets={'accepted_partial_sensitivity_confirmation.json':'accepted_partial_sensitivity_confirmation.json','validation.json':'accepted_partial_sensitivity_confirmation_validation.json'}
 receipt_path=ROOT/'sensitivity_confirmation_partial_acceptance_receipt.json'
 assert not (ROOT/'accepted_sensitivity_confirmation.json').exists() and not receipt_path.exists() and all(not(ROOT/n).exists() for n in targets.values())
 drive=c.DRIVE/'phase-evidence'/('sensitivity_partial_confirmation_acceptance_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));drive.mkdir(parents=True,exist_ok=False)
 copy_checked(package,drive/package.name,PACKAGE_ID)
 for src,dst in targets.items():
  with (ROOT/dst).open('xb') as f:f.write(blobs[src])
 result=dict(status=STATUS,phase_plan_sha256=PLAN_SHA,archive=dict(PACKAGE_ID,drive_path=str(drive/package.name)),acceptance=bid(blobs[NAMES[0]]),local_validation=bid(blobs['validation.json']),failed_registered_jobs=[20],completed_jobs=23,n_original_object_clusters=280,physical_cache_files=180,campaign_spent_seconds=l['spent_seconds'],complete_four_condition_confirmation=False)
 with receipt_path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 (drive/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 print('AGENT_B_PARTIAL_CONFIRMATION_ACCEPTANCE',json.dumps(result),flush=True)
 print('AGENT_B_SENSITIVITY_PARTIAL_CONFIRMATION_ACCEPTANCE_COMPLETE',flush=True)
if __name__=='__main__':main()
