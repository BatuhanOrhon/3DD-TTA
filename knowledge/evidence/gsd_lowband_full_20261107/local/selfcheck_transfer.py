"""Exercise real transfer functions with injected pre-ACK failure; no remote operation."""
import argparse,ast,hashlib,importlib.util,json,os,re,shutil,subprocess,tempfile,time,types,uuid,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
source=(HERE/'transfer_daemon.py').read_text();tree=ast.parse(source)
functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
with tempfile.TemporaryDirectory() as temporary:
 root=Path(temporary);(root/'accepted').mkdir();remote=Path('/content/studies/gsd-lowband-full-20261107')
 raw=b'exact synthetic archive byte payload';digest=hashlib.sha256(raw).hexdigest();size=len(raw);label='shard000_control'
 manifest=dict(label=label,kind='inference',archive=dict(path=str(remote/'raw_runs'/label/'run.zip'),drive='/content/drive/full/run.zip',bytes=size,sha256=digest),
  parts=[dict(path=str(remote/'transfers'/label/'part00000.bin'),bytes=size,sha256=digest)])
 state=dict(downloads=0,uploads=0,remote_ack=None,failed=False,journal={label:dict(status='accepted_pending_ack')},fallback_downloads=[])
 ns=dict(Path=Path,ROOT=root,REMOTE=remote,json=json,os=os,hashlib=hashlib,re=re,shutil=shutil,time=time,uuid=uuid,
  args=types.SimpleNamespace(session='owned_test'),shards={'shard000':dict(label='shard000',arms=['control','half0','M100'])},PYTHON='fake_python',HERE=HERE,tempfile=tempfile)
 exec(compile(functions,'actual_transfer_functions','exec'),ns)
 def cli(*args):
  if args[0]=='download' and args[3]=='manifest':Path(args[4]).write_text(json.dumps(manifest));return ''
  if args[0]=='download' and args[3]==str(remote/'campaign_journal.json'):
   state['fallback_downloads'].append(args[3]);Path(args[4]).write_text(json.dumps(dict(status='running',jobs=state['journal'])));return ''
  if args[0]=='download' and args[3]==str(remote/'transfers'/label/'transfer_manifest.json'):
   state['fallback_downloads'].append(args[3]);Path(args[4]).write_text(json.dumps(manifest));return ''
  if args[0]=='download':
   if state['remote_ack'] is None:raise RuntimeError('File or directory not found')
   Path(args[4]).write_text(json.dumps(state['remote_ack']));return ''
  assert args[0]=='upload';state['uploads']+=1
  accepted=root/'accepted'/(label+'.json');assert accepted.is_file(),'ACK must follow durable local receipt'
  if not state['failed']:state['failed']=True;raise RuntimeError('Injected acknowledgement network failure')
  state['remote_ack']=json.loads(Path(args[3]).read_text());return ''
 def run(command,**kwargs):
  state['downloads']+=1;out=Path(command[-1]);out.mkdir(parents=True);(out/'parts').mkdir();(out/'run.zip').write_bytes(raw);(out/'parts/part00000.bin').write_bytes(raw)
  return types.SimpleNamespace(returncode=0)
 ns.update(cli=cli,subprocess=types.SimpleNamespace(run=run,STDOUT=subprocess.STDOUT),accept_inference=lambda *args:dict(validated_fixture=True))
 event=dict(label=label,kind='inference',manifest='manifest',archive_identity=dict(bytes=size,sha256=digest))
 try:ns['process'](event)
 except RuntimeError as error:assert 'Injected acknowledgement' in str(error)
 else:raise AssertionError('Injected failure not reached')
 receipt=json.loads((root/'accepted'/(label+'.json')).read_text());assert not receipt['ack_uploaded'] and Path(receipt['local_archive']).read_bytes()==raw
 ns['process'](event);assert state['downloads']==1 and state['uploads']==2
 receipt=json.loads((root/'accepted'/(label+'.json')).read_text());assert receipt['ack_uploaded']
 assert not (Path(receipt['local_archive']).parent/'parts').exists()
 ns['process'](event);assert state['downloads']==1 and state['uploads']==2
 assert not list((root/'accepted').glob('*.pending.*'))
 events,complete=ns['journal_events']()
 assert not complete and events==[dict(event,manifest=str(remote/'transfers'/label/'transfer_manifest.json'))]
 assert state['fallback_downloads']==[str(remote/'campaign_journal.json'),str(remote/'transfers'/label/'transfer_manifest.json')]
 ns['process'](events[0]);assert state['downloads']==1 and state['uploads']==2,'Journal recovery must replay the durable ack without redownloading raw data'
 state['journal'][label]['status']='acked_and_cleaned'
 assert ns['journal_events']()==([],False)
 state['journal'][label]['status']='running'
 assert ns['journal_events']()==([],False)
 state['journal'][label]['status']='accepted_pending_ack'
 manifest['archive']['sha256']='bad'
 try:ns['journal_events']()
 except AssertionError:pass
 else:raise AssertionError('Fallback accepted an invalid manifest SHA')
 manifest['archive']['sha256']=digest
 state['journal']['../peer_cache']=dict(status='accepted_pending_ack')
 try:ns['journal_events']()
 except AssertionError:pass
 else:raise AssertionError('Fallback accepted an unowned journal label')
 del state['journal']['../peer_cache']
 label='shard000_half0';manifest.update(label=label)
 manifest['archive']['path']=str(remote/'raw_runs'/label/'run.zip')
 manifest['parts'][0]['path']=str(remote/'transfers'/label/'part00000.bin')
 state['journal']={label:dict(status='accepted_pending_ack')};state['remote_ack']=None
 events,complete=ns['journal_events']();assert len(events)==1 and events[0]['label']==label and not complete
 ns['process'](events[0])
 recovered=json.loads((root/'accepted'/(label+'.json')).read_text())
 assert recovered['ack_uploaded'] and state['downloads']==2 and state['uploads']==3
 assert Path(recovered['local_archive']).read_bytes()==raw,'An unseen monitor event must recover entirely through the journal'
print('FULL_TRANSFER_ACK_CRASH_REPLAY_AND_PARTS_BOUNDARY_SELFCHECK_PASSED')
