"""Bounded CLI transfers; reassemble the exact immutable raw package."""
import argparse,concurrent.futures,hashlib,json,re,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--session',default='tta-gsd-lowband-halfmix-a100-20261107-r1');p.add_argument('manifest',type=Path);p.add_argument('destination',type=Path);args=p.parse_args()
record=json.loads(args.manifest.read_text());out=args.destination;out.mkdir(parents=True,exist_ok=False)
parts=out/'parts';parts.mkdir();logs=out/'transfer_logs';logs.mkdir()
def identity(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return dict(bytes=Path(path).stat().st_size,sha256=h.hexdigest())
def transfer(item):
 target=parts/Path(item['path']).name;assert not target.exists()
 command=['colab','download','-s',args.session,item['path'],str(target)]
 r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
 output=re.sub(r'(colab-runtime-proxy-token=)[^&\s\x1b]+',r'\1[REDACTED]',r.stdout)
 report=dict(command=command,returncode=r.returncode,output=output,expected={k:item[k] for k in ('bytes','sha256')})
 (logs/(target.name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
 assert r.returncode==0,output
 assert identity(target)==report['expected'],str(target)
 return target
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(transfer,item) for item in record['parts']]
 for i,future in enumerate(futures):
  future.result();print('LOWBAND_LOCAL_PART_VERIFIED',i+1,len(futures),flush=True)
archive=out/Path(record['archive']['path']).name
with archive.open('xb') as f:
 for item in record['parts']:
  path=parts/Path(item['path']).name;assert identity(path)=={k:item[k] for k in ('bytes','sha256')}
  with path.open('rb') as source:shutil.copyfileobj(source,f)
assert identity(archive)=={k:record['archive'][k] for k in ('bytes','sha256')}
(out/'raw_receipt.json').write_text(json.dumps(record['archive'],indent=2)+'\n')
shutil.copyfile(args.manifest,out/'transfer_manifest.json')
report=dict(status='CLI chunks and reassembled whole SHA verified',archive=str(archive),**identity(archive),parts=len(record['parts']))
(out/'download_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
