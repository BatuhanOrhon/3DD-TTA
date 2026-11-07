"""Local acknowledged-shard progress; no remote access or automatic Scan launch."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

ANALYZER_SHA256='e13ee21e8a48295e510e657ea5fae17a3a482679afbbc79a1e8adc20b94a7f44'
OWNER='<!-- owned by gsd-lowband-full-20261107/watch_progress.py -->'
HEADER='# Full ModelNet40-C progress — 2026-11-07\n\n'+OWNER+'\n\nNative repeat gates failed; these single-seed object-cluster intervals exclude native numerical variability. Fixed39-shard coverage continues regardless of interim scores. Only the full accepted result can evaluate the frozen external gate; this watcher never launches Scan.\n\n| UTC | Shards | Objects | Predictions/arm | PM correct control / half0 / M100 | M100 − control pp [CI95] | Status | External gate |\n| --- | ---: | ---: | ---: | --- | --- | --- | --- |\n'


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def identity(path):
    raw=Path(path).read_bytes()
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def exclusive(path,raw):
    with Path(path).open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


def atomic(path,raw):
    path=Path(path);temporary=path.with_name(path.name+'.pending.'+uuid.uuid4().hex)
    exclusive(temporary,raw);os.replace(temporary,path)


def encoded(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def receipt_snapshot(root,plan,cache=None):
    root=Path(root);cache={} if cache is None else cache
    expected={shard['label']+'_'+role:role for shard in plan['shards'] for role in ('cache','control','half0','M100')}
    records={}
    for path in (root/'accepted').glob('shard*.json'):
        label=path.stem;assert label in expected,'Unexpected owned receipt label'
        stat=path.stat();stamp=(stat.st_ino,stat.st_mtime_ns,stat.st_size)
        previous=cache.get(label)
        if previous is not None and previous[0]==stamp:
            record=previous[1]
        else:
            raw=json.loads(path.read_text())
            record={key:raw[key] for key in ('label','kind','ack_uploaded','status','archive_identity','ack')}
            cache[label]=(stamp,record)
        role=expected[label];kind='cache' if role=='cache' else 'inference'
        assert record['label']==label and record['kind']==kind and type(record['ack_uploaded']) is bool
        assert record['status']==('accepted and cleanup ack uploaded' if record['ack_uploaded'] else 'locally validated; cleanup ack pending')
        archive=record['archive_identity'];assert type(archive['bytes']) is int and archive['bytes']>0 and re.fullmatch('[0-9a-f]{64}',archive['sha256'])
        ack=record['ack'];assert ack['label']==label and ack['kind']==kind and ack['status']=='local_and_drive_verified'
        assert {key:ack[key] for key in ('bytes','sha256')}==archive
        records[label]=record
    complete=[];all_present=[]
    for shard in plan['shards']:
        labels=[shard['label']+'_'+role for role in ('cache','control','half0','M100')]
        all_present.append(all(label in records for label in labels))
        complete.append(all(label in records and records[label]['ack_uploaded'] for label in labels))
    contiguous=0
    for ready in complete:
        if not ready:break
        contiguous+=1
    return dict(updated_utc=utc(),ack_cache_jobs=sum(r['ack_uploaded'] and r['kind']=='cache' for r in records.values()),
        ack_inference_jobs=sum(r['ack_uploaded'] and r['kind']=='inference' for r in records.values()),
        awaiting_ack=sum(not r['ack_uploaded'] for r in records.values()),contiguous_shards=contiguous,
        complete_shards=sum(complete),all_roles_present_shards=sum(all_present),
        safe_to_aggregate=sum(all_present)==contiguous)


class Watcher:
    def __init__(self,root,knowledge,analyzer,analyzer_path):
        self.root=Path(root).resolve();self.knowledge=Path(knowledge).resolve()
        self.analyzer=analyzer;self.analyzer_path=Path(analyzer_path).resolve()
        assert identity(self.analyzer_path)['sha256']==ANALYZER_SHA256,'Frozen analyzer SHA differs'
        self.plan=json.loads((self.root/'operator/plan.json').read_text())
        assert len(self.plan['shards'])==39
        self.ids=[]
        for ordinal,shard in enumerate(self.plan['shards']):
            assert shard['label']=='shard%03d'%ordinal and shard['count']==(36 if ordinal==38 else 64)
            assert len(shard['arms'])==3 and set(shard['arms'])=={'control','half0','M100'}
            selection=(self.root/'operator'/shard['selection_file']).resolve()
            assert (self.root/'operator') in selection.parents
            ids=json.loads(selection.read_text())['indices']
            assert ids==list(range(ordinal*64,min((ordinal+1)*64,2468)))
            self.ids.append(ids)
        self.derived=self.root.parent/'derived' if self.root.name=='downloaded' else self.root/'derived'
        self.derived.mkdir(exist_ok=True)
        self.cache={}
        if not self.knowledge.exists():exclusive(self.knowledge,HEADER.encode())
        assert self.knowledge.read_text().startswith(HEADER),'Progress file is not exclusively watcher-owned'
        self.entries=self.read_entries()

    def path(self,count):
        return self.derived/('full_final.json' if count==39 else 'partial_shards%03d.json'%count)

    def read_entries(self):
        text=self.knowledge.read_text()
        matches=re.findall(r'<!-- progress shards=(\d{3}) report_sha256=([0-9a-f]{64}) -->',text)
        assert text.count('<!-- progress shards=')==len(matches)
        counts=[int(count) for count,_ in matches]
        assert counts==sorted(set(counts)) and all(1<=count<=39 for count in counts)
        result=dict((int(count),sha) for count,sha in matches)
        for count,sha in result.items():
            assert identity(self.path(count))['sha256']==sha,'Recorded immutable progress report changed'
        return result

    def check_report(self,report,count):
        assert 1<=count<=39
        ids=[index for values in self.ids[:count] for index in values];objects=len(ids)
        coverage=dict(shards=count,objects=objects,indices=ids,predictions_per_arm=15*objects,
            inference_jobs=3*count,cache_jobs=count,actual_prepared_caches=15*objects//2)
        assert report['coverage']==coverage,'Derived coverage differs from acknowledged contiguous prefix'
        assert report['status']==('complete' if count==39 else 'partial')
        if count<39:assert report['external_gate'] is None,'Partial result cannot evaluate external gate'
        else:assert type(report['external_gate']['advance_to_external']) is bool and report['provenance']['raw_zip_sha_verified'] is True
        proof=report['provenance']
        for key,path in (('plan',self.root/'operator/plan.json'),('common_source_identity',self.root/'common_identity.json')):
            assert {field:proof[key][field] for field in ('bytes','sha256')}==identity(path)
        expected={shard['label']+'_'+role for shard in self.plan['shards'][:count] for role in ('cache','control','half0','M100')}
        assert len(proof['receipts'])==4*count and {row['label'] for row in proof['receipts']}==expected
        for row in proof['receipts']:
            path=self.root/'accepted'/(row['label']+'.json');receipt=json.loads(path.read_text())
            assert receipt['ack_uploaded'] is True
            assert {key:row['receipt'][key] for key in ('bytes','sha256')}==identity(path)
            assert row['archive']==receipt['archive_identity']
        for arm in ('control','half0','M100'):
            rows=report['per_corruption'][arm]
            assert len(rows)==15 and all(type(row['n']) is int and row['n']==objects and type(row['pm_correct']) is int and 0<=row['pm_correct']<=objects for row in rows.values())
        metrics=report['comparisons']['M100_vs_control']['pm'];ci=metrics['object_cluster_ci95_pp']
        assert len(ci)==2 and ci[0]<=ci[1] and all(type(v) in (int,float) and math.isfinite(v) for v in (metrics['delta_pp'],*ci))

    def ensure_report(self,count):
        assert identity(self.analyzer_path)['sha256']==ANALYZER_SHA256
        path=self.path(count);legacy=self.derived/'partial_shard000.json'
        if path.exists():raw=path.read_bytes();report=json.loads(raw)
        elif count==1 and legacy.exists():raw=legacy.read_bytes();report=json.loads(raw)
        else:
            report=self.analyzer.aggregate(self.root,final=count==39);raw=encoded(report)
        self.check_report(report,count)
        metadata=path.with_suffix('.meta.json')
        expected=dict(schema_version=1,analyzer=dict(path=str(self.analyzer_path),**identity(self.analyzer_path)),
            root=str(self.root),coverage=report['coverage'],report=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        if metadata.exists():
            saved=json.loads(metadata.read_text())
            assert all(saved[key]==value for key,value in expected.items()),'Immutable analyzer/report metadata differs'
        else:exclusive(metadata,encoded(dict(expected,created_utc=utc())))
        if not path.exists():exclusive(path,raw)
        assert identity(path)=={key:expected['report'][key] for key in ('bytes','sha256')}
        return report,path,expected['report']['sha256']

    def step(self):
        state=receipt_snapshot(self.root,self.plan,self.cache);count=state['contiguous_shards']
        assert not self.entries or count>=max(self.entries),'Acknowledged progress regressed; inspect preserved evidence'
        if count and state['safe_to_aggregate']:
            report,path,sha=self.ensure_report(count)
            if count not in self.entries:
                correct={arm:sum(row['pm_correct'] for row in report['per_corruption'][arm].values()) for arm in ('control','half0','M100')}
                metric=report['comparisons']['M100_vs_control']['pm'];ci=metric['object_cluster_ci95_pp']
                marker='<!-- progress shards=%03d report_sha256=%s -->'%(count,sha)
                row='| %s | %d/39 | %d/2468 | %d | %d / %d / %d | %.6f [%.6f, %.6f] | %s | %s |\n'%(utc(),count,report['coverage']['objects'],report['coverage']['predictions_per_arm'],correct['control'],correct['half0'],correct['M100'],metric['delta_pp'],ci[0],ci[1],report['status'],json.dumps(report['external_gate'],sort_keys=True))
                text=self.knowledge.read_text();assert text.startswith(HEADER)
                row=row.replace('|\n',marker+' |\n')
                atomic(self.knowledge,(text+row).encode());self.entries[count]=sha
            else:assert self.entries[count]==sha
            state.update(report=str(path),external_gate=report['external_gate'],status=report['status'])
        else:state.update(report=str(self.path(max(self.entries))) if self.entries else None,external_gate=None,status='waiting_for_acknowledged_contiguous_shards')
        atomic(self.derived/'latest_progress.json',encoded(state))
        return state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--knowledge-progress',type=Path,required=True)
    parser.add_argument('--analyzer',type=Path,default=Path(__file__).with_name('analyze_full.py'))
    args=parser.parse_args(argv)
    assert identity(args.analyzer)['sha256']==ANALYZER_SHA256
    spec=importlib.util.spec_from_file_location('frozen_full_analyzer',args.analyzer)
    analyzer=importlib.util.module_from_spec(spec);spec.loader.exec_module(analyzer)
    watcher=Watcher(args.root,args.knowledge_progress,analyzer,args.analyzer)
    with (watcher.derived/'watch_progress.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        stopped=threading.Event()
        def status():
            while not stopped.wait(55):
                print('FULL_LOCAL_PROGRESS',json.dumps(receipt_snapshot(watcher.root,watcher.plan,watcher.cache),sort_keys=True),flush=True)
        thread=threading.Thread(target=status,daemon=True);thread.start()
        try:
            while True:
                state=watcher.step()
                if state['contiguous_shards']==39 and state['status']=='complete':
                    print('FULL_LOCAL_FINAL_ANALYSIS_READY',json.dumps(dict(path=state['report'],external_gate=state['external_gate']),sort_keys=True),flush=True)
                    return
                time.sleep(10)
        finally:stopped.set();thread.join(timeout=2)


if __name__=='__main__':main()
