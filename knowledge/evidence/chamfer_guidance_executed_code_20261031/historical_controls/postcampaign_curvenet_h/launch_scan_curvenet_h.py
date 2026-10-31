"""Forward one reviewed new-scope phase to the sole outer monitor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import scan_curvenet_h_controls as controls


def forward(command,cwd,log_path,marker):
    seen=False
    # The phase owns its bounded worker process group. This launcher never kills
    # a controller on an unrelated outer timeout and leaves an unknown child.
    with Path(log_path).open('x') as log:
        child=subprocess.Popen(command,cwd=str(cwd),stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in child.stdout:
                log.write(line);log.flush();sys.stdout.write(line);sys.stdout.flush()
                seen=seen or line.strip()==marker
            code=child.wait()
        finally:child.stdout.close()
    if code!=0 or not seen:
        raise RuntimeError('Phase failed or exact completion marker missing; inspect '+str(log_path))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--stage',choices=('deploy',*controls.STAGES),required=True)
    args=parser.parse_args(argv)
    controls.require(controls.identity(args.plan)['sha256']==args.plan_sha256,'Plan SHA differs')
    plan=json.loads(args.plan.read_text());controls.validate_plan(plan)
    controls.verify_control_sources(plan,required=Path(__file__).name)
    controller=plan['control_sources']['scan_curvenet_h_controls.py']['path']
    log=controls.owned(Path(plan['research_root'])/('launch_'+args.stage+'.log'),plan['research_root'])
    log.parent.mkdir(parents=True,exist_ok=True)
    command=[plan['python'],'-u',controller,'--plan',str(args.plan),
             '--plan-sha256',args.plan_sha256,'--stage',args.stage]
    forward(command,Path(controller).parent,log,'SCAN_CURVENET_H_STAGE_COMPLETE '+args.stage)


if __name__=='__main__':main()
