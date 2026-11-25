from pathlib import Path
import json
from tangent_actor import train

if __name__=='__main__':
 c=json.loads((Path(__file__).resolve().parents[1]/'configs/smoke_joint.json').read_text())
 print(json.dumps(train(c,Path('runs/joint_smoke')),default=str,indent=2))
