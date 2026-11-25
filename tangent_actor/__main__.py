import argparse
import json
from .api import train


def main():
    p=argparse.ArgumentParser(description='Train an extracted Tangent-Actor implementation')
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda']);p.add_argument('--variant',choices=['full','standard','no_mcv','no_terminal_anchor','no_updates','low_rank'])
    a=p.parse_args();c=json.load(open(a.config))
    if a.device:c['device']=a.device
    if a.variant:c['variant']=a.variant
    r=train(c,a.output)
    print(json.dumps({k:r[k] for k in ('pde','dimension','updates','variant','wall_seconds','finite','output_mean_plain','output_mean_mcv')},indent=2))


if __name__=='__main__':main()
