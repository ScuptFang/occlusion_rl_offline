"""Compare paired continuous-error rollouts and export a standalone figure."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',action='append',required=True,help='Label=JSON path')
    p.add_argument('--out',required=True);args=p.parse_args()
    series=[]
    for spec in args.input:
        label,file=spec.split('=',1);series.append((label,json.loads(Path(file).read_text())))
    reference={(r['case'],r['band']):r for r in series[0][1]['cases']}
    for _,data in series[1:]:
        for row in data['cases']:
            other=reference[(row['case'],row['band'])];n=min(row['episodes'],other['episodes'])
            np.testing.assert_allclose(row['initial_rotations'][:n],other['initial_rotations'][:n],atol=1e-7)
            np.testing.assert_allclose(row['initial_translations'][:n],other['initial_translations'][:n],atol=1e-7)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    table=[]
    for label,data in series:
        for zero,axrow in [(False,0),(True,1)]:
            rows=[r for r in data['cases'] if (r['band']=='zero')==zero]
            for col,key in enumerate(['rotation_deg','translation_mm']):
                curves=[np.array([np.mean(s[key]) for s in r['trace']]) for r in rows]
                values=np.mean(curves,axis=0)
                axes[axrow,col].plot(np.arange(len(values)),values,label=label)
        table.append({'label':label,'episodes':sum(r['episodes'] for r in data['cases']),
                      'initial_error':data['initial_macro_error'],'macro':data['macro']})
    for row in range(2):
        for col in range(2):
            ax=axes[row,col];ax.grid(alpha=.25);ax.set_xlabel('Executed actions');ax.legend(fontsize=8)
            ax.set_ylabel('Rotation error (deg)' if col==0 else 'Translation error (mm)')
            ax.set_title(('Near / medium / wide macro average' if row==0 else 'Drift from original pose'))
    fig.tight_layout();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=160);out.with_suffix('.json').write_text(json.dumps(table,indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
