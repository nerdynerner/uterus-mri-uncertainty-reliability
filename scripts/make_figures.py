"""Generate manuscript Figures 1-3 from the recorded aggregate analysis."""
import argparse,csv,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--results",type=Path,default=Path(__file__).resolve().parents[1]/"results");p.add_argument("--output",type=Path,required=True);p.add_argument("--rscript",default="Rscript");args=p.parse_args()
F=args.output;F.mkdir(parents=True,exist_ok=False)
A=json.loads((args.results/"final_analysis.json").read_text())
METHODS=['deterministic','ensemble','dropout']
LABEL={'deterministic':'Baseline entropy','ensemble':'Five-model ensemble','dropout':'MC dropout (30 passes)'}
COLOUR={'deterministic':'#333333','ensemble':'#1764A0','dropout':'#AE5220'}
PARTS=['internal_test','external_test']
PL={'internal_test':'Internal D2 (n=11)','external_test':'External D1 (n=26)'}
fmt=lambda x: 'Undefined' if x is None else f'{x:.3f}'
def ci(r):
    return fmt(r['estimate'])+' ('+('undefined' if r['ci95'] is None else ', '.join(fmt(v) for v in r['ci95']))+')'
def short(m): return {'deterministic':'Baseline','ensemble':'Ensemble','dropout':'Dropout'}[m]
def csv_table(name,headers,rows):
    with (F/name).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
curve_rows=[];point_rows=[];effect_rows=[];bin_rows=[]
for part in PARTS:
 for m in METHODS:
    r=A[part]['methods'][m];p=r['policy'];b=r['primary_effect_ci']
    curve_rows += [[part,m,c,risk] for c,risk in r['risk_coverage']]
    point_rows.append([part,m,p['coverage'],p['retained_error']])
    effect_rows.append([part,m,b['estimate'],*(b['ci95'] or [None,None])])
    bins=A[part]['secondary'][m]['probability_calibration']['temperature_scaled']['reliability_bins']
    bin_rows += [[part,m,b['mean_predicted_probability'],b['observed_target_fraction']] for b in bins if b['weight']]
csv_table('risk_coverage.csv',['cohort','method','coverage','risk'],curve_rows)
csv_table('policy_points.csv',['cohort','method','coverage','risk'],point_rows)
csv_table('effects.csv',['cohort','method','estimate','lower','upper'],effect_rows)
csv_table('reliability.csv',['cohort','method','probability','observed'],bin_rows)
subprocess.run([args.rscript,"--vanilla",str(Path(__file__).with_name("plot_figures.R")),str(F)],check=True)
