"""Supplementary reviewer experiments; original outputs are never overwritten."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name] = '2'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'deps'))
sys.path.insert(0, str(ROOT.parent))
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from lifelines import CoxPHFitter
from lifetimes import BetaGeoFitter
from run_experiment import load_data, build_snapshot, best_threshold, metrics, SEED
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)

def panel_for(data, cutoffs, obs=90, horizon=30):
    frames=[]
    for cutoff in pd.to_datetime(cutoffs):
        assert cutoff-pd.Timedelta(days=obs) >= pd.Timestamp('2010-12-01')
        assert cutoff+pd.Timedelta(days=horizon) <= pd.Timestamp('2011-12-09')
        p=build_snapshot(data,cutoff,obs,horizon)
        future=data[(data.InvoiceDate>=cutoff)&(data.InvoiceDate<cutoff+pd.Timedelta(days=horizon))]
        first=future.groupby('CustomerID').InvoiceDate.min()
        p['duration']=(p.CustomerID.map(first)-cutoff).dt.total_seconds().div(86400).fillna(horizon).clip(lower=1e-6)
        assert np.array_equal(p.label.values,p.CustomerID.isin(first.index).astype(int).values)
        frames.append(p)
    return pd.concat(frames,ignore_index=True)

def bg_summary(data,p,obs=90):
    frames=[]
    for cutoff,part in p.groupby('cutoff',sort=False):
        h=data[(data.InvoiceDate>=cutoff-pd.Timedelta(days=obs))&(data.InvoiceDate<cutoff)]
        orders=h.groupby(['CustomerID','InvoiceNo']).InvoiceDate.min().reset_index()
        g=orders.groupby('CustomerID').InvoiceDate.agg(['count','min','max'])
        q=pd.DataFrame({'x':g['count']-1,'tx':(g['max']-g['min']).dt.total_seconds()/86400,'T':(cutoff-g['min']).dt.total_seconds()/86400})
        q=q.loc[part.CustomerID].reset_index(drop=True);q.index=part.index
        frames.append(q)
    return pd.concat(frames).loc[p.index]

def bg_probability(model,s,h):
    r,alpha=model.params_['r'],model.params_['alpha']
    alive=model.conditional_probability_alive(s.x,s.tx,s['T'])
    # At least one event = alive now times one minus gamma-Poisson zero-event probability.
    prob=alive*(-np.expm1((r+s.x)*np.log((alpha+s['T'])/(alpha+s['T']+h))))
    assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
    return np.asarray(prob)

def report(name,valid,test,pv,pt,extra=None):
    threshold=best_threshold(valid.label.values,pv)
    row={'model':name,**metrics(test.label.values,pt,threshold),'brier':brier_score_loss(test.label,pt)}
    if extra:row.update(extra)
    return row

def main():
    data=load_data();print('Loaded',len(data),flush=True)
    p=panel_for(data,pd.date_range('2011-04-01','2011-11-01',freq='MS'))
    train=p[p.cutoff<='2011-08-01'];valid=p[p.cutoff=='2011-09-01'];test=p[p.cutoff>='2011-10-01']
    features=[x for x in p if x not in ['CustomerID','cutoff','label','duration']]
    rows=[]; predictions=test[['CustomerID','cutoff','label']].copy()
    # Fixed mild ridge penalty stabilizes correlated aggregate covariates; no test tuning.
    scale=StandardScaler().fit(train[features])
    def sx(frame):return pd.DataFrame(scale.transform(frame[features]),columns=features,index=frame.index)
    fit=sx(train);fit['duration']=train.duration;fit['event']=train.label
    cox=CoxPHFitter(penalizer=0.1).fit(fit,'duration','event',show_progress=False)
    pv=1-cox.predict_survival_function(sx(valid),times=[30]).iloc[0].values
    pt=1-cox.predict_survival_function(sx(test),times=[30]).iloc[0].values
    rows.append(report('Cox_All',valid,test,pv,pt));predictions['prob_Cox_All']=pt
    cox.summary.to_csv(OUT/'cox_coefficients.csv')
    print(rows[-1],flush=True)
    # One training landmark avoids counting the same history repeatedly in population likelihood.
    bg_train=bg_summary(data,train[train.cutoff=='2011-08-01'])
    bg=BetaGeoFitter(penalizer_coef=0.01).fit(bg_train.x,bg_train.tx,bg_train['T'])
    bv=bg_summary(data,valid);bt=bg_summary(data,test)
    pv=bg_probability(bg,bv,30);pt=bg_probability(bg,bt,30)
    assert np.allclose(bg_probability(bg,bt,0),0)
    assert (bg_probability(bg,bt,60)>=pt-1e-12).all()
    rows.append(report('BG_NBD',valid,test,pv,pt));predictions['prob_BG_NBD']=pt
    print(rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'additional_baselines.csv',index=False)
    predictions.to_csv(OUT/'additional_predictions.csv',index=False)
    (OUT/'model_config.json').write_text(json.dumps({'cox_penalizer':0.1,'bg_penalizer':0.01,'bg_training_cutoff':'2011-08-01','bg_params':bg.params_.to_dict(),'features':features},indent=2))
    # Shared landmarks valid for all horizons and observation lengths. July+60d precedes Aug? No:
    # Use June only for training, August for validation, October for test to purge label overlap.
    sensitivity=[]
    panels={}
    for obs,horizon in [(60,30),(90,30),(180,30),(90,60)]:
        panels[(obs,horizon)]=panel_for(data,['2011-06-01','2011-08-01','2011-10-01'],obs,horizon)
    # Common customer-cutoff cohort isolates window effects from eligibility changes.
    keys=None
    for q in panels.values():
        current=set(zip(q.CustomerID,q.cutoff)); keys=current if keys is None else keys&current
    for (obs,horizon),q in panels.items():
        q=q[[k in keys for k in zip(q.CustomerID,q.cutoff)]]
        tr=q[q.cutoff=='2011-06-01'];va=q[q.cutoff=='2011-08-01'];te=q[q.cutoff=='2011-10-01']
        assert tr.cutoff.max()+pd.Timedelta(days=horizon)<=va.cutoff.min()
        assert va.cutoff.max()+pd.Timedelta(days=horizon)<=te.cutoff.min()
        for name,cols in [('Logistic_RFM',['recency','frequency','monetary']),('Logistic_All',features)]:
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced',random_state=SEED))
            model.fit(tr[cols],tr.label)
            pv=model.predict_proba(va[cols])[:,1];pt=model.predict_proba(te[cols])[:,1]
            row=report(name,va,te,pv,pt,{'obs_days':obs,'pred_days':horizon,'train_n':len(tr),'valid_n':len(va),'test_n':len(te),'positive_rate':te.label.mean()})
            sensitivity.append(row);print(row,flush=True)
    pd.DataFrame(sensitivity).to_csv(OUT/'window_sensitivity.csv',index=False)
    (OUT/'checks.json').write_text(json.dumps({'complete_history_and_followup':True,'no_cross_split_label_overlap':True,'cox_event_matches_binary_label':True,'bg_probability_bounds_and_horizon_monotonicity':True,'sensitivity_same_customer_cutoff_cohort':True,'seed':SEED},indent=2))

if __name__=='__main__':main()
