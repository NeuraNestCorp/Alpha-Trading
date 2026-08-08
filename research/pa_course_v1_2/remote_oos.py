#!/usr/bin/env python3
from __future__ import annotations
import base64,hashlib,io,json,math,sys,tarfile,urllib.request
from pathlib import Path
FROZEN_CORE={"src/trading_research/features.py":"948979d14b20d9f742f0a3548849e2654fd35b0d99995ae09f7d4dd684510b78","src/trading_research/features_course.py":"8860651077e11873645f7f0c2c33c58d19c52475b438d8bdee7c01da575eab26","src/trading_research/regimes.py":"fd2e2fb3de3f0d146989709562a317c67f99827ecec7d6c1e2f97a958b51b16d","src/trading_research/context.py":"3b78a6c4316fd7e3e904d068736ef2ae158a07ac7ab4a26b37854b9431d757a2","src/trading_research/strategies/pa_course.py":"e1d7d59b7e868b54ebf2e29b3bdd572dca7cde1ca1a4d91c400cfbf3410eac7f","src/trading_research/backtest/course_engine.py":"0f3486c8aa96cba19b971f159e5d47c0c4791fcaddb8287ee040fbbe45c2f613","src/trading_research/risk.py":"eab6ed74cb7724e545e4f4cd20bdbb0a83fcba922811b6ff5668d512008641f2","src/trading_research/course_runtime.py":"58cf3c3acf99acc95672804c687ef002c88188c31ec80309f3068c4d9386f8f2","configs/pa_course_v1_2.yaml":"035c2f9338008df056449daf2984d2110c27bbb73eb580a78d85bdadcfa77173"}
DATA_URL="https://github.com/getdata-finance/xauusd-5m-ohlcv-metals-historical-data/releases/download/sample-2026-07-31/XAUUSD_5m.csv"
DATA_SHA256="dcdda974e53aa83b7b7043fb816419f3057a18924c5f5c09c9e2f1317541f5ee"
OOS_CUTOFF="2026-05-06T00:00:00+00:00"
HERE=Path(__file__).parent
ARCHIVE_B64=''.join((HERE/f"archive_{i}.b64").read_text().strip() for i in range(4))
WORK=Path("pa_course_large_oos_runtime"); WORK.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE_B64)),mode='r:gz') as tf: tf.extractall(WORK)
for rel,h in FROZEN_CORE.items():
 got=hashlib.sha256((WORK/rel).read_bytes()).hexdigest()
 if got!=h: raise SystemExit(f"FROZEN HASH FAIL {rel} {got} != {h}")
print("FROZEN_HASH_CHECK=PASS",len(FROZEN_CORE))
data_path=WORK/'data/XAUUSD_5m_public_2026-02-02_2026-07-31.csv'
with urllib.request.urlopen(DATA_URL,timeout=120) as r,data_path.open('wb') as f:
 while True:
  chunk=r.read(1024*1024)
  if not chunk: break
  f.write(chunk)
data_hash=hashlib.sha256(data_path.read_bytes()).hexdigest(); print("DATA_SHA256",data_hash)
if data_hash!=DATA_SHA256: raise SystemExit("DATA HASH FAIL")
import numpy as np
import pandas as pd
sys.path.insert(0,str((WORK/'src').resolve()))
from trading_research.features_course import add_course_features
from trading_research.regimes import classify_regime
from trading_research.context import add_session_context,add_event_context,attach_higher_timeframe_context,EconomicEvent
from trading_research.strategies.pa_course import generate_course_setups
from trading_research.backtest.course_engine import run_course_backtest
from trading_research.course_runtime import load_course_config,course_params,backtest_kwargs
CFG=load_course_config(WORK/'configs/pa_course_v1_2.yaml')
def load_events(path):
 e=pd.read_csv(path); return [EconomicEvent(pd.Timestamp(r.timestamp),r.currency,r.impact,r.name) for r in e.itertuples()]
ALL_EVENTS=load_events(WORK/'data/economic_events_sample.csv')
def prep_df(d):
 fkw={**CFG['features'],**CFG['course_gap_context']}; rkw=CFG['regime']
 feat=lambda q:add_course_features(q,**fkw); regime=lambda q:classify_regime(q,**rkw)
 x=regime(feat(d)); x=add_session_context(x,**CFG['context']['session']); dates=set(x.index.date)
 x=add_event_context(x,[e for e in ALL_EVENTS if e.timestamp.date() in dates],**CFG['context']['event']); return attach_higher_timeframe_context(x,feat,regime)
def load_csv(path):
 z=pd.read_csv(path); tc='datetime' if 'datetime' in z.columns else 'timestamp'; z[tc]=pd.to_datetime(z[tc],utc=True); z=z.set_index(tc)[['open','high','low','close','volume']].astype(float).sort_index(); return z[~z.index.duplicated(keep='first')]
def wilson(k,n,z=1.959963984540054):
 if n==0:return [None,None]
 p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return [c-h,c+h]
def bootstrap_ci(vals,seed=20260808,nboot=20000):
 a=np.asarray(vals,dtype=float)
 if len(a)==0:return [None,None]
 rng=np.random.default_rng(seed); means=[]
 for start in range(0,nboot,2000): means.append(rng.choice(a,(min(2000,nboot-start),len(a)),replace=True).mean(axis=1))
 q=np.quantile(np.concatenate(means),[.025,.975]); return [float(q[0]),float(q[1])]
def max_drawdown_r(t):
 if len(t)==0:return 0.0
 eq=t.r.cumsum().to_numpy(); peak=np.maximum.accumulate(np.maximum(eq,0)); return float(np.max(peak-eq))
def summary(t):
 if len(t)==0:return {'trades':0}
 wins=t[t.r>0].r; losses=t[t.r<0].r
 return {'trades':int(len(t)),'win_rate':float((t.r>0).mean()),'win_rate_wilson95':wilson(int((t.r>0).sum()),len(t)),'expectancy_r':float(t.r.mean()),'expectancy_bootstrap95':bootstrap_ci(t.r),'total_r':float(t.r.sum()),'profit_factor':float(wins.sum()/(-losses.sum())) if len(losses) else None,'avg_win_r':float(wins.mean()) if len(wins) else None,'avg_loss_r':float(losses.mean()) if len(losses) else None,'median_r':float(t.r.median()),'avg_cost_r':float(t.cost_r.mean()),'max_actual_risk_r':float(t.actual_risk_r.max()),'avg_actual_risk_r':float(t.actual_risk_r.mean()),'max_drawdown_r_additive':max_drawdown_r(t),'long_trades':int((t.direction==1).sum()),'short_trades':int((t.direction==-1).sum())}
d=load_csv(data_path); print("DATA_ROWS",len(d),"START",d.index.min(),"END",d.index.max()); x=prep_df(d); setups=generate_course_setups(x,course_params(CFG)); print("SETUPS_TOTAL",len(setups))
full_t,full_m=run_course_backtest(x,setups,**backtest_kwargs(CFG)); cut=pd.Timestamp(OOS_CUTOFF); oos_setups=[s for s in setups if x.index[int(s['entry_i'])]>=cut]; print("SETUPS_OOS",len(oos_setups)); oos_t,oos_m=run_course_backtest(x,oos_setups,**backtest_kwargs(CFG))
for t in (full_t,oos_t):
 if len(t): t['entry_time']=pd.to_datetime(t.entry_time,utc=True); t['exit_time']=pd.to_datetime(t.exit_time,utc=True)
out=WORK/'results'; out.mkdir(exist_ok=True); full_t.to_csv(out/'trades_full.csv',index=False); oos_t.to_csv(out/'trades_oos_2026-05-06_onward.csv',index=False)
report={'strategy_id':CFG['strategy_id'],'frozen_hash_check':'PASS','data':{'rows':int(len(d)),'start':str(d.index.min()),'end':str(d.index.max()),'sha256':data_hash},'setups_total':int(len(setups)),'setups_oos':int(len(oos_setups)),'full_sample':summary(full_t),'strict_oos_cutoff':OOS_CUTOFF,'strict_oos':summary(oos_t),'engine_metrics_full':full_m,'engine_metrics_oos':oos_m,'fundamental_event_coverage_note':'Only project economic_events_sample.csv is present (2026-05-04/05); no full-calendar event feed is injected for the rest of the large sample.'}
if len(oos_t):
 oos_t['entry_month']=oos_t.entry_time.dt.strftime('%Y-%m'); report['oos_by_month']={m:summary(g.copy()) for m,g in oos_t.groupby('entry_month')}; report['oos_by_direction']={'long':summary(oos_t[oos_t.direction==1].copy()),'short':summary(oos_t[oos_t.direction==-1].copy())}; report['oos_by_outcome']={str(k):int(v) for k,v in oos_t.outcome.value_counts().items()}; report['oos_by_setup']={str(k):summary(g.copy()) for k,g in oos_t.groupby('setup')}; report['oos_by_session']={str(k):summary(g.copy()) for k,g in oos_t.groupby('session')}
(out/'large_oos_report.json').write_text(json.dumps(report,indent=2,default=str)); print("=== LARGE_OOS_REPORT_JSON ==="); print(json.dumps(report,indent=2,default=str))
