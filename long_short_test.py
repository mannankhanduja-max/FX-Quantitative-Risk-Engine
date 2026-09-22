"""
Is the long advantage a directional edge, or the 2024-26 bull market?

Test: split every (instrument, month) cell by that instrument's OWN
return over the month. If longs beat shorts only when the market
rose, the advantage is the trend. If the gap holds in down months
too, it is structural.

Contemporaneous by design - the question is whether the edge is
explained by the move it happened inside, not whether the move is
predictable.
"""
import argparse, numpy as np, pandas as pd, config, mtf_test as M
from fxrisk.data import intraday

args=argparse.Namespace(rr=2.0,stop_sigma=1.5,lookback=20,trigger_window=6,max_bars=20,
    cost_bp=1.0,cost="real",commission_bp=0.0,cost_power=0.5,
    sessions="per-instrument",blackout=True,bias=False)

pool=[]; rets={}
for inst in config.UNIVERSE_INTRADAY:
    t,p,_=M.run_one(inst,args,("breakout","fakeout","retrace"))
    t=t.copy(); t["instrument"]=inst.name; pool.append(t)
    b=intraday.load_symbol(inst.yahoo,"5m")["Close"]
    rets[inst.name]=b.resample("ME").last().pct_change()

a=pd.concat(pool,ignore_index=True)
a["month"]=a["entry_time"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.to_period("M")

rows=[]
for (i,m),g in a.groupby(["instrument","month"]):
    r=rets[i]
    idx=[x for x in r.index if x.tz_localize(None).to_period("M")==m]
    if not idx or not np.isfinite(r.loc[idx[0]]): continue
    mr=float(r.loc[idx[0]])
    L=g[g["side"]>0]["unit_net_R"]; S=g[g["side"]<0]["unit_net_R"]
    if len(L)<15 or len(S)<15: continue
    rows.append({"instrument":i,"month":str(m),"mret":mr,"n":len(g),
                 "long":L.mean(),"short":S.mean(),"gap":L.mean()-S.mean()})
d=pd.DataFrame(rows)
print(f"cells: {len(d)}  ({d['instrument'].nunique()} instruments)")

up=d[d["mret"]>0]; dn=d[d["mret"]<=0]
print(f"\n{'':<14}{'cells':>6}{'long R':>9}{'short R':>9}{'gap':>9}")
for nm,x in (("UP months",up),("DOWN months",dn)):
    print(f"{nm:<14}{len(x):>6}{x['long'].mean():>+9.4f}{x['short'].mean():>+9.4f}{x['gap'].mean():>+9.4f}")

# Is the gap significant within DOWN months alone?
for nm,x in (("UP",up),("DOWN",dn)):
    t=x["gap"].mean()/(x["gap"].std(ddof=1)/np.sqrt(len(x)))
    print(f"  {nm:<5} gap t = {t:+.2f}   ({(x['gap']>0).mean():.0%} of cells positive)")

c=np.corrcoef(d["mret"],d["gap"])[0,1]
print(f"\ncorr(month return, long-short gap) = {c:+.3f}")
b1=np.polyfit(d["mret"],d["gap"],1)
print(f"slope {b1[0]:+.3f}   intercept {b1[1]:+.4f}  <- gap at ZERO monthly return")

print("\nby instrument, DOWN months only")
for i,g in dn.groupby("instrument"):
    print(f"  {i:<9} cells={len(g):<3} long {g['long'].mean():+.4f}  short {g['short'].mean():+.4f}  gap {g['gap'].mean():+.4f}")
