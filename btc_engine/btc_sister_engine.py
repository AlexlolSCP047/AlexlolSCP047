from __future__ import annotations

import json, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

SYMBOL='BTCUSDT'
TZ=ZoneInfo('Europe/Madrid')
BASE='https://api.binance.com/api/v3/klines'
OUT=Path('btc_reports')
OUT.mkdir(exist_ok=True)
(OUT/'history').mkdir(exist_ok=True)


def fetch_klines(interval:str, start_ms:int, end_ms:int, limit:int=1000)->pd.DataFrame:
    rows=[]; cur=start_ms
    step_ms={'1h':3600_000,'4h':4*3600_000}.get(interval, 24*3600_000)
    while cur < end_ms:
        r=requests.get(BASE, params={'symbol':SYMBOL,'interval':interval,'startTime':cur,'endTime':end_ms,'limit':limit}, timeout=30)
        r.raise_for_status(); batch=r.json()
        if not batch: break
        rows.extend(batch)
        nxt=batch[-1][0]+step_ms
        if nxt<=cur: break
        cur=nxt
        time.sleep(0.05)
    cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']
    df=pd.DataFrame(rows,columns=cols).drop_duplicates('open_time').sort_values('open_time')
    for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c])
    df['dt_utc']=pd.to_datetime(df.open_time,unit='ms',utc=True)
    df['dt_local']=df.dt_utc.dt.tz_convert(TZ)
    return df


def daily_from_1h(h:pd.DataFrame)->pd.DataFrame:
    x=h.set_index('dt_local')
    d=x.resample('1D',label='left',closed='left').agg(open=('open','first'),high=('high','max'),low=('low','min'),close=('close','last'),volume=('volume','sum'))
    return d.dropna().reset_index().rename(columns={'dt_local':'date'})


def z(a:np.ndarray)->np.ndarray:
    s=np.std(a)
    return (a-np.mean(a))/(s if s>1e-12 else 1.0)


def corr_score(a,b):
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12: return 50.0
    return float(np.clip((np.corrcoef(a,b)[0,1]+1)*50,0,100))


def rmse_score(a,b):
    az=z(a); bz=z(b)
    rmse=float(np.sqrt(np.mean((az-bz)**2)))
    return float(np.clip(100*(1-rmse/3.0),0,100))


def candle_features(df:pd.DataFrame)->np.ndarray:
    rng=np.maximum(df.high-df.low,1e-9)
    body=np.abs(df.close-df.open)/rng
    upper=(df.high-np.maximum(df.open,df.close))/rng
    lower=(np.minimum(df.open,df.close)-df.low)/rng
    hh=np.r_[0,np.sign(np.diff(df.high))]
    ll=np.r_[0,np.sign(np.diff(df.low))]
    return np.c_[body,upper,lower,hh,ll].ravel()


def similarity(cur:pd.DataFrame, cand:pd.DataFrame)->float:
    rc=np.diff(np.log(cur.close.to_numpy())); rh=np.diff(np.log(cand.close.to_numpy()))
    s1=corr_score(z(rc),z(rh))
    pc=100*np.exp(np.r_[0,np.cumsum(rc)]); ph=100*np.exp(np.r_[0,np.cumsum(rh)])
    s2=rmse_score(pc,ph)
    vc=np.array([np.std(rc),np.mean((cur.high-cur.low)/cur.close)])
    vh=np.array([np.std(rh),np.mean((cand.high-cand.low)/cand.close)])
    s3=rmse_score(vc,vh)
    s4=rmse_score(candle_features(cur),candle_features(cand))
    return round(.45*s1+.30*s2+.15*s3+.10*s4,2)


def classify(ret:float)->str:
    if ret>0.0075:return 'alcista'
    if ret<-0.0075:return 'bajista'
    return 'lateral'


def dedupe(cands:list[dict], n:int)->list[dict]:
    out=[]
    for c in sorted(cands,key=lambda x:x['score'],reverse=True):
        a0,a1=c['start_i'],c['end_i']
        keep=True
        for o in out:
            b0,b1=o['start_i'],o['end_i']
            ov=max(0,min(a1,b1)-max(a0,b0)+1)
            if ov/n>0.5: keep=False; break
        if keep: out.append(c)
    return sorted(out,key=lambda x:x['end_i'])


def sisters(d:pd.DataFrame,n:int)->dict:
    cur=d.iloc[-n:].copy(); cutoff=len(d)-n-1
    cands=[]
    for end_i in range(n-1,cutoff+1):
        start_i=end_i-n+1
        cand=d.iloc[start_i:end_i+1]
        sc=similarity(cur,cand)
        if sc<78: continue
        fut_close=d.iloc[end_i+1].close
        ret=fut_close/d.iloc[end_i].close-1
        cands.append({'start_i':start_i,'end_i':end_i,'start':str(d.iloc[start_i].date.date()),'end':str(d.iloc[end_i].date.date()),'score':sc,'tier':'fuerte' if sc>=85 else 'valida','result':classify(ret),'ret24h':ret})
    kept=dedupe(cands,n)
    weights={k:0.0 for k in ['alcista','lateral','bajista']}
    counts={k:0 for k in weights}
    for c in kept:
        w=c['score']/100; weights[c['result']]+=w; counts[c['result']]+=1
    total=sum(weights.values())
    pct={k:(100*v/total if total else 0.0) for k,v in weights.items()}
    return {'n':n,'strong':sum(c['tier']=='fuerte' for c in kept),'valid':sum(c['tier']=='valida' for c in kept),'total':len(kept),'counts':counts,'pct':pct,'items':kept}

MODELS={
 'M1 14D+4D':{14:.60,4:.40},
 'M2 14D+7D+4D':{14:.45,7:.30,4:.25},
 'M3 7D+4D':{7:.55,4:.45},
 'M4 14D+7D':{14:.60,7:.40},
}


def combine(results:dict,weights:dict)->dict:
    out={k:0.0 for k in ['alcista','lateral','bajista']}
    sample=sum(results[n]['total'] for n in weights)
    for n,w in weights.items():
        for k in out: out[k]+=w*results[n]['pct'][k]
    order=sorted(out,key=out.get,reverse=True)
    return {'pct':out,'principal':order[0],'alternativo':order[1],'weak':out[order[0]]-out[order[1]]<7,'sample':sample,'insufficient':sample<5}


def duration_stats(d:pd.DataFrame, sis:dict, target:str)->dict:
    vals=[]
    for c in sis['items']:
        end=c['end_i']; run=0
        for j in range(end+1,min(end+8,len(d))):
            r=d.iloc[j].close/d.iloc[j-1].close-1
            if classify(r)==target: run+=1
            else: break
        if run: vals.append(run)
    if not vals:return {'cases':0,'median':None,'mean':None,'iqr':None}
    q1,q3=np.percentile(vals,[25,75])
    return {'cases':len(vals),'median':float(np.median(vals)),'mean':float(np.mean(vals)),'iqr':[float(q1),float(q3)]}


def activation_window(h:pd.DataFrame, sis:dict)->dict:
    hours=[]
    for c in sis['items']:
        day=pd.Timestamp(c['end']).date()+timedelta(days=1)
        x=h[h.dt_local.dt.date==day].copy()
        if len(x)<8: continue
        bodies=(x.close-x.open).abs().to_numpy(); found=None
        for i in range(6,len(x)):
            prev=x.iloc[i-6:i]; close=x.iloc[i].close
            broke=close>prev.high.max() or close<prev.low.min()
            if broke and bodies[i]>=1.25*np.median(bodies[i-6:i]): found=x.iloc[i].dt_local; break
        if found is None:
            i=int(np.argmax(np.abs(np.log(x.close/x.open)))); found=x.iloc[i].dt_local
        hours.append((found.hour, c['score']/100))
    if not hours:return {'window':None,'cases':0}
    scores=[sum(w for hr,w in hours if ((hr-start)%24)<3) for start in range(24)]
    st=int(np.argmax(scores)); return {'window':f'{st:02d}:00–{(st+3)%24:02d}:00','cases':len(hours)}


def markdown(report:dict)->str:
    lines=[f"# BTC Sister Models — {report['generated_at']}", '', f"Precio último cierre: **{report['last_close']:.2f} USDT**",'', '## Gráficas hermanas', '', '| Marco | Fuertes | Válidas | Total | Alcistas | Laterales | Bajistas |','|---|---:|---:|---:|---:|---:|---:|']
    for n in [14,7,4]:
        r=report['frames'][str(n)]; p=r['pct']; c=r['counts']
        lines.append(f"| {n}D | {r['strong']} | {r['valid']} | {r['total']} | {c['alcista']} ({p['alcista']:.1f}%) | {c['lateral']} ({p['lateral']:.1f}%) | {c['bajista']} ({p['bajista']:.1f}%) |")
    lines+=['','## Modelos','', '| Modelo | Principal | Alternativo | A/L/B | Muestra | Duración | Ventana |','|---|---|---|---|---:|---|---|']
    for name,m in report['models'].items():
        p=m['pct']; dur=m['duration']; dur_txt='n/d' if not dur['cases'] else f"mediana {dur['median']:.1f} d; IQR {dur['iqr'][0]:.1f}–{dur['iqr'][1]:.1f}"
        lines.append(f"| {name} | {m['principal']} {'(débil)' if m['weak'] else ''} | {m['alternativo']} | {p['alcista']:.1f}/{p['lateral']:.1f}/{p['bajista']:.1f}% | {m['sample']} | {dur_txt} | {m['activation']['window'] or 'n/d'} |")
    lines+=['','> Cálculo automático con velas spot BTCUSDT de Binance; similitud por forma normalizada.']
    return '\n'.join(lines)+'\n'


def main():
    now=datetime.now(timezone.utc); start=now-timedelta(days=210)
    h=fetch_klines('1h',int(start.timestamp()*1000),int(now.timestamp()*1000))
    d=daily_from_1h(h)
    today_local=now.astimezone(TZ).date()
    d=d[d.date.dt.date<today_local].reset_index(drop=True)
    frames={n:sisters(d,n) for n in [14,7,4]}
    models={}
    for name,w in MODELS.items():
        m=combine(frames,w); primary_frame=max(w,key=w.get)
        m['duration']=duration_stats(d,frames[primary_frame],m['principal'])
        m['activation']=activation_window(h,frames[primary_frame])
        models[name]=m
    report={'generated_at':now.astimezone(TZ).isoformat(timespec='seconds'),'last_close':float(d.iloc[-1].close),'frames':{str(k):v for k,v in frames.items()},'models':models}
    (OUT/'latest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'latest.md').write_text(markdown(report),encoding='utf-8')
    day=now.astimezone(TZ).date().isoformat()
    (OUT/'history'/f'{day}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(markdown(report))

if __name__=='__main__': main()
