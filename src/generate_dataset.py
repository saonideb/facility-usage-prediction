from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"data"/"facility_bookings.csv"
SEED=20261019
FACILITIES=["Gym","Swimming Pool","Badminton Court","Tennis Court","Clubhouse","Multipurpose Hall"]

def generate_dataset(n_residents=450,n_bookings=12000,seed=SEED):
    rng=np.random.default_rng(seed)
    start=pd.Timestamp("2024-01-01")
    end=pd.Timestamp("2026-09-30 23:59:59")
    pop=np.array([.31,.20,.17,.11,.12,.09])
    residents=[f"R-{i:04d}" for i in range(1,n_residents+1)]
    prefs={}
    hour_pref={}
    weekday_pref={}
    lead_base={}
    for r in residents:
        dom=int(rng.choice(len(FACILITIES),p=pop))
        alpha=np.full(len(FACILITIES),.30); alpha[dom]=4.0
        prefs[r]=rng.dirichlet(alpha)
        hour_pref[r]=int(rng.choice([7,8,9,12,17,18,19,20],p=[.08,.10,.08,.07,.12,.21,.20,.14]))
        weekday_pref[r]=int(rng.choice(7))
        lead_base[r]=float(rng.lognormal(np.log(rng.choice([18,30,48,96])),.28))
    day_probs={
      "Gym":[.18,.16,.15,.15,.14,.12,.10],
      "Swimming Pool":[.11,.11,.12,.12,.13,.20,.21],
      "Badminton Court":[.13,.13,.13,.14,.15,.16,.16],
      "Tennis Court":[.12,.12,.13,.14,.15,.17,.17],
      "Clubhouse":[.10,.10,.11,.12,.14,.21,.22],
      "Multipurpose Hall":[.10,.10,.11,.12,.14,.21,.22]}
    hours={
      "Gym":([6,7,8,9,12,17,18,19,20,21],[.10,.17,.14,.09,.05,.10,.14,.11,.07,.03]),
      "Swimming Pool":([6,7,8,10,12,15,16,17,18,19,20],[.07,.09,.08,.06,.05,.08,.10,.13,.15,.12,.07]),
      "Badminton Court":([7,8,9,16,17,18,19,20,21],[.04,.05,.05,.08,.12,.17,.20,.19,.10]),
      "Tennis Court":([7,8,9,16,17,18,19,20,21],[.05,.05,.05,.09,.13,.17,.19,.17,.10]),
      "Clubhouse":([10,11,12,15,16,17,18,19,20,21],[.06,.06,.05,.06,.08,.13,.18,.17,.13,.08]),
      "Multipurpose Hall":([9,10,11,12,14,15,16,17,18,19,20],[.06,.08,.08,.07,.08,.09,.11,.14,.13,.10,.06])}
    rows=[]
    # Allocate event counts approximately evenly, with a realistic sparse tail.
    counts=rng.poisson(lam=n_bookings/n_residents,size=n_residents)
    counts=np.maximum(counts,8)
    while counts.sum()<n_bookings: counts[int(rng.integers(n_residents))]+=1
    while counts.sum()>n_bookings:
        candidates=np.where(counts>8)[0]
        if len(candidates)==0: break
        counts[int(rng.choice(candidates))]-=1
    for ri,r in enumerate(residents):
        n=int(counts[ri])
        # Resident activity begins at a random point; inter-use intervals produce meaningful recurrence.
        current=start+pd.Timedelta(days=int(rng.integers(0,180)))
        for j in range(n):
            if j:
                gap=int(np.clip(rng.gamma(shape=2.2,scale=5.0),3,28))
                current += pd.Timedelta(days=gap)
            if current>end.normalize(): break
            p=prefs[r].copy()
            if current>=pd.Timestamp("2025-07-01"): p[1]*=1.18; p[4]*=1.10
            if current.month in [4,5,6,7,8]: p[1]*=1.30
            if current.weekday()>=5: p[4]*=1.25; p[5]*=1.25
            if current>=pd.Timestamp("2026-01-01") and rng.random()<.12: p=np.roll(p,int(rng.integers(1,len(p))))
            p/=p.sum()
            facility=FACILITIES[int(rng.choice(len(FACILITIES),p=p))]
            # Choose usage weekday using facility and resident-level preferences.
            dp=np.array(day_probs[facility],dtype=float)
            dp[weekday_pref[r]]*=1.65
            if current>=pd.Timestamp("2025-07-01"): dp[5:]*=1.05
            dp/=dp.sum()
            usage_weekday=int(rng.choice(7,p=dp))
            usage_date=current+pd.Timedelta(days=(usage_weekday-current.weekday())%7)
            if j and usage_date <= pd.Timestamp(rows[-1]["usage_timestamp"]) if rows and rows[-1]["resident_ref"]==r else False:
                usage_date=current+pd.Timedelta(days=7)
            hrs,probs=hours[facility]; probs=np.array(probs,dtype=float)
            probs[np.argmin(np.abs(np.array(hrs)-hour_pref[r]))]*=1.8
            probs/=probs.sum()
            hour=int(rng.choice(hrs,p=probs))
            minute=int(rng.choice([0,15,30,45],p=[.48,.18,.24,.10]))
            usage=usage_date+pd.Timedelta(hours=hour,minutes=minute)
            lead=lead_base[r]*rng.lognormal(0,.42)*{"Gym":.75,"Swimming Pool":1.0,"Badminton Court":1.1,"Tennis Court":1.2,"Clubhouse":2.5,"Multipurpose Hall":2.9}[facility]
            if rng.random()<.035: lead*=rng.uniform(3,6)
            lead=float(np.clip(lead,1,24*14))
            booking=usage-pd.Timedelta(hours=lead)
            if rng.random()<.045:
                usage+=pd.Timedelta(hours=int(rng.choice([-2,-1,1,2,3])))
                if usage<=booking: usage=booking+pd.Timedelta(hours=1)
            rows.append({"resident_ref":r,"facility_ref":facility,"booking_timestamp":booking,"usage_timestamp":usage})
    df=pd.DataFrame(rows)
    # Enforce exact requested size if small count clipping changed the total.
    if len(df)>n_bookings: df=df.sample(n=n_bookings,random_state=seed).copy()
    df=df.sort_values(["booking_timestamp","resident_ref"]).reset_index(drop=True)
    df.insert(0,"record_ref",[f"B-{i+1:06d}" for i in range(len(df))])
    OUT.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUT,index=False,date_format="%Y-%m-%d %H:%M:%S")
    return df

if __name__=="__main__":
    d=generate_dataset()
    print(f"Wrote {len(d):,} bookings across {d.resident_ref.nunique()} residents to {OUT}")
