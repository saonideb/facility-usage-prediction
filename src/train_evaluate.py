from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"facility_bookings.csv"; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
FACILITIES=["Gym","Swimming Pool","Badminton Court","Tennis Court","Clubhouse","Multipurpose Hall"]
SEED=42

def make_examples(events):
    events=events.sort_values(["booking_timestamp","record_ref"]).reset_index(drop=True)
    examples=[]
    for resident,group in events.groupby("resident_ref",sort=False):
        group=group.sort_values(["booking_timestamp","record_ref"]).reset_index(drop=True)
        hist=[]
        for i in range(len(group)-1):
            anchor=group.iloc[i]; target=group.iloc[i+1]
            bt=pd.Timestamp(anchor.booking_timestamp)
            # History is strictly through the anchor event; target is never used in features.
            hist_plus=hist+[{"facility":anchor.facility_ref,"booking":bt,"usage":pd.Timestamp(anchor.usage_timestamp),
                "lead":max(.25,(pd.Timestamp(anchor.usage_timestamp)-bt).total_seconds()/3600)}]
            fs=[x["facility"] for x in hist_plus]; us=[x["usage"] for x in hist_plus]; ls=[x["lead"] for x in hist_plus]
            counts={f:fs.count(f) for f in FACILITIES}
            top=max(FACILITIES,key=lambda f:counts[f]) if hist_plus else "UNKNOWN"
            features={
              "example_ref":f"{anchor.record_ref}__NEXT","resident_ref":resident,"anchor_facility":anchor.facility_ref,
              "anchor_booking_timestamp":bt,"anchor_booking_hour":bt.hour,"anchor_booking_weekday":bt.weekday(),
              "anchor_booking_month":bt.month,"anchor_weekend":int(bt.weekday()>=5),
              "history_count":len(hist_plus),"history_unique_facilities":len(set(fs)),"history_top_facility":top,
              "history_avg_lead_hours":float(np.mean(ls)),"history_avg_usage_hour":float(np.mean([u.hour for u in us])),
              "history_avg_usage_weekday":float(np.mean([u.weekday() for u in us])),
              "history_mode_usage_weekday":int(pd.Series([u.weekday() for u in us]).mode().iloc[0]),
              "history_mode_usage_hour":int(pd.Series([u.hour for u in us]).mode().iloc[0]),
              "days_since_previous_booking":(bt-hist_plus[-2]["booking"]).total_seconds()/86400 if len(hist_plus)>1 else np.nan,
              "history_avg_booking_gap_days":float(np.mean([(hist_plus[k]["booking"]-hist_plus[k-1]["booking"]).total_seconds()/86400 for k in range(1,len(hist_plus))])) if len(hist_plus)>1 else np.nan,
              "history_median_booking_gap_days":float(np.median([(hist_plus[k]["booking"]-hist_plus[k-1]["booking"]).total_seconds()/86400 for k in range(1,len(hist_plus))])) if len(hist_plus)>1 else np.nan,
              "target_booking_timestamp":pd.Timestamp(target.booking_timestamp),"target_usage_timestamp":pd.Timestamp(target.usage_timestamp),
              "target_facility":target.facility_ref,"target_usage_weekday":pd.Timestamp(target.usage_timestamp).weekday(),
              "target_usage_hour":pd.Timestamp(target.usage_timestamp).hour,
              "target_lead_hours":max(.25,(pd.Timestamp(target.usage_timestamp)-pd.Timestamp(target.booking_timestamp)).total_seconds()/3600),
              "target_days_to_usage":max(.1,(pd.Timestamp(target.usage_timestamp)-bt).total_seconds()/86400),
              "target_days_to_booking":max(.01,(pd.Timestamp(target.booking_timestamp)-bt).total_seconds()/86400)
            }
            for f in FACILITIES: features[f"resident_count__{f.lower().replace(' ','_')}"]=counts[f]
            examples.append(features)
            hist.append({"facility":anchor.facility_ref,"booking":bt,"usage":pd.Timestamp(anchor.usage_timestamp),
                         "lead":max(.25,(pd.Timestamp(anchor.usage_timestamp)-bt).total_seconds()/3600)})
    return pd.DataFrame(examples)

def align_weekday_hour(base_ts,weekday,hour):
    base_ts=pd.Timestamp(base_ts)
    # Use the predicted continuous offset for date scale; then select the nearest date with predicted weekday.
    candidate=base_ts.normalize()+pd.Timedelta(days=max(1,int(round((base_ts-pd.Timestamp(base_ts.normalize())).total_seconds()/86400))))
    delta=(int(weekday)-candidate.weekday())%7
    if delta>3: delta-=7
    candidate=candidate+pd.Timedelta(days=delta)
    if candidate<=base_ts.normalize(): candidate+=pd.Timedelta(days=7)
    return candidate+pd.Timedelta(hours=int(hour))

def main():
    events=pd.read_csv(DATA,parse_dates=["booking_timestamp","usage_timestamp"])
    examples=make_examples(events)
    cutoff=examples.target_booking_timestamp.quantile(.80)
    tr=examples[examples.target_booking_timestamp<cutoff].copy()
    te=examples[examples.target_booking_timestamp>=cutoff].copy()
    cat=["resident_ref","anchor_facility","history_top_facility"]
    excluded={"example_ref","anchor_booking_timestamp","target_booking_timestamp","target_usage_timestamp","target_facility",
      "target_usage_weekday","target_usage_hour","target_lead_hours","target_days_to_usage","target_days_to_booking"}
    feature_cols=[c for c in examples.columns if c not in excluded]
    nums=[c for c in feature_cols if c not in cat]
    def prep():
        return ColumnTransformer([
          ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore"))]),cat),
          ("num",Pipeline([("imputer",SimpleImputer(strategy="median"))]),nums)])
    def clf():
        return Pipeline([("prep",prep()),("model",ExtraTreesClassifier(n_estimators=90,min_samples_leaf=3,class_weight="balanced",max_features=.8,random_state=SEED,n_jobs=-1))])
    def clf_unbalanced():
        return Pipeline([("prep",prep()),("model",ExtraTreesClassifier(n_estimators=90,min_samples_leaf=3,class_weight=None,max_features=.8,random_state=SEED,n_jobs=-1))])
    def reg():
        return Pipeline([("prep",prep()),("model",ExtraTreesRegressor(n_estimators=70,min_samples_leaf=4,max_features=.8,random_state=SEED,n_jobs=-1))])
    models={"facility":clf(),"weekday":clf_unbalanced(),"hour":clf_unbalanced(),"days_to_usage":reg(),"days_to_booking":reg()}
    Xtr,Xte=tr[feature_cols],te[feature_cols]
    ys={"facility":tr.target_facility,"weekday":tr.target_usage_weekday.astype(str),"hour":tr.target_usage_hour.astype(str),
        "days_to_usage":tr.target_days_to_usage.clip(.1,60),"days_to_booking":tr.target_days_to_booking.clip(.01,60)}
    for k,m in models.items(): m.fit(Xtr,ys[k])
    pf=models["facility"].predict(Xte); day=models["weekday"].predict(Xte).astype(int); hour=models["hour"].predict(Xte).astype(int)
    # Personalized fallback: for established residents, their historical mode is a strong low-variance baseline.
    established=te.history_count.to_numpy()>=3
    day=np.where(established,te.history_mode_usage_weekday.to_numpy(dtype=int),day)
    hour=np.where(established,te.history_mode_usage_hour.to_numpy(dtype=int),hour)
    offset=np.clip(models["days_to_usage"].predict(Xte),.1,60)
    days_to_booking=np.clip(models["days_to_booking"].predict(Xte),.01,60)
    pred_usage=[]
    for bt,off,d,h in zip(te.anchor_booking_timestamp,offset,day,hour):
        rough=pd.Timestamp(bt)+pd.Timedelta(days=float(off))
        # Shift to the nearest occurrence of the predicted weekday and then set the predicted hour.
        delta=(int(d)-rough.weekday())%7
        if delta>3: delta-=7
        adjusted=rough+pd.Timedelta(days=delta)
        adjusted=adjusted.normalize()+pd.Timedelta(hours=int(h))
        if adjusted<=pd.Timestamp(bt): adjusted+=pd.Timedelta(days=7)
        pred_usage.append(adjusted)
    pred_nudge=[pd.Timestamp(bt)+pd.Timedelta(days=float(d)) for bt,d in zip(te.anchor_booking_timestamp,days_to_booking)]
    review=te[["example_ref","resident_ref","history_count","anchor_booking_timestamp","anchor_facility","target_booking_timestamp","target_usage_timestamp",
      "target_facility","target_usage_weekday","target_usage_hour","target_lead_hours"]].copy()
    review=review.rename(columns={"target_booking_timestamp":"actual_booking_timestamp","target_usage_timestamp":"actual_usage_timestamp",
      "target_facility":"actual_facility","target_usage_weekday":"actual_usage_weekday","target_usage_hour":"actual_usage_hour"})
    review["predicted_facility"]=pf; review["predicted_usage_weekday"]=day; review["predicted_usage_hour"]=hour
    review["predicted_usage_timestamp"]=pred_usage; review["predicted_notification_timestamp"]=pred_nudge
    review["actual_notification_timestamp_proxy"]=review.actual_booking_timestamp
    review["facility_match"]=review.predicted_facility==review.actual_facility
    review["weekday_match"]=review.predicted_usage_weekday==review.actual_usage_weekday
    review["hour_match"]=review.predicted_usage_hour==review.actual_usage_hour
    review["notification_error_hours"]=(pd.to_datetime(review.predicted_notification_timestamp)-review.actual_notification_timestamp_proxy).dt.total_seconds()/3600
    review["notification_within_24h"]=review.notification_error_hours.abs()<=24
    review["notification_within_72h"]=review.notification_error_hours.abs()<=72
    mcols=["facility_match","weekday_match","hour_match","notification_within_24h"]
    review["outputs_matched"]=review[mcols].sum(axis=1); review["match_summary"]=review.outputs_matched.astype(str)+" of 4"
    review["all_outputs_match"]=review.outputs_matched==4
    # Transparent majority/median baseline fitted only on training labels.
    mode_fac=tr.target_facility.mode().iloc[0]; mode_day=int(tr.target_usage_weekday.mode().iloc[0]); mode_hour=int(tr.target_usage_hour.mode().iloc[0])
    med_offset=float(tr.target_days_to_usage.median()); med_booking=float(tr.target_days_to_booking.median())
    base_fac_acc=float((te.target_facility==mode_fac).mean()); base_day_acc=float((te.target_usage_weekday==mode_day).mean())
    base_hour_acc=float((te.target_usage_hour==mode_hour).mean())
    base_usages=[]
    for bt in te.anchor_booking_timestamp:
        rough=pd.Timestamp(bt)+pd.Timedelta(days=med_offset)
        delta=(mode_day-rough.weekday())%7
        if delta>3: delta-=7
        u=(rough+pd.Timedelta(days=delta)).normalize()+pd.Timedelta(hours=mode_hour)
        if u<=pd.Timestamp(bt): u+=pd.Timedelta(days=7)
        base_usages.append(u)
    base_nudges=[pd.Timestamp(bt)+pd.Timedelta(days=med_booking) for bt in te.anchor_booking_timestamp]
    base_notif=float((np.abs(pd.Series(base_nudges).astype("int64")/3.6e12-te.target_booking_timestamp.astype("int64")/3.6e12)<=24).mean())
    metrics=[
      ("Facility","Accuracy",accuracy_score(review.actual_facility,review.predicted_facility),"Exact next-facility match"),
      ("Facility","Macro F1",f1_score(review.actual_facility,review.predicted_facility,average="macro",zero_division=0),"Balances facility classes"),
      ("Usage weekday","Accuracy",accuracy_score(review.actual_usage_weekday,review.predicted_usage_weekday),"Exact weekday match"),
      ("Usage weekday","Macro F1",f1_score(review.actual_usage_weekday.astype(str),review.predicted_usage_weekday.astype(str),average="macro",zero_division=0),"Balances weekday classes"),
      ("Usage hour","Accuracy",accuracy_score(review.actual_usage_hour,review.predicted_usage_hour),"Exact hour match"),
      ("Usage hour","Macro F1",f1_score(review.actual_usage_hour.astype(str),review.predicted_usage_hour.astype(str),average="macro",zero_division=0),"Balances hour classes"),
      ("Notification time","MAE (hours)",mean_absolute_error(review.actual_notification_timestamp_proxy.astype("int64")/3.6e12,pd.to_datetime(review.predicted_notification_timestamp).astype("int64")/3.6e12),"Against booking timestamp proxy, not observed send time"),
      ("Notification time","Within 24h",review.notification_within_24h.mean(),"Absolute timestamp error <= 24h"),
      ("Notification time","Within 72h",review.notification_within_72h.mean(),"Absolute timestamp error <= 72h"),
      ("Overall","Mean output score",(review.facility_match.mean()+review.weekday_match.mean()+review.hour_match.mean()+review.notification_within_24h.mean())/4,"Mean of three exact-match rates and notification within-24h rate"),
      ("Overall","All four criteria match",review.all_outputs_match.mean(),"All three labels exact and notification within 24h"),
      ("Baseline","Facility accuracy",base_fac_acc,"Training majority-class baseline"),
      ("Baseline","Weekday accuracy",base_day_acc,"Training majority-class baseline"),
      ("Baseline","Hour accuracy",base_hour_acc,"Training majority-class baseline"),
      ("Baseline","Notification within 24h",base_notif,"Median usage offset/lead and mode weekday/hour baseline")]
    mdf=pd.DataFrame(metrics,columns=["output","metric","value","interpretation"])
    review.to_csv(OUT/"prediction_review.csv",index=False,date_format="%Y-%m-%d %H:%M:%S"); mdf.to_csv(OUT/"metrics.csv",index=False)
    # Small, auditable error-analysis tables by target facility, weekday, and history depth.
    facility_err=review.groupby("actual_facility").agg(records=("example_ref","count"),facility_accuracy=("facility_match","mean"),
        weekday_accuracy=("weekday_match","mean"),hour_accuracy=("hour_match","mean"),notification_mae_hours=("notification_error_hours",lambda x: float(np.mean(np.abs(x))))).reset_index()
    facility_err.insert(0,"group_type","actual_facility"); facility_err=facility_err.rename(columns={"actual_facility":"group"})
    weekday_err=review.groupby("actual_usage_weekday").agg(records=("example_ref","count"),weekday_accuracy=("weekday_match","mean"),
        hour_accuracy=("hour_match","mean"),notification_mae_hours=("notification_error_hours",lambda x: float(np.mean(np.abs(x))))).reset_index()
    weekday_err.insert(0,"group_type","actual_weekday"); weekday_err=weekday_err.rename(columns={"actual_usage_weekday":"group"})
    history_err=review.assign(history_band=pd.cut(review.history_count,[-1,1,3,7,10000],labels=["1","2-3","4-7","8+"])).groupby("history_band",observed=False).agg(
        records=("example_ref","count"),
        facility_accuracy=("facility_match","mean"),
        weekday_accuracy=("weekday_match","mean"),
        hour_accuracy=("hour_match","mean"),
        notification_mae_hours=("notification_error_hours",lambda x: float(np.mean(np.abs(x))))
    ).reset_index()
    history_err.insert(0,"group_type","history_depth"); history_err=history_err.rename(columns={"history_band":"group"})
    pd.concat([facility_err,weekday_err,history_err],ignore_index=True,sort=False).to_csv(OUT/"error_analysis.csv",index=False)
    joblib.dump(models,OUT/"models.joblib",compress=3)
    metadata={"seed":SEED,"source_events":len(events),"resident_count":int(events.resident_ref.nunique()),"supervised_examples":len(examples),
      "train_examples":len(tr),"holdout_examples":len(te),"target_booking_cutoff":str(cutoff),"features":feature_cols,
      "models":"ExtraTrees classifiers for facility/weekday/hour; regressors for days-to-usage and days-to-booking",
      "target_definition":"Predict next resident booking from anchor event and prior resident history",
      "leakage_controls":["Target event excluded from all features","Chronological split by target booking timestamp","Preprocessors fitted on training rows only"],
      "notification_evaluation":"Actual booking timestamp is a proxy, not observed notification-send time","python_version":sys.version.split()[0]}
    (OUT/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    with pd.ExcelWriter(OUT/"prediction_review.xlsx",engine="openpyxl") as writer:
      review.to_excel(writer,sheet_name="Prediction Review",index=False); mdf.to_excel(writer,sheet_name="Metrics Summary",index=False)
      events.head(100).to_excel(writer,sheet_name="Dataset Sample",index=False); pd.read_csv(OUT/"error_analysis.csv").to_excel(writer,sheet_name="Error Analysis",index=False)
      from openpyxl.styles import Font,PatternFill,Alignment
      from openpyxl.utils import get_column_letter
      for ws in writer.book.worksheets:
        ws.freeze_panes="A2"; ws.auto_filter.ref=ws.dimensions
        for c in ws[1]: c.font=Font(bold=True,color="FFFFFF"); c.fill=PatternFill("solid",fgColor="1F4E78"); c.alignment=Alignment(wrap_text=True,vertical="top")
        for col in ws.columns: ws.column_dimensions[get_column_letter(col[0].column)].width=min(max(max(len(str(c.value or "")) for c in list(col)[:250])+2,12),38)
    print(f"Events: {len(events):,}; next-event examples: {len(examples):,}; train: {len(tr):,}; holdout: {len(te):,}")
    print(mdf.to_string(index=False))
if __name__=="__main__": main()
