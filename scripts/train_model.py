"""Train + evaluate the optional learned layer on protocol recordings.

Reports three evaluation protocols side by side so the leakage problem of the
first prototype is visible in your own numbers:

  (1) V1 replica - per-frame RandomForest on [left EAR, lip distance], random 80/20 frame split
  (2) same model, leave-one-subject-out (LOSO)
  (3) FocusGuard window features (3 s windows, 1 s hop) + rule outputs, LOSO
  (4) FocusGuard rule engine alone (no training at all)

  python scripts/train_model.py --data data/protocol --out results/protocol
"""
import argparse, glob, json, os, sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focusguard.features import FEATURE_NAMES  # noqa: E402
from focusguard.ml import cv_evaluate, default_models, window_features  # noqa: E402

LABEL = {"focused": 0, "glance": 0, "distracted": 1, "drowsy": 1}


def load(data_dir):
    dfs = [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(data_dir, "*.csv")))]
    if not dfs:
        raise SystemExit(f"no protocol CSVs in {data_dir}")
    df = pd.concat(dfs, ignore_index=True)
    df = df[df.label != "transition"].copy()
    df["y"] = df.label.map(LABEL)
    return df


def windows(df, win=3.0, hop=1.0):
    rows = []
    for (subj, sess), g in df.groupby(["subject", "session"]):
        t = g.t.values
        X = g[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").values
        for s in np.arange(t[0], t[-1] - win, hop):
            m = (t >= s) & (t < s + win)
            if m.sum() < 10:
                continue
            lab = g.y.values[m]
            if (lab == lab[0]).mean() < 0.8:          # skip windows spanning two instructions
                continue
            f = window_features(X[m], t[m], FEATURE_NAMES)
            f["rule_bad_frac"] = float(np.isin(g.rule_state.values[m], ["DISTRACTED", "DROWSY"]).mean())
            f.update(subject=subj, y=int(round(lab.mean())))
            rows.append(f)
    return pd.DataFrame(rows)


def main():
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, train_test_split
    from sklearn.ensemble import RandomForestClassifier
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/protocol")
    ap.add_argument("--out", default="results/protocol")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load(a.data)
    subjects = df.subject.nunique()
    print(f"{len(df)} frames, {subjects} subjects")
    res = {}

    # (1) V1 replica with random frame split
    fr = df.dropna(subset=["ear_l", "mar"])
    Xv, yv, gv = fr[["ear_l", "mar"]].values, fr.y.values, fr.subject.values
    Xtr, Xte, ytr, yte = train_test_split(Xv, yv, test_size=0.2, random_state=0, stratify=yv)
    rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xtr, ytr)
    res["V1 per-frame RF, random 80/20"] = {"acc": float((rf.predict(Xte) == yte).mean())}
    # (2) V1 replica, LOSO
    if subjects > 1:
        r = cv_evaluate(Xv, yv, lambda: RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
                        LeaveOneGroupOut(), gv)
        res["V1 per-frame RF, LOSO"] = {k: v for k, v in r.items() if k != "oof"}

    # (3) window features
    W = windows(df)
    feat_cols = [c for c in W.columns if c not in ("subject", "y")]
    X, y, g = W[feat_cols].values.astype(float), W.y.values, W.subject.values
    split = LeaveOneGroupOut() if subjects > 1 else StratifiedKFold(5, shuffle=True, random_state=0)
    best, best_auc = None, -1
    for name, fac in default_models().items():
        r = cv_evaluate(X, y, fac, split, g)
        res[f"FocusGuard windows + {name}, {'LOSO' if subjects > 1 else '5-fold'}"] = {k: v for k, v in r.items() if k != "oof"}
        if r["auc"] > best_auc:
            best, best_auc = name, r["auc"]
    # (4) rules only
    yh = (W.rule_bad_frac.values > 0.5).astype(int)
    res["FocusGuard rules only (no training)"] = {"acc": float((yh == y).mean())}

    for k, v in res.items():
        print(f"{k:55s} " + "  ".join(f"{m}={x:.3f}" for m, x in v.items() if isinstance(x, float)))
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    model = default_models()[best]()
    model.fit(X, y)
    joblib.dump({"model": model, "features": feat_cols, "window_s": 3.0}, os.path.join(a.out, "window_model.joblib"))
    print("saved model:", best)


if __name__ == "__main__":
    main()
