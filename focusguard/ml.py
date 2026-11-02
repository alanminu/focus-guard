"""Optional learned layer + evaluation utilities.

The rule engine is annotation-free. When labelled data *is* available (your
own protocol recordings, or a public dataset) this module learns a
window-level classifier on top of the same interpretable attributes, and -
crucially - evaluates it with **subject-disjoint** cross-validation.

Random frame-level 80/20 splits (used by the first prototype) put
near-identical neighbouring frames of the same person in train and test, so
they measure memorisation, not generalisation.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .behavior import BehaviorEngine
from .calibration import Baseline
from .config import Config


# ------------------------------------------------------------------ helpers
def _runs(mask: np.ndarray, t: np.ndarray) -> List[float]:
    """Durations (s) of consecutive True runs."""
    if mask.size == 0:
        return []
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.033
    return [float(t[e] - t[s] + dt) for s, e in zip(starts, ends)]


def window_features(X: np.ndarray, t: np.ndarray, names: Sequence[str]) -> Dict[str, float]:
    """Interpretable temporal statistics of one window / clip."""
    names = list(names)
    col = lambda k: X[:, names.index(k)]
    face = np.isfinite(col("ear"))
    out: Dict[str, float] = {"face_missing_frac": float(1 - face.mean())}
    if face.sum() < 3:
        return out
    Xf, tf = X[face], t[face]
    c = lambda k: Xf[:, names.index(k)]
    ear, mar, jaw = c("ear"), c("mar"), c("jaw_open")
    ear_open = np.nanpercentile(ear, 90)
    for k, v in {"ear": ear, "mar": mar, "jaw_open": jaw, "pitch": c("pitch"), "yaw": c("yaw"),
                 "roll": c("roll"), "blink_bs": c("blink_bs"), "bs_eye_pitch": c("bs_eye_pitch"),
                 "bs_eye_yaw": c("bs_eye_yaw"), "ear_l": c("ear_l"), "ear_r": c("ear_r")}.items():
        out[f"{k}_mean"] = float(np.nanmean(v))
        out[f"{k}_std"] = float(np.nanstd(v))
        out[f"{k}_p10"] = float(np.nanpercentile(v, 10))
        out[f"{k}_p90"] = float(np.nanpercentile(v, 90))
    closed_rel = ear < 0.65 * ear_open            # relative-to-self closure (calibration-free)
    closed_abs = ear < 0.18
    out["closed_rel_frac"] = float(closed_rel.mean())
    out["closed_abs_frac"] = float(closed_abs.mean())
    out["blinkbs_frac"] = float((c("blink_bs") > 0.5).mean())
    r = _runs(closed_rel, tf)
    out["closure_max_s"] = max(r) if r else 0.0
    out["closure_n"] = float(len(r))
    out["long_closure_n"] = float(sum(x >= 1.0 for x in r))
    wide = (mar > 0.45) | (jaw > 0.45)
    rw = _runs(wide, tf)
    out["mouth_wide_frac"] = float(wide.mean())
    out["mouth_wide_max_s"] = max(rw) if rw else 0.0
    dp = np.diff(c("pitch"))
    out["pitch_vel_std"] = float(np.nanstd(dp)) if dp.size else 0.0
    out["nod_n"] = float(np.sum(dp < -3.0))
    out["ear_open_p90"] = float(ear_open)
    return out


V1_FEATURES = ["ear_l_mean", "ear_l_std", "ear_l_p10", "ear_l_p90", "mar_mean", "mar_std", "mar_p10", "mar_p90"]


def rule_features(X: np.ndarray, t: np.ndarray, names: Sequence[str], cfg: Optional[Config] = None,
                  self_calibrate: bool = True) -> Dict[str, float]:
    """Run the annotation-free rule engine over a clip and summarise its outputs.

    With ``self_calibrate`` the open-eye EAR / resting mouth are estimated from the
    clip itself (robust percentiles) - the offline analogue of the live calibration.
    """
    cfg = cfg or Config.for_context("driving")
    names = list(names)
    base = Baseline.population_default(cfg)
    if self_calibrate:
        ear = X[:, names.index("ear")]
        if np.isfinite(ear).sum() > 5:
            base.median["ear"] = float(np.nanpercentile(ear, 90))
            base.median["mar"] = float(np.nanpercentile(X[:, names.index("mar")], 20))
            for k in ("yaw", "pitch", "bs_eye_yaw", "bs_eye_pitch"):
                base.median[k] = float(np.nanmedian(X[:, names.index(k)]))
        base.calibrated = True
    cfg.adaptive_baseline = False
    eng = BehaviorEngine(cfg, base)
    states = []
    for row, ti in zip(X, t):
        f = None if not np.isfinite(row[names.index("ear")]) else dict(zip(names, map(float, row)))
        states.append(eng.update(float(ti), f))
    st = np.array([s.state for s in states])
    ev = pd.DataFrame(eng.events)
    cnt = lambda k: float((ev.type == k).sum()) if len(ev) else 0.0
    return {
        "rule_drowsy_frac": float((st == "DROWSY").mean()),
        "rule_distracted_frac": float((st == "DISTRACTED").mean()),
        "rule_perclos": float(states[-1].metrics.get("perclos", 0.0)) if states else 0.0,
        "rule_long_closures": cnt("long_eye_closure"),
        "rule_yawns": cnt("yawn"),
        "rule_blinks": cnt("blink"),
        "rule_glances": cnt("glance"),
        "rule_mean_score": float(np.mean([s.score for s in states])) if states else 100.0,
    }


# ------------------------------------------------------------------ evaluation
def cv_evaluate(X: np.ndarray, y: np.ndarray, model_factory: Callable, splitter, groups=None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    accs, baccs, f1s, aucs = [], [], [], []
    oof = np.zeros(len(y), dtype=float)
    for tr, te in splitter.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else m.predict(X[te])
        oof[te] = p
        yh = (p >= 0.5).astype(int)
        accs.append(accuracy_score(y[te], yh))
        baccs.append(balanced_accuracy_score(y[te], yh))
        f1s.append(f1_score(y[te], yh, zero_division=0))
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te], p))
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)), "bal_acc": float(np.mean(baccs)),
            "f1": float(np.mean(f1s)), "auc": float(np.mean(aucs)) if aucs else float("nan"), "oof": oof}


def default_models(seed: int = 0) -> Dict[str, Callable]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "LogReg": lambda: make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "RandomForest": lambda: make_pipeline(SimpleImputer(), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)),
        "HistGB": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                         max_leaf_nodes=15, random_state=seed),
    }
