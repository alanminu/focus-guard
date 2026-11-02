"""Evaluation on SUST-DDD (real-drive drowsiness, 19 drivers, 2,074 x 10 s clips).

Requires features from scripts/extract_features.py. Produces:
  results/sust/clip_features.csv      one row per clip
  results/sust/results.json/.csv      all metrics
  results/sust/*.png                  figures used in the paper

Protocols
  random-4fold  : stratified 4-fold on clips, as in the SUST-DDD paper (clips of the
                  same driver end up in both train and test)
  grouped-4fold : StratifiedGroupKFold on *pseudo-subjects*. The dataset does not
                  publish driver IDs, so clips are clustered (default 12 groups for 19 drivers,
                  erring towards merging drivers rather than splitting them) from
                  (a) the mean colour thumbnail of the scene (each driver used their own
                  car and phone) and (b) mean aligned face geometry. This approximates a
                  subject-disjoint split; see the cluster montage for a visual check.
"""
import argparse, glob, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from focusguard.config import Config  # noqa: E402
from focusguard.ml import V1_FEATURES, cv_evaluate, default_models, rule_features, window_features  # noqa: E402


def load(feat_dir):
    rows, meta = [], []
    files = sorted(glob.glob(os.path.join(feat_dir, "*.npz")))
    for p in files:
        z = np.load(p, allow_pickle=True)
        X, t, names = z["X"], z["t"], list(z["names"])
        y = 1 if str(z["label"]).startswith("drows") else 0
        f = window_features(X, t, names)
        f.update(rule_features(X, t, names, Config.for_context("driving")))
        f.update(clip=str(z["clip"]), y=y)
        rows.append(f)
        meta.append({"clip": str(z["clip"]), "scene": z["scene"], "shape": z["shape"],
                     "w": float(z["size"][0]), "h": float(z["size"][1]), "fps": float(z["fps"]),
                     "face_rate": float(np.isfinite(X[:, 0]).mean())})
    return pd.DataFrame(rows), meta


def pseudo_subjects(meta, k=19, seed=0):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    S = np.stack([m["scene"] for m in meta]) / 255.0
    F = np.stack([m["shape"] for m in meta])
    A = np.c_[np.array([[m["w"], m["h"], m["fps"]] for m in meta])]
    Zs = PCA(10, random_state=seed).fit_transform(StandardScaler().fit_transform(S))
    Zf = PCA(6, random_state=seed).fit_transform(StandardScaler().fit_transform(F))
    Za = StandardScaler().fit_transform(A)
    Z = np.c_[Zs, 0.7 * Zf, 0.5 * Za]
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)


def montage(meta, groups, video_root, path, per=4):
    import cv2
    tiles = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0][:per]
        row = []
        for i in idx:
            clip = meta[i]["clip"]
            cand = glob.glob(os.path.join(video_root, "**", clip + ".mp4"), recursive=True)
            img = np.zeros((96, 72, 3), np.uint8)
            if cand:
                cap = cv2.VideoCapture(cand[0])
                cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
                ok, f = cap.read()
                if ok:
                    img = cv2.resize(f, (72, 96))
            row.append(img)
        while len(row) < per:
            row.append(np.zeros((96, 72, 3), np.uint8))
        r = np.hstack(row)
        cv2.putText(r, str(g), (2, 14), 0, 0.45, (0, 255, 255), 1)
        tiles.append(r)
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(path, grid)


def main():
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
    from sklearn.ensemble import RandomForestClassifier
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/sust")
    ap.add_argument("--videos", default=None, help="SUST-DDD root (only for the cluster montage)")
    ap.add_argument("--out", default="results/sust")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--groups", type=int, default=19, help="number of pseudo-subject clusters")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df, meta = load(a.features)
    groups = pseudo_subjects(meta, k=a.groups)
    df["group"] = groups
    df["face_rate"] = [m["face_rate"] for m in meta]
    df.to_csv(os.path.join(a.out, "clip_features.csv"), index=False)
    y = df.y.values
    print(f"{len(df)} clips  drowsy={y.sum()}  groups={len(np.unique(groups))}  "
          f"face detected in {df.face_rate.mean()*100:.1f}% of frames")
    if a.videos:
        montage(meta, groups, a.videos, os.path.join(a.out, "pseudo_subject_montage.jpg"))

    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    all_feat = [c for c in df.columns if c not in ("clip", "y", "group", "face_rate")]
    eye = [c for c in all_feat if c.startswith(("ear", "closed", "closure", "long_closure", "blink"))]
    mouth = [c for c in all_feat if c.startswith(("mar", "jaw", "mouth"))]
    pose = [c for c in all_feat if c.startswith(("pitch", "yaw", "roll", "nod", "bs_eye"))]
    sets = {
        "V1 (left EAR + lip)": V1_FEATURES,
        "Eyes": eye,
        "Eyes + mouth": eye + mouth,
        "Eyes + mouth + head/gaze": eye + mouth + pose,
        "Full (+ rule-engine outputs)": all_feat,
    }
    results = []
    for seed in range(a.seeds):
        splits = {"random-4fold": (StratifiedKFold(4, shuffle=True, random_state=seed), None),
                  "grouped-4fold": (StratifiedGroupKFold(4, shuffle=True, random_state=seed), groups)}
        for sname, cols in sets.items():
            X = df[cols].values.astype(float)
            for mname, fac in default_models(seed).items():
                for pname, (sp, g) in splits.items():
                    r = cv_evaluate(X, y, fac, sp, g)
                    results.append({"features": sname, "model": mname, "protocol": pname, "seed": seed,
                                    **{k: v for k, v in r.items() if k != "oof"}})
        print(f"seed {seed} done", flush=True)
    R = pd.DataFrame(results)
    agg = R.groupby(["features", "model", "protocol"], sort=False)[["acc", "bal_acc", "f1", "auc"]].mean().reset_index()
    agg.to_csv(os.path.join(a.out, "results_models.csv"), index=False)

    # ---- annotation-free rule engine (no training at all) ----
    rule = {}
    for col in ("rule_drowsy_frac", "rule_long_closures", "rule_perclos"):
        rule[col + "_auc"] = float(roc_auc_score(y, df[col]))
    yh = (df.rule_drowsy_frac > 0).astype(int)
    rule["rule_any_drowsy_acc"] = float(accuracy_score(y, yh))
    rule["rule_any_drowsy_bal_acc"] = float(balanced_accuracy_score(y, yh))
    inv = -df.rule_mean_score
    rule["rule_attention_index_auc"] = float(roc_auc_score(y, inv))

    # ---- V1 replica: per-frame RF with random frame split (the prototype's protocol) ----
    frames = []
    for p, (i, m) in zip(sorted(glob.glob(os.path.join(a.features, "*.npz"))), enumerate(meta)):
        z = np.load(p, allow_pickle=True)
        names = list(z["names"])
        Xf = z["X"][:, [names.index("ear_l"), names.index("mar")]]
        ok = np.isfinite(Xf).all(1)
        frames.append(pd.DataFrame({"ear_l": Xf[ok, 0], "mar": Xf[ok, 1], "y": df.y.iloc[i], "clip": i,
                                    "group": groups[i]}))
    FR = pd.concat(frames, ignore_index=True)
    v1 = {}
    tr, te = train_test_split(np.arange(len(FR)), test_size=0.2, random_state=0, stratify=FR.y)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0).fit(FR.loc[tr, ["ear_l", "mar"]], FR.y[tr])
    v1["frame_random_80_20_acc"] = float((rf.predict(FR.loc[te, ["ear_l", "mar"]]) == FR.y[te]).mean())
    v1["frame_random_80_20_train_acc"] = float((rf.predict(FR.loc[tr, ["ear_l", "mar"]]) == FR.y[tr]).mean())
    gk = StratifiedGroupKFold(4, shuffle=True, random_state=0)
    accs = []
    for trg, teg in gk.split(FR[["ear_l", "mar"]], FR.y, FR.group):
        m = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, min_samples_leaf=5)
        m.fit(FR.loc[trg, ["ear_l", "mar"]], FR.y[trg])
        accs.append((m.predict(FR.loc[teg, ["ear_l", "mar"]]) == FR.y[teg]).mean())
    v1["frame_grouped_4fold_acc"] = float(np.mean(accs))
    out = {"n_clips": int(len(df)), "n_drowsy": int(y.sum()), "face_detect_rate": float(df.face_rate.mean()),
           "group_sizes": np.bincount(groups).tolist(), "rules_no_training": rule, "v1_replica": v1,
           "sust_paper_baseline": {"VGG19+LSTM_random_4fold_acc": 0.9053}}
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    best = agg.sort_values(["protocol", "acc"]).groupby("protocol").tail(3)
    print(best.to_string())

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hg = agg[agg.model == "HistGB"].set_index(["features", "protocol"]).acc.unstack()
    hg = hg.loc[list(sets.keys())]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    x = np.arange(len(hg))
    ax.bar(x - 0.2, hg["random-4fold"] * 100, 0.4, label="random 4-fold (leaky)", color="#9fb7d5")
    ax.bar(x + 0.2, hg["grouped-4fold"] * 100, 0.4, label="pseudo-subject 4-fold", color="#1f4e79")
    for i, (r, g_) in enumerate(zip(hg["random-4fold"], hg["grouped-4fold"])):
        ax.text(i - 0.2, r * 100 + 0.8, f"{r*100:.1f}", ha="center", fontsize=7.5)
        ax.text(i + 0.2, g_ * 100 + 0.8, f"{g_*100:.1f}", ha="center", fontsize=7.5)
    ax.axhline(90.53, ls="--", color="#d9534f", lw=1)
    ax.text(len(hg) - 0.5, 91.3, "SUST-DDD paper VGG19+LSTM (random 4-fold) 90.5%", ha="right", fontsize=7.5, color="#d9534f")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" (", "\n(") for s in hg.index], fontsize=8)
    ax.set_ylabel("clip accuracy (%)")
    ax.set_ylim(40, 100)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("SUST-DDD drowsiness: feature groups x evaluation protocol (HistGB)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "sust_protocols.png"), dpi=160)
    plt.close(fig)

    # permutation importance of the full model (grouped protocol, one fold-free fit for ranking)
    from sklearn.inspection import permutation_importance
    X = df[all_feat].values.astype(float)
    m = default_models(0)["HistGB"]()
    sgk = StratifiedGroupKFold(4, shuffle=True, random_state=0)
    imps = []
    for trn, tst in sgk.split(X, y, groups):
        m.fit(X[trn], y[trn])
        imps.append(permutation_importance(m, X[tst], y[tst], n_repeats=5, random_state=0).importances_mean)
    imp = pd.Series(np.mean(imps, 0), index=all_feat).sort_values(ascending=False)
    imp.to_csv(os.path.join(a.out, "permutation_importance.csv"))
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(top.index, top.values * 100, color="#1f4e79")
    ax.set_xlabel("accuracy drop when shuffled (pp), held-out pseudo-subjects")
    ax.set_title("Most informative attributes (SUST-DDD)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "sust_importance.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
