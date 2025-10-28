import os, numpy as np, pandas as pd
from .io_utils import ensure_dir
try:
    from lightgbm import LGBMRanker
except Exception:
    LGBMRanker = None
try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None
try:
    import joblib
except Exception:
    joblib = None

def label_chunks_from_annotations(df: pd.DataFrame, ann_csv: str) -> pd.DataFrame:
    ann = pd.read_csv(ann_csv)
    req = {"video_id","start_sec","end_sec"}
    if not (set(ann.columns) >= req):
        raise RuntimeError("annotations.csv must have columns: video_id,start_sec,end_sec")
    def iou(a_s,a_e,b_s,b_e):
        inter = max(0.0, min(a_e,b_e) - max(a_s,b_s))
        union = (a_e-a_s) + (b_e-b_s) - inter
        return inter/union if union>0 else 0.0
    labels = []
    for _, r in df.iterrows():
        v = ann[ann.video_id == r["video_id"]]
        lbl = 0
        for _, a in v.iterrows():
            if iou(r["start"], r["end"], float(a["start_sec"]), float(a["end_sec"])) >= 0.5:
                lbl = 1; break
        labels.append(lbl)
    df = df.copy(); df["label"] = labels; return df

def train_quick_model(features_dir: str, ann_csv: str, model_path: str):
    if LogisticRegression is None or joblib is None:
        raise RuntimeError("Install scikit-learn and joblib: pip install scikit-learn joblib")
    files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.csv')]
    if not files: raise RuntimeError("No feature CSVs found. Run prepare first.")
    dfs = [pd.read_csv(f) for f in files]; X = pd.concat(dfs, ignore_index=True)
    X = label_chunks_from_annotations(X, ann_csv)
    feat_cols = ['motion','audio','cap_overlap','kw_density']; y = X['label'].values.astype(np.int32)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X[feat_cols].values, y)
    ensure_dir(os.path.dirname(model_path)); joblib.dump(clf, model_path); print(f'✅ Quick model saved to {model_path}')

def train_ranker(features_dir: str, ann_csv: str, model_path: str):
    if LGBMRanker is None or joblib is None:
        raise RuntimeError("Install lightgbm and joblib: pip install lightgbm joblib")
    files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.csv')]
    if not files: raise RuntimeError("No feature CSVs found. Run prepare first.")
    dfs = [pd.read_csv(f) for f in files]; X = pd.concat(dfs, ignore_index=True)
    X = label_chunks_from_annotations(X, ann_csv)
    grp_sizes = X.groupby('video_id').size().values
    feat_cols = ['motion','audio','cap_overlap','kw_density']; y = X['label'].values.astype(np.float32)
    ranker = LGBMRanker(objective='lambdarank', metric='ndcg', n_estimators=300, learning_rate=0.05)
    ranker.fit(X[feat_cols], y, group=grp_sizes)
    ensure_dir(os.path.dirname(model_path)); joblib.dump(ranker, model_path); print(f'✅ Model saved to {model_path}')

def predict_scores_with_model(chunks, model_path: str) -> bool:
    if joblib is None or not os.path.exists(model_path): return False
    model = joblib.load(model_path)
    feats = np.array([[c.motion, c.audio, c.cap_overlap, c.kw_density] for c in chunks], dtype=np.float32)
    scores = None
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(feats)
            if proba.ndim == 2 and proba.shape[1] >= 2: scores = proba[:,1]
            else: scores = proba.ravel()
        except Exception: scores = None
    if scores is None and hasattr(model, 'decision_function'):
        try: scores = model.decision_function(feats)
        except Exception: scores = None
    if scores is None and hasattr(model, 'predict'):
        try: scores = model.predict(feats)
        except Exception: scores = None
    if scores is None: return False
    scores = np.asarray(scores, dtype=np.float32)
    mn, mx = float(np.min(scores)), float(np.max(scores))
    p = (scores - mn) / (mx - mn + 1e-12)
    for i, c in enumerate(chunks): c.score = float(p[i])
    return True

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _rank_metrics_per_video(labels, scores, ks=(5,10)):
    import numpy as np
    labels = np.asarray(labels, dtype=np.int32); scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores); y = labels[order]; npos = int(y.sum()); res = {}
    for K in ks:
        K = int(min(K, len(y))); topk = y[:K]; tp = int(topk.sum())
        prec = tp / max(1, K); rec = tp / max(1, npos) if npos>0 else 0.0
        f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        gains = (2**topk - 1) / np.log2(np.arange(2, K+2)); dcg = gains.sum()
        ideal = np.sort(labels)[::-1][:K]; idcg = ((2**ideal - 1) / np.log2(np.arange(2, len(ideal)+2))).sum()
        ndcg = (dcg / idcg) if idcg>0 else 0.0
        res[f'P@{K}'] = prec; res[f'R@{K}'] = rec; res[f'F1@{K}'] = f1; res[f'NDCG@{K}'] = ndcg
    ap = 0.0
    if npos>0:
        hits=0; precs=[]; 
        for i, rel in enumerate(y, start=1):
            if rel==1: hits+=1; precs.append(hits/i)
        ap = float(np.mean(precs)) if precs else 0.0
    res['MAP']=ap; res['positives']=npos; res['total']=len(y); return res

def evaluate_model(features_dir: str, annotations_csv: str, model_path: str, out_csv: str, out_png: str, ks=(5,10)) -> dict:
    if joblib is None: raise RuntimeError("joblib needed for evaluation.")
    if not os.path.exists(model_path): raise RuntimeError(f"Model not found at {model_path}.")
    files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.csv')]
    if not files: raise RuntimeError("No feature CSVs found. Run prepare first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = label_chunks_from_annotations(df, annotations_csv)
    feat_cols = ['motion','audio','cap_overlap','kw_density']
    model = joblib.load(model_path); df['pred'] = model.predict(df[feat_cols].values)
    vids = sorted(df['video_id'].unique().tolist())
    rows=[]; agg={f'P@{k}':0.0 for k in ks}; agg.update({f'R@{k}':0.0 for k in ks}); agg.update({f'F1@{k}':0.0 for k in ks}); agg.update({f'NDCG@{k}':0.0 for k in ks}); agg['MAP']=0.0
    n=0
    for vid in vids:
        sub=df[df.video_id==vid]; m=_rank_metrics_per_video(sub['label'].values, sub['pred'].values, ks=ks)
        row={'video_id':vid}; row.update(m); rows.append(row)
        for k in ks: agg[f'P@{k}']+=m[f'P@{k}']; agg[f'R@{k}']+=m[f'R@{k}']; agg[f'F1@{k}']+=m[f'F1@{k}']; agg[f'NDCG@{k}']+=m[f'NDCG@{k}']
        agg['MAP']+=m['MAP']; n+=1
    if n>0:
        for k in ks: agg[f'P@{k}']/=n; agg[f'R@{k}']/=n; agg[f'F1@{k}']/=n; agg[f'NDCG@{k}']/=n
        agg['MAP']/=n
    import os
    os.makedirs(os.path.dirname(out_csv), exist_ok=True); pd.DataFrame(rows).to_csv(out_csv, index=False)
    labels=[*(f'P@{k}' for k in ks), *(f'R@{k}' for k in ks), *(f'F1@{k}' for k in ks), *(f'NDCG@{k}' for k in ks), 'MAP']
    values=[agg[x] for x in labels]
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4)); plt.bar(labels, values); plt.ylim(0,1.0); plt.title('Ranking Model Evaluation'); plt.ylabel('Score'); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True); plt.savefig(out_png); plt.close()
    return {'per_video_csv': out_csv, 'chart_png': out_png, 'averages': agg}
