"""ESANN attribution drift development and frozen study engine, protocol 1.0.

All monitoring functions take features and model outputs. Labels enter training,
source quality checks, and the separate evaluation layer. Run roots are immutable
with respect to configuration and engine hashes; complete units can be resumed.
"""
from __future__ import annotations

import dataclasses
import concurrent.futures
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ks_2samp, rankdata, wasserstein_distance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

VERSION = "2.0-path-a-plus"
DATASETS = ("sea", "hyperplane", "adult_numeric", "electricity", "interaction")
MODELS = ("xgboost", "mlp")
STUDY_SEEDS = (11, 29, 47, 71, 101)
PINNED = {"numpy": "2.2.6", "scipy": "1.15.3", "pandas": "2.2.3",
          "scikit-learn": "1.6.1", "matplotlib": "3.10.1", "xgboost": "3.0.5",
          "shap": "0.49.1", "river": "0.22.0"}


@dataclasses.dataclass
class Config:
    stage: str = "development"
    datasets: tuple = DATASETS
    models: tuple = MODELS
    seeds: tuple = (11,)
    train_n: int = 4096
    gate_n: int = 384
    reference_n: int = 256
    calibration_n: int = 1536
    stream_n: int = 1200
    background_n: int = 64
    lime_n: int = 500
    lime_sigma: float = .5
    lime_ridge: float = .001
    ig_steps: int = 64
    epochs: int = 30
    model_batch: int = 256
    explain_batch: int = 32
    xgb_trees: int = 120
    xgb_depth: int = 3
    xgb_subsample: float = .8
    xgb_colsample: float = .8
    learning_rate: float = .001
    windows: tuple = (50,100,250)
    primary_window: int = 100
    bootstrap_n: int = 199
    block_length: int = 20
    alpha: float = .05
    review_budgets: tuple = (.1,.2)
    false_negative_cost: float = 5.
    false_positive_cost: float = 1.
    js_weight: float = .5
    top_k: int = 3
    adwin_delta: float = .002
    data_seed: int = 20261110
    explanation_seed: int = 801
    monitor_seed: int = 1701
    interaction_rho: float = .95
    boundary_strength: float = 1.5
    threads: int = 2
    workers: int = 4

    @classmethod
    def study(cls):
        return cls(stage="confirmation",seeds=STUDY_SEEDS,stream_n=2400)

    def validate(self):
        assert self.stage in ("development","confirmation","smoke")
        assert self.primary_window in self.windows
        assert self.calibration_n%3==0 and self.calibration_n//3>=max(self.windows)
        assert self.stream_n>=4*max(self.windows)
        assert self.lime_n>=20 and self.ig_steps>=8
        assert set(self.datasets)<=set(DATASETS) and set(self.models)<=set(MODELS)
        assert self.workers>=1 and self.threads>=1
        if self.stage=="confirmation":
            assert tuple(self.datasets)==DATASETS and tuple(self.models)==MODELS
            assert tuple(self.seeds)==STUDY_SEEDS and self.core()==Config().core()
            assert (self.train_n,self.gate_n,self.reference_n,self.calibration_n,self.stream_n)==(4096,384,256,1536,2400)

    def core(self):
        omit={"stage","seeds","train_n","gate_n","reference_n","calibration_n","stream_n","threads","explain_batch","workers"}
        return {k:v for k,v in dataclasses.asdict(self).items() if k not in omit}


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def encoded(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(plain(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def stable_seed(*parts):
    return int(digest(parts)[:8], 16)


def seed_all(seed, threads=2):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def download(url, path):
    path = Path(path)
    if path.exists() and path.stat().st_size > 100:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "ESANNResearch/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, open(temp, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    assert temp.stat().st_size > 100, f"Download size check: {url}"
    temp.replace(path)
    return path


@dataclasses.dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    onset: int | None = None
    transition_end: int | None = None
    role: str = "source"
    eta: np.ndarray | None = None


@dataclasses.dataclass
class Bundle:
    name: str
    feature_names: list
    source: dict
    streams: dict
    provenance: dict


def _source_synthetic_bundle(name, cfg, seed):
    stage_key = "development" if cfg.stage == "development" else cfg.stage
    rng = np.random.default_rng(stable_seed("data", name, stage_key, seed))
    d = 3 if name == "sea" else 10
    w = np.array([1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

    def X_source(n):
        X = rng.uniform(0, 1, (n, d))
        if name == "hyperplane":
            X[:, 1] = np.clip(X[:, 0] + rng.normal(0, 0.06, n), 0, 1)
        return X.astype("float32")

    def labels(X, alternate=0.0, flips=None):
        if name == "sea":
            y = (10 * (X[:, 0] + X[:, 1]) > (8.0 + 4.0 * alternate)).astype(int)
        else:
            new_w = w.copy()
            new_w[:4] *= -1
            mix_w = (1 - np.asarray(alternate)[..., None]) * w + np.asarray(alternate)[..., None] * new_w
            y = (((X - 0.5) * mix_w).sum(axis=1) > 0).astype(int)
        if flips is None:
            flips = rng.random(len(X)) < 0.05
        return np.where(flips, 1 - y, y).astype("int64")

    source = {}
    counts = {"train": cfg.train_n, "gate": cfg.gate_n,
              "reference": cfg.reference_n, "calibration": cfg.calibration_n}
    for role, n in counts.items():
        X = X_source(n)
        source[role] = Split(X, labels(X), np.array([f"{name}:{stage_key}:{seed}:{role}:{i}" for i in range(n)]))
    n = cfg.stream_n
    onset = n // 2
    end = onset if name == "sea" else 3 * n // 4
    X = X_source(n)
    flips = rng.random(n) < 0.05
    y = labels(X, flips=flips)
    ids = np.array([f"{name}:{stage_key}:{seed}:stream:{i}" for i in range(n)])
    severity = np.clip((np.arange(n) - onset + 1) / max(1, end - onset), 0, 1)
    if end == onset:
        severity = (np.arange(n) >= onset).astype(float)
    changed = X.copy()
    if name == "sea":
        changed[:, 0] = (1 - severity) * X[:, 0] + severity * np.sqrt(X[:, 0])
        shift_name = "covariate_abrupt"
    else:
        target = np.clip(1 - X[:, 0] + rng.normal(0, 0.06, n), 0, 1)
        changed[:, 1] = (1 - severity) * X[:, 1] + severity * target
        shift_name = "covariate_correlation_gradual"
    streams = {
        "stationary": Split(X, y, ids, role="stationary_control"),
        shift_name: Split(changed, labels(changed, flips=flips), ids, onset, end, "controlled_covariate"),
        "label_only": Split(X.copy(), labels(X, severity, flips), ids.copy(), onset, end, "label_only_control"),
    }
    return Bundle(name, [f"x{i}" for i in range(d)], source, streams,
                  {"kind": "synthetic", "generator": "ESANN controlled SEA / hyperplane variants",
                   "seed": seed, "stage": cfg.stage, "label_noise": 0.05,
                   "paired_stream_ids": True, "transition": [onset, end],
                   "description": "The hyperplane covariate arm changes feature correlation; the label-only arm rotates its label rule."})


def adult_bundle(cfg,seed,cache):
    url="https://archive.ics.uci.edu/static/public/2/adult.zip"
    path=download(url,Path(cache)/"adult.zip")
    columns=["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native_country","income"]
    feats=["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
    with zipfile.ZipFile(path) as z:
        frames=[]
        for member in ("adult.data","adult.test"):
            df=pd.read_csv(z.open(member),names=columns,skipinitialspace=True,comment="|",na_values="?")
            df["income"]=df.income.astype(str).str.rstrip(".").str.strip()
            df["id"]=[f"{member}:{i}" for i in range(len(df))]
            df["row_hash"]=[hashlib.sha256(s.encode()).hexdigest() for s in df[columns].fillna("?").astype(str).agg("|".join,axis=1)]
            df["official"]=member;frames.append(df)
    merged=pd.concat(frames,ignore_index=True)
    before=len(merged);merged=merged.drop_duplicates("row_hash",keep="first")
    source_pool=merged.loc[merged.official=="adult.data"].sort_values("row_hash")
    pool=source_pool.iloc[::2] if cfg.stage=="development" else source_pool.iloc[1::2]
    rng=np.random.default_rng(stable_seed("adult_v2",cfg.stage,cfg.data_seed))
    pool=pool.iloc[rng.permutation(len(pool))]
    manifest,manifest_sha=revision_manifest()
    excluded=set(manifest["adult_excluded_deployment_ids"])
    source={};offset=0
    def split(df):
        return Split(df[feats].to_numpy("float32"),(df.income==">50K").to_numpy("int64"),df.id.to_numpy())
    for role,n in (("train",cfg.train_n),("gate",cfg.gate_n),("reference",cfg.reference_n),("calibration",cfg.calibration_n)):
        source[role]=split(pool.iloc[offset:offset+n]);offset+=n
    deployment=pool.iloc[offset:] if cfg.stage=="development" else merged.loc[merged.official=="adult.test"]
    deployment=deployment.loc[~deployment.id.isin(excluded)]
    n=cfg.stream_n
    assert len(deployment)>=n,("Unused Adult deployment observations",len(deployment),n)
    stationary=split(deployment.iloc[rng.choice(len(deployment),n,replace=False)])
    pre=rng.choice(len(deployment),n//2,replace=False)
    remaining=np.setdiff1d(np.arange(len(deployment)),pre)
    mean=source["train"].X.mean(0);sd=source["train"].X.std(0)+1e-6
    Z=(deployment[feats].to_numpy()-mean)/sd
    logits=np.clip(1.2*Z[remaining,0]+.8*Z[remaining,2],-8,8)
    weight=np.exp(logits-logits.max())
    post=rng.choice(remaining,n-n//2,replace=False,p=weight/weight.sum())
    shifted=split(deployment.iloc[np.r_[pre,post]])
    shifted.onset=shifted.transition_end=n//2;shifted.role="controlled_covariate_selection"
    stationary.role="stationary_control"
    return Bundle("adult_numeric",feats,source,{"stationary":stationary,"covariate_selection":shifted},
                  {"url":url,"sha256":file_hash(path),"original_rows":before,"removed_full_row_duplicates":before-len(merged),
                   "protocol":VERSION,"data_seed":cfg.data_seed,"source_pool":"even official-training hash order for development; odd for confirmation",
                   "unused_deployment_pool_n":len(deployment),"previous_observed_deployment_ids_excluded":True,
                   "revision_data_manifest_sha256":manifest_sha,"feature_subset":feats,
                   "selection_rule":"without replacement exp(clip(1.2*z_age+0.8*z_education,-8,8))"})


def electricity_bundle(cfg, seed, cache):
    os.environ["RIVER_DATA"] = str(Path(cache).resolve() / "river")
    from river import datasets
    data = datasets.Elec2()
    rows = []
    for i, (x, y) in enumerate(data):
        rows.append({**x, "__y": int(y), "__id": i})
    df = pd.DataFrame(rows)
    features = [c for c in df.columns if c not in ("date", "__y", "__id")]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="raise")
    start = 8000 if cfg.stage == "development" else 30000
    source = {}
    offset = start
    for role, n in (("train", cfg.train_n), ("gate", cfg.gate_n), ("reference", cfg.reference_n), ("calibration", cfg.calibration_n)):
        part = df.iloc[offset:offset+n]
        assert len(part) == n
        source[role] = Split(part[features].to_numpy("float32"), part.__y.to_numpy("int64"), np.array([f"elec2:{i}" for i in part.__id]))
        offset += n
    part = df.iloc[offset:offset+cfg.stream_n]
    assert len(part) == cfg.stream_n
    stream = Split(part[features].to_numpy("float32"), part.__y.to_numpy("int64"), np.array([f"elec2:{i}" for i in part.__id]), role="natural_temporal")
    data_path = Path(data.path)
    return Bundle("electricity", features, source, {"natural": stream},
                  {"provider": "river.datasets.Elec2 0.22.0", "url": str(getattr(data, "url", "River Elec2")),
                   "sha256": file_hash(data_path), "original_rows": len(df), "range_start": start,
                   "deployment_range": [offset, offset + cfg.stream_n], "order": "original chronological",
                   "features": features, "date_feature": "excluded; chronological position is retained in identifiers",
                   "onset_annotation": "natural chronology"})


def make_bundle(name,cfg,seed,cache):
    if name in ("sea","hyperplane"):
        return synthetic_bundle(name,cfg,seed)
    if name=="interaction":
        return interaction_bundle(cfg,seed)
    if name=="adult_numeric":
        return adult_bundle(cfg,seed,cache)
    if name=="electricity":
        return electricity_bundle(cfg,cfg.data_seed,cache)
    raise ValueError(name)


def validate_bundle(bundle):
    seen = set()
    counts = {}
    d = len(bundle.feature_names)
    for role, split in bundle.source.items():
        assert len(split.X) == len(split.y) == len(split.ids)
        assert split.X.shape[1] == d and np.isfinite(split.X).all()
        assert len(set(split.ids)) == len(split.ids)
        assert seen.isdisjoint(set(split.ids)), f"Split overlap in {role}"
        seen.update(split.ids)
        counts[role] = len(split.X)
    for name, split in bundle.streams.items():
        assert seen.isdisjoint(set(split.ids)), f"Source/deployment overlap in {name}"
        assert len(set(split.ids)) == len(split.ids)
        assert np.isfinite(split.X).all()
        counts[name] = len(split.X)
    # Paired scenarios deliberately share underlying rows; source roles stay disjoint.
    hashes = {role: digest(sorted(s.ids.tolist())) for role, s in bundle.source.items()}
    return {"role_counts": counts, "source_id_hashes": hashes, "all_source_roles_disjoint": True,
            "source_deployment_disjoint": True}


class Predictor:
    def __init__(self, kind, cfg, seed):
        self.kind, self.cfg, self.seed = kind, cfg, seed
        self.scaler = StandardScaler()
        self.device = "cpu"
        self.training_log = []

    def fit(self, X, y):
        import torch
        seed_all(self.seed, self.cfg.threads)
        Z = self.scaler.fit_transform(X).astype("float32")
        if self.kind == "xgboost":
            from xgboost import XGBClassifier
            self.model = XGBClassifier(n_estimators=self.cfg.xgb_trees, max_depth=self.cfg.xgb_depth,
                                       learning_rate=0.05, subsample=self.cfg.xgb_subsample, colsample_bytree=self.cfg.xgb_colsample,
                                       reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
                                       tree_method="hist", n_jobs=self.cfg.threads, random_state=self.seed)
            self.model.fit(Z, y)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 64), torch.nn.SiLU(),
                                             torch.nn.Linear(64, 64), torch.nn.SiLU(),
                                             torch.nn.Linear(64, 32), torch.nn.SiLU(), torch.nn.Linear(32, 1)).to(self.device)
            opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=0.0001)
            tx = torch.tensor(Z, device=self.device)
            ty = torch.tensor(y.astype("float32"), device=self.device)
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            self.model.train()
            for epoch in range(self.cfg.epochs):
                order = torch.randperm(len(Z), generator=generator, device=self.device)
                epoch_loss = 0.0
                for idx in order.split(self.cfg.model_batch):
                    opt.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(self.model(tx[idx]).flatten(), ty[idx])
                    loss.backward()
                    opt.step()
                    epoch_loss += float(loss.detach()) * len(idx)
                self.training_log.append({"epoch": epoch + 1, "loss": epoch_loss / len(Z)})
            self.model.eval()
        return self

    def transform(self, X):
        return self.scaler.transform(X).astype("float32")

    def margin(self, Z):
        if self.kind == "xgboost":
            return self.model.predict(Z, output_margin=True).astype(float)
        import torch
        values = []
        with torch.no_grad():
            for start in range(0, len(Z), 8192):
                tx = torch.as_tensor(Z[start:start+8192], dtype=torch.float32, device=self.device)
                values.append(self.model(tx).flatten().cpu().numpy())
        return np.concatenate(values).astype(float)

    def save(self, folder):
        import torch
        folder = Path(folder)
        if self.kind == "xgboost":
            self.model.save_model(folder / "model.ubj")
        else:
            torch.save({"state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                        "input_dim": len(self.scaler.mean_), "hidden": [64, 64, 32], "activation": "SiLU"}, folder / "model.pt")
            self.model.to(self.device)
        write_json(folder / "scaler.json", {"mean": self.scaler.mean_, "scale": self.scaler.scale_, "var": self.scaler.var_})
        pd.DataFrame(self.training_log).to_csv(folder / "training.csv", index=False)


def probability_simplex(phi):
    v = np.abs(np.asarray(phi, dtype=float))
    total = v.sum(axis=-1, keepdims=True)
    return np.divide(v, total, out=np.full_like(v, 1 / v.shape[-1]), where=total > 1e-12)


def disagreement(A, B, cfg):
    p, q = probability_simplex(A), probability_simplex(B)
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        js = 0.5 * (np.where(p > 0, p * np.log2(np.maximum(p, 1e-300) / np.maximum(m, 1e-300)), 0).sum(1) +
                    np.where(q > 0, q * np.log2(np.maximum(q, 1e-300) / np.maximum(m, 1e-300)), 0).sum(1))
    gaps = []
    for a, b in zip(np.abs(A), np.abs(B)):
        ra = rankdata(-a, method="average")
        rb = rankdata(-b, method="average")
        k = min(cfg.top_k, len(a))
        ia = np.lexsort((np.arange(len(a)), -a))[:k]
        ib = np.lexsort((np.arange(len(b)), -b))[:k]
        union = np.union1d(ia, ib)
        num = den = 0.0
        for offset, i in enumerate(union):
            for j in union[offset+1:]:
                sa, sb = np.sign(ra[i] - ra[j]), np.sign(rb[i] - rb[j])
                gap = 0.0 if sa == sb else 0.5 if sa == 0 or sb == 0 else 1.0
                weight = 1.0 / (min(ra[i], rb[i]) * min(ra[j], rb[j]))
                num += weight * gap
                den += weight
        gaps.append(num / den if den else 0.0)
    rank = np.asarray(gaps)
    js = np.clip(js, 0, 1)
    return cfg.js_weight * js + (1 - cfg.js_weight) * rank, js, rank


class Explainers:
    def __init__(self, predictor, trainZ, cfg, seed):
        self.p, self.cfg, self.seed = predictor, cfg, cfg.explanation_seed
        seed = cfg.explanation_seed
        rng = np.random.default_rng(stable_seed("background", seed))
        self.background = trainZ[rng.choice(len(trainZ), min(cfg.background_n, len(trainZ)), replace=False)]
        self.baseline = self.background.mean(0)
        self.reference_output = float(self.p.margin(self.background).mean())
        if self.p.kind == "xgboost":
            import shap
            self.tree = shap.TreeExplainer(self.p.model, data=self.background, model_output="raw", feature_perturbation="interventional")
            self.reference_output = float(np.asarray(self.tree.expected_value).reshape(-1)[0])
        else:
            self.reference_output = float(self.p.margin(self.baseline[None])[0])

    def primary(self, Z):
        if self.p.kind == "xgboost":
            return np.asarray(self.tree.shap_values(Z, check_additivity=False), dtype=float)
        import torch
        nodes, weights = np.polynomial.legendre.leggauss(self.cfg.ig_steps)
        alphas = torch.tensor((nodes + 1) / 2, dtype=torch.float32, device=self.p.device)
        weights = torch.tensor(weights / 2, dtype=torch.float32, device=self.p.device)
        base = torch.tensor(self.baseline, device=self.p.device)
        tx = torch.tensor(Z, device=self.p.device)
        points = (base[None, None, :] + alphas[None, :, None] * (tx - base)[:, None, :]).reshape(-1, Z.shape[1])
        points.requires_grad_(True)
        output = self.p.model(points).sum()
        gradient = torch.autograd.grad(output, points)[0].reshape(len(Z), self.cfg.ig_steps, -1)
        integrated = (gradient * weights[None, :, None]).sum(1) * (tx - base)
        return integrated.detach().cpu().numpy().astype(float)

    def local(self, Z, ids, repetition=0):
        n, d = Z.shape
        count = self.cfg.lime_n
        deltas = []
        for row_id in ids:
            rng = np.random.default_rng(stable_seed("lime", self.seed, str(row_id), repetition))
            half = rng.normal(0, self.cfg.lime_sigma, (math.ceil((count - 1) / 2), d))
            delta = np.concatenate([np.zeros((1, d)), half, -half], axis=0)[:count]
            deltas.append(delta)
        delta = np.stack(deltas)
        query = (Z[:, None, :] + delta).reshape(-1, d).astype("float32")
        output = self.p.margin(query).reshape(n, count)
        design = np.concatenate([np.ones((n, count, 1)), delta], axis=2)
        weight = np.exp(-np.sum(delta**2, axis=2) / (d * 0.75**2))
        lhs = np.einsum("bni,bn,bnj->bij", design, weight, design)
        penalty = np.eye(d+1) * self.cfg.lime_ridge * count
        penalty[0, 0] = 0
        lhs += penalty[None]
        rhs = np.einsum("bni,bn,bn->bi", design, weight, output)
        coef = np.linalg.solve(lhs, rhs[..., None]).squeeze(-1)
        fit = np.einsum("bni,bi->bn", design, coef)
        average = np.sum(weight * output, axis=1) / weight.sum(1)
        denominator = np.sum(weight * (output - average[:, None])**2, axis=1)
        numerator = np.sum(weight * (output - fit)**2, axis=1)
        r2 = np.where(denominator > 1e-12, 1 - numerator / np.maximum(denominator, 1e-12), 1.0)
        # Contributions to the same class-1 logit, measured from background mean.
        contribution = coef[:, 1:] * (Z - self.baseline)
        return contribution, r2

    def explain(self, Z, ids, progress=None):
        As, Bs, B2s, r2s, residuals = [], [], [], [], []
        begin = time.perf_counter()
        primary_s = local_s = 0.0
        for start in range(0, len(Z), self.cfg.explain_batch):
            batch = Z[start:start+self.cfg.explain_batch]
            batch_ids = ids[start:start+self.cfg.explain_batch]
            t = time.perf_counter()
            a = self.primary(batch)
            primary_s += time.perf_counter() - t
            t = time.perf_counter()
            b, r2 = self.local(batch, batch_ids)
            b2, _ = self.local(batch, batch_ids, repetition=1)
            local_s += time.perf_counter() - t
            margin = self.p.margin(batch)
            residual = np.abs(a.sum(1) - (margin - self.reference_output)) / np.maximum(1.0, np.abs(margin - self.reference_output))
            As.append(a); Bs.append(b); B2s.append(b2); r2s.append(r2); residuals.append(residual)
            if progress and start and start % (self.cfg.explain_batch * 50) == 0:
                progress(f"  explanations {start}/{len(Z)}")
        A, B, B2 = map(np.concatenate, (As, Bs, B2s))
        D, js, rank = disagreement(A, B, self.cfg)
        noise, _, _ = disagreement(B, B2, self.cfg)
        margin = self.p.margin(Z)
        p = np.clip(expit(margin), 1e-12, 1 - 1e-12)
        msp = np.maximum(p, 1-p)
        entropy = -(p * np.log2(p) + (1-p) * np.log2(1-p))
        _, _, cost = expected_cost_components(p, self.cfg)
        # Canonical symmetric binary logits (-margin/2, margin/2).
        energy = -np.logaddexp(-margin/2, margin/2)
        scalar = np.column_stack([D, js, rank, noise, entropy, 1-msp, energy, cost, np.sqrt((Z**2).mean(1))])
        return {"X": Z, "A": A, "B": B, "B2": B2, "scalar": scalar, "margin": margin,
                "p": p, "lime_r2": np.concatenate(r2s), "residual": np.concatenate(residuals),
                "timing": {"n": len(Z), "total_seconds": time.perf_counter() - begin,
                           "primary_seconds": primary_s, "local_two_repetitions_seconds": local_s,
                           "local_query_rows": int(len(Z) * self.cfg.lime_n * 2),
                           "ig_gradient_rows": int(len(Z) * self.cfg.ig_steps) if self.p.kind == "mlp" else 0}}


SCALAR_NAMES = ("disagreement", "js_only", "rank_only", "lime_noise", "entropy", "msp_uncertainty", "energy", "expected_cost", "input_distance")
DETECTOR_NAMES = SCALAR_NAMES + ("single_A", "single_B", "signed_difference", "input_KS", "input_RFF_MMD")


def subset_data(data, idx):
    return {key: value[idx] if isinstance(value, np.ndarray) and len(value) == len(data["X"]) else value
            for key, value in data.items()}


def block_indices(n, size, block, rng):
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, math.ceil(size / block))
    return np.concatenate([np.arange(i, i+block) for i in starts])[:size]


class Detector:
    def __init__(self,reference,calibration,cfg,seed):
        self.reference,self.cfg=reference,cfg
        seed=cfg.monitor_seed
        rng=np.random.default_rng(stable_seed("rff",seed))
        X=reference["X"]
        distance=np.sqrt(((X[:128,None,:]-X[None,:128,:])**2).sum(-1))
        positive=distance[distance>1e-12]
        bandwidth=float(np.median(positive)) if len(positive) else 1.
        self.rff_w=rng.normal(size=(X.shape[1],64))/bandwidth
        self.rff_b=rng.uniform(0,2*np.pi,64)
        self.ref_rff=self.features(X).mean(0)
        self.thresholds,self.bootstrap={},{}
        for window in cfg.windows:
            rng=np.random.default_rng(stable_seed("bootstrap",seed,window))
            samples=np.stack([self.statistics(calibration,block_indices(len(calibration["X"]),window,cfg.block_length,rng)) for _ in range(cfg.bootstrap_n)])
            self.bootstrap[window]=samples
            self.thresholds[window]=np.quantile(samples,1-cfg.alpha,axis=0,method="higher")
    def features(self,X):
        return math.sqrt(2/64)*np.cos(X@self.rff_w+self.rff_b)
    def statistics(self,data,idx):
        ref=self.reference
        values=[wasserstein_distance(ref["scalar"][:,j],data["scalar"][idx,j]) for j in range(len(SCALAR_NAMES))]
        # Exactly the same operator and dimension for A, B and their signed difference.
        for key in ("A","B","signed_difference"):
            reference=ref["A"]-ref["B"] if key=="signed_difference" else ref[key]
            current=data["A"][idx]-data["B"][idx] if key=="signed_difference" else data[key][idx]
            values.append(np.mean([wasserstein_distance(reference[:,j],current[:,j]) for j in range(reference.shape[1])]))
        values.append(max(ks_2samp(ref["X"][:,j],data["X"][idx,j]).statistic for j in range(ref["X"].shape[1])))
        values.append(float(np.sum((self.features(data["X"][idx]).mean(0)-self.ref_rff)**2)))
        return np.asarray(values)
    def monitor(self,data,stream_name,onset=None,transition_end=None):
        records=[];gamma_by_window={}
        reference_mean=float(self.reference["scalar"][:,0].mean())
        for window in self.cfg.windows:
            stride=max(1,window//4)
            two=np.zeros(len(data["X"]));directional=two.copy()
            previous_end=0;previous_two=previous_dir=0.
            for end in range(window,len(data["X"])+1,stride):
                two[previous_end:end-1]=previous_two;directional[previous_end:end-1]=previous_dir
                start=end-window
                scores=self.statistics(data,np.arange(start,end));thresholds=self.thresholds[window]
                mean=float(data["scalar"][start:end,0].mean())
                d,q=scores[0],thresholds[0]
                g=float(np.clip(1-q/max(d,1e-12),0,1)) if d>q else 0.
                gd=g if mean>reference_mean else 0.
                two[end-1]=g;directional[end-1]=gd
                previous_end,previous_two,previous_dir=end,g,gd
                phase="natural"
                if onset is not None:
                    phase="pre" if end<=onset else "post" if start>=transition_end else "transition"
                elif stream_name=="stationary":
                    phase="stationary"
                for j,method in enumerate(DETECTOR_NAMES):
                    records.append({"stream":stream_name,"window":window,"start":start,"end":end,"phase":phase,
                                    "method":method,"statistic":scores[j],"threshold":thresholds[j],"alarm":bool(scores[j]>thresholds[j]),
                                    "mean_disagreement":mean,"reference_mean_disagreement":reference_mean,
                                    "gamma_two_sided":g,"gamma_directional":gd})
            two[previous_end:]=previous_two;directional[previous_end:]=previous_dir
            gamma_by_window[window]={"two_sided":two,"directional":directional}
        return pd.DataFrame(records),gamma_by_window


def adwin_trace(reference, stream, cfg):
    from river.drift import ADWIN
    detector = ADWIN(delta=cfg.adwin_delta, clock=1, min_window_length=5, grace_period=20)
    for val in 1 - reference["scalar"][:, 5]:
        detector.update(float(val))
    flags = []
    for val in 1 - stream["scalar"][:, 5]:
        detector.update(float(val))
        flags.append(bool(detector.drift_detected))
    return np.asarray(flags)


def risk_scores(data,gamma,mapper):
    D,js,rank,noise,entropy,msp,energy,_,distance=data["scalar"].T
    u,w,c=expected_cost_components(data["p"],data["cfg"])
    matched=mapper.transform(D)
    g=gamma["two_sided"];gd=gamma["directional"]
    return {"expected_cost":c,"naive_unscaled":(1-g)*c+g*D,
            "cost_preserved":w*((1-gd)*u+gd*matched),
            "cost_quantile_two_sided":w*((1-g)*u+g*matched),
            "cost_only_two_sided":w*((1-g)*u+g*D),
            "quantile_only_two_sided":(1-g)*c+g*matched,
            "naive_directional":(1-gd)*c+gd*D,
            "static_disagreement":D,"entropy":entropy,"msp_uncertainty":msp,"energy":energy,
            "input_distance":distance,"within_surrogate_disagreement":noise}


def loss_vector(y, p, cfg):
    pred = p >= 0.5
    return np.where((y == 1) & ~pred, cfg.false_negative_cost,
                    np.where((y == 0) & pred, cfg.false_positive_cost, 0.0))


def risk_curve(loss, score):
    # Every tied score group enters together; this makes AURC tie invariant.
    order = np.argsort(score, kind="stable")
    score, loss = np.asarray(score)[order], np.asarray(loss)[order]
    ends = np.r_[np.flatnonzero(np.diff(score) != 0) + 1, len(score)]
    sums = np.cumsum(loss)[ends-1]
    risk = sums / ends
    coverage = ends / len(loss)
    weights = np.diff(np.r_[0.0, coverage])
    return float(np.sum(weights * risk)), coverage, risk


def detector_metrics(windows, split, cfg, adwin):
    rows = []
    for (window, method), group in windows.groupby(["window", "method"], sort=False):
        pre = group.loc[group.phase.isin(["pre", "stationary"])]
        defined = group.loc[group.phase.isin(["pre", "post"])]
        auc = float(roc_auc_score(defined.phase == "post", defined.statistic)) if defined.phase.nunique() == 2 else np.nan
        alarms = group.loc[group.alarm]
        delay = np.nan
        detected = np.nan
        if split.onset is not None:
            eligible = alarms.loc[alarms.end > split.onset]
            detected = bool(len(eligible))
            delay = int(eligible.end.iloc[0] - split.onset) if len(eligible) else len(split.X) - split.onset
        rows.append({"window": window, "method": method, "drift_auroc": auc,
                     "pre_or_stationary_alarm_fraction": float(pre.alarm.mean()) if len(pre) else np.nan,
                     "any_pre_alarm": bool(pre.alarm.any()) if len(pre) else np.nan,
                     "alarm_windows": len(alarms), "evaluated_windows": len(group),
                     "detected": detected, "delay_or_censor_samples": delay,
                     "delay_is_censored": detected is False if isinstance(detected, bool) else np.nan,
                     "role": split.role})
    positions = np.flatnonzero(adwin) + 1
    pre_n = split.onset if split.onset is not None else len(split.X) if split.role == "stationary_control" else 0
    eligible = positions[positions > split.onset] if split.onset is not None else np.array([])
    rows.append({"window": 0, "method": "ADWIN_confidence_native", "drift_auroc": np.nan,
                 "pre_or_stationary_alarm_fraction": float(np.sum(positions <= pre_n) / pre_n) if pre_n else np.nan,
                 "any_pre_alarm": bool(np.any(positions <= pre_n)) if pre_n else np.nan,
                 "alarm_windows": len(positions), "evaluated_windows": len(split.X),
                 "detected": bool(len(eligible)) if split.onset is not None else np.nan,
                 "delay_or_censor_samples": int(eligible[0] - split.onset) if len(eligible) else len(split.X) - split.onset if split.onset is not None else np.nan,
                 "delay_is_censored": len(eligible) == 0 if split.onset is not None else np.nan,
                 "role": split.role})
    return rows


def selection_metrics(y, p, score_map, cutoffs, cfg, stream, window, split):
    losses = loss_vector(y, p, cfg)
    rows, curves = [], []
    masks = {"all": np.ones(len(y), dtype=bool)}
    if split.onset is not None:
        masks["fully_shifted"] = np.arange(len(y)) >= split.transition_end
    for scope, subset in masks.items():
        indices = np.flatnonzero(subset)
        eta = split.eta[subset] if split.eta is not None else None
        conditional_loss = np.where(p[subset]<.5, cfg.false_negative_cost*eta, cfg.false_positive_cost*(1-eta)) if eta is not None else None
        for method, scores in score_map.items():
            score, loss = scores[subset], losses[subset]
            aurc, coverage, risk = risk_curve(loss, score)
            for c, r in zip(coverage, risk) if method in ("expected_cost","naive_unscaled","cost_preserved") and window==cfg.primary_window else []:
                curves.append({"stream": stream, "window": window, "scope": scope, "method": method, "coverage": c, "risk": r})
            for budget in cfg.review_budgets:
                cutoff = cutoffs[(window, method, budget)]
                accept = score <= cutoff
                n_accept = int(accept.sum())
                rows.append({"stream": stream, "role": split.role, "window": window, "scope": scope,
                             "method": method, "review_budget_source": budget, "threshold": cutoff,
                             "n": len(score), "accepted": n_accept, "coverage": accept.mean(),
                             "selective_cost_risk": float(loss[accept].mean()) if n_accept else np.nan,
                             "unconditional_cost": float(loss.mean()), "aurc_cost": aurc,
                             "accuracy": float(((p[subset] >= .5) == y[subset]).mean()),
                             "true_conditional_aurc": risk_curve(conditional_loss,score)[0] if eta is not None else np.nan,
                             "true_conditional_selective_cost": float(conditional_loss[accept].mean()) if eta is not None and n_accept else np.nan,
                             "false_negative_accepted": int(((y[subset] == 1) & (p[subset] < .5) & accept).sum()),
                             "false_positive_accepted": int(((y[subset] == 0) & (p[subset] >= .5) & accept).sum())})
    return rows, curves


def ece(y, p):
    confidence = np.maximum(p, 1-p)
    correct = ((p >= 0.5) == y).astype(float)
    result = 0.0
    for lo, hi in zip(np.linspace(.5, 1, 11)[:-1], np.linspace(.5, 1, 11)[1:]):
        take = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if take.any():
            result += take.mean() * abs(correct[take].mean() - confidence[take].mean())
    return float(result)


def save_explanations(folder, name, split, data, feature_names):
    np.savez_compressed(folder / f"{name}_explanations.npz", X=data["X"], A=data["A"], B=data["B"], B2=data["B2"],
                        scalar=data["scalar"], margin=data["margin"], ids=split.ids.astype(str),
                        eta=split.eta if split.eta is not None else np.array([]),
                        feature_names=np.array(feature_names), scalar_names=np.array(SCALAR_NAMES))
    frame = pd.DataFrame({"id": split.ids, "y": split.y, "p": data["p"], "margin": data["margin"],
                          "lime_r2": data["lime_r2"], "primary_residual": data["residual"]})
    if split.eta is not None:
        frame["true_eta"] = split.eta
    for i, key in enumerate(SCALAR_NAMES):
        frame[key] = data["scalar"][:, i]
    frame.to_csv(folder / f"{name}_rows.csv", index=False)


def source_quality(predictor, gate, explanations):
    accuracy = float(balanced_accuracy_score(gate.y, explanations["p"] >= .5))
    residual = float(np.quantile(explanations["residual"], .95))
    r2 = float(np.median(explanations["lime_r2"]))
    std = float(np.std(explanations["scalar"][:, 0]))
    gates = {"balanced_accuracy_at_least_060": accuracy >= .60,
             "primary_p95_residual_at_most_005": residual <= .05,
             "local_median_r2_at_least_010": r2 >= .10,
             "disagreement_sd_at_least_00001": std >= .0001,
             "finite_outputs": bool(all(np.isfinite(explanations[k]).all() for k in ("A", "B", "B2", "scalar", "p")))}
    return {"balanced_accuracy": accuracy, "primary_p95_residual": residual, "local_median_r2": r2,
            "disagreement_sd": std, "gates": gates, "all_passed": all(gates.values())}


def run_unit(bundle, kind, seed, cfg, folder, log):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "data_provenance.json", bundle.provenance)
    write_json(folder / "split_checks.json", validate_bundle(bundle))
    write_json(folder / "stream_manifest.json", {name: {"role": s.role, "onset": s.onset,
               "transition_end": s.transition_end, "n": len(s.X), "id_sha256": digest(s.ids.tolist())}
               for name, s in bundle.streams.items()})
    predictor = Predictor(kind, cfg, seed)
    started = time.perf_counter()
    predictor.fit(bundle.source["train"].X, bundle.source["train"].y)
    training_s = time.perf_counter() - started
    predictor.save(folder)
    source_arrays = {role: predictor.transform(s.X) for role, s in bundle.source.items()}
    np.savez_compressed(folder / "source_split.npz", **{f"{role}_{key}": value for role, s in bundle.source.items()
                         for key, value in (("X_raw", s.X), ("y", s.y), ("ids", s.ids.astype(str)))})
    explainers = Explainers(predictor, source_arrays["train"], cfg, seed)
    np.savez_compressed(folder / "background.npz", background=explainers.background, baseline=explainers.baseline,
                        reference_output=explainers.reference_output)
    log("source quality check")
    gate = explainers.explain(source_arrays["gate"], bundle.source["gate"].ids, log)
    save_explanations(folder, "gate", bundle.source["gate"], gate, bundle.feature_names)
    quality = source_quality(predictor, bundle.source["gate"], gate)
    write_json(folder / "quality.json", quality)
    if cfg.stage == "confirmation" and not quality["all_passed"]:
        return {"quality": quality, "status": "source_quality_review", "training_seconds": training_s}
    explained = {}
    for role in ("reference", "calibration"):
        log(role + " explanations")
        explained[role] = explainers.explain(source_arrays[role], bundle.source[role].ids, log)
        save_explanations(folder, role, bundle.source[role], explained[role], bundle.feature_names)
    ref, cal = explained["reference"], explained["calibration"]
    third = len(cal["X"]) // 3
    for data in explained.values():
        data["cfg"] = cfg
    cal_mapping = subset_data(cal, np.arange(third))
    cal_detector = subset_data(cal, np.arange(third,2*third))
    cal_selection = subset_data(cal, np.arange(2*third,len(cal["X"])))
    mapper = QuantileMatcher(cal_mapping["scalar"][:,0], expected_cost_components(cal_mapping["p"],cfg)[0])
    mapper.save(folder / "quantile_mapping.json", bundle.source["calibration"].ids[:third])
    calibration_rows = pd.read_csv(folder / "calibration_rows.csv", float_precision="round_trip")
    calibration_rows["calibration_role"] = ["mapping"]*third + ["detector"]*third + ["selection"]*third
    calibration_rows.to_csv(folder / "calibration_rows.csv", index=False)
    detector = Detector(ref, cal_detector, cfg, seed)
    threshold_rows = [{"window": w, "method": name, "threshold": detector.thresholds[w][j],
                       "alpha": cfg.alpha, "bootstrap_replicates": cfg.bootstrap_n, "block_length": cfg.block_length}
                      for w in cfg.windows for j, name in enumerate(DETECTOR_NAMES)]
    pd.DataFrame(threshold_rows).to_csv(folder / "detector_thresholds.csv", index=False)
    np.savez_compressed(folder / "bootstrap_statistics.npz", **{f"W{w}": arr for w, arr in detector.bootstrap.items()},
                        detector_names=np.array(DETECTOR_NAMES), rff_w=detector.rff_w, rff_b=detector.rff_b)
    cal_windows, cal_gamma = detector.monitor(cal_selection, "calibration_selection")
    cal_windows["row_offset_in_calibration_table"] = 2*third
    cal_windows.to_csv(folder / "calibration_windows.csv", index=False)
    cutoffs, cutoff_rows = {}, []
    for w in cfg.windows:
        for method, scores in risk_scores(cal_selection, cal_gamma[w], mapper).items():
            for budget in cfg.review_budgets:
                cutoff = float(np.quantile(scores, 1-budget, method="higher"))
                cutoffs[(w, method, budget)] = cutoff
                cutoff_rows.append({"window": w, "method": method, "review_budget": budget, "threshold": cutoff,
                                    "source_coverage": float(np.mean(scores <= cutoff)),
                                    "calibration_role": "selection", "calibration_n": len(scores)})
    pd.DataFrame(cutoff_rows).to_csv(folder / "selection_thresholds.csv", index=False)
    all_windows, all_detection, all_selection, all_curves, timings, stream_quality = [], [], [], [], [], []
    timing = {"split": "gate", **gate["timing"]}; timings.append(timing)
    for role, data in explained.items():
        timings.append({"split": role, **data["timing"]})
    cached_stationary = None
    cache_explanations = {}
    paired_check = None
    for stream_name, split in bundle.streams.items():
        log(stream_name + " stream")
        if stream_name == "label_only":
            assert cached_stationary is not None
            baseline_split = bundle.streams["stationary"]
            assert np.array_equal(split.X, baseline_split.X) and np.array_equal(split.ids, baseline_split.ids)
            data = cached_stationary
            paired_check = {"identical_X_and_ids": True, "identical_monitor_inputs": True,
                            "changed_label_fraction": float(np.mean(split.y != baseline_split.y)),
                            "exact_feature_score_reuse": True}
        else:
            cache_key = digest({"X": hashlib.sha256(split.X.tobytes()).hexdigest(), "ids": split.ids.tolist()})
            if cache_key in cache_explanations:
                data = cache_explanations[cache_key]
            else:
                data = explainers.explain(predictor.transform(split.X), split.ids, log)
                cache_explanations[cache_key] = data
                timings.append({"split": stream_name, **data["timing"]})
        data["cfg"] = cfg
        if stream_name == "stationary":
            cached_stationary = data
        save_explanations(folder, stream_name, split, data, bundle.feature_names)
        windows, gammas = detector.monitor(data, stream_name, split.onset, split.transition_end)
        flags = adwin_trace(ref, data, cfg)
        rows = pd.read_csv(folder / f"{stream_name}_rows.csv", float_precision="round_trip")
        rows["ADWIN_alarm"] = flags
        rows["uncertainty_u"], rows["cost_weight_w"], _ = expected_cost_components(data["p"],cfg)
        rows["matched_disagreement"] = mapper.transform(data["scalar"][:,0])
        for w in cfg.windows:
            rows[f"gamma_two_sided_W{w}"] = gammas[w]["two_sided"]
            rows[f"gamma_directional_W{w}"] = gammas[w]["directional"]
            for method, scores in risk_scores(data, gammas[w], mapper).items():
                rows[f"risk_W{w}_{method}"] = scores
            r, c = selection_metrics(split.y, data["p"], risk_scores(data, gammas[w], mapper), cutoffs, cfg, stream_name, w, split)
            all_selection.extend(r); all_curves.extend(c)
        rows.to_csv(folder / f"{stream_name}_rows.csv", index=False)
        all_windows.append(windows)
        for row in detector_metrics(windows, split, cfg, flags):
            all_detection.append({"stream": stream_name, **row})
        stream_quality.append({"stream": stream_name, "primary_residual_p95": float(np.quantile(data["residual"], .95)),
                               "local_r2_median": float(np.median(data["lime_r2"])),
                               "disagreement_mean": float(data["scalar"][:, 0].mean()),
                               "within_lime_disagreement_mean": float(data["scalar"][:, 3].mean()),
                               "ece": ece(split.y, data["p"]), "n": len(split.y),
                               "brier": float(np.mean((data["p"]-split.y)**2)),
                               "posterior_mse": float(np.mean((data["p"]-split.eta)**2)) if split.eta is not None else np.nan})
    pd.concat(all_windows, ignore_index=True).to_csv(folder / "monitor_windows.csv", index=False)
    pd.DataFrame(all_detection).to_csv(folder / "detection_metrics.csv", index=False)
    pd.DataFrame(all_selection).to_csv(folder / "selection_metrics.csv", index=False)
    pd.DataFrame(all_curves).to_csv(folder / "risk_curves.csv", index=False)
    pd.DataFrame(timings).to_csv(folder / "timing.csv", index=False)
    pd.DataFrame(stream_quality).to_csv(folder / "stream_quality.csv", index=False)
    if paired_check is not None:
        write_json(folder / "label_only_observability_check.json", paired_check)
    if cfg.stage == "development" and "stationary" in bundle.streams:
        controls = pd.DataFrame(all_detection)
        primary_control = controls.loc[(controls.stream == "stationary") & (controls.window == cfg.primary_window)
                                       & (controls.method == "disagreement")]
        realized = float(primary_control.pre_or_stationary_alarm_fraction.iloc[0])
        quality["development_stationary_alarm_fraction"] = realized
        quality["gates"]["development_stationary_alarm_fraction_at_most_020"] = realized <= .20
        quality["all_passed"] = all(quality["gates"].values())
        write_json(folder / "quality.json", quality)
    return {"quality": quality, "status": "complete", "training_seconds": training_s,
            "feature_count": len(bundle.feature_names), "label_only_check": paired_check}


def verify_archive(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), "Archive member uniqueness"
        assert z.testzip() is None, "ZIP CRC verification"
        manifest = json.loads(z.read("file_manifest.json"))
        assert set(manifest) == set(names) - {"file_manifest.json"}, "Manifest membership"
        for name, record in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == record["sha256"], f"Checksum: {name}"
        return json.loads(z.read("run_status.json"))


def validate_freeze(path,cfg,engine_hash):
    status=verify_archive(path)
    assert status["stage"]=="development" and status["status"]=="complete"
    with zipfile.ZipFile(path) as z:
        freeze=json.loads(z.read("freeze_manifest.json"))
        assert freeze["all_technical_gates_passed"]
        assert freeze["engine_sha256"]==engine_hash
        assert freeze["core_sha256"]==digest(Config().core())==digest(cfg.core())
        # Operational settings are included in exact archival identity.
        assert freeze["study_config_sha256"]==digest(dataclasses.asdict(Config.study()))
        assert freeze["torch_version"]==importlib.metadata.version("torch")
        assert freeze["data_manifest_sha256"]==revision_manifest()[1]
        assert sorted(freeze["completed_units"])==sorted(f"{d}/{m}/seed_11" for d in DATASETS for m in MODELS)
        assert json.loads(z.read("config.json"))==plain(dataclasses.asdict(Config()))
    return {"source_zip_name":Path(path).name,"source_zip_sha256":file_hash(path),"freeze":freeze}


def aggregate(root,cfg):
    root=Path(root)
    collected={}
    for kind in ("detection","selection","timing","stream_quality"):
        parts=[]
        filename=kind+"_metrics.csv" if kind in ("detection","selection") else kind+".csv"
        for path in sorted((root/"units").rglob(filename)):
            rel=path.relative_to(root/"units").parts
            frame=pd.read_csv(path,float_precision="round_trip")
            frame.insert(0,"dataset",rel[0]);frame.insert(1,"model",rel[1]);frame.insert(2,"seed",int(rel[2].split("_")[1]))
            parts.append(frame)
        if parts:
            collected[kind]=pd.concat(parts,ignore_index=True)
            collected[kind].to_csv(root/(kind+"_all.csv"),index=False)
    if "selection" not in collected:
        return
    frame=collected["selection"].drop_duplicates(["dataset","model","seed","stream","window","scope","method"])
    frame.groupby(["dataset","model","stream","window","scope","method"],dropna=False).aurc_cost.agg(["mean","std","count"]).reset_index().to_csv(root/"aurc_summary.csv",index=False)
    table=frame.loc[frame.window==cfg.primary_window].pivot(index=["dataset","model","seed","stream","scope"],columns="method",values="aurc_cost").reset_index()
    for method in ("naive_unscaled","cost_preserved","cost_quantile_two_sided","cost_only_two_sided","quantile_only_two_sided","naive_directional"):
        table[method+"_minus_expected_cost"]=table[method]-table.expected_cost
    table.to_csv(root/"paired_aurc_contrasts.csv",index=False)
    # Both all-stream and fully-shifted scopes are reported, retaining every outcome.
    summary=table.groupby(["dataset","model","stream","scope"]).agg({k:["mean","std","count"] for k in ("expected_cost","naive_unscaled","cost_preserved","naive_unscaled_minus_expected_cost","cost_preserved_minus_expected_cost")})
    summary.to_csv(root/"table1_all_scopes.csv")
    detect=collected["detection"]
    detect.loc[detect.window==cfg.primary_window].groupby(["dataset","model","stream","method"]).drift_auroc.agg(["mean","std","count"]).reset_index().to_csv(root/"table2_all_streams.csv",index=False)


def make_figures(root, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    root = Path(root)
    figures = root / "figures"; figures.mkdir(exist_ok=True)
    for source in sorted((root / "units").rglob("monitor_windows.csv")):
        unit = source.parent
        windows = pd.read_csv(source, float_precision="round_trip")
        for stream in windows.stream.unique():
            data = pd.read_csv(unit / f"{stream}_rows.csv", float_precision="round_trip")
            sub = windows.loc[(windows.stream == stream) & (windows.window == cfg.primary_window)]
            fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
            block = max(1, len(data) // 60)
            time_x = np.arange(len(data)) + 1
            confidence = np.maximum(data.p, 1-data.p)
            correct = ((data.p >= .5) == data.y).astype(float)
            axes[0].plot(time_x, pd.Series(confidence).rolling(block, min_periods=1).mean(), label="Confidence", lw=1)
            axes[0].plot(time_x, pd.Series(correct).rolling(block, min_periods=1).mean(), label="Accuracy", lw=1)
            axes[0].set(xlabel="Stream sample", ylabel="Rolling mean", ylim=(0, 1.03))
            axes[0].legend(fontsize=7)
            for method in ("disagreement", "entropy", "single_A", "input_RFF_MMD"):
                m = sub.loc[sub.method == method]
                denom = np.maximum(m.threshold.to_numpy(), 1e-12)
                axes[1].plot(m.end, m.statistic.to_numpy()/denom, label=method, lw=1)
            axes[1].axhline(1, color="black", ls="--", lw=.7)
            axes[1].set(xlabel="Stream sample", ylabel="Statistic / source threshold")
            axes[1].legend(fontsize=6)
            curves = pd.read_csv(unit / "risk_curves.csv", float_precision="round_trip")
            scope = "fully_shifted" if "fully_shifted" in curves.loc[curves.stream == stream, "scope"].unique() else "all"
            curves = curves.loc[(curves.stream == stream) & (curves.scope == scope) & (curves.window == cfg.primary_window)]
            for method in ("adaptive", "expected_cost", "static_disagreement", "entropy"):
                c = curves.loc[curves.method == method]
                axes[2].plot(c.coverage, c.risk, label=method, lw=1)
            axes[2].set(xlabel="Coverage", ylabel="Mean accepted error cost")
            axes[2].legend(fontsize=6)
            label = "_".join(unit.relative_to(root / "units").parts) + "_" + stream
            fig.suptitle(label.replace("_", " "), fontsize=10)
            fig.tight_layout()
            fig.savefig(figures / f"{label}.pdf")
            fig.savefig(figures / f"{label}.png", dpi=180)
            plt.close(fig)


def package(root):
    root = Path(root)
    manifest = {str(p.relative_to(root)).replace("\\", "/"): {"sha256": file_hash(p), "bytes": p.stat().st_size}
                for p in sorted(root.rglob("*")) if p.is_file() and p.name != "file_manifest.json" and not p.name.endswith(".tmp")}
    write_json(root / "file_manifest.json", manifest)
    output = root.parent / f"RETURN_{root.name}.zip"
    temp = output.with_suffix(".partial")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in list(manifest) + ["file_manifest.json"]:
            z.write(root / name, name)
    temp.replace(output)
    verify_archive(output)
    return output


def run(cfg=None,base="esann_runs",development_zip=None,log=print):
    cfg=cfg or Config();cfg.validate()
    engine_file=Path(__file__).resolve();engine_sha=file_hash(engine_file)
    config=dataclasses.asdict(cfg)
    data_manifest,data_sha=revision_manifest()
    identity=digest({"engine":engine_sha,"config":config,"data_manifest":data_sha})[:12]
    base=Path(base);root=base/f"{cfg.stage}_{identity}";root.mkdir(parents=True,exist_ok=True)
    cache=base/"data_cache";cache.mkdir(exist_ok=True)
    (root/"engine.py").write_bytes(engine_file.read_bytes())
    write_json(root/"revision_data_manifest.json",data_manifest)
    write_json(root/"config.json",config)
    versions={name:importlib.metadata.version(name) for name in (*PINNED,"torch")}
    import torch
    write_json(root/"environment.json",{"python":sys.version,"platform":platform.platform(),"versions":versions,
               "engine_sha256":engine_sha,"data_manifest_sha256":data_sha,"protocol_version":VERSION,
               "cuda_available":torch.cuda.is_available(),"deterministic_algorithms":True,
               "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
    status={"stage":cfg.stage,"status":"running","identity":identity,"units":{},"engine_sha256":engine_sha,"data_manifest_sha256":data_sha}
    write_json(root/"run_status.json",status)
    events=[]
    def record(message):
        log(message,flush=True)
        events.append({"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"message":message})
        write_json(root/"progress.json",events)
    try:
        assert all(versions[name]==version for name,version in PINNED.items()),versions
        if cfg.stage=="confirmation":
            assert development_zip,"Development freeze required"
            write_json(root/"development_receipt.json",validate_freeze(development_zip,cfg,engine_sha))
        # Load public datasets once in the parent to populate cache without download races.
        for name in cfg.datasets:
            validate_bundle(make_bundle(name,cfg,cfg.seeds[0],cache))
        tasks=[]
        for name in cfg.datasets:
            for kind in cfg.models:
                for seed in cfg.seeds:
                    key=f"{name}/{kind}/seed_{seed}"
                    tasks.append((key,(cfg,name,kind,seed,str(cache),str(root/"units"/key),identity)))
        record(f"Starting {len(tasks)} units with {cfg.workers} processes and {cfg.threads} threads each")
        with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures={pool.submit(unit_worker,task):key for key,task in tasks}
            for future in concurrent.futures.as_completed(futures):
                key=futures[future]
                try:
                    result=future.result()
                except Exception:
                    result={"status":"failed","traceback":traceback.format_exc()}
                status["units"][key]=result
                write_json(root/"run_status.json",status)
                record(f"Finished {len(status['units'])}/{len(tasks)}: {key}: {result['status']}")
        if any(v["status"]!="complete" for v in status["units"].values()):
            raise RuntimeError("Review source-quality or worker diagnostics in run_status.json")
        aggregate(root,cfg)
        all_passed=all(v["quality"]["all_passed"] for v in status["units"].values())
        status["status"]="complete" if all_passed else "development_review"
        status["all_technical_gates_passed"]=all_passed
        if cfg.stage=="development" and all_passed and config==dataclasses.asdict(Config()):
            write_json(root/"freeze_manifest.json",{"protocol_version":VERSION,"engine_sha256":engine_sha,
                       "data_manifest_sha256":data_sha,"core_sha256":digest(cfg.core()),
                       "study_config_sha256":digest(dataclasses.asdict(Config.study())),
                       "all_technical_gates_passed":True,"completed_units":sorted(status["units"]),
                       "selection_rule":"Technical feasibility; every observed effect direction retained.",
                       "torch_version":versions["torch"],"study_config":dataclasses.asdict(Config.study())})
        record(f"Run status: {status['status']}")
    except Exception as error:
        status["status"]="failed";status["error"]=str(error)
        (root/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        record(f"Diagnostics: {error}")
    finally:
        status["ended_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        write_json(root/"run_status.json",status)
        output=package(root)
    log(f"RETURN THIS ZIP: {output.resolve()}",flush=True)
    return output,status



def revision_manifest():
    path=Path(__file__).with_name("revision_data_manifest.json")
    return json.loads(path.read_text()),file_hash(path)



def synthetic_bundle(name,cfg,seed):
    bundle=_source_synthetic_bundle(name,cfg,cfg.data_seed)
    bundle.provenance.update(protocol=VERSION,data_seed=cfg.data_seed,fit_seed_separate=True)
    return bundle



def interaction_bundle(cfg,seed):
    rng=np.random.default_rng(stable_seed("interaction",cfg.stage,cfg.data_seed))
    rho=cfg.interaction_rho
    def inputs(n,corr):
        X=rng.normal(size=(n,6))
        X[:,1]=corr*X[:,0]+np.sqrt(1-corr*corr)*X[:,1]
        return X.astype("float32")
    def logit(X,boundary=0.):
        g=X[:,0]+X[:,1]+1.5*(X[:,0]*X[:,1]-rho)+.25*X[:,2]
        local=np.exp(-((X[:,0]+X[:,1])/.5)**2)*np.tanh(2*(X[:,0]-X[:,1]))
        return 2*(g+cfg.boundary_strength*np.asarray(boundary)*local)
    def split(role,X,uniform,eta,ids=None,onset=None):
        if ids is None:
            ids=np.array([f"interaction:v2:{cfg.stage}:{cfg.data_seed}:{role}:{i}" for i in range(len(X))])
        return Split(X,(uniform<eta).astype("int64"),ids,onset,onset,role,eta)
    source={}
    for role,n in (("train",cfg.train_n),("gate",cfg.gate_n),("reference",cfg.reference_n),("calibration",cfg.calibration_n)):
        X=inputs(n,rho)
        source[role]=split(role,X,rng.random(n),expit(logit(X)))
    n=cfg.stream_n;t0=n//2
    X=inputs(n,rho);U=rng.random(n)
    changed=X.copy()
    changed[t0:,1]=-rho*changed[t0:,0]+np.sqrt(1-rho*rho)*rng.normal(size=n-t0)
    active=(np.arange(n)>=t0).astype(float)
    ids=np.array([f"interaction:v2:{cfg.stage}:{cfg.data_seed}:stream:{i}" for i in range(n)])
    stationary=split("stationary_control",X,U,expit(logit(X)),ids)
    covariate=split("controlled_covariate",changed,U,expit(logit(changed)),ids,t0)
    boundary=split("controlled_observable_conditional",changed,U,expit(logit(changed,active)),ids,t0)
    labels=split("label_only_control",X.copy(),U,expit(logit(X,active)),ids.copy(),t0)
    return Bundle("interaction",[f"x{i}" for i in range(6)],source,
                  {"stationary":stationary,"covariate_interaction":covariate,"local_boundary_shift":boundary,"label_only":labels},
                  {"kind":"synthetic","protocol":VERSION,"data_seed":cfg.data_seed,"stage":cfg.stage,
                   "source_correlation":rho,"target_correlation":-rho,"onset":t0,
                   "source_logit":"2*(x0+x1+1.5*(x0*x1-rho)+0.25*x2)",
                   "boundary_addition_inside_logit":"1.5*exp(-((x0+x1)/0.5)^2)*tanh(2*(x0-x1))",
                   "boundary_strength":cfg.boundary_strength,"eta_recorded_for_evaluation":True,
                   "paired_covariate_and_conditional_inputs":True,"label_rule_uses_model_or_explanation":False})



def expected_cost_components(p,cfg):
    p=np.asarray(p,dtype=float)
    u=np.minimum(p,1-p)
    denominator=max(cfg.false_negative_cost,cfg.false_positive_cost)
    weight=np.where(p<.5,cfg.false_negative_cost,cfg.false_positive_cost)/denominator
    return u,weight,weight*u



class QuantileMatcher:
    """Right-continuous empirical CDF followed by inverse empirical CDF."""
    def __init__(self,disagreement,uncertainty):
        self.sorted_D=np.sort(np.asarray(disagreement,dtype=float))
        self.sorted_u=np.sort(np.asarray(uncertainty,dtype=float))
        assert len(self.sorted_D)==len(self.sorted_u)>0
        assert np.isfinite(self.sorted_D).all() and np.isfinite(self.sorted_u).all()
        assert self.sorted_u.min()>=0 and self.sorted_u.max()<=.5
    def transform(self,D):
        q=np.searchsorted(self.sorted_D,np.asarray(D),side="right")/len(self.sorted_D)
        index=np.clip(np.ceil(q*len(self.sorted_u)).astype(int)-1,0,len(self.sorted_u)-1)
        return self.sorted_u[index]
    def save(self,path,ids):
        write_json(path,{"sorted_D":self.sorted_D,"sorted_u":self.sorted_u,"fit_ids":list(ids),
                         "cdf":"right-continuous empirical","inverse":"inverted empirical CDF, left quantile",
                         "fit_role":"mapping_calibration","uses_labels":False})



def unit_worker(task):
    cfg,name,kind,seed,cache,folder,identity=task
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    bundle=make_bundle(name,cfg,seed,cache)
    marker=folder/"complete.json"
    if marker.exists():
        prior=json.loads(marker.read_text())
        assert prior["unit_identity"]==identity and prior["provenance_sha256"]==digest(bundle.provenance)
        for member,expected in prior["unit_files"].items():
            assert file_hash(folder/member)==expected,(str(folder/member),"checksum")
        return prior["result"]
    with open(folder/"console.log","a",encoding="utf-8",buffering=1) as f:
        def log(message):
            f.write(time.strftime("%H:%M:%S ",time.gmtime())+message+"\n")
        result=run_unit(bundle,kind,seed,cfg,folder,log)
    if result["status"]=="complete":
        files={str(p.relative_to(folder)).replace("\\","/"):file_hash(p) for p in folder.rglob("*") if p.is_file() and p.name!="complete.json"}
        write_json(marker,{"unit_identity":identity,"provenance_sha256":digest(bundle.provenance),"unit_files":files,"result":result})
    return result

