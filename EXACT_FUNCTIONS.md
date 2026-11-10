# Exact score and detector implementations

Frozen scientific engine SHA256: `53a0d22652fdff2683dc9053f8ad072f09958b79b518f0ba5edb63ccebcbf4e4`.

## disagreement
`source/engine.py:450`
```python
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
```

## Detector
`source/engine.py:591`
```python
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
```

## risk_scores
`source/engine.py:666`
```python
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
```

## expected_cost_components
`source/engine.py:1190`
```python
def expected_cost_components(p,cfg):
    p=np.asarray(p,dtype=float)
    u=np.minimum(p,1-p)
    denominator=max(cfg.false_negative_cost,cfg.false_positive_cost)
    weight=np.where(p<.5,cfg.false_negative_cost,cfg.false_positive_cost)/denominator
    return u,weight,weight*u
```

## QuantileMatcher
`source/engine.py:1199`
```python
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
```
