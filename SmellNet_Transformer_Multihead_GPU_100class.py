#[SmellNet_Transformer_Multihead_GPU_100class.py]

import os
import glob
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class SmellVectorizeConfig:
    feature_cols: Optional[List[str]] = None
    drop_cols: Tuple[str, ...] = ("time", "timestamp", "ts", "index", "label", "class")

    fillna: str = "ffill"          # "ffill", "bfill", "zero"
    standardize: str = "zscore"    # "zscore" or "none"
    eps: float = 1e-6

    use_temporal_diff: bool = True
    p_lag: int = 25
    diff_mode: str = "diff"        # "diff" or "concat"

    window_w: int = 100
    stride: int = 1
    drop_short: bool = True


def _choose_feature_columns(df: pd.DataFrame, cfg: SmellVectorizeConfig) -> List[str]:
    if cfg.feature_cols is not None:
        missing = [c for c in cfg.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols에 없는 컬럼: {missing}")
        return cfg.feature_cols

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    filtered = []
    for c in num_cols:
        c_low = str(c).lower()
        if any(dc in c_low for dc in cfg.drop_cols):
            continue
        filtered.append(c)
    if not filtered:
        raise ValueError("자동 선택된 feature 컬럼이 없습니다. feature_cols를 직접 지정하세요.")
    return filtered


def _fillna(df_feat: pd.DataFrame, cfg: SmellVectorizeConfig) -> pd.DataFrame:
    if cfg.fillna == "ffill":
        return df_feat.ffill().bfill()
    if cfg.fillna == "bfill":
        return df_feat.bfill().ffill()
    if cfg.fillna == "zero":
        return df_feat.fillna(0.0)
    raise ValueError("fillna는 'ffill'/'bfill'/'zero' 중 하나여야 합니다.")


def _standardize(x: np.ndarray, cfg: SmellVectorizeConfig) -> np.ndarray:
    if cfg.standardize == "none":
        return x
    if cfg.standardize == "zscore":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.maximum(std, cfg.eps)
        return (x - mean) / std
    raise ValueError("standardize는 'zscore' 또는 'none' 이어야 합니다.")


def _temporal_difference(x: np.ndarray, cfg: SmellVectorizeConfig) -> np.ndarray:
    if not cfg.use_temporal_diff:
        return x

    p = int(cfg.p_lag)
    if p <= 0:
        return x

    diff = np.zeros_like(x)
    if x.shape[0] > p:
        diff[p:] = x[p:] - x[:-p]

    if cfg.diff_mode == "diff":
        return diff
    if cfg.diff_mode == "concat":
        return np.concatenate([x, diff], axis=-1)
    raise ValueError("diff_mode는 'diff' 또는 'concat' 이어야 합니다.")


def _sliding_window(x: np.ndarray, cfg: SmellVectorizeConfig) -> np.ndarray:
    T_total, D = x.shape
    w = int(cfg.window_w)
    s = int(cfg.stride)

    if T_total < w:
        if cfg.drop_short:
            return np.zeros((0, w, D), dtype=x.dtype)
        out = np.zeros((1, w, D), dtype=x.dtype)
        out[0, :T_total] = x
        return out

    starts = np.arange(0, T_total - w + 1, s, dtype=int)
    windows = np.empty((len(starts), w, D), dtype=x.dtype)
    for i, st in enumerate(starts):
        windows[i] = x[st:st + w]
    return windows


def prepare_dataset(
    data_root: str,
    out_npz: str = "smell_dataset.npz",
    out_meta_json: str = "smell_dataset_meta.json",
    cfg: Optional[SmellVectorizeConfig] = None,
):
    if cfg is None:
        cfg = SmellVectorizeConfig()

    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    if len(classes) == 0:
        raise RuntimeError(f"클래스 폴더가 없습니다: {data_root}")

    #if len(classes) != 10:
    #    print(f"[주의] classes={len(classes)} 입니다. 요청은 10 classes입니다.")
    if len(classes) != 100:
        print(f"[주의] classes={len(classes)} 입니다. 요청은 100 classes입니다.")

    class_to_id = {c: i for i, c in enumerate(classes)}

    all_X, all_y = [], []
    feature_cols_final = None
    files_used = 0

    # for c in classes:
    #     c_dir = os.path.join(data_root, c)
    #     for p in sorted(glob.glob(os.path.join(c_dir, "*.csv"))):
    #         df = pd.read_csv(p)
    #         feat_cols = _choose_feature_columns(df, cfg)

    #         if feature_cols_final is None:
    #             feature_cols_final = feat_cols
    #         elif feat_cols != feature_cols_final:
    #             raise ValueError(f"Feature columns mismatch: {p}")

    #         df_feat = _fillna(df[feat_cols].copy(), cfg)
    #         x = df_feat.to_numpy(dtype=np.float32)
    #         x = _standardize(x, cfg)
    #         x = _temporal_difference(x, cfg)
    #         Xw = _sliding_window(x, cfg)

    #         if Xw.shape[0] == 0:
    #             continue

    #         all_X.append(Xw)
    #         all_y.append(np.full((Xw.shape[0],), class_to_id[c], dtype=np.int64))
    #         files_used += 1

    for c in classes:
        c_dir = os.path.join(data_root, c)

        for p in sorted(glob.glob(os.path.join(c_dir, "*.csv"))):

            df = pd.read_csv(p)

            # 사용할 컬럼
            #feat_cols = ["NO2", "VOC",	"CO", "Alcohol", "LPG", "Benzene", "Temperature", 
            #    "Pressure", "Humidity", "Gas_Resistance", "Altitude"]

            feat_cols = ["NO2", "VOC", "CO"]

            # C2H5OH 또는 C2H5CH 중 하나를 사용
            if "C2H5CH" in df.columns:
                df["C2H5OH"] = df["C2H5CH"]
            elif "C2H5OH" in df.columns:
                pass
            else:
                raise ValueError(f"{p}에 C2H5OH 또는 C2H5CH 컬럼이 없습니다.")

            feat_cols.insert(1, "C2H5OH")

            # 컬럼 존재 여부 확인
            missing_cols = [col for col in feat_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"{p}에 다음 컬럼이 없습니다: {missing_cols}"
                )

            # 필요한 컬럼만 추출
            df_feat = df[feat_cols].copy()
            
            # 결측치 처리
            df_feat = _fillna(df_feat, cfg)

            # numpy 변환
            x = df_feat.to_numpy(dtype=np.float32)

            # 표준화
            x = _standardize(x, cfg)

            # Temporal Difference
            x = _temporal_difference(x, cfg)

            # Sliding Window
            Xw = _sliding_window(x, cfg)

            if Xw.shape[0] == 0:
                continue

            all_X.append(Xw)
            all_y.append(
                np.full(
                    (Xw.shape[0],),
                    class_to_id[c],
                    dtype=np.int64
                )
            )

            files_used += 1

    if not all_X:
        raise RuntimeError("생성된 윈도우가 없습니다. window_w/데이터 길이를 확인하세요.")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)

    meta = {
        "data_root": data_root,
        "classes": classes,
        "class_to_id": class_to_id,
        "num_files_used": files_used,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "vectorize_config": cfg.__dict__,
        "feature_cols": feature_cols_final,
    }

    np.savez_compressed(out_npz, X=X, y=y)
    with open(out_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_npz}")
    print(f"Saved: {out_meta_json}")
    print("X:", X.shape, "y:", y.shape, "num_classes:", len(classes))


if __name__ == "__main__":
    cfg = SmellVectorizeConfig(
        window_w=100,
        stride=1,
        use_temporal_diff=True,
        p_lag=25,
        #diff_mode="concat",   # diff/concat 실험용 옵션 (원하면 "diff"로 변경)
        diff_mode="diff",   
        standardize="zscore",
        fillna="ffill",
    )
    prepare_dataset(
        #data_root="data_smell", #smell data가 있는 폴더명 데이터는 .csv 형태로 저장됨
        data_root="data_100\smell", #smell data가 있는 폴더명 데이터는 .csv 형태로 저장됨
        out_npz="smell_dataset.npz",
        out_meta_json="smell_dataset_meta.json",
        cfg=cfg
    )

import os
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List
from sklearn.decomposition import PCA

import numpy as np
import cupy as cp
cp.cuda.Device(0).use()
x = cp.random.randn(100, 100)
print(x.device)   # cuda:0 이면 성공

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# =========================
# Epoch DataLoader (shuffle then iterate all)
# =========================
def make_split_80_20(N: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(N * 0.8)
    return idx[:n_train], idx[n_train:]


def epoch_batches(X: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool = True):
    """
    NumPy 인덱스 기반 epoch loader:
    - train set 전체를 셔플한 뒤 batch로 순회
    """
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    for st in range(0, N, batch_size):
        ed = min(st + batch_size, N)
        b = idx[st:ed]
        yield X[b], y[b]


# =========================
# Math utils (CuPy)
# =========================
def softmax(x, axis=-1):
    x = x - cp.max(x, axis=axis, keepdims=True)
    e = cp.exp(x)
    return e / cp.sum(e, axis=axis, keepdims=True)


def one_hot(y, C):
    oh = cp.zeros((y.shape[0], C), dtype=cp.float32)
    oh[cp.arange(y.shape[0]), y] = 1.0
    return oh


# =========================
# Layers: Linear / LayerNorm
# =========================
class Linear:
    def __init__(self, in_dim, out_dim, scale=0.02, name="linear"):
        self.W = (cp.random.randn(in_dim, out_dim).astype(cp.float32) * scale)
        self.b = cp.zeros((out_dim,), dtype=cp.float32)
        self.name = name

        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)

    def forward(self, x):
        # x: (..., in_dim)
        y = x @ self.W + self.b
        cache = (x,)
        return y, cache

    def backward(self, dy, cache):
        (x,) = cache
        # dy: (..., out_dim)
        x2 = x.reshape(-1, x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])

        self.dW = x2.T @ dy2
        self.db = cp.sum(dy2, axis=0)

        dx2 = dy2 @ self.W.T
        dx = dx2.reshape(x.shape)
        return dx


class LayerNorm:
    def __init__(self, dim, eps=1e-5, name="ln"):
        self.g = cp.ones((dim,), dtype=cp.float32)
        self.b = cp.zeros((dim,), dtype=cp.float32)
        self.eps = eps
        self.name = name

        self.dg = cp.zeros_like(self.g)
        self.db = cp.zeros_like(self.b)

    def forward(self, x):
        # x: (..., dim)
        mu = cp.mean(x, axis=-1, keepdims=True)
        var = cp.mean((x - mu) ** 2, axis=-1, keepdims=True)
        inv = 1.0 / cp.sqrt(var + self.eps)
        xhat = (x - mu) * inv
        y = xhat * self.g + self.b
        cache = (x, xhat, mu, var, inv)
        return y, cache

    def backward(self, dy, cache):
        x, xhat, mu, var, inv = cache
        D = x.shape[-1]

        # grads for affine
        dy2 = dy.reshape(-1, D)
        xhat2 = xhat.reshape(-1, D)

        self.dg = cp.sum(dy2 * xhat2, axis=0)
        self.db = cp.sum(dy2, axis=0)

        # backprop through normalization
        dxhat = dy * self.g
        dxhat2 = dxhat.reshape(-1, D)
        xmu2 = (x - mu).reshape(-1, D)

        dvar = cp.sum(dxhat2 * xmu2 * (-0.5) * (var.reshape(-1,1) + self.eps) ** (-1.5), axis=1, keepdims=True)
        dmu = cp.sum(dxhat2 * (-inv.reshape(-1,1)), axis=1, keepdims=True) + dvar * cp.mean(-2.0 * xmu2, axis=1, keepdims=True)

        dx = (dxhat2 * inv.reshape(-1,1)) + (dvar * 2.0 * xmu2 / D) + (dmu / D)
        return dx.reshape(x.shape)


# =========================
# GELU
# =========================
def gelu(x):
    # approximation
    return 0.5 * x * (1.0 + cp.tanh(cp.sqrt(2.0 / cp.pi) * (x + 0.044715 * x**3)))


def gelu_backward(dy, x):
    # derivative of tanh approximation (good enough for training)
    t = cp.sqrt(2.0 / cp.pi) * (x + 0.044715 * x**3)
    u = cp.tanh(t)
    dt = cp.sqrt(2.0 / cp.pi) * (1 + 3 * 0.044715 * x**2)
    du = (1 - u**2) * dt
    dgelu = 0.5 * (1 + u) + 0.5 * x * du
    return dy * dgelu


# =========================
# Single-Head Self-Attention (direct backprop)
# =========================
# =========================
# Multi-Head Self-Attention (direct backprop, CuPy)
# =========================
class SelfAttentionMultiHead:
    """
    Multi-head scaled dot-product attention (manual backprop)
    x -> Q,K,V -> split heads -> softmax(QK^T/sqrt(dh))V -> concat -> Wo
    """

    def __init__(self, d_model, num_heads=4, name="attn_mh"):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = int(num_heads)
        self.dh = d_model // self.h

        s = 0.02
        # (D,D) projections (standard MHA style: one big mat then reshape)
        self.Wq = (cp.random.randn(d_model, d_model).astype(cp.float32) * s)
        self.Wk = (cp.random.randn(d_model, d_model).astype(cp.float32) * s)
        self.Wv = (cp.random.randn(d_model, d_model).astype(cp.float32) * s)
        self.Wo = (cp.random.randn(d_model, d_model).astype(cp.float32) * s)

        self.bq = cp.zeros((d_model,), dtype=cp.float32)
        self.bk = cp.zeros((d_model,), dtype=cp.float32)
        self.bv = cp.zeros((d_model,), dtype=cp.float32)
        self.bo = cp.zeros((d_model,), dtype=cp.float32)

        self.name = name

        # grads
        self.dWq = cp.zeros_like(self.Wq); self.dWk = cp.zeros_like(self.Wk)
        self.dWv = cp.zeros_like(self.Wv); self.dWo = cp.zeros_like(self.Wo)
        self.dbq = cp.zeros_like(self.bq); self.dbk = cp.zeros_like(self.bk)
        self.dbv = cp.zeros_like(self.bv); self.dbo = cp.zeros_like(self.bo)

    def _split_heads(self, x):
        # x: (B,T,D) -> (B,H,T,dh)
        B, T, D = x.shape
        x = x.reshape(B, T, self.h, self.dh)
        return cp.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, xh):
        # xh: (B,H,T,dh) -> (B,T,D)
        B, H, T, dh = xh.shape
        x = cp.transpose(xh, (0, 2, 1, 3)).reshape(B, T, H * dh)
        return x

    def forward(self, x, attn_mask=None):
        """
        x: (B,T,D)
        attn_mask: (B,T) bool or 0/1. True/1=valid token, False/0=pad
                   (key/query 둘 다 마스킹)
        """
        B, T, D = x.shape

        Q = x @ self.Wq + self.bq  # (B,T,D)
        K = x @ self.Wk + self.bk
        V = x @ self.Wv + self.bv

        Qh = self._split_heads(Q)  # (B,H,T,dh)
        Kh = self._split_heads(K)
        Vh = self._split_heads(V)

        scale = 1.0 / np.sqrt(self.dh)
        # scores: (B,H,T,T)
        scores = cp.matmul(Qh, cp.transpose(Kh, (0, 1, 3, 2))) * scale

        if attn_mask is not None:
            m = attn_mask.astype(cp.bool_)           # (B,T)
            key_mask = m[:, None, None, :]           # (B,1,1,T)
            scores = cp.where(key_mask, scores, -1e9)

        A = softmax(scores, axis=-1)                 # (B,H,T,T)
        Oh = cp.matmul(A, Vh)                        # (B,H,T,dh)
        O = self._merge_heads(Oh)                    # (B,T,D)

        Y = O @ self.Wo + self.bo                    # (B,T,D)

        if attn_mask is not None:
            qmask = attn_mask.astype(cp.float32)[:, :, None]  # (B,T,1)
            Y = Y * qmask

        cache = (x, Qh, Kh, Vh, A, Oh, scale, attn_mask, O)
        return Y, cache

    def backward(self, dY, cache):
        """
        dY: (B,T,D)
        returns dx: (B,T,D)
        """
        x, Qh, Kh, Vh, A, Oh, scale, attn_mask, O = cache
        B, T, D = x.shape

        if attn_mask is not None:
            qmask = attn_mask.astype(cp.float32)[:, :, None]
            dY = dY * qmask

        # Y = O Wo + bo
        dO = dY @ self.Wo.T
        self.dWo = O.reshape(-1, D).T @ dY.reshape(-1, D)
        self.dbo = cp.sum(dY.reshape(-1, D), axis=0)

        # dO -> heads
        dOh = self._split_heads(dO)  # (B,H,T,dh)

        # Oh = A Vh
        dA = cp.matmul(dOh, cp.transpose(Vh, (0, 1, 3, 2)))     # (B,H,T,T)
        dVh = cp.matmul(cp.transpose(A, (0, 1, 3, 2)), dOh)     # (B,H,T,dh)

        if attn_mask is not None:
            m = attn_mask.astype(cp.float32)[:, None, None, :]  # (B,1,1,T) key mask
            dA = dA * m

        # A = softmax(scores)
        tmp = cp.sum(dA * A, axis=-1, keepdims=True)            # (B,H,T,1)
        dScores = A * (dA - tmp)                                # (B,H,T,T)

        if attn_mask is not None:
            m = attn_mask.astype(cp.float32)[:, None, None, :]
            dScores = dScores * m

        # scores = Qh Kh^T * scale
        dQh = cp.matmul(dScores, Kh) * scale                    # (B,H,T,dh)
        dKh = cp.matmul(cp.transpose(dScores, (0, 1, 3, 2)), Qh) * scale

        # merge heads: (B,H,T,dh) -> (B,T,D)
        dQ = self._merge_heads(dQh)
        dK = self._merge_heads(dKh)
        dV = self._merge_heads(dVh)

        # Q = xWq + bq
        self.dWq = x.reshape(-1, D).T @ dQ.reshape(-1, D)
        self.dbq = cp.sum(dQ.reshape(-1, D), axis=0)
        dx_q = dQ @ self.Wq.T

        # K = xWk + bk
        self.dWk = x.reshape(-1, D).T @ dK.reshape(-1, D)
        self.dbk = cp.sum(dK.reshape(-1, D), axis=0)
        dx_k = dK @ self.Wk.T

        # V = xWv + bv
        self.dWv = x.reshape(-1, D).T @ dV.reshape(-1, D)
        self.dbv = cp.sum(dV.reshape(-1, D), axis=0)
        dx_v = dV @ self.Wv.T

        dx = dx_q + dx_k + dx_v

        if attn_mask is not None:
            qmask = attn_mask.astype(cp.float32)[:, :, None]
            dx = dx * qmask

        return dx

# =========================
# Transformer Encoder Block (1-layer)
# =========================
class Transformer1Layer:
    def __init__(self, d_in, d_model=128, dim_ff=256, num_classes=100):
        self.d_in = d_in
        self.d_model = d_model
        self.num_classes = num_classes

        self.in_proj = Linear(d_in, d_model, name="in_proj")

        # CLS token
        self.cls = (cp.random.randn(1, 1, d_model).astype(cp.float32) * 0.02)
        self.dcls = cp.zeros_like(self.cls)

        self.ln1 = LayerNorm(d_model, name="ln1")
        
        # 기존:
        # self.attn = SelfAttention1Head(d_model, name="attn")

        # 변경:
        self.attn = SelfAttentionMultiHead(d_model, num_heads=4, name="attn_mh")
        #num_heads=4는 예시입니다. d_model=128이면 2/4/8 모두 가능(128 % heads == 0 만족).
        self.ln2 = LayerNorm(d_model, name="ln2")

        # FFN
        self.ff1 = Linear(d_model, dim_ff, name="ff1")
        self.ff2 = Linear(dim_ff, d_model, name="ff2")

        self.final_ln = LayerNorm(d_model, name="final_ln")
        self.head = Linear(d_model, num_classes, name="head")

    def forward(self, x, lengths=None):
        """
        x: (B,T,D_in)
        logits: (B,C)
        """
        B, T, _ = x.shape

        if lengths is None:
            tok_mask = cp.ones((B, T), dtype=cp.bool_)
        else:
            lengths = lengths.astype(cp.int64)
            tok_mask = (cp.arange(T)[None, :] < lengths[:, None])  # (B,T) valid=True

        h, c_in = self.in_proj.forward(x)
        cls = cp.tile(self.cls, (B, 1, 1))
        h = cp.concatenate([cls, h], axis=1)  # (B,T+1,D)

        # CLS 포함 mask: (B,T+1)
        attn_mask = cp.concatenate([cp.ones((B,1), dtype=cp.bool_), tok_mask], axis=1)

        # pre-norm attention
        h1, c_ln1 = self.ln1.forward(h)
        a_out, c_attn = self.attn.forward(h1, attn_mask=attn_mask)  # <-- 여기!
        h = h + a_out
        
        # pre-norm FFN
        h2, c_ln2 = self.ln2.forward(h)
        ff1_out, c_ff1 = self.ff1.forward(h2)
        ff1_act = gelu(ff1_out)
        ff2_out, c_ff2 = self.ff2.forward(ff1_act)
        h = h + ff2_out

        # CLS pooling
        cls_vec = h[:, 0, :]              # (B,d_model)
        z, c_fln = self.final_ln.forward(cls_vec)
        logits, c_head = self.head.forward(z)  # (B,C)

        cache = {
            "x": x,
            "c_in": c_in,
            "B": B,
            "c_ln1": c_ln1,
            "c_attn": c_attn,
            "h_after_attn": None,  # not needed
            "c_ln2": c_ln2,
            "c_ff1": c_ff1,
            "ff1_out": ff1_out,
            "ff1_act": ff1_act,
            "c_ff2": c_ff2,
            "h_after_ffn": h,      # needed for CLS gradient path
            "c_fln": c_fln,
            "c_head": c_head,
            "T_plus": h.shape[1],
        }
        return logits, cache

    def backward(self, dlogits, cache):
        """
        dlogits: (B,C)
        """
        B = cache["B"]

        # head backward
        dz = self.head.backward(dlogits, cache["c_head"])   # (B,d_model)

        # final ln backward
        dcls_vec = self.final_ln.backward(dz, cache["c_fln"])  # (B,d_model)

        # propagate into h (only CLS position)
        h = cache["h_after_ffn"]
        dh = cp.zeros_like(h)
        dh[:, 0, :] = dcls_vec

        # back through FFN residual: h = h_prev + ff2_out
        # so d_ff2_out = dh, and dh_prev accumulates dh
        dff2_out = dh
        dh_prev = dh

        # ff2 backward
        dff1_act = self.ff2.backward(dff2_out, cache["c_ff2"])  # (B,T+,dim_ff)

        # gelu backward
        dff1_out = gelu_backward(dff1_act, cache["ff1_out"])

        # ff1 backward
        dh2 = self.ff1.backward(dff1_out, cache["c_ff1"])       # (B,T+,d_model)

        # ln2 backward
        dh_ln2 = self.ln2.backward(dh2, cache["c_ln2"])          # (B,T+,d_model)

        # residual add at attn output stage: h_after_attn + ff2_out
        # dh_prev already contains gradients flowing into h_after_attn
        dh_attn_stage = dh_prev + dh_ln2

        # back through attention residual: h = h_before_attn + attn_out
        datt_out = dh_attn_stage
        dh_before_attn = dh_attn_stage

        # attn backward (input was h1 = ln1(h_before_attn))
        dh1 = self.attn.backward(datt_out, cache["c_attn"])      # (B,T+,d_model)

        # ln1 backward
        dh0 = self.ln1.backward(dh1, cache["c_ln1"])             # (B,T+,d_model)

        # residual: dh_before_attn accumulates ln path too
        dh0 = dh0 + dh_before_attn

        # split CLS + token part
        dcls = cp.sum(dh0[:, :1, :], axis=0, keepdims=True)   # (1,1,d_model)
        self.dcls = dcls

        dh_tokens = dh0[:, 1:, :]  # (B,T,d_model)

        # in_proj backward
        dx = self.in_proj.backward(dh_tokens, cache["c_in"])   # (B,T,D_in)
        return dx

    def zero_grads(self):
        # linear
        self.in_proj.dW.fill(0); self.in_proj.db.fill(0)
        self.ff1.dW.fill(0); self.ff1.db.fill(0)
        self.ff2.dW.fill(0); self.ff2.db.fill(0)
        self.head.dW.fill(0); self.head.db.fill(0)

        # ln
        self.ln1.dg.fill(0); self.ln1.db.fill(0)
        self.ln2.dg.fill(0); self.ln2.db.fill(0)
        self.final_ln.dg.fill(0); self.final_ln.db.fill(0)

        # attn
        self.attn.dWq.fill(0); self.attn.dWk.fill(0); self.attn.dWv.fill(0); self.attn.dWo.fill(0)
        self.attn.dbq.fill(0); self.attn.dbk.fill(0); self.attn.dbv.fill(0); self.attn.dbo.fill(0)

        # cls
        self.dcls.fill(0)

    def params_and_grads(self):
        """
        AdamW 업데이트를 위해 파라미터 딕셔너리로 반환
        """
        P = {}
        G = {}

        def add(name, p, g):
            P[name] = p
            G[name] = g

        add("in_proj.W", self.in_proj.W, self.in_proj.dW)
        add("in_proj.b", self.in_proj.b, self.in_proj.db)
        add("cls", self.cls, self.dcls)

        add("ln1.g", self.ln1.g, self.ln1.dg); add("ln1.b", self.ln1.b, self.ln1.db)
        add("attn.Wq", self.attn.Wq, self.attn.dWq); add("attn.Wk", self.attn.Wk, self.attn.dWk)
        add("attn.Wv", self.attn.Wv, self.attn.dWv); add("attn.Wo", self.attn.Wo, self.attn.dWo)
        add("attn.bq", self.attn.bq, self.attn.dbq); add("attn.bk", self.attn.bk, self.attn.dbk)
        add("attn.bv", self.attn.bv, self.attn.dbv); add("attn.bo", self.attn.bo, self.attn.dbo)

        add("ln2.g", self.ln2.g, self.ln2.dg); add("ln2.b", self.ln2.b, self.ln2.db)

        add("ff1.W", self.ff1.W, self.ff1.dW); add("ff1.b", self.ff1.b, self.ff1.db)
        add("ff2.W", self.ff2.W, self.ff2.dW); add("ff2.b", self.ff2.b, self.ff2.db)

        add("final_ln.g", self.final_ln.g, self.final_ln.dg); add("final_ln.b", self.final_ln.b, self.final_ln.db)
        add("head.W", self.head.W, self.head.dW); add("head.b", self.head.b, self.head.db)

        return P, G


# =========================
# Softmax CE loss (forward/backward)
# =========================
def softmax_ce_loss(logits, y):
    """
    logits: (B,C)
    y: (B,) int
    returns loss scalar, dlogits (B,C), acc
    """
    B, C = logits.shape
    prob = softmax(logits, axis=1)
    loss = -cp.mean(cp.log(prob[cp.arange(B), y] + 1e-12))
    pred = cp.argmax(prob, axis=1)
    acc = cp.mean((pred == y).astype(cp.float32))

    #cross entropy loss = -xp.sum(xp.log(probs[xp.arange(B), y] + 1e-9)) / B

    # dlogits = (prob - onehot) / B
    oh = one_hot(y, C)
    dlogits = (prob - oh) / B
    return loss, dlogits, acc


# =========================
# AdamW optimizer (CuPy)
# =========================
class AdamW:
    def __init__(self, params: Dict[str, cp.ndarray], lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-4):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = {k: cp.zeros_like(v) for k, v in params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in params.items()}

    def step(self, params: Dict[str, cp.ndarray], grads: Dict[str, cp.ndarray]):
        self.t += 1
        lr = self.lr
        b1, b2 = self.b1, self.b2

        for k in params.keys():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)

            mhat = self.m[k] / (1 - (b1 ** self.t))
            vhat = self.v[k] / (1 - (b2 ** self.t))

            # decoupled weight decay
            params[k] -= lr * (mhat / (cp.sqrt(vhat) + self.eps) + self.wd * params[k])

# ================================
# compute_confusion_matrix/heatmap
# ================================

def compute_confusion_matrix(y_true, y_pred, num_classes: int):
    """
    y_true, y_pred: 1D numpy arrays of int
    returns: (num_classes, num_classes) cm  (rows=true, cols=pred)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix_heatmap(cm, class_names=None, normalize: bool = True, title="Confusion Matrix"):
    """
    cm: numpy array (C,C)
    normalize=True면 행(row) 기준 %로 표시
    """
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_show = (cm / np.maximum(row_sum, 1)) * 100.0
        fmt = "{:.1f}"
        suffix = " (%)"
    else:
        cm_show = cm.astype(np.float32)
        fmt = "{:d}"
        suffix = " (count)"

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_show, interpolation="nearest")
    plt.title(title + suffix)
    plt.colorbar()

    C = cm.shape[0]
    ticks = np.arange(C)
    if class_names is None:
        class_names = [str(i) for i in range(C)]

    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # 셀 값 표시
    for i in range(C):
        for j in range(C):
            if normalize:
                text = fmt.format(cm_show[i, j])
            else:
                text = fmt.format(int(cm[i, j]))
            plt.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

    
# =========================
# Save / Load
# =========================
def save_params(model: Transformer1Layer, meta: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    P, _ = model.params_and_grads()
    np_pack = {k: cp.asnumpy(v) for k, v in P.items()}
    np.savez_compressed(path, **np_pack)
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved params:", path, "and", path + ".meta.json")


def load_params(model: Transformer1Layer, path: str):
    pack = np.load(path)
    P, _ = model.params_and_grads()
    for k in P.keys():
        P[k][...] = cp.array(pack[k], dtype=cp.float32)
    print("Loaded params:", path)


# =========================
# Train / Eval
# =========================
#@cp.no_fusion()
def evaluate(model: Transformer1Layer, X_np: np.ndarray, y_np: np.ndarray, batch_size: int):
    """
    전체 test set을 batch로 순회하며 loss/acc 계산 (GPU)
    """
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    all_true = []
    all_pred = []

    for xb, yb in epoch_batches(X_np, y_np, batch_size=batch_size, seed=0, shuffle=False):
        xb_cp = cp.asarray(xb, dtype=cp.float32)
        yb_cp = cp.asarray(yb, dtype=cp.int32)

        logits, cache = model.forward(xb_cp)
        loss, _, acc = softmax_ce_loss(logits, yb_cp)

        prob = softmax(logits, axis=1)
        pred = cp.argmax(prob, axis=1)

        bsz = yb.shape[0]
        total_loss += float(loss.get()) * bsz
        total_acc += float(acc.get()) * bsz
        total += bsz

        all_true.append(yb)
        all_pred.append(cp.asnumpy(pred))

    avg_loss = total_loss / max(1, total)
    avg_acc = total_acc / max(1, total)
    y_true = np.concatenate(all_true) if all_true else np.array([])
    y_pred = np.concatenate(all_pred) if all_pred else np.array([])
    return avg_loss, avg_acc, y_true, y_pred


def save_confusion_heatmap(cm: np.ndarray, class_names: List[str], out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(cm_sum, 1)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_norm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix (row-normalized)")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            if v >= 0.05:
                plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if v > 0.5 else "black", fontsize=8)

    plt.tight_layout()
    plt.show() #화면에 먼저그리고 save
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved heatmap:", out_png)

# ============================================
# pca_trajectory_plot
# ============================================

def pca_trajectory_plot(
    model: Transformer1Layer,
    X_np: np.ndarray,
    y_np: np.ndarray,
    class_names: List[str],
    out_png: str,
    num_samples: int = 12,
    seed: int = 0,
    use_cls: bool = False,
):
    """
    Figure2 스타일 PCA trajectory:
    - 각 샘플의 시퀀스 토큰 임베딩을 PCA 2D로 투영
    - time(토큰 위치) 순서대로 선으로 연결해서 trajectory로 그림
    - use_cls=False: 입력 토큰들(1..T) trajectory
    - use_cls=True : CLS 포함(0..T)
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    rng = np.random.default_rng(seed)

    N = X_np.shape[0]
    if N == 0:
        print("[WARN] No samples for PCA plot.")
        return

    # 샘플 인덱스 선택(랜덤)
    pick = rng.choice(np.arange(N), size=min(num_samples, N), replace=False)

    # 1) 선택 샘플들의 token embedding 수집 (PCA fit용)
    traj_list = []   # 각 원소: (T_or_T+1, D)
    label_list = []  # 각 원소: int label

    for i in pick:
        xb = cp.asarray(X_np[i:i+1], dtype=cp.float32)   # (1,T,D_in)
        logits, cache = model.forward(xb)

        # cache["h_after_ffn"] : (B, T+1, d_model)  (CLS 포함)
        h = cache["h_after_ffn"]  # cupy
        h_np = cp.asnumpy(h[0])   # (T+1, d_model)

        if use_cls:
            seq = h_np  # (T+1, d_model) CLS 포함
        else:
            seq = h_np[1:, :]  # (T, d_model) CLS 제외(토큰만)

        traj_list.append(seq)
        label_list.append(int(y_np[i]))

    # PCA는 (전체 샘플의 모든 time-point)을 하나의 point cloud로 보고 fit
    all_points = np.concatenate(traj_list, axis=0)  # (sum_len, d_model)
    pca = PCA(n_components=2, random_state=seed)
    all_2d = pca.fit_transform(all_points)

    # 2) 다시 샘플별로 잘라서 plot
    plt.figure(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(class_names))))

    st = 0
    for seq, lab in zip(traj_list, label_list):
        L = seq.shape[0]
        pts = all_2d[st:st+L, :]
        st += L

        c = colors[lab % len(colors)]
        name = class_names[lab] if lab < len(class_names) else str(lab)

        # trajectory line
        plt.plot(pts[:, 0], pts[:, 1], "-", color=c, alpha=0.75, linewidth=2)

        # start / end marker
        plt.scatter(pts[0, 0], pts[0, 1], marker="o", s=40, color=c, alpha=0.9)
        plt.scatter(pts[-1, 0], pts[-1, 1], marker="X", s=60, color=c, alpha=0.9)

        # end에 라벨 텍스트(겹치면 지저분해질 수 있어 alpha 낮게)
        plt.text(pts[-1, 0], pts[-1, 1], name, fontsize=9, color=c, alpha=0.9)

    plt.title(f"PCA Trajectory Plot ({'CLS+tokens' if use_cls else 'tokens only'})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show() #화면에 그리고 save
    plt.savefig(out_png, dpi=150)
    plt.close()
    print("Saved PCA trajectory plot:", out_png)


# =========================
# Config / Main
# =========================
@dataclass
class TrainConfig:
    npz_path: str = "smell_dataset.npz"
    meta_path: str = "smell_dataset_meta.json"

    seed: int = 43 #42
    batch_size: int = 64
    epochs: int = 2 #10
    lr: float = 1e-3
    weight_decay: float = 1e-4

    d_model: int = 128
    dim_ff: int = 256

    ckpt_path: str = "checkpoints/transformer_cupy_min.npz"
    plots_dir: str = "plots_cupy"


def main():
    cfg = TrainConfig()
    os.makedirs(cfg.plots_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # load data
    pack = np.load(cfg.npz_path)
    X = pack["X"].astype(np.float32)  # (N,T,D)
    y = pack["y"].astype(np.int32)    # (N,)

    meta = json.load(open(cfg.meta_path, "r", encoding="utf-8"))
    classes = meta["classes"]
    C = len(classes)

    tr_idx, te_idx = make_split_80_20(X.shape[0], seed=cfg.seed)
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    d_in = X.shape[2]
    model = Transformer1Layer(d_in=d_in, d_model=cfg.d_model, dim_ff=cfg.dim_ff, num_classes=C)

    # optimizer
    P, G = model.params_and_grads()
    opt = AdamW(P, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # logs
    train_loss_hist, train_acc_hist = [], []
    test_loss_hist, test_acc_hist = [], []
    epoch_hist = []

    best_test_acc = -1.0

    for ep in range(cfg.epochs):
        # epoch train (shuffle and iterate all)
        total_loss = 0.0
        total_acc = 0.0
        total = 0

        for xb, yb in tqdm(epoch_batches(X_tr, y_tr, batch_size=cfg.batch_size, seed=cfg.seed + ep, shuffle=True),
                           desc=f"epoch {ep+1}/{cfg.epochs}"):
            xb_cp = cp.asarray(xb, dtype=cp.float32)
            yb_cp = cp.asarray(yb, dtype=cp.int32)

            model.zero_grads()

            logits, cache = model.forward(xb_cp)
            loss, dlogits, acc = softmax_ce_loss(logits, yb_cp)

            # backward
            model.backward(dlogits, cache)

            # update
            P, G = model.params_and_grads()
            opt.step(P, G)

            bsz = yb.shape[0]
            total_loss += float(loss.get()) * bsz
            total_acc += float(acc.get()) * bsz
            total += bsz

        tr_loss = total_loss / max(1, total)
        tr_acc = total_acc / max(1, total)

        te_loss, te_acc, y_true, y_pred = evaluate(model, X_te, y_te, batch_size=cfg.batch_size)
        print(f"[Epochs: {ep + 1}, Train loss={tr_loss:.4f}, Train acc={tr_acc:.4f}, TEST(20%) loss={te_loss:.4f} acc={te_acc:.4f}")
        
        epoch_hist.append(ep + 1)
        train_loss_hist.append(tr_loss)
        train_acc_hist.append(tr_acc)
        test_loss_hist.append(te_loss)
        test_acc_hist.append(te_acc)

        # save best
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            save_params(model, meta={"classes": classes, "d_in": d_in, "d_model": cfg.d_model, "dim_ff": cfg.dim_ff}, path=cfg.ckpt_path)
    
    # ---- final evaluation on held-out test(20%) ----
    te_loss, te_acc, y_true, y_pred = evaluate(model, X_te, y_te, batch_size=cfg.batch_size)
    print(f"[RESULT] TEST(20%) loss={te_loss:.4f} acc={te_acc:.4f}")
    
    # final confusion matrix + heatmap
    _, _, y_true, y_pred = evaluate(model, X_te, y_te, batch_size=cfg.batch_size)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    print("[INFO] Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    
    np.save(os.path.join(cfg.plots_dir, "confusion_matrix.npy"), cm)
    save_confusion_heatmap(cm, classes, os.path.join(cfg.plots_dir, "confusion_matrix.png"))

   
    
    # plot curves
    #plt.figure(figsize=(12, 8))

    #plt.subplot(2, 1, 1)
    plt.plot(epoch_hist, train_loss_hist, label="train_loss")
    plt.plot(epoch_hist, test_loss_hist, label="test_loss")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.show()
    
    #plt.subplot(2, 1, 2)
    plt.plot(epoch_hist, train_acc_hist, label="train_acc")
    plt.plot(epoch_hist, test_acc_hist, label="test_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.show()

    plt.plot(epoch_hist, train_acc_hist, label="train_acc")
    plt.plot(epoch_hist, test_acc_hist, label="test_acc")
    plt.plot(epoch_hist, train_loss_hist, label="train_loss")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.show()
    
    #plt.tight_layout()
    out = os.path.join(cfg.plots_dir, "train_curves.png")
    plt.savefig(out, dpi=150)
    #plt.show()
    print("Saved curves:", out)

    # save metrics json
    metrics = {
        "epoch": epoch_hist,
        "train_loss": train_loss_hist,
        "train_acc": train_acc_hist,
        "test_loss": test_loss_hist,
        "test_acc": test_acc_hist,
        "best_test_acc": best_test_acc,
    }
    with open(os.path.join(cfg.plots_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved metrics:", os.path.join(cfg.plots_dir, "metrics.json"))

    # =========================
    # PCA trajectory plot (Figure2 style)
    # =========================
    pca_trajectory_plot(
        model=model,
        X_np=X_te,
        y_np=y_te,
        class_names=classes,
        out_png=os.path.join(cfg.plots_dir, "pca_trajectory_tokens.png"),
        num_samples=12,   # 원하는 만큼
        seed=cfg.seed,
        use_cls=False     # 토큰(시간축) trajectory
    )

    # (선택) CLS까지 포함한 trajectory도 저장
    pca_trajectory_plot(
        model=model,
        X_np=X_te,
        y_np=y_te,
        class_names=classes,
        out_png=os.path.join(cfg.plots_dir, "pca_trajectory_with_cls.png"),
        num_samples=12,
        seed=cfg.seed,
        use_cls=True
    )



if __name__ == "__main__":
    main()