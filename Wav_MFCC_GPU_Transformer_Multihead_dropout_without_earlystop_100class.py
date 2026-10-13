#[Wav_MFCC_GPU_Transformer_Multihead_dropout_without_earlystop_100class.py]

import os, glob, hashlib, random
import numpy as np
import librosa
try:
    import cupy as cp
    print("CuPy:", cp.__version__)
    print("driverGetVersion:", cp.cuda.runtime.driverGetVersion())   # ex) 12060
    print("runtimeGetVersion:", cp.cuda.runtime.runtimeGetVersion()) # ex) 12080
    print(cp.cuda.runtime.runtimeGetVersion(), cp.cuda.runtime.driverGetVersion())
    print(cp.arange(3))
    cp.cuda.Device(0).use()
    x = cp.random.randn(100, 100)
    print(x.device)   # cuda:0 이면 성공
except Exception as e:
    print("ERROR:", e)

import os, glob, hashlib, random, json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================
# 0) NumPy / CuPy 자동 선택 (실연산 테스트 후 폴백)
# =============================
GPU = False
xp = None

def _select_xp():
    global GPU, xp
    if os.environ.get("FORCE_CPU", "0") == "1":
        import numpy as _np
        xp = _np
        GPU = False
        print("[INFO] FORCE_CPU=1 -> Using NumPy (CPU)")
        return

    try:
        import cupy as _cp
        # 실제 연산 테스트 (nvrtc 문제 있으면 여기서 예외)
        _ = (_cp.arange(1) + 1).sum().item()
        xp = _cp
        GPU = True
        print("[INFO] Using CuPy (GPU)")
    except Exception as e:
        import numpy as _np
        xp = _np
        GPU = False
        print("[WARN] CuPy unusable -> fallback to NumPy (CPU)")
        print("       reason:", repr(e))

_select_xp()


def to_numpy(a):
    """xp array -> numpy array (GPU면 cupy->numpy)"""
    if GPU:
        return xp.asnumpy(a)
    return a


# =============================
# Augmentation (wav 단계)
# =============================
def add_noise_snr(y: np.ndarray, snr_db: float, rng: np.random.Generator):
    """SNR(dB) 기준 가우시안 노이즈 추가"""
    y = y.astype(np.float32)
    sig_rms = np.sqrt(np.mean(y**2) + 1e-12)
    noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=y.shape).astype(np.float32)
    return y + noise

def time_shift_zero_pad(y: np.ndarray, shift_samples: int):
    """시간 쉬프트(0 padding)"""
    y = y.astype(np.float32)
    if shift_samples == 0:
        return y
    n = y.shape[0]
    if abs(shift_samples) >= n:
        return np.zeros_like(y)

    out = np.zeros_like(y)
    if shift_samples > 0:
        out[shift_samples:] = y[: n - shift_samples]
    else:
        s = -shift_samples
        out[: n - s] = y[s:]
    return out

def augment_waveform(
    y: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    p_noise: float = 0.7,
    snr_db_min: float = 10.0,
    snr_db_max: float = 30.0,
    p_shift: float = 0.7,
    max_shift_ms: float = 80.0,
):
    """확률적으로 time-shift + noise (강도 랜덤)"""
    y_aug = y.astype(np.float32)

    if rng.random() < p_shift:
        max_shift = int(sr * (max_shift_ms / 1000.0))
        shift = int(rng.integers(-max_shift, max_shift + 1))
        y_aug = time_shift_zero_pad(y_aug, shift)

    if rng.random() < p_noise:
        snr_db = float(rng.uniform(snr_db_min, snr_db_max))
        y_aug = add_noise_snr(y_aug, snr_db, rng)

    # 클리핑 방지(선택)
    peak = np.max(np.abs(y_aug)) + 1e-12
    if peak > 1.0:
        y_aug = y_aug / peak

    return y_aug

def plot_mfcc_compare(mfcc_orig, mfcc_aug, title="MFCC: Original vs Augmented"):
    """
    mfcc_orig, mfcc_aug: (T, F)
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(mfcc_orig.T, aspect="auto", origin="lower")
    plt.title("Original MFCC (F x T)")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coeff")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.imshow(mfcc_aug.T, aspect="auto", origin="lower")
    plt.title("Augmented MFCC (F x T)")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coeff")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
# =============================
# 1) wav -> MFCC(고정크기 T x F) + 캐싱
# =============================
def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def _hash_path(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def wav_to_mfcc_fixed(
    wav_path,
    sr=16000,
    n_mfcc=40,
    n_fft=400,
    hop_length=160,
    T=64,
    augment=False,
    aug_cfg=None,
    rng=None,
):
    """
    return: (T, n_mfcc) numpy float32
    augment=True면 wav 단계에서 증강 후 MFCC 추출
    """
    y, _ = librosa.load(wav_path, sr=sr)

    if augment:
        if rng is None:
            rng = np.random.default_rng()
        if aug_cfg is None:
            aug_cfg = {}
        y = augment_waveform(y, sr, rng=rng, **aug_cfg)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )  # (n_mfcc, time)
    mfcc = mfcc.T  # (time, n_mfcc)

    t = mfcc.shape[0]
    if t >= T:
        mfcc = mfcc[:T, :]
    else:
        mfcc = np.pad(mfcc, ((0, T - t), (0, 0)), mode="constant", constant_values=0.0)

    return mfcc.astype(np.float32)

def cache_mfcc_dataset(
    paths,
    labels=None,  # ★ 라벨을 같이 받음 (train 증강 시 필수)
    cache_dir="cache_mfcc",
    sr=16000,
    n_mfcc=40,
    n_fft=400,
    hop_length=160,
    T=64,
    force_recompute=False,
    # --- augmentation options ---
    augment=False,
    num_aug=2,
    aug_prob=0.8,
    aug_cfg=None,
    seed=42,
):
    """
    캐시 생성:
    - augment=False: 원본만 저장
    - augment=True : 원본 + (확률적으로) 증강본 num_aug개 추가 저장
    반환:
    - (cached_paths, cached_labels)  (labels가 None이면 labels는 None)
    """
    _safe_mkdir(cache_dir)
    rng = np.random.default_rng(seed)

    if aug_cfg is None:
        aug_cfg = dict(
            p_noise=0.7, snr_db_min=10.0, snr_db_max=30.0,
            p_shift=0.7, max_shift_ms=80.0
        )

    cached_paths = []
    cached_labels = [] if labels is not None else None

    if labels is None:
        labels_iter = [None] * len(paths)
    else:
        if len(labels) != len(paths):
            raise ValueError("labels 길이는 paths 길이와 같아야 합니다.")
        labels_iter = labels

    for p, ylab in zip(paths, labels_iter):
        key = _hash_path(os.path.abspath(p))

        # 1) 원본 캐시
        out0 = os.path.join(cache_dir, f"{key}_orig_T{T}_F{n_mfcc}.npy")
        if force_recompute or (not os.path.exists(out0)):
            mfcc0 = wav_to_mfcc_fixed(
                p, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, T=T,
                augment=False
            )
            np.save(out0, mfcc0)

        cached_paths.append(out0)
        if cached_labels is not None:
            cached_labels.append(int(ylab))

        # 2) 증강 캐시
        if augment:
            for k in range(num_aug):
                if rng.random() > aug_prob:
                    continue
                outk = os.path.join(cache_dir, f"{key}_aug{k}_T{T}_F{n_mfcc}.npy")
                if force_recompute or (not os.path.exists(outk)):
                    mfcck = wav_to_mfcc_fixed(
                        p, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, T=T,
                        augment=True, aug_cfg=aug_cfg, rng=rng
                    )
                    np.save(outk, mfcck)

                cached_paths.append(outk)
                if cached_labels is not None:
                    cached_labels.append(int(ylab))

    return cached_paths, cached_labels


# =============================
# 2) 데이터셋 로드 & split
#   - total 100% -> train 80% / test 20%
#   - train 내부에서 val = train의 30%
# =============================
def load_dataset_from_folders(data_dir, word_list=None, seed=42):
    random.seed(seed)
    if word_list is None:
        word_list = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
    word2idx = {w: i for i, w in enumerate(word_list)}
    idx2word = {i: w for w, i in word2idx.items()}

    paths, labels = [], []
    for w in word_list:
        wavs = glob.glob(os.path.join(data_dir, w, "*.wav"))
        for p in wavs:
            paths.append(p)
            labels.append(word2idx[w])

    tmp = list(zip(paths, labels))
    random.shuffle(tmp)
    paths, labels = zip(*tmp) if tmp else ([], [])
    return list(paths), list(labels), idx2word, word2idx

def split_train_val_test(paths, labels, seed=42, train_ratio=0.8, val_ratio_in_train=0.1):
#def split_train_val_test(paths, labels, seed=42, train_ratio=0.7, val_ratio_in_train=0.3):
    """
    total -> train(80%) / test(20%)
    train -> train_sub(100% of train data) / val(30% random selection from train data)
    """
    rng = random.Random(seed)
    idx = list(range(len(paths)))
    rng.shuffle(idx)

    n_total = len(idx)
    n_train = int(n_total * train_ratio)
    train_idx = idx[:n_train] #80%
    test_idx  = idx[n_train:] #20%

    # train 내부 val 분리
    rng.shuffle(train_idx)
    n_val = int(len(train_idx) * val_ratio_in_train)
    val_idx = train_idx[:n_val] #10%
    train_sub_idx = train_idx[n_val:]
    #train_sub_idx = train_idx[:] #100%

    def pick(idxs):
        return [paths[i] for i in idxs], [labels[i] for i in idxs]

    tr_p, tr_y = pick(train_sub_idx)
    va_p, va_y = pick(val_idx)
    te_p, te_y = pick(test_idx)

    return tr_p, tr_y, va_p, va_y, te_p, te_y


# =============================
# 3) MFCC 정규화 + Padding Mask
# =============================
def make_frame_mask(x_np, eps=1e-8):
    """
    x_np: (T,F) 또는 (B,T,F)

    MFCC 캐시에서 zero-padding된 프레임을 검출한다.
    실제 프레임=1, padding=0
    """
    return (
        np.max(np.abs(x_np), axis=-1) > eps
    ).astype(np.float32)


def compute_norm_stats(npy_paths):
    """
    Train 데이터의 실제 프레임만 사용하여 feature별 mean/std 계산.
    Padding 행은 통계 계산에서 제외한다.
    """
    sum_ = None
    sumsq = None
    count = 0

    for path in npy_paths:
        x = np.load(path).astype(np.float32)  # (T,F)
        mask = make_frame_mask(x).astype(bool)

        valid_x = x[mask]

        if valid_x.shape[0] == 0:
            continue

        if sum_ is None:
            sum_ = valid_x.sum(axis=0, dtype=np.float64)
            sumsq = (valid_x * valid_x).sum(
                axis=0, dtype=np.float64
            )
        else:
            sum_ += valid_x.sum(axis=0, dtype=np.float64)
            sumsq += (valid_x * valid_x).sum(
                axis=0, dtype=np.float64
            )

        count += valid_x.shape[0]

    if count == 0:
        raise ValueError("정규화 통계를 계산할 유효 MFCC 프레임이 없습니다.")

    mean = sum_ / count
    var = sumsq / count - mean * mean

    mean = mean.astype(np.float32)
    std = np.sqrt(
        np.maximum(var, 1e-8)
    ).astype(np.float32)

    return mean, std


def apply_norm_np(x_np, mean, std, frame_mask=None):
    """
    실제 프레임만 정규화하고 padding 프레임은 다시 0으로 유지한다.
    """
    if frame_mask is None:
        frame_mask = make_frame_mask(x_np)

    normalized = (x_np - mean) / std

    if normalized.ndim == 2:
        normalized *= frame_mask[:, None]
    elif normalized.ndim == 3:
        normalized *= frame_mask[:, :, None]
    else:
        raise ValueError(
            f"지원하지 않는 MFCC 차원입니다: {normalized.shape}"
        )

    return normalized.astype(np.float32)


def save_norm_stats(path, mean, std):
    np.savez(path, mean=mean, std=std)


def load_norm_stats(path):
    data = np.load(path)
    return (
        data["mean"].astype(np.float32),
        data["std"].astype(np.float32)
    )


# =============================
# 4) 미니배치 로더
# =============================
def iterate_minibatches(
    npy_paths,
    labels,
    mean,
    std,
    batch_size=32,
    shuffle=True,
    seed=42
):
    indices = np.arange(len(npy_paths))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        X_list = []
        mask_list = []

        for index in batch_indices:
            x_raw = np.load(
                npy_paths[index]
            ).astype(np.float32)

            frame_mask = make_frame_mask(x_raw)

            x_normalized = apply_norm_np(
                x_raw,
                mean,
                std,
                frame_mask=frame_mask
            )

            X_list.append(x_normalized)
            mask_list.append(frame_mask)

        X = np.stack(X_list, axis=0).astype(np.float32)
        frame_masks = np.stack(
            mask_list, axis=0
        ).astype(np.float32)

        y = np.asarray(
            [labels[index] for index in batch_indices],
            dtype=np.int64
        )

        yield (
            xp.asarray(X),
            xp.asarray(y),
            xp.asarray(frame_masks)
        )

# =============================
# 5) Softmax / Label Smoothing
# =============================
def softmax2d(logits):
    z = logits - xp.max(
        logits, axis=1, keepdims=True
    )
    exp_z = xp.exp(z)

    return exp_z / xp.sum(
        exp_z, axis=1, keepdims=True
    )


def cross_entropy_loss(
    logits,
    y,
    label_smoothing=0.0
):
    """
    학습:
        label_smoothing=0.05

    평가:
        label_smoothing=0.0
    """
    probs = softmax2d(logits)

    B, C = logits.shape
    smoothing = float(label_smoothing)

    targets = xp.full(
        (B, C),
        smoothing / C,
        dtype=xp.float32
    )

    targets[xp.arange(B), y] += 1.0 - smoothing

    loss = -xp.sum(
        targets * xp.log(probs + 1e-9)
    ) / B

    return loss, probs, targets


def accuracy_from_probs(probs, y):
    pred = xp.argmax(probs, axis=1)

    return float(
        xp.mean(
            (pred == y).astype(xp.float32)
        )
    )


# =============================
# 6) AdamW
# =============================
class AdamW:
    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    ):
        self.params = params
        self.lr = float(lr)
        self.b1, self.b2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        self.t = 0

        self.m = [
            xp.zeros_like(param)
            for param in params
        ]

        self.v = [
            xp.zeros_like(param)
            for param in params
        ]

    def step(self, grads):
        self.t += 1

        bias_correction = (
            xp.sqrt(1.0 - self.b2 ** self.t)
            / (1.0 - self.b1 ** self.t)
        )

        step_lr = self.lr * bias_correction

        for index, (param, grad) in enumerate(
            zip(self.params, grads)
        ):
            self.m[index] = (
                self.b1 * self.m[index]
                + (1.0 - self.b1) * grad
            )

            self.v[index] = (
                self.b2 * self.v[index]
                + (1.0 - self.b2) * (grad * grad)
            )

            adam_update = (
                self.m[index]
                / (
                    xp.sqrt(self.v[index])
                    + self.eps
                )
            )

            param -= step_lr * adam_update

            # 행렬/embedding 파라미터에만 weight decay 적용
            # bias와 LayerNorm vector는 제외
            if param.ndim >= 2:
                param -= (
                    self.lr
                    * self.weight_decay
                    * param
                )

# =============================
# Dropout
# =============================
def dropout_forward(
    x,
    dropout_p=0.1,
    training=True
):
    if (not training) or dropout_p <= 0.0:
        return x, None

    keep_prob = 1.0 - dropout_p

    mask = (
        xp.random.random(x.shape) < keep_prob
    ).astype(xp.float32)

    output = x * mask / keep_prob

    return output, mask


def dropout_backward(
    dout,
    mask,
    dropout_p=0.1
):
    if mask is None or dropout_p <= 0.0:
        return dout

    keep_prob = 1.0 - dropout_p

    return dout * mask / keep_prob    
    
# =============================
# 7) Multi-Head Transformer Classifier
#    NumPy / CuPy 공용 수동 역전파
# =============================

def xavier_uniform(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return xp.random.uniform(
        -limit, limit, size=(in_dim, out_dim)
    ).astype(xp.float32)


def softmax_last(x):
    z = x - xp.max(x, axis=-1, keepdims=True)
    e = xp.exp(z)
    return e / xp.sum(e, axis=-1, keepdims=True)


def layer_norm_forward(x, gamma, beta, eps=1e-5):
    """
    x: (..., D)
    gamma, beta: (D,)
    """
    mean = xp.mean(x, axis=-1, keepdims=True)
    var = xp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    inv_std = 1.0 / xp.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    out = gamma * x_hat + beta

    cache = {
        "x_hat": x_hat,
        "inv_std": inv_std,
        "gamma": gamma,
    }
    return out, cache


def layer_norm_backward(dout, cache):
    """
    LayerNorm backward.
    """
    x_hat = cache["x_hat"]
    inv_std = cache["inv_std"]
    gamma = cache["gamma"]

    D = dout.shape[-1]

    dgamma = xp.sum(
        dout * x_hat,
        axis=tuple(range(dout.ndim - 1))
    )
    dbeta = xp.sum(
        dout,
        axis=tuple(range(dout.ndim - 1))
    )

    dxhat = dout * gamma

    sum_dxhat = xp.sum(dxhat, axis=-1, keepdims=True)
    sum_dxhat_xhat = xp.sum(
        dxhat * x_hat,
        axis=-1,
        keepdims=True
    )

    dx = (
        inv_std / D
        * (
            D * dxhat
            - sum_dxhat
            - x_hat * sum_dxhat_xhat
        )
    )

    return dx, dgamma, dbeta


class TransformerEncoderLayer:
    def __init__(
        self,
        d_model=64,
        num_heads=4,
        d_ff=128,
        dropout_p=0.15,
        seed=0
    ):
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model은 num_heads로 나누어 떨어져야 합니다."
            )

        xp.random.seed(seed)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_ff = d_ff
        self.dropout_p = dropout_p

        self.Wq = xavier_uniform(d_model, d_model)
        self.Wk = xavier_uniform(d_model, d_model)
        self.Wv = xavier_uniform(d_model, d_model)
        self.Wo = xavier_uniform(d_model, d_model)

        self.bq = xp.zeros(d_model, dtype=xp.float32)
        self.bk = xp.zeros(d_model, dtype=xp.float32)
        self.bv = xp.zeros(d_model, dtype=xp.float32)
        self.bo = xp.zeros(d_model, dtype=xp.float32)

        self.gamma1 = xp.ones(
            d_model, dtype=xp.float32
        )
        self.beta1 = xp.zeros(
            d_model, dtype=xp.float32
        )

        self.W1 = xavier_uniform(d_model, d_ff)
        self.b1 = xp.zeros(d_ff, dtype=xp.float32)
        self.W2 = xavier_uniform(d_ff, d_model)
        self.b2 = xp.zeros(d_model, dtype=xp.float32)

        self.gamma2 = xp.ones(
            d_model, dtype=xp.float32
        )
        self.beta2 = xp.zeros(
            d_model, dtype=xp.float32
        )

        self.params = [
            self.Wq, self.Wk, self.Wv, self.Wo,
            self.bq, self.bk, self.bv, self.bo,
            self.gamma1, self.beta1,
            self.W1, self.b1, self.W2, self.b2,
            self.gamma2, self.beta2,
        ]

        self.cache = None
        self.attention_weights = None

    def _split_heads(self, x):
        B, L, _ = x.shape

        return (
            x.reshape(
                B,
                L,
                self.num_heads,
                self.head_dim
            )
            .transpose(0, 2, 1, 3)
        )

    def _merge_heads(self, x):
        B, H, L, Dh = x.shape

        return (
            x.transpose(0, 2, 1, 3)
            .reshape(B, L, H * Dh)
        )

    def attention_forward(
        self,
        x,
        valid_mask
    ):
        """
        valid_mask: (B,L)
        CLS와 실제 MFCC 토큰=1, padding=0
        """
        B, L, D = x.shape

        q_linear = x @ self.Wq + self.bq
        k_linear = x @ self.Wk + self.bk
        v_linear = x @ self.Wv + self.bv

        Q = self._split_heads(q_linear)
        K = self._split_heads(k_linear)
        V = self._split_heads(v_linear)

        scale = np.sqrt(self.head_dim)

        scores = (
            Q @ K.transpose(0, 1, 3, 2)
        ) / scale

        # Key padding mask
        key_mask = valid_mask[:, None, None, :]

        scores = xp.where(
            key_mask > 0,
            scores,
            xp.asarray(-1e9, dtype=xp.float32)
        )

        attention = softmax_last(scores)

        heads = attention @ V
        merged = self._merge_heads(heads)
        output = merged @ self.Wo + self.bo

        self.attention_weights = attention

        cache = {
            "x": x,
            "Q": Q,
            "K": K,
            "V": V,
            "attention": attention,
            "merged": merged,
        }

        return output, cache

    def attention_backward(self, dout, cache):
        x = cache["x"]
        Q = cache["Q"]
        K = cache["K"]
        V = cache["V"]
        attention = cache["attention"]
        merged = cache["merged"]

        _, _, D = x.shape

        dout_2d = dout.reshape(-1, D)
        merged_2d = merged.reshape(-1, D)

        dWo = merged_2d.T @ dout_2d
        dbo = xp.sum(dout_2d, axis=0)

        dmerged = dout @ self.Wo.T
        dheads = self._split_heads(dmerged)

        dattention = (
            dheads
            @ V.transpose(0, 1, 3, 2)
        )

        dV = (
            attention.transpose(0, 1, 3, 2)
            @ dheads
        )

        dscores = attention * (
            dattention
            - xp.sum(
                dattention * attention,
                axis=-1,
                keepdims=True
            )
        )

        scale = np.sqrt(self.head_dim)

        dQ = (dscores @ K) / scale

        dK = (
            dscores.transpose(0, 1, 3, 2)
            @ Q
        ) / scale

        dq_linear = self._merge_heads(dQ)
        dk_linear = self._merge_heads(dK)
        dv_linear = self._merge_heads(dV)

        x_2d = x.reshape(-1, D)
        dq_2d = dq_linear.reshape(-1, D)
        dk_2d = dk_linear.reshape(-1, D)
        dv_2d = dv_linear.reshape(-1, D)

        dWq = x_2d.T @ dq_2d
        dWk = x_2d.T @ dk_2d
        dWv = x_2d.T @ dv_2d

        dbq = xp.sum(dq_2d, axis=0)
        dbk = xp.sum(dk_2d, axis=0)
        dbv = xp.sum(dv_2d, axis=0)

        dx = (
            dq_linear @ self.Wq.T
            + dk_linear @ self.Wk.T
            + dv_linear @ self.Wv.T
        )

        grads = [
            dWq, dWk, dWv, dWo,
            dbq, dbk, dbv, dbo,
        ]

        return dx, grads

    def ffn_forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = xp.maximum(z1, 0.0)
        output = a1 @ self.W2 + self.b2

        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
        }

        return output, cache

    def ffn_backward(self, dout, cache):
        x = cache["x"]
        z1 = cache["z1"]
        a1 = cache["a1"]

        D = x.shape[-1]
        FF = a1.shape[-1]

        dout_2d = dout.reshape(-1, D)
        a1_2d = a1.reshape(-1, FF)
        x_2d = x.reshape(-1, D)

        dW2 = a1_2d.T @ dout_2d
        db2 = xp.sum(dout_2d, axis=0)

        da1 = dout @ self.W2.T

        dz1 = da1 * (
            z1 > 0
        ).astype(xp.float32)

        dz1_2d = dz1.reshape(-1, FF)

        dW1 = x_2d.T @ dz1_2d
        db1 = xp.sum(dz1_2d, axis=0)

        dx = dz1 @ self.W1.T

        return dx, [dW1, db1, dW2, db2]

    def forward(
        self,
        x,
        valid_mask,
        training=True
    ):
        # Pre-Norm Multi-Head Attention
        norm1, ln1_cache = layer_norm_forward(
            x,
            self.gamma1,
            self.beta1
        )

        attention_output, attention_cache = (
            self.attention_forward(
                norm1,
                valid_mask
            )
        )

        attention_output, attention_drop_mask = (
            dropout_forward(
                attention_output,
                dropout_p=self.dropout_p,
                training=training
            )
        )

        residual1 = x + attention_output

        # Pre-Norm FFN
        norm2, ln2_cache = layer_norm_forward(
            residual1,
            self.gamma2,
            self.beta2
        )

        ffn_output, ffn_cache = self.ffn_forward(norm2)

        ffn_output, ffn_drop_mask = dropout_forward(
            ffn_output,
            dropout_p=self.dropout_p,
            training=training
        )

        output = residual1 + ffn_output

        self.cache = {
            "ln1": ln1_cache,
            "attention": attention_cache,
            "attention_drop_mask": attention_drop_mask,
            "ln2": ln2_cache,
            "ffn": ffn_cache,
            "ffn_drop_mask": ffn_drop_mask,
        }

        return output

    def backward(self, dout):
        # FFN residual
        dresidual1 = dout.copy()

        dffn_output = dropout_backward(
            dout,
            self.cache["ffn_drop_mask"],
            dropout_p=self.dropout_p
        )

        dnorm2, ffn_grads = self.ffn_backward(
            dffn_output,
            self.cache["ffn"]
        )

        (
            dresidual1_ln,
            dgamma2,
            dbeta2
        ) = layer_norm_backward(
            dnorm2,
            self.cache["ln2"]
        )

        dresidual1 += dresidual1_ln

        # Attention residual
        dx = dresidual1.copy()

        dattn_output = dropout_backward(
            dresidual1,
            self.cache["attention_drop_mask"],
            dropout_p=self.dropout_p
        )

        dnorm1, attention_grads = (
            self.attention_backward(
                dattn_output,
                self.cache["attention"]
            )
        )

        dx_ln, dgamma1, dbeta1 = (
            layer_norm_backward(
                dnorm1,
                self.cache["ln1"]
            )
        )

        dx += dx_ln

        grads = [
            *attention_grads,
            dgamma1,
            dbeta1,
            *ffn_grads,
            dgamma2,
            dbeta2,
        ]

        return dx, grads

#class TransformerClassifier:
class TransformerClassifier:
    def __init__(
        self,
        input_dim=40,
        seq_len=64,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=1,
        #num_classes=35,
        num_classes=100,
        dropout_p=0.15,
        seed=42
    ):
        xp.random.seed(seed)

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.training = True

        self.W_in = xavier_uniform(
            input_dim,
            d_model
        )
        self.b_in = xp.zeros(
            d_model,
            dtype=xp.float32
        )

        self.cls_token = (
            xp.random.randn(
                1, 1, d_model
            ).astype(xp.float32)
            * 0.02
        )

        self.pos_embed = (
            xp.random.randn(
                1,
                seq_len + 1,
                d_model
            ).astype(xp.float32)
            * 0.02
        )

        self.layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_p=dropout_p,
                seed=seed + layer_index + 1
            )
            for layer_index in range(num_layers)
        ]

        self.gamma_final = xp.ones(
            d_model,
            dtype=xp.float32
        )
        self.beta_final = xp.zeros(
            d_model,
            dtype=xp.float32
        )

        self.W_cls = xavier_uniform(
            d_model,
            num_classes
        )
        self.b_cls = xp.zeros(
            num_classes,
            dtype=xp.float32
        )

        self.params = [
            self.W_in,
            self.b_in,
            self.cls_token,
            self.pos_embed,
        ]

        for layer in self.layers:
            self.params.extend(layer.params)

        self.params.extend([
            self.gamma_final,
            self.beta_final,
            self.W_cls,
            self.b_cls,
        ])

        self.cache = None
        self.last_sequence = None
        self.last_cls = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(
        self,
        X,
        frame_mask=None
    ):
        """
        X: (B,T,F)
        frame_mask: (B,T)
        
        """
        B, T, _ = X.shape

        if T != self.seq_len:
            raise ValueError(
                f"입력 T={T}, 설정 seq_len={self.seq_len}"
            )

        if frame_mask is None:
            frame_mask = xp.ones(
                (B, T),
                dtype=xp.float32
            )

        projected = X @ self.W_in + self.b_in

        cls = xp.broadcast_to(
            self.cls_token,
            (B, 1, self.d_model)
        )

        sequence = xp.concatenate(
            [cls, projected],
            axis=1
        )

        sequence = sequence + self.pos_embed

        # Input/embedding dropout
        sequence, input_drop_mask = dropout_forward(
            sequence,
            dropout_p=self.dropout_p,
            training=self.training
        )

        cls_mask = xp.ones(
            (B, 1),
            dtype=xp.float32
        )

        valid_mask = xp.concatenate(
            [cls_mask, frame_mask],
            axis=1
        )

        for layer in self.layers:
            sequence = layer.forward(
                sequence,
                valid_mask=valid_mask,
                training=self.training
            )

        self.last_sequence = sequence

        cls_hidden = sequence[:, 0, :]

        cls_norm, final_ln_cache = layer_norm_forward(
            cls_hidden,
            self.gamma_final,
            self.beta_final
        )

        # Classification head 직전 dropout
        cls_dropped, cls_drop_mask = dropout_forward(
            cls_norm,
            dropout_p=self.dropout_p,
            training=self.training
        )

        logits = (
            cls_dropped @ self.W_cls
            + self.b_cls
        )

        self.last_cls = cls_dropped

        self.cache = {
            "X": X,
            "frame_mask": frame_mask,
            "valid_mask": valid_mask,
            "input_drop_mask": input_drop_mask,
            "final_ln": final_ln_cache,
            "cls_drop_mask": cls_drop_mask,
        }

        return logits

    def backward(
        self,
        probs,
        targets,
        grad_clip=1.0
    ):
        """
        targets는 label smoothing이 적용된 (B,C) 행렬.
        """
        X = self.cache["X"]
        B = probs.shape[0]

        dlogits = (probs - targets) / B

        dW_cls = self.last_cls.T @ dlogits
        db_cls = xp.sum(dlogits, axis=0)

        dcls_dropped = dlogits @ self.W_cls.T

        dcls_norm = dropout_backward(
            dcls_dropped,
            self.cache["cls_drop_mask"],
            dropout_p=self.dropout_p
        )

        (
            dcls_hidden,
            dgamma_final,
            dbeta_final
        ) = layer_norm_backward(
            dcls_norm,
            self.cache["final_ln"]
        )

        dsequence = xp.zeros_like(
            self.last_sequence
        )
        dsequence[:, 0, :] = dcls_hidden

        layer_grads_reversed = []

        for layer in reversed(self.layers):
            dsequence, layer_grads = (
                layer.backward(dsequence)
            )
            layer_grads_reversed.append(
                layer_grads
            )

        dsequence = dropout_backward(
            dsequence,
            self.cache["input_drop_mask"],
            dropout_p=self.dropout_p
        )

        dpos_embed = xp.sum(
            dsequence,
            axis=0,
            keepdims=True
        )

        dcls_token = xp.sum(
            dsequence[:, 0:1, :],
            axis=0,
            keepdims=True
        )

        dprojected = dsequence[:, 1:, :]

        # padding 위치의 projection gradient 제거
        dprojected *= (
            self.cache["frame_mask"][:, :, None]
        )

        X_2d = X.reshape(
            -1,
            self.input_dim
        )

        dprojected_2d = dprojected.reshape(
            -1,
            self.d_model
        )

        dW_in = X_2d.T @ dprojected_2d
        db_in = xp.sum(
            dprojected_2d,
            axis=0
        )

        grads = [
            dW_in,
            db_in,
            dcls_token,
            dpos_embed,
        ]

        for layer_grads in reversed(
            layer_grads_reversed
        ):
            grads.extend(layer_grads)

        grads.extend([
            dgamma_final,
            dbeta_final,
            dW_cls,
            db_cls,
        ])

        grads = [
            xp.clip(
                grad,
                -grad_clip,
                grad_clip
            )
            for grad in grads
        ]

        return grads

    def save_params(self, path):
        save_dict = {
            f"param_{index:03d}": to_numpy(param)
            for index, param in enumerate(self.params)
        }

        np.savez(path, **save_dict)

    def load_params(self, path):
        data = np.load(path)

        if len(data.files) != len(self.params):
            raise ValueError(
                "저장 모델과 현재 모델의 파라미터 수가 다릅니다."
            )

        loaded_params = [
            xp.asarray(data[f"param_{index:03d}"])
            for index in range(len(self.params))
        ]

        pointer = 0

        self.W_in = loaded_params[pointer]
        pointer += 1

        self.b_in = loaded_params[pointer]
        pointer += 1

        self.cls_token = loaded_params[pointer]
        pointer += 1

        self.pos_embed = loaded_params[pointer]
        pointer += 1

        for layer in self.layers:
            count = len(layer.params)

            layer.params = loaded_params[
                pointer:pointer + count
            ]

            (
                layer.Wq,
                layer.Wk,
                layer.Wv,
                layer.Wo,
                layer.bq,
                layer.bk,
                layer.bv,
                layer.bo,
                layer.gamma1,
                layer.beta1,
                layer.W1,
                layer.b1,
                layer.W2,
                layer.b2,
                layer.gamma2,
                layer.beta2,
            ) = layer.params

            pointer += count

        self.gamma_final = loaded_params[pointer]
        pointer += 1

        self.beta_final = loaded_params[pointer]
        pointer += 1

        self.W_cls = loaded_params[pointer]
        pointer += 1

        self.b_cls = loaded_params[pointer]

        self.params = loaded_params
        
# =============================
# 8) 평가 / Confusion Matrix
# =============================
def eval_loop(
    model,
    npy_paths,
    labels,
    mean,
    std,
    batch_size=64
):
    previous_mode = model.training
    model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for X, y, frame_mask in iterate_minibatches(
        npy_paths,
        labels,
        mean,
        std,
        batch_size=batch_size,
        shuffle=False
    ):
        logits = model.forward(
            X,
            frame_mask=frame_mask
        )

        loss, probs, _ = cross_entropy_loss(
            logits,
            y,
            label_smoothing=0.0
        )

        acc = accuracy_from_probs(probs, y)
        n = int(y.shape[0])

        total_loss += float(loss) * n
        total_acc += acc * n
        total_n += n

    if previous_mode:
        model.train()

    return (
        total_loss / max(1, total_n),
        total_acc / max(1, total_n)
    )


def confusion_matrix(
    model,
    npy_paths,
    labels,
    mean,
    std,
    #num_classes=35,
    num_classes=100,
    batch_size=64
):
    previous_mode = model.training
    model.eval()

    cm = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64
    )

    for X, y, frame_mask in iterate_minibatches(
        npy_paths,
        labels,
        mean,
        std,
        batch_size=batch_size,
        shuffle=False
    ):
        logits = model.forward(
            X,
            frame_mask=frame_mask
        )

        pred = to_numpy(
            xp.argmax(logits, axis=1)
        ).astype(np.int64)

        true = to_numpy(y).astype(np.int64)

        for true_label, pred_label in zip(true, pred):
            cm[true_label, pred_label] += 1

    if previous_mode:
        model.train()

    return cm

def plot_confusion_matrix(cm, idx2word, title="Confusion Matrix"):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(idx2word))
    plt.xticks(ticks, [idx2word[i] for i in ticks], rotation=45, ha="right")
    plt.yticks(ticks, [idx2word[i] for i in ticks])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


# =============================
# 9) Trainer
#    - Best model 사용 안 함
#    - Early stopping 사용 안 함
#    - 모든 epoch 실행
#    - epoch마다 Validation / Test 평가
# =============================
class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        max_iter=100,
        #max_iter=2,
        batch_size=32,
        label_smoothing=0.05,
        grad_clip=1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.grad_clip = grad_clip

        self.train_loss = []
        self.train_acc = []

        self.val_loss = []
        self.val_acc = []

        self.test_loss = []
        self.test_acc = []

    def fit(
        self,
        tr_cache,
        tr_y,
        va_cache,
        va_y,
        te_cache,
        te_y,
        mean,
        std,
        seed=42
    ):
        for epoch in tqdm(range(self.max_iter)):
            # =========================
            # Train
            # =========================
            self.model.train()

            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_n = 0

            for X, y, frame_mask in iterate_minibatches(
                tr_cache,
                tr_y,
                mean,
                std,
                batch_size=self.batch_size,
                shuffle=True,
                seed=seed + epoch
            ):
                logits = self.model.forward(
                    X,
                    frame_mask=frame_mask
                )

                loss, probs, targets = cross_entropy_loss(
                    logits,
                    y,
                    label_smoothing=self.label_smoothing
                )

                grads = self.model.backward(
                    probs,
                    targets,
                    grad_clip=self.grad_clip
                )

                self.optimizer.step(grads)

                acc = accuracy_from_probs(probs, y)
                batch_n = int(y.shape[0])

                epoch_loss += float(loss) * batch_n
                epoch_acc += acc * batch_n
                epoch_n += batch_n

            train_loss = (
                epoch_loss / max(1, epoch_n)
            )

            train_acc = (
                epoch_acc / max(1, epoch_n)
            )

            # =========================
            # Validation
            # =========================
            val_loss, val_acc = eval_loop(
                self.model,
                va_cache,
                va_y,
                mean,
                std,
                batch_size=max(
                    64,
                    self.batch_size
                )
            )

            # =========================
            # Test
            # =========================
            test_loss, test_acc = eval_loop(
                self.model,
                te_cache,
                te_y,
                mean,
                std,
                batch_size=max(
                    64,
                    self.batch_size
                )
            )

            # =========================
            # History 저장
            # =========================
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            self.val_loss.append(val_loss)
            self.val_acc.append(val_acc)

            self.test_loss.append(test_loss)
            self.test_acc.append(test_acc)

            print(
                "\n"
                f"epoch={epoch + 1:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_acc={val_acc:.4f} | "
                f"test_loss={test_loss:.6f} | "
                f"test_acc={test_acc:.4f}"
            )

        # 마지막 epoch 모델 그대로 사용
        self.model.eval()

        print(
            "\n======================================"
        )
        print(
            f"[TRAINING FINISHED] "
            f"last epoch={len(self.train_loss)}"
        )
        print(
            f"[LAST VALIDATION] "
            f"loss={self.val_loss[-1]:.4f} "
            f"acc={self.val_acc[-1]:.4f}"
        )
        print(
            f"[LAST TEST] "
            f"loss={self.test_loss[-1]:.4f} "
            f"acc={self.test_acc[-1]:.4f}"
        )
        print(
            "======================================"
        )

        return self

# =============================
def plot_transformer_pca_trajectory(
    model,
    npy_paths,
    labels,
    mean,
    std,
    idx2word,
    out_path,
    num_samples=12,
    include_cls=False,
    seed=42
):
    """
    Transformer 최종 Encoder 출력의 PCA trajectory.

    include_cls=False:
        MFCC 프레임 토큰만 사용

    include_cls=True:
        CLS + MFCC 프레임 토큰 사용
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print(
            "[WARN] PCA를 사용하려면 "
            "'pip install scikit-learn'을 실행하세요."
        )
        return

    if len(npy_paths) == 0:
        print("[WARN] PCA plot 데이터가 없습니다.")
        return

    rng = np.random.default_rng(seed)
    count = min(num_samples, len(npy_paths))

    selected = rng.choice(
        len(npy_paths),
        size=count,
        replace=False
    )

    trajectories = []
    selected_labels = []

    for index in selected:
        x_np = np.load(npy_paths[index]).astype(np.float32)
        x_np = apply_norm_np(x_np, mean, std)

        x_raw = np.load(
            npy_paths[index]
        ).astype(np.float32)

        frame_mask_np = make_frame_mask(x_raw)

        x_np = apply_norm_np(
            x_raw,
            mean,
            std,
            frame_mask=frame_mask_np
        )

        X = xp.asarray(x_np[None, :, :])
        frame_mask = xp.asarray(
            frame_mask_np[None, :]
        )

        previous_mode = model.training
        model.eval()

        model.forward(
            X,
            frame_mask=frame_mask
        )

        if previous_mode:
            model.train()

        sequence_np = to_numpy(
            model.last_sequence[0]
        )

        #if not include_cls:
        #    sequence_np = sequence_np[1:, :]
        if include_cls:
            valid_length = int(frame_mask_np.sum())
            sequence_np = sequence_np[:valid_length + 1]
        else:
            valid_length = int(frame_mask_np.sum())
            sequence_np = sequence_np[1:valid_length + 1]
        
        trajectories.append(sequence_np)
        selected_labels.append(int(labels[index]))

    all_points = np.concatenate(
        trajectories,
        axis=0
    )

    pca = PCA(n_components=2)
    all_points_2d = pca.fit_transform(all_points)

    plt.figure(figsize=(11, 8))

    colors = plt.cm.tab10(
        np.linspace(0, 1, max(10, len(idx2word)))
    )

    offset = 0
    used_labels = set()

    for trajectory, label in zip(
        trajectories,
        selected_labels
    ):
        length = trajectory.shape[0]
        points = all_points_2d[
            offset:offset + length
        ]
        offset += length

        color = colors[label % len(colors)]
        class_name = idx2word[label]

        legend_label = (
            class_name
            if class_name not in used_labels
            else None
        )
        used_labels.add(class_name)

        plt.plot(
            points[:, 0],
            points[:, 1],
            "-",
            color=color,
            linewidth=1.6,
            alpha=0.75,
            label=legend_label
        )

        # 시작
        plt.scatter(
            points[0, 0],
            points[0, 1],
            marker="o",
            color=color,
            s=45
        )

        # 종료
        plt.scatter(
            points[-1, 0],
            points[-1, 1],
            marker="X",
            color=color,
            s=70
        )

    mode = "CLS + MFCC tokens" if include_cls else "MFCC tokens"

    plt.title(
        f"Transformer PCA Trajectory: {mode}\n"
        f"PC1={pca.explained_variance_ratio_[0] * 100:.1f}%, "
        f"PC2={pca.explained_variance_ratio_[1] * 100:.1f}%"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.show()

    print(f"[INFO] PCA trajectory saved -> {out_path}")


def plot_attention_heatmap(
    model,
    npy_path,
    mean,
    std,
    out_path,
    layer_index=-1,
    head_index=0
):
    """
    특정 Transformer layer/head의 Attention weight를 출력합니다.

    Parameters
    ----------
    model:
        학습된 TransformerClassifier

    npy_path:
        MFCC 캐시 파일 경로

    mean, std:
        Train 데이터에서 계산한 정규화 통계

    out_path:
        Heatmap 저장 경로

    layer_index:
        출력할 Transformer layer 번호
        -1이면 마지막 layer

    head_index:
        출력할 Multi-Head Attention head 번호
    """

    # ---------------------------------
    # 1. MFCC 로드
    # ---------------------------------
    x_raw = np.load(
        npy_path
    ).astype(np.float32)

    # 실제 프레임=1, padding=0
    frame_mask_np = make_frame_mask(x_raw)

    # padding을 제외하고 정규화
    x_np = apply_norm_np(
        x_raw,
        mean,
        std,
        frame_mask=frame_mask_np
    )

    # Batch 차원 추가
    X = xp.asarray(
        x_np[None, :, :]
    )

    frame_mask = xp.asarray(
        frame_mask_np[None, :]
    )

    # ---------------------------------
    # 2. 평가 모드에서 forward
    # ---------------------------------
    previous_mode = model.training
    model.eval()

    model.forward(
        X,
        frame_mask=frame_mask
    )

    # ---------------------------------
    # 3. Attention 추출
    # ---------------------------------
    attention = model.layers[
        layer_index
    ].attention_weights

    if attention is None:
        raise RuntimeError(
            "Attention weight가 없습니다. "
            "model.forward()가 정상적으로 실행되었는지 확인하세요."
        )

    # attention shape:
    # (B, num_heads, L, L)
    num_heads = int(attention.shape[1])

    if not 0 <= head_index < num_heads:
        raise ValueError(
            f"head_index={head_index}는 잘못된 값입니다. "
            f"사용 가능한 범위는 0~{num_heads - 1}입니다."
        )

    attention_np = to_numpy(
        attention[0, head_index]
    ).astype(np.float32)

    # 실제 MFCC 프레임 수 + CLS token 1개
    valid_length = int(
        frame_mask_np.sum()
    ) + 1

    # Padding 영역 제거
    attention_np = attention_np[
        :valid_length,
        :valid_length
    ]

    # forward가 끝난 뒤 기존 mode 복원
    if previous_mode:
        model.train()

    # ---------------------------------
    # 4. Heatmap 출력
    # ---------------------------------
    plt.figure(figsize=(9, 8))

    image = plt.imshow(
        attention_np,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0.0
    )

    plt.colorbar(
        image,
        label="Attention weight"
    )

    plt.xlabel("Key position")
    plt.ylabel("Query position")

    plt.title(
        f"Attention Heatmap: "
        f"layer={layer_index}, "
        f"head={head_index}"
    )

    # 0번은 CLS token
    tick_count = min(
        valid_length,
        10
    )

    tick_positions = np.linspace(
        0,
        valid_length - 1,
        tick_count,
        dtype=int
    )

    tick_labels = [
        "CLS" if position == 0 else str(position)
        for position in tick_positions
    ]

    plt.xticks(
        tick_positions,
        tick_labels
    )

    plt.yticks(
        tick_positions,
        tick_labels
    )

    plt.tight_layout()

    out_parent = os.path.dirname(out_path)

    if out_parent:
        os.makedirs(
            out_parent,
            exist_ok=True
        )

    plt.savefig(
        out_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"[INFO] attention heatmap saved -> {out_path}"
    )    
    
 # =============================   

def plot_curves(
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    test_loss,
    test_acc,
    out_dir=None
):
    epochs = np.arange(
        1,
        len(train_loss) + 1
    )

    # =============================
    # Loss / Accuracy
    # =============================
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        train_loss,
        label="train_loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="val_loss"
    )

    plt.plot(
        epochs,
        test_loss,
        label="test_loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation / Test Loss")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        train_acc,
        label="train_acc"
    )

    plt.plot(
        epochs,
        val_acc,
        label="val_acc"
    )

    plt.plot(
        epochs,
        test_acc,
        label="test_acc"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train / Validation / Test Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(
            out_dir,
            exist_ok=True
        )

        out_path = os.path.join(
            out_dir,
            "training_validation_test_curves.png"
        )

        plt.savefig(
            out_path,
            dpi=160,
            bbox_inches="tight"
        )

        print(
            f"[INFO] training curves saved -> {out_path}"
        )

    plt.show()
    plt.close()

    # =============================
    # Test 지표만 별도 출력
    # =============================
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        test_loss,
        color="red",
        label="test_loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.title("Test Loss per Epoch")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        test_acc,
        color="green",
        label="test_acc"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy per Epoch")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if out_dir is not None:
        test_out_path = os.path.join(
            out_dir,
            "test_metrics_per_epoch.png"
        )

        plt.savefig(
            test_out_path,
            dpi=160,
            bbox_inches="tight"
        )

        print(
            f"[INFO] test curves saved -> "
            f"{test_out_path}"
        )

    plt.show()
    plt.close()

# =============================
# 10) main
# =============================
def main():
    # ---- config ----
    data_dir = "data_100\speech"
    #data_dir = "data_speech_35"

    # 기존 강한 증강 cache와 섞이지 않도록 별도 폴더 사용
    cache_dir = "cache_mfcc_transformer_regularized"

    out_dir = "speech_artifacts"
    _safe_mkdir(out_dir)

    # MFCC
    T = 64
    F = 40

    # 학습
    batch_size = 32
    max_iter = 100
    learning_rate = 3e-4
    weight_decay = 1e-4
    label_smoothing = 0.05
    dropout_p = 0.15
    
    grad_clip = 1.0
    seed = 42

    # 축소된 Transformer
    d_model = 64
    num_heads = 4
    d_ff = 128
    num_layers = 1

    # ---- load dataset ----
    paths, labels, idx2word, word2idx = (
        load_dataset_from_folders(
            data_dir,
            seed=seed
        )
    )

    #if len(word2idx) != 10:
    #    raise ValueError(
    #        f"classes=10 이어야 합니다. 현재: {len(word2idx)}"
    #    )
    if len(word2idx) != 100:
        raise ValueError(
            f"classes=100 이어야 합니다. 현재: {len(word2idx)}"
        )

    (
        tr_p,
        tr_y,
        va_p,
        va_y,
        te_p,
        te_y
    ) = split_train_val_test(
        paths,
        labels,
        seed=seed,
        train_ratio=0.8,
        val_ratio_in_train=0.1
    )

    print(
        f"[INFO] total={len(paths)} | "
        f"train={len(tr_p)} | "
        f"val={len(va_p)} | "
        f"test={len(te_p)}"
    )
    print("[INFO] mapping:", word2idx)

    # ---- 완화된 augmentation ----
    aug_cfg = {
        "p_noise": 0.5,
        "snr_db_min": 15.0,
        "snr_db_max": 35.0,
        "p_shift": 0.5,
        "max_shift_ms": 50.0,
    }

    tr_cache, tr_y_aug = cache_mfcc_dataset(
        tr_p,
        tr_y,
        cache_dir=cache_dir,
        n_mfcc=F,
        T=T,
        augment=False, # 증강 사용 안함
        force_recompute=True,
        seed=seed
    )

    va_cache, va_y2 = cache_mfcc_dataset(
        va_p,
        va_y,
        cache_dir=cache_dir,
        n_mfcc=F,
        T=T,
        augment=False, # 증강 사용 안함
        force_recompute=True,
        seed=seed
    )

    te_cache, te_y2 = cache_mfcc_dataset(
        te_p,
        te_y,
        cache_dir=cache_dir,
        n_mfcc=F,
        T=T,
        augment=False, # 증강 사용 안함
        force_recompute=True,
        seed=seed
    )

    print(
        f"[INFO] augmented train={len(tr_cache)} | "
        f"val={len(va_cache)} | "
        f"test={len(te_cache)}"
    )

    # ---- normalization ----
    norm_path = os.path.join(
        out_dir,
        "mfcc_norm_stats.npz"
    )

    mean, std = compute_norm_stats(tr_cache)
    save_norm_stats(norm_path, mean, std)

    print(
        f"[INFO] saved norm stats -> {norm_path}"
    )

    # ---- model ----
    model = TransformerClassifier(
        input_dim=F,
        seq_len=T,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        #num_classes=35,
        num_classes=100,
        dropout_p=dropout_p,
        seed=seed
    )

    optimizer = AdamW(
        model.params,
        lr=learning_rate,
        weight_decay=weight_decay
    )


    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        max_iter=max_iter,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        grad_clip=grad_clip
    )

    trainer.fit(
        tr_cache=tr_cache,
        tr_y=tr_y_aug,
        va_cache=va_cache,
        va_y=va_y2,
        te_cache=te_cache,
        te_y=te_y2,
        mean=mean,
        std=std,
        seed=seed
    )

    # Trainer에서 매 epoch Test 평가가 끝났으므로
    # 마지막 epoch 결과를 그대로 사용
    test_loss = trainer.test_loss[-1]
    test_acc = trainer.test_acc[-1]

    val_loss = trainer.val_loss[-1]
    val_acc = trainer.val_acc[-1]

    print(
        "\n======================================"
    )

    print(
        f"[FINAL MODEL] epoch={max_iter}"
    )

    print(
        f"[VALIDATION] "
        f"loss={val_loss:.4f} "
        f"acc={val_acc:.4f}"
    )

    print(
        f"[TEST] "
        f"loss={test_loss:.4f} "
        f"acc={test_acc:.4f}"
    )

    print(
        "======================================"
    )

    # 마지막 epoch 모델 저장
    final_model_path = os.path.join(
        out_dir,
        "transformer_multihead_last_epoch_params.npz"
    )

    model.save_params(final_model_path)

    print(
        f"[INFO] last epoch model saved -> "
        f"{final_model_path}"
    )

    # ---- Confusion Matrix ----
    cm = confusion_matrix(
        model,
        te_cache,
        te_y2,
        mean,
        std,
        
        #num_classes=35,
        num_classes=100,
        batch_size=max(64, batch_size)
    )

    print(
        "[INFO] Confusion Matrix "
        "(rows=true, cols=pred):"
    )
    print(cm)

    plot_confusion_matrix(
        cm,
        idx2word,
        title="Best Transformer Test Confusion Matrix"
    )

    # ---- 최적 모델을 최종 모델명으로도 저장 ----
    #final_model_path = os.path.join(
    #    out_dir,
    #    "transformer_multihead_params.npz"
    #)

    #model.save_params(final_model_path)

    meta_path = os.path.join(
        out_dir,
        "transformer_multihead_meta.json"
    )

    meta = {
        "model_type": "TransformerClassifier",
        "T": T,
        "F": F,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "dropout_p": dropout_p,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "last_epoch": max_iter,
        "last_val_loss": val_loss,
        "last_val_acc": val_acc,
        "last_test_loss": test_loss,
        "last_test_acc": test_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "num_classes": 100,
        "idx2word": {
            str(key): value
            for key, value in idx2word.items()
        },
        "word2idx": word2idx,
        "norm_path": os.path.basename(norm_path),
        "model_path": os.path.basename(
            final_model_path
        ),
    }

    with open(
        meta_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            meta,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"[INFO] saved model -> {final_model_path}"
    )
    print(
        f"[INFO] saved meta -> {meta_path}"
    )

    plot_curves(
        trainer.train_loss,
        trainer.train_acc,
        trainer.val_loss,
        trainer.val_acc,
        trainer.test_loss,
        trainer.test_acc,
        out_dir=out_dir
    )
    # 이하 MFCC 비교, PCA, Attention heatmap 코드 실행
    
    # ---- (DEBUG) 원본 vs 증강 MFCC 비교 플롯 ----
    DEBUG_PLOT_AUG = False #True  # 필요 없으면 False

    if DEBUG_PLOT_AUG and len(tr_p) > 0:
        rng = np.random.default_rng(seed)
        sample_wav = tr_p[0]  # 원하는 파일로 바꿔도 됨

        mfcc_orig = wav_to_mfcc_fixed(
            sample_wav, sr=16000, n_mfcc=F, n_fft=400, hop_length=160, T=T,
            augment=False
        )
        mfcc_aug = wav_to_mfcc_fixed(
            sample_wav, sr=16000, n_mfcc=F, n_fft=400, hop_length=160, T=T,
            augment=True, aug_cfg=aug_cfg, rng=rng
        )

        plot_mfcc_compare(mfcc_orig, mfcc_aug, title=f"MFCC Compare: {os.path.basename(sample_wav)}")
        
    # ---- Transformer PCA trajectory ----
    plot_transformer_pca_trajectory(
        model=model,
        npy_paths=te_cache,
        labels=te_y2,
        mean=mean,
        std=std,
        idx2word=idx2word,
        out_path=os.path.join(
            out_dir,
            "pca_trajectory_tokens.png"
        ),
        num_samples=12,
        include_cls=False,
        seed=seed
    )

    plot_transformer_pca_trajectory(
        model=model,
        npy_paths=te_cache,
        labels=te_y2,
        mean=mean,
        std=std,
        idx2word=idx2word,
        out_path=os.path.join(
            out_dir,
            "pca_trajectory_with_cls.png"
        ),
        num_samples=12,
        include_cls=True,
        seed=seed
    )

    # ---- Attention heatmap ----
    if len(te_cache) > 0:
        plot_attention_heatmap(
            model=model,
            npy_path=te_cache[0],
            mean=mean,
            std=std,
            out_path=os.path.join(
                out_dir,
                "attention_heatmap.png"
            ),
            layer_index=-1,
            head_index=0
        )

if __name__ == "__main__":
    main()



