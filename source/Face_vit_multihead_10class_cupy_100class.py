# Face_vit_multihead_10class_cupy_100class.py
# ============================================================
# CuPy 기반 Vision Transformer Multi-Head 이미지 분류
#
# 기능
# 1. 100개 클래스 이미지 폴더 로딩
# 2. Stratified Train/Test 분할
# 3. PIL 기반 On-the-fly 데이터 증강
# 4. 이미지 Patch Embedding
# 5. CLS Token + Positional Embedding
# 6. Multi-Head Self-Attention
# 7. Transformer Encoder
# 8. Softmax Cross Entropy
# 9. CuPy 수동 역전파
# 10. AdamW 최적화
# 11. Epoch별 Train/Test Loss와 Accuracy
# 12. 학습 곡선, Confusion Matrix, 클래스별 정확도
# ============================================================

import os
import json
import pickle
from collections import OrderedDict

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm


# ============================================================
# GPU 선택
# ============================================================
GPU_DEVICE_ID = 0
cp.cuda.Device(GPU_DEVICE_ID).use()


# ============================================================
# 재현성 설정
# ============================================================
SEED = 42

np.random.seed(SEED)
cp.random.seed(SEED)


# ============================================================
# Utility
# ============================================================
def scalar_to_float(value):
    if hasattr(value, "get"):
        return float(value.get())
    return float(value)


def softmax(x):
    x = x - cp.max(x, axis=-1, keepdims=True)

    exp_x = cp.exp(x)

    return exp_x / cp.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


def softmax_cross_entropy(logits, labels):
    """
    Parameters
    ----------
    logits : CuPy array, shape (B, C)
    labels : CuPy array, shape (B,)

    Returns
    -------
    loss : scalar
    dlogits : shape (B, C)
    """
    labels = labels.astype(cp.int32)

    probabilities = softmax(logits)

    batch_size = logits.shape[0]

    correct_probabilities = probabilities[
        cp.arange(batch_size),
        labels
    ]

    loss = -cp.mean(
        cp.log(correct_probabilities + 1e-12)
    )

    dlogits = probabilities.copy()

    dlogits[
        cp.arange(batch_size),
        labels
    ] -= 1.0

    dlogits /= batch_size

    return loss, dlogits


# ============================================================
# Linear
# ============================================================
class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(
            2.0 / (in_features + out_features)
        )

        self.W = (
            cp.random.randn(
                in_features,
                out_features
            ).astype(cp.float32)
            * scale
        )

        self.b = cp.zeros(
            out_features,
            dtype=cp.float32
        )

        self.x = None

        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x

        return cp.matmul(x, self.W) + self.b

    def backward(self, dout):
        x_shape = self.x.shape

        x_2d = self.x.reshape(
            -1,
            x_shape[-1]
        )

        dout_2d = dout.reshape(
            -1,
            dout.shape[-1]
        )

        self.dW = cp.matmul(
            x_2d.T,
            dout_2d
        )

        self.db = cp.sum(
            dout_2d,
            axis=0
        )

        dx = cp.matmul(
            dout,
            self.W.T
        )

        return dx


# ============================================================
# Layer Normalization
# ============================================================
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = cp.ones(
            dim,
            dtype=cp.float32
        )

        self.beta = cp.zeros(
            dim,
            dtype=cp.float32
        )

        self.eps = eps

        self.x_hat = None
        self.inv_std = None

        self.dgamma = None
        self.dbeta = None

    def forward(self, x):
        mean = cp.mean(
            x,
            axis=-1,
            keepdims=True
        )

        centered = x - mean

        variance = cp.mean(
            centered ** 2,
            axis=-1,
            keepdims=True
        )

        inv_std = 1.0 / cp.sqrt(
            variance + self.eps
        )

        x_hat = centered * inv_std

        self.x_hat = x_hat
        self.inv_std = inv_std

        return (
            self.gamma * x_hat
            + self.beta
        )

    def backward(self, dout):
        feature_dim = dout.shape[-1]

        parameter_axes = tuple(
            range(dout.ndim - 1)
        )

        self.dgamma = cp.sum(
            dout * self.x_hat,
            axis=parameter_axes
        )

        self.dbeta = cp.sum(
            dout,
            axis=parameter_axes
        )

        dx_hat = dout * self.gamma

        sum_dx_hat = cp.sum(
            dx_hat,
            axis=-1,
            keepdims=True
        )

        sum_dx_hat_xhat = cp.sum(
            dx_hat * self.x_hat,
            axis=-1,
            keepdims=True
        )

        dx = (
            self.inv_std / feature_dim
            * (
                feature_dim * dx_hat
                - sum_dx_hat
                - self.x_hat * sum_dx_hat_xhat
            )
        )

        return dx


# ============================================================
# GELU
# ============================================================
class GELU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x

        coefficient = np.sqrt(
            2.0 / np.pi
        )

        inner = coefficient * (
            x + 0.044715 * x ** 3
        )

        return 0.5 * x * (
            1.0 + cp.tanh(inner)
        )

    def backward(self, dout):
        x = self.x

        coefficient = np.sqrt(
            2.0 / np.pi
        )

        inner = coefficient * (
            x + 0.044715 * x ** 3
        )

        tanh_inner = cp.tanh(inner)

        dinner_dx = coefficient * (
            1.0
            + 3.0 * 0.044715 * x ** 2
        )

        derivative = (
            0.5 * (1.0 + tanh_inner)
            + 0.5
            * x
            * (1.0 - tanh_inner ** 2)
            * dinner_dx
        )

        return dout * derivative


# ============================================================
# Dropout
# ============================================================
class Dropout:
    def __init__(self, dropout_ratio=0.1):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if (
            not train_flg
            or self.dropout_ratio <= 0.0
        ):
            self.mask = None
            return x

        keep_probability = (
            1.0 - self.dropout_ratio
        )

        self.mask = (
            cp.random.rand(*x.shape)
            < keep_probability
        ).astype(x.dtype)

        return (
            x
            * self.mask
            / keep_probability
        )

    def backward(self, dout):
        if self.mask is None:
            return dout

        keep_probability = (
            1.0 - self.dropout_ratio
        )

        return (
            dout
            * self.mask
            / keep_probability
        )


# ============================================================
# Multi-Head Self-Attention
# ============================================================
class MultiHeadSelfAttention:
    def __init__(
        self,
        d_model=128,
        num_heads=4
    ):
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model은 num_heads로 "
                "나누어져야 합니다."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = (
            d_model // num_heads
        )

        self.scale = (
            1.0 / np.sqrt(self.head_dim)
        )

        init_scale = np.sqrt(
            2.0 / d_model
        )

        self.Wq = (
            cp.random.randn(
                d_model,
                d_model
            ).astype(cp.float32)
            * init_scale
        )

        self.Wk = (
            cp.random.randn(
                d_model,
                d_model
            ).astype(cp.float32)
            * init_scale
        )

        self.Wv = (
            cp.random.randn(
                d_model,
                d_model
            ).astype(cp.float32)
            * init_scale
        )

        self.Wo = (
            cp.random.randn(
                d_model,
                d_model
            ).astype(cp.float32)
            * init_scale
        )

        self.bq = cp.zeros(
            d_model,
            dtype=cp.float32
        )

        self.bk = cp.zeros(
            d_model,
            dtype=cp.float32
        )

        self.bv = cp.zeros(
            d_model,
            dtype=cp.float32
        )

        self.bo = cp.zeros(
            d_model,
            dtype=cp.float32
        )

        self.x = None
        self.q = None
        self.k = None
        self.v = None

        self.attention_weights = None
        self.concat = None

        self.dWq = None
        self.dWk = None
        self.dWv = None
        self.dWo = None

        self.dbq = None
        self.dbk = None
        self.dbv = None
        self.dbo = None

    def _split_heads(self, x):
        batch_size, token_count, _ = x.shape

        return x.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim
        ).transpose(
            0, 2, 1, 3
        )

    def _combine_heads(self, x):
        batch_size = x.shape[0]
        token_count = x.shape[2]

        return x.transpose(
            0, 2, 1, 3
        ).reshape(
            batch_size,
            token_count,
            self.d_model
        )

    def forward(self, x):
        self.x = x

        q_linear = (
            cp.matmul(x, self.Wq)
            + self.bq
        )

        k_linear = (
            cp.matmul(x, self.Wk)
            + self.bk
        )

        v_linear = (
            cp.matmul(x, self.Wv)
            + self.bv
        )

        q = self._split_heads(q_linear)
        k = self._split_heads(k_linear)
        v = self._split_heads(v_linear)

        scores = cp.matmul(
            q,
            k.transpose(0, 1, 3, 2)
        )

        scores *= self.scale

        attention_weights = softmax(
            scores
        )

        context = cp.matmul(
            attention_weights,
            v
        )

        concat = self._combine_heads(
            context
        )

        output = (
            cp.matmul(concat, self.Wo)
            + self.bo
        )

        self.q = q
        self.k = k
        self.v = v

        self.attention_weights = (
            attention_weights
        )

        self.concat = concat

        return output

    def backward(self, dout):
        batch_size, token_count, _ = (
            dout.shape
        )

        concat_2d = self.concat.reshape(
            -1,
            self.d_model
        )

        dout_2d = dout.reshape(
            -1,
            self.d_model
        )

        self.dWo = cp.matmul(
            concat_2d.T,
            dout_2d
        )

        self.dbo = cp.sum(
            dout_2d,
            axis=0
        )

        dconcat = cp.matmul(
            dout,
            self.Wo.T
        )

        dcontext = self._split_heads(
            dconcat
        )

        # context = attention_weights @ V
        dattention = cp.matmul(
            dcontext,
            self.v.transpose(
                0, 1, 3, 2
            )
        )

        dv = cp.matmul(
            self.attention_weights.transpose(
                0, 1, 3, 2
            ),
            dcontext
        )

        # Softmax backward
        softmax_dot = cp.sum(
            dattention
            * self.attention_weights,
            axis=-1,
            keepdims=True
        )

        dscores = (
            self.attention_weights
            * (
                dattention
                - softmax_dot
            )
        )

        dscores *= self.scale

        # scores = Q @ K^T
        dq = cp.matmul(
            dscores,
            self.k
        )

        dk = cp.matmul(
            dscores.transpose(
                0, 1, 3, 2
            ),
            self.q
        )

        dq_linear = self._combine_heads(
            dq
        )

        dk_linear = self._combine_heads(
            dk
        )

        dv_linear = self._combine_heads(
            dv
        )

        x_2d = self.x.reshape(
            -1,
            self.d_model
        )

        dq_2d = dq_linear.reshape(
            -1,
            self.d_model
        )

        dk_2d = dk_linear.reshape(
            -1,
            self.d_model
        )

        dv_2d = dv_linear.reshape(
            -1,
            self.d_model
        )

        self.dWq = cp.matmul(
            x_2d.T,
            dq_2d
        )

        self.dWk = cp.matmul(
            x_2d.T,
            dk_2d
        )

        self.dWv = cp.matmul(
            x_2d.T,
            dv_2d
        )

        self.dbq = cp.sum(
            dq_2d,
            axis=0
        )

        self.dbk = cp.sum(
            dk_2d,
            axis=0
        )

        self.dbv = cp.sum(
            dv_2d,
            axis=0
        )

        dx_q = cp.matmul(
            dq_linear,
            self.Wq.T
        )

        dx_k = cp.matmul(
            dk_linear,
            self.Wk.T
        )

        dx_v = cp.matmul(
            dv_linear,
            self.Wv.T
        )

        return dx_q + dx_k + dx_v


# ============================================================
# Feed Forward Network
# ============================================================
class FeedForward:
    def __init__(
        self,
        d_model=128,
        d_ff=256,
        dropout_ratio=0.1
    ):
        self.fc1 = Linear(
            d_model,
            d_ff
        )

        self.activation = GELU()

        self.dropout1 = Dropout(
            dropout_ratio
        )

        self.fc2 = Linear(
            d_ff,
            d_model
        )

        self.dropout2 = Dropout(
            dropout_ratio
        )

    def forward(
        self,
        x,
        train_flg=True
    ):
        x = self.fc1.forward(x)

        x = self.activation.forward(x)

        x = self.dropout1.forward(
            x,
            train_flg
        )

        x = self.fc2.forward(x)

        x = self.dropout2.forward(
            x,
            train_flg
        )

        return x

    def backward(self, dout):
        dout = self.dropout2.backward(
            dout
        )

        dout = self.fc2.backward(
            dout
        )

        dout = self.dropout1.backward(
            dout
        )

        dout = self.activation.backward(
            dout
        )

        dout = self.fc1.backward(
            dout
        )

        return dout


# ============================================================
# Transformer Encoder
# Pre-LayerNorm 구조
# ============================================================
class TransformerEncoder:
    def __init__(
        self,
        d_model=128,
        num_heads=4,
        d_ff=256,
        dropout_ratio=0.1
    ):
        self.norm1 = LayerNorm(
            d_model
        )

        self.attention = (
            MultiHeadSelfAttention(
                d_model=d_model,
                num_heads=num_heads
            )
        )

        self.attention_dropout = Dropout(
            dropout_ratio
        )

        self.norm2 = LayerNorm(
            d_model
        )

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_ratio=dropout_ratio
        )

    def forward(
        self,
        x,
        train_flg=True
    ):
        # Pre-LN Attention
        normalized_x = self.norm1.forward(
            x
        )

        attention_output = (
            self.attention.forward(
                normalized_x
            )
        )

        attention_output = (
            self.attention_dropout.forward(
                attention_output,
                train_flg
            )
        )

        hidden = x + attention_output

        # Pre-LN FFN
        normalized_hidden = (
            self.norm2.forward(hidden)
        )

        ffn_output = self.ffn.forward(
            normalized_hidden,
            train_flg
        )

        output = hidden + ffn_output

        return output

    def backward(self, dout):
        # output = hidden + ffn_output
        dhidden_residual = dout
        dffn_output = dout

        dnormalized_hidden = (
            self.ffn.backward(
                dffn_output
            )
        )

        dhidden_normalized = (
            self.norm2.backward(
                dnormalized_hidden
            )
        )

        dhidden = (
            dhidden_residual
            + dhidden_normalized
        )

        # hidden = x + attention_output
        dx_residual = dhidden
        dattention_output = dhidden

        dattention_output = (
            self.attention_dropout.backward(
                dattention_output
            )
        )

        dnormalized_x = (
            self.attention.backward(
                dattention_output
            )
        )

        dx_normalized = (
            self.norm1.backward(
                dnormalized_x
            )
        )

        dx = (
            dx_residual
            + dx_normalized
        )

        return dx


# ============================================================
# Vision Transformer Multi-Head
# ============================================================
class VisionTransformerMultiHead:
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        d_model=128,
        num_heads=4,
        d_ff=256,
        num_layers=2,
        #num_classes=10,
        num_classes=100,
        dropout_ratio=0.1
    ):
        if image_size % patch_size != 0:
            raise ValueError(
                "image_size는 patch_size로 "
                "나누어져야 합니다."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        grid_size = (
            image_size // patch_size
        )

        self.num_patches = (
            grid_size * grid_size
        )

        patch_dim = (
            in_channels
            * patch_size
            * patch_size
        )

        self.patch_projection = Linear(
            patch_dim,
            d_model
        )

        self.cls_token = (
            cp.random.randn(
                1,
                1,
                d_model
            ).astype(cp.float32)
            * 0.02
        )

        self.pos_embedding = (
            cp.random.randn(
                1,
                self.num_patches + 1,
                d_model
            ).astype(cp.float32)
            * 0.02
        )

        self.embedding_dropout = Dropout(
            dropout_ratio
        )

        self.encoders = [
            TransformerEncoder(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_ratio=dropout_ratio
            )
            for _ in range(num_layers)
        ]

        self.final_norm = LayerNorm(
            d_model
        )

        self.classifier = Linear(
            d_model,
            num_classes
        )

        self.d_cls_token = None
        self.d_pos_embedding = None

        self.last_loss = None

        self.params = OrderedDict()
        self.grads = OrderedDict()

        self._register_params()

    def _register_params(self):
        self.params["patch_W"] = (
            self.patch_projection.W
        )

        self.params["patch_b"] = (
            self.patch_projection.b
        )

        self.params["cls_token"] = (
            self.cls_token
        )

        self.params["pos_embedding"] = (
            self.pos_embedding
        )

        for layer_index, encoder in enumerate(
            self.encoders
        ):
            prefix = (
                f"encoder{layer_index}"
            )

            self.params[
                f"{prefix}_ln1_gamma"
            ] = encoder.norm1.gamma

            self.params[
                f"{prefix}_ln1_beta"
            ] = encoder.norm1.beta

            self.params[
                f"{prefix}_Wq"
            ] = encoder.attention.Wq

            self.params[
                f"{prefix}_bq"
            ] = encoder.attention.bq

            self.params[
                f"{prefix}_Wk"
            ] = encoder.attention.Wk

            self.params[
                f"{prefix}_bk"
            ] = encoder.attention.bk

            self.params[
                f"{prefix}_Wv"
            ] = encoder.attention.Wv

            self.params[
                f"{prefix}_bv"
            ] = encoder.attention.bv

            self.params[
                f"{prefix}_Wo"
            ] = encoder.attention.Wo

            self.params[
                f"{prefix}_bo"
            ] = encoder.attention.bo

            self.params[
                f"{prefix}_ln2_gamma"
            ] = encoder.norm2.gamma

            self.params[
                f"{prefix}_ln2_beta"
            ] = encoder.norm2.beta

            self.params[
                f"{prefix}_ff1_W"
            ] = encoder.ffn.fc1.W

            self.params[
                f"{prefix}_ff1_b"
            ] = encoder.ffn.fc1.b

            self.params[
                f"{prefix}_ff2_W"
            ] = encoder.ffn.fc2.W

            self.params[
                f"{prefix}_ff2_b"
            ] = encoder.ffn.fc2.b

        self.params[
            "final_ln_gamma"
        ] = self.final_norm.gamma

        self.params[
            "final_ln_beta"
        ] = self.final_norm.beta

        self.params[
            "head_W"
        ] = self.classifier.W

        self.params[
            "head_b"
        ] = self.classifier.b

    def _patchify(self, x):
        """
        Input:
            x: (B, C, H, W)

        Output:
            patches: (B, N, patch_dim)
        """
        batch_size, channels, height, width = (
            x.shape
        )

        if (
            height != self.image_size
            or width != self.image_size
        ):
            raise ValueError(
                f"입력 이미지 크기는 "
                f"{self.image_size}×"
                f"{self.image_size}여야 합니다. "
                f"현재 입력은 "
                f"{height}×{width}입니다."
            )

        patch_size = self.patch_size

        grid_h = (
            height // patch_size
        )

        grid_w = (
            width // patch_size
        )

        patches = x.reshape(
            batch_size,
            channels,
            grid_h,
            patch_size,
            grid_w,
            patch_size
        )

        patches = patches.transpose(
            0, 2, 4, 1, 3, 5
        )

        patches = patches.reshape(
            batch_size,
            grid_h * grid_w,
            channels
            * patch_size
            * patch_size
        )

        return patches

    def forward(
        self,
        x,
        train_flg=False
    ):
        batch_size = x.shape[0]

        patches = self._patchify(x)

        patch_tokens = (
            self.patch_projection.forward(
                patches
            )
        )

        cls_tokens = cp.broadcast_to(
            self.cls_token,
            (
                batch_size,
                1,
                self.d_model
            )
        )

        tokens = cp.concatenate(
            [
                cls_tokens,
                patch_tokens
            ],
            axis=1
        )

        tokens = (
            tokens
            + self.pos_embedding
        )

        tokens = (
            self.embedding_dropout.forward(
                tokens,
                train_flg
            )
        )

        for encoder in self.encoders:
            tokens = encoder.forward(
                tokens,
                train_flg
            )

        cls_output = tokens[:, 0, :]

        cls_output = (
            self.final_norm.forward(
                cls_output
            )
        )

        logits = self.classifier.forward(
            cls_output
        )

        return logits

    def predict(
        self,
        x,
        train_flg=False
    ):
        return self.forward(
            x,
            train_flg=train_flg
        )

    def loss(
        self,
        x,
        labels,
        train_flg=True
    ):
        logits = self.forward(
            x,
            train_flg=train_flg
        )

        loss, _ = softmax_cross_entropy(
            logits,
            labels
        )

        return loss

    def gradient(self, x, labels):
        logits = self.forward(
            x,
            train_flg=True
        )

        loss, dlogits = (
            softmax_cross_entropy(
                logits,
                labels
            )
        )

        self.last_loss = loss

        # Classification head
        dcls = self.classifier.backward(
            dlogits
        )

        # Final LayerNorm
        dcls = self.final_norm.backward(
            dcls
        )

        batch_size = x.shape[0]

        dtokens = cp.zeros(
            (
                batch_size,
                self.num_patches + 1,
                self.d_model
            ),
            dtype=cp.float32
        )

        # 분류 출력은 CLS Token에서만 생성
        dtokens[:, 0, :] = dcls

        # Encoder 역전파
        for encoder in reversed(
            self.encoders
        ):
            dtokens = encoder.backward(
                dtokens
            )

        dtokens = (
            self.embedding_dropout.backward(
                dtokens
            )
        )

        self.d_pos_embedding = cp.sum(
            dtokens,
            axis=0,
            keepdims=True
        )

        self.d_cls_token = cp.sum(
            dtokens[:, :1, :],
            axis=0,
            keepdims=True
        )

        dpatch_tokens = (
            dtokens[:, 1:, :]
        )

        self.patch_projection.backward(
            dpatch_tokens
        )

        self._collect_grads()

        return self.grads

    def _collect_grads(self):
        self.grads = OrderedDict()

        self.grads["patch_W"] = (
            self.patch_projection.dW
        )

        self.grads["patch_b"] = (
            self.patch_projection.db
        )

        self.grads["cls_token"] = (
            self.d_cls_token
        )

        self.grads["pos_embedding"] = (
            self.d_pos_embedding
        )

        for layer_index, encoder in enumerate(
            self.encoders
        ):
            prefix = (
                f"encoder{layer_index}"
            )

            attention = encoder.attention

            self.grads[
                f"{prefix}_ln1_gamma"
            ] = encoder.norm1.dgamma

            self.grads[
                f"{prefix}_ln1_beta"
            ] = encoder.norm1.dbeta

            self.grads[
                f"{prefix}_Wq"
            ] = attention.dWq

            self.grads[
                f"{prefix}_bq"
            ] = attention.dbq

            self.grads[
                f"{prefix}_Wk"
            ] = attention.dWk

            self.grads[
                f"{prefix}_bk"
            ] = attention.dbk

            self.grads[
                f"{prefix}_Wv"
            ] = attention.dWv

            self.grads[
                f"{prefix}_bv"
            ] = attention.dbv

            self.grads[
                f"{prefix}_Wo"
            ] = attention.dWo

            self.grads[
                f"{prefix}_bo"
            ] = attention.dbo

            self.grads[
                f"{prefix}_ln2_gamma"
            ] = encoder.norm2.dgamma

            self.grads[
                f"{prefix}_ln2_beta"
            ] = encoder.norm2.dbeta

            self.grads[
                f"{prefix}_ff1_W"
            ] = encoder.ffn.fc1.dW

            self.grads[
                f"{prefix}_ff1_b"
            ] = encoder.ffn.fc1.db

            self.grads[
                f"{prefix}_ff2_W"
            ] = encoder.ffn.fc2.dW

            self.grads[
                f"{prefix}_ff2_b"
            ] = encoder.ffn.fc2.db

        self.grads[
            "final_ln_gamma"
        ] = self.final_norm.dgamma

        self.grads[
            "final_ln_beta"
        ] = self.final_norm.dbeta

        self.grads[
            "head_W"
        ] = self.classifier.dW

        self.grads[
            "head_b"
        ] = self.classifier.db

    def save_params(
        self,
        file_name="vit_multihead_params_100.pkl"
    ):
        directory = os.path.dirname(
            file_name
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        cpu_params = {
            key: cp.asnumpy(value)
            for key, value
            in self.params.items()
        }

        with open(file_name, "wb") as file:
            pickle.dump(
                cpu_params,
                file
            )

    def load_params(self, file_name):
        with open(file_name, "rb") as file:
            loaded_params = pickle.load(
                file
            )

        for key in self.params:
            if key not in loaded_params:
                raise KeyError(
                    f"저장 파일에 '{key}' "
                    f"파라미터가 없습니다."
                )

            loaded_value = cp.asarray(
                loaded_params[key],
                dtype=cp.float32
            )

            if (
                loaded_value.shape
                != self.params[key].shape
            ):
                raise ValueError(
                    f"{key} shape 불일치: "
                    f"model={self.params[key].shape}, "
                    f"file={loaded_value.shape}"
                )

            self.params[key][...] = (
                loaded_value
            )


# ============================================================
# Optimizer
# ============================================================
class SGD:
    def __init__(self, lr=0.001):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            if (
                key in grads
                and grads[key] is not None
            ):
                params[key] -= (
                    self.lr * grads[key]
                )


class AdamW:
    def __init__(
        self,
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
        grad_clip=1.0
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.weight_decay = (
            weight_decay
        )

        self.grad_clip = grad_clip

        self.iteration = 0

        self.m = {}
        self.v = {}

    def _clip_gradients(self, grads):
        if self.grad_clip is None:
            return

        total_norm_squared = cp.array(
            0.0,
            dtype=cp.float32
        )

        for grad in grads.values():
            if grad is not None:
                total_norm_squared += (
                    cp.sum(grad ** 2)
                )

        total_norm = cp.sqrt(
            total_norm_squared
        )

        clip_coefficient = cp.minimum(
            1.0,
            self.grad_clip
            / (total_norm + 1e-6)
        )

        for key in grads:
            if grads[key] is not None:
                grads[key] *= (
                    clip_coefficient
                )

    @staticmethod
    def _use_weight_decay(key):
        no_decay_patterns = (
            "_b",
            "_beta",
            "_gamma",
            "cls_token",
            "pos_embedding"
        )

        return not any(
            pattern in key
            for pattern in no_decay_patterns
        )

    def update(self, params, grads):
        self._clip_gradients(grads)

        self.iteration += 1

        beta1_correction = (
            1.0
            - self.beta1
            ** self.iteration
        )

        beta2_correction = (
            1.0
            - self.beta2
            ** self.iteration
        )

        for key, parameter in params.items():
            if key not in grads:
                continue

            grad = grads[key]

            if grad is None:
                continue

            if key not in self.m:
                self.m[key] = cp.zeros_like(
                    parameter
                )

                self.v[key] = cp.zeros_like(
                    parameter
                )

            self.m[key] = (
                self.beta1 * self.m[key]
                + (1.0 - self.beta1)
                * grad
            )

            self.v[key] = (
                self.beta2 * self.v[key]
                + (1.0 - self.beta2)
                * grad ** 2
            )

            m_hat = (
                self.m[key]
                / beta1_correction
            )

            v_hat = (
                self.v[key]
                / beta2_correction
            )

            adam_update = (
                m_hat
                / (
                    cp.sqrt(v_hat)
                    + self.eps
                )
            )

            if self._use_weight_decay(
                key
            ):
                adam_update += (
                    self.weight_decay
                    * parameter
                )

            parameter -= (
                self.lr * adam_update
            )


# ============================================================
# PIL 기반 이미지 증강
# ============================================================
class SimpleImageDataGenerator:
    def __init__(
        self,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=5.0,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        rescale=1.0 / 255.0
    ):
        self.rotation_range = (
            rotation_range
        )

        self.width_shift_range = (
            width_shift_range
        )

        self.height_shift_range = (
            height_shift_range
        )

        self.shear_range = (
            shear_range
        )

        self.zoom_range = (
            zoom_range
        )

        self.horizontal_flip = (
            horizontal_flip
        )

        self.fill_mode = fill_mode
        self.rescale = rescale

    @staticmethod
    def _uniform(low, high):
        return np.random.uniform(
            low,
            high
        )

    def augment_pil(self, image):
        width, height = image.size

        angle = self._uniform(
            -self.rotation_range,
            self.rotation_range
        )

        shift_x = int(
            self._uniform(
                -self.width_shift_range,
                self.width_shift_range
            )
            * width
        )

        shift_y = int(
            self._uniform(
                -self.height_shift_range,
                self.height_shift_range
            )
            * height
        )

        zoom = self._uniform(
            1.0 - self.zoom_range,
            1.0 + self.zoom_range
        )

        shear_degree = self._uniform(
            -self.shear_range,
            self.shear_range
        )

        image = image.rotate(
            angle,
            resample=Image.BILINEAR,
            expand=False
        )

        shear_radian = np.deg2rad(
            shear_degree
        )

        affine_matrix = (
            zoom,
            np.tan(shear_radian),
            shift_x,
            0.0,
            zoom,
            shift_y
        )

        image = image.transform(
            (width, height),
            Image.AFFINE,
            affine_matrix,
            resample=Image.BILINEAR
        )

        if (
            self.horizontal_flip
            and np.random.rand() < 0.5
        ):
            image = image.transpose(
                Image.FLIP_LEFT_RIGHT
            )

        return image

    def pil_to_chw_float(
        self,
        image,
        size=224
    ):
        image = image.resize(
            (size, size),
            resample=Image.BILINEAR
        )

        array = np.asarray(
            image.convert("RGB"),
            dtype=np.float32
        )

        if self.rescale is not None:
            array *= self.rescale

        return array.transpose(
            2, 0, 1
        )


# ============================================================
# 데이터 경로 수집
# ============================================================
def collect_paths_10class(
    dataset_path
):
    if not os.path.isdir(
        dataset_path
    ):
        raise FileNotFoundError(
            f"데이터 폴더가 없습니다: "
            f"{dataset_path}"
        )

    class_folders = [
        folder_name
        for folder_name
        in sorted(
            os.listdir(dataset_path)
        )
        if os.path.isdir(
            os.path.join(
                dataset_path,
                folder_name
            )
        )
    ]

    if len(class_folders) != 100:
        print(
            "[경고] 클래스 폴더 개수가 "
            f"100개가 아닙니다: "
            f"{len(class_folders)}"
        )

        print(
            "클래스 폴더:",
            class_folders
        )

    valid_extensions = (
        ".bmp",
        ".png",
        ".jpg",
        ".jpeg"
    )

    paths = []
    labels = []

    for label_index, class_name in enumerate(
        class_folders
    ):
        class_directory = os.path.join(
            dataset_path,
            class_name
        )

        file_names = sorted(
            os.listdir(class_directory)
        )

        for file_name in file_names:
            if file_name.lower().endswith(
                valid_extensions
            ):
                paths.append(
                    os.path.join(
                        class_directory,
                        file_name
                    )
                )

                labels.append(
                    label_index
                )

    if not paths:
        raise RuntimeError(
            "학습 가능한 이미지 파일을 "
            "찾지 못했습니다."
        )

    return (
        np.asarray(paths),
        np.asarray(
            labels,
            dtype=np.int32
        ),
        class_folders
    )


# ============================================================
# Stratified Train/Test Split
# ============================================================
def stratified_split(
    paths,
    labels,
    test_ratio=0.2,
    seed=42
):
    rng = np.random.default_rng(
        seed
    )

    train_indices = []
    test_indices = []

    for class_id in np.unique(
        labels
    ):
        class_indices = np.where(
            labels == class_id
        )[0]

        rng.shuffle(
            class_indices
        )

        class_count = len(
            class_indices
        )

        if class_count <= 1:
            train_indices.extend(
                class_indices
            )
            continue

        test_count = int(
            round(
                class_count
                * test_ratio
            )
        )

        test_count = max(
            1,
            test_count
        )

        test_count = min(
            test_count,
            class_count - 1
        )

        test_indices.extend(
            class_indices[:test_count]
        )

        train_indices.extend(
            class_indices[test_count:]
        )

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64
    )

    test_indices = np.asarray(
        test_indices,
        dtype=np.int64
    )

    rng.shuffle(
        train_indices
    )

    rng.shuffle(
        test_indices
    )

    return (
        paths[train_indices],
        labels[train_indices],
        paths[test_indices],
        labels[test_indices]
    )


# ============================================================
# Trainer
# ============================================================
class Trainer:
    def __init__(
        self,
        network,
        train_paths,
        train_labels,
        test_paths,
        test_labels,
        epochs,
        mini_batch_size,
        optimizer,
        optimizer_param,
        augmenter,
        img_size=224,
        evaluate_sample_num_per_epoch=500,
        evaluation_batch_size=16,
        verbose=True
    ):
        self.network = network

        self.train_paths = train_paths
        self.train_labels = train_labels

        self.test_paths = test_paths
        self.test_labels = test_labels

        self.epochs = epochs
        self.batch_size = mini_batch_size

        self.augmenter = augmenter
        self.img_size = img_size

        self.evaluate_sample_num_per_epoch = (
            evaluate_sample_num_per_epoch
        )

        self.evaluation_batch_size = (
            evaluation_batch_size
        )

        self.verbose = verbose

        optimizer_classes = {
            "sgd": SGD,
            "adamw": AdamW
        }

        optimizer_name = (
            optimizer.lower()
        )

        if optimizer_name not in optimizer_classes:
            raise ValueError(
                f"지원하지 않는 optimizer: "
                f"{optimizer}"
            )

        self.optimizer = (
            optimizer_classes[
                optimizer_name
            ](**optimizer_param)
        )

        self.train_size = len(
            train_paths
        )

        self.iter_per_epoch = max(
            int(
                np.ceil(
                    self.train_size
                    / self.batch_size
                )
            ),
            1
        )

        self.max_iter = (
            self.epochs
            * self.iter_per_epoch
        )

        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []

        self.test_loss_list = []
        self.test_acc_list = []

    def _load_batch(
        self,
        batch_paths,
        batch_labels,
        train_aug=True
    ):
        batch_size = len(
            batch_paths
        )

        xs = np.empty(
            (
                batch_size,
                3,
                self.img_size,
                self.img_size
            ),
            dtype=np.float32
        )

        for index, path in enumerate(
            batch_paths
        ):
            with Image.open(path) as image:
                image = image.convert(
                    "RGB"
                )

                if train_aug:
                    image = (
                        self.augmenter
                        .augment_pil(image)
                    )

                xs[index] = (
                    self.augmenter
                    .pil_to_chw_float(
                        image,
                        size=self.img_size
                    )
                )

        ys = batch_labels.astype(
            np.int32,
            copy=False
        )

        return (
            cp.asarray(xs),
            cp.asarray(ys)
        )

    def _evaluate_from_paths(
        self,
        paths,
        labels,
        batch_size=None
    ):
        if batch_size is None:
            batch_size = (
                self.evaluation_batch_size
            )

        total_loss = 0.0
        correct = 0
        total = 0

        for start in range(
            0,
            len(paths),
            batch_size
        ):
            end = min(
                start + batch_size,
                len(paths)
            )

            x_batch, y_batch = (
                self._load_batch(
                    paths[start:end],
                    labels[start:end],
                    train_aug=False
                )
            )

            logits = self.network.predict(
                x_batch,
                train_flg=False
            )

            loss, _ = (
                softmax_cross_entropy(
                    logits,
                    y_batch
                )
            )

            predictions = cp.argmax(
                logits,
                axis=1
            )

            current_batch_size = (
                end - start
            )

            total_loss += (
                scalar_to_float(loss)
                * current_batch_size
            )

            correct += int(
                cp.sum(
                    predictions == y_batch
                ).get()
            )

            total += (
                current_batch_size
            )

            del x_batch
            del y_batch
            del logits
            del predictions

        average_loss = (
            total_loss / max(total, 1)
        )

        accuracy = (
            correct / max(total, 1)
        )

        return average_loss, accuracy

    def _select_evaluation_data(self):
        if (
            self.evaluate_sample_num_per_epoch
            is None
        ):
            return (
                self.train_paths,
                self.train_labels,
                self.test_paths,
                self.test_labels
            )

        train_count = min(
            self.evaluate_sample_num_per_epoch,
            len(self.train_paths)
        )

        test_count = min(
            self.evaluate_sample_num_per_epoch,
            len(self.test_paths)
        )

        return (
            self.train_paths[:train_count],
            self.train_labels[:train_count],
            self.test_paths[:test_count],
            self.test_labels[:test_count]
        )

    def train_step(self):
        replace = (
            self.train_size
            < self.batch_size
        )

        batch_indices = np.random.choice(
            self.train_size,
            self.batch_size,
            replace=replace
        )

        batch_paths = (
            self.train_paths[
                batch_indices
            ]
        )

        batch_labels = (
            self.train_labels[
                batch_indices
            ]
        )

        x_batch, y_batch = (
            self._load_batch(
                batch_paths,
                batch_labels,
                train_aug=True
            )
        )

        grads = self.network.gradient(
            x_batch,
            y_batch
        )

        self.optimizer.update(
            self.network.params,
            grads
        )

        self.current_iter += 1

        del x_batch
        del y_batch
        del grads

        # 1 epoch 완료 시 평가
        if (
            self.current_iter
            % self.iter_per_epoch
            == 0
        ):
            self.current_epoch += 1

            (
                train_paths_eval,
                train_labels_eval,
                test_paths_eval,
                test_labels_eval
            ) = self._select_evaluation_data()

            train_loss, train_acc = (
                self._evaluate_from_paths(
                    train_paths_eval,
                    train_labels_eval
                )
            )

            test_loss, test_acc = (
                self._evaluate_from_paths(
                    test_paths_eval,
                    test_labels_eval
                )
            )

            self.train_loss_list.append(
                train_loss
            )

            self.train_acc_list.append(
                train_acc
            )

            self.test_loss_list.append(
                test_loss
            )

            self.test_acc_list.append(
                test_acc
            )

            print(
                f"\nEpoch "
                f"{self.current_epoch:03d}/"
                f"{self.epochs:03d} | "
                f"Train Loss="
                f"{train_loss:.6f} | "
                f"Train Acc="
                f"{train_acc:.4f} | "
                f"Test Loss="
                f"{test_loss:.6f} | "
                f"Test Acc="
                f"{test_acc:.4f}"
            )

    def train(self):
        print(
            "\n=============== "
            "Training Start "
            "==============="
        )

        print(
            f"Epochs          : "
            f"{self.epochs}"
        )

        print(
            f"Iterations/Epoch: "
            f"{self.iter_per_epoch}"
        )

        print(
            f"Batch Size      : "
            f"{self.batch_size}"
        )

        for _ in tqdm(
            range(self.max_iter),
            desc="Training"
        ):
            self.train_step()

        final_test_loss, final_test_acc = (
            self._evaluate_from_paths(
                self.test_paths,
                self.test_labels
            )
        )

        print(
            "\n=============== "
            "Final Test Result "
            "==============="
        )

        print(
            f"Final Test Loss : "
            f"{final_test_loss:.6f}"
        )

        print(
            f"Final Test Acc  : "
            f"{final_test_acc:.4f}"
        )

        return (
            final_test_loss,
            final_test_acc
        )

    def confusion_matrix_and_per_class_acc(
        self,
        num_classes=10,
        batch_size=16
    ):
        confusion_matrix = np.zeros(
            (
                num_classes,
                num_classes
            ),
            dtype=np.int64
        )

        for start in range(
            0,
            len(self.test_paths),
            batch_size
        ):
            end = min(
                start + batch_size,
                len(self.test_paths)
            )

            x_batch, y_true = (
                self._load_batch(
                    self.test_paths[
                        start:end
                    ],
                    self.test_labels[
                        start:end
                    ],
                    train_aug=False
                )
            )

            logits = self.network.predict(
                x_batch,
                train_flg=False
            )

            y_pred = cp.argmax(
                logits,
                axis=1
            ).get()

            y_true_np = y_true.get()

            for true_label, predicted_label in zip(
                y_true_np,
                y_pred
            ):
                confusion_matrix[
                    int(true_label),
                    int(predicted_label)
                ] += 1

            del x_batch
            del y_true
            del logits

        per_class_accuracy = np.zeros(
            num_classes,
            dtype=np.float64
        )

        for class_index in range(
            num_classes
        ):
            class_total = (
                confusion_matrix[
                    class_index
                ].sum()
            )

            if class_total > 0:
                per_class_accuracy[
                    class_index
                ] = (
                    confusion_matrix[
                        class_index,
                        class_index
                    ]
                    / class_total
                )

        total_count = (
            confusion_matrix.sum()
        )

        if total_count > 0:
            overall_accuracy = (
                np.trace(
                    confusion_matrix
                )
                / total_count
            )
        else:
            overall_accuracy = 0.0

        return (
            confusion_matrix,
            per_class_accuracy,
            overall_accuracy
        )

    def save_metrics(
        self,
        file_name
    ):
        metrics = {
            "train_loss": (
                self.train_loss_list
            ),
            "train_accuracy": (
                self.train_acc_list
            ),
            "test_loss": (
                self.test_loss_list
            ),
            "test_accuracy": (
                self.test_acc_list
            )
        }

        directory = os.path.dirname(
            file_name
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        with open(
            file_name,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                metrics,
                file,
                ensure_ascii=False,
                indent=2
            )


# ============================================================
# 시각화
# ============================================================
def plot_training_curves(
    trainer,
    save_path=None,
    show=True
):
    epoch_axis = np.arange(
        1,
        len(
            trainer.train_loss_list
        ) + 1
    )

    figure = plt.figure(
        figsize=(12, 5)
    )

    loss_axis = figure.add_subplot(
        1, 2, 1
    )

    loss_axis.plot(
        epoch_axis,
        trainer.train_loss_list,
        label="Train Loss"
    )

    loss_axis.plot(
        epoch_axis,
        trainer.test_loss_list,
        label="Test Loss"
    )

    loss_axis.set_xlabel(
        "Epoch"
    )

    loss_axis.set_ylabel(
        "Loss"
    )

    loss_axis.set_title(
        "Loss Curve"
    )

    loss_axis.grid(True)
    loss_axis.legend()

    accuracy_axis = (
        figure.add_subplot(
            1, 2, 2
        )
    )

    accuracy_axis.plot(
        epoch_axis,
        trainer.train_acc_list,
        label="Train Accuracy"
    )

    accuracy_axis.plot(
        epoch_axis,
        trainer.test_acc_list,
        label="Test Accuracy"
    )

    accuracy_axis.set_xlabel(
        "Epoch"
    )

    accuracy_axis.set_ylabel(
        "Accuracy"
    )

    accuracy_axis.set_title(
        "Accuracy Curve"
    )

    accuracy_axis.grid(True)
    accuracy_axis.legend()

    figure.tight_layout()

    if save_path is not None:
        directory = os.path.dirname(
            save_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        figure.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        print(
            f"학습 곡선 저장: "
            f"{save_path}"
        )

    if show:
        plt.show()
    else:
        plt.close(figure)


def plot_confusion_matrix(
    confusion_matrix,
    class_names,
    save_path=None,
    show=True
):
    figure, axis = plt.subplots(
        figsize=(9, 8)
    )

    image = axis.imshow(
        confusion_matrix,
        interpolation="nearest",
        cmap="Blues"
    )

    figure.colorbar(
        image,
        ax=axis
    )

    axis.set_title(
        "Confusion Matrix"
    )

    axis.set_xlabel(
        "Predicted Label"
    )

    axis.set_ylabel(
        "True Label"
    )

    tick_positions = np.arange(
        len(class_names)
    )

    axis.set_xticks(
        tick_positions
    )

    axis.set_yticks(
        tick_positions
    )

    axis.set_xticklabels(
        class_names,
        rotation=45,
        ha="right"
    )

    axis.set_yticklabels(
        class_names
    )

    threshold = (
        confusion_matrix.max() / 2.0
        if confusion_matrix.size > 0
        else 0.0
    )

    for row in range(
        confusion_matrix.shape[0]
    ):
        for column in range(
            confusion_matrix.shape[1]
        ):
            value = confusion_matrix[
                row,
                column
            ]

            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                )
            )

    figure.tight_layout()

    if save_path is not None:
        directory = os.path.dirname(
            save_path
        )

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        figure.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight"
        )

        print(
            f"Confusion Matrix 저장: "
            f"{save_path}"
        )

    if show:
        plt.show()
    else:
        plt.close(figure)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # --------------------------------------------------------
    # 사용자 환경에 맞게 수정
    # --------------------------------------------------------
    DATASET_PATH = (
        r"E:\cnn2020_code"
        r"\deepfake_detection"
        r"\2026_01"
        r"\multi_auth_py"
        r"\data_100"
        r"\face"
    )

    OUTPUT_DIRECTORY = (
        "vit_outputs_100"
    )

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 학습 설정
    # --------------------------------------------------------
    IMAGE_SIZE = 224
    PATCH_SIZE = 16

    D_MODEL = 128
    NUM_HEADS = 4
    D_FF = 256
    NUM_LAYERS = 2

    DROPOUT_RATIO = 0.1
    NUM_CLASSES = 100

    EPOCHS = 2 #100 #50
    MINI_BATCH_SIZE = 16
    EVALUATION_BATCH_SIZE = 16

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0

    EVALUATE_SAMPLE_NUM_PER_EPOCH = 500

    # --------------------------------------------------------
    # 데이터 경로 수집
    # --------------------------------------------------------
    (
        paths,
        labels,
        class_names
    ) = collect_paths_10class(
        DATASET_PATH
    )

    print(
        "클래스 폴더 순서:",
        class_names
    )

    print(
        "전체 이미지 수:",
        len(paths)
    )

    # --------------------------------------------------------
    # Stratified Train/Test split
    # --------------------------------------------------------
    (
        train_paths,
        train_labels,
        test_paths,
        test_labels
    ) = stratified_split(
        paths,
        labels,
        test_ratio=0.2,
        seed=SEED
    )

    print(
        "Train:",
        len(train_paths)
    )

    print(
        "Test:",
        len(test_paths)
    )

    # --------------------------------------------------------
    # 이미지 증강
    # --------------------------------------------------------
    augmenter = SimpleImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=5.0,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        rescale=1.0 / 255.0
    )

    # --------------------------------------------------------
    # Vision Transformer Multi-Head
    # --------------------------------------------------------
    network = VisionTransformerMultiHead(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout_ratio=DROPOUT_RATIO
    )

    print(
        "\n=============== "
        "Model Configuration "
        "==============="
    )

    print(
        f"Image Size     : "
        f"{IMAGE_SIZE}×{IMAGE_SIZE}"
    )

    print(
        f"Patch Size     : "
        f"{PATCH_SIZE}×{PATCH_SIZE}"
    )

    print(
        f"Patch Count    : "
        f"{network.num_patches}"
    )

    print(
        f"Token Count    : "
        f"{network.num_patches + 1}"
    )

    print(
        f"d_model        : "
        f"{D_MODEL}"
    )

    print(
        f"Attention Heads: "
        f"{NUM_HEADS}"
    )

    print(
        f"Head Dimension : "
        f"{D_MODEL // NUM_HEADS}"
    )

    print(
        f"FFN Dimension  : "
        f"{D_FF}"
    )

    print(
        f"Encoder Layers : "
        f"{NUM_LAYERS}"
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    trainer = Trainer(
        network=network,

        train_paths=train_paths,
        train_labels=train_labels,

        test_paths=test_paths,
        test_labels=test_labels,

        epochs=EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE,

        optimizer="adamw",

        optimizer_param={
            "lr": LEARNING_RATE,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP
        },

        augmenter=augmenter,
        img_size=IMAGE_SIZE,

        evaluate_sample_num_per_epoch=(
            EVALUATE_SAMPLE_NUM_PER_EPOCH
        ),

        evaluation_batch_size=(
            EVALUATION_BATCH_SIZE
        ),

        verbose=True
    )

    # --------------------------------------------------------
    # 학습 실행
    # --------------------------------------------------------
    trainer.train()

    # --------------------------------------------------------
    # 마지막 epoch 모델 저장
    # --------------------------------------------------------
    model_path = os.path.join(
        OUTPUT_DIRECTORY,
        "vit_multihead_10class_100.pkl"
    )

    network.save_params(
        model_path
    )

    print(
        f"모델 저장 완료: "
        f"{model_path}"
    )

    # --------------------------------------------------------
    # Epoch별 metric 저장
    # --------------------------------------------------------
    metric_path = os.path.join(
        OUTPUT_DIRECTORY,
        "training_metrics_100.json"
    )

    trainer.save_metrics(
        metric_path
    )

    print(
        f"학습 기록 저장 완료: "
        f"{metric_path}"
    )

    # --------------------------------------------------------
    # 학습 곡선
    # --------------------------------------------------------
    curve_path = os.path.join(
        OUTPUT_DIRECTORY,
        "training_curves_100.png"
    )

    plot_training_curves(
        trainer,
        save_path=curve_path,
        show=True
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------
    (
        confusion_matrix,
        per_class_accuracy,
        overall_accuracy
    ) = trainer.confusion_matrix_and_per_class_acc(
        num_classes=NUM_CLASSES,
        batch_size=EVALUATION_BATCH_SIZE
    )

    print(
        "\n=============== "
        "Confusion Matrix "
        "==============="
    )

    print(
        confusion_matrix
    )

    print(
        "\n=============== "
        "Per-class Accuracy "
        "==============="
    )

    for class_index, accuracy in enumerate(
        per_class_accuracy
    ):
        if class_index < len(
            class_names
        ):
            class_name = (
                class_names[
                    class_index
                ]
            )
        else:
            class_name = str(
                class_index
            )

        print(
            f"Class {class_index:02d} "
            f"({class_name}): "
            f"{accuracy:.4f}"
        )

    print(
        "\nOverall Accuracy: "
        f"{overall_accuracy:.4f}"
    )

    confusion_matrix_npy_path = (
        os.path.join(
            OUTPUT_DIRECTORY,
            "confusion_matrix_100.npy"
        )
    )

    np.save(
        confusion_matrix_npy_path,
        confusion_matrix
    )

    confusion_matrix_png_path = (
        os.path.join(
            OUTPUT_DIRECTORY,
            "confusion_matrix_100.png"
        )
    )

    plot_confusion_matrix(
        confusion_matrix,
        class_names,
        save_path=(
            confusion_matrix_png_path
        ),
        show=True
    )

    print(
        "\n모든 작업이 완료되었습니다."
    )