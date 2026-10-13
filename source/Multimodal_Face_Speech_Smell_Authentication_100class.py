#```python
# ============================================================
# Multimodal_Face_Speech_Smell_Authentication_100class.py
#
# 기존 세 Transformer를 frozen encoder로 사용하고
# Face + Speech + Smell embedding을 concatenate하여
# 사용자 인증 분류기를 학습합니다.
#
# 필요한 기존 파일
# 1. Face_vit_multihead_10class_cupy.py
# 2. SmellNet_Transformer_Multihead_GPU.py
# 3. Wav_MFCC_GPU_Transformer_Multihead_dropout_without_earlystop.py
#
# manifest.csv 필수 컬럼
# sample_id,user_id,face_path,wav_path,smell_path,split
#
# split 값:
# train / val / test
#
# label 컬럼을 직접 사용할 경우:
# sample_id,label,face_path,wav_path,smell_path,split
# ============================================================

import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import cupy as cp
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm


# ============================================================
# 기존 단일모달 파일 import
# ============================================================
import Face_vit_multihead_10class_cupy_100class as face_module
import SmellNet_Transformer_Multihead_GPU_100class as smell_module
import Wav_MFCC_GPU_Transformer_Multihead_dropout_without_earlystop_100class as speech_module


# ============================================================
# 공통 설정
# ============================================================
GPU_DEVICE_ID = 0
SEED = 42

cp.cuda.Device(GPU_DEVICE_ID).use()

np.random.seed(SEED)
cp.random.seed(SEED)


@dataclass
class Config:
    # --------------------------------------------------------
    # 데이터
    # --------------------------------------------------------
    manifest_csv: str = "multimodal_manifest_100.csv"

    # --------------------------------------------------------
    # 사전 학습 모델
    # --------------------------------------------------------
    face_checkpoint: str = (
        "vit_outputs_100/vit_multihead_10class_100.pkl"
    )

    speech_checkpoint: str = (
        "speech_artifacts/transformer_multihead_last_epoch_params.npz"
    )

    speech_norm_stats: str = (
        "speech_artifacts/mfcc_norm_stats.npz"
    )

    smell_checkpoint: str = (
        "checkpoints/transformer_cupy_min.npz"
        #"smell_outputs/smell_transformer_multihead.npz"
    )

    # --------------------------------------------------------
    # 출력
    # --------------------------------------------------------
    output_dir: str = "multimodal_outputs_100"

    # --------------------------------------------------------
    # 클래스
    # --------------------------------------------------------
    num_classes: int = 100

    # --------------------------------------------------------
    # Face 설정
    # 사전 학습 모델과 동일해야 합니다.
    # --------------------------------------------------------
    image_size: int = 224
    patch_size: int = 16
    face_d_model: int = 128
    face_num_heads: int = 4
    face_d_ff: int = 256
    face_num_layers: int = 2

    # --------------------------------------------------------
    # Fusion Weight 각 모달리티의 기여도를 조절
    # 가중치를 높이면 해당 요소가 fake일경우 Acc가
    # 낮아질 것으로 기대함
    # --------------------------------------------------------
    face_weight: float = 0.16
    speech_weight: float = 0.03
    smell_weight: float = 0.81

    print(f"face_weight: " f"{face_weight:}")
    print(f"speech_weight: " f"{speech_weight:}")
    print(f"smell_weight: " f"{smell_weight:}")
    
    
    
    # --------------------------------------------------------
    # Speech 설정
    # 업로드한 최신 speech 코드의 권장 기본값과 일치
    # --------------------------------------------------------
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 400
    hop_length: int = 160
    speech_seq_len: int = 64

    speech_d_model: int = 64
    speech_num_heads: int = 4
    speech_d_ff: int = 128
    speech_num_layers: int = 1
    speech_dropout: float = 0.15

    # --------------------------------------------------------
    # Smell 설정
    # --------------------------------------------------------
    smell_window: int = 100
    smell_stride: int = 1
    smell_lag: int = 25
    smell_diff_mode: str = "diff"

    smell_d_model: int = 128
    smell_d_ff: int = 256

    # CSV 하나에서 여러 window가 생성될 경우
    # "last", "first", "mean" 중 선택
    smell_window_select: str = "last"

    # --------------------------------------------------------
    # Fusion 설정
    # 각 modality를 128차원으로 정렬
    # --------------------------------------------------------
    common_embedding_dim: int = 128
    fusion_hidden_dim: int = 256
    fusion_output_dim: int = 128
    fusion_dropout: float = 0.20

    # --------------------------------------------------------
    # 학습 설정
    # --------------------------------------------------------
    epochs: int = 100
    batch_size: int = 8

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # encoder는 고정하고 fusion만 학습
    freeze_encoders: bool = True


# ============================================================
# 기본 수학 함수
# ============================================================
def softmax(logits):
    shifted = logits - cp.max(
        logits,
        axis=1,
        keepdims=True
    )

    exp_logits = cp.exp(shifted)

    return exp_logits / cp.sum(
        exp_logits,
        axis=1,
        keepdims=True
    )


def cross_entropy_loss(logits, labels):
    labels = labels.astype(cp.int32)

    probabilities = softmax(logits)
    batch_size = logits.shape[0]

    loss = -cp.mean(
        cp.log(
            probabilities[
                cp.arange(batch_size),
                labels
            ] + 1e-9
        )
    )

    dlogits = probabilities.copy()

    dlogits[
        cp.arange(batch_size),
        labels
    ] -= 1.0

    dlogits /= batch_size

    return loss, dlogits


def l2_normalize(x, eps=1e-8):
    norm = cp.sqrt(
        cp.sum(
            x * x,
            axis=1,
            keepdims=True
        ) + eps
    )

    return x / norm


# ============================================================
# Fusion용 Linear
# ============================================================
class Linear:
    def __init__(
        self,
        input_dim,
        output_dim,
        name
    ):
        limit = np.sqrt(
            6.0 / (input_dim + output_dim)
        )

        self.W = cp.random.uniform(
            -limit,
            limit,
            size=(input_dim, output_dim)
        ).astype(cp.float32)

        self.b = cp.zeros(
            output_dim,
            dtype=cp.float32
        )

        self.name = name
        self.x = None

        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = cp.sum(dout, axis=0)

        return dout @ self.W.T


class GELU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x

        coefficient = np.sqrt(
            2.0 / np.pi
        )

        value = coefficient * (
            x + 0.044715 * x ** 3
        )

        return 0.5 * x * (
            1.0 + cp.tanh(value)
        )

    def backward(self, dout):
        x = self.x

        coefficient = np.sqrt(
            2.0 / np.pi
        )

        value = coefficient * (
            x + 0.044715 * x ** 3
        )

        tanh_value = cp.tanh(value)

        dvalue = coefficient * (
            1.0
            + 3.0 * 0.044715 * x ** 2
        )

        derivative = (
            0.5 * (1.0 + tanh_value)
            + 0.5
            * x
            * (1.0 - tanh_value ** 2)
            * dvalue
        )

        return dout * derivative


class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, training=True):
        if not training or self.dropout_ratio <= 0.0:
            self.mask = None
            return x

        keep_probability = (
            1.0 - self.dropout_ratio
        )

        self.mask = (
            cp.random.random(x.shape)
            < keep_probability
        ).astype(cp.float32)

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
# Multimodal Fusion 인증 Head
# ============================================================
class MultimodalFusionHead:
    def __init__(
        self,
        face_dim,
        speech_dim,
        smell_dim,
        common_dim,
        hidden_dim,
        output_dim,
        num_classes,
        dropout_ratio,
        face_weight=0.0,
        speech_weight=0.0,
        smell_weight=0.0
    ):
        # modality별 차원 정렬
        self.face_projection = Linear(
            face_dim,
            common_dim,
            "face_projection"
        )

        self.speech_projection = Linear(
            speech_dim,
            common_dim,
            "speech_projection"
        )

        self.smell_projection = Linear(
            smell_dim,
            common_dim,
            "smell_projection"
        )

        fusion_dim = common_dim * 3

        self.fc1 = Linear(
            fusion_dim,
            hidden_dim,
            "fusion_fc1"
        )

        self.act1 = GELU()
        self.dropout1 = Dropout(
            dropout_ratio
        )

        self.fc2 = Linear(
            hidden_dim,
            output_dim,
            "fusion_fc2"
        )

        self.act2 = GELU()
        self.dropout2 = Dropout(
            dropout_ratio / 2.0
        )

        self.classifier = Linear(
            output_dim,
            num_classes,
            "classifier"
        )

        self.params = {
            "face_projection_W":
                self.face_projection.W,
            "face_projection_b":
                self.face_projection.b,

            "speech_projection_W":
                self.speech_projection.W,
            "speech_projection_b":
                self.speech_projection.b,

            "smell_projection_W":
                self.smell_projection.W,
            "smell_projection_b":
                self.smell_projection.b,

            "fusion_fc1_W":
                self.fc1.W,
            "fusion_fc1_b":
                self.fc1.b,

            "fusion_fc2_W":
                self.fc2.W,
            "fusion_fc2_b":
                self.fc2.b,

            "classifier_W":
                self.classifier.W,
            "classifier_b":
                self.classifier.b
        }

        self.grads = {}
        total = (
            face_weight +
            speech_weight +
            smell_weight
        )

        self.face_weight = face_weight / total
        self.speech_weight = speech_weight / total
        self.smell_weight = smell_weight / total

    def forward(
        self,
        face_embedding,
        speech_embedding,
        smell_embedding,
        training=True
    ):
        face_projected = (
            self.face_projection.forward(
                face_embedding
            )
        )

        speech_projected = (
            self.speech_projection.forward(
                speech_embedding
            )
        )

        smell_projected = (
            self.smell_projection.forward(
                smell_embedding
            )
        )

        # 각 modality 크기 차이 완화
        face_projected = l2_normalize(
            face_projected
        )

        speech_projected = l2_normalize(
            speech_projected
        )

        smell_projected = l2_normalize(
            smell_projected
        )

        face_projected *= self.face_weight
        speech_projected *= self.speech_weight
        smell_projected *= self.smell_weight

        fusion = cp.concatenate(
            [
                face_projected,
                speech_projected,
                smell_projected
            ],
            axis=1
        )

        hidden = self.fc1.forward(
            fusion
        )

        hidden = self.act1.forward(
            hidden
        )

        hidden = self.dropout1.forward(
            hidden,
            training=training
        )

        authentication_embedding = (
            self.fc2.forward(hidden)
        )

        authentication_embedding = (
            self.act2.forward(
                authentication_embedding
            )
        )

        authentication_embedding = (
            self.dropout2.forward(
                authentication_embedding,
                training=training
            )
        )

        logits = self.classifier.forward(
            authentication_embedding
        )

        return logits

    def backward(self, dlogits):
        dout = self.classifier.backward(
            dlogits
        )

        dout = self.dropout2.backward(
            dout
        )

        dout = self.act2.backward(
            dout
        )

        dout = self.fc2.backward(
            dout
        )

        dout = self.dropout1.backward(
            dout
        )

        dout = self.act1.backward(
            dout
        )

        dfusion = self.fc1.backward(
            dout
        )

        common_dim = (
            dfusion.shape[1] // 3
        )

        dface = dfusion[
            :,
            :common_dim
        ]

        dspeech = dfusion[
            :,
            common_dim:2 * common_dim
        ]

        dsmell = dfusion[
            :,
            2 * common_dim:
        ]

        # encoder가 frozen이므로 반환 gradient는 사용하지 않음
        self.face_projection.backward(
            dface
        )

        self.speech_projection.backward(
            dspeech
        )

        self.smell_projection.backward(
            dsmell
        )

        self.grads = {
            "face_projection_W":
                self.face_projection.dW,
            "face_projection_b":
                self.face_projection.db,

            "speech_projection_W":
                self.speech_projection.dW,
            "speech_projection_b":
                self.speech_projection.db,

            "smell_projection_W":
                self.smell_projection.dW,
            "smell_projection_b":
                self.smell_projection.db,

            "fusion_fc1_W":
                self.fc1.dW,
            "fusion_fc1_b":
                self.fc1.db,

            "fusion_fc2_W":
                self.fc2.dW,
            "fusion_fc2_b":
                self.fc2.db,

            "classifier_W":
                self.classifier.dW,
            "classifier_b":
                self.classifier.db
        }

        return self.grads

    def save(self, path):
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        data = {
            key: cp.asnumpy(value)
            for key, value in self.params.items()
        }

        np.savez_compressed(
            path,
            **data
        )

    def load(self, path):
        checkpoint = np.load(path)

        for key, parameter in self.params.items():
            if key not in checkpoint:
                raise KeyError(
                    f"Fusion checkpoint에 "
                    f"{key}가 없습니다."
                )

            value = cp.asarray(
                checkpoint[key],
                dtype=cp.float32
            )

            if value.shape != parameter.shape:
                raise ValueError(
                    f"{key} shape 불일치: "
                    f"{value.shape} != "
                    f"{parameter.shape}"
                )

            parameter[...] = value


# ============================================================
# AdamW
# ============================================================
class AdamW:
    def __init__(
        self,
        params,
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=1e-4,
        grad_clip=1.0
    ):
        self.params = params

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        self.step_count = 0

        self.m = {
            key: cp.zeros_like(value)
            for key, value in params.items()
        }

        self.v = {
            key: cp.zeros_like(value)
            for key, value in params.items()
        }

    def _clip(self, grads):
        total_squared = cp.asarray(
            0.0,
            dtype=cp.float32
        )

        for grad in grads.values():
            total_squared += cp.sum(
                grad * grad
            )

        total_norm = cp.sqrt(
            total_squared
        )

        coefficient = cp.minimum(
            1.0,
            self.grad_clip
            / (total_norm + 1e-6)
        )

        for key in grads:
            grads[key] *= coefficient

    def update(self, grads):
        self._clip(grads)

        self.step_count += 1

        for key, parameter in self.params.items():
            grad = grads[key]

            self.m[key] = (
                self.beta1 * self.m[key]
                + (1.0 - self.beta1) * grad
            )

            self.v[key] = (
                self.beta2 * self.v[key]
                + (1.0 - self.beta2)
                * grad * grad
            )

            m_hat = self.m[key] / (
                1.0
                - self.beta1 ** self.step_count
            )

            v_hat = self.v[key] / (
                1.0
                - self.beta2 ** self.step_count
            )

            update = m_hat / (
                cp.sqrt(v_hat)
                + self.eps
            )

            # bias에는 weight decay를 적용하지 않음
            if parameter.ndim >= 2:
                update += (
                    self.weight_decay
                    * parameter
                )

            parameter -= (
                self.lr * update
            )


# ============================================================
# Face encoder adapter
# ============================================================
class FaceEncoderAdapter:
    def __init__(self, cfg):
        self.model = (
            face_module
            .VisionTransformerMultiHead(
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                in_channels=3,
                d_model=cfg.face_d_model,
                num_heads=cfg.face_num_heads,
                d_ff=cfg.face_d_ff,
                num_layers=cfg.face_num_layers,
                num_classes=cfg.num_classes,
                dropout_ratio=0.1
            )
        )

        self.model.load_params(
            cfg.face_checkpoint
        )

        self.output_dim = (
            cfg.face_d_model
        )

    def encode(self, images):
        """
        images:
            (B,3,224,224)
        """
        patches = self.model._patchify(
            images
        )

        tokens = (
            self.model
            .patch_projection
            .forward(patches)
        )

        batch_size = images.shape[0]

        cls_tokens = cp.broadcast_to(
            self.model.cls_token,
            (
                batch_size,
                1,
                self.model.d_model
            )
        )

        tokens = cp.concatenate(
            [
                cls_tokens,
                tokens
            ],
            axis=1
        )

        tokens = (
            tokens
            + self.model.pos_embedding
        )

        for encoder in self.model.encoders:
            tokens = encoder.forward(
                tokens,
                train_flg=False
            )

        cls_embedding = tokens[
            :,
            0,
            :
        ]

        cls_embedding = (
            self.model
            .final_norm
            .forward(cls_embedding)
        )

        return cls_embedding


# ============================================================
# Speech encoder adapter
# ============================================================
class SpeechEncoderAdapter:
    def __init__(self, cfg):
        self.model = (
            speech_module
            .TransformerClassifier(
                input_dim=cfg.n_mfcc,
                seq_len=cfg.speech_seq_len,
                d_model=cfg.speech_d_model,
                num_heads=cfg.speech_num_heads,
                d_ff=cfg.speech_d_ff,
                num_layers=cfg.speech_num_layers,
                num_classes=cfg.num_classes,
                dropout_p=cfg.speech_dropout,
                seed=SEED
            )
        )

        self.model.load_params(
            cfg.speech_checkpoint
        )

        self.model.eval()

        norm_data = np.load(
            cfg.speech_norm_stats
        )

        self.mean = norm_data[
            "mean"
        ].astype(np.float32)

        self.std = norm_data[
            "std"
        ].astype(np.float32)

        self.output_dim = (
            cfg.speech_d_model
        )

    def encode(
        self,
        mfcc_batch,
        frame_mask
    ):
        self.model.eval()

        self.model.forward(
            mfcc_batch,
            frame_mask=frame_mask
        )

        if self.model.last_cls is None:
            raise RuntimeError(
                "Speech model의 last_cls가 "
                "생성되지 않았습니다."
            )

        return self.model.last_cls


# ============================================================
# Smell encoder adapter
# ============================================================
class SmellEncoderAdapter:
    def __init__(
        self,
        cfg,
        smell_input_dim
    ):
        self.model = (
            smell_module
            .Transformer1Layer(
                d_in=smell_input_dim,
                d_model=cfg.smell_d_model,
                dim_ff=cfg.smell_d_ff,
                num_classes=cfg.num_classes
            )
        )

        smell_module.load_params(
            self.model,
            cfg.smell_checkpoint
        )

        self.output_dim = (
            cfg.smell_d_model
        )

    def encode(
        self,
        smell_batch,
        lengths=None
    ):
        _, cache = self.model.forward(
            smell_batch,
            lengths=lengths
        )

        # Smell 코드의 PCA 함수에서도 사용하는
        # Transformer 최종 token sequence
        hidden = cache[
            "h_after_ffn"
        ]

        cls_vector = hidden[
            :,
            0,
            :
        ]

        embedding, _ = (
            self.model
            .final_ln
            .forward(cls_vector)
        )

        return embedding


# ============================================================
# 데이터 전처리
# ============================================================
def load_face_image(
    path,
    image_size
):
    with Image.open(path) as image:
        image = image.convert(
            "RGB"
        )

        image = image.resize(
            (
                image_size,
                image_size
            ),
            resample=Image.BICUBIC
        )

        array = np.asarray(
            image,
            dtype=np.float32
        )

    array /= 255.0

    return array.transpose(
        2,
        0,
        1
    )


def load_speech_feature(
    path,
    cfg,
    mean,
    std
):
    mfcc = (
        speech_module
        .wav_to_mfcc_fixed(
            path,
            sr=cfg.sample_rate,
            n_mfcc=cfg.n_mfcc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            T=cfg.speech_seq_len,
            augment=False
        )
    )

    frame_mask = (
        speech_module
        .make_frame_mask(mfcc)
    )

    normalized = (
        speech_module
        .apply_norm_np(
            mfcc,
            mean,
            std,
            frame_mask=frame_mask
        )
    )

    return (
        normalized,
        frame_mask
    )


def load_smell_feature(
    path,
    cfg,
    expected_feature_cols=None
):
    # vector_cfg = (
    #     smell_module
    #     .SmellVectorizeConfig(
    #         feature_cols=expected_feature_cols,
    #         fillna="ffill",
    #         standardize="zscore",
    #         use_temporal_diff=True,
    #         p_lag=cfg.smell_lag,
    #         diff_mode=cfg.smell_diff_mode,
    #         window_w=cfg.smell_window,
    #         stride=cfg.smell_stride,
    #         drop_short=False
    #     )
    # )

    # dataframe = pd.read_csv(path)

    # feature_columns = (
    #     smell_module
    #     ._choose_feature_columns(
    #         dataframe,
    #         vector_cfg
    #     )
    # )

    dataframe = pd.read_csv(path)

    # ----------------------------------------------------
    # 사용할 센서 4개 고정
    # C2H5CH -> C2H5OH 이름 통일
    # ----------------------------------------------------
    if "C2H5CH" in dataframe.columns:
        dataframe["C2H5OH"] = dataframe["C2H5CH"]
    elif "C2H5OH" not in dataframe.columns:
        raise ValueError(
            f"{path}에 C2H5OH 또는 C2H5CH 컬럼이 없습니다."
        )

    feature_columns = [
        "NO2",
        "C2H5OH",
        "VOC",
        "CO"
    ]
    #feature_columns = ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG", "Benzene", "Temperature", 
    #        "Pressure", "Humidity", "Gas_Resistance", "Altitude"]
    # 존재 여부 확인
    missing = [
        c for c in feature_columns
        if c not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{path}에 컬럼이 없습니다 : {missing}"
        )

    vector_cfg = smell_module.SmellVectorizeConfig(
        feature_cols=feature_columns,
        fillna="ffill",
        standardize="zscore",
        use_temporal_diff=True,
        p_lag=cfg.smell_lag,
        diff_mode=cfg.smell_diff_mode,
        window_w=cfg.smell_window,
        stride=cfg.smell_stride,
        drop_short=False
    )

    #====================================
    feature_frame = (
        smell_module
        ._fillna(
            dataframe[
                feature_columns
            ].copy(),
            vector_cfg
        )
    )

    values = feature_frame.to_numpy(
        dtype=np.float32
    )

    values = smell_module._standardize(
        values,
        vector_cfg
    )

    values = (
        smell_module
        ._temporal_difference(
            values,
            vector_cfg
        )
    )

    windows = smell_module._sliding_window(
        values,
        vector_cfg
    )

    if windows.shape[0] == 0:
        raise ValueError(
            f"Smell window 생성 실패: {path}"
        )

    if cfg.smell_window_select == "first":
        selected = windows[0]

    elif cfg.smell_window_select == "last":
        selected = windows[-1]

    elif cfg.smell_window_select == "mean":
        selected = np.mean(
            windows,
            axis=0
        ).astype(np.float32)

    else:
        raise ValueError(
            "smell_window_select는 "
            "'first', 'last', 'mean' 중 "
            "하나여야 합니다."
        )

    return (
        selected.astype(np.float32),
        feature_columns
    )


# ============================================================
# Manifest
# ============================================================
def load_manifest(path):
    dataframe = pd.read_csv(path)

    required_columns = {
        "face_path",
        "wav_path",
        "smell_path",
        "split"
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"manifest 필수 컬럼 누락: "
            f"{sorted(missing)}"
        )

    if "label" not in dataframe.columns:
        if "user_id" not in dataframe.columns:
            raise ValueError(
                "manifest에는 label 또는 "
                "user_id 컬럼이 필요합니다."
            )

        users = sorted(
            dataframe[
                "user_id"
            ].astype(str).unique()
        )

        user_to_label = {
            user: index
            for index, user
            in enumerate(users)
        }

        dataframe["label"] = (
            dataframe[
                "user_id"
            ].astype(str).map(
                user_to_label
            )
        )

    else:
        user_to_label = None

    dataframe["label"] = (
        dataframe["label"]
        .astype(np.int32)
    )

    valid_splits = {
        "train",
        "val",
        "test"
    }

    unknown_splits = (
        set(
            dataframe[
                "split"
            ].astype(str).str.lower()
        )
        - valid_splits
    )

    if unknown_splits:
        raise ValueError(
            f"잘못된 split 값: "
            f"{unknown_splits}"
        )

    dataframe["split"] = (
        dataframe[
            "split"
        ].astype(str).str.lower()
    )

    return (
        dataframe,
        user_to_label
    )


# ============================================================
# Multimodal batch loader
# ============================================================
class MultimodalBatchLoader:
    def __init__(
        self,
        dataframe,
        cfg,
        speech_encoder,
        smell_feature_cols=None
    ):
        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.cfg = cfg
        self.speech_encoder = (
            speech_encoder
        )

        self.smell_feature_cols = (
            smell_feature_cols
        )

    def __len__(self):
        return len(
            self.dataframe
        )

    def iterate(
        self,
        batch_size,
        shuffle,
        seed
    ):
        indices = np.arange(
            len(self.dataframe)
        )

        if shuffle:
            rng = np.random.default_rng(
                seed
            )

            rng.shuffle(
                indices
            )

        for start in range(
            0,
            len(indices),
            batch_size
        ):
            batch_indices = indices[
                start:start + batch_size
            ]

            face_list = []
            mfcc_list = []
            frame_mask_list = []
            smell_list = []
            label_list = []

            for index in batch_indices:
                row = self.dataframe.iloc[
                    int(index)
                ]

                face = load_face_image(
                    row["face_path"],
                    self.cfg.image_size
                )

                mfcc, frame_mask = (
                    load_speech_feature(
                        row["wav_path"],
                        self.cfg,
                        self.speech_encoder.mean,
                        self.speech_encoder.std
                    )
                )

                smell, feature_cols = (
                    load_smell_feature(
                        row["smell_path"],
                        self.cfg,
                        expected_feature_cols=(
                            self.smell_feature_cols
                        )
                    )
                )

                if self.smell_feature_cols is None:
                    self.smell_feature_cols = (
                        feature_cols
                    )

                face_list.append(
                    face
                )

                mfcc_list.append(
                    mfcc
                )

                frame_mask_list.append(
                    frame_mask
                )

                smell_list.append(
                    smell
                )

                label_list.append(
                    int(row["label"])
                )

            face_batch = cp.asarray(
                np.stack(
                    face_list,
                    axis=0
                ),
                dtype=cp.float32
            )

            speech_batch = cp.asarray(
                np.stack(
                    mfcc_list,
                    axis=0
                ),
                dtype=cp.float32
            )

            frame_mask_batch = cp.asarray(
                np.stack(
                    frame_mask_list,
                    axis=0
                ),
                dtype=cp.float32
            )

            smell_batch = cp.asarray(
                np.stack(
                    smell_list,
                    axis=0
                ),
                dtype=cp.float32
            )

            labels = cp.asarray(
                np.asarray(
                    label_list,
                    dtype=np.int32
                )
            )

            smell_lengths = cp.full(
                (
                    smell_batch.shape[0],
                ),
                smell_batch.shape[1],
                dtype=cp.int32
            )

            yield {
                "face": face_batch,
                "speech": speech_batch,
                "speech_mask":
                    frame_mask_batch,
                "smell": smell_batch,
                "smell_lengths":
                    smell_lengths,
                "labels": labels
            }


# ============================================================
# 전체 Multimodal 모델
# ============================================================
class MultimodalAuthenticationModel:
    def __init__(
        self,
        face_encoder,
        speech_encoder,
        smell_encoder,
        fusion_head
    ):
        self.face_encoder = (
            face_encoder
        )

        self.speech_encoder = (
            speech_encoder
        )

        self.smell_encoder = (
            smell_encoder
        )

        self.fusion_head = fusion_head

    def forward(
        self,
        batch,
        training=True
    ):
        face_embedding = (
            self.face_encoder.encode(
                batch["face"]
            )
        )

        speech_embedding = (
            self.speech_encoder.encode(
                batch["speech"],
                batch["speech_mask"]
            )
        )

        smell_embedding = (
            self.smell_encoder.encode(
                batch["smell"],
                batch["smell_lengths"]
            )
        )

        logits = self.fusion_head.forward(
            face_embedding,
            speech_embedding,
            smell_embedding,
            training=training
        )

        return logits


# ============================================================
# 평가
# ============================================================
def evaluate(
    model,
    loader,
    cfg
):
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    true_labels = []
    predicted_labels = []

    for batch in loader.iterate(
        batch_size=cfg.batch_size,
        shuffle=False,
        seed=SEED
    ):
        logits = model.forward(
            batch,
            training=False
        )

        loss, _ = cross_entropy_loss(
            logits,
            batch["labels"]
        )

        predictions = cp.argmax(
            logits,
            axis=1
        )

        batch_size = int(
            batch["labels"].shape[0]
        )

        total_loss += (
            float(loss.get())
            * batch_size
        )

        total_correct += int(
            cp.sum(
                predictions
                == batch["labels"]
            ).get()
        )

        total_count += batch_size

        true_labels.append(
            cp.asnumpy(
                batch["labels"]
            )
        )

        predicted_labels.append(
            cp.asnumpy(
                predictions
            )
        )

    average_loss = (
        total_loss
        / max(total_count, 1)
    )

    accuracy = (
        total_correct
        / max(total_count, 1)
    )

    if true_labels:
        true_labels = np.concatenate(
            true_labels
        )

        predicted_labels = np.concatenate(
            predicted_labels
        )

    else:
        true_labels = np.asarray(
            [],
            dtype=np.int32
        )

        predicted_labels = np.asarray(
            [],
            dtype=np.int32
        )

    return (
        average_loss,
        accuracy,
        true_labels,
        predicted_labels
    )


def confusion_matrix(
    true_labels,
    predicted_labels,
    num_classes
):
    matrix = np.zeros(
        (
            num_classes,
            num_classes
        ),
        dtype=np.int64
    )

    for true_label, prediction in zip(
        true_labels,
        predicted_labels
    ):
        matrix[
            int(true_label),
            int(prediction)
        ] += 1

    return matrix


# ============================================================
# 학습
# ============================================================
def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    cfg
):
    train_loss_history = []
    train_acc_history = []

    val_loss_history = []
    val_acc_history = []

    best_val_accuracy = -1.0

    best_model_path = os.path.join(
        cfg.output_dir,
        "best_multimodal_fusion.npz"
    )

    for epoch in range(
        1,
        cfg.epochs + 1
    ):
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        iterator = train_loader.iterate(
            batch_size=cfg.batch_size,
            shuffle=True,
            seed=SEED + epoch
        )

        for batch in tqdm(
            iterator,
            desc=(
                f"Epoch "
                f"{epoch:03d}/"
                f"{cfg.epochs:03d}"
            )
        ):
            logits = model.forward(
                batch,
                training=True
            )

            loss, dlogits = (
                cross_entropy_loss(
                    logits,
                    batch["labels"]
                )
            )

            grads = (
                model
                .fusion_head
                .backward(dlogits)
            )

            optimizer.update(
                grads
            )

            predictions = cp.argmax(
                logits,
                axis=1
            )

            batch_size = int(
                batch["labels"].shape[0]
            )

            total_loss += (
                float(loss.get())
                * batch_size
            )

            total_correct += int(
                cp.sum(
                    predictions
                    == batch["labels"]
                ).get()
            )

            total_count += batch_size

        train_loss = (
            total_loss
            / max(total_count, 1)
        )

        train_accuracy = (
            total_correct
            / max(total_count, 1)
        )

        val_loss, val_accuracy, _, _ = (
            evaluate(
                model,
                val_loader,
                cfg
            )
        )

        train_loss_history.append(
            train_loss
        )

        train_acc_history.append(
            train_accuracy
        )

        val_loss_history.append(
            val_loss
        )

        val_acc_history.append(
            val_accuracy
        )

        print(
            f"\nEpoch {epoch:03d} | "
            f"Train Loss="
            f"{train_loss:.6f} | "
            f"Train Acc="
            f"{train_accuracy:.4f} | "
            f"Val Loss="
            f"{val_loss:.6f} | "
            f"Val Acc="
            f"{val_accuracy:.4f}"
        )

        #if val_accuracy > best_val_accuracy:
        
        if val_accuracy > best_val_accuracy and epoch % 10 == 0:
            best_val_accuracy = (
                val_accuracy
            )

            model.fusion_head.save(
                best_model_path
            )

            print(
                f"[SAVE] best fusion model: "
                f"{best_model_path}"
            )

    # best validation 모델 로드
    model.fusion_head.load(
        best_model_path
    )

    (
        test_loss,
        test_accuracy,
        true_labels,
        predicted_labels
    ) = evaluate(
        model,
        test_loader,
        cfg
    )

    print(
        "\n=============== "
        "Final Test Result "
        "==============="
    )

    print(
        f"Test Loss: "
        f"{test_loss:.6f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    history = {
        "train_loss":
            train_loss_history,
        "train_accuracy":
            train_acc_history,
        "val_loss":
            val_loss_history,
        "val_accuracy":
            val_acc_history,
        "test_loss":
            test_loss,
        "test_accuracy":
            test_accuracy
    }

    return (
        history,
        true_labels,
        predicted_labels
    )


# ============================================================
# 시각화
# ============================================================
def plot_history(
    history,
    output_path
):
    epochs = np.arange(
        1,
        len(
            history[
                "train_loss"
            ]
        ) + 1
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        epochs,
        history[
            "train_accuracy"
        ],
        label="Train Accuracy"
    )

    axes[1].plot(
        epochs,
        history[
            "val_accuracy"
        ],
        label="Validation Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].grid(True)
    axes[1].legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


def plot_confusion_matrix(
    matrix,
    output_path
):
    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    image = axis.imshow(
        matrix,
        cmap="Blues"
    )

    figure.colorbar(
        image,
        ax=axis
    )

    axis.set_xlabel(
        "Predicted"
    )

    axis.set_ylabel(
        "True"
    )

    axis.set_title(
        "Multimodal Authentication "
        "Confusion Matrix"
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size
        else 0.0
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            value = matrix[
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

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


# ============================================================
# Main
# ============================================================
def main():
    cfg = Config()

    os.makedirs(
        cfg.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Manifest 로드
    # --------------------------------------------------------
    dataframe, user_to_label = (
        load_manifest(
            cfg.manifest_csv
        )
    )

    train_dataframe = dataframe[
        dataframe["split"] == "train"
    ].reset_index(drop=True)

    val_dataframe = dataframe[
        dataframe["split"] == "val"
    ].reset_index(drop=True)

    test_dataframe = dataframe[
        dataframe["split"] == "test"
    ].reset_index(drop=True)

    if len(train_dataframe) == 0:
        raise ValueError(
            "train sample이 없습니다."
        )

    if len(val_dataframe) == 0:
        raise ValueError(
            "val sample이 없습니다."
        )

    if len(test_dataframe) == 0:
        raise ValueError(
            "test sample이 없습니다."
        )

    print(
        "Train samples:",
        len(train_dataframe)
    )

    print(
        "Validation samples:",
        len(val_dataframe)
    )

    print(
        "Test samples:",
        len(test_dataframe)
    )

    # --------------------------------------------------------
    # 첫 smell 파일에서 feature dimension 확인
    # --------------------------------------------------------
    first_smell_path = (
        train_dataframe.iloc[0][
            "smell_path"
        ]
    )

    first_smell, smell_feature_cols = (
        load_smell_feature(
            first_smell_path,
            cfg
        )
    )

    smell_input_dim = (
        first_smell.shape[-1]
    )

    print(
        "Smell input dimension:",
        smell_input_dim
    )

    print(
        "Smell feature columns:",
        smell_feature_cols
    )

    # --------------------------------------------------------
    # 기존 Transformer encoder 로드
    # --------------------------------------------------------
    face_encoder = FaceEncoderAdapter(
        cfg
    )

    speech_encoder = (
        SpeechEncoderAdapter(
            cfg
        )
    )

    smell_encoder = (
        SmellEncoderAdapter(
            cfg,
            smell_input_dim
        )
    )

    # --------------------------------------------------------
    # Fusion head
    # --------------------------------------------------------
    fusion_head = MultimodalFusionHead(
        face_dim=(
            face_encoder.output_dim
        ),

        speech_dim=(
            speech_encoder.output_dim
        ),

        smell_dim=(
            smell_encoder.output_dim
        ),

        common_dim=(
            cfg.common_embedding_dim
        ),

        hidden_dim=(
            cfg.fusion_hidden_dim
        ),

        output_dim=(
            cfg.fusion_output_dim
        ),

        num_classes=(
            cfg.num_classes
        ),

        dropout_ratio=(
            cfg.fusion_dropout
        ),

        face_weight=cfg.face_weight,
        speech_weight=cfg.speech_weight,
        smell_weight=cfg.smell_weight
    )

    model = (
        MultimodalAuthenticationModel(
            face_encoder,
            speech_encoder,
            smell_encoder,
            fusion_head
        )
    )

    optimizer = AdamW(
        fusion_head.params,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip
    )

    # --------------------------------------------------------
    # Data loader
    # --------------------------------------------------------
    train_loader = (
        MultimodalBatchLoader(
            train_dataframe,
            cfg,
            speech_encoder,
            smell_feature_cols
        )
    )

    val_loader = (
        MultimodalBatchLoader(
            val_dataframe,
            cfg,
            speech_encoder,
            smell_feature_cols
        )
    )

    test_loader = (
        MultimodalBatchLoader(
            test_dataframe,
            cfg,
            speech_encoder,
            smell_feature_cols
        )
    )

    # --------------------------------------------------------
    # 학습
    # --------------------------------------------------------
    (
        history,
        true_labels,
        predicted_labels
    ) = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        cfg
    )

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------
    history_path = os.path.join(
        cfg.output_dir,
        "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=2
        )

    if user_to_label is not None:
        mapping_path = os.path.join(
            cfg.output_dir,
            "user_to_label.json"
        )

        with open(
            mapping_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                user_to_label,
                file,
                ensure_ascii=False,
                indent=2
            )

    curve_path = os.path.join(
        cfg.output_dir,
        "training_curves.png"
    )

    plot_history(
        history,
        curve_path
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        cfg.num_classes
    )

    matrix_path = os.path.join(
        cfg.output_dir,
        "confusion_matrix.npy"
    )

    np.save(
        matrix_path,
        matrix
    )

    matrix_image_path = os.path.join(
        cfg.output_dir,
        "confusion_matrix.png"
    )

    plot_confusion_matrix(
        matrix,
        matrix_image_path
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        matrix
    )

    print(
        "\n모든 작업이 완료되었습니다."
    )

    #model, history = run_training(
    #    train_loader=train_loader,
    #    val_loader=val_loader,
    #    test_loader=test_loader,
    #    smell_input_dim=smell_input_dim,
    #    num_classes=100,
    #    epochs=100,
    #    batch_size=4,
    #    output_directory="multimodal_outputs_100"
    #)

if __name__ == "__main__":
    main()


