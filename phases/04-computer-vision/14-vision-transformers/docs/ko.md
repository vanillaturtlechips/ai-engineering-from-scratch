# Vision Transformers (ViT)

> 이미지를 패치로 분할하고, 각 패치를 단어처럼 처리한 후 표준 트랜스포머를 실행합니다. 뒤돌아보지 마세요.

**유형:** Build  
**언어:** Python  
**선수 지식:** Phase 7 Lesson 02 (Self-Attention), Phase 4 Lesson 04 (Image Classification)  
**소요 시간:** ~45분

## 학습 목표

- 패치 임베딩(patch embedding), 학습된 위치 임베딩(learned positional embedding), 클래스 토큰(class token), 트랜스포머 인코더 블록(transformer encoder block)을 직접 구현하여 최소한의 ViT(Visual Transformer)를 구축
- DeiT와 MAE가 등장하기 전까지 ViT가 대규모 사전 학습 데이터를 필요로 한다고 여겨졌던 이유를 설명
- ViT, Swin, ConvNeXt의 아키텍처 사전 조건(architectural priors) 비교: (사전 조건 없음, 로컬 윈도우 어텐션(local window attention), 컨볼루션 백본(conv backbone))
- `timm` 라이브러리와 표준 선형 프로브(linear-probe) / 파인튜닝(fine-tuning) 레시피를 사용하여 소규모 데이터셋에서 사전 학습된 ViT를 파인튜닝(fine-tuning) 수행

## 문제 정의

10년 동안 합성곱(convolution)은 컴퓨터 비전(computer vision)과 동의어로 사용되었습니다. CNN은 강력한 귀납적 편향(inductive bias) — 지역성(locality), 이동 등가성(translation equivariance) — 을 가지고 있어 이를 대체할 수 없다고 여겨졌습니다. 그러나 Dosovitskiy et al. (2020)은 합성곱 연산 없이도 평탄화된 이미지 패치(flattened image patches)에 일반 트랜스포머(plain transformer)를 적용하면 대규모에서 최고의 CNN과 맞먹거나 능가할 수 있음을 보였습니다.

단, "대규모에서"라는 조건이 있었습니다. ImageNet-1k에서 ViT는 ResNet에 패배했습니다. 반면 ImageNet-21k 또는 JFT-300M으로 사전 학습된 후 ImageNet-1k에서 파인튜닝(fine-tuning)된 ViT는 ResNet을 능가했습니다. 결론은 트랜스포머는 유용한 사전 지식(prior)이 부족하지만 충분한 데이터로부터 이를 학습할 수 있다는 것이었습니다. 이후 연구(DeiT, MAE, DINO)는 적절한 학습 방법 — 강력한 증강(strong augmentation), 자기 지도 사전 학습(self-supervised pretraining), 증류(distillation) — 을 통해 ViT도 소규모 데이터에서 잘 학습될 수 있음을 보였습니다.

2026년 기준으로 순수 CNN은 여전히 엣지 디바이스(edge devices)에서 경쟁력을 유지하고 있지만(ConvNeXt가 가장 강력함), 트랜스포머는 다른 모든 분야에서 우위를 점하고 있습니다: 분할(Segmentation, Mask2Former, SegFormer), 검출(Detection, DETR, RT-DETR), 멀티모달(Multimodal, CLIP, SigLIP), 비디오(VideoMAE, VJEPA). ViT 블록 구조는 반드시 알아야 할 핵심 구조입니다.

## 개념

### 파이프라인

```mermaid
flowchart LR
    IMG["이미지<br/>(3, 224, 224)"] --> PATCH["패치 임베딩<br/>conv 16x16 s=16<br/>-> (768, 14, 14)"]
    PATCH --> FLAT["평탄화(flatten) to<br/>(196, 768) 토큰"]
    FLAT --> CAT["[CLS] 토큰 추가"]
    CAT --> POS["학습된 위치 임베딩 추가"]
    POS --> ENC["N개의 트랜스포머<br/>인코더 블록"]
    ENC --> CLS["[CLS] 토큰 출력 추출"]
    CLS --> HEAD["MLP 분류기"]

    style PATCH fill:#dbeafe,stroke:#2563eb
    style ENC fill:#fef3c7,stroke:#d97706
    style HEAD fill:#dcfce7,stroke:#16a34a
```

7단계. 패치 → 토큰 → 어텐션 → 분류기. 모든 변형(DeiT, Swin, ConvNeXt, MAE 사전학습)은 7단계 중 1~2단계를 변경하고 나머지는 그대로 유지합니다.

### 패치 임베딩

첫 번째 컨볼루션(convolution)이 핵심입니다. 커널 크기 16, 스트라이드 16으로 224x224 이미지는 14x14 그리드의 16x16 패치로 변환되고, 각 패치는 768차원 임베딩으로 투영됩니다. 이 단일 컨볼루션은 패치화와 선형 투영을 동시에 수행합니다.

```
입력:  (3, 224, 224)
컨볼루션 (3 → 768, k=16, s=16, 패딩 없음):
출력: (768, 14, 14)
공간 차원 평탄화: (196, 768)
```

196개 패치 = 196개 토큰. 각 토큰의 특징 차원은 768(ViT-B), 1024(ViT-L), 또는 1280(ViT-H)입니다.

### 클래스 토큰

시퀀스 앞에 추가되는 단일 학습 벡터입니다:

```
토큰 = [CLS; 패치_1; 패치_2; ...; 패치_196]   형태 (197, 768)
```

N개의 트랜스포머 블록 이후, `[CLS]` 출력은 전체 이미지 표현입니다. 분류 헤드는 이 단일 벡터만 읽습니다.

### 위치 임베딩

트랜스포머는 공간 위치에 대한 내재적 개념이 없습니다. 모든 토큰에 학습된 벡터를 추가합니다:

```
토큰 = 토큰 + 학습된 위치 임베딩   (형태 (197, 768))
```

임베딩은 모델의 매개변수이며, 경사 기반 학습은 이를 2D 이미지 구조에 적응시킵니다. 사인/코사인 2D 대안도 존재하지만 실제로는 거의 사용되지 않습니다.

### 트랜스포머 인코더 블록

표준 구조입니다. 멀티헤드 자기 어텐션, MLP, 잔차 연결, 사전 레이어 정규화(pre-LayerNorm)를 포함합니다.

```
x = x + MSA(LN(x))
x = x + MLP(LN(x))

MLP는 GELU 활성화 함수를 사용하는 2층 구조: Linear(d → 4d) → GELU → Linear(4d → d)
```

ViT-B/16은 12개의 블록을 쌓으며, 각 블록은 12개의 어텐션 헤드를 가지며 총 86M 매개변수를 가집니다.

### 사전 레이어 정규화(pre-LN) 이유

초기 트랜스포머는 사후 레이어 정규화(`x = LN(x + sublayer(x))`)를 사용했고, 워밍업 없이 6~8층 이상 학습이 어려웠습니다. 사전 레이어 정규화(`x = x + sublayer(LN(x))`)는 워밍업 없이도 깊은 네트워크를 안정적으로 학습시킵니다. 모든 ViT와 현대 LLM은 사전 레이어 정규화를 사용합니다.

### 패치 크기 트레이드오프

- 16x16 패치 → 196 토큰, 표준.
- 32x32 패치 → 49 토큰, 빠르지만 해상도 낮음.
- 8x8 패치 → 784 토큰, 세밀하지만 O(n²) 어텐션 비용이 급증.

큰 패치 = 적은 토큰 = 빠르지만 공간 세부 정보 감소. SwinV2는 계층적 윈도우에서 4x4 패치를 사용합니다.

### DeiT의 ImageNet-1k 학습 레시피

원본 ViT는 CNN을 능가하기 위해 JFT-300M이 필요했습니다. DeiT(Touvron et al., 2020)는 ImageNet-1k만으로 ViT-B를 81.8% Top-1 정확도로 학습시켰으며, 4가지 변경 사항을 적용했습니다:

1. 강력한 증강: RandAugment, Mixup, CutMix, Random Erasing.
2. 확률적 깊이(학습 중 랜덤하게 전체 블록 드롭).
3. 반복 증강(동일 이미지를 배치당 3번 샘플링).
4. CNN 교사 모델로부터의 지식 증류(선택 사항, 정확도 추가 향상).

모든 현대 ViT 학습 레시피는 DeiT에서 파생되었습니다.

### Swin vs ConvNeXt

- **Swin** (Liu et al., 2021) — 윈도우 기반 어텐션. 각 블록은 로컬 윈도우 내에서만 어텐션을 수행하며, 번갈아 가며 윈도우를 이동시켜 정보를 혼합합니다. 어텐션 연산자를 유지하면서 CNN과 같은 지역성 사전 정보를 복원합니다.
- **ConvNeXt** (Liu et al., 2022) — Swin의 아키텍처 선택(깊이별 컨볼루션, LayerNorm, GELU, 인버티드 병목)을 따르는 재설계된 CNN. 격차는 "어텐션 vs 컨볼루션"이 아니라 "현대적 학습 레시피 + 아키텍처"임을 입증했습니다.

2026년에는 ConvNeXt-V2와 Swin-V2가 모두 프로덕션 등급이며, 올바른 선택은 추론 스택(에지 장치에 더 잘 컴파일되는 ConvNeXt)과 사전학습 코퍼스에 따라 달라집니다.

### MAE 사전학습

마스크드 오토인코더(He et al., 2022): 무작위로 75% 패치를 마스킹하고, 인코더가 보이는 25%만 처리하도록 학습시킨 후, 작은 디코더가 인코더 출력으로부터 마스킹된 패치를 재구성하도록 학습시킵니다. 사전학습 후 디코더를 버리고 인코더를 미세 조정합니다.

MAE는 ViT를 ImageNet-1k만으로 학습 가능하게 하며, SOTA를 달성하고 현재 기본 자기지도 학습 레시피입니다.

## 구축 방법

### 1단계: 패치 임베딩

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, dim=192, image_size=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)
```

하나의 합성곱(Convolution), 하나의 평탄화(Flatten), 하나의 전치(Transpose)가 전체 이미지-토큰 변환 단계입니다.

### 2단계: 트랜스포머 블록

사전 정규화(Pre-LN), 멀티헤드 자기 어텐션(Multi-head Self-Attention), GELU 활성화 함수를 사용한 MLP, 잔차 연결(Residual Connections)로 구성됩니다.

```python
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x
```

`nn.MultiheadAttention`은 헤드 분할, 스케일링된 닷-프로덕트, 출력 투영을 처리합니다. `batch_first=True`이므로 형태는 `(N, seq, dim)`입니다.

### 3단계: ViT 구현

```python
class ViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 num_classes=10, dim=192, depth=6, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.patch = PatchEmbedding(in_channels, patch_size, dim, image_size)
        num_patches = self.patch.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x[:, 0])
        return self.head(x)

vit = ViT(image_size=64, patch_size=16, num_classes=10, dim=192, depth=6, num_heads=3)
x = torch.randn(2, 3, 64, 64)
print(f"출력: {vit(x).shape}")
print(f"파라미터 수: {sum(p.numel() for p in vit.parameters()):,}")
```

약 280만 개의 파라미터를 가진 작은 ViT로 CPU에서도 실행 가능합니다. 실제 ViT-Base는 8,600만 개의 파라미터를 가지며, 동일한 클래스 정의에 `dim=768, depth=12, num_heads=12`로 설정하면 됩니다.

### 4단계: 단일 이미지 추론 검증

```python
logits = vit(torch.randn(1, 3, 64, 64))
print(f"로그잇: {logits}")
print(f"확률:  {logits.softmax(-1)}")
```

오류 없이 실행되어야 하며, 확률의 합은 1이 되어야 합니다.

## 사용 방법

`timm`은 모든 ViT(Vision Transformer) 변형 모델을 ImageNet 사전 훈련 가중치와 함께 제공합니다. 한 줄로 사용 가능:

```python
import timm

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=10)
```

`timm`은 2026년 기준 비전 트랜스포머의 프로덕션 기본 라이브러리입니다. ViT, DeiT, Swin, Swin-V2, ConvNeXt, ConvNeXt-V2, MaxViT, MViT, EfficientFormer 및 동일한 API 하에서 수십 가지 다른 모델을 지원합니다.

멀티모달 작업(이미지 + 텍스트)의 경우, `transformers`는 CLIP, SigLIP, BLIP-2, LLaVA를 제공합니다. 이 모든 모델의 이미지 인코더는 ViT 변형 모델입니다.

## Ship It

이 레슨은 다음을 생성합니다:

- `outputs/prompt-vit-vs-cnn-picker.md` — 데이터셋 크기, 컴퓨팅 자원, 추론 스택에 따라 ViT, ConvNeXt, Swin 중 선택하는 프롬프트.
- `outputs/skill-vit-patch-and-pos-embed-inspector.md` — ViT의 패치 임베딩과 위치 임베딩 형태가 모델의 예상 시퀀스 길이와 일치하는지 확인하는 스킬로, 가장 흔한 포팅 버그를 잡아냅니다.

## 연습 문제

1. **(쉬움)** 위의 작은 ViT를 통한 순전파 과정에서 모든 중간 텐서의 형태를 출력하세요. 확인: 입력 `(N, 3, 64, 64)` -> 패치 `(N, 16, 192)` -> CLS 추가 `(N, 17, 192)` -> 분류기 입력 `(N, 192)` -> 출력 `(N, num_classes)`.
2. **(중간)** 4강의 합성-CIFAR 데이터셋에서 사전 학습된 `timm` ViT-S/16을 파인튜닝(fine-tuning)하세요. 동일한 데이터에서 ResNet-18 파인튜닝과 비교하세요. 학습 시간과 최종 정확도를 보고하세요.
3. **(어려움)** 작은 ViT에 대해 MAE 사전 학습을 구현하세요: 패치의 75%를 마스킹하고, 인코더 + 작은 디코더를 훈련시켜 마스킹된 패치를 복원하세요. 사전 학습 전후의 합성 데이터에서 선형 평가(linear-probe) 정확도를 평가하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| 패치 임베딩(patch embedding) | "첫 번째 컨볼루션(conv)" | 커널 크기 = 스트라이드 = 패치 크기인 컨볼루션; 이미지를 토큰 임베딩 그리드로 변환 |
| 클래스 토큰(class token) | "[CLS]" | 토큰 시퀀스 앞에 추가된 학습된 벡터; 최종 출력은 전체 이미지 표현 |
| 위치 임베딩(positional embedding) | "학습된 위치 정보(pos)" | 모든 토큰에 추가되는 학습된 벡터; 트랜스포머가 각 패치의 위치를 알 수 있도록 함 |
| 프리-정규화(Pre-LN) | "서브레이어 전 레이어 정규화" | 안정적인 트랜스포머 변형: `x + sublayer(LN(x))` 대신 `LN(x + sublayer(x))` 사용 |
| 멀티헤드 어텐션(multi-head attention) | "병렬 어텐션" | 표준 트랜스포머 어텐션을 num_heads 개의 독립 부분 공간으로 분할 후 연결 |
| ViT-B/16 | "베이스, 패치 16" | 표준 크기: dim=768, depth=12, heads=12, patch_size=16, 이미지=224; 약 86M 파라미터 |
| DeiT | "데이터 효율적 ViT" | 강력한 증강 기법으로 ImageNet-1k만으로 훈련된 ViT; 대규모 사전 훈련 데이터셋이 필수적이지 않음을 입증 |
| MAE | "마스킹된 오토인코더" | 자기 지도 사전 훈련: 75% 패치 마스킹 후 복원; ViT 사전 훈련의 주요 방법 |

## 추가 자료

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) — ViT 논문
- [DeiT: Data-efficient Image Transformers (Touvron et al., 2020)](https://arxiv.org/abs/2012.12877) — ImageNet-1k만으로 ViT를 학습시키는 방법
- [Masked Autoencoders are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377) — MAE 사전 학습
- [timm documentation](https://huggingface.co/docs/timm) — 프로덕션에서 사용할 모든 비전 트랜스포머의 레퍼런스