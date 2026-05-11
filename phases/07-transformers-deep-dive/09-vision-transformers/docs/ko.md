# Vision Transformers (ViT)

> 이미지는 패치들의 격자이고, 문장은 토큰들의 격자입니다. 동일한 트랜스포머가 둘 다 처리합니다.

**Type:** 구축
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (풀 트랜스포머), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers 소개)
**Time:** ~45분

## 문제 정의

2020년 이전에는 컴퓨터 비전이 컨볼루션(convolution)을 의미했습니다. ImageNet, COCO 및 검출 벤치마크에서 모든 SOTA(State-of-the-Art) 모델은 CNN 백본을 사용했습니다. 트랜스포머(Transformer)는 언어 처리 전용이었습니다.

Dosovitskiy et al. (2020) — "An Image is Worth 16x16 Words" — 는 컨볼루션을 완전히 제거할 수 있음을 보였습니다. 이미지를 고정 크기 패치로 분할하고, 각 패치를 임베딩(embedding)으로 선형 투영한 후, 시퀀스를 기본 트랜스포머 인코더에 입력합니다. 충분한 규모(ImageNet-21k 사전 학습 또는 그 이상)에서 ViT(Vision Transformer)는 ResNet 기반 모델과 성능에서 맞먹거나 능가합니다.

ViT는 2026년에 더 넓은 패턴의 시작점이 되었습니다: 하나의 아키텍처로 여러 모달리티를 처리합니다. Whisper는 오디오를 토큰화합니다. ViT는 이미지를 토큰화합니다. 로봇 공학을 위한 액션 토큰(action tokens), 비디오를 위한 픽셀 토큰(pixel tokens)이 있습니다. 트랜스포머는 신경 쓰지 않습니다 — 시퀀스를 입력하면 학습합니다.

2026년에는 ViT와 그 파생 모델(DeiT, Swin, DINOv2, ViT-22B, SAM 3)이 대부분의 컴퓨터 비전 분야를 장악했습니다. CNN은 여전히 엣지 장치 및 지연 시간에 민감한 작업에서 우위를 차지합니다. 그 외 모든 작업에는 스택 어딘가에 ViT가 있습니다.

## 개념

![Image → 패치 → 토큰 → 트랜스포머](../assets/vit.svg)

### 단계 1 — 패치화(patchify)

`H × W × C` 이미지를 `N × (P·P·C)`의 평탄한 패치 시퀀스로 분할합니다. 일반적인 설정: `224 × 224` 이미지, `16 × 16` 패치 → 각각 768개의 값을 가진 196개 패치.

```
image (224, 224, 3) → 14 × 14 그리드의 16x16x3 패치 → 길이 768의 196개 벡터
```

패치 크기는 조절 가능한 레버입니다. 작은 패치 = 더 많은 토큰, 더 나은 해상도, 이차적 어텐션 비용. 큰 패치 = 더 거친 표현, 더 저렴한 비용.

### 단계 2 — 선형 임베딩(linear embedding)

단일 학습된 행렬이 각 평탄한 패치를 `d_model`로 투영합니다. 커널 크기 `P`와 스트라이드 `P`의 컨볼루션과 동일합니다. PyTorch에서는 문자 그대로 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — 2줄 구현입니다.

### 단계 3 — `[CLS]` 토큰 추가, 위치 임베딩 추가

- 학습 가능한 `[CLS]` 토큰을 앞에 추가합니다. 최종 은닉 상태는 분류에 사용되는 이미지 표현입니다.
- 학습 가능한 위치 임베딩(ViT-원본) 또는 사인파 2D(후기 변형)를 추가합니다.
- 2024+ RoPE는 위치를 위해 2D로 확장되었으며, 때로는 명시적 임베딩 없이 사용됩니다.

### 단계 4 — 표준 트랜스포머 인코더

`LayerNorm → Self-Attention → + → LayerNorm → MLP → +`의 L개 블록을 쌓습니다. BERT와 동일합니다. 비전 전용 레이어가 없습니다. 이것이 논문의 교육적 핵심 포인트입니다.

### 단계 5 — 헤드(head)

분류용: `[CLS]` 은닉 상태 → 선형 → 소프트맥스. DINOv2 또는 SAM의 경우: `[CLS]`를 버리고 패치 임베딩을 직접 사용합니다.

### 중요한 변형들

| 모델 | 연도 | 변경 사항 |
|-------|------|--------|
| ViT | 2020 | 원본. 고정된 패치 크기, 전체 글로벌 어텐션. |
| DeiT | 2021 | 지식 증류; ImageNet-1k에서만 학습 가능. |
| Swin | 2021 | 시프트된 윈도우를 사용한 계층적 구조. 고정 하위 이차 비용. |
| DINOv2 | 2023 | 자기 지도 학습(라벨 없음). 최고의 일반 비전 특징. |
| ViT-22B | 2023 | 220억 파라미터; 확장 법칙 적용. |
| SigLIP | 2023 | ViT + 언어 쌍, 시그모이드 대조 손실. |
| SAM 3 | 2025 | 모든 것 분할; ViT-Large + 프롬프트 가능 마스크 디코더. |

### 왜 시간이 걸렸는가

ViT는 CNN의 귀납적 편향(번역 불변성, 지역성)이 없기 때문에 CNN과 맞먹으려면 *많은* 데이터가 필요합니다. 1억 개 이상의 라벨된 이미지나 강력한 자기 지도 사전 학습이 없으면, 동일한 계산량에서 CNN이 여전히 우세합니다. DeiT는 2021년 증류 트릭으로 이를 해결했고, DINOv2는 2023년 자기 지도 학습으로 영구적으로 해결했습니다.

## 구축 방법

`code/main.py`를 참조하세요. 순수 표준 라이브러리(pure-stdlib) 패치화(patchify) + 선형 임베딩(linear embedding) + 검증(sanity checks) 구현. 학습(training)은 없음 — 현실적인 규모의 ViT는 PyTorch와 GPU 시간이 수 시간 필요합니다.

### 1단계: 가상 이미지 생성

24 × 24 RGB 이미지를 `(R, G, B)` 튜플의 행 리스트로 표현. 6×6 패치 사용 → 16개 패치, 각각 108차원 임베딩 벡터.

### 2단계: 패치화(patchify)

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

래스터 순서(raster order): 그리드 전체를 행 우선(row-major)으로 처리. 모든 ViT에서 이 순서를 사용합니다.

### 3단계: 선형 임베딩(linear embed)

각 평탄화된 패치(flat patch)를 랜덤 `(patch_flat_size, d_model)` 행렬과 곱합니다. `[CLS]` 토큰을 앞에 추가한 후 출력 형태가 `(N_patches + 1, d_model)`인지 검증합니다.

### 4단계: 현실적인 ViT 파라미터 수 계산

ViT-Base의 파라미터 수를 출력: 12개 레이어, 12개 헤드, d=768, 패치=16. ResNet-50(~25M)과 비교. ViT-Base는 ~86M, ViT-Large는 ~307M, ViT-Huge는 ~632M 파라미터를 가집니다.

## 사용 방법

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 패치
cls_emb = out[:, 0]                       # 이미지 표현
```

**DINOv2 임베딩은 이미지 특징의 2026년 기본값입니다.** 백본을 고정하고 작은 헤드만 학습시킵니다. 분류, 검색, 검출, 캡션 생성에 모두 작동합니다. Meta의 DINOv2 체크포인트는 텍스트가 없는 모든 비전 작업에서 CLIP을 능가합니다.

**패치 크기 선택.** 소형 모델은 16×16(ViT-B/16)을 사용합니다. 밀집 예측(세분화)은 8×8 또는 14×14(SAM, DINOv2)를 사용합니다. 매우 큰 모델은 14×14를 사용합니다.

## Ship It

`outputs/skill-vit-configurator.md`를 참조하세요. 이 스킬은 데이터셋 크기, 해상도, 컴퓨팅 예산을 고려하여 새로운 비전 작업에 적합한 ViT 변형(variant)과 패치 크기를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. 패치 수가 `(H/P) * (W/P)`와 같고 평탄화된 패치 차원이 `P*P*C`와 같은지 확인합니다.
2. **중간.** 2D 사인 곡선 위치 임베딩 구현 — 각 패치의 `행(row)`과 `열(col)`에 대해 독립적인 두 개의 사인 곡선 코드를 생성하고 연결합니다. 이를 소형 PyTorch ViT에 입력하고 CIFAR-10에서 학습 가능한 위치 임베딩 대비 정확도를 비교합니다.
3. **어려움.** 3계층 ViT(PyTorch)를 구축하고 4×4 패치로 1,000개의 MNIST 이미지를 학습시킵니다. 테스트 정확도를 측정합니다. 이제 동일한 1,000개 이미지에 대해 DINOv2 사전 학습을 추가합니다(단순화: 마스킹된 패치로부터 패치 임베딩을 예측하도록 인코더를 학습). 정확도가 향상되는지 확인합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 패치(Patch) | "비전 트랜스포머 토큰" | 이미지의 `P × P × C` 영역에 대한 픽셀 값의 평탄한 벡터. |
| 패치화(Patchify) | "잘라 + 평탄화" | 이미지를 겹치지 않는 패치로 분할하고, 각각을 벡터로 평탄화. |
| `[CLS]` 토큰 | "이미지 요약" | 앞에 추가된 학습 가능한 토큰; 최종 임베딩이 이미지 표현. |
| 귀납적 편향(Inductive bias) | "모델이 가정하는 것" | ViT는 CNN보다 사전 지식이 적음; 부족한 부분을 보완하려면 더 많은 데이터 필요. |
| DINOv2 | "자기 지도 학습 ViT" | 이미지 증강 + 모멘텀 티처를 사용해 라벨 없이 학습. 2026년 최고의 일반 이미지 특징. |
| SigLIP | "CLIP의 후속 모델" | 시그모이드 대조 손실로 학습된 ViT + 텍스트 인코더; 동일 계산량에서 CLIP보다 우수. |
| Swin | "윈도우 ViT" | 계층적 ViT에 지역 어텐션 + 시프트 윈도우 적용; 이차보다 낮은 복잡도. |
| 레지스터 토큰(Register tokens) | "2023년 트릭" | 어텐션 싱크를 흡수하는 몇 개의 추가 학습 가능 토큰; DINOv2 특징 개선. |

## 추가 자료

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 논문.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2를 위한 레지스터 토큰(register-token) 수정 방법.