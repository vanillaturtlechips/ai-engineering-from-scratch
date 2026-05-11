# ControlNet, LoRA & Conditioning

> 텍스트만으로는 서투른 제어 신호입니다. ControlNet을 사용하면 사전 훈련된 확산 모델을 복제하고 깊이 맵, 포즈 스켈레톤, 낙서, 에지 이미지로 이를 조종할 수 있습니다. LoRA를 사용하면 1000만 개의 파라미터를 훈련시켜 20억 개 파라미터 모델을 미세 조정할 수 있습니다. 이 둘을 함께 사용하면 Stable Diffusion이 장난감에서 2026년 모든 기관에서 사용하는 이미지 파이프라인으로 변모했습니다.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — LoRA 기반)  
**Time:** ~75 minutes

## 문제 정의

"빨간 드레스를 입고 번화가에서 개를 산책시키는 여성"과 같은 프롬프트는 모델이 *개가 어디에 위치하는지*, *여성의 포즈는 무엇인지*, 또는 *거리의 시점*에 대한 정보를 전혀 제공하지 않습니다. 텍스트는 이미지를 지정하는 데 필요한 정보의 약 10%만 고정할 수 있습니다. 나머지는 시각적인 요소이며, 이를 효율적으로 단어로 설명할 수 없습니다.

모든 신호(포즈, 깊이, 캐니 엣지, 분할 마스크)마다 새로운 조건부 모델을 처음부터 훈련시키는 것은 비효율적입니다. 2.6B 파라미터의 SDXL 백본을 고정한 상태로 유지하면서, 조건 정보를 읽는 작은 사이드 네트워크를 부착하고, 이 네트워크가 백본의 중간 특징을 조정하도록 하고 싶습니다. 이것이 바로 ControlNet입니다.

또한 전체 모델을 재훈련하지 않고 새로운 개념(얼굴, 제품, 스타일 등)을 모델에 가르치고자 합니다. 100배 더 작은 델타(delta)를 원하는데, 이는 기존 어텐션 가중치에 플러그인되는 저랭크 어댑터인 LoRA(Low-Rank Adapters)입니다.

ControlNet + LoRA + 텍스트 = 2026년 실무자의 도구 세트입니다. 대부분의 프로덕션 이미지 파이프라인은 SDXL/SD3/Flux 기반 위에 2-5개의 LoRA, 1-3개의 ControlNet, IP-Adapter를 레이어링합니다.

## 개념

![ControlNet은 인코더를 복제하고, LoRA는 저랭크 델타를 추가합니다](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

사전 훈련된 SD를 가져옵니다. U-Net의 인코더 절반을 *복제*합니다. 원본은 동결합니다. 복제본을 훈련시켜 추가 조건 입력(윤곽선, 깊이, 포즈)을 수용하도록 합니다. 복제본을 원본의 디코더 절반에 *제로 컨볼루션* 스킵 연결(0으로 초기화된 1×1 컨볼루션 — 처음에는 무연산으로 시작, 델타 학습)로 다시 연결합니다.

```
SD U-Net 디코더:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

제로 컨볼루션 초기화는 ControlNet이 항등 함수로 시작함을 의미합니다 — 훈련 전에도 해가 없습니다. 표준 확산 손실로 (프롬프트, 조건, 이미지) 삼중항 100만 개를 훈련합니다.

모달리티별 ControlNet은 작은 사이드 모델(SDXL 기준 ~360M, SD 1.5 기준 ~70M)로 제공됩니다. 추론 시 다음과 같이 조합할 수 있습니다:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

모델 내 모든 선형 레이어 `W ∈ R^{d×d}`에 대해, `W`를 동결하고 저랭크 델타를 추가합니다:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

여기서 `r << d`입니다. 어텐션에는 랭크 4-16, 대규모 파인튜닝에는 64-128이 일반적입니다. 새로운 파라미터 수: `2 · d · r` (기존 `d²` 대비). SDXL 어텐션에서 `d=640`, `r=16`일 때: 어댑터당 20k 파라미터(기존 410k 대비) — 20배 감소. 전체 모델 기준: LoRA는 일반적으로 20-200MB(기준 모델 5GB 대비).

추론 시 LoRA를 스케일링할 수 있습니다: `W' = W + α · B @ A`. `α = 0.5-1.5`가 일반적입니다. 여러 LoRA는 가산적으로 중첩됩니다(비선형 상호작용 주의).

### IP-Adapter (Ye et al., 2023)

*이미지*를 조건(텍스트와 함께)으로 수용하는 초소형 어댑터입니다. CLIP 이미지 인코더를 사용해 이미지 토큰을 생성하고, 이를 텍스트 토큰과 함께 크로스 어텐션에 주입합니다. 기준 모델당 ~20MB. LoRA 없이도 "이 참조 이미지의 스타일로 생성" 작업을 수행할 수 있습니다.

## 컴포저빌리티 매트릭스

| 도구 | 제어 대상 | 크기 | 사용 시기 |
|------|------------------|------|-------------|
| ControlNet | 공간 구조(포즈, 깊이, 에지) | 70-360MB | 정확한 레이아웃, 구성 |
| LoRA | 스타일, 주제, 개념 | 20-200MB | 개인화, 스타일 |
| IP-Adapter | 참조 이미지의 스타일 또는 주제 | 20MB | 텍스트로 설명할 수 없는 외형 |
| Textual Inversion | 새 토큰으로서의 단일 개념 | 10KB | 레거시, 대부분 LoRA로 대체됨 |
| DreamBooth | 주제에 대한 전체 파인튜닝 | 2-5GB | 강한 정체성, 높은 계산 자원 |
| T2I-Adapter | 가벼운 ControlNet 대안 | 70MB | 에지 장치, 추론 예산 제약 |

ControlNet ≈ 공간. LoRA ≈ 의미. 둘 다 사용.

## 구축 방법

`code/main.py`는 1-D에서 두 가지 메커니즘을 시뮬레이션합니다:

1. **LoRA.** 사전 훈련된 선형 계층 `W`. 이를 동결합니다. `W + BA`가 목표 선형 계층과 일치하도록 저랭크 `B @ A`를 훈련합니다. `r = 1`로 랭크-1 보정을 완벽하게 학습할 수 있음을 보여줍니다.

2. **ControlNet-lite.** "동결된 기본" 예측기와 추가 신호를 읽는 "사이드 네트워크". 사이드 네트워크의 출력은 0으로 초기화된 학습 가능한 스칼라(제로-컨브 버전)로 게이팅됩니다. 훈련 시 게이트가 증가하는 것을 관찰합니다.

### 단계 1: LoRA 수학

```python
def lora(W, A, B, x, alpha=1.0):
    # W는 동결; A, B는 훈련 가능한 저랭크 인자입니다.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 단계 2: 제로-초기화 사이드 네트워크

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate는 0으로 초기화됨
h = base(x) + gated
```

0단계에서 출력은 `base`와 동일합니다. 초기 훈련에서는 `gate`가 천천히 업데이트되며 — 치명적인 드리프트가 발생하지 않습니다.

## 주의 사항

- **LoRA 과도한 스케일링.** `α = 2` 또는 `α = 3`은 "더 강력하게"라는 일반적인 해킹 기법이지만 과도하게 스타일화된/깨진 출력을 생성합니다. `α ≤ 1.5`를 유지하세요.
- **ControlNet 가중치 충돌.** Pose ControlNet을 가중치 1.0으로, Depth ControlNet을 가중치 1.0으로 동시에 사용하면 일반적으로 과도한 효과가 발생합니다. 가중치 합계 ≈ 1.0이 안전한 기본값입니다.
- **잘못된 베이스 모델에 LoRA 적용.** SDXL LoRA는 어텐션 차원이 일치하지 않아 SD 1.5에서 무음(no-op) 처리됩니다. Diffusers 0.30+에서는 경고가 표시됩니다.
- **텍스트 인버전 드리프트.** 한 체크포인트에서 학습된 토큰은 다른 체크포인트에서 심하게 드리프트됩니다. LoRA가 더 이식성이 좋습니다.
- **LoRA 가중치 병합 및 저장.** 더 빠른 추론을 위해 LoRA를 베이스 모델 가중치에 병합할 수 있지만(런타임 추가 없음), 런타임에서 `α`를 조정할 수 있는 기능이 사라집니다. 두 버전을 모두 보관하세요.

## 사용 방법

| 목표 | 2026 파이프라인 |
|------|---------------|
| 브랜드의 아트 스타일 재현 | 랭크 32에서 30개의 선별된 이미지로 학습된 LoRA(LoRA) |
| 생성된 이미지에 내 얼굴 넣기 | DreamBooth 또는 LoRA + IP-Adapter-FaceID |
| 특정 포즈 + 프롬프트 | ControlNet-Openpose + SDXL + 텍스트 |
| 깊이 인식 구성 | ControlNet-Depth + SD3 |
| 참조 이미지 + 프롬프트 | IP-Adapter + 텍스트 |
| 정확한 레이아웃 | ControlNet-Scribble 또는 ControlNet-Canny |
| 배경 교체 | ControlNet-Seg + 인페인팅(Lesson 09) |
| 빠른 1단계 스타일 | SDXL-Turbo에 적용된 LCM-LoRA |

## Ship It

`outputs/skill-sd-toolkit-composer.md`를 저장하세요. 이 스킬은 작업(입력 자산: 프롬프트, 선택적 참조 이미지, 선택적 포즈, 선택적 깊이, 선택적 스크리블)을 받아 툴스택, 가중치, 재현 가능한 시드 프로토콜을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`에서 LoRA 랭크 `r`을 1부터 4까지 변화시킵니다. 어떤 랭크에서 LoRA가 정확히 랭크-2 목표 델타(delta)와 일치합니까?
2. **중간.** 두 개의 서로 다른 목표 변환(transform)에 대해 별도의 LoRA를 훈련시킵니다. 이를 함께 로드하고 가법적(additive) 상호작용을 보여줍니다. 상호작용이 선형성(linearity)을 잃는 경우는 언제입니까?
3. **어려움.** diffusers를 사용하여 다음을 스택으로 구성합니다: SDXL-base + Canny-ControlNet(가중치 0.8) + 스타일 LoRA(α 0.8) + IP-Adapter(가중치 0.6). 스택 가중치가 변할 때 FID(프셰 거리) 대 프롬프트 준수도(prompt-adherence) 트레이드오프를 측정합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| ControlNet | "공간 제어" | 복제된 인코더 + 제로 컨볼루션(zero-conv) 스킵; 조건부 이미지를 읽습니다. |
| Zero convolution | "항등 함수로 시작" | 1×1 컨볼루션(conv)을 0으로 초기화; ControlNet은 처음에는 아무 작업도 하지 않습니다. |
| LoRA | "저랭크 어댑터" | `W + B @ A`, `r << d`; 전체 파인튜닝보다 100배 적은 파라미터. |
| rank r | "조절 장치" | LoRA 압축; 일반적으로 4-16, 64+는 강한 개인화용. |
| α | "LoRA 강도" | 런타임에서 LoRA 델타(delta)를 스케일링합니다. |
| IP-Adapter | "참조 이미지" | CLIP-이미지 토큰을 통한 소형 이미지 조건부 어댑터. |
| DreamBooth | "전체 주제 파인튜닝" | 특정 주제(subject)의 ~30개 이미지로 전체 모델을 학습시킵니다. |
| Textual Inversion | "새 토큰" | 새로운 단어 임베딩만 학습; 레거시 기술, 대부분 대체됨. |

## 프로덕션 노트: LoRA 스왑, ControlNet 레인, 멀티테넌트 서빙

실제 텍스트-이미지 SaaS는 동일한 기본 체크포인트 위에서 수백 개의 LoRA와 수십 개의 ControlNet을 서빙합니다. 이 서빙 문제는 LLM 멀티테넌시와 매우 유사합니다(프로덕션 문헌에서는 연속 배치 처리 및 LoRAX/S-LoRA 하에서 LLM 사례를 다룹니다):

- **핫 스왑 LoRA, 병합하지 마세요.** `W' = W + α·B·A`를 기본 모델에 병합하면 단계별 추론이 ~3-5% 빨라지지만 `α`와 기본 모델이 고정됩니다. LoRA를 VRAM에 랭크-r 델타로 핫 상태로 유지하세요. `diffusers`는 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`를 제공하여 요청별 활성화를 지원합니다. 스왑 비용은 `2 · d · r · num_layers` 가중치이며, MB 규모이고 수 초 미만입니다.
- **ControlNet을 두 번째 어텐션 레인으로.** 복제된 인코더는 기본 모델과 병렬로 실행됩니다. 각각 가중치 1.0인 두 개의 ControlNet은 단계별 병합된 패스가 아닌 두 번의 추가 순방향 패스를 의미합니다. 배치 크기 여유 공간이 기하급수적으로 감소합니다. 활성 ControlNet당 ~1.5× 단계 비용을 예산에 반영하세요.
- **양자화된 LoRA도 가능.** 기본 모델을 양자화했다면(레슨 07, 8GB의 Flux 참조), LoRA 델타도 8비트 또는 4비트로 깔끔하게 양자화할 수 있습니다. QLoRA 스타일 로딩을 사용하면 4비트 Flux 기본 모델 위에 5-10개의 LoRA를 메모리 초과 없이 쌓을 수 있습니다.

Flux 관련: Niels의 Flux-on-8GB 노트북은 기본 모델을 4비트로 양자화합니다. 이 양자화된 기본 모델 위에 스타일 LoRA(`pipe.load_lora_weights("user/style-lora")`)를 `weight_name="pytorch_lora_weights.safetensors"`로 로드하는 것이 여전히 작동합니다. 이는 2026년 대부분의 SaaS 에이전시가 사용하는 레시피입니다.

## 추가 자료

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (원래 LLM용; diffusion으로 이식).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet의 경량 대안.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter 문서](https://huggingface.co/docs/diffusers/training/controlnet) — 참조 파이프라인.