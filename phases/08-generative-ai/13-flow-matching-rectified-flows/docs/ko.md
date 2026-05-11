# 플로우 매칭 & 렉티파이드 플로우

> 확산 모델은 잡음에서 데이터까지 곡선 경로를 걷기 때문에 20-50개의 샘플링 단계가 필요합니다. 플로우 매칭(Lipman et al., 2023)과 렉티파이드 플로우(Liu et al., 2022)는 직선 경로를 학습했습니다. 더 직선적인 경로는 더 적은 단계를 의미하며, 이는 더 빠른 추론을 가능하게 합니다. Stable Diffusion 3, Flux.1, AudioCraft 2는 모두 2024년에 플로우 매칭으로 전환했습니다.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** Phase 8 · 06 (DDPM), Phase 1 · 미적분학
**소요 시간:** ~45분

## 문제 정의

DDPM의 역과정은 `N(0, I)`에서 데이터 분포로의 1000단계 확률적 과정입니다. DDIM은 이를 20-50단계의 결정적 단계로 축소했습니다. 우리는 더 적은 단계—이상적으로는 1단계—를 원합니다. 문제는 역과정을 푸는 ODE가 강성(stiff)하며 경로가 곡선 형태라는 점입니다.

만약 노이즈에서 데이터로의 경로가 *직선*이 되도록 모델을 훈련시킬 수 있다면, `t=1`에서 `t=0`으로의 단일 오일러 단계(Euler step)로도 충분할 것입니다. 플로우 매칭(flow matching)은 이를 직접 구현합니다: `x_1 ∼ N(0, I)`에서 `x_0 ∼ data`로의 직선 보간법을 정의하고, 시간 도함수를 일치시키는 벡터 필드 `v_θ(x, t)`를 훈련시킨 후 추론 시 적분합니다.

직교 흐름(Rectified flow, Liu 2022)은 더 나아갑니다: 재흐름(reflow) 절차를 통해 경로를 반복적으로 직선화하며, 점진적으로 선형 ODE에 가까워지는 경로를 생성합니다. 2회의 재흐름 반복 후, 2단계 샘플러가 50단계 DDPM 품질과 동등해집니다.

## 개념

![Flow matching: 노이즈와 데이터 사이의 직선 보간](../assets/flow-matching.svg)

### 직선 흐름

정의:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

여기서 `x_0 ~ 데이터`이고 `x_1 ~ N(0, I)`입니다. 이 직선 경로를 따라 시간 미분은 상수입니다:

```
dx_t / dt = x_1 - x_0
```

신경망 벡터장 `v_θ(x_t, t)`를 정의하고 이 미분과 일치하도록 훈련시킵니다:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

이것은 **조건부 흐름 매칭** 손실 함수(Lipman 2023)입니다. 훈련은 시뮬레이션 없이 진행됩니다: ODE를 전개하지 않습니다. 단순히 `(x_0, x_1, t)`를 샘플링하고 회귀 분석을 수행합니다.

### 샘플링

추론 시 학습된 벡터장을 시간 역방향으로 적분합니다:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

`x_1 ~ N(0, I)`에서 시작하여 오일러 방법으로 `t=0`까지 하강합니다.

### 정류 흐름(Rectified flow) (Liu 2022)

직선 흐름은 작동하지만 학습된 경로는 실제로 직선이 아닙니다 — 여러 `x_0`가 동일한 `x_1`로 매핑될 수 있기 때문에 경로가 휘어집니다. 정류 흐름의 재흐름 단계:

1. 무작위 쌍으로 흐름 모델 v_1을 훈련시킵니다.
2. v_1을 `x_1`에서 그 도착점 `x_0`까지 적분하여 N개의 쌍 `(x_1, x_0)`을 샘플링합니다.
3. 이 쌍으로 v_2를 훈련시킵니다. 쌍이 이제 "ODE-매칭"되었기 때문에 이들 사이의 직선 보간은 실제로 더 평탄해집니다.
4. 반복합니다.

실제로 2번의 재흐름 반복으로 거의 선형에 도달하여 2-4단계 추론이 가능해집니다. SDXL-Turbo, SD3-Turbo, LCM은 모두 흐름 매칭 모델에서 증류된 모델입니다.

### 2024년 이미지 분야에서 승리한 이유

세 가지 이유:

1. **시뮬레이션 없는 훈련** — 훈련 중 ODE 전개 불필요, 구현이 간단합니다.
2. **더 나은 손실 기하학** — 직선 경로는 일관된 신호 대 잡음비를 가지는 반면, DDPM ε-손실은 스케줄의 경계에서 나쁜 SNR을 가집니다.
3. **더 빠른 추론** — SDXL-Turbo 품질로 4-8단계, 일관성 증류로 1단계 가능합니다.

## Flow matching vs DDPM — 정확한 연결 관계

가우시안 조건부 경로를 사용한 Flow matching은 *특정 노이즈 스케줄*을 가진 확산 모델(diffusion model)입니다. `x_t = α(t) x_0 + σ(t) x_1` 스케줄을 선택하면 Flow matching은 `v = α'(t)·x_0 - σ'(t)·x_1`인 Stratonovich 재구성 확산 모델을 복원합니다. 가우시안 경로의 경우 두 방법은 대수적으로 동등합니다.

Flow matching이 추가한 것: 목표(단순한 속도)의 *명확성*, 더 깔끔한 손실 함수(loss function), 그리고 비가우시안 보간법(interpolant) 실험을 위한 자유도입니다.

## 빌드하기

`code/main.py`는 두 모드 가우시안 혼합에 대한 1-D 흐름 매칭을 구현합니다. 벡터 필드 `v_θ(x, t)`는 직선 타겟으로 학습된 작은 MLP입니다. 추론 시 1, 2, 4, 20개의 오일러 단계를 통합하고 샘플 품질을 비교합니다.

### 단계 1: 학습 손실

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # 역전파 + 업데이트
```

### 단계 2: 다중 단계 추론

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 단계 3: 단계 수 비교

4단계 샘플러가 이미 20단계 품질과 일치할 것으로 예상됩니다 — 지연 시간 측면에서 큰 의미가 있습니다.

## 함정(Pitfalls)

- **시간 매개변수화(Time parameterization).** Flow matching은 `t ∈ [0, 1]`을 사용하며 `t=0`은 데이터, `t=1`은 노이즈에 해당합니다. DDPM은 `t ∈ [0, T]`를 사용하며 `t=0`은 데이터, `t=T`는 노이즈에 해당합니다. 방향은 동일하지만 스케일이 다릅니다. 논문에서 이 부분을 자주 잘못 기술합니다.
- **스케줄 선택(Schedule choice).** Rectified flow의 직선 경로는 "the" flow-matching 스케줄이지만, 더 나은 스케일 커버리지를 위해 코사인(cosine) 또는 로짓-정규(logit-normal) t-샘플링을 사용할 수 있습니다(SD3는 이 방식을 사용합니다).
- **리플로우 비용(Reflow cost).** 리플로우를 위한 쌍 데이터셋 생성은 샘플당 전체 추론 패스가 필요합니다. 1-2단계 추론이 정말 필요할 때만 리플로우를 수행하세요.
- **분류기-자유 가이드(Classifier-free guidance)는 여전히 적용 가능.** 선형 결합에서 ε을 v로 교체하면 됩니다: `v_cfg = (1+w) v_cond - w v_uncond`.

## 사용 방법

| 사용 사례 | 2026 스택 |
|----------|-----------|
| 텍스트-이미지, 최고 품질 | 플로우 매칭: SD3, Flux.1-dev |
| 텍스트-이미지, 1-4단계 | 증류된 플로우 매칭: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 실시간 추론 | 플로우 매칭 기반 일관성 증류 (LCM, PCM) |
| 오디오 생성 | 플로우 매칭: Stable Audio 2.5, AudioCraft 2 |
| 비디오 생성 | 플로우 매칭과 확산 혼합 (Sora, Veo, Stable Video) |
| 과학 / 물리학 (입자 궤적, 분자) | 플로우 매칭 + 등변 벡터 필드 |

2025-2026년에 어떤 논문에서 "확산보다 빠르다"고 언급할 때, 거의 항상 플로우 매칭 + 증류를 의미합니다.

## Ship It

`outputs/skill-fm-tuner.md`를 저장하세요. Skill은 확산 스타일 모델 사양을 받아 흐름 매칭(flow-matching) 훈련 구성으로 변환합니다: 스케줄 선택, 시간 샘플링 분포(균일/로짓-정규), 옵티마이저, 리플로우(reflow) 계획, 목표 스텝 수, 평가 프로토콜.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하고 1-step vs 20-step MSE를 실제 데이터 분포와 비교해 보세요.
2. **중간.** 균일 `t` 샘플링에서 로짓-정규 분포(중간 t에 샘플링을 집중)로 전환해 보세요. 모델 품질이 향상되나요?
3. **어려움.** 재흐름(reflow) 반복 한 번 구현: 첫 번째 모델을 적분하여 (x_0, x_1) 쌍을 생성하고, 이 쌍으로 두 번째 모델을 훈련시킨 후 1-step 샘플 품질을 비교해 보세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| Flow matching | "직선 확산(Straight-line diffusion)" | 보간 경로를 따라 `x_1 - x_0`와 일치하도록 `v_θ(x, t)`를 학습. |
| Rectified flow | "리플로우(Reflow)" | 학습된 흐름을 직선화하는 반복적 절차. |
| Velocity field | "v_θ" | 모델 출력 — `x_t`를 이동시킬 방향. |
| Straight-line interpolant | "경로(The path)" | `x_t = (1-t)·x_0 + t·x_1`; 단순한 목표 미분값. |
| Euler sampler | "1차 ODE 솔버(1st order ODE solver)" | 가장 간단한 적분기; 경로가 직선일 때 효과적. |
| Logit-normal t | "SD3 샘플링" | 그래디언트가 가장 강한 중간값 쪽으로 `t` 샘플링을 집중. |
| Consistency distillation | "1-스텝 샘플러" | 모든 `x_t`를 직접 `x_0`로 매핑하도록 학생 모델을 학습. |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; 동일한 기법, 새로운 변수. |

## 프로덕션 노트: Flux.1-schnell은 가장 빠른 플로우 매칭 구현

플로우 매칭의 프로덕션 승자는 **Flux.1-schnell**입니다. 이는 1-4회의 추론 단계만으로 Flux-dev 수준의 품질을 유지하는 플로우 매칭 DiT(DiT distilled)입니다. Niels의 "8GB 머신에서 Flux 실행하기" 노트북은 참조 배포 레시피입니다: T5 + CLIP 인코딩, 양자화된 MMDiT 노이즈 제거(schnell은 4단계, dev는 50단계), VAE 디코딩. 비용 분석:

| 변형체 | 단계 | L4에서 1024² 지연 시간 | 총 FLOPs (상대적) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (원본) | 50 | ~15초 | 1.0× |
| Flux.1-schnell | 4 | ~1.2초 | 0.08× (12배 빠름) |
| SDXL-base | 30 | ~4초 | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3초 | 0.03× |

프로덕션 규칙: **플로우 매칭 기반 모델 + 증류 = 2026년 텍스트-이미지 생성의 기본 방식**. 모든 주요 벤더가 이 조합을 제공합니다: SD3-Turbo(SD3 + 플로우 + 증류), Flux-schnell(Flux-dev + 정류된 플로우 스트레이트닝), CogView-4-Flash. 순수 확산 기반 모델은 레거시 체크포인트용으로만 존재합니다.

## 추가 자료

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — 정류 흐름(rectified flow).
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — 흐름 정합(flow matching).
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, 대규모 정류 흐름(rectified flow at scale).
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — FM + 확산(diffusion)을 포괄하는 일반 프레임워크.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 확산(diffusion)/흐름(flow)의 1단계 증류(1-step distillation).
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — 터보 변형(turbo variant).
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 프로덕션에서의 흐름 정합(flow matching in production).