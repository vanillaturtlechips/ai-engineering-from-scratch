# GANs — 생성기(Generator) vs 판별기(Discriminator)

> 2014년 Goodfellow의 트릭은 밀도 추정을 완전히 건너뛰는 것이었습니다. 두 네트워크가 있습니다. 하나는 가짜를 만들고, 하나는 이를 탐지합니다. 가짜가 실제와 구별 불가능할 때까지 경쟁합니다. 이 방법은 작동해서는 안 됩니다. 실제로도 자주 실패합니다. 하지만 성공할 경우, 좁은 도메인에서 문헌상 가장 선명한 샘플을 생성합니다.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** Phase 3 · 02 (역전파), Phase 3 · 08 (옵티마이저), Phase 8 · 02 (VAE)
**소요 시간:** ~75분

## 문제

VAE는 MSE 디코더 손실이 *평균* 이미지에 대해 베이즈-최적(Bayes-optimal)이기 때문에 흐릿한 샘플을 생성합니다. 여러 타당한 숫자의 평균은 흐릿한 숫자가 됩니다. 픽셀 단위의 근접성이 아닌 *타당성*을 보상하는 손실 함수가 필요합니다. 타당성에 대한 닫힌 형태(closed-form)는 존재하지 않습니다. 이를 학습해야 합니다.

Goodfellow의 아이디어: 실제 이미지와 가짜 이미지를 구분하는 분류기 `D(x)`를 훈련시킵니다. 생성기 `G(z)`는 `D`를 속이도록 훈련됩니다. `G`에 대한 손실 신호는 `D`가 현재 실제처럼 보이는 것으로 판단하는 모든 것입니다. 이 신호는 `G`가 개선됨에 따라 업데이트되며, 움직이는 표적을 쫓습니다. 두 네트워크가 수렴하면 `G`는 `log p(x)`를 명시적으로 정의하지 않고도 데이터 분포를 학습한 것입니다.

이것이 적대적 훈련(adversarial training)입니다. 수식은 미니맥스 게임(minimax game)으로 표현됩니다:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026년 현재 GAN은 더 이상 최첨단(SOTA) 생성 모델이 아닙니다(확산 모델과 흐름 매칭(flow matching)이 그 자리를 차지함). 그러나 StyleGAN 2/3는 여전히 가장 선명한 얼굴 생성 모델로 남아 있으며, GAN 판별자는 확산 모델 훈련에서 *지각 손실(perceptual loss)*로 사용되고, 적대적 훈련은 실시간 확산 모델을 가능하게 하는 빠른 1단계 증류(SDXL-Turbo, SD3-Turbo, LCM)를 구동합니다.

## 개념

![GAN 훈련: 생성자와 판별자의 미니맥스](../assets/gan.svg)

**생성자 `G(z)`.** 노이즈 벡터 `z ~ N(0, I)`를 샘플 `x̂`로 매핑합니다. 디코더 형태의 네트워크(밀집 또는 전치 합성곱)입니다.

**판별자 `D(x)`.** 샘플을 스칼라 확률(또는 점수)로 매핑합니다. 실제 데이터 → 1, 가짜 데이터 → 0.

**손실 함수.** 두 가지 교차 업데이트:

- **판별자 `D` 훈련:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. 실제 데이터=1, 가짜 데이터=0에 대한 이진 교차 엔트로피입니다.
- **생성자 `G` 훈련:** `loss_G = -log D(G(z))`. 이는 Goodfellow가 사용한 *비포화(non-saturating)* 형태입니다 (원래 `log(1 - D(G(z)))`는 `D`가 확신하면 포화되어 그래디언트가 사라집니다).

**훈련 루프.** `D` 1단계, `G` 1단계. 반복.

**작동 원리.** `G`가 `p_data`와 완벽히 일치하면 `D`는 확률 0.5 이상의 성능을 낼 수 없으며 모든 곳에서 0.5를 출력합니다. 이때 `G`는 더 이상 그래디언트를 받지 않습니다. 균형 상태.

**실패 원인.** 모드 붕괴(`G`가 `D`가 분류하지 못하는 단일 모드를 찾아 계속 생성), 그래디언트 소실(`D`가 너무 빨리 학습하여 `log D`가 포화됨), 훈련 불안정성(학습률, 배치 크기 등 모든 요소).

## GAN을 작동하게 만든 변형들

| 연도 | 혁신 | 해결 |
|------|------------|-----|
| 2015 | DCGAN | 합성곱/역합성곱, 배치 정규화, LeakyReLU — 최초의 안정적인 아키텍처. |
| 2017 | WGAN, WGAN-GP | BCE를 Wasserstein 거리 + 그래디언트 패널티로 대체. 사라지는 그래디언트 문제 해결. |
| 2017 | 스펙트럴 정규화(spectral normalization) | 판별자(discriminator)의 Lipschitz 경계 설정. 2026년 판별자에서도 여전히 사용됨. |
| 2018 | 프로그레시브 GAN(Progressive GAN) | 저해상도부터 훈련 시작, 레이어 추가. 최초의 메가픽셀 결과. |
| 2019 | StyleGAN / StyleGAN2 | 매핑 네트워크(mapping network) + 적응형 인스턴스 정규화(adaptive instance norm). 고정 도메인 사진 사실주의(state of the art) 기준. |
| 2021 | StyleGAN3 | 에일리어싱 제거(alias-free), 이동 등가성(translation-equivariant) — 2026년 얼굴 생성 골드 스탠더드. |
| 2022 | StyleGAN-XL | 조건부(conditional), 클래스 인식(class-aware), 대규모 확장. |
| 2024 | R3GAN | 더 강력한 정규화(regularization)로 재브랜딩; 트릭 없이 1024² 해상도에서 작동. |

## 구축

`code/main.py`는 1-D 데이터(2개의 가우시안 혼합)에 대해 작은 GAN을 훈련합니다. 생성자(Generator)와 판별자(Discriminator)는 단일 은닉층 MLP입니다. 순전파(forward), 역전파(backward), 미니맥스 루프(minimax loop)를 직접 구현합니다. 목표는 두 가지 주요 실패 모드(모드 붕괴(mode collapse) + 소실 기울기(vanishing gradient))가 발생하는 것을 관찰하는 것입니다.

### 단계 1: 비포화 손실(non-saturating loss)

기존의 Goodfellow 손실 `log(1 - D(G(z)))`는 판별자(D)가 생성자(G)의 가짜 샘플을 높은 확신으로 가짜로 분류할 때 0에 수렴합니다. 이 시점에서 G의 기울기는 기본적으로 0이 되어 G는 개선될 수 없습니다. 비포화 형태 `-log D(G(z))`는 반대 극점을 가집니다: 판별자가 확신하면 손실이 발산하여 G에 강한 신호를 제공합니다.

```python
def g_loss(d_fake):
    # log D(G(z)) 최대화 <=> -log D(G(z)) 최소화
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 단계 2: 생성자 단계당 하나의 판별자 단계

```python
for step in range(steps):
    # D 훈련
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # G 훈련
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # 새로운 가짜 샘플
    update_G(fake_batch)
```

G에 대한 새로운 가짜 샘플을 제공하지 않으면 기울기가 오래되어(stale) 성능이 저하됩니다.

### 단계 3: 모드 붕괴(mode collapse) 감시

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] 모드 붕괴: 한 모드가 소멸됨")
```

전형적인 증상: 두 실제 모드 중 하나가 생성되지 않습니다. 판별자는 가짜로 인식하지 못하므로 이를 수정하지 않습니다.

## 함정(Pitfalls)

- **판별자(Discriminator)가 너무 강함.** D의 학습률(learning rate)을 2-5배 줄이거나, 인스턴스/레이어 노이즈(instance/layer noise)를 추가. D가 95% 이상의 정확도에 도달하면 생성자(G)는 학습 불가능해짐.
- **생성자(Generator)가 특정 모드(mode)를 암기함.** D 입력에 노이즈 추가, 미니배치-판별자(minibatch-discriminator) 레이어 사용, 또는 WGAN-GP로 전환.
- **배치 정규화(Batch norm) 통계 누출.** 실제 배치(real batch)와 가짜 배치(fake batch)가 동일한 BN 레이어를 통과하면 통계가 혼합됨. 대신 인스턴스 정규화(instance norm) 또는 스펙트럴 정규화(spectral norm) 사용.
- **Inception-score 조작.** FID(Fréchet Inception Distance)와 IS(Inception Score)는 샘플 수가 적을 때 노이즈가 큼. 평가 시 ≥10k 샘플 사용.
- **조건부 작업(conditional tasks)에서 원샷 샘플링(one-shot sampling)은 거짓말.** 여전히 CFG 스케일(CFG scales), 절단(truncation) 기법, 재샘플링(re-sampling)이 필요하여 사용 가능한 출력을 얻을 수 있음.

## 사용 방법

2026 GAN 스택:

| 상황 | 선택 |
|-----------|------|
| 사실적인 인간 얼굴, 고정된 포즈 | StyleGAN3 (가장 선명하고 작음) |
| 애니메이션 / 스타일화된 얼굴 | StyleGAN-XL 또는 Stable Diffusion LoRA |
| 이미지-이미지 변환 | Pix2Pix / CycleGAN (Phase 8 · 04) 또는 ControlNet (Phase 8 · 08) |
| 빠른 1단계 텍스트-이미지 생성 | 확산 모델의 적대적 증류 (SDXL-Turbo, SD3-Turbo) |
| 확산 모델 트레이너 내 지각 손실 | 이미지 크롭에 대한 소형 GAN 판별기 |
| 다중 모달, 개방형 작업 | 사용하지 마세요 — 확산 모델 또는 플로우 매칭 사용 |

GAN은 선명하지만 적용 범위가 좁습니다. 도메인이 확장되면 — 사진, 임의의 텍스트 프롬프트, 비디오 — 확산 모델로 전환하세요. 적대적 트릭은 독립형 생성기가 아닌 구성 요소(지각 손실, 증류)로 계속 사용됩니다.

## Ship It

`outputs/skill-gan-debugger.md`를 저장하세요. Skill은 실패한 GAN 실행(손실 곡선, 샘플 그리드, 데이터셋 크기)을 입력으로 받아, 가능한 원인 목록(순위별), 한 줄짜리 해결 방법, 재실행 프로토콜을 출력합니다.

## 연습 문제

1. **쉬움.** 기본 설정으로 `code/main.py`를 실행합니다. 그런 다음 `D_LR = 5 * G_LR`로 설정하고 다시 실행합니다. G의 손실(loss)이 상수로 수렴하는 속도는 얼마나 빠른가요?  
2. **중간.** Goodfellow BCE 손실 함수를 WGAN 손실 함수로 교체합니다: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, 그리고 D(Discriminator)의 가중치를 `[-0.01, 0.01]` 범위로 클리핑합니다. 학습이 더 안정적인가요? 벽시계(wall-clock) 기준 수렴 속도를 비교하세요.  
3. **어려움.** 1-D 예제를 2-D 데이터(원형 배열의 8개 가우시안 혼합)로 확장합니다. 1k, 5k, 10k 단계에서 생성자(generator)가 8개의 모드(mode) 중 몇 개를 포착하는지 추적합니다. 미니배치 판별(minibatch discrimination)을 구현하고 다시 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 생성기(Generator) | "G" | 노이즈-샘플 네트워크, `G: z → x̂`. |
| 판별기(Discriminator) | "D" | 분류기 `D: x → [0, 1]`, 실제 vs 가짜. |
| 미니맥스(Minimax) | "The game" | `min_G max_D`의 공동 목적 함수. |
| 비포화 손실(Non-saturating loss) | "The fix" | G에 대해 `log(1 - D(G(z)))` 대신 `-log D(G(z))` 사용. |
| 모드 붕괴(Mode collapse) | "G가 한 가지만 외웠어" | 생성기가 다양한 데이터에도 불구하고 몇 가지 유사한 출력만 생성. |
| WGAN(Wasserstein GAN) | "Wasserstein" | BCE를 Earth-Mover 거리 + 그래디언트 페널티로 대체; 더 부드러운 그래디언트. |
| 스펙트럴 노름(Spectral norm) | "Lipschitz 트릭" | 판별기(D)의 가중치 노름 제약으로 기울기 제한; 학습 안정화. |
| 스타일GAN(StyleGAN) | "잘 작동하는 것" | 매핑 네트워크 + AdaIN; 얼굴 생성 분야 최고 수준(2026년 기준). |

## 프로덕션 노트: 원샷 추론은 GAN의 지속적인 강점

GAN은 더 이상 오픈 도메인 생성에서 샘플 품질 면에서 우위를 점하지 않지만, 추론 비용 측면에서는 여전히 우위를 유지합니다. 프로덕션 추론 문헌에서 GAN은 다음과 같은 특징을 가집니다:

- **프리필(prefill) 및 디코딩 단계 없음.** 단일 `G(z)` 순전파(forward pass)로 구성됩니다. TTFT(Thinking Time From First Token) ≈ 총 지연 시간.
- **KV-캐시 압력 없음.** 유일한 상태는 가중치입니다. 배치 크기는 캐시(cahce)가 아닌 활성화 메모리에 의해 제한됩니다.
- **간단한 연속 배치 처리.** 모든 요청이 동일한 고정 FLOPs를 소모하므로, 서버의 목표 점유율(target occupancy)에서 정적 배치(static batch)가 일반적으로 최적입니다. 인플라이트 스케줄러(in-flight scheduler)가 필요하지 않습니다.

이것이 2026년 빠른 텍스트-이미지 생성을 위한 주요 기술인 GAN 증류(SDXL-Turbo, SD3-Turbo, ADD, LCM)가 지배적인 이유입니다: 20-50단계 디퓨전 파이프라인을 1-4회의 GAN 스타일 순전파로 압축하면서도 디퓨전 기반의 분포를 유지합니다. 적대적 손실(adversarial loss)은 느린 생성기를 빠른 생성기로 변환하는 훈련 시간 조절 장치(knob)로 남아 있습니다.

## 추가 자료

- [Goodfellow et al. (2014). 생성적 적대 신경망(Generative Adversarial Nets)](https://arxiv.org/abs/1406.2661) — 원본 GAN 논문.
- [Radford et al. (2015). DCGAN을 이용한 비지도 표현 학습(Unsupervised Representation Learning with DCGAN)](https://arxiv.org/abs/1511.06434) — 최초의 안정적인 아키텍처.
- [Arjovsky, Chintala, Bottou (2017). 워셔스테인 GAN(Wasserstein GAN)](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). GAN을 위한 스펙트럼 정규화(Spectral Normalization for GANs)](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). StyleGAN의 이미지 품질 분석 및 개선(Analyzing and Improving the Image Quality of StyleGAN)](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). 에일리어스 프리 생성적 적대 신경망(Alias-Free Generative Adversarial Networks)](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). 적대적 확산 증류(Adversarial Diffusion Distillation)](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.