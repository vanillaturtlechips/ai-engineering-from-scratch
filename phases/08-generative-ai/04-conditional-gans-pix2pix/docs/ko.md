# 조건부 GAN(Conditional GANs) & Pix2Pix

> 2014-2017년의 첫 번째 큰 발전은 GAN이 생성하는 것을 제어하는 것이었습니다. 레이블, 이미지 또는 문장을 첨부할 수 있습니다. Pix2Pix는 이미지 버전을 구현했으며, 여전히 좁은 이미지-이미지 작업에서는 모든 일반적인 텍스트-이미지 모델을 능가합니다.

**유형:** 구현  
**언어:** Python  
**선수 지식:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)  
**소요 시간:** ~75분

## 문제 정의

비조건부 GAN은 임의의 얼굴을 샘플링한다. 데모에는 유용하지만 실제 적용에는 쓸모가 없다. 당신이 원하는 것은 *스케치를 사진으로 매핑*, *지도를 항공 사진으로 매핑*, *주간 장면을 야간으로 매핑*, *그레이스케일 이미지에 컬러화*이다. 이 모든 경우에서 입력 이미지 `x`가 주어지고, 어떤 의미적 대응 관계를 갖는 `y`를 출력해야 한다. 하나의 `x`에 대해 많은 타당한 `y`가 존재할 수 있다. 평균 제곱 오차는 이들을 뭉개버려서 흐릿하게 만든다. 반면 적대적 손실은 그렇지 않다. 왜냐하면 "실제처럼 보인다"는 기준이 날카롭기 때문이다.

조건부 GAN(Mirza & Osindero, 2014)은 조건 `c`를 생성기 `G`와 판별기 `D` 모두의 입력으로 추가한다. Pix2Pix(Isola et al., 2017)는 이를 특수화했다: 조건은 전체 입력 이미지이며, 생성기는 U-Net, 판별기는 *패치 기반* 분류기(PatchGAN)이며, 손실 함수는 적대적 손실 + L1이다. 이 레시피는 2026년에도 여전히 좁은 이미지-이미지 영역에서 처음부터 학습하는 텍스트-이미지 모델보다 우수한 성능을 보인다. 왜냐하면 *쌍으로 된 데이터*로 학습되기 때문이다 — 필요한 신호를 정확히 가지고 있다.

## 개념

![Pix2Pix: U-Net 생성기, PatchGAN 판별기](../assets/pix2pix.svg)

**조건부 생성기(Conditional G).** `G(x, z) → y`. Pix2Pix에서 `z`는 G 내부의 드롭아웃입니다(명시적 노이즈 입력 없음 — Isola는 명시적 노이즈가 무시됨을 발견).

**조건부 판별기(Conditional D).** `D(x, y) → [0, 1]`. 입력은 *쌍*(조건, 출력)입니다. 이것이 핵심 차이점: D는 `y`가 실제처럼 보이는지뿐만 아니라 `y`가 `x`와 일관되는지 판단해야 합니다.

**U-Net 생성기.** 병목(bottleneck)을 가로지르는 스킵 연결(skip connections)이 있는 인코더-디코더 구조. 입력과 출력이 저수준 구조(경계, 실루엣)를 공유하는 작업에 중요합니다. 스킵 연결이 없으면 고주파 세부 정보가 사라집니다.

**PatchGAN 판별기.** 단일 real/fake 점수를 출력하는 대신, D는 각 셀이 ~70×70 픽셀의 수용 영역(receptive field)을 판단하는 `N×N` 그리드를 출력합니다. 평균화됩니다. 이는 마르코프 랜덤 필드 가정: 현실성은 지역적입니다. 훈련 속도가 훨씬 빠르고, 매개변수가 적으며, 출력이 더 선명합니다.

**손실 함수(Loss).**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 항은 훈련을 안정화하고 G를 알려진 타겟으로 밀어줍니다. L1은 L2(평균이 아닌 중앙값)보다 더 선명한 경계를 제공합니다. `λ = 100`이 Pix2Pix의 기본값이었습니다.

## CycleGAN — 페어 데이터가 없을 때

Pix2Pix는 페어된 `(x, y)` 데이터가 필요합니다. CycleGAN(Zhu et al., 2017)은 이 요구사항을 제거하지만, 대신 *사이클 일관성(cycle consistency)* 손실이라는 추가 손실을 도입합니다. 두 생성기 `G: X → Y`와 `F: Y → X`를 사용합니다. `F(G(x)) ≈ x`와 `G(F(y)) ≈ y`가 되도록 훈련시켜, 페어된 예제 없이도 말을 얼룩말로, 여름을 겨울로 변환할 수 있습니다.

2026년에는 페어되지 않은 이미지-투-이미지 변환이 대부분 CycleGAN이 아닌 디퓨전(ControlNet, IP-Adapter)을 통해 수행되지만, 사이클 일관성 아이디어는 거의 모든 페어되지 않은 도메인 적응 논문에서 여전히 살아남아 있습니다.

## 구축 방법

`code/main.py`는 1-D 데이터에 대한 소형 조건부 GAN을 구현합니다. 조건 `c`는 클래스 레이블(0 또는 1)입니다. 작업: 주어진 클래스에 대한 조건부 분포에서 샘플을 생성합니다.

### 1단계: 생성자(G)와 판별자(D) 입력에 조건 추가

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

원-핫 인코딩은 가장 간단한 방법입니다. 더 큰 모델들은 학습된 임베딩(embedding), FiLM 변조, 또는 교차 어텐션(cross-attention)을 사용합니다.

### 2단계: 조건부 학습

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

생성자는 주변 분포(marginal)가 아닌 *주어진 조건에 대한 실제 분포*를 일치시켜야 합니다.

### 3단계: 클래스별 출력 검증

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 함정(Pitfalls)

- **조건 무시(Condition ignored).** G는 주변화(marginalize)하는 법을 학습하지만, 조건 신호가 약해 D이 절대 페널티를 주지 않음. 해결: D을 더 공격적으로 조건화(early layer에서, late layer뿐만 아니라), 투영 판별기(projection discriminator) 사용 (Miyato & Koyama 2018).
- **L1 가중치 너무 낮음(L1 weight too low).** G가 임의의 실제 같은 출력으로 표류(drift)하며, 충실도(faithful) 없는 결과 생성. Pix2Pix-style 작업에서는 λ≈100으로 시작.
- **L1 가중치 너무 높음(L1 weight too high).** L1이 여전히 L_p 노름이므로 G가 흐릿한(blurry) 출력 생성. 학습이 안정화되면 가중치를 점차 감소(anneal down).
- **D에서의 실제값 누출(Ground-truth leakage in D).** D 입력으로 `y`가 아닌 `(x, y)`를 연결(concatenate). 이 방법이 아니면 D이 일관성(check consistency)을 검증할 수 없음.
- **클래스별 모드 붕괴(Mode collapse per class).** 각 클래스가 독립적으로 붕괴 가능. 클래스 조건부 다양성(class-conditional diversity) 검증 수행.

## 사용 방법

2026년 이미지-투-이미지 작업 현황:

| 작업 | 최적 접근법 |
|------|-------------|
| 스케치 → 사진, 동일 도메인, 페어링 데이터 | Pix2Pix / Pix2PixHD (여전히 빠르고 선명함) |
| 스케치 → 사진, 언페어링 데이터 | ControlNet with a Scribble conditioning model |
| 시맨틱 분할 → 사진 | SPADE / GauGAN2 또는 SD + ControlNet-Seg |
| 스타일 변환 | IP-Adapter 또는 LoRA를 사용한 Diffusion; GAN 방법은 레거시 |
| 깊이 → 사진 | Stable Diffusion 기반 ControlNet-Depth |
| 초해상도 | Real-ESRGAN (GAN), ESRGAN-Plus, 또는 SD-Upscale (diffusion) |
| 컬러화 | ColTran, diffusion 기반 컬러화 도구, 또는 Pix2Pix-color |
| 주간 → 야간, 계절, 날씨 | CycleGAN 또는 ControlNet 기반 |

Pix2Pix는 (a) 수천 개의 페어링 예제가 있을 때, (b) 작업이 좁고 반복 가능할 때, (c) 빠른 추론이 필요할 때 여전히 적합한 도구입니다. 일반적인 오픈 도메인 작업에서는 diffusion이 우세합니다.

## Ship It

`outputs/skill-img2img-chooser.md`를 저장하세요. 이 스킬은 작업 설명, 데이터 가용성(paired vs unpaired, N 샘플), 지연 시간/품질 예산을 입력받아 다음을 출력합니다: 접근 방식(Pix2Pix, CycleGAN, ControlNet 변형, SDXL + IP-Adapter), 훈련 데이터 요구사항, 추론 비용, 평가 프로토콜(LPIPS, FID, 작업별 평가).

## 연습 문제

1. **쉬움.** `code/main.py`를 수정하여 세 번째 클래스를 추가하세요. G가 여전히 각 클래스의 노이즈를 올바른 모드로 매핑하는지 확인하세요.
2. **중간.** 1-D 설정에서 L1을 지각(perceptual) 스타일 손실 함수로 대체하세요 (예: 작은 동결된 D를 특징 추출기로 사용). 조건부 분포의 예리함(sharpness)이 변하는지 확인하세요.
3. **어려움.** 1-D 설정에서 CycleGAN을 스케치하세요: 두 분포, 두 생성기, 사이클 손실. 페어링된 데이터 없이도 이들 사이를 매핑하는 방법을 학습하는지 보여주세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 조건부 GAN(Conditional GAN) | "레이블이 있는 GAN" | G(z, c), D(x, c). 두 네트워크 모두 조건(condition)을 입력받음. |
| Pix2Pix | "이미지-투-이미지 GAN" | U-Net 생성기(G)와 PatchGAN 판별기(D) + L1 손실 함수를 사용하는 페어드(paired) cGAN. |
| U-Net | "스킵 연결(skips)이 있는 인코더-디코더" | 대칭적인 합성곱 네트워크; 스킵 연결은 고주파 정보(high-freq)를 보존함. |
| PatchGAN | "지역적 현실성(local-realism) 분류기" | 판별기(D)가 전역(global) 점수 대신 패치별 점수를 출력함. |
| CycleGAN | "언페어드(unpaired) 이미지 변환" | 두 개의 생성기(G) + 사이클 일관성(cycle-consistency) 손실 함수; 페어드 데이터 불필요. |
| SPADE | "GauGAN" | 시맨틱 맵(semantic map)으로 중간 활성화(intermediate activations)를 정규화; 세그멘테이션-투-이미지(segmentation-to-image). |
| FiLM | "특성별 선형 변조(Feature-wise linear modulation)" | 조건(condition)으로부터 특성별(per-feature) 아핀 변환(affine transform) 적용; 저비용 조건화(cheap conditioning). |

## 프로덕션 노트: 지연 시간 기준인 Pix2Pix

페어링된 데이터와 좁은 작업(스케치 → 렌더링, 의미론적 지도 → 사진, 낮 → 밤)이 있을 때, Pix2Pix의 원샷 추론은 지연 시간 측면에서 확산 모델보다 10배 더 빠릅니다. 일반적인 프로덕션 비교는 다음과 같습니다:

| 경로 | 단계 | 단일 L4에서 512² 기준 일반 지연 시간 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net 순전파) | 1 | ~30 ms |
| SD-Inpaint 또는 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL 기본 모델 | 20-30 | ~3-5 s |

Pix2Pix는 정적 배치(모든 요청이 동일한 FLOPs)에서 처리량 면에서 우세합니다. 확산 모델은 품질과 일반화 능력에서 우수합니다. 현대적인 접근 방식은 좁은 작업을 위한 Pix2Pix 스타일의 증류 모델과 꼬리 입력(tail inputs)을 위한 확산 모델 폴백을 함께 제공하는 것입니다.

## 추가 자료

- [Mirza & Osindero (2014). 조건부 생성적 적대 네트워크(Conditional Generative Adversarial Nets)](https://arxiv.org/abs/1411.1784) — cGAN 논문.
- [Isola et al. (2017). 조건부 적대 네트워크를 이용한 이미지-이미지 변환(Image-to-Image Translation with Conditional Adversarial Networks)](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). 순환 일관성 적대 네트워크를 이용한 비쌍대 이미지-이미지 변환(Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks)](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). 조건부 GAN을 이용한 고해상도 이미지 합성(High-Resolution Image Synthesis with Conditional GANs)](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). 공간 적응형 정규화를 이용한 의미론적 이미지 합성(Semantic Image Synthesis with Spatially-Adaptive Normalization)](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). 투영 판별자를 갖는 cGANs(cGANs with Projection Discriminator)](https://arxiv.org/abs/1802.05637) — 투영 판별자(D).