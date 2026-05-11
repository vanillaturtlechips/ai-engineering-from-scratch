# StyleGAN

> 대부분의 생성기는 `z`를 모든 레이어에 동시에 혼합합니다. StyleGAN은 이를 분리했습니다: 먼저 `z`를 중간 `w`로 매핑한 다음, AdaIN을 통해 모든 해상도 수준에서 `w`를 *주입*합니다. 이 단일 변경으로 잠재 공간이 분리되었고, 7년 연속 사실적인 얼굴 생성이 해결된 문제가 되었습니다.

**유형:** Build  
**언어:** Python  
**선수 지식:** Phase 8 · 03 (GANs), Phase 4 · 08 (정규화), Phase 3 · 07 (CNNs)  
**소요 시간:** ~45분

## 문제 정의

DCGAN은 `z`를 전치 합성곱(transposed convolution) 스택을 통해 이미지로 매핑합니다. 문제는 `z`가 모든 것(포즈, 조명, 신원, 배경)을 제어하며 이들이 서로 얽혀 있다는 점입니다. `z`의 한 축을 따라 이동하면 네 가지 모두가 변화합니다. "같은 사람, 다른 포즈"와 같은 요청을 모델에 할 수 없는 이유는 표현이 그러한 방식으로 분리되지 않기 때문입니다.

Karras et al. (2019, NVIDIA)는 다음과 같은 방법을 제안했습니다: `z`를 직접 합성곱 레이어에 입력하지 말고, 네트워크 입력으로 `4×4×512` 상수 텐서를 사용합니다. 8층 MLP를 학습시켜 `z ∈ Z → w ∈ W`로 매핑합니다. *적응형 인스턴스 정규화*(AdaIN)를 통해 모든 해상도에서 `w`를 주입합니다: 각 합성곱 특징 맵을 정규화한 후, `w`의 아핀 변환(affine projection)으로 스케일링 및 시프트를 적용합니다. 확률적 디테일(피부 모공, 머리카락 가닥)을 위해 레이어별 노이즈를 추가합니다.

결과: `W`는 "고수준 스타일"(포즈, 신원)과 "세부 스타일"(조명, 색상)에 대해 대략 직교하는 축을 가집니다. 두 이미지 간에 스타일을 교체할 수 있습니다. 예를 들어 저해상도 레벨에는 이미지 A의 `w`를, 고해상도 레벨에는 이미지 B의 `w`를 사용할 수 있습니다. 이를 통해 편집, 도메인 간 스타일 변환, 그리고 "StyleGAN-역변환" 연구 라인이 가능해졌습니다.

## 개념

![StyleGAN: 매핑 네트워크 + AdaIN + 레이어별 노이즈](../assets/stylegan.svg)

**매핑 네트워크.** `f: Z → W`, 8층 MLP. `Z = N(0, I)^512`. `W`는 강제로 가우시안이 되지 않음 — 데이터에 적응된 형태를 학습.

**합성 네트워크.** 학습된 상수 `4×4×512`에서 시작. 각 해상도 블록: `업샘플 → 합성곱 → AdaIN(w_i) → 노이즈 → 합성곱 → AdaIN(w_i) → 노이즈`. 해상도는 2배씩 증가: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

여기서 `y_scale`과 `y_bias`는 `w`의 아핀 변환에서 나옴. 특징 맵별로 정규화한 후 재스타일링. 여기서 "스타일"은 특징 맵의 1차 및 2차 통계량.

**레이어별 노이즈.** 각 특징 맵에 단일 채널 가우시안 노이즈를 추가, 학습된 채널별 계수로 스케일링. 전역 구조에는 영향을 주지 않으면서 확률적 디테일 제어.

**절단 기법(Truncation trick).** 추론 시, `z`를 샘플링하고 `w = mapping(z)`를 계산한 후, `w' = ŵ + ψ·(w - ŵ)`를 적용. 여기서 `ŵ`는 많은 샘플에 대한 `w`의 평균. `ψ < 1`은 다양성을 품질로 교환. 거의 모든 StyleGAN 데모는 `ψ ≈ 0.7`을 사용.

## StyleGAN 1 → 2 → 3

| 버전 | 연도 | 혁신 |
|---------|------|------------|
| StyleGAN | 2019 | 매핑 네트워크(mapping network) + AdaIN + 노이즈 + 점진적 성장(progressive growing). |
| StyleGAN2 | 2020 | 가중치 디모듈레이션(weight demodulation)이 AdaIN 대체 (물방울 아티팩트 수정); 스킵/잔차 아키텍처; 경로 길이 정규화(path-length regularization). |
| StyleGAN3 | 2021 | 에일리어싱 프리 컨볼루션(alias-free convolution) + 등변성 커널(equivariant kernels); 픽셀 그리드에 텍스처 고정 현상 제거. |
| StyleGAN-XL | 2022 | 클래스 조건부(class-conditional), 1024², ImageNet. |
| R3GAN | 2024 | 더 강력한 정규화로 재브랜딩; 20배 적은 파라미터로 FFHQ-1024에서 확산 모델과의 격차 해소. |

2026년 현재 StyleGAN3는 (a) 높은 FPS에서의 좁은 도메인 포토리얼리즘, (b) 소수 샷 도메인 적응(100개 이미지로 새 데이터셋 학습, 매핑 네트워크 고정), (c) 인버전 기반 편집(실제 사진을 재구성하는 `w` 찾기 → 해당 `w` 편집) 분야에서 여전히 기본 모델로 사용됩니다. 오픈 도메인 텍스트-이미지 생성에는 적합하지 않으며, 이 경우 확산 모델이 더 효과적입니다.

## 빌드하기

`code/main.py`는 1-D에서 "style-GAN lite" 토이를 구현합니다: 매핑 MLP, 학습된 상수 벡터를 `w`에서 파생된 스케일/편향으로 변조하는 합성 함수, 그리고 레이어별 노이즈. `w`를 아핀 변조(affine-modulation)로 주입하는 것이 생성기 입력에 `z`를 연결(concatenation)하는 것과 비슷하거나 더 나은 성능을 보인다는 것을 보여줍니다.

### 단계 1: 매핑 네트워크

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 단계 2: 적응형 인스턴스 정규화(adaptive instance normalization)

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

특징 맵별(feature-map) 스케일과 편향은 `w`의 선형 투영(linear projection)에서 나옵니다.

### 단계 3: 레이어별 노이즈

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

채널별(sigma per-channel) 표준 편차는 학습 가능합니다.

## 주의 사항

- **드롭렛 아티팩트(droplet artifacts).** StyleGAN 1은 AdaIN이 평균을 0으로 만들어 특징 맵에 덩어리진 드롭렛이 발생했습니다. StyleGAN 2의 가중치 디모듈레이션(weight demodulation)은 컨볼루션 가중치를 스케일링하여 이를 해결합니다.
- **텍스처 고정(texture sticking).** StyleGAN 1과 2의 텍스처는 객체 좌표가 아닌 픽셀 좌표를 따라 이동했습니다(보간 시 확인 가능). StyleGAN 3의 앨리어싱-프리 컨볼루션(alias-free convolutions)은 윈도우드 싱크 필터(windowed sinc filters)로 이 문제를 해결합니다.
- **모드 커버리지(mode coverage).** 트렁케이션(truncation) `ψ < 0.7`은 깨끗해 보이지만 좁은 콘(cone)에서 샘플링됩니다. 다양성이 필요하면 `ψ = 1.0`을 사용하세요.
- **인버전은 손실 발생.** 실제 사진을 `W`로 인버팅하는 작업은 일반적으로 최적화 또는 인코더(e4e, ReStyle, HyperStyle)를 통해 수행되며, 많은 반복 시 결과가 드리프트(drift)될 수 있습니다.

## 사용 방법

| 사용 사례 | 접근 방식 |
|----------|----------|
| 사실적인 인간 얼굴 (애니메이션, 제품, 좁은 범위) | StyleGAN3 FFHQ / 커스텀 파인튜닝 |
| 사진에서의 얼굴 편집 | e4e 인버전 + StyleSpace / InterFaceGAN 방향 |
| 얼굴 교체 / 재연 | StyleGAN + 인코더 + 블렌딩 |
| 아바타 파이프라인 | 저데이터 파인튜닝을 위한 ADA가 적용된 StyleGAN3 |
| 소수 이미지로부터 도메인 적응 | 매핑 네트워크 고정, 합성 네트워크 파인튜닝 |
| 멀티모달 또는 텍스트 조건부 생성 | 사용하지 않음 — 확산 모델 사용 |

"사람의 얼굴 사진"이 정답인 프로덕션급 데모의 경우, StyleGAN은 동일한 품질 기준에서 추론 비용(단일 순전파, 4090 기준 <10ms)과 선명도에서 확산 모델을 능가합니다.

## Ship It

`outputs/skill-stylegan-inversion.md`를 저장하세요. Skill은 실제 사진을 입력받아 다음을 출력합니다: 인버전 방법(e4e / ReStyle / HyperStyle), 예상 잠재 손실(latent loss), 편집 예산(`W`에서 아티팩트 발생 전까지 이동 가능한 거리), 그리고 알려진 좋은 편집 방향 목록(나이, 표정, 포즈).

## 연습 문제

1. **쉬움.** `adain_on=True`와 `adain_on=False`로 `code/main.py`를 실행해 보세요. 고정된 잠재 변수(fixed latent) 대 교란된 잠재 변수(perturbed latent)에 대한 출력의 분포를 비교하세요.
2. **중간.** 혼합 정규화(mixing regularization) 구현: 훈련 배치에 대해 `w_a`, `w_b`를 계산하고, 합성의 첫 번째 절반에는 `w_a`를, 두 번째 절반에는 `w_b`를 적용하세요. 디코더가 분리된 스타일(disentangled styles)을 학습하는지 확인해 보세요.
3. **어려움.** 사전 훈련된 StyleGAN3 FFHQ 모델(ffhq-1024.pkl)을 사용하세요. 라벨이 지정된 샘플에 대해 SVM을 훈련시켜 "미소(smile)"를 제어하는 `w` 방향을 찾고, 신원(identity)이 흐트러지기 전까지 얼마나 멀리까지 밀어낼 수 있는지 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 매핑 네트워크 | "The MLP" | `f: Z → W`, 8층, 잠재 기하학과 데이터 통계를 분리합니다. |
| W 공간 | "The style space" | 매핑 네트워크의 출력; 대략 분리(disentangled)되어 있습니다. |
| AdaIN | "Adaptive instance norm" | 특성 맵을 정규화한 후 `w`-투영으로 스케일 + 시프트합니다. |
| 절단 기법 | "Psi" | `w = 평균 + ψ·(w - 평균)`, ψ<1은 다양성을 품질로 교환합니다. |
| 경로 길이 정규화 | "PL reg" | `w` 단위 변화당 이미지 큰 변화를 페널티; `W`를 더 부드럽게 만듭니다. |
| 가중치 디모듈레이션 | "The StyleGAN2 fix" | 활성화 대신 컨볼루션 가중치를 정규화; 물방울 아티팩트를 제거합니다. |
| 앨리어싱 제거 | "StyleGAN3의 기법" | 윈도우드 싱크 필터; 픽셀 그리드에 달라붙는 텍스처를 제거합니다. |
| 인버전 | "실제 이미지에 대한 w 찾기" | `x → w`를 최적화하거나 인코딩하여 `G(w) ≈ x`를 만듭니다. |

## 2026년에도 StyleGAN이 여전히 사용되는 이유: 프로덕션 관점

StyleGAN3는 4090에서 1024² 해상도의 FFHQ 얼굴을 10ms 미만에 생성합니다 — `num_steps = 1`, VAE 디코딩 없음, 크로스 어텐션 패스 없음. 프로덕션 측면에서 이는 모든 이미지 생성기의 최저 지연 시간입니다. 동일한 해상도에서 50단계 SDXL + VAE 디코딩 파이프라인은 약 3초가 소요됩니다. 이는 **300배 차이**이며, 좁은 도메인 제품(아바타 서비스, 신분증 문서 파이프라인, 스톡 얼굴 생성)에서는 TCO(총 소유 비용) 측면에서 승리합니다.

두 가지 운영적 결과:

- **스케줄러 없음, 배치 처리기 없음.** 목표 점유율에서의 정적 배치가 최적입니다. 연속 배치 처리(LLM 및 디퓨전에 필수적)는 모든 요청이 동일한 FLOPs를 소모하기 때문에 아무런 이점이 없습니다.
- **절삭(truncation) `ψ`가 안전 조절 장치입니다.** `ψ < 0.7`은 매핑 네트워크 범위의 좁은 콘에서 샘플링합니다. 이는 서빙 레이어가 샘플 분산에 대해 가진 유일한 조절 장치입니다. 피크 부하 시 `ψ`를 낮추고, 프리미엄 사용자에게는 높입니다.

## 추가 자료

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN(스타일GAN).
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2(스타일GAN2).
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3(스타일GAN3).
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e 인버전(e4e inversion).
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL(스타일GAN-XL).
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — 현대식 최소 GAN 레시피(modern minimal GAN recipe).