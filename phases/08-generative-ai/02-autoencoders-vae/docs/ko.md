# 오토인코더 & 변분 오토인코더(VAE)

> 일반 오토인코더는 압축한 후 재구성합니다. 이는 암기합니다. 생성하지 않습니다. 하나의 트릭을 추가하세요 — 코드를 가우시안 분포처럼 강제하면 — 샘플러가 됩니다. `z = μ + σ·ε`의 재파라미터화라는 단일 트릭이 2026년에 사용하는 모든 잠재 확산 및 흐름 매칭 이미지 모델의 입력에 VAE가 있는 이유입니다.

**유형:** 구축
**언어:** Python
**선수 지식:** 3단계 · 02 (역전파), 3단계 · 07 (CNNs), 8단계 · 01 (분류 체계)
**소요 시간:** ~75분

## 문제 정의

784픽셀의 MNIST 숫자 이미지를 16개의 숫자로 압축한 후 복원하는 과정에서, 일반 오토인코더는 복원 MSE(평균 제곱 오차)에서는 우수한 성능을 보이지만 코드 공간은 불규칙하고 엉망이다. 코드 공간에서 임의의 점을 선택해 디코딩하면 노이즈가 발생한다. 이는 샘플러가 없는 압축 모델에 불과하다.

실제로 원하는 것은 다음과 같다: (a) 코드 공간이 샘플링 가능한 깔끔하고 부드러운 분포(예: 등방성 가우시안 `N(0, I)`)를 가지며, (b) 어떤 샘플을 디코딩하더라도 그럴듯한 숫자 이미지를 생성하고, (c) 인코더와 디코더가 여전히 효과적으로 압축하는 것. 이 세 가지 목표를 하나의 아키텍처와 하나의 손실 함수로 달성해야 한다.

Kingma의 2013년 VAE(변분 오토인코더)는 인코더가 *분포* `q(z|x) = N(μ(x), σ(x)²)`를 출력하도록 훈련시키고, KL 페널티를 통해 이 분포를 사전 분포 `N(0, I)`에 근접시킨 후, `q(z|x)`에서 `z`를 샘플링해 디코딩한다. 추론 시에는 인코더를 제거하고 `z ~ N(0, I)`에서 샘플링한 후 디코딩한다. KL 페널티는 코드 공간이 구조화되도록 강제하는 역할을 한다.

2026년 현재 VAE는 단독으로 사용되기보다는 잠재 확산 모델(SD 1/2/XL/3, Flux, AudioCraft 등)의 인코더로 주로 활용된다. VAE를 학습하면 모든 이미지 파이프라인의 보이지 않는 첫 번째 레이어를 이해하는 것과 같다.

## 개념

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**오토인코더.** `z = encoder(x)`, `x̂ = decoder(z)`, 손실 = `||x - x̂||²`. 코드 공간이 구조화되지 않음.

**VAE 인코더.** 두 벡터를 출력: `μ(x)`와 `log σ²(x)`. 이들은 `q(z|x) = N(μ, diag(σ²))`를 정의함.

**리파라미터화 트릭.** `q(z|x)`에서 샘플링은 미분 불가능. 샘플을 `z = μ + σ·ε`로 재작성 (여기서 `ε ~ N(0, I)`). 이제 `z`는 `(μ, σ)`의 결정적 함수에 비파라미터 노이즈가 추가된 형태 — 그래디언트가 `μ`와 `σ`를 통해 흐름.

**손실.** 증거 하한(ELBO), 두 항으로 구성:

```
loss = 재구성 + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

재구성 항은 `x̂`를 `x`에 가깝게 유도. KL 항은 `q(z|x)`를 사전 분포로 유도. 두 항은 상충 관계. 작은 β(<1) = 더 선명한 샘플, 덜 가우시안인 코드 공간. 큰 β(>1) = 더 깨끗한 코드 공간, 흐릿한 샘플. β-VAE(Higgins 2017)는 이 조절 장치를 유명하게 만들었고, 분리 표현 연구를 촉발시킴.

**샘플링.** 추론 시: `z ~ N(0, I)`에서 샘플링 후 디코더를 통해 전달. 단일 순전파 — 확산과 같은 반복적 샘플링 불필요.

## 구축 방법

`code/main.py`는 numpy나 torch 없이 작은 VAE를 구현합니다. 입력은 8차원 공간에서 2-성분 가우시안 혼합 분포에서 추출된 합성 데이터입니다. 인코더와 디코더는 단일 은닉층 MLP입니다. tanh 활성화 함수, 순전파, 손실 함수, 그리고 수작업 역전파를 구현했습니다. 프로덕션용이 아닌 교육용입니다.

### 1단계: 인코더 순전파

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`σ` 대신 `log σ²`를 사용하여 네트워크 출력이 제한되지 않도록 합니다(σ에 softplus를 적용하는 것은 함정입니다 — σ ≈ 0에서 그래디언트가 소멸합니다).

### 2단계: 재매개변수화 및 디코딩

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 3단계: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

두 분포가 모두 가우시안이므로 정확한 닫힌 형태의 KL 발산을 사용합니다. 수치적분을 사용하지 마세요. 2026년에도 사람들은 이유 없이 몬테카를로 KL 추정기를 사용하는 코드를 출시합니다 — 이는 3배 느립니다.

### 4단계: 생성

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

이것이 생성 모델입니다. 단 5줄입니다.

## 함정

- **사후 붕괴(Posterior collapse).** KL 항이 `q(z|x) → N(0, I)`로 너무 공격적으로 수렴시켜 `z`가 `x`에 대한 정보를 전혀 담지 못하는 현상. 해결: β-어닐링(β=0에서 시작해 1로 점진적 증가), free bits, 또는 비활성 차원에 대한 KL 항 생략.
- **흐릿한 샘플(Blurry samples).** 가우시안 디코더 우도는 MSE 재구성을 의미하며, 이는 L2(mean)에 대해 베이지안 최적 — 여러 가능한 숫자 집합의 평균은 흐릿한 숫자. 해결: 이산 디코더(VQ-VAE, NVAE) 사용, 또는 VAE를 인코더로만 활용하고 잠재 공간에 확산 모델 적층(Stable Diffusion의 접근 방식).
- **β가 너무 크고 너무 이르게 증가.** 사후 붕괴 참조. β≈0.01에서 시작해 점진적으로 증가.
- **잠재 차원(latent dim) 너무 작음.** MNIST에는 16-D, ImageNet 256²에는 256-D, ImageNet 1024²에는 2048-D가 필요. Stable Diffusion의 VAE는 512×512×3 → 64×64×4로 압축(공간 영역에서 32x 다운샘플링, 채널에서 32x).

## 사용 방법

2026 VAE 스택:

| 상황 | 선택 |
|-----------|------|
| 디퓨전용 이미지-잠재 인코더 | Stable Diffusion VAE (`sd-vae-ft-ema`) 또는 Flux VAE |
| 오디오-잠재 인코더 | Encodec (Meta), SoundStream, 또는 DAC (Descript) |
| 비디오 잠재 공간 | Sora의 시공간 패치(spatiotemporal patches), Latte VAE, WAN VAE |
| 분리된 표현 학습(disentangled representation learning) | β-VAE, FactorVAE, TCVAE |
| 이산 잠재 공간(트랜스포머 모델링용) | VQ-VAE, RVQ (ResidualVQ) |
| 생성용 연속 잠재 공간 | 일반 VAE, 이후 해당 잠재 공간에서 플로우/디퓨전 모델 조건화 |

잠재-디퓨전 모델은 인코더와 디코더 사이에 디퓨전 모델이 위치한 VAE입니다. VAE는 거친 압축을 수행하고, 디퓨전 모델이 주요 작업을 담당합니다. 비디오(VAE + 비디오-디퓨전 DiT)와 오디오(Encodec + MusicGen 트랜스포머)에서도 동일한 패턴이 적용됩니다.

## Ship It

`outputs/skill-vae-trainer.md`를 저장하세요.

Skill은 다음 입력을 받습니다:  
- 데이터셋 프로파일  
- 잠재 차원(latent-dim) 목표값  
- 다운스트림 용도(재구성, 샘플링, 또는 잠재-확산 입력)  

그리고 다음 출력을 생성합니다:  
- 아키텍처 선택(plain/β/VQ/RVQ)  
- β 스케줄  
- 잠재 차원(latent dim)  
- 디코더 우도(Gaussian vs categorical)  
- 평가 계획(재구성 MSE, 차원당 KL, `q(z|x)`와 `N(0, I)` 간 Fréchet 거리)

## 연습 문제

1. **쉬움.** `code/main.py`에서 `β`를 `0.01`, `0.1`, `1.0`, `5.0`으로 변경해 보세요. 최종 재구성 MSE와 KL 값을 기록하세요. 합성 데이터에 대해 파레토 최적(Pareto-best)인 `β`는 무엇인가요?
2. **중간.** 가우시안 디코더 우도(Gaussian decoder likelihood)를 베르누이 우도(Bernoulli likelihood, 크로스-엔트로피 손실)로 교체하세요. 동일한 합성 데이터의 이진화된 버전에서 샘플 품질을 비교하세요.
3. **어려움.** `code/main.py`를 미니 VQ-VAE로 확장하세요: 연속 `z`를 K=32개의 엔트리를 가진 코드북(codebook)에서 최근접 이웃(nearest-neighbour) 조회로 교체하세요. 재구성 MSE를 비교하고 몇 개의 코드북 엔트리가 사용되는지 보고하세요 (코드북 붕괴(codebook collapse)는 실제로 발생합니다).

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 오토인코더(Autoencoder) | 인코더-디코더 네트워크 | `x → z → x̂`, MSE 학습. 생성 모델이 아님. |
| VAE(Variational Autoencoder) | 샘플러가 있는 AE | 인코더가 분포를 출력, KL 페널티가 잠재 공간을 형성. |
| ELBO(Evidence Lower Bound) | 증거 하한 | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`; `q = p(z|x)`일 때 tight. |
| 재매개변수화(Reparameterization) | `z = μ + σ·ε` | 확률적 노드를 결정론적 + 순수 노이즈로 재작성. 샘플링을 통한 역전파 가능. |
| 사전 분포(Prior) | `p(z)` | 잠재 변수의 목표 분포, 일반적으로 `N(0, I)`. |
| 사후 붕괴(Posterior collapse) | "KL 항이 이김" | 인코더가 `x`를 무시하고 사전 분포를 출력; 디코더가 허구를 생성해야 함. |
| β-VAE | 조절 가능한 KL 가중치 | `loss = recon + β·KL`. 높은 β = 더 분리된 표현이지만 흐릿함. |
| VQ-VAE(Vector Quantized VAE) | 이산 잠재 변수 | 연속 `z`를 가장 가까운 코드북 벡터로 대체; 트랜스포머 모델링 가능. |

## 프로덕션 노트: VAE는 디퓨전 서버에서 가장 핫한 경로

Stable Diffusion / Flux / SD3 파이프라인에서 VAE는 요청당 두 번 호출됩니다 — img2img 또는 인페인팅을 수행할 경우 인코딩 시 한 번, 디코딩 시 한 번 호출됩니다. 1024² 해상도에서 디코더 패스는 종종 전체 파이프라인에서 가장 큰 활성화 메모리 피크를 발생시킵니다. 이는 `128×128×16` 잠재 공간을 `1024×1024×3`로 업샘플링하기 때문입니다. 두 가지 실용적인 결과:

- **디코딩 슬라이스 또는 타일링.** `diffusers`는 `pipe.vae.enable_slicing()`과 `pipe.vae.enable_tiling()`을 제공합니다. 타일링은 작은 이음새 아티팩트를 희생하는 대신 `O(H·W)` 대신 `O(tile²)` 메모리를 사용합니다. 1024² 이상의 해상도에서 소비자 GPU에 필수적입니다.
- **bf16 디코더, 최종 리사이징을 위한 fp32 수치 연산.** SD 1.x VAE는 fp32로 출시되었으며 1024²+에서 fp16으로 캐스팅 시 *무증상으로 NaN을 생성*합니다. SDXL은 `madebyollin/sdxl-vae-fp16-fix`를 제공합니다 — 항상 fp16-픽스 변형을 선호하거나 bf16을 사용하세요.

## 추가 자료

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 논문.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — 분리형 β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — 최신 이미지 VAE.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAE를 인코더로 사용.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, 오디오 VAE 표준.