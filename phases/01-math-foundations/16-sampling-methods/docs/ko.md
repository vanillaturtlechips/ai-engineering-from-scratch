# 샘플링 방법

> 샘플링은 AI가 가능성 공간을 탐색하는 방식입니다.

**유형:** 구축(Build)  
**언어:** Python  
**선수 지식:** 1단계, 레슨 06-07 (확률, 베이즈 정리)  
**소요 시간:** ~120분

## 학습 목표

- 균일 난수만을 사용하여 역 CDF, 거부, 중요도 샘플링을 처음부터 구현
- 언어 모델 토큰 생성을 위한 온도, 상위-k, 상위-p(핵) 샘플링 구축
- VAE에서 샘플링을 통한 역전파를 가능하게 하는 재매개변수화 트릭과 그 이유 설명
- 비정규화된 목표 분포에서 샘플링하기 위한 메트로폴리스-헤이스팅스 MCMC 실행

## 문제 정의

언어 모델은 사용자의 프롬프트를 처리한 후 50,000개의 로짓(logit)으로 구성된 벡터를 생성합니다. 이는 모델의 어휘집에 있는 모든 토큰에 대한 점수입니다. 이제 모델은 이 중 하나를 선택해야 합니다. 어떻게 선택할까요?

항상 가장 높은 확률의 토큰을 선택하면 모든 응답이 동일해집니다. 결정론적이고 지루합니다. 반면 균일하게 무작위 선택하면 출력은 무의미한 문자열이 됩니다. 해결책은 이 두 극단 사이의 어딘가에 있으며, 그 지점을 제어하는 것이 바로 샘플링(sampling)입니다.

샘플링은 텍스트 생성에만 국한되지 않습니다. 강화 학습(reinforcement learning)은 궤적(trajectory)을 샘플링하여 정책 경사(policy gradient)를 추정합니다. VAE(Variational Autoencoder)는 학습된 분포에서 샘플링하고 무작위성을 역전파(backpropagation)하여 잠재 표현(latent representation)을 학습합니다. 확산 모델(diffusion model)은 노이즈를 샘플링하고 반복적으로 노이즈를 제거(denoising)하여 이미지를 생성합니다. 몬테카를로(Monte Carlo) 방법은 닫힌 형태의 해가 없는 적분을 추정합니다. MCMC(Markov Chain Monte Carlo) 알고리즘은 열거가 불가능한 고차원 사후 분포(posterior distribution)를 탐색합니다.

모든 생성형 AI 시스템은 샘플링 시스템입니다. 샘플링 전략은 출력의 품질, 다양성, 제어 가능성을 결정합니다. 이 강의에서는 균일 난수(uniform random numbers)에서 시작하여 현대 LLM(Large Language Model)과 생성 모델을 구동하는 기술까지 모든 주요 샘플링 방법을 처음부터 구축합니다.

## 개념

### 샘플링이 중요한 이유

샘플링은 AI 및 머신러닝 전반에서 네 가지 근본적인 역할을 합니다:

**생성.** 언어 모델, 확산 모델, GAN은 모두 샘플링을 통해 출력을 생성합니다. 샘플링 알고리즘은 창의성, 일관성, 다양성을 직접 제어합니다. 온도(temperature), top-k, nucleus 샘플링은 엔지니어가 매일 조정하는 핵심 파라미터입니다.

**훈련.** 확률적 경사 하강법(SGD)은 미니배치를 샘플링합니다. 드롭아웃은 비활성화할 뉴런을 샘플링합니다. 데이터 증강은 무작위 변환을 샘플링합니다. 중요도 샘플링은 강화 학습(PPO, TRPO)에서 그래디언트 분산을 줄이기 위해 샘플 가중치를 재조정합니다.

**추정.** ML의 많은 양은 닫힌 형태의 해가 없습니다. 데이터 분포에 대한 기대 손실, 에너지 기반 모델의 분할 함수, 베이지안 추론의 증거 등이 있습니다. 몬테카를로 추정은 샘플 평균을 통해 이 모든 것을 근사합니다.

**탐색.** MCMC 알고리즘은 베이지안 추론에서 사후 분포를 탐색합니다. 진화 전략은 파라미터 섭동을 샘플링합니다. 톰슨 샘플링은 밴딧 문제에서 탐색과 활용을 균형 있게 조정합니다.

핵심 과제: 단순한 분포(균등, 정규)에서만 직접 샘플링할 수 있습니다. 그 외 모든 분포에는 단순 샘플을 목표 분포 샘플로 변환하는 방법이 필요합니다.

### 균등 무작위 샘플링

모든 샘플링 방법은 여기서 시작합니다. 균등 무작위 생성기는 [0, 1) 구간에서 값을 생성하며, 같은 길이의 모든 부분 구간은 동일한 확률을 가집니다.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

특성:
  E[U] = 0.5
  Var(U) = 1/12
```

n개의 이산 집합에서 균등하게 샘플링하려면 U를 생성하고 floor(n * U)를 반환합니다. 연속 범위 [a, b]에서 샘플링하려면 a + (b - a) * U를 계산합니다.

핵심 통찰: 단일 균등 무작위 수는 어떤 분포에서든 하나의 샘플을 생성하는 데 필요한 정확한 무작위성을 포함합니다. 올바른 변환을 찾는 것이 관건입니다.

### 역 CDF 방법 (역변환 샘플링)

누적 분포 함수(CDF)는 값을 확률로 매핑합니다:

```
F(x) = P(X <= x)

특성:
  F는 비감소 함수
  F(-inf) = 0
  F(+inf) = 1
  F는 실수선을 [0, 1]로 매핑
```

역 CDF는 확률을 다시 값으로 매핑합니다. U ~ Uniform(0, 1)이면 X = F_inverse(U)는 목표 분포를 따릅니다.

```
알고리즘:
  1. u ~ Uniform(0, 1) 생성
  2. F_inverse(u) 반환

작동 원리:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**지수 분포 예시:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

F(x) = u에 대해 x를 풀면:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

(1 - U)와 U는 동일한 분포를 가지므로:
  x = -ln(u) / lambda
```

이 방법은 F_inverse를 닫힌 형태로 쓸 수 있을 때 완벽하게 작동합니다. 정규 분포의 경우 닫힌 형태의 역 CDF가 없으므로 다른 방법(Box-Muller, 수치 근사)을 사용합니다.

**이산 버전:** 이산 분포의 경우 누적 합을 CDF로 구성하고, U를 생성한 후 누적 합이 U를 초과하는 첫 번째 인덱스를 찾습니다. 이는 Lesson 06의 `sample_categorical` 작동 방식입니다.

### 리젝션 샘플링

CDF를 역전할 수 없지만 상수를 제외한 목표 PDF를 평가할 수 있을 때 리젝션 샘플링을 사용합니다.

```
목표 분포: p(x)  (평가 가능, 정규화되지 않을 수 있음)
제안 분포: q(x)  (샘플링 가능)
상한: M, 모든 x에 대해 p(x) <= M * q(x)

알고리즘:
  1. x ~ q(x) 샘플링
  2. u ~ Uniform(0, 1) 샘플링
  3. u < p(x) / (M * q(x))이면 x를 수용
  4. 아니면 거부하고 1단계로 돌아감

수용률 = 1/M
```

상한 M이 더 타이트할수록 수용률이 높아집니다. 저차원(1-3차원)에서는 리젝션 샘플링이 잘 작동합니다. 고차원에서는 제안 부피의 대부분이 거부되어 수용률이 지수적으로 감소합니다. 이는 리젝션 샘플링의 차원의 저주입니다.

**예시: 절단된 정규 분포 샘플링.** 절단된 범위 내에서 균등 제안을 사용합니다. M은 해당 범위에서 정규 PDF의 최댓값입니다.

**예시: 반원 샘플링.** 경계 사각형 내에서 균등하게 제안합니다. 점이 반원 내부에 있으면 수용합니다. 이는 몬테카를로가 π를 계산하는 방식입니다: 수용률은 면적 비율 π/4와 같습니다.

### 중요도 샘플링

때로는 목표 분포 p(x)의 샘플이 필요하지 않습니다. p(x) 하에서의 기대값을 추정해야 하고, 다른 분포 q(x)에서 샘플을 가지고 있을 수 있습니다.

```
목표: E_p[f(x)] = f(x) * p(x) dx의 적분 추정

재작성:
  E_p[f(x)] = f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

여기서 w(x) = p(x) / q(x)는 중요도 가중치입니다.

추정기:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    x_i ~ q(x)
```

이는 강화 학습에서 중요합니다. PPO(Proximal Policy Optimization)에서는 이전 정책 pi_old로 궤적을 수집하지만 새로운 정책 pi_new를 최적화하려고 합니다. 중요도 가중치는 pi_new(a|s) / pi_old(a|s)입니다. PPO는 이 가중치를 클리핑하여 새 정책이 이전 정책과 너무 멀어지지 않도록 합니다.

중요도 샘플링 추정기의 분산은 q가 p와 얼마나 유사한지에 따라 달라집니다. q가 p와 매우 다르면 소수의 샘플이 엄청난 가중치를 얻어 추정을 지배합니다. 자기 정규화 중요도 샘플링은 가중치의 합으로 나누어 이 문제를 완화합니다:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 몬테카를로 추정

몬테카를로 추정은 적분을 무작위 샘플 평균으로 근사합니다. 대수의 법칙에 따라 수렴이 보장됩니다.

```
목표: I = 영역 D에서 g(x) dx의 적분 추정

방법:
  1. D에서 x_1, ..., x_N을 균등하게 샘플링
  2. I ~ (D의 부피 / N) * sum(g(x_i))

오차: O(1 / sqrt(N))   차원에 무관
```

오차율은 차원에 독립적입니다. 이 때문에 그리드 기반 적분이 불가능한 고차원에서 몬테카를로 방법이 우세합니다.

**π 추정:**

```
(x, y)를 [-1, 1] x [-1, 1]에서 균등하게 샘플링
단위 원 내부(x^2 + y^2 <= 1)에 있는 점들의 수를 세고
π ~ 4 * (내부 점 수) / (전체 점 수)
```

**기대값 추정:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    x_i ~ p(x)

표본 평균은 참 기대값으로 수렴합니다.
추정기의 분산 = Var(f(X)) / N
```

### 마르코프 체인 몬테카를로(MCMC): 메트로폴리스-헤이스팅스

MCMC는 정상 분포가 목표 분포 p(x)인 마르코프 체인을 구성합니다. 충분한 단계 후에는 체인의 샘플이 (대략적으로) p(x)의 샘플이 됩니다.

```
목표: p(x)  (정규화 상수까지 알려짐)
제안: q(x'|x)  (현재 상태에서 다음 상태를 제안하는 방법)

메트로폴리스-헤이스팅스 알고리즘:
  1. 임의의 x_0에서 시작
  2. t = 1, 2, ..., T에 대해:
     a. x' ~ q(x'|x_t) 제안
     b. 수용 비율 계산:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. min(1, alpha) 확률로 수용:
        - u < alpha (u ~ Uniform(0,1))이면 x_{t+1} = x'
        - 아니면 x_{t+1} = x_t
  3. 처음 B개 샘플 제거 (번인)
  4. 나머지 샘플 반환
```

대칭 제안(q(x'|x) = q(x|x'))의 경우 비율은 p(x')/p(x)로 단순화됩니다. 이는 원래 메트로폴리스 알고리즘입니다.

**작동 원리.** 수용 규칙은 상세 균형을 보장합니다: x에서 x'로 이동할 확률과 x'에서 x로 이동할 확률이 같습니다. 상세 균형은 p(x)가 체인의 정상 분포임을 의미합니다.

**실용적 고려사항:**
- 번인: 체인이 평형 상태에 도달하기 전의 초기 샘플 제거
- 얇게 자르기: 자기상관을 줄이기 위해 k번째 샘플만 보관
- 제안 스케일: 너무 작으면 체인이 느리게 이동(고수용률, 느린 탐색); 너무 크면 대부분의 제안이 거부됨(저수용률, 제자리 맴돌기)
- 고차원에서 가우시안 제안의 최적 수용률은 약 0.234

### 깁스 샘플링

깁스 샘플링은 다변량 분포를 위한 MCMC의 특수한 경우입니다. 모든 차원에서 한 번에 이동을 제안하는 대신, 한 번에 하나의 변수를 조건부 분포에서 업데이트합니다.

```
목표: p(x_1, x_2, ..., x_d)

알고리즘:
  각 반복 t에 대해:
    x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t) 샘플링
    x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t) 샘플링
    ...
    x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1}) 샘플링
```

깁스 샘플링은 각 조건부 분포 p(x_i | x_{-i})에서 샘플링할 수 있어야 합니다. 이는 많은 모델에서 간단합니다:
- 베이지안 네트워크: 그래프 구조에서 조건부 분포가 유도됨
- 가우시안 혼합: 조건부 분포가 가우시안
- 이징 모델: 각 스핀의 조건부는 이웃에만 의존

수용률은 항상 1입니다(모든 제안이 수용됨). 정확한 조건부 분포에서 샘플링하면 상세 균형이 자동으로 만족됩니다.

**한계.** 변수들이 강하게 상관되어 있을 때 깁스 샘플링은 느리게 혼합됩니다. 한 번에 하나의 변수만 업데이트하기 때문에 분포를 가로지르는 큰 대각선 이동이 불가능합니다.

### 온도 샘플링 (LLM에서 사용)

언어 모델은 각 어휘 토큰에 대한 로짓 z_1, ..., z_V를 출력합니다. 소프트맥스는 이를 확률로 변환합니다. 온도는 소프트맥스 전에 로짓을 재조정합니다:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: 표준 소프트맥스 (원래 분포)
T -> 0:  argmax (결정적, 항상 최고 로짓 선택)
T -> inf: 균등 (모든 토큰 동일한 확률)
T < 1.0: 분포를 날카롭게 (더 확신, 덜 다양)
T > 1.0: 분포를 평평하게 (덜 확신, 더 다양)
```

**작동 원리.** T < 1로 로짓을 나누면 로짓 간 차이가 증폭됩니다. z_1 = 2, z_2 = 1일 때 T = 0.5로 나누면 z_1/T = 4, z_2/T = 2가 되어 격차가 커집니다. 소프트맥스 후 최고 로짓 토큰이 훨씬 더 큰 확률을 가집니다.

**실제 적용:**
- T = 0.0: 탐욕적 디코딩, 사실 기반 Q&A에 최적
- T = 0.3-0.7: 약간 창의적, 코드 생성에 적합
- T = 0.7-1.0: 균형 잡힌, 일반 대화에 적합
- T = 1.0-1.5: 창의적 글쓰기, 브레인스토밍
- T > 1.5: 점점 무작위, 거의 유용하지 않음

온도는 가능한 토큰을 변경하지 않습니다. 각 토큰에 할당된 확률 질량을 변경합니다.

### Top-k 샘플링

Top-k 샘플링은 후보 집합을 확률이 가장 높은 k개의 토큰으로 제한한 후, 그 제한된 집합에서 재정규화하고 샘플링합니다.

```
알고리즘:
  1. 모든 V개 토큰에 대해 소프트맥스 확률 계산
  2. 토큰을 확률 내림차순으로 정렬
  3. 상위 k개 토큰만 유지
  4. 재정규화: p_i' = p_i / sum(p_j for j in top-k)
  5. 재정규화된 분포에서 샘플링

k = 1:  탐욕적 디코딩
k = V:  필터링 없음 (표준 샘플링)
k = 40: 일반적인 설정, 어휘 분포의 긴 꼬리 제거
```

Top-k는 모델이 어휘 분포의 긴 꼬리에 존재하는 극히 낮은 확률의 토큰(오타, 무의미한 토큰)을 선택하지 못하게 합니다. 문제점: k는 문맥과 무관하게 고정됩니다. 모델이 확신하는 경우(하나의 토큰이 95% 확률) k = 40은 여전히 39개의 대안을 허용합니다. 모델이 불확실한 경우(확률이 1000개 토큰에 분산) k = 40은 타당한 옵션을 잘라냅니다.

### Top-p (Nucleus) 샘플링

Top-p 샘플링은 후보 집합 크기를 동적으로 조정합니다. 고정된 수의 토큰을 유지하는 대신, 누적 확률이 p를 초과하는 가장 작은 토큰 집합을 유지합니다.

```
알고리즘:
  1. 모든 V개 토큰에 대해 소프트맥스 확률 계산
  2. 토큰을 확률 내림차순으로 정렬
  3. 상위 k개 토큰의 누적 확률 합이 p 이상이 되는 가장 작은 k 찾기
  4. 해당 k개 토큰만 유지
  5. 재정규화 후 샘플링

p = 0.9:  확률 질량의 90%를 포함하는 토큰 유지
p = 1.0:  필터링 없음
p = 0.1:  매우 제한적, 거의 탐욕적
```

모델이 확신하는 경우 nucleus 샘플링은 적은 수의 토큰(아마도 2-3개)을 유지합니다. 모델이 불확실한 경우 많은 토큰(아마도 200개)을 유지합니다. 이 적응적 행동 때문에 nucleus 샘플링은 일반적으로 top-k보다 더 나은 텍스트를 생성합니다.

**일반적인 조합:**
- 온도 0.7 + top-p 0.9: 일반적인 목적 설정
- 온도 0.0 (탐욕적): 결정적 작업에 최적
- 온도 1.0 + top-k 50: Fan et al. (2018) 원본 논문 설정

Top-k와 top-p를 결합할 수 있습니다. 먼저 top-k를 적용한 후 남은 집합에 top-p를 적용합니다.

### 재파라미터화 트릭 (VAE에서 사용)

변분 오토인코더(VAE)는 입력을 잠재 공간의 분포로 인코딩하고, 그 분포에서 샘플링한 후 샘플을 다시 디코딩하여 학습합니다. 문제점: 샘플링 연산을 통해 역전파를 할 수 없습니다.

```
표준 샘플링 (미분 불가능):
  z ~ N(mu, sigma^2)

  무작위성이 그래디언트 흐름을 차단합니다.
  d/d_mu [N(mu, sigma^2)에서 샘플링] = ???
```

재파라미터화 트릭은 무작위성을 파라미터와 분리합니다:

```
재파라미터화 샘플링:
  epsilon ~ N(0, 1)          (고정된 무작위 노이즈, 파라미터 없음)
  z = mu + sigma * epsilon   (파라미터의 결정적 함수)

  이제 z는 mu와 sigma의 결정적이고 미분 가능한 함수입니다.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  그래디언트가 mu와 sigma를 통해 흐릅니다.
```

이는 N(mu, sigma^2)가 mu + sigma * N(0, 1)과 동일한 분포를 가지기 때문에 가능합니다. 핵심 통찰: 무작위성을 파라미터 없는 소스(epsilon)로 이동한 후, 샘플을 파라미터의 미분 가능한 변환으로 표현합니다.

**VAE 학습 루프에서:**
1. 인코더는 각 입력에 대해 mu와 log(sigma^2)를 출력
2. epsilon ~ N(0, 1) 샘플링
3. z = mu + sigma * epsilon 계산
4. z를 디코딩하여 입력 재구성
5. 4, 3, 2, 1단계를 통해 역전파 (3단계가 미분 가능하므로 가능)

재파라미터화 트릭이 없으면 VAE는 표준 역전파로 학습할 수 없습니다. 이 단일 통찰이 VAE를 실용적으로 만들었습니다.

### Gumbel-Softmax (미분 가능한 범주형 샘플링)

재파라미터화 트릭은 연속 분포(가우시안)에 대해 작동합니다. 이산 범주형 분포에는 다른 접근법이 필요합니다. Gumbel-Softmax는 범주형 샘플링에 대한 미분 가능한 근사를 제공합니다.

**Gumbel-Max 트릭 (미분 불가능):**

```
log-확률 log(p_1), ..., log(p_k)를 가진 범주형 분포에서 샘플링:
  1. 각 범주에 대해 g_i ~ Gumbel(0, 1) 샘플링
     (g = -log(-log(u)), u ~ Uniform(0, 1))
  2. argmax(log(p_i) + g_i) 반환

이는 정확한 범주형 샘플을 생성합니다.
```

**Gumbel-Softmax (미분 가능한 근사):**

```
하드 argmax를 소프트 소프트맥스로 대체:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau(온도)는 근사를 제어:
  tau -> 0:  원-핫 벡터에 근접 (하드 범주형)
  tau -> inf:  균등 분포에 근접 (1/k, 1/k, ..., 1/k)
  tau = 1.0:  소프트 근사
```

Gumbel-Softmax는 이산 샘플의 연속 완화를 생성합니다. 출력은 하드 원-핫 대신 확률 벡터(소프트 원-핫)입니다. 그래디언트는 소프트맥스를 통해 흐릅니다. 학습 중 순전파에서는 "직진" 추정기를 사용할 수 있습니다: 순전파에는 하드 argmax를, 역전파에는 소프트 Gumbel-Softmax 그래디언트를 사용합니다.

**응용 분야:**
- VAE의 이산 잠재 변수
- 신경망 구조 탐색(이산 연산 선택)
- 하드 어텐션 메커니즘
- 이산 액션을 가진 강화 학습

### 층화 샘플링

표준 몬테카를로 샘플링은 우연히 샘플 공간에 빈틈을 남길 수 있습니다. 층화 샘플링은 공간을 층(strata)으로 나누고 각 층에서 샘플링하여 균일한 커버리지를 강제합니다.

```
표준 몬테카를로:
  [0, 1]에서 N개 점을 균등하게 샘플링
  일부 영역에는 클러스터, 다른 영역에는 빈틈이 생길 수 있음

층화 샘플링:
  [0, 1]을 N개의 동일한 층 [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)로 분할
  각 층 내에서 하나의 점을 균등하게 샘플링
  x_i = (i + u_i) / N   u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

층화 샘플링은 항상 표준 몬테카를로보다 낮거나 같은 분산을 가집니다:

```
Var(층화) <= Var(표준 몬테카를로)

개선 효과는 f(x)가 부드럽게 변할 때 가장 큽니다.
구간 상수 함수의 경우 층화 샘플링은 정확합니다.
```

**응용 분야:**
- 수치 적분 (준-몬테카를로)
- 훈련 데이터 분할 (각 폴드에서 클래스 균형 보장)
- 층화를 통한 중요도 샘플링 (두 기법 결합)
- NeRF(Neural Radiance Fields)는 카메라 광선을 따라 층화 샘플링을 사용

### 확산 모델과의 연결

확산 모델은 샘플링 과정을 통해 이미지를 생성합니다. 순방향 과정은 T단계에 걸쳐 이미지에 가우시안 노이즈를 추가하여 순수 노이즈가 되게 합니다. 역방향 과정은 노이즈를 제거하며 원본 이미지를 단계적으로 복원합니다.

```
순방향 과정 (알려짐):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  T단계 후: x_T ~ N(0, I)  (순수 노이즈)

역방향 과정 (학습됨):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  각 노이즈 제거 단계는 샘플링 단계입니다.
```

이 레슨의 방법들과의 연결:
- 각 노이즈 제거 단계는 재파라미터화 트릭을 사용(노이즈 샘플링, 결정적 변환 적용)
- 노이즈 스케줄 {alpha_t}는 온도 어닐링의 형태를 제어
- 훈련은 ELBO(증거 하한)를 근사하기 위해 몬테카를로 추정을 사용
- 확산 모델의 조상 샘플링은 마르코프 체인(각 단계는 현재 상태에만 의존)

전체 이미지 생성 과정은 반복적 샘플링입니다: 노이즈에서 시작하여 각 단계에서 학습된 노이즈 제거 모델에 조건부로 약간 덜 노이즈가 있는 버전을 샘플링합니다.

## 구축 방법

### 1단계: 균일 분포 및 역 CDF 샘플링

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

10,000개의 지수 분포 샘플을 생성하고 평균이 1/lambda인지 확인합니다.

### 2단계: 거부 샘플링

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

거부 샘플링을 사용하여 절단된 정규 분포에서 샘플을 추출합니다. 샘플 히스토그램을 통해 형태를 확인합니다.

### 3단계: 중요도 샘플링

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

균일 제안 분포를 사용하여 정규 분포 하에서 E[X^2]를 추정합니다. 알려진 답(mu^2 + sigma^2)과 비교합니다.

### 4단계: 몬테카를로 파이 추정

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 5단계: 메트로폴리스-헤이스팅스 MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

이중 모드 분포(두 개의 가우시안 혼합)에서 샘플을 추출합니다. 체인의 궤적을 시각화합니다.

### 6단계: 깁스 샘플링

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 7단계: 온도 샘플링

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

온도 변화가 토큰 로짓의 출력 분포에 미치는 영향을 보여줍니다.

### 8단계: Top-k 및 Top-p 샘플링

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 9단계: 재매개변수화 트릭

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

재매개변수화된 샘플을 통해 그래디언트가 흐르지만 직접 샘플링에서는 그래디언트가 흐르지 않음을 보여줍니다.

### 10단계: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

온도를 낮추면 출력이 원-핫 벡터에 가까워지는 것을 보여줍니다.

모든 시각화가 포함된 전체 구현은 `code/sampling.py`에 있습니다.

## 사용 방법

NumPy와 SciPy를 사용한 프로덕션 버전:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"지수 분포 평균: {exponential_samples.mean():.4f} (예상 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"1.96에서의 CDF: {normal.cdf(1.96):.4f}")
print(f"0.975에서의 역 CDF: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"샘플링된 토큰 인덱스: {token}")
```

대규모 MCMC의 경우 전용 라이브러리 사용:
- PyMC: NUTS(적응형 HMC)를 활용한 베이지안 모델링
- emcee: 앙상블 MCMC 샘플러
- NumPyro/JAX: GPU 가속 MCMC

이제 직접 구현한 내용을 바탕으로 라이브러리 함수들이 수행하는 작업을 이해할 수 있습니다.

## 연습 문제

1. 코시 분포(Cauchy distribution)에 대한 역 CDF 샘플링을 구현하세요. CDF는 F(x) = 0.5 + arctan(x)/pi입니다. 10,000개의 샘플을 생성하고 히스토그램을 실제 PDF와 비교하세요. 중심에서 멀리 떨어진 극단값(heavy tails)을 확인하세요.

2. 균일 분포(Uniform(0, 1)) 제안 분포를 사용하여 거부 샘플링(rejection sampling)으로 베타 분포(Beta(2, 5)) 샘플을 생성하세요. 수용된 샘플을 실제 베타 PDF와 비교하세요. 이론적 수용률은 얼마인가요?

3. 1,000, 10,000, 100,000개의 샘플로 몬테 카를로(Monte Carlo) 방법을 사용하여 0부터 pi까지 sin(x)의 적분을 추정하세요. 각 수준에서 오차를 비교하세요. 오차가 O(1/sqrt(N)) 비율로 감소하는지 확인하세요.

4. 2D 분포 p(x, y) ∝ exp(-(x² * y² + x² + y² - 8*x - 8*y) / 2)에서 샘플링하기 위해 메트로폴리스-헤이스팅스(Metropolis-Hastings) 알고리즘을 구현하세요. 샘플과 체인 궤적(chain trajectory)을 플롯하세요. 다양한 제안 표준 편차를 실험해 보세요.

5. 완전한 텍스트 생성 데모를 구축하세요: 로짓(logits)이 주어진 10개 단어 어휘(vocabulary)에서 (a) 탐욕적(greedy), (b) 온도=0.7, (c) top-k=3, (d) top-p=0.9 방법을 사용하여 20토큰 시퀀스를 생성하세요. 5회 실행 간 출력 다양성을 비교하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 샘플링 | "무작위 값 추출" | 확률 분포에 따라 값을 생성. 모든 생성형 AI의 기반 메커니즘 |
| 균일 분포 | "모두 동일한 확률" | [a, b] 구간 내 모든 값이 확률 밀도 1/(b-a)를 가짐. 모든 샘플링 방법의 시작점 |
| 역 CDF | "확률 변환" | F_inverse(U)는 균일 샘플을 알려진 CDF를 가진 분포의 샘플로 변환. 정확하고 효율적 |
| 리젝션 샘플링 | "제안 후 수락/거부" | 간단한 제안 분포에서 생성, 목표/제안 비율에 비례하는 확률로 수락. 정확하지만 샘플 낭비 발생 |
| 중요도 샘플링 | "샘플 가중치 재조정" | q(x) 샘플에 p(x)/q(x) 가중치를 적용해 p(x) 하의 기댓값 추정. 강화학습의 PPO 핵심 |
| 몬테카를로 | "무작위 샘플 평균" | 적분을 샘플 평균으로 근사. 오차는 차원 무관하게 O(1/sqrt(N)) |
| MCMC | "수렴하는 랜덤 워크" | 정적 분포가 목표 분포인 마르코프 체인 구성. 메트로폴리스-헤이스팅스가 기본 알고리즘 |
| 메트로폴리스-헤이스팅스 | "상승은 항상 수락, 하강은 가끔 수락" | 이동 제안, 밀도 비율 기반 수락. 상세 균형은 목표 분포 수렴 보장 |
| 깁스 샘플링 | "한 번에 하나의 변수" | 다른 변수를 고정한 채 각 변수의 조건부 분포에서 업데이트. 100% 수락률 |
| 온도 | "확신도 조절기" | 소프트맥스 전 로짓을 T로 나눔. T<1은 확신도 증가(뾰족), T>1은 다양성 증가(평탄) |
| Top-k 샘플링 | "상위 k개 유지" | k개 최고 확률 토큰 외 0 처리, 재정규화 후 샘플링. 고정된 후보 집합 크기 |
| 뉴클레우스 샘플링 (top-p) | "확률 높은 것들 유지" | 누적 확률이 p를 초과하는 최소 토큰 집합 유지. 적응형 후보 집합 크기 |
| 재파라미터화 트릭 | "무작위성 외부로 이동" | z = mu + sigma * epsilon (epsilon ~ N(0,1))로 표현. 샘플링을 미분 가능하게 함. VAE 훈련 필수 |
| 검블-소프트맥스 | "부드러운 범주형 샘플링" | 검블 노이즈 + 온도 적용 소프트맥스를 이용한 범주형 샘플링의 미분 가능 근사 |
| 층화 샘플링 | "강제 커버리지" | 샘플 공간을 층(strata)으로 분할, 각 층에서 샘플링. 순진 몬테카를로보다 항상 낮은 분산 |
| 번인 | "워밍업 기간" | 체인이 정적 분포에 도달하기 전 버려지는 초기 MCMC 샘플 |
| 상세 균형 | "가역성 조건" | p(x) * T(x->y) = p(y) * T(y->x). p가 마르코프 체인의 정적 분포임을 보장하는 충분 조건 |
| 확산 샘플링 | "반복적 노이즈 제거" | 노이즈에서 시작해 학습된 노이즈 제거 단계를 적용해 데이터 생성. 각 단계는 조건부 샘플링 연산 |

## 추가 자료

- [Holbrook (2023): 메트로폴리스-헤이스팅스 알고리즘](https://arxiv.org/abs/2304.07010) - MCMC 기초에 대한 상세 튜토리얼
- [Jang, Gu, Poole (2017): Gumbel-Softmax를 이용한 범주형 재매개변수화](https://arxiv.org/abs/1611.01144) - 원본 Gumbel-Softmax 논문
- [Holtzman et al. (2020): 신경망 텍스트 생성의 흥미로운 사례](https://arxiv.org/abs/1904.09751) - 뉴클레우스(탑-p) 샘플링 논문
- [Kingma & Welling (2014): 오토인코딩 변분 베이지안](https://arxiv.org/abs/1312.6114) - 재매개변수화 트릭을 소개한 VAE 논문
- [Ho, Jain, Abbeel (2020): 노이즈 제거 확산 확률 모델](https://arxiv.org/abs/2006.11239) - DDPM이 샘플링을 이미지 생성에 연결