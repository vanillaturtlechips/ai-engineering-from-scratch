# 확률과 분포

> 확률은 AI가 불확실성을 표현하는 언어입니다.

**유형:** 학습  
**언어:** Python  
**선수 지식:** 1단계, 레슨 01-04  
**소요 시간:** ~75분

## 학습 목표

- 베르누이(Bernoulli), 범주형(categorical), 포아송(Poisson), 균등(uniform), 정규(normal) 분포에 대한 PMF(Probability Mass Function) 및 PDF(Probability Density Function)를 처음부터 구현
- 기댓값(expected value), 분산(variance) 계산 및 중심 극한 정리(Central Limit Theorem)를 활용해 가우시안(Gaussian)이 지배적인 이유 설명
- 수치적 안정성 트릭(최대 로짓(logit) 빼기)을 적용한 소프트맥스(softmax) 및 로그-소프트맥스(log-softmax) 함수 구현
- 로짓(logits)으로부터 교차 엔트로피 손실(cross-entropy loss) 계산 및 음의 로그-우도(negative log-likelihood)와의 연결 관계 설명

## 문제

분류기가 `[0.03, 0.91, 0.06]`을 출력합니다. 언어 모델은 50,000개의 후보 단어 중 다음 단어를 선택합니다. 확산 모델은 학습된 분포에서 샘플링하여 이미지를 생성합니다. 이 모든 것은 확률의 실제 적용 사례입니다.

모델이 만드는 모든 예측은 확률 분포입니다. 모든 손실 함수(loss function)는 예측된 분포와 실제 분포가 얼마나 떨어져 있는지를 측정합니다. 모든 학습 단계는 한 분포가 다른 분포와 더 비슷해지도록 매개변수를 조정합니다. 확률이 없다면 단일 ML 논문을 읽거나, 단일 모델을 디버깅하거나, 학습 손실(loss)이 왜 NaN인지 이해할 수 없습니다.

## 개념

### 사건, 표본 공간, 확률

표본 공간 S는 모든 가능한 결과의 집합입니다. 사건은 표본 공간의 부분집합입니다. 확률은 사건을 0과 1 사이의 숫자로 매핑합니다.

```
동전 던지기:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

주사위 한 번 던지기:
  S = {1, 2, 3, 4, 5, 6}
  P(짝수) = P({2, 4, 6}) = 3/6 = 0.5
```

확률을 정의하는 세 가지 공리:
1. 모든 사건 A에 대해 P(A) >= 0
2. P(S) = 1 (항상 어떤 일이 발생함)
3. A와 B가 동시에 발생할 수 없을 때, P(A 또는 B) = P(A) + P(B)

베이즈 정리, 기댓값, 분포 등 모든 다른 개념은 이 세 규칙에서 파생됩니다.

### 조건부 확률과 독립성

P(A|B)는 B가 발생했을 때 A의 확률입니다.

```
P(A|B) = P(A와 B) / P(B)

예시: 카드 덱
  P(King | Face 카드) = P(King와 Face 카드) / P(Face 카드)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

두 사건이 독립적일 때, 한 사건을 아는 것이 다른 사건에 대한 정보를 주지 않습니다:

```
독립:   P(A|B) = P(A)
동치:   P(A와 B) = P(A) * P(B)
```

동전 던지기는 독립적입니다. 카드를 다시 넣지 않고 뽑는 것은 독립적이지 않습니다.

### 확률 질량 함수 vs 확률 밀도 함수

이산 확률 변수는 확률 질량 함수(PMF)를 가집니다. 각 결과는 직접 읽을 수 있는 특정 확률을 가집니다.

```
PMF: P(X = k)

공정한 주사위:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  모든 확률의 합 = 1
```

연속 확률 변수는 확률 밀도 함수(PDF)를 가집니다. 단일 지점의 밀도는 확률이 아닙니다. 확률은 구간에서 밀도를 적분하여 얻습니다.

```
PDF: f(x)

P(a <= X <= b) = a부터 b까지 f(x)의 적분

f(x)는 1보다 클 수 있음 (밀도, 확률이 아님)
f(x)의 -inf부터 +inf까지 적분 = 1
```

이 구분은 ML에서 중요합니다. 분류 출력은 PMF(이산 선택)입니다. VAE 잠재 공간은 PDF(연속)를 사용합니다.

### 일반적인 분포

**베르누이:** 한 번의 시행, 두 가지 결과. 이진 분류를 모델링합니다.

```
P(X = 1) = p
P(X = 0) = 1 - p
평균 = p,  분산 = p(1-p)
```

**범주형:** 한 번의 시행, k가지 결과. 다중 클래스 분류(softmax 출력)를 모델링합니다.

```
P(X = i) = p_i,  여기서 p_i의 합 = 1
예시: P(고양이) = 0.7,  P(개) = 0.2,  P(새) = 0.1
```

**균등:** 모든 결과가 동일한 확률을 가짐. 무작위 초기화에 사용됩니다.

```
이산: P(X = k) = 1/n (k는 {1, ..., n}에 속함)
연속: f(x) = 1/(b-a) (x는 [a, b]에 속함)
```

**정규(가우시안):** 종 모양의 곡선. 평균(mu)과 분산(sigma^2)으로 매개변수화됩니다.

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

표준 정규 분포: mu = 0, sigma = 1
  68% 데이터는 1 시그마 내에 있음
  95% 데이터는 2 시그마 내에 있음
  99.7% 데이터는 3 시그마 내에 있음
```

**포아송:** 고정 구간 내 희귀 사건의 발생 횟수. 사건 발생률을 모델링합니다.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
평균 = lambda,  분산 = lambda
```

### 기댓값과 분산

기댓값은 가중 평균 결과입니다.

```
이산:   E[X] = x_i * P(X = x_i)의 합
연속: E[X] = x * f(x)의 적분
```

분산은 평균 주변의 퍼짐을 측정합니다.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
표준 편차 = sqrt(Var(X))
```

ML에서 기댓값은 손실 함수(데이터 분포에 대한 평균 손실)로 나타납니다. 분산은 모델 안정성을 알려줍니다. 그래디언트의 높은 분산은 노이즈가 많은 훈련을 의미합니다.

### 결합 분포와 주변 분포

결합 분포 P(X, Y)는 두 확률 변수를 함께 설명합니다.

결합 PMF 예시 (X = 날씨, Y = 우산):

| | Y=0 (우산 없음) | Y=1 (우산 있음) | 주변 P(X) |
|---|---|---|---|
| X=0 (맑음) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (비) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **주변 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

주변 분포는 다른 변수를 합산하여 구합니다:

```
P(X = x) = 모든 y에 대해 P(X = x, Y = y)의 합
```

위 표의 행과 열 합계는 주변 분포입니다.

### 정규 분포가 어디에나 나타나는 이유

중심 극한 정리: 많은 독립 확률 변수의 합(또는 평균)은 원래 분포와 관계없이 정규 분포에 수렴합니다.

```
주사위 1개 던지기:  균등 분포 (평평함)
주사위 2개 평균:  삼각형 (정점 있음)
주사위 30개 평균: 거의 완벽한 종 모양

이는 모든 시작 분포에 대해 성립합니다.
```

이것이 이유인 경우:
- 측정 오차는 대략 정규 분포 (많은 작은 독립 원인)
- 신경망의 가중치 초기화는 정규 분포 사용
- SGD의 그래디언트 노이즈는 대략 정규 분포 (많은 샘플 그래디언트의 합)
- 정규 분포는 주어진 평균과 분산에 대한 최대 엔트로피 분포

### 로그 확률

원시 확률은 수치적 문제를 일으킵니다. 많은 작은 확률을 곱하면 빠르게 0으로 언더플로우됩니다.

```
P(문장) = P(단어1) * P(단어2) * ... * P(단어_n)
        = 0.01 * 0.003 * 0.02 * ...
        -> 0.0 (약 30개 항 후 언더플로우)
```

로그 확률은 이를 해결합니다. 곱셈은 덧셈으로 변환됩니다.

```
log P(문장) = log P(단어1) + log P(단어2) + ... + log P(단어_n)
            = -4.6 + -5.8 + -3.9 + ...
            -> 유한한 수 (언더플로우 없음)
```

규칙:
- log(a * b) = log(a) + log(b)
- 로그 확률은 항상 <= 0 (0 < P <= 1이므로)
- 더 음수일수록 덜 발생 가능
- 교차 엔트로피 손실은 정답 클래스의 음의 로그 확률

### 확률 분포로서의 소프트맥스

신경망은 원시 점수(로짓)를 출력합니다. 소프트맥스는 이를 유효한 확률 분포로 변환합니다.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

특성:
  - 모든 출력은 (0, 1) 범위
  - 모든 출력의 합은 1
  - 입력의 상대적 순서 유지
  - exp()는 로짓 간 차이를 증폭
```

소프트맥스 트릭: 지수 계산 전 최대 로짓을 빼서 오버플로우를 방지합니다.

```
z = [100, 101, 102]
exp(102) = 오버플로우

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (안전)

동일한 결과, 오버플로우 없음.
```

로그-소프트맥스는 수치적 안정성을 위해 소프트맥스와 로그를 결합합니다. PyTorch는 교차 엔트로피 손실 계산 시 이를 내부적으로 사용합니다.

### 샘플링

샘플링은 분포에서 무작위 값을 추출하는 것을 의미합니다. ML에서:
- 드롭아웃은 무작위로 어떤 뉴런을 0으로 만들지 샘플링
- 데이터 증강은 무작위 변환을 샘플링
- 언어 모델은 예측된 분포에서 다음 토큰을 샘플링
- 확산 모델은 노이즈를 샘플링하고 점진적으로 제거

임의의 분포에서 샘플링하려면 역변환 샘플링, 거부 샘플링 또는 재매개변수화 트릭(VAE에서 사용)과 같은 기법이 필요합니다.

## 구축 방법

### 1단계: 확률 기초

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 2단계: PMF와 PDF 직접 구현

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 3단계: 기댓값과 분산

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"주사위: E[X] = {mu:.4f}, Var(X) = {var:.4f}, 표준편차 = {var**0.5:.4f}")
```

### 4단계: 분포에서 샘플링

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 5단계: 소프트맥스와 로그 확률

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 6단계: 중심 극한 정리 시연

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 7단계: 시각화

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

모든 시각화가 포함된 전체 구현은 `code/probability.py`에 있습니다.

## 사용 방법

NumPy와 SciPy를 사용하면 위의 모든 작업이 한 줄 코드로 가능합니다:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"평균: {np.mean(samples):.4f}, 표준편차: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"소프트맥스: {probs}")
print(f"로그-소프트맥스: {log_probs}")
```

이제 직접 구현해본 기능들을 라이브러리로 호출할 때 어떤 작업이 수행되는지 알게 되었습니다.

## 연습 문제

1. 지수 분포에 대한 역변환 샘플링(inverse transform sampling)을 구현하세요. 10,000개의 값을 샘플링하고 히스토그램을 실제 PDF와 비교하여 검증하세요.

2. 두 개의 조작된 주사위(loaded dice)에 대한 결합 분포 테이블을 만드세요. 주변 분포(marginal distribution)를 계산하고 주사위들이 독립(independent)인지 확인하세요.

3. 정답 클래스가 인덱스 3일 때, 로짓(logits) `[2.0, 0.5, -1.0, 3.0, 0.1]`을 출력하는 5-클래스 분류기의 교차 엔트로피 손실(cross-entropy loss)을 계산하세요. 그런 다음 PyTorch의 `nn.CrossEntropyLoss`로 답을 검증하세요.

4. 로그 확률(log probability) 리스트를 입력받아 가장 가능성 높은 시퀀스, 총 로그 확률, 그리고 동등한 원시 확률(raw probability)을 반환하는 함수를 작성하세요. 각 단어의 확률이 0.01인 50단어 문장으로 테스트하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| 표본 공간(Sample space) | "모든 가능성" | 실험의 모든 가능한 결과의 집합 S |
| 확률 질량 함수(PMF) | "확률 함수" | 각 이산 결과의 정확한 확률을 제공하며 합이 1이 되는 함수 |
| 확률 밀도 함수(PDF) | "확률 곡선" | 연속 변수를 위한 밀도 함수. 구간에 대해 적분하면 확률을 얻음 |
| 조건부 확률(Conditional probability) | "무언가 주어졌을 때의 확률" | P(A\|B) = P(A and B) / P(B). 베이지안 사고와 베이즈 정리의 기초 |
| 독립성(Independence) | "서로 영향을 주지 않음" | P(A and B) = P(A) * P(B). 한 사건을 안다고 다른 사건에 대한 정보가 없음 |
| 기댓값(Expected value) | "평균" | 모든 결과의 확률 가중치 합. 손실 함수는 기댓값임 |
| 분산(Variance) | "퍼져 있는 정도" | 평균으로부터의 제곱 편차의 기댓값. 높은 분산은 노이즈가 많고 불안정한 추정치를 의미 |
| 정규 분포(Normal distribution) | "종 모양 곡선" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). 중심 극한 정리(CLT)로 인해 어디서나 나타남 |
| 중심 극한 정리(Central Limit Theorem) | "평균은 정규 분포를 따름" | 많은 독립 표본의 평균은 원본 분포와 무관하게 정규 분포로 수렴 |
| 결합 분포(Joint distribution) | "두 변수가 함께" | P(X, Y)는 X와 Y 결과의 모든 조합에 대한 확률을 설명 |
| 주변 분포(Marginal distribution) | "다른 변수를 합산" | P(X) = sum_y P(X, Y). 결합 분포에서 한 변수의 분포를 복원 |
| 로그 확률(Log probability) | "확률의 로그" | log P(x). 곱을 합으로 변환하여 긴 시퀀스에서 수치적 언더플로우 방지 |
| 소프트맥스(Softmax) | "점수를 확률로 변환" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). 실수 값 로짓을 유효한 확률 분포로 매핑 |
| 교차 엔트로피(Cross-entropy) | "손실 함수" | -sum(p_true * log(p_predicted)). 두 분포가 얼마나 다른지 측정. 낮을수록 좋음 |
| 로짓(Logits) | "원시 모델 출력" | 소프트맥스 이전의 비정규화된 점수. 로지스틱 함수에서 이름 유래 |
| 샘플링(Sampling) | "무작위 값 추출" | 확률 분포에 따라 값을 생성. 모델이 출력을 생성하는 방식

## 추가 자료

- [3Blue1Brown: 하지만 중심 극한 정리란 무엇인가?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 평균이 왜 정규분포를 따르는지 시각적 증명
- [Stanford CS229 확률론 복습](https://cs229.stanford.edu/section/cs229-prob.pdf) - 여기 포함된 모든 내용과 그 이상을 다루는 간결한 참고 자료
- [로그-합-지수(Log-Sum-Exp) 트릭](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 수치적 안정성의 중요성과 이를 달성하는 방법