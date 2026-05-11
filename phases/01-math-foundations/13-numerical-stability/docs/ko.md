# 수치적 안정성

> 부동 소수점은 누수 있는 추상화입니다. 학습 중에 문제가 발생할 수 있으며, 당신은 이를 예상하지 못할 것입니다.

**유형:** 빌드  
**언어:** Python  
**선수 지식:** 1단계, 레슨 01-04  
**소요 시간:** ~120분

## 학습 목표

- 최댓값 빼기 기법(max-subtraction trick)을 사용하여 수치적으로 안정적인 소프트맥스(softmax)와 로그-합-지수(log-sum-exp) 구현
- 부동소수점 연산에서 오버플로우(overflow), 언더플로우(underflow), 치명적 상쇄(catastrophic cancellation) 식별
- 중심 유한 차분법(centered finite differences)을 사용하여 분석적 기울기(analytical gradient)와 수치적 기울기(numerical gradient) 검증
- 훈련 시 bfloat16이 float16보다 선호되는 이유 설명 및 손실 스케일링(loss scaling)이 기울기 언더플로우(gradient underflow)를 방지하는 방법 설명

## 문제

모델이 3시간 동안 훈련한 후 손실(loss)이 NaN이 됩니다. print 문을 추가하니, 9,000번째 스텝에서는 로짓(logit)이 정상입니다. 하지만 9,001번째 스텝에서는 로짓이 `inf`가 됩니다. 9,002번째 스텝에서는 모든 그래디언트(gradient)가 `nan`이 되고 훈련이 중단됩니다.

또는: 모델은 훈련을 완료했지만 정확도가 논문에서 주장한 것보다 2% 낮습니다. 모든 것을 확인했습니다. 아키텍처(architecture)도 일치하고, 하이퍼파라미터(hyperparameter)도 일치하며, 데이터도 일치합니다. 문제는 논문에서 float32를 사용했는데, 당신은 적절한 스케일링(scaling) 없이 float16을 사용했기 때문입니다. 32비트의 누적 반올림 오차(accumulated rounding error)가 조용히 정확도를 갉아먹었습니다.

또는: 크로스-엔트로피 손실(cross-entropy loss)을 처음부터 구현했습니다. 작은 로짓에서는 작동합니다. 하지만 로짓이 100을 초과하면 `inf`를 반환합니다. `exp(100)`이 float32로 표현할 수 있는 값보다 커서 소프트맥스(softmax)가 오버플로(overflow)되었기 때문입니다. 모든 ML 프레임워크는 이 문제를 2줄짜리 트릭으로 해결합니다. 하지만 당신은 그 트릭이 존재하는지 몰랐습니다.

수치적 안정성(numerical stability)은 이론적인 문제가 아닙니다. 훈련이 성공하는 것과 소리 없이 실패하는 것의 차이입니다. 당신이 디버깅하게 될 모든 심각한 ML 버그는 결국 부동소수점(floating point) 문제로 귀결됩니다.

## 개념

### IEEE 754: 컴퓨터가 실수(real number)를 저장하는 방법

컴퓨터는 IEEE 754 표준을 따라 실수를 부동소수점(floating point) 값으로 저장합니다. 부동소수점(float)은 부호 비트(sign bit), 지수(exponent), 가수(mantissa, significand)의 세 부분으로 구성됩니다.

```
Float32 구조 (총 32비트):
[1 부호] [8 지수] [23 가수]

값 = (-1)^부호 * 2^(지수 - 127) * 1.가수
```

가수는 정밀도(유의 숫자 개수)를 결정하고, 지수는 범위(숫자의 크기)를 결정합니다.

```
형식     비트   지수  가수  소수점 이하 자릿수  범위 (약)
float64  64     11    52    ~15-16              +/- 1.8e308
float32  32     8     23    ~7-8                +/- 3.4e38
float16  16     5     10    ~3-4                +/- 65,504
bfloat16 16     8     7     ~2-3                +/- 3.4e38
```

float32는 약 7자리 소수점 정밀도를 제공합니다. 즉, 1.0000001과 1.0000002는 구분할 수 있지만, 1.00000001과 1.00000002는 구분할 수 없습니다. 7자리 이후에는 모든 것이 반올림 오차입니다.

float16은 약 3자리 정밀도를 제공합니다. 표현 가능한 최대값은 65,504로, 로짓(logits), 그래디언트(gradients), 활성화값(activations)이 이 값을 쉽게 초과하는 ML에서는 문제가 됩니다.

bfloat16은 float16의 범위 문제를 해결하기 위한 Google의 대안입니다. float32와 동일한 8비트 지수(범위: 최대 3.4e38)를 가지지만, 가수는 7비트로 float16보다 정밀도가 낮습니다. 신경망 훈련에서는 정밀도보다 범위가 더 중요하므로, 일반적으로 bfloat16이 더 적합합니다.

### 0.1 + 0.2 != 0.3인 이유

0.1은 이진 부동소수점으로 정확히 표현할 수 없습니다. 이진수로 표현하면 다음과 같이 반복 소수입니다:

```
0.1 (이진수) = 0.0001100110011001100110011... (무한 반복)
```

float32는 이를 23비트 가수로 자릅니다. 저장된 값은 약 0.100000001490116입니다. 마찬가지로 0.2는 약 0.200000002980232로 저장됩니다. 두 수의 합은 0.300000004470348로, 0.3이 아닙니다.

```
Python 예시:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

이 문제는 ML에서 다음과 같은 영향을 미칩니다:

1. `if loss < threshold`와 같은 손실 비교에서 잘못된 결과가 나올 수 있음
2. 수천 단계의 그래디언트 업데이트에서 작은 값들이 누적되면 실제 합과 차이가 발생함
3. 체크섬(checksum) 및 재현성(reproducibility) 테스트에서 `==`로 부동소수점을 비교하면 실패함

해결 방법: 부동소수점을 `==`로 비교하지 마세요. `abs(a - b) < epsilon` 또는 `math.isclose()`를 사용하세요.

### 치명적 상쇄(Catastrophic Cancellation)

거의 같은 두 부동소수점 수를 빼면 유의 숫자가 상쇄되고, 반올림 오차가 선두 숫자로 승격됩니다.

```
a = 1.0000001    (float32로 저장된 값: 1.00000011920929)
b = 1.0000000    (float32로 저장된 값: 1.00000000000000)

실제 차이:  0.0000001
계산된 값:  0.00000011920929

상대 오차: 19.2%
```

단일 뺄셈에서 19%의 상대 오차가 발생할 수 있습니다. ML에서는 다음과 같은 경우에 발생합니다:

- 큰 평균값을 가진 데이터의 분산 계산: `E[x^2] - E[x]^2`에서 E[x]가 큰 경우
- 거의 같은 로그-확률(log-probabilities) 뺄셈
- 너무 작은 epsilon으로 유한 차분(finite-difference) 그래디언트 계산

해결 방법: 큰 거의 같은 수를 빼는 공식을 재구성하세요. 분산의 경우 Welford 알고리즘이나 데이터 중심화를 사용하세요. 로그-확률의 경우 로그 공간(log-space)에서 작업하세요.

### 오버플로우(Overflow)와 언더플로우(Underflow)

오버플로우는 결과가 표현 가능한 최대값을 초과할 때 발생합니다. 언더플로우는 결과가 너무 작아(0에 너무 가까워) 표현 가능한 최소 양수보다 작을 때 발생합니다.

```
float32 경계:
  최대값:  3.4028235e+38
  최소 양수 (정규): 1.175e-38
  최소 양수 (비정규): 1.401e-45
  오버플로우: 3.4e38 초과 시 inf
  언더플로우: 1.4e-45 미만 시 0.0
```

`exp()` 함수는 ML에서 오버플로우의 주요 원인입니다:

```
exp(88.7)  = 3.40e+38   (float32에 간신히 들어감)
exp(89.0)  = inf         (오버플로우)
exp(-87.3) = 1.18e-38   (언더플로우 직전)
exp(-104)  = 0.0         (0으로 언더플로우)
```

`log()` 함수는 반대 방향 문제를 일으킵니다:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (정상)
log(1e-46) = -inf        (입력이 0으로 언더플로우된 후 log(0) = -inf)
```

ML에서 `exp()`는 소프트맥스(softmax), 시그모이드(sigmoid), 확률 계산에 사용됩니다. `log()`는 교차 엔트로피(cross-entropy), 로그-우도(log-likelihood), KL 발산(KL divergence)에 사용됩니다. `log(exp(x))`는 적절한 트릭 없이는 위험한 연산입니다.

### 로그-합-지수(Log-Sum-Exp) 트릭

`log(sum(exp(x_i)))`를 직접 계산하면 수치적 불안정성이 발생합니다. 어떤 `x_i`가 크면 `exp(x_i)`가 오버플로우되고, 모든 `x_i`가 매우 음수면 `exp(x_i)`가 0으로 언더플로우되어 `log(0)`은 `-inf`가 됩니다.

트릭: 지수 계산 전에 최대값을 빼세요.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

이 방법이 작동하는 이유: `max(x)`를 빼면 가장 큰 지수가 `exp(0) = 1`이 됩니다. 오버플로우가 불가능합니다. 합계의 최소값은 1이므로 `log(1) = 0`이 되어 `-inf`로 언더플로우되지 않습니다.

증명:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (c를 더하고 뺌)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (exp(c) 인수분해)
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

`c = max(x)`로 설정하면 오버플로우가 제거됩니다.

이 트릭은 ML에서 다음과 같은 곳에 사용됩니다:
- 소프트맥스 정규화
- 교차 엔트로피 손실 계산
- 시퀀스 모델의 로그-확률 합산
- 가우시안 혼합 모델
- 변분 추론

### 소프트맥스가 최대값-뺄셈 트릭을 필요로 하는 이유

소프트맥스는 로짓(logits)을 확률로 변환합니다:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

트릭을 사용하지 않으면 [100, 101, 102] 같은 로짓은 오버플로우를 일으킵니다:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
합계     = 2.99e44

이 값들은 float32 최대값(~3.4e38)을 초과하는가? 아니요, 2.69e43 < 3.4e38? 실제로:
exp(88.7)은 이미 float32 한계입니다.
exp(100) = float32에서 inf.
```

트릭을 사용하면 최대값 `max(x) = 102`를 뺍니다:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
합계 = 1.503

softmax = [0.090, 0.245, 0.665]
```

확률은 동일하지만 계산은 안전합니다. 이는 최적화가 아니라 정확성을 위한 필수 조건입니다.

### NaN과 Inf: 탐지 및 예방

`nan`(Not a Number)과 `inf`(무한대)는 계산을 통해 전파됩니다. 그래디언트 업데이트에서 하나의 `nan`이 발생하면 가중치가 `nan`이 되고, 이후 모든 출력이 `nan`이 됩니다. 훈련은 한 단계 내에 실패합니다.

`inf`가 발생하는 경우:
- 큰 양수의 `exp()`
- 0으로 나누기: `1.0 / 0.0`
- float32 오버플로우

`nan`이 발생하는 경우:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 음수의 제곱근: `sqrt(-1.0)`
- 음수의 로그: `log(-1.0)`
- 기존 `nan`이 포함된 모든 산술 연산

탐지:

```python
import math

math.isnan(x)       # x가 nan이면 True
math.isinf(x)       # x가 +inf 또는 -inf이면 True
math.isfinite(x)    # x가 nan 또는 inf가 아니면 True
```

예방 전략:

1. `exp()` 입력값 제한: `exp(clamp(x, -80, 80))`
2. 분모에 엡실론 추가: `x / (y + 1e-8)`
3. `log()` 내부에 엡실론 추가: `log(x + 1e-8)`
4. 안정적인 구현 사용 (log-sum-exp, 안정적인 소프트맥스)
5. 그래디언트 클리핑으로 가중치 폭발 방지
6. 디버깅 시 매 순전파 후 `nan`/`inf` 확인

### 수치적 그래디언트 검증

역전파(backpropagation)로 계산한 해석적 그래디언트(analytical gradient)에 버그가 있을 수 있습니다. 수치적 그래디언트 검증은 유한 차분(finite differences)으로 그래디언트를 계산하여 검증합니다.

중심 차분 공식:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

이는 O(h^2) 정확도로, `(f(x+h) - f(x)) / h`의 O(h) 정확도보다 훨씬 좋습니다.

h 선택: 너무 크면 근사치가 틀리고, 너무 작으면 치명적 상쇄로 결과가 파괴됩니다. 일반적으로 `h = 1e-5`에서 `1e-7`을 사용합니다.

검증: 해석적 그래디언트와 수치적 그래디언트의 상대 오차를 계산합니다.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

경험적 규칙:
- `relative_error < 1e-7`: 완벽, 그래디언트 정확
- `relative_error < 1e-5`: 허용 가능, 아마도 정확
- `relative_error > 1e-3`: 문제 있음
- `relative_error > 1`: 그래디언트 완전히 잘못됨

새로운 레이어 또는 손실 함수를 구현할 때는 항상 그래디언트를 검증하세요. PyTorch는 `torch.autograd.gradcheck()`를 제공합니다.

### 혼합 정밀도 훈련(Mixed Precision Training)

현대 GPU는 float16 행렬 곱셈을 float32보다 2-8배 빠르게 계산하는 전용 하드웨어(Tensor Cores)를 갖추고 있습니다. 혼합 정밀도 훈련은 이를 활용합니다:

```
1. float32로 가중치 마스터 복사본 유지
2. float16으로 순전파 (빠름)
3. float32로 손실 계산 (오버플로우 방지)
4. float16으로 역전파 (빠름)
5. 그래디언트를 float32로 스케일링
6. float32 마스터 가중치 업데이트
```

순수 float16 훈련의 문제점: 그래디언트는 종종 매우 작습니다(1e-8 이하). float16은 ~6e-8 미만의 값을 0으로 언더플로우시킵니다. 모든 그래디언트 업데이트가 0이 되어 모델이 학습하지 않습니다.

해결 방법: 손실 스케일링(loss scaling)

```
1. 손실에 큰 스케일 팩터(예: 1024) 곱하기
2. 역전파에서 (손실 * 1024)의 그래디언트 계산
3. 모든 그래디언트가 1024배 커져 float16 언더플로우 방지
4. 가중치 업데이트 전 그래디언트를 1024로 나누기
5. 순효과: 동일한 업데이트, 언더플로우 없음
```

동적 손실 스케일링은 스케일 팩터를 자동으로 조정합니다. 큰 값(65536)으로 시작합니다. 그래디언트가 `inf`로 오버플로우되면 반으로 줄입니다. N단계 동안 오버플로우가 없으면 두 배로 늘립니다.

### bfloat16 vs float16: 훈련에 bfloat16이 더 나은 이유

```
float16:   [1 부호] [5 지수]  [10 가수]
bfloat16:  [1 부호] [8 지수]  [7 가수]
```

float16은 더 높은 정밀도(10비트 가수 vs 7비트)를 제공하지만, 범위가 제한적입니다(최대 ~65,504). bfloat16은 정밀도는 낮지만 float32와 동일한 범위를 가집니다(최대 ~3.4e38).

신경망 훈련에서:

- 활성화값(activations)과 로짓(logits)은 훈련 중 정기적으로 65,504를 초과합니다. float16은 오버플로우되지만, bfloat16은 처리 가능합니다.
- float16은 손실 스케일링이 필요하지만, bfloat16은 일반적으로 필요 없습니다. 범위가 그래디언트 크기 스펙트럼을 커버하기 때문입니다.
- bfloat16은 float32의 단순 절단입니다: 가수의 하위 16비트를 버립니다. 변환은 간단하고 지수 부분은 손실되지 않습니다.

float16은 값이 유계(bounded)이고 정밀도가 더 중요한 추론(inference)에 선호됩니다. bfloat16은 범위가 더 중요한 훈련(training)에 선호됩니다. TPU와 현대 NVIDIA GPU(A100, H100)가 네이티브 bfloat16 지원을 제공하는 이유입니다.

### 그래디언트 클리핑(Gradient Clipping)

폭발하는 그래디언트(exploding gradients)는 그래디언트가 많은 레이어를 통해 기하급수적으로 증가할 때 발생합니다(RNN, 심층 네트워크, 트랜스포머에서 흔함). 하나의 큰 그래디언트가 한 단계 내에 모든 가중치를 손상시킬 수 있습니다.

두 가지 클리핑 유형:

**값 클리핑(Clip by value):** 각 그래디언트 요소를 독립적으로 클램핑합니다.

```
grad = clamp(grad, -max_val, max_val)
```

간단하지만 그래디언트 벡터의 방향을 변경할 수 있습니다.

**노름 클리핑(Clip by norm):** 그래디언트 벡터의 노름이 임계값을 초과하지 않도록 전체 벡터를 스케일링합니다.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

그래디언트 방향을 보존합니다. `torch.nn.utils.clip_grad_norm_()`이 이 방법을 사용합니다. 표준적인 선택입니다.

일반적인 값: 트랜스포머에 `max_norm=1.0`, RL에 `max_norm=0.5`, 단순 네트워크에 `max_norm=5.0`.

그래디언트 클리핑은 임시 방편이 아닙니다. 안전 장치입니다. 이 없이는 하나의 이상치 배치(outlier batch)가 몇 주간의 훈련을 망칠 수 있습니다.

### 정규화 레이어(Normalization Layers)의 수치적 안정화 역할

배치 정규화(batch normalization), 레이어 정규화(layer normalization), RMS 정규화는 일반적으로 훈련 수렴을 돕는 정규화기로 소개됩니다. 또한 수치적 안정제 역할을 합니다.

정규화가 없으면 활성화값이 레이어를 통과할 때마다 지수적으로 증가하거나 감소할 수 있습니다:

```
레이어 1: 값 범위 [0, 1]
레이어 5: 값 범위 [0, 100]
레이어 10: 값 범위 [0, 10,000]
레이어 50: 값 범위 [0, inf]
```

정규화는 매 레이어에서 활성화값을 재중심화(recentering)하고 재스케일링(rescaling)합니다:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`(일반적으로 1e-5)은 모든 활성화값이 동일할 때 0으로 나누는 것을 방지합니다. 학습된 파라미터 `gamma`와 `beta`는 네트워크가 필요한 스케일을 복원할 수 있게 합니다.

이를 통해 네트워크 전체에서 값을 수치적으로 안전한 범위로 유지하여 순전파 시 오버플로우와 역전파 시 그래디언트 폭발을 방지합니다.

### 일반적인 ML 수치적 버그

**버그: 몇 에포크 후 손실이 NaN이 됨.**
원인: 로짓이 너무 커져 소프트맥스가 오버플로우됨. 또는 학습률이 너무 높아 가중치가 발산함.
해결: 안정적인 소프트맥스(최대값 뺄셈) 사용, 학습률 감소, 그래디언트 클리핑 추가.

**버그: 손실이 log(num_classes)에 멈춤.**
원인: 모델 출력이 균일한 확률에 가까움. 그래디언트가 소실되거나 모델이 전혀 학습하지 않음을 의미함.
해결: 데이터 레이블 확인, 손실 함수 검증, 죽은 ReLU 확인.

**버그: 검증 정확도가 예상보다 1-3% 낮음.**
원인: 적절한 손실 스케일링 없이 혼합 정밀도 사용. 그래디언트 언더플로우로 작은 업데이트가 무음하게 사라짐.
해결: 동적 손실 스케일링 활성화, 또는 bfloat16으로 전환.

**버그: 일부 레이어의 그래디언트 노름이 0.0.**
원인: 죽은 ReLU 뉴런(모든 입력이 음수), 또는 float16 언더플로우.
해결: LeakyReLU 또는 GELU 사용, 그래디언트 스케일링 사용, 가중치 초기화 확인.

**버그: 한 GPU에서는 모델이 작동하지만 다른 GPU에서는 결과가 다름.**
원인: 비결정적 부동소수점 누적 순서. GPU 병렬 감소(parallel reductions)가 하드웨어마다 다른 순서로 합산하며, 부동소수점 덧셈은 결합 법칙이 성립하지 않음.
해결: 작은 차이(1e-6) 수용, 또는 `torch.use_deterministic_algorithms(True)` 설정 및 속도 저하 수용.

**버그: 손실 계산에서 `exp()`가 `inf`를 반환함.**
원인: 최대값-뺄셈 트릭 없이 `exp()`에 원시 로짓 전달.
해결: 내부적으로 log-sum-exp를 구현하는 `torch.nn.functional.log_softmax()` 사용.

**버그: float32에서 float16으로 전환 후 훈련 발산.**
원인: float16은 6e-8 미만의 그래디언트 크기나 65,504 초과의 활성화값을 표현할 수 없음.
해결: 손실 스케일링과 함께 혼합 정밀도(AMP) 사용, 또는 bfloat16 사용.

## 구축 단계

### 1단계: 부동소수점 정밀도 한계 시연

```python
print("=== 부동소수점 정밀도 ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"차이: {(0.1 + 0.2) - 0.3:.2e}")
```

### 2단계: 순진(naive) vs 안정(stable) 소프트맥스 구현

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"순진:  {softmax_naive(safe_logits)}")
print(f"안정: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"안정: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits)는 [nan, nan, nan]을 반환
```

### 3단계: 안정 로그-합-지수(log-sum-exp) 구현

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"순진:  {logsumexp_naive(safe):.6f}")
print(f"안정: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"안정: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large)는 inf를 반환
```

### 4단계: 안정 교차 엔트로피 구현

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"순진:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"안정: {cross_entropy_stable(true_class, logits):.6f}")
```

### 5단계: 그래디언트 검증

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  파라미터 {i}: 해석적={a:.8f} 수치적={n:.8f} "
              f"상대오차={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 사용 방법

### 혼합 정밀도 시뮬레이션

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 그래디언트 클리핑

```python
import math

def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"원본 노름: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"클리핑된 노름:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"방향 유지: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 감지

```python
import math

def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

전체 구현 및 모든 에지 케이스 예시는 `code/numerical.py`를 참조하세요.

## Ship It

이 레슨은 다음을 생성합니다:
- `code/numerical.py`에 안정적인 소프트맥스(stable softmax), 로그-합-지수(log-sum-exp), 교차 엔트로피(cross-entropy), 그래디언트 체크(gradient checking), 혼합 정밀도 시뮬레이션(mixed precision simulation) 구현
- `outputs/prompt-numerical-debugger.md`를 통해 학습 중 NaN/Inf 및 수치 문제 진단

이러한 안정적인 구현은 Phase 3에서 훈련 루프(training loop)를 구축할 때와 Phase 4에서 어텐션 메커니즘(attention mechanisms)을 구현할 때 다시 등장합니다.

## 연습 문제

1. **파국적 상쇄(catastrophic cancellation).** [1000000.0, 1000001.0, 1000002.0]의 분산을 float32에서 순진한 공식 `E[x^2] - E[x]^2`로 계산하세요. 그런 다음 Welford의 온라인 알고리즘(online algorithm)을 사용하여 계산하세요. 실제 분산(0.6667) 대비 오차를 비교하세요.

2. **정밀도 탐색(precision hunt).** Python에서 `1.0 + x == 1.0`을 만족하는 가장 작은 양의 float32 값 `x`를 찾으세요. 이는 기계 엡실론(machine epsilon)입니다. `numpy.finfo(numpy.float32).eps`와 일치하는지 확인하세요.

3. **로그-합-지수(log-sum-exp) 경계 조건.** `logsumexp_stable` 함수를 다음 경우로 테스트하세요: (a) 모든 값이 동일한 경우, (b) 한 값이 나머지보다 훨씬 큰 경우, (c) 모든 값이 매우 음수인 경우(-1000). 순진한 버전이 실패하는 경우 정확한 결과를 제공하는지 확인하세요.

4. **신경망 레이어 기울기 확인(gradient checking).** 단일 선형 레이어 `y = Wx + b`와 그 해석적 역전파(analytical backward pass)를 구현하세요. 3x2 가중치 행렬에 대해 `numerical_gradient`를 사용하여 정확성을 검증하세요.

5. **손실 스케일링(loss scaling) 실험.** float16으로 훈련 시뮬레이션: [1e-9, 1e-3] 범위의 무작위 기울기를 생성하고 float16으로 변환한 후 0이 되는 비율을 측정하세요. 그런 다음 손실 스케일링(1024 곱하기)을 적용하고 float16으로 변환한 후 다시 스케일링하여 0이 되는 비율을 다시 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| IEEE 754 | "부동소수점 표준" | 이진 부동소수점 형식, 반올림 규칙, 특수 값(inf, nan)을 정의하는 국제 표준. 모든 현대 CPU와 GPU가 구현함. |
| 머신 엡실론(Machine epsilon) | "정밀도 한계" | 주어진 부동소수점 형식에서 1.0 + e != 1.0을 만족하는 가장 작은 값 e. float32의 경우 약 1.19e-7. |
| 치명적 상쇄(Catastrophic cancellation) | "뺄셈으로 인한 정밀도 손실" | 거의 같은 크기의 부동소수점 수를 뺄 때 유효숫자가 상쇄되고 반올림 오차가 결과를 지배하는 현상. |
| 오버플로우(Overflow) | "너무 큰 숫자" | 표현 가능한 최대값을 초과하여 결과가 inf가 되는 현상. exp(89)는 float32에서 오버플로우 발생. |
| 언더플로우(Underflow) | "너무 작은 숫자" | 표현 가능한 최소 양수보다 0에 가까워져 결과가 0.0이 되는 현상. exp(-104)는 float32에서 언더플로우 발생. |
| 로그-합-지수 트릭(Log-sum-exp trick) | "최댓값을 먼저 빼라" | exp(max(x))를 인수분해하여 log(sum(exp(x)))를 계산함으로써 오버플로우와 언더플로우를 방지하는 기법. 소프트맥스, 크로스엔트로피, 로그 확률 계산에 사용. |
| 안정적 소프트맥스(Stable softmax) | "폭발하지 않는 소프트맥스" | 지수 연산 전에 max(logits)를 빼는 방식. 수치적으로 동일한 결과를 제공하며 오버플로우 가능성 없음. |
| 그래디언트 검증(Gradient checking) | "역전파 검증" | 유한 차분법으로 구한 수치적 그래디언트와 역전파로 구한 해석적 그래디언트를 비교하여 구현 오류를 찾는 기법. |
| 혼합 정밀도(Mixed precision) | "float16 순전파, float32 역전파" | 속도 중심 연산에는 낮은 정밀도 부동소수점을, 수치적으로 민감한 연산에는 높은 정밀도 부동소수점을 사용하는 기법. 일반적으로 2-3배 속도 향상. |
| 손실 스케일링(Loss scaling) | "그래디언트 언더플로우 방지" | 역전파 전에 손실에 큰 상수를 곱하여 그래디언트가 float16 표현 범위 내에 머물게 한 후, 가중치 업데이트 전에 동일한 상수로 나눔. |
| bfloat16 | "브레인 부동소수점" | 8비트 지수(float32와 동일한 범위)와 7비트 가수(float16보다 낮은 정밀도)를 가진 구글의 16비트 형식. 학습에 선호됨. |
| 그래디언트 클리핑(Gradient clipping) | "그래디언트 노름 제한" | 그래디언트 벡터의 노름이 임계값을 넘지 않도록 스케일링. 폭발하는 그래디언트가 가중치를 망치는 것을 방지. |
| NaN | "숫자가 아님" | 정의되지 않은 연산(0/0, inf-inf, sqrt(-1))에서 발생하는 특수 부동소수점 값. 이후 모든 산술 연산에 전파됨. |
| Inf | "무한대" | 오버플로우 또는 0으로 나누기 연산에서 발생하는 특수 부동소수점 값. NaN을 생성할 수 있음(inf - inf, inf * 0). |
| 수치적 그래디언트(Numerical gradient) | "무식한 미분법" | f(x+h)와 f(x-h)를 평가한 후 2h로 나누어 미분을 근사하는 방법. 느리지만 검증에 신뢰할 수 있음.

## 추가 자료

- [부동소수점 산술에 대해 모든 컴퓨터 과학자가 알아야 할 것 (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 권위 있는 참고 자료, 밀도 있지만 완벽함
- [혼합 정밀도 훈련 (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- float16 훈련을 위한 손실 스케일링(loss scaling)을 소개한 NVIDIA 논문
- [AMP: 자동 혼합 정밀도 (PyTorch 문서)](https://pytorch.org/docs/stable/amp.html) -- PyTorch에서 혼합 정밀도(AMP) 실용 가이드
- [bfloat16 형식 (Google Cloud TPU 문서)](https://cloud.google.com/tpu/docs/bfloat16) -- Google이 TPU에 이 형식을 선택한 이유
- [카한 합산 (위키피디아)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 부동소수점 합산의 반올림 오차 감소 알고리즘