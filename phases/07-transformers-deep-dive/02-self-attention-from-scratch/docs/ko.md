# Self-Attention from Scratch

> 어텐션은 모든 단어가 "누가 나에게 중요한가?"라고 묻고 그 답을 학습하는 룩업 테이블입니다.

**유형:** 구현
**언어:** Python
**선수 지식:** Phase 3 (딥러닝 코어), Phase 5 Lesson 10 (시퀀스-투-시퀀스)
**소요 시간:** ~90분

## 학습 목표

- NumPy만 사용하여 쿼리/키/값 투영과 소프트맥스 가중치 합을 포함한 스케일드 닷-프로덕트 자기 어텐션(scaled dot-product self-attention)을 처음부터 구현
- 헤드를 분할하고 병렬 어텐션을 계산한 후 결과를 연결하는 멀티-헤드 어텐션(multi-head attention) 레이어 구축
- 어텐션 행렬(attention matrix)이 토큰 간 관계를 포착하는 과정을 추적하고, sqrt(d_k)로 스케일링하는 것이 소프트맥스 포화(softmax saturation)를 방지하는 이유 설명
- 양방향 어텐션(bidirectional attention)을 자기회귀적(디코더 스타일) 어텐션(autoregressive attention)으로 변환하기 위해 인과적 마스킹(causal masking) 적용

## 문제

RNN은 시퀀스를 한 번에 하나의 토큰씩 처리합니다. 50번째 토큰에 도달할 때쯤이면 1번째 토큰의 정보는 50번의 압축 단계를 거치게 됩니다. 장거리 의존성은 고정 크기의 은닉 상태(hidden state)로 압축되는데, 이는 LSTM 게이팅으로도 완전히 해결되지 않는 병목 현상입니다.

2014년 Bahdanau의 어텐션(attention) 논문은 해결책을 제시했습니다: 디코더가 모든 인코더 위치를 다시 살펴보고 현재 단계에 중요한 위치를 결정하도록 하는 것입니다. 하지만 이는 여전히 RNN에 추가된 구조였습니다. 2017년 "Attention Is All You Need" 논문은 더 근본적인 질문을 던졌습니다: 어텐션이 *유일한* 메커니즘이라면 어떨까? 순환(recurrence)도 없고, 합성곱(convolution)도 없이 오직 어텐션만 있다면?

셀프 어텐션(self-attention)은 시퀀스의 모든 위치가 다른 모든 위치에 병렬적으로 주의를 기울일 수 있게 합니다. 이것이 트랜스포머(transformer)를 빠르고 확장 가능하며 지배적인 모델로 만든 핵심 요소입니다.

## 개념

### 데이터베이스 조회 비유

어텐션(attention)을 소프트 데이터베이스 조회로 생각해 보세요:

```
기존 데이터베이스:
  쿼리: "프랑스의 수도"  -->  정확한 일치  -->  "파리"

어텐션:
  쿼리: "프랑스의 수도"  -->  모든 키와의 유사도  -->  모든 값의 가중치 혼합
```

모든 토큰은 세 가지 벡터를 생성합니다:
- **쿼리(Q)**: "내가 찾는 것은 무엇인가?"
- **키(K)**: "내가 포함하는 것은 무엇인가?"
- **값(V)**: "선택되면 제공할 정보는 무엇인가?"

쿼리와 모든 키 간의 내적(dot product)은 어텐션 점수를 생성합니다. 높은 점수는 "이 키가 내 쿼리와 일치한다"는 의미입니다. 이 점수는 값에 가중치를 부여합니다. 출력은 값의 가중치 합입니다.

### Q, K, V 계산

각 토큰 임베딩은 세 개의 학습된 가중치 행렬을 통해 투영됩니다:

```
입력 임베딩 (n개 토큰 시퀀스, 각각 d차원):

  X = [x1, x2, x3, ..., xn]       형태: (n, d)

세 가중치 행렬:

  Wq  형태: (d, dk)
  Wk  형태: (d, dk)
  Wv  형태: (d, dv)

투영:

  Q = X @ Wq    형태: (n, dk)      각 토큰의 쿼리
  K = X @ Wk    형태: (n, dk)      각 토큰의 키
  V = X @ Wv    형태: (n, dv)      각 토큰의 값
```

시각적으로, 하나의 토큰에 대해:

```
             Wq
  x_i ------[*]------> q_i    "내가 찾는 것은 무엇인가?"
       |
       |     Wk
       +----[*]------> k_i    "내가 포함하는 것은 무엇인가?"
       |
       |     Wv
       +----[*]------> v_i    "내가 제공할 것은 무엇인가?"
```

### 어텐션 행렬

모든 토큰에 대한 Q, K, V를 얻은 후, 어텐션 점수는 행렬을 형성합니다:

```
점수 = Q @ K^T    형태: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- q1이 각 키에 얼마나 주목하는가
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

각 행: 전체 시퀀스에 대한 하나의 토큰 어텐션
```

### 왜 스케일링하는가?

내적(dot product)은 차원 dk에 따라 증가합니다. dk = 64일 때, 내적 값은 수십 범위에 있을 수 있으며, 이는 소프트맥스(softmax)를 기울기가 소실되는 영역으로 밀어넣습니다. 해결책: sqrt(dk)로 나눕니다.

```
스케일링된 점수 = (Q @ K^T) / sqrt(dk)
```

이는 소프트맥스가 유용한 기울기를 생성하는 범위 내에서 값을 유지합니다.

### 소프트맥스가 점수를 가중치로 변환

소프트맥스는 원시 점수를 각 행에 대한 확률 분포로 변환합니다:

```
q1의 원시 점수:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         소프트맥스
                            |
어텐션 가중치:   [0.52, 0.09, 0.07, 0.14, 0.08]   (합 ~1.0)
```

이제 각 토큰은 다른 모든 토큰에 얼마나 주목할지 나타내는 가중치 집합을 가집니다.

### 값의 가중치 합

각 토큰의 최종 출력은 모든 값 벡터의 가중치 합입니다:

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

토큰 1의 경우:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### 전체 파이프라인

```
                    +-------+
  X (입력)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (입력)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (입력)  ----->|  @ Wv  |-----> V ---------->| 가중치 합 |----> 출력
                    +-------+          ^          |          |
                                       |          +----------+
                              +--------+--------+
                              |    소프트맥스    |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

한 줄 공식:

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## 구축 방법

### 1단계: 소프트맥스(softmax) 직접 구현

소프트맥스는 원시 로짓(logits)을 확률로 변환합니다. 수치적 안정성을 위해 최댓값을 뺍니다.

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### 2단계: 스케일드 닷-프로덕트 어텐션(scaled dot-product attention)

핵심 함수입니다. Q, K, V 행렬을 입력으로 받아 어텐션 출력과 가중치 행렬을 반환합니다.

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### 3단계: 학습된 투영을 포함한 셀프 어텐션(self-attention) 클래스

Wq, Wk, Wv 가중치 행렬이 Xavier 유사 스케일링으로 초기화된 완전한 셀프 어텐션 모듈입니다.

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### 4단계: 문장에 적용 실행

문장에 대한 가짜 임베딩을 생성하고 어텐션 가중치를 확인합니다.

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("어텐션 가중치 (각 행: 해당 토큰이 주목하는 위치):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### 5단계: ASCII 히트맵으로 어텐션 시각화

어텐션 가중치를 문자로 매핑하여 빠르게 시각화합니다.

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## 사용 방법

PyTorch의 `nn.MultiheadAttention`은 우리가 구현한 기능에 멀티 헤드 분할과 출력 투영을 추가한 것입니다:

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

주요 차이점: 멀티 헤드 어텐션은 여러 어텐션 함수를 병렬로 실행하며, 각각 dk = d_model / n_heads 크기의 고유한 Q, K, V 투영을 사용한 후 결과를 연결(concatenate)합니다. 이를 통해 모델이 동시에 다양한 관계 유형에 어텐션할 수 있게 됩니다.

## Ship It

이 레슨은 다음을 생성합니다:
- `outputs/prompt-attention-explainer.md` - 데이터베이스 조회 비유를 통한 어텐션(attention) 설명용 프롬프트

## 연습 문제

1. `scaled_dot_product_attention`을 수정하여 소프트맥스(softmax) 이전에 특정 위치를 음의 무한대로 설정하는 선택적 마스크 행렬(mask matrix)을 지원하도록 구현하세요 (이것이 인과적/디코더 마스킹(causal/decoder masking)의 작동 방식입니다)

2. 멀티헤드 어텐션(multi-head attention)을 처음부터 구현하세요: Q, K, V를 `n_heads` 개의 청크로 분할하고, 각각에 대해 어텐션을 실행한 후 연결(concatenate)하고, 최종 가중치 행렬 Wo를 통해 투영(project)하세요

3. 길이가 같은 두 개의 서로 다른 문장을 동일한 SelfAttention 인스턴스에 입력하고 어텐션 패턴을 비교하세요. 어떤 부분이 달라지나요? 어떤 부분이 동일하게 유지되나요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| Query (Q) | "질문 벡터" | 입력이 찾는 정보를 나타내는 학습된 투영(learned projection) |
| Key (K) | "라벨 벡터" | 쿼리와 대조되는, 해당 토큰이 포함하는 정보를 나타내는 학습된 투영 |
| Value (V) | "콘텐츠 벡터" | 어텐션 점수에 기반해 집계되는 실제 정보를 운반하는 학습된 투영 |
| Scaled dot-product attention | "어텐션 공식" | softmax(QK^T / sqrt(dk)) @ V - 고차원에서 softmax 포화(saturation) 방지 |
| Self-attention | "토큰이 자기 자신과 다른 토큰을 바라봄" | Q, K, V가 모두 동일한 시퀀스에서 나오는 어텐션, 모든 위치가 다른 모든 위치를 참조할 수 있음 |
| Attention weights | "집중 정도" | 스케일링된 내적(dot product)에 대한 softmax로 생성된 위치에 대한 확률 분포 |
| Multi-head attention | "병렬 어텐션" | 서로 다른 투영으로 여러 어텐션 함수를 실행한 후 결과를 결합해 더 풍부한 표현 생성 |

## 추가 학습 자료

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - 원본 트랜스포머 논문
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - 전체 아키텍처를 가장 잘 시각화한 설명
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - 설명이 포함된 라인별 PyTorch 구현