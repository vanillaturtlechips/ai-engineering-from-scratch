# 어텐션 메커니즘 — 획기적인 발전

> 디코더는 압축된 요약을 흘끗 보는 것을 멈추고 전체 소스를 살펴보기 시작합니다. 이후의 모든 것은 어텐션(attention) + 엔지니어링입니다.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** 5단계 · 09 (시퀀스-투-시퀀스 모델)
**소요 시간:** ~45분

## 문제

레슨 09는 측정 가능한 실패로 끝났습니다. 장난감 복사 작업에 훈련된 GRU 인코더-디코더는 길이 5에서 89% 정확도를 보이지만 길이 80에서는 무작위 추측 수준에 가깝습니다. 이 문제는 구조적 문제이며 훈련 버그가 아닙니다. 인코더가 추출한 모든 정보는 고정된 크기의 하나의 은닉 상태에 맞아야 하며, 디코더는 그 외의 다른 정보를 전혀 보지 못합니다.

Bahdanau, Cho, Bengio는 2014년에 세 줄짜리 해결책을 발표했습니다. 디코더에 최종 인코더 상태만 제공하는 대신 모든 인코더 상태를 유지합니다. 각 디코더 단계에서 "디코더가 현재 인코더 위치 `i`를 얼마나 주의 깊게 봐야 하는가?"를 나타내는 가중치를 사용해 인코더 상태들의 가중 평균을 계산합니다. 이 가중 평균이 컨텍스트(context)이며, 매 디코더 단계마다 변경됩니다.

이것이 전체 아이디어입니다. 트랜스포머는 이를 확장했습니다. 셀프 어텐션(self-attention)은 단일 시퀀스에 적용했고, 멀티헤드 어텐션(multi-head attention)은 병렬로 실행했습니다. 하지만 2014년 버전만으로도 병목 현상을 해결했으며, 일단 이를 구현하면 트랜스포머로의 전환은 개념적 변화가 아닌 엔지니어링 문제입니다.

## 개념

![Bahdanau 어텐션: 디코더가 모든 인코더 상태에 쿼리](../assets/attention.svg)

각 디코더 단계 `t`에서:

1. 이전 디코더 은닉 상태 `s_{t-1}`을 **쿼리**로 사용합니다.
2. 모든 인코더 은닉 상태 `h_1, ..., h_T`와 점수를 매깁니다. 인코더 위치당 하나의 스칼라 값입니다.
3. 점수를 소프트맥스하여 합이 1이 되는 어텐션 가중치 `α_{t,1}, ..., α_{t,T}`를 얻습니다.
4. 컨텍스트 벡터 `c_t = Σ α_{t,i} * h_i`. 인코더 상태들의 가중 평균입니다.
5. 디코더는 `c_t`와 이전 출력 토큰을 입력으로 받아 다음 토큰을 생성합니다.

가중 평균이 핵심입니다. 디코더가 "Je"를 "I"로 번역할 때는 "Je" 위의 인코더 상태 가중치를 높게, 나머지는 낮게 설정합니다. "not"이 필요할 때는 "pas" 가중치를 높게 설정합니다. 컨텍스트 벡터는 각 단계를 재구성합니다.

## 형태(shape) (모두를 괴롭히는 그 것)

모든 주의(attention) 구현이 처음 실패할 때 문제가 되는 부분입니다. 천천히 읽으세요.

| 항목 | 형태(shape) | 비고 |
|-------|-------|-------|
| 인코더 은닉 상태 `H` | `(T_enc, d_h)` | BiLSTM인 경우 `d_h = 2 * d_hidden` |
| 디코더 은닉 상태 `s_{t-1}` | `(d_s,)` | 하나의 벡터 |
| 주의 점수 `e_{t,i}` | 스칼라 | 인코더 위치마다 하나씩 |
| 주의 가중치 `α_{t,i}` | 스칼라 | 모든 `i`에 대한 소프트맥스 적용 후 |
| 컨텍스트 벡터 `c_t` | `(d_h,)` | 인코더 상태와 동일한 형태 |

**Bahdanau (가법) 점수.** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}`의 형태는 `(d_s,)`, `h_i`의 형태는 `(d_h,)`입니다.
- `W_a`의 형태는 `(d_attn, d_s)`입니다. `U_a`의 형태는 `(d_attn, d_h)`입니다.
- tanh 내부의 합은 `(d_attn,)` 형태를 가집니다.
- `v_α`의 형태는 `(d_attn,)`입니다. `v_α`와의 내적은 스칼라로 축소됩니다. **이것이 `v_α`의 역할입니다.** 마법이 아닙니다. 주의 차원 벡터를 스칼라 점수로 변환하는 투영입니다.

**Luong (승법) 점수.** 세 가지 변형:

- `dot`: `e_{t,i} = s_t^T * h_i`. `d_s == d_h`가 필요합니다. 엄격한 제약 조건입니다. 인코더가 양방향이면 건너뜁니다.
- `general`: `e_{t,i} = s_t^T * W * h_i`이며 `W`의 형태는 `(d_s, d_h)`입니다. 차원 일치 제약을 제거합니다.
- `concat`: 본질적으로 Bahdanau 형태입니다. 처음 두 가지가 더 저렴하기 때문에 거의 사용되지 않습니다.

**Bahdanau / Luong 주의점 하나.** Bahdanau는 `s_{t-1}`(현재 단어 생성 *전* 디코더 상태)를 사용합니다. Luong은 `s_t`(생성 *후* 상태)를 사용합니다. 둘을 혼동하면 디버깅이 극히 어려운 미묘한 기울기 오류가 발생합니다. 한 논문을 선택하고 그 관례를 따르세요.

## 구축

### 단계 1: additive (Bahdanau) 어텐션

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

위 표와 함께 형태를 확인하세요. `encoder_states`는 형태 `(T_enc, d_h)`를 가집니다. `projected_enc`는 형태 `(T_enc, d_attn)`을 가집니다. `projected_dec`는 형태 `(d_attn,)`을 가지며 브로드캐스트됩니다. `combined`는 형태 `(T_enc, d_attn)`을 가집니다. `scores`는 형태 `(T_enc,)`를 가집니다. `weights`는 형태 `(T_enc,)`를 가집니다. `context`는 형태 `(d_h,)`를 가집니다. 이제 실행하세요.

### 단계 2: Luong dot 및 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

각각 3줄입니다. 이것이 Luong의 논문이 주목받은 이유입니다. 대부분의 작업에서 동일한 정확도를 유지하면서 훨씬 적은 코드입니다.

### 단계 3: 수치 예제

"cat", "sat", "mat"에 해당하는 세 개의 인코더 상태와 첫 번째와 가장 잘 정렬되는 디코더 상태가 주어졌을 때, 어텐션 분포는 위치 0에 집중됩니다. 디코더 상태가 마지막과 정렬되도록 이동하면 어텐션은 위치 2로 이동합니다. 컨텍스트 벡터가 이를 추적합니다.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

첫 번째 행이 승리합니다. 그런 다음 디코더 상태를 세 번째 인코더 상태에 더 가깝게 이동시키고 가중치가 이동하는 것을 관찰하세요. 이것이 전부입니다. 어텐션은 명시적인 정렬입니다.

### 단계 4: 이것이 트랜스포머로 가는 다리인 이유

위의 내용을 Q/K/V로 번역해 보겠습니다:

- **쿼리(Query)** = 디코더 상태 `s_{t-1}`
- **키(Key)** = 인코더 상태 (점수를 매기는 대상)
- **값(Value)** = 인코더 상태 (가중치를 적용하고 합산하는 대상)

고전적인 어텐션에서는 키와 값이 동일한 것입니다. 셀프 어텐션은 이를 분리합니다: 시퀀스를 자신에 대해 쿼리할 수 있으며, K와 V에 대해 서로 다른 학습된 투영을 사용합니다. 멀티헤드 어텐션은 서로 다른 학습된 투영으로 병렬로 실행합니다. 트랜스포머는 전체 단계를 여러 번 쌓고 RNN을 제거합니다.

수학은 동일합니다. 형태도 동일합니다. Bahdanau 어텐션에서 스케일링된 닷-프로덕트 어텐션으로의 교육적 도약은 대부분 표기법 차이일 뿐입니다.

## 사용 방법

PyTorch와 TensorFlow는 어텐션(attention)을 직접 제공합니다.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

이것은 트랜스포머(Transformer) 어텐션 레이어입니다. 5개 위치의 쿼리 배치, 10개 위치의 키/값 배치, 각각 128차원, 8개의 헤드를 가집니다. `output`은 새로운 컨텍스트가 보강된 쿼리입니다. `weights`는 시각화할 수 있는 5x10 정렬 행렬입니다.

### 고전적인 어텐션이 여전히 중요한 경우

- 교육(Pedagogy). 단일 헤드, 단일 레이어, RNN 기반 버전은 모든 개념을 가시적으로 만듭니다.
- 트랜스포머가 적합하지 않은 온디바이스(on-device) 시퀀스 작업.
- 2014-2017년 사이의 모든 논문. Bahdanau의 관례를 모르면 잘못 읽을 수 있습니다.
- 기계 번역(MT)에서의 세밀한 정렬 분석. 원시 어텐션 가중치는 트랜스포머 모델에서도 해석 가능성 도구이며, 이를 읽으려면 그것이 무엇인지 알아야 합니다.

### 어텐션 가중치를 설명으로 사용하는 함정

어텐션 가중치는 해석 가능해 보입니다. 위치별로 합이 1이 되는 가중치이며, 플롯으로 시각화할 수 있고, 높은 값은 "이것을 주목했다"는 의미입니다. 리뷰어들은 이를 좋아합니다.

하지만 보이는 것만큼 해석 가능하지는 않습니다. Jain과 Wallace(2019)는 일부 작업에서 어텐션 분포를 치환하거나 임의의 대안으로 대체해도 모델 예측이 변하지 않음을 보였습니다. 제거 실험(ablation)이나 반사실적 검증(counterfactual check) 없이 어텐션 가중치를 추론 증거로 보고하지 마십시오.

## Ship It

`outputs/prompt-attention-shapes.md`로 저장:

```markdown
---
name: attention-shapes
description: 어텐션 구현에서 발생하는 형태(shape) 불일치를 디버깅합니다.
phase: 5
lesson: 10
---

깨진 어텐션 구현이 주어졌을 때, 형태 불일치 문제를 식별합니다. 다음 내용을 출력하세요:

1. 잘못된 형태를 가진 행렬. 텐서 이름을 명시하세요.
2. (d_s, d_h, d_attn, T_enc, T_dec, batch_size)로부터 유도된 올바른 형태.
3. 한 줄짜리 수정 방법. 전치(transpose), 형태 변경(reshape), 또는 프로젝션(project).
4. 회귀(regression)를 잡을 수 있는 테스트. 일반적으로: `assert output.shape == (batch, T_dec, d_h)` 및 `weights.shape == (batch, T_dec, T_enc)` 및 `weights.sum(dim=-1)`이 1에 근접하는지 확인.

묵묵히 브로드캐스팅하는 수정은 권장하지 마세요. 브로드캐스팅으로 숨겨진 버그는 나중에 정확도 저하로 나타나며, 이는 가장 심각한 어텐션 버그 유형입니다.

Bahdanau 어텐션의 경우, 디코더 입력이 `s_{t-1}`(이전 단계 상태)임을 강조하세요. Luong 어텐션의 경우 `s_t`(현재 단계 상태)임을 명시하세요. Dot-product 어텐션의 경우, 쿼리와 키 간 차원 불일치를 가장 흔한 첫 번째 오류로 표시하세요.
```

## 연습 문제

1. **쉬움.** 인코더의 패딩 토큰에 어텐션 가중치 0이 적용되도록 `softmax` 마스킹을 구현하세요. 가변 길이 시퀀스가 포함된 배치에서 테스트하세요.
2. **중간.** Luong `general` 형태에 멀티 헤드 어텐션을 추가하세요. `d_h`를 `n_heads` 그룹으로 분할하고, 각 헤드별로 어텐션을 실행한 후 연결(concatenate)하세요. 싱글 헤드 경우가 이전 구현과 일치하는지 검증하세요.
3. **어려움.** 레슨 09의 토이 복사(copy) 태스크에서 Bahdanau 어텐션을 적용한 GRU 인코더-디코더를 학습시키세요. 정확도 대 시퀀스 길이 그래프를 그리세요. 어텐션 미적용 베이스라인과 비교하세요. 길이가 증가함에 따라 성능 차이가 벌어지는 것을 확인할 수 있어야 하며, 이는 어텐션이 병목 현상을 해결함을 입증합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 어텐션(attention) | 사물을 보는 것 | 값 시퀀스의 가중 평균. 가중치는 쿼리-키 유사도로 계산됨. |
| 쿼리(Query), 키(Key), 값(Value) | QKV | 세 가지 투영: Q는 질문, K는 매칭 대상, V는 반환할 내용. |
| 가법 어텐션(additive attention) | 바다니우(Bahdanau) | 피드포워드 점수: `v^T tanh(W q + U k)`. |
| 승법 어텐션(multiplicative attention) | 룽(dot / general) | 점수는 `q^T k` 또는 `q^T W k`. 계산 비용이 저렴하며 대부분의 작업에서 동일한 정확도. |
| 정렬 행렬(alignment matrix) | 예쁜 그림 | 디코더 시간(T_dec)과 인코더 시간(T_enc) 격자로 표현된 어텐션 가중치. 모델이 어디에 집중했는지 확인 가능. |

## 추가 자료

- [Bahdanau, Cho, Bengio (2014). 정렬과 번역을 동시에 학습하는 신경망 기계 번역](https://arxiv.org/abs/1409.0473) — 원본 논문.
- [Luong, Pham, Manning (2015). 어텐션 기반 신경망 기계 번역의 효과적 접근법](https://arxiv.org/abs/1508.04025) — 세 가지 점수 변형 및 비교.
- [Jain and Wallace (2019). 어텐션은 설명이 아니다](https://arxiv.org/abs/1902.10186) — 해석 가능성에 대한 주의 사항.
- [Dive into Deep Learning — Bahdanau 어텐션](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — PyTorch를 활용한 실행 가능한 튜토리얼.