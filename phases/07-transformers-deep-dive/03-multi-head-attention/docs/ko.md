# 멀티 헤드 어텐션(Multi-Head Attention)

> 하나의 어텐션 헤드(attention head)는 한 번에 하나의 관계(relation)를 학습합니다. 8개의 헤드는 8개의 관계를 학습합니다. 헤드들은 자유롭습니다. 더 많은 헤드를 사용하세요.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** 7단계 · 02 (기초부터 시작하는 셀프 어텐션(Self-Attention from Scratch))
**소요 시간:** ~75분

## 문제

단일 자기 어텐션 헤드(self-attention head)는 하나의 어텐션 행렬(attention matrix)을 계산합니다. 이 행렬은 일반적으로 훈련 신호(training signal)의 손실(loss)을 최소화하는 하나의 관계 유형을 포착합니다. 데이터에 주어-동사 일치(subject-verb agreement), 공참조(co-reference), 장거리 담화(long-range discourse), 구문 청킹(syntactic chunking)이 모두 얽혀 있는 경우, 단일 헤드는 이들을 하나의 소프트맥스 분포(soft-max distribution)로 혼합하고 신호의 절반을 잃어버립니다.

2017년 Vaswani 논문의 해결책: 여러 어텐션 함수를 병렬로 실행하고, 각각 고유한 Q(Query), K(Key), V(Value) 투영(projection)을 사용한 후 출력을 연결(concatenate)합니다. 각 헤드는 `d_model / n_heads` 차원의 더 작은 부분 공간에서 작동합니다. 총 파라미터 수는 동일하게 유지되지만, 표현 능력(expressive power)은 증가합니다.

멀티헤드 어텐션(multi-head attention)은 2026년에 출시되는 모든 트랜스포머(transformer)의 기본 구성 요소입니다. 유일한 논점은 *헤드의 개수*와 키(key)와 값(value)이 투영을 공유하는지 여부(그룹드-쿼리 어텐션(Grouped-Query Attention), 멀티-쿼리 어텐션(Multi-Query Attention), 멀티헤드 잠재 어텐션(Multi-head Latent Attention))에 관한 것입니다.

## 개념

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**분할(Split).** 형태 `(N, d_model)`인 `X`를 가져옵니다. 각각 형태 `(N, d_model)`인 Q, K, V로 투영합니다. `d_head = d_model / n_heads`가 되도록 `(N, n_heads, d_head)`로 재구성합니다. 그리고 `(n_heads, N, d_head)`로 전치합니다.

**병렬 어텐션(Attend in parallel).** 각 헤드 내부에서 스케일링된 닷-프로덕트 어텐션(scaled dot-product attention)을 실행합니다. 각 헤드는 `(N, d_head)`를 출력합니다. 헤드들은 임베딩의 서로 다른 부분 공간에서 작동하며, 어텐션 계산 중에는 서로 통신하지 않습니다.

**결합 및 투영(Concatenate and project).** 헤드들을 다시 `(N, d_model)`로 쌓고, 형태 `(d_model, d_model)`인 학습된 출력 행렬 `W_o`와 곱합니다. `W_o`는 헤드들이 혼합되는 지점입니다.

**작동 원리(Why it works).** 각 헤드는 표현 예산(representational budget)을 놓고 다른 헤드들과 경쟁하지 않고 전문화될 수 있습니다. 2019–2024년의 프로브 연구(probing studies)는 뚜렷한 헤드 역할을 보여줍니다: 위치 헤드(positional heads), 이전 토큰에 어텐션하는 헤드, 복사 헤드(copy heads), 개체명 헤드(named-entity heads), 유도 헤드(induction heads, 인-컨텍스트 학습(in-context learning)의 기반).

**2026년 기준 변형(variants) 계보:**

| 변형(Variant) | Q 헤드 | K/V 헤드 | 사용 모델 |
|---------|---------|-----------|---------|
| 멀티헤드(MHA) | N | N | GPT-2, BERT, T5 |
| 멀티쿼리(MQA) | N | 1 | PaLM, Falcon |
| 그룹화 쿼리(GQA) | N | G (예: N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| 멀티헤드 잠재(MLA) | N | 저랭크(low-rank)로 압축 | DeepSeek-V2, V3 |

GQA는 KV-캐시 메모리를 `N/G` 비율로 줄이면서도 거의 완전한 품질을 유지하기 때문에 현대적인 기본값입니다. MLA는 K/V를 잠재 공간으로 압축한 후 계산 시간에 다시 투영하는 방식으로 더 나아가며, FLOPs 비용은 증가시키지만 메모리를 훨씬 더 절약합니다.

## 구축 방법

## 1단계: 기존 단일 헤드 어텐션에서 헤드 분할

Lesson 02의 `SelfAttention`을 가져와 분할/결합 레이어로 감싸세요. NumPy 구현은 `code/main.py`를 참조하세요. 로직은 다음과 같습니다:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (헤드, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

리쉐이프(reshape) 1회와 전치(transpose) 1회로 구성됩니다. 루프가 없습니다. 이는 PyTorch의 `nn.MultiheadAttention`이 내부적으로 수행하는 작업과 정확히 일치합니다.

## 2단계: 헤드별 스케일링된 점곱 어텐션 실행

각 헤드는 Q, K, V의 고유한 슬라이스를 받습니다. 어텐션은 배치 행렬 곱셈으로 처리됩니다:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (헤드, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (헤드, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

실제 하드웨어에서 `Qh @ Kh.transpose(...)`는 단일 `bmm`(배치 행렬 곱셈)입니다. GPU는 `(헤드, N, d_head) × (헤드, d_head, N) -> (헤드, N, N)` 형태의 단일 배치 행렬 곱셈으로 인식합니다. 헤드 추가는 추가 비용이 없습니다.

## 3단계: 그룹화 쿼리 어텐션(GQA) 변형

키와 값 프로젝션만 변경됩니다. Q는 `n_heads` 그룹을 받고, K와 V는 `n_kv_heads < n_heads` 그룹을 받은 후 반복되어 헤드 수를 맞춥니다:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_헤드, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_헤드, n, d_head)
```

추론 시 KV 캐시에 `n_kv_heads` 복사본만 유지되므로 메모리가 절약됩니다. Llama 3 70B는 64개의 쿼리 헤드와 8개의 KV 헤드를 사용하여 캐시 크기를 8배 줄입니다.

## 4단계: 각 헤드가 학습한 내용 분석

4개 헤드로 짧은 문장에 MHA를 실행하세요. 각 헤드에 대해 `(N, N)` 어텐션 행렬을 출력하면, 무작위 초기화 상태에서도 서로 다른 헤드가 다른 구조를 포착하는 것을 확인할 수 있습니다. 이는 부분적으로는 신호, 부분적으로는 부분 공간의 회전 대칭성 때문입니다.

## 사용 방법

PyTorch에서 한 줄 버전:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+의 GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention은 CUDA에서 Flash Attention을 자동 디스패치합니다.
# GQA의 경우, (B, n_heads, N, d_head) 형태의 Q와 (B, n_kv_heads, N, d_head) 형태의 K,V를 전달합니다. PyTorch가 반복을 처리합니다.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**헤드 수는 몇 개?** 2026년 프로덕션 모델들의 경험적 규칙:

| 모델 크기 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 소형 (~125M) | 768 | 12 | 64 |
| 베이스 (~350M) | 1024 | 16 | 64 |
| 대형 (~1B) | 2048 | 16 | 128 |
| 프론티어 (~70B) | 8192 | 64 | 128 |

`d_head`는 거의 항상 64 또는 128로 설정됩니다. 이는 하나의 헤드가 "볼 수 있는" 단위입니다. 32 미만으로 떨어지면 헤드들이 스케일링 팩터 `sqrt(d_head)`와 충돌하기 시작하고, 256을 넘으면 "많은 소규모 전문가" 이점을 잃게 됩니다.

## Ship It

`outputs/skill-mha-configurator.md`를 참조하세요. 이 스킬은 파라미터 예산, 시퀀스 길이, 배포 대상에 따라 새로운 트랜스포머에 대한 헤드 수(head count), 키-값 헤드 수(kv-head count), 투영 전략(projection strategy)을 추천합니다.

## 연습 문제

1. **쉬움.** `code/main.py`의 MHA(Multi-Head Attention)에서 `d_model=64`로 고정한 상태에서 `n_heads`를 1에서 16으로 변경해 보세요. 합성 복사 작업에서 1층짜리 작은 모델의 손실(loss)을 그래프로 그려보세요. 헤드(head)가 더 많아지면 도움이 되는지, 정체되는지, 아니면 성능이 떨어지는지 확인해 보세요.
2. **중간.** MQA(Multi-Query Attention)를 구현하세요(모든 쿼리 헤드(query head)가 하나의 KV 헤드를 공유). 전체 MHA 대비 파라미터 수가 얼마나 감소하는지 측정해 보세요. 추론 시 N=2048일 때 KV-캐시 크기가 얼마나 줄어드는지 계산해 보세요.
3. **어려움.** Multi-head Latent Attention의 작은 버전을 구현하세요: K,V를 랭크(rank)-`r` 잠재 공간으로 압축하고, KV-캐시에 잠재 공간을 저장한 후 어텐션(attention) 시간에 복원하세요. 검증 ppl(perplexity)이 1비트 이내를 유지하면서 캐시 메모리가 전체 MHA의 1/8 아래로 떨어지는 `r` 값은 얼마인가요?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 헤드(Head) | "단일 어텐션 회로" | 차원 `d_head = d_model / n_heads`를 가진 하나의 Q/K/V 투영과 자체 어텐션 행렬. |
| d_head | "헤드 차원" | 헤드당 은닉 너비; 프로덕션 환경에서는 거의 항상 64 또는 128. |
| 분할/결합(Split / combine) | "리쉐이프 트릭" | 어텐션 주변의 `(N, d_model) ↔ (n_heads, N, d_head)` 리쉐이프+전치. |
| W_o | "출력 투영" | 헤드 연결 후 적용되는 `(d_model, d_model)` 행렬; 헤드들이 혼합되는 지점. |
| MQA | "하나의 KV 헤드" | 멀티-쿼리 어텐션: 단일 공유 K/V 투영. 가장 작은 KV 캐시, 일부 품질 손실 발생. |
| GQA | "Llama 2 이후 기본값" | `n_kv_heads < n_heads`인 그룹화-쿼리 어텐션; Q와 일치하도록 반복. |
| MLA | "DeepSeek의 트릭" | 멀티-헤드 잠재 어텐션: K,V를 저랭크 잠재 공간으로 압축, 어텐션 시 복원. |
| 귀납 헤드(Induction head) | "인-컨텍스트 학습의 회로" | 이전 발생 사례를 감지하고 뒤따른 내용을 복사하는 한 쌍의 헤드.

## 추가 자료

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 원본 멀티 헤드 스펙.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 논문.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 학습 후 MHA를 GQA로 변환하는 방법.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA와 캐시 메모리에서 MHA/GQA를 능가하는 이유.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 헤드의 실제 동작을 메커니즘적으로 분석.