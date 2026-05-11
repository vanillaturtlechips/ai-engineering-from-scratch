# 위치 인코딩 — 사인곡선, RoPE, ALiBi

> 어텐션은 순열 불변(permutation-invariant)합니다. "The cat sat on the mat"와 "mat the on sat cat the"는 위치 신호 없이 동일한 출력을 생성합니다. 세 가지 알고리즘은 이를 해결합니다 — 각각 "위치"의 의미에 대해 다른 가정을 합니다.

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** 7단계 · 02 (셀프 어텐션), 7단계 · 03 (멀티 헤드 어텐션)
**소요 시간:** ~45분

## 문제

Scaled dot-product 어텐션은 순서(순서)를 무시합니다. 어텐션 행렬 `softmax(Q K^T / √d) V`는 쌍별 유사도(pairwise similarities)로부터 계산됩니다. 입력 `X`의 행을 섞으면 출력 행도 같은 방식으로 섞입니다. 어텐션 내부에서는 위치(position)를 전혀 고려하지 않습니다.

이는 단어 주머니(bag-of-words) 모델에서는 문제가 되지 않습니다. 하지만 언어, 코드, 오디오, 비디오 등 순서가 의미를 전달하는 모든 분야에서는 치명적입니다.

해결 방법은 임베딩에 위치 정보를 주입하는 것입니다. 세 가지 주요 접근 방식:

1. **절대 사인/코사인 임베딩(Absolute sinusoidal)** (Vaswani 2017). 임베딩에 위치의 `sin/cos` 값을 추가합니다. 간단하고 학습 가능한 매개변수가 없지만, 훈련된 길이 이상으로의 외삽(extrapolation) 성능이 떨어집니다.
2. **RoPE — 회전형 위치 임베딩(Rotary Position Embeddings)** (Su 2021). 위치에 비례하는 각도로 Q와 K 벡터를 회전시킵니다. *상대적* 위치 정보를 내적(dot product)에 직접 인코딩합니다. 2026년 현재 주류 방식입니다.
3. **ALiBi — 선형 편향 어텐션(Attention with Linear Biases)** (Press 2022). 임베딩을 완전히 생략하고, 거리에 기반한 헤드별 선형 페널티를 어텐션 점수에 추가합니다. 길이 외삽 성능이 우수합니다.

2026년 기준으로, 거의 모든 최첨단 오픈 모델은 RoPE를 사용합니다: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. 일부 장문 컨텍스트(long-context) 모델은 ALiBi 또는 그 변형 방식을 사용합니다. 절대 사인/코사인 임베딩은 역사적인 접근 방식입니다.

## 개념

![Sinusoidal 절대 vs RoPE 회전 vs ALiBi 거리 편향](../assets/positional-encoding.svg)

### 절대 사인파

`(max_len, d_model)` 형태의 고정 행렬 `PE`를 사전 계산:

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

그 후 어텐션 전에 `X' = X + PE[:N]`. 각 차원은 서로 다른 주파수의 사인파입니다. 모델은 위상 패턴에서 위치를 읽는 방법을 학습합니다. `max_len`을 넘어서면 실패: 모델이 0–2047 위치만 보고 2048 위치에서 어떤 일이 일어나는지 알 수 없습니다.

### RoPE

Q와 K 벡터(임베딩이 아님)를 회전. 차원 쌍 `(2i, 2i+1)`에 대해:

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 기본값
```

위치 `pos_k`에 동일한 회전을 키에도 적용. 내적 `q'_m · k'_n`은 `(m - n)`만의 함수가 됩니다. 즉: **회전 절대 위치를 기반으로 했지만 어텐션 점수는 상대적 거리에만 의존**합니다. 아름다운 트릭입니다.

RoPE 확장: `base`를 스케일링(NTK-aware, YaRN, LongRoPE)하여 재학습 없이 더 긴 컨텍스트로 외삽할 수 있습니다. Llama 3는 이 방식으로 8K에서 128K 컨텍스트로 확장했습니다.

### ALiBi

임베딩 트릭을 건너뜁니다. 어텐션 점수에 직접 편향을 적용:

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

여기서 `m_h`는 헤드별 기울기(예: `1 / 2^(8·h/H)`). 가까운 토큰은 강화, 먼 토큰은 페널티. 학습 시 추가 비용 없음. 논문에서는 길이 외삽이 사인파를 능가하고 RoPE의 원래 학습 길이에서 RoPE와 동등함을 보여줍니다.

### 2026년에 선택할 것

| 변형 | 외삽 | 학습 비용 | 사용처 |
|---------|---------------|---------------|---------|
| 절대 사인파 | 낮음 | 무료 | 원본 트랜스포머, 초기 BERT |
| 학습된 절대 | 없음 | 미미 | GPT-2, GPT-3 |
| RoPE | 스케일링 시 우수 | 무료 | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | 우수 | 파인튜닝 단계 | Qwen2-1M, Llama 3.1 128K |
| ALiBi | 우수 | 무료 | BLOOM, MPT, Baichuan |

RoPE는 아키텍처를 변경하지 않고 어텐션에 통합되며, 상대적 위치를 인코딩하고, `base` 하이퍼파라미터가 긴 컨텍스트 파인튜닝을 위한 깔끔한 조절 장치를 제공하기 때문에 승리했습니다.

## 구축 방법

### 1단계: 사인파 인코딩

`code/main.py` 참조. 4줄 계산:

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

첫 번째 어텐션(attention) 레이어 이전 임베딩 행렬에 추가합니다.

### 2단계: Q, K에 RoPE 적용

RoPE는 Q와 K에 인-플레이스로 작동합니다. 각 차원 쌍에 대해:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

중요: 위치 `m`의 Q와 위치 `n`의 K에 동일한 함수를 적용합니다. 이들의 내적(dot product)은 모든 좌표 쌍에서 `cos((m-n)·θ_i)` 인자를 포착합니다. 어텐션은 상대 위치를 자동으로 학습합니다.

### 3단계: ALiBi 기울기 및 편향

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # 소프트맥스 전 어텐션 점수에 추가
```

헤드 `h`의 `(seq_len, seq_len)` 어텐션 점수 행렬에 `bias[h]`를 추가한 후 소프트맥스를 적용합니다.

### 4단계: RoPE의 상대 거리 특성 검증

두 무작위 벡터 `a, b`를 선택합니다. `(pos_a, pos_b)`로 회전한 후 `(pos_a + k, pos_b + k)`로 다시 회전합니다. 두 내적 값은 부동소수점 오차 범위 내에서 일치해야 합니다. 이 특성이 RoPE의 핵심 - 절대 오프셋에 불변하며 상대 간격만 고려합니다.

## 사용 방법

PyTorch 2.5+는 `torch.nn.functional`에 RoPE 유틸리티를 포함합니다. 대부분의 프로덕션 코드는 `flash_attn` 또는 `xformers`를 사용하며, 여기서 RoPE는 어텐션 커널 내부에 적용됩니다.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026년 장문맥 처리 기법:**

- **NTK-aware 보간법.** 4K에서 16K+로 확장할 때 `base`를 `base * (scale_factor)^(d/(d-2))`로 재조정합니다.
- **YaRN.** 장문맥에서 어텐션 엔트로피를 보존하는 더 스마트한 보간법. Llama 3.1 128K에서 사용됩니다.
- **LongRoPE.** Microsoft의 2024년 기법으로, 진화형 검색을 사용해 차원별 스케일 팩터를 선택합니다. Phi-3-Long에서 사용됩니다.
- **위치 보간 + 파인튜닝.** 확장 계수로 위치를 축소한 후 1–5B 토큰에 대해 파인튜닝합니다. 놀랍게도 효과적입니다.

## Ship It

`outputs/skill-positional-encoding-picker.md`를 참조하세요. 이 스킬은 목표 컨텍스트 길이, 외삽 필요성, 훈련 예산을 고려하여 새로운 모델에 대한 인코딩 전략을 선택합니다.

## 연습 문제

1. **쉬움.** `max_len=512, d=128`에 대해 사인파 `PE` 행렬을 히트맵으로 시각화하세요. "차원 인덱스가 커질수록 줄무늬가 넓어지는" 패턴을 확인하세요.
2. **중간.** NTK-aware RoPE 스케일링을 구현하세요. 길이 256의 시퀀스로 작은 언어 모델을 훈련한 후, 스케일링 적용 여부에 따라 길이 1024에서 테스트하세요. 퍼플렉서티(perplexity)를 측정하세요.
3. **어려움.** 동일한 어텐션 모듈에 ALiBi와 RoPE를 구현하세요. 길이 512의 시퀀스로 4층 트랜스포머를 복사 작업(copy task)에 훈련시킨 후, 테스트 시 2048로 외삽(extrapolate)하세요. 성능 저하를 비교하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 위치 인코딩(Positional encoding) | "어텐션에 순서를 알려준다" | 임베딩 또는 어텐션에 추가되는 위치 정보를 인코딩하는 모든 신호. |
| 사인/코사인(Sinusoidal) | "원래 방식" | 기하학적 주파수의 `sin/cos`를 임베딩에 추가; 외삽(extrapolation) 불가. |
| RoPE(Rotary Position Embedding) | "회전 임베딩" | 위치 의존적 각도로 Q, K를 회전; 내적(dot product)이 상대적 거리를 인코딩. |
| ALiBi(Attention with Linear Biases) | "선형 편향 트릭" | 어텐션 점수에 `-m·|i-j|` 추가; 임베딩 불필요, 외삽 성능 우수. |
| 베이스(base) | "RoPE의 조절기" | RoPE의 주파수 스케일러; 추론 시 컨텍스트 확장을 위해 증가시킴. |
| NTK-aware | "RoPE 스케일링 트릭" | 컨텍스트 확장 시 고주파 차원이 압축되지 않도록 `base`를 재조정. |
| YaRN | "고급 방식" | 어텐션 엔트로피를 보존하는 차원별 보간+외삽 기법. |
| 외삽(Extrapolation) | "학습 길이 이상에서도 작동" | 위치 체계가 훈련 시 본 `max_len`을 넘어 올바른 출력을 제공할 수 있는가? |

## 추가 자료

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 원본 사인 곡형(sinusoidal) 위치 임베딩.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE(회전 위치 임베딩) 논문.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi(선형 편향 어텐션).
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 최신 RoPE 확장 기법.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta의 Llama 2 장문 처리 논문.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long에서 사용된 Microsoft의 200만 토큰 이상 확장 방법.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 모든 RoPE 확장 기법(기본, 선형, 동적, YaRN, LongRoPE, Llama-3)의 프로덕션급 구현체.