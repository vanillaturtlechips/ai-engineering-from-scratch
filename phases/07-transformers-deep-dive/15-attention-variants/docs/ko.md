# 어텐션 변형 — 슬라이딩 윈도우, 스파스, 디퍼렌셜

> 전체 어텐션은 원(circle)입니다. 모든 토큰이 모든 토큰을 보고, 메모리는 그 대가를 치릅니다. 네 가지 변형은 원의 형태를 구부려 비용의 절반을 회복합니다.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** 7단계 · 02 (셀프 어텐션), 7단계 · 03 (멀티 헤드), 7단계 · 12 (KV 캐시 / 플래시 어텐션)
**소요 시간:** ~60분

## 문제

전체 어텐션(full attention)은 시퀀스 길이에 대해 `O(N²)` 메모리와 `O(N²)` 계산 비용이 듭니다. 128K 컨텍스트를 가진 Llama 3 70B의 경우, 각 레이어당 160억 개의 어텐션 항목이 발생하며, 80개의 레이어를 곱하게 됩니다. Flash Attention(레슨 12)은 `O(N²)` 활성화 메모리를 숨기지만 산술적 비용은 변경하지 않습니다 — 모든 토큰은 여전히 다른 모든 토큰에 어텐션합니다.

어텐션 행렬 자체의 토폴로지를 변경하는 세 가지 범주의 변형(variant)이 있습니다:

1. **슬라이딩 윈도우 어텐션(SWA).** 각 토큰은 전체 접두사(prefix)가 아닌 고정된 이웃 윈도우에 어텐션합니다. 메모리와 계산 비용은 `O(N · W)`로 감소하며, 여기서 `W`는 윈도우 크기입니다. Gemma 2/3, Mistral 7B 초기 레이어, Phi-3-Long에서 사용됩니다.
2. **희소/블록 어텐션.** 선택된 `(i, j)` 쌍만 점수가 부여되며, 나머지는 가중치가 0으로 강제됩니다. Longformer, BigBird, OpenAI 희소 트랜스포머에서 사용됩니다.
3. **차등 어텐션.** 별도의 Q/K 투영으로 두 개의 어텐션 맵을 계산한 후 하나를 다른 것에서 뺍니다. 처음 몇 토큰으로 가중치가 유출되는 "어텐션 싱크"를 제거합니다. Microsoft의 DIFF 트랜스포머(2024)에서 제안되었습니다.

이들은 공존합니다. 2026년 프론티어 모델은 종종 이들을 혼합합니다: 대부분의 레이어는 SWA-1024를 사용하고, 5번째 레이어마다 글로벌 전체 어텐션을 적용하며, 소수는 검색(retrieval)을 정리하는 차등 헤드를 가집니다. Gemma 3의 5:1 SWA-대-글로벌 비율은 현재 교과서적 기본값입니다.

## 개념

### 슬라이딩 윈도우 어텐션(SWA)

각 위치 `i`의 쿼리는 `[i - W, i]`(인과적 SWA) 또는 `[i - W/2, i + W/2]`(양방향) 내의 위치만 어텐션합니다. 윈도우 밖의 토큰은 점수 행렬에서 `-inf`를 받습니다.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

`N = 8192` 및 `W = 1024`의 경우, 점수 행렬은 기대값으로 1024 × 8192개의 비제로 행을 가지며, 이는 8배 감소입니다.

**KV 캐시는 SWA로 축소됩니다.** 각 레이어당 K와 V의 마지막 `W` 토큰만 유지하면 됩니다. Gemma-3 구성(1024 윈도우, 128K 컨텍스트)의 경우 KV 캐시는 128배 감소합니다.

**품질 비용.** SWA 전용 트랜스포머는 장거리 검색에 어려움을 겪습니다. 해결책: SWA 레이어와 전체 어텐션 레이어를 교차 배치합니다. Gemma 3은 5:1 SWA:글로벌 비율을 사용합니다. Mistral 7B는 중첩 윈도우를 통해 정보가 "앞으로 흐르는" 인과적 SWA 스택을 사용했으며, 각 레이어는 유효 수용 필드를 `W`만큼 확장하고, `L` 레이어 후에는 모델이 `L × W` 토큰을 어텐션할 수 있습니다.

### 희소/블록 어텐션

미리 `N × N` 희소성 패턴을 선택합니다. 세 가지 표준 형태:

- **로컬 + 스트라이드(OpenAI 희소 트랜스포머).** 마지막 `W` 토큰과 그 이전의 `stride` 간격 토큰에 어텐션합니다. `O(N · sqrt(N))` 계산으로 로컬 및 장거리 정보를 포착합니다.
- **Longformer / BigBird.** 로컬 윈도우 + 모든 토큰에 어텐션하고 모든 토큰의 어텐션을 받는 소수의 글로벌 토큰(예: `[CLS]`) + 무작위 희소 연결. 동일한 품질에서 2배 컨텍스트를 달성합니다.
- **네이티브 희소 어텐션(DeepSeek, 2025).** `(Q, K)`의 어떤 블록이 중요한지 학습하고, 커널 수준에서 제로 블록을 건너뜁니다. FlashAttention과 호환됩니다.

희소 어텐션은 커널 엔지니어링 이야기입니다. 수학(점수 행렬 마스킹)은 간단하지만, 승리는 SRAM에 제로 항목을 로드하지 않는 데서 옵니다. FlashAttention-3과 2026 FlexAttention API는 PyTorch에서 사용자 정의 희소 패턴을 1급 객체로 만듭니다.

### 미분 어텐션(DIFF 트랜스포머, 2024)

일반 어텐션은 "어텐션 싱크" 문제가 있습니다: softmax는 모든 행의 합이 1이 되도록 강제하므로, 특정 토큰에 어텐션하고 싶지 않은 토큰은 첫 번째 토큰(또는 처음 몇 개)에 가중치를 할당합니다. 이는 실제 콘텐츠에 할당되어야 할 용량을 빼앗습니다.

미분 어텐션은 두 개의 어텐션 맵을 계산하고 차이를 구해 이 문제를 해결합니다:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

여기서 `λ`는 학습된 스칼라(일반적으로 0.5–0.8)입니다. A1은 실제 콘텐츠 가중치를 포착하고, A2는 싱크를 포착합니다. 차이를 구하면 싱크가 상쇄되고, 관련 토큰에 가중치가 재할당됩니다.

보고된 결과(Microsoft 2024): 5–10% 낮은 퍼플렉서티, 동일한 훈련 길이에서 1.5–2배 더 긴 유효 컨텍스트, 더 선명한 니들-인-헤이스택 검색.

### 변형 비교

| 변형 | 계산 | KV 캐시 | 전체 대비 품질 | 프로덕션 사용 |
|---------|---------|----------|-----------------|----------------|
| 전체 어텐션 | O(N²) | O(N) per layer | 기준 | 모든 모델의 기본 레이어 |
| SWA (윈도우 1024) | O(N·W) | O(W) per layer | -0.1 ppl, 글로벌 레이어와 함께 우수 | Gemma 2/3, Phi-3-Long |
| 로컬 + 스트라이드 희소 | O(N·√N) | 혼합 | SWA와 유사 | OpenAI 희소 트랜스포머, Longformer |
| BigBird (로컬 + 글로벌 + 무작위) | O(N) 근사 | 혼합 | 2배 컨텍스트에서 전체와 일치 | 초기 장문 컨텍스트 BERT |
| 네이티브 희소 (DeepSeek-V3.2) | O(N · 활성 비율) | O(N) | 0.05 ppl 이내 | DeepSeek-V3.2, 2025 |
| 미분 | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF 트랜스포머, 2026년 초 모델 |

## 빌드하기

`code/main.py`를 참조하세요. 장난감 시퀀스에서 전체(full), SWA, 로컬+스트라이드(local+strided), 미분 어텐션(differential attention) 마스크를 나란히 비교하는 인과적 마스크 비교기를 구현합니다.

### 1단계: 전체 인과 마스크(기준선)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

레슨 07의 기준선. 하위 삼각 행렬; 대각선 위쪽의 가중치는 0입니다.

### 2단계: 슬라이딩 윈도우 인과 마스크

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

하나의 파라미터 — `window`. `window >= n`일 때 전체 인과 어텐션을 복원합니다. `window = 1`일 때 각 토큰은 자기 자신에만 어텐션합니다.

### 3단계: 로컬 + 스트라이드 희소 마스크

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

밀집된 로컬 윈도우와 시퀀스 시작까지의 `stride` 간격 토큰. 추가 레이어와 함께 로그 단계로 수용 필드(receptive field)가 증가합니다.

### 4단계: 미분 어텐션

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

두 어텐션 패스를 학습된 혼합 계수(lam)로 차감. 코드에서는 단일 vs 미분 어텐션의 어텐션 싱크 히트맵을 비교하고 싱크가 붕괴되는 것을 관찰합니다.

### 5단계: KV 캐시 크기

각 변형에 대해 `N = 131072`에서 레이어당 캐시 크기를 출력합니다. SWA와 희소 변형은 10–100배 감소합니다. 미분 어텐션은 2배 증가합니다. 메모리 비용을 의식적으로 지불하세요.

## 사용 방법

2026년 프로덕션 패턴:

```python
from transformers import AutoModelForCausalLM
# Gemma 3은 SWA(윈도우=1024)와 5:1 비율의 글로벌 레이어를 혼합합니다.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+의 FlexAttention은 마스크 함수를 허용합니다:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

이는 커스텀 Triton 커널로 컴파일됩니다. 일반적인 패턴에서 FlashAttention-3 속도의 10% 이내이며, 마스크 함수는 Python 호출 가능합니다.

**각 방법 선택 시기:**

- **순수 전체 어텐션** — ~16K 컨텍스트까지 모든 레이어, 또는 검색 품질이 최우선일 때.
- **SWA + 글로벌 혼합** — 긴 컨텍스트(>32K), 훈련 및 추론 시 메모리 제약. 32K 이상에서 2026년 기본값.
- **희소 블록 어텐션** — 커스텀 커널, 커스텀 패턴. 특수 워크로드(검색, 오디오)에 한정.
- **차등 어텐션** — 어텐션 싱크 오염이 문제되는 모든 워크로드(긴 컨텍스트 RAG, 니들-인-헤이스택).

## Ship It

`outputs/skill-attention-variant-picker.md`를 참조하세요. 이 스킬은 목표 컨텍스트 길이, 검색 요구사항, 훈련/추론 컴퓨팅 프로파일을 기반으로 새로운 모델에 적합한 어텐션(attention) 토폴로지를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. `window=4`에서 SWA가 행당 마지막 4개 토큰 외부를 모두 0으로 만드는지 확인합니다. `window=n`이 전체 인과적 어텐션을 비트 단위로 재현하는지 검증합니다.
2. **중간.** Lesson 07 캡스톤 위에 `window=1024`로 인과적 SWA를 구현합니다. tinyshakespeare에서 1,000스텝 동안 훈련합니다. 검증 손실(val loss)이 전체 어텐션 대비 얼마나 회귀하는지, 최대 메모리 사용량은 얼마나 감소하는지 측정합니다.
3. **어려움.** 캡스톤 모델에 Gemma-3 스타일의 5:1 레이어 믹스(5개 SWA, 1개 글로벌)를 구현합니다. 동일한 파라미터 규모에서 순수-SWA 및 순수-글로벌 베이스라인과 손실, 메모리 사용량, 생성 품질을 비교합니다.
4. **어려움.** 헤드별 학습 가능한 `λ`를 사용한 차등 어텐션(differential attention)을 구현합니다. 합성 검색 작업(1개의 니들, 2,000개의 방해 요소)에서 훈련합니다. 동일한 파라미터 규모에서 단일 어텐션 베이스라인 대비 검색 정확도를 측정합니다.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 슬라이딩 윈도우 어텐션(SWA) | "로컬 어텐션" | 각 쿼리는 마지막 `W` 토큰에 어텐션; KV 캐시가 `O(W)`로 축소. |
| 유효 수용 필드 | "모델이 얼마나 멀리까지 보는지" | 윈도우 `W`를 가진 `L`-계층 SWA 스택에서 최대 `L × W` 토큰. |
| 롱포머 / 빅버드 | "로컬 + 글로벌 + 랜덤" | 몇 개의 항상 어텐션하는 글로벌 토큰을 포함한 희소 패턴; 초기 장문맥 접근법. |
| 네이티브 희소 어텐션 | "DeepSeek의 커널 트릭" | 블록 수준 희소성 학습; 커널 레벨에서 0 블록을 건너뛰면서 품질 유지. |
| 차분 어텐션 | "두 맵, 하나 빼기" | DIFF 트랜스포머: 첫 번째 어텐션 맵에서 두 번째 어텐션 맵에 학습된 `λ`를 곱한 값을 빼서 어텐션 싱크 제거. |
| 어텐션 싱크 | "가중치가 토큰 0으로 유출" | 소프트맥스 정규화로 행 합이 1이 됨; 비정보성 쿼리가 위치 0에 가중치 집중. |
| 플렉스어텐션 | "마스크-애즈-파이썬" | PyTorch 2.5+ API로 임의의 마스크 함수를 FlashAttention-형태 커널로 컴파일. |
| 계층 유형 혼합 | "5:1 SWA-투-글로벌" | 스택 내 희소 및 전체 어텐션 계층을 교차 배치하여 낮은 메모리에서 품질 유지.

## 추가 자료

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 캐노니컬 슬라이딩 윈도우 + 글로벌 토큰 논문.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 로컬 + 글로벌 + 랜덤.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI의 로컬+스트라이드 패턴.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:글로벌 혼합.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 현재 교과서 기본값인 윈도우=1024의 5:1 혼합.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF 트랜스포머 논문.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2의 학습 기반 희소 어텐션.
- [PyTorch — FlexAttention 블로그 및 문서](https://pytorch.org/blog/flexattention/) — "사용하기"의 마스크-콜러블 패턴 API 참조.