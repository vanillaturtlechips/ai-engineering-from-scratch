# KV 캐시, 플래시 어텐션 & 추론 최적화

> 훈련은 병렬 처리 및 FLOP 제한적입니다. 추론은 직렬 처리 및 메모리 제한적입니다. 다른 병목 현상, 다른 기법.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** 7단계 · 02 (셀프 어텐션), 7단계 · 05 (전체 트랜스포머), 7단계 · 07 (GPT)
**소요 시간:** ~75분

## 문제

순진한 자기회귀 디코더는 `N`개의 토큰을 생성하기 위해 `O(N²)` 작업을 수행합니다. 각 단계에서 전체 접두사에 대한 어텐션을 재계산하기 때문입니다. 4K 토큰 응답의 경우 16M 어텐션 연산이 발생하며, 이 중 대부분은 중복됩니다. 접두사 토큰의 모든 은닉 상태는 한 번 계산되면 결정론적입니다. 새로 생성된 토큰의 쿼리만 이전 모든 토큰의 캐시된 키(keys)와 값(values)에 대해 실행하면 됩니다.

게다가 어텐션 자체는 많은 데이터를 이동시킵니다. 표준 어텐션은 N×N 점수 행렬, N×d 소프트맥스 출력, N×d 최종 출력을 구체화(materialize)합니다. 이는 HBM에 대한 너무 많은 읽기/쓰기 작업을 유발합니다. N≥2K인 경우, 어텐션은 FLOP에 제한되기 전에 메모리-바운드(memory-bound) 상태가 됩니다. 기존 어텐션 커널은 현대 GPU를 4–10배 정도 활용하지 못합니다.

Dao et al.의 두 가지 최적화는 "느린" 추론(frontier inference)을 "빠른" 추론으로 전환했습니다:

1. **KV 캐시.** 모든 접두사 토큰의 K와 V 벡터를 저장합니다. 새 토큰의 어텐션은 캐시된 키에 대한 단일 쿼리 실행으로 처리됩니다. 추론은 생성 단계당 `O(N²)`에서 `O(N)`으로 감소합니다.
2. **플래시 어텐션(Flash Attention).** 어텐션 계산을 타일링하여 전체 N×N 행렬이 HBM에 저장되지 않도록 합니다. 소프트맥스 + 행렬 곱셈(matmul) 전체가 SRAM에서 발생합니다. A100에서 2–4배, FP8을 사용하는 H100에서 5–10배의 벽시계 시간(wall-clock) 속도 향상을 제공합니다.

2026년까지 이 두 기술은 보편화되었습니다. 모든 프로덕션 추론 스택(vLLM, TensorRT-LLM, SGLang, llama.cpp)은 이를 전제로 합니다. 모든 최신 모델(frontier model)은 플래시 어텐션이 활성화된 상태로 출시됩니다.

## 개념

![KV 캐시 증가와 Flash Attention 타일링](../assets/kv-cache-flash-attn.svg)

### KV 캐시 수학

디코더 레이어, 토큰, 헤드당:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K와 V
```

32 레이어, 32 헤드, d_head=128, fp16을 사용하는 7B 모델의 경우:

```
레이어당 토큰 = 2 * 128 * 2 = 512 바이트
토큰당 (32 레이어) = 16 KB
32K 컨텍스트당 = 512 MB
```

Llama 3 70B(80 레이어, d_head=128, GQA 8 KV 헤드)의 경우:

```
레이어당 토큰 = 2 * 8 * 128 * 2 = 4096 바이트 (4 KB)
32K 컨텍스트당 = 10.4 GB
```

이 10GB가 바로 Llama 3 70B가 128K 컨텍스트에서 배치 크기 1로 KV 캐시만 위해 40GB A100의 대부분을 사용하는 이유입니다.

**GQA는 KV-캐시의 승자.** 64 헤드를 사용하는 MHA는 32GB가 필요합니다. MLA는 더 압축합니다.

### Flash Attention — 타일링 트릭

표준 어텐션:

```
S = Q @ K^T          (HBM 읽기, N×N, HBM 쓰기)
P = softmax(S)       (HBM 읽기, HBM 쓰기)
O = P @ V            (HBM 읽기, HBM 쓰기)
```

3번의 HBM 왕복. H100에서 HBM 대역폭은 3TB/s, SRAM은 30TB/s입니다. 모든 HBM 접근은 온칩 유지 대비 10배 느립니다.

Flash Attention:

```
Q의 각 블록(타일 크기 ~128 × 128)에 대해:
    Q_tile을 SRAM에 로드
    K, V의 각 블록에 대해:
        K_tile, V_tile을 SRAM에 로드
        S_tile = Q_tile @ K_tile^T 계산     (SRAM)
        실행 중인 소프트맥스 집계             (SRAM)
        O_tile에 누적                        (SRAM)
    O_tile을 HBM에 쓰기
```

타일당 1번의 HBM 접근. 총 메모리 사용량은 `O(N²)`에서 `O(N)`으로 감소합니다. 역전파 시 순방향 패스의 일부 값을 재계산하여 저장하지 않음 — 또 다른 메모리 절약.

**수치적 트릭.** 실행 중인 소프트맥스는 타일 간 `(max, sum)`을 유지하여 최종 정규화가 정확합니다. 근사가 아닌 — Flash Attention은 표준 어텐션과 비트 단위 동일한 출력을 계산합니다(fp16 비결합성 제외).

**버전 진화:**

| 버전 | 연도 | 주요 변경 사항 | 기준 하드웨어 대비 속도 향상 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | 타일링된 SRAM 커널 | A100에서 2배 |
| Flash 2 | 2023 | 향상된 병렬성, 인과적 순서 우선 | A100에서 3배 |
| Flash 3 | 2024 | Hopper 비동기성, FP8 | H100에서 1.5–2배 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5단계 파이프라인, 소프트웨어 exp2 | 추론 우선 (초기엔 순방향만) |

Flash 4는 출시 시 순방향 패스만 지원합니다. 훈련에는 여전히 Flash 3을 사용합니다. Flash 4의 GQA 및 가변 길이 지원은 예정되어 있습니다(2026년 중반).

### 추측 디코딩 — 다른 지연 시간 개선

저렴한 모델이 N개의 토큰을 제안합니다. 큰 모델이 N개를 병렬로 검증합니다. 검증이 k개의 토큰을 수락하면, k개 생성에 대해 1번의 큰 모델 순방향 패스만 지불합니다. 코드 및 산문에서는 일반적으로 k=3–5입니다.

2026년 기본값:
- **EAGLE 2 / 메두사.** 검증자의 은닉 상태를 공유하는 통합 초안 헤드. 품질 손실 없이 2–3배 속도 향상.
- **초안 모델과의 추측 디코딩.** 소비자 하드웨어에서 2–4배 속도 향상.
- **선행 디코딩.** 야코비 반복; 초안 모델 불필요. 니치하지만 무료.

### 연속 배치

클래식 배치 추론: 가장 느린 시퀀스가 완료될 때까지 기다린 후 새 배치 시작. 짧은 응답이 일찍 완료될 때 GPU를 낭비합니다.

연속 배치(Orca에서 처음 출시, 현재 vLLM, TensorRT-LLM, SGLang에 포함): 이전 요청이 완료되는 즉시 새 요청을 배치에 교체합니다. 일반적인 채팅 워크로드에서 5–10배 처리량 향상.

### PagedAttention — 가상 메모리로서의 KV 캐시

vLLM의 주요 기능. KV 캐시는 16토큰 블록으로 할당되며, 페이지 테이블이 논리적 위치를 물리적 블록에 매핑합니다. 병렬 샘플(빔 검색, 병렬 샘플링) 간 KV 공유, 프롬프트 캐싱을 위한 접두사 핫 스왑, 메모리 조각화 해제가 가능합니다. 순차적 할당 대비 4배 처리량 향상.

## 빌드하기

`code/main.py`를 참조하세요. 다음 내용을 구현합니다:

1. 순진한 `O(N²)` 증분 디코더.
2. `O(N)` KV-캐시 디코더.
3. Flash Attention의 실행-최대 알고리즘을 시뮬레이션하는 타일드 소프트맥스.

### 단계 1: KV 캐시

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

간단합니다: 각 레이어, 각 헤드별로 토큰이 증가할 때마다 K, V 벡터를 유지합니다.

### 단계 2: 타일드 소프트맥스

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention 스타일의 softmax(qK^T)V로 실행-최대/합을 구현합니다."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

한 번에 `softmax(qK) V`와 비트-동일한 출력을 생성하지만, 작업 세트는 전체 `N × d_head`가 아닌 `tile × d_head` 블록입니다.

### 단계 3: 100토큰 생성 시 순진 vs 캐시 디코딩 비교

어텐션 연산 횟수를 계산합니다. 순진: `O(N²)` = 5050. 캐시: `O(N)` = 100. 코드는 두 값을 모두 출력합니다.

## 사용 방법

```python
# HuggingFace transformers는 디코더 전용 generate()에서 KV 캐시를 자동 활성화합니다.
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # Hopper인 경우 FA3 사용
    torch_dtype="bfloat16",
)
# generate()는 KV 캐시를 자동으로 사용합니다.
```

vLLM 프로덕션:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

요청 간 접두사 캐싱은 큰 2026년 승리입니다 — 동일한 시스템 프롬프트, 몇 가지 예시, 또는 긴 컨텍스트 문서는 호출 간 KV를 재사용합니다. 반복되는 도구 프롬프트가 있는 에이전트 워크로드의 경우 접두사 캐싱은 일반적으로 처리량이 5배 증가합니다.

## Ship It

`outputs/skill-inference-optimizer.md`를 참조하세요. 이 스킬은 새로운 추론 배포를 위해 어텐션(attention) 구현 방식, KV 캐시 전략, 양자화(quantization), 그리고 추측적 디코딩(speculative decoding)을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. 순진(naive) 디코더와 캐시된(cached) 디코더가 동일한 출력을 생성하는지 확인하고, 연산 횟수(op-count) 차이를 기록합니다.
2. **중간.** 접두사(prefix) 캐싱 구현: 프롬프트 P와 여러 완성(completion)이 주어졌을 때, P에 대한 단일 순방향(forward) 패스를 실행하여 KV 캐시를 채운 후, 각 완성별로 분기(branch)합니다. 각 완성마다 P를 다시 인코딩하는 경우와 비교하여 속도 향상(speedup)을 측정합니다.
3. **어려움.** 장난감(toy) PagedAttention 구현: 고정 16-토큰 블록으로 KV 캐시를 구성하고, 빈 블록 목록(free-list)을 관리합니다. 시퀀스가 완료되면 해당 블록을 풀로 반환합니다. 길이가 다양한 1,000개의 채팅 완성을 시뮬레이션합니다. 연속 할당(contiguous allocation) 대비 메모리 단편화(memory fragmentation) 정도를 비교합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| KV 캐시 | "디코딩을 빠르게 만드는 비결" | 모든 접두사 토큰에서 저장된 K와 V; 새로운 쿼리는 재계산 대신 이들에 어텐션(attention)합니다. |
| HBM | "GPU 메인 메모리" | 고대역폭 메모리(High Bandwidth Memory); H100에서 80GB, B200에서 192GB. ~3TB/s 대역폭. |
| SRAM | "온칩 메모리" | SM당 고속 메모리, H100에서 SM당 ~256KB. ~30TB/s 대역폭. |
| 플래시 어텐션(Flash Attention) | "타일링된 어텐션 커널" | HBM에 N×N을 구체화하지 않고 어텐션을 계산합니다. |
| 연속 배치(Continuous batching) | "대기 없는 배치" | 완료된 시퀀스를 교체하고 새 시퀀스를 배치에 추가하며, 배치를 비우지 않습니다. |
| 페이징 어텐션(PagedAttention) | "vLLM의 핵심 기술" | 페이지 테이블로 고정 블록에 KV 캐시 할당; 조각화를 제거합니다. |
| 접두사 캐싱(Prefix caching) | "긴 프롬프트 재사용" | 요청 간 공유 접두사에 대한 KV를 캐시; 에이전트 비용을 대폭 절감합니다. |
| 추측 디코딩(Speculative decoding) | "초안 + 검증" | 저렴한 초안 모델이 토큰을 제안하고, 큰 모델이 한 번에 k개를 검증합니다. |

## 추가 자료

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5-stage pipeline과 software-exp2 트릭; 이 강의에서 언급한 forward-only launch 주의 사항은 레포지토리 README를 참조하세요.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 논문.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — spec decoding.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — 강의에서 인용한 통합 초안(integrated-draft) 접근법의 EAGLE-1/2 논문.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — EAGLE과 함께 참조된 Medusa 접근법.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16-토큰 블록과 페이지 테이블 설계에 대한 공식 심층 분석.