# Speculative Decoding — 초안 작성, 검증, 반복

> 자기회귀적 디코딩은 직렬 방식입니다. 각 토큰은 이전 토큰이 생성될 때까지 기다려야 합니다. Speculative 디코딩은 이 체인을 끊습니다: 저렴한 모델이 N개의 토큰을 초안으로 작성하고, 고가의 모델이 한 번의 순방향 전달로 N개 전체를 검증합니다. 초안이 정확할 경우 N개의 생성을 위해 단 한 번의 큰 순방향 전달 비용만 지불하면 됩니다.

**유형:** 구현  
**언어:** Python  
**선수 지식:** 7단계 · 07 (GPT 인과적 언어 모델), 7단계 · 12 (KV 캐시 & 플래시 어텐션)  
**소요 시간:** ~60분

## 문제

70B LLM이 하나의 토큰을 샘플링하는 데 H100에서 약 30ms가 소요됩니다. 3B 초안 모델은 약 3ms가 소요됩니다. 3B 모델로 5토큰을 미리 생성한 다음, 70B 모델을 한 번 실행하여 5토큰 전체를 검증하면 총 소요 시간은 `5×3 + 30 = 45ms`로, 최대 5개의 수락된 토큰을 생성할 수 있습니다. 이는 직선형 생성(`5×30 = 150ms`)에 비해 빠릅니다. 이것이 바로 추측 디코딩(speculative decoding)의 핵심 아이디어입니다: 소량의 추가 GPU 메모리(초안 모델)를 희생하여 디코딩 지연 시간을 2–4배 줄이는 것입니다.

이 기법은 분포를 보존해야 합니다. Leviathan 등(2023)과 Chen 등(동시기)이 제안한 추측 샘플링은 출력 시퀀스가 대형 모델이 단독으로 생성한 것과 **동일한 분포**를 갖도록 보장합니다. 품질 저하는 없으며, 단지 더 빠를 뿐입니다.

2026년 추론 분야를 주도하는 초안-검증 모델 4가지 계열:

1. **바닐라 추측 디코딩(Leviathan 2023).** 별도의 초안 모델(예: Llama 3 1B)과 검증 모델(예: Llama 3 70B)을 사용합니다.
2. **메두사(Cai 2024).** 검증 모델에 여러 디코딩 헤드를 추가하여 위치 `t+1..t+k`를 병렬로 예측합니다. 별도의 초안 모델이 필요 없습니다.
3. **EAGLE 계열(Li 2024, 2025).** 검증 모델의 은닉 상태를 재사용하는 경량 초안 모델; 바닐라 대비 높은 수락률; 일반적으로 3–4배 성능 향상.
4. **룩어헤드 디코딩(Fu 2024).** 야코비 반복법; 초안 모델 불필요. 자체 추측(self-speculation). 특정 용도에 적합하지만 의존성 없음.

2026년의 모든 프로덕션 추론 스택은 기본적으로 추측 디코딩을 탑재합니다. vLLM, TensorRT-LLM, SGLang, llama.cpp는 모두 최소 바닐라 + EAGLE-2를 지원합니다.

## 개념

### 핵심 알고리즘

검증기 `M_q`와 더 저렴한 초안 모델 `M_p`가 주어졌을 때:

1. 이미 디코딩된 접두사 `x_1..x_k`를 정의합니다.
2. **초안 생성**: `M_p`를 사용하여 `d_{k+1}, d_{k+2}, ..., d_{k+N}`을 자기회귀적으로 제안하고, 초안 확률 `p_1..p_N`을 계산합니다.
3. **병렬 검증**: `M_q`를 한 번 실행하여 `x_1..x_k, d_{k+1}, ..., d_{k+N}`에 대한 검증기 확률 `q_1..q_{N+1}`을 위치 `k+1..k+N+1`에 대해 얻습니다.
4. **좌측에서 우측으로 초안 토큰 수락/거부**: 각 `i`에 대해 `min(1, q_i(d_i) / p_i(d_i))` 확률로 수락합니다.
5. 위치 `j`에서 첫 거부 발생 시: "잔여" 분포 `(q_j - p_j)_+`를 정규화한 분포에서 `t_j`를 샘플링합니다. `j` 이후의 모든 초안은 폐기됩니다.
6. 모든 `N`을 수락할 경우: `q_{N+1}`에서 추가 토큰 `t_{N+1}`을 샘플링합니다(무료 보너스 토큰).

잔여 분포 기법은 출력이 마치 `M_q`가 처음부터 샘플링한 것처럼 정확히 분포되도록 유지하는 수학적 통찰입니다.

### 속도 향상 결정 요소

`α` = 초안 토큰당 기대 수락률, `c` = 초안 대 검증기 비용 비율일 때, 단계별:

- 순진 생성은 토큰당 1회의 대형 모델 호출을 수행합니다.
- 추측 생성은 `α`가 높을 때 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 토큰당 1회의 대형 모델 호출을 수행합니다.

`α = 0.75` 및 `N = 5`일 때의 일반적인 경험 법칙: 대형 모델 호출 3배 감소. 초안 비용은 5배 저렴. 총 벽시계 시간은 ~2.5배 감소.

**α에 영향을 주는 요소:**

- 초안이 검증기를 얼마나 잘 근사하는지. 동일한 패밀리/동일한 학습 데이터는 α를 크게 향상시킵니다.
- 디코딩 전략. 탐욕적 초안 대 탐욕적 검증기: 높은 α. 온도 샘플링: 일치하기 어려움; 수락률 감소.
- 작업 유형. 코드 및 구조화된 출력은 더 많이 수락함(예측 가능); 자유 형식의 창의적 글쓰기는 덜 수락함.

### Medusa — 초안 모델 없는 추측

Medusa는 초안 모델을 검증기에 추가된 출력 헤드로 대체합니다. 위치 `t`에서:

```
공유 트렁크 → 은닉 상태 h_t
    ├── head_0: t+1 위치 토큰 예측 (표준 LM 헤드)
    ├── head_1: t+2 위치 토큰 예측
    ├── head_2: t+3 위치 토큰 예측
    ├── head_3: t+4 위치 토큰 예측
```

각 헤드는 자체 로짓을 출력합니다. 추론 시 각 헤드에서 샘플링하여 후보 시퀀스를 얻은 후, 모든 후보 연속을 동시에 고려하는 트리-어텐션 방식을 사용하여 한 번의 순전파로 검증합니다.

장점: 두 번째 모델 불필요. 단점: 학습 가능한 매개변수 추가; 지도 미세 조정 단계 필요(~1B 토큰); 수락률은 좋은 초안을 사용하는 일반 추측보다 약간 낮음.

### EAGLE — 은닉 상태 재사용을 통한 향상된 초안

EAGLE-1/2/3 (Li et al., 2024–2025)는 초안 모델을 검증기의 마지막 레이어 은닉 상태를 입력으로 받는 소형 트랜스포머(일반적으로 1층)로 구성합니다. 초안이 검증기의 특징 표현을 보기 때문에 예측 분포가 검증기 출력과 강하게 상관됩니다. 수락률은 ~0.6(일반)에서 0.85+로 상승합니다.

EAGLE-3(2025)은 후보 연속에 대한 트리 탐색을 추가했습니다. vLLM과 SGLang은 Llama 3/4 및 Qwen 3의 기본 추측 경로로 EAGLE-2/3을 제공합니다.

### KV 캐시 관리

검증은 한 번의 순전파로 `N`개의 초안 토큰을 검증기에 입력합니다. 이는 검증기의 KV 캐시를 `N`개 항목만큼 확장합니다. 일부 초안이 거부되면 캐시를 수락된 접두사 길이로 되돌려야 합니다.

프로덕션 구현(vLLM의 `--speculative-model`, TensorRT-LLM의 LookaheadDecoder)은 스크래치 KV 버퍼로 이를 처리합니다. 먼저 쓴 후 수락 시 커밋합니다. 개념적으로 어렵지는 않지만 세부 사항이 까다롭습니다.

## 빌드하기

`code/main.py`를 참조하세요. 다음 요소를 사용하여 핵심 추측 샘플링 알고리즘(거부 단계 + 잔여 분포)을 구현합니다:

- 핸드코딩된 분포에 대한 결정론적 소프트맥스인 "큰 모델"(수용 수학 검증 가능).
- 큰 모델의 변형인 "초안 모델".
- 직접 샘플링과 동일한 주변 분포를 생성하는 수용/거부 루프.

### 1단계: 거부 단계

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u`는 균일 난수입니다. `q_prob`는 검증자(큰 모델)의 초안 토큰 확률입니다. `p_prob`는 초안 모델의 확률입니다. 레비아탄 정리는 이 베르누이 결정 후 거부 시 잔여 분포 샘플링이 검증자 분포를 정확히 보존한다는 것입니다.

### 2단계: 잔여 분포

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

`q`에서 `p`를 요소별로 빼고, 음수 값을 0으로 클램핑한 후 재정규화합니다. 거부 시 이 분포에서 샘플링합니다.

### 3단계: 단일 추측 단계

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

5회 수용 → 1회 보너스 → 1회 검증자 패스로 6개 토큰 생성.

### 4단계: 수용률 측정

초안 품질 수준을 다양하게 하여 10,000회 추측 단계를 실행합니다. 초안 모델과 검증자 모델 간 KL 발산과 수용률을 플롯으로 표시하면 단조 관계가 나타납니다.

### 5단계: 분포 동등성 검증

경험적 검증: 추측 루프로 생성된 토큰 히스토그램은 검증자에서 직접 샘플링한 히스토그램과 일치해야 합니다. 이는 레비아탄 정리의 실제 적용 사례입니다. 카이제곱 검정은 샘플링 오차 내에서 이를 확인합니다.

## 사용 방법

프로덕션:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM은 2026년 중반 기준으로 가장 빠른 Medusa 경로를 제공합니다. `faster-whisper`는 Whisper-large에 대한 추측 디코딩을 소형 드래프트 모델로 래핑합니다.

**드래프트 모델 선택:**

| 전략 | 선택 시기 | 속도 향상 |
|----------|--------------|---------|
| Vanilla 드래프트 (1B/3B Llama 계열) | 빠른 프로토타이핑, 훈련 불필요 | 1.8–2.3× |
| Medusa 헤드 | 검증기 파인튜닝 가능 | 2–3× |
| EAGLE-2 / 3 | 프로덕션, 최대 속도 | 3–4× |
| Lookahead | 드래프트 불필요, 훈련 불필요, 추가 파라미터 불필요 | 1.3–1.6× |

**추측 디코딩을 사용하지 말아야 할 경우:**

- 1–5토큰 단일 시퀀스 생성. 오버헤드가 지배적입니다.
- 매우 창의적인 / 고온 샘플링(α 감소).
- 메모리 제약 환경(드래프트 모델이 VRAM 추가 사용).

## Ship It

`outputs/skill-spec-decode-picker.md`를 참조하세요. 이 스킬은 새로운 추론 작업(workload)을 위해 추측 디코딩 전략(vanilla / Medusa / EAGLE / lookahead)과 튜닝 파라미터(N, draft temperature)를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. 50,000개 토큰에 대해 카이제곱 검정 p > 0.05 수준에서 추측 토큰 분포가 검증기의 직접 샘플링 분포와 일치하는지 확인합니다.
2. **중간.** `α = 0.5, 0.7, 0.85`에 대해 `N`의 함수로 속도 향상(큰 모델 순방향당 토큰 수)을 그래프로 그립니다. 각 α에 대한 최적 `N`을 식별하세요. (힌트: 검증 호출당 예상 토큰 수 = `(1 - α^{N+1}) / (1 - α)`.)
3. **어려움.** 작은 Medusa를 구현합니다: 14강의 캡스톤 GPT를 가져와 t+2, t+3, t+4 위치를 예측하는 3개의 추가 LM 헤드를 추가합니다. tinyshakespeare 데이터로 공동 다중 헤드 손실을 사용해 훈련합니다. 동일한 모델을 잘라내어 만든 일반 드래프트 대비 수용률을 비교합니다.
4. **어려움.** 롤백을 구현합니다: 10토큰 접두사 KV 캐시로 시작해 5개 드래프트 토큰을 입력합니다. 위치 3에서 거부를 시뮬레이션합니다. 다음 반복에서 캐시 읽기가 "접두사 + 첫 2개 수용된 드래프트"와 정확히 일치하는지 확인합니다.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| Draft model | "저렴한 모델" | 후보 토큰을 제안하는 더 작은 모델; 일반적으로 검증 모델보다 10–50배 저렴합니다. |
| Verifier | "큰 모델" | 분포를 보존하는 대상 모델; 추측 단계마다 한 번 실행됩니다. |
| Acceptance rate (α) | "초안이 맞는 빈도" | 검증 모델이 초안을 수락하는 토큰별 확률. 일반적으로 0.7–0.9입니다. |
| Residual distribution | "거부 시 대체 분포" | `(q - p)_+` 정규화; 거부 시 이 분포에서 샘플링하면 검증 모델의 분포가 보존됩니다. |
| Bonus token | "무료 토큰" | 모든 N개의 초안이 수락되면 검증 모델의 다음 단계 분포에서 하나 더 샘플링합니다. |
| Medusa | "초안 없는 추측" | 검증 모델에 여러 LM 헤드를 추가하여 t+1..t+k 위치를 병렬로 예측합니다. |
| EAGLE | "은닉 상태 초안" | 검증 모델의 마지막 레이어 은닉 상태에 조건화된 소형 트랜스포머 초안 모델입니다. |
| Lookahead decoding | "야코비 반복" | 고정점 반복을 사용한 자기 추측; 초안 모델이 필요 없습니다. |
| Tree attention | "한 번에 여러 후보 검증" | 여러 초안 연속을 동시에 고려하는 분기 검증입니다. |
| KV rollback | "거부된 초안 취소" | 임시 KV 버퍼; 수락 시 커밋, 거부 시 폐기합니다.

## 추가 자료

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 핵심 알고리즘 및 동치 정리.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 동시 연구; 베르누이-거부 증명.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 논문; 트리-어텐션 검증.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; 은닉 상태 조건부 초안.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; 동적 트리 깊이.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 룩어헤드, 초안 없음 접근법.
- [vLLM 문서 — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 4가지 전략이 모두 구현된 표준 프로덕션 참조.
- [SafeAILab / EAGLE 참조 구현](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3의 참조 코드.