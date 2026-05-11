# 추론 최적화 및 EAGLE

> 프론트티어 LLM이 하나의 토큰을 생성하려면 수십억 개의 파라미터에 대한 전체 순전파가 필요합니다. 이 순전파는 과도하게 프로비저닝된 것입니다: 대부분의 경우 훨씬 더 작은 모델이 다음 3-5개 토큰을 정확히 추측할 수 있으며, 큰 모델은 추측을 *검증*하기만 하면 됩니다. 추측이 맞을 경우 하나의 비용으로 5개 토큰을 얻을 수 있습니다. 추론 최적화(Leviathan et al. 2023)는 이를 정확히 구현했으며, EAGLE-3(2025)은 검증당 약 4.5개 토큰의 수용률을 달성했습니다. 이는 출력 분포를 일치시킨 상태에서 4-5배 속도 향상을 의미합니다.

**유형:** 구현
**언어:** Python (NumPy 사용)
**선수 지식:** 10단계 12강(추론 최적화), 10단계 4강(사전 학습 미니-GPT)
**소요 시간:** ~75분

## 문제

H100에서 70B급 모델의 디코딩 처리량(decode throughput)은 일반적으로 초당 40-80 토큰입니다. 각 토큰은 HBM에서 모든 모델 가중치를 읽는 전체 순전파(forward pass)를 필요로 합니다. 출력을 변경하지 않고는 모델을 더 작게 만들 수 없습니다. 메모리 한계 이상으로 배치 크기(batch size)를 늘릴 수도 없습니다. 모델이 단일 순전파로 여러 토큰을 출력할 수 없다면 이 상황에 갇히게 됩니다.

자기회귀적 생성(autoregressive generation)은 본질적으로 직렬적입니다: `x_{t+1} = sample(p(· | x_{1:t}))`. 하지만 동시성(concurrency) 기회가 존재합니다. "다음 4개 토큰은 아마도 [a, b, c, d]일 것"이라고 예측하는 저비용 예측기가 있다면, **대형 모델의 단일 순전파**에서 5개 위치 모두를 검증하고 가장 긴 일치 접두사를 수용할 수 있습니다.

Leviathan, Kalai, Matias(2023, "Fast Inference from Transformers via Speculative Decoding")는 대상 모델의 샘플링 분포를 보존하는 영리한 수용/거부(accept/reject) 규칙을 통해 이를 정확히 구현했습니다. 동일한 출력 분포를 유지하면서 2-4배 더 빠른 속도를 달성했습니다.

## 개념

### 두 모델 구성

- **타겟 모델** `M_p`: 실제로 샘플을 생성하려는 크고 느리지만 고품질 모델. 분포: `p(x)`.
- **드래프트 모델** `M_q`: 작고 빠르지만 낮은 품질의 모델. 분포: `q(x)`. 5-30배 더 작음.

단계별 동작:

1. 드래프트 모델이 `K`개의 토큰을 자기회귀적으로 제안: `x_1, x_2, ..., x_K ~ q`.
2. 타겟 모델이 `K+1` 위치에 대해 병렬로 한 번의 순전파를 실행하여 각 제안된 토큰에 대한 `p(x_k)`를 생성.
3. 아래 수정된 거부 샘플링 규칙을 통해 각 토큰을 좌에서 우로 수락/거부. 가장 긴 일치하는 접두사를 수락.
4. 어떤 토큰이 거부되면 수정된 분포에서 대체 토큰을 샘플링하고 중단. 그렇지 않으면 `p(· | x_1...x_K)`에서 보너스 토큰 하나를 샘플링.

드래프트가 타겟과 완벽하게 일치하면 타겟 순전파당 `K+1`개의 토큰을 얻음. 드래프트가 첫 번째 위치에서 틀리면 1개의 토큰만 얻음.

### 정확성 규칙

스펙큘레이티브 디코딩은 **분포상 `p`에서 샘플링하는 것과 수학적으로 동등**합니다. 거부 규칙:

```
각 드래프트 토큰 x_t에 대해:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        x_t를 수락
    else:
        잔차 분포 (p - q)+ / ||(p - q)+||_1에서 대체 토큰을 샘플링하고 중단
```

여기서 `(p - q)+`는 점별 차이의 양의 부분을 나타냅니다. 드래프트와 타겟이 일치할 때(`p ≈ q`) 수락률은 거의 1입니다. 불일치 시 잔차 분포가 전체 샘플이 정확히 `p`가 되도록 구성됩니다.

**탐욕적 경우.** 온도=0 샘플링에서는 `argmax(p) == x_t`를 확인. 참이면 수락, 거짓이면 `argmax(p)`를 출력하고 중단.

### 기대 속도 향상

드래프트 모델의 토큰 수준 수락률이 `α`일 때, 타겟 순전파당 생성되는 기대 토큰 수는:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = 드래프트 길이, α ∈ [0, 1]
```

`α = 0.8, K = 4`일 때: `(1 - 0.8^5)/(1 - 0.8) = 3.36` 토큰/순전파. 단일 타겟 순전파 비용은 대략 `cost_q * K + cost_p` (K개의 드래프트 단계 + 1회 타겟 검증). `cost_p >> cost_q * K`이면 처리량 속도 향상 비율은 `3.36× / 1 = 3.36×`입니다.

유일한 실제 매개변수는 `α`이며, 이는 드래프트-타겟 정렬에 완전히 의존합니다. 좋은 드래프트가 전부입니다.

### 드래프트 모델 훈련: 지식 증류

임의의 작은 모델은 드래프트로 부적합합니다. 표준 방법은 타겟에서 증류하는 것:

1. 작은 아키텍처 선택 (~70B 타겟에 1B, ~7B 타겟에 500M).
2. 대규모 텍스트 코퍼스에 대해 타겟 모델을 실행; 다음 토큰 분포를 저장.
3. 타겟 분포(실제 토큰이 아닌)에 대해 KL 발산으로 드래프트 훈련.

결과: `α`는 일반적으로 코딩에서 0.6-0.8, 자연어 채팅에서 0.7-0.85. 생산 환경에서 2-3× 속도 향상.

### EAGLE: 트리 드래프트 + 특징 재사용

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty")는 표준 스펙큘레이티브 디코딩의 두 가지 비효율성을 지적:

1. 드래프트는 K개의 직렬 단계를 수행하지만, 드래프트는 가장 최근 검증에서 타겟의 특징(은닉 상태)을 재사용할 수 있음 — 타겟이 이미 풍부한 표현을 계산했는데 드래프트는 이를 처음부터 재계산.
2. 드래프트는 선형 체인을 출력. 드래프트가 *트리* 형태의 후보(각 노드에 여러 추측)를 출력할 수 있다면, 타겟의 단일 순전파는 트리 어텐션 마스크를 통해 여러 후보 경로를 병렬로 검증하고 가장 긴 수락된 분기를 선택할 수 있음.

EAGLE-1 변경 사항:
- 드래프트 입력 = 원시 토큰이 아닌 위치 t에서의 타겟 최종 은닉 상태.
- 드래프트 아키텍처 = 별도의 소형 모델이 아닌 1개의 트랜스포머 디코더 레이어.
- 출력 = 깊이 4-6, 각 깊이당 K=4-8 후보의 트리.

EAGLE-2 (2024)는 동적 트리 토폴로지 추가: 드래프트가 불확실한 곳에서는 트리가 넓게 성장하고 확신하는 곳에서는 좁게 유지. 검증 비용을 증가시키지 않으면서 `α_effective`를 상승.

EAGLE-3 (Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test")는 고정된 최상위 계층 특징 의존성을 제거하고 새로운 "테스트 시간 시뮬레이션" 손실로 드래프트를 훈련 — 드래프트는 교사 강요 훈련 분포가 아닌 타겟의 테스트 시간 분포와 일치하는 출력으로 훈련. 수락률은 0.75(EAGLE-2)에서 0.82(EAGLE-3)로, 평균 토큰/검증 수는 3.0에서 4.5로 상승.

### 트리 어텐션 검증

드래프트가 트리를 출력하면 타겟 모델은 **트리 어텐션 마스크**를 사용하여 단일 순전파로 검증 — 순수 선형 대신 트리 토폴로지를 인코딩하는 인과적 마스크. 각 토큰은 트리 내 조상 노드에만 어텐션. 검증 패스는 여전히 1회 순전파, 1회 행렬 곱셈; 토폴로지 마스크는 몇 개의 추가 KV 항목만 필요.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

`a, b`가 경쟁하는 첫 번째 토큰 후보이고 `c, d, e, f`가 두 번째 토큰 후보일 때, 6개 위치 모두 1회 순전파로 검증. 출력은 수락된 어떤 경로에서든 가장 긴 접두사.

### 승리하는 경우 vs. 실패하는 경우

**승리:**
- 예측 가능한 텍스트(코드, 일반적인 영어, 구조화된 출력)의 채팅/완성. `α`가 높음.
- 디코딩 중 미사용 GPU 컴퓨트가 있는 환경(메모리 바운드 단계). 트리 드래프트는 사용 가능한 FLOPs를 활용.

**패배 / 효과 없음:**
- 고엔트로피 출력(고온도의 창의적 글쓰기). `α`가 `|vocab|` 수준으로 감소.
- 매우 높은 동시성을 가진 배치 서빙 — 배치 처리로 FLOPs가 이미 가득 차 있어 트리 검증 여유 공간 부족.
- 드래프트가 크게 작지 않은 매우 작은 타겟 모델.

생산 환경에서는 일반적으로 채팅에서 2-3×, 코드 생성에서 3-5×의 벽시계 시간 속도 향상을 보고하며, 창의적 글쓰기에서는 거의 효과가 없음.

## 빌드하기

`code/main.py`:

- 정확한 거부 규칙을 구현하고 대상 분포를 보존하는지 검증하는 참조 `speculative_decode(target, draft, prompt, K, temperature)` (경험적 KL < 0.01 vs 일반 대상 샘플링).
- 상위-p 브랜칭으로 깊이-K 트리를 구축하는 EAGLE 스타일 트리 드래프터.
- 검증자를 위한 올바른 인과 패턴을 생성하는 트리 어텐션 마스크 빌더.
- 소형 LM(GPT-2-medium 대상으로부터 GPT-2-small을 증류한 모델)에서 모두 실행되는 수용률 측정 하니스.

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """한 라운드의 추측 디코딩. 수용된 토큰 목록을 반환합니다."""
    # 1. K개 토큰 드래프트
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. 대상은 모든 드래프트 위치 + 1개 추가 위치에서 p 계산
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. 왼쪽에서 오른쪽으로 수용/거부
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. K개 모두 수용 → 대상에서 보너스 토큰 샘플링
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 사용 방법

- **vLLM**과 **SGLang**은 1급 추측 디코딩을 제공합니다. 플래그: `--speculative_model`, `--num_speculative_tokens`. EAGLE-2/3 지원은 `--spec_decoding_algorithm eagle` 플래그를 통해 가능합니다.
- **NVIDIA TensorRT-LLM**은 Medusa와 EAGLE 트리를 네이티브로 지원합니다.
- **참조 초안 모델**: `Qwen/Qwen3-0.6B-spec` (Qwen3-32B용 초안), `meta-llama/Llama-3.2-1B-Instruct-spec` (70B용 초안).
- **Medusa 헤드** (Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): 초안 모델 대신 대상 모델 자체에 K개의 병렬 예측 헤드를 추가합니다. 배포가 더 간단하며, EAGLE보다 약간 낮은 수용률을 보입니다.

## Ship It

이 레슨은 `outputs/skill-speculative-tuning.md`를 생성합니다 — 대상 모델의 워크로드를 프로파일링하고 다음 사항을 선택하는 스킬입니다: 초안 모델, K(초안 길이), 트리 폭, 온도, 그리고 일반 디코딩으로 폴백할 시점.

## 연습 문제

1. 정확한 거부 규칙(rejection rule)을 구현하고 이를 경험적으로 검증하세요. `speculative_decode`와 일반 타겟 샘플링을 통해 10K 샘플을 실행하고, 두 출력 분포 간의 TV 거리(Total Variation distance)를 계산하세요. 거리는 0.01 미만이어야 합니다.

2. 속도 향상(speedup) 공식을 계산하세요. 고정된 `α`와 `K`를 사용하여, 타겟 순방향 연산당 예상 토큰 수를 그래프로 나타내세요. `α ∈ {0.5, 0.7, 0.9}`에 대한 최적의 `K`를 찾으세요.

3. 소형 초안 모델(draft)을 학습하세요. 124M GPT-2 타겟 모델을 사용하여, 100M 토큰으로 KL 손실(KL loss)을 적용해 30M GPT-2 초안 모델을 지식 증류(knowledge distillation)하세요. 보유 텍스트(held-out text)에서 `α`를 측정하세요. 예상 값: 0.6-0.7.

4. EAGLE 스타일의 트리 초안(tree drafting)을 구현하세요. 체인 대신, 초안이 각 깊이에서 상위 3개 분기(branches)를 출력하도록 합니다. 트리 어텐션 마스크(tree attention mask)를 구성하세요. 타겟 모델이 가장 긴 올바른 분기를 수용하는지 검증하세요.

5. 실패 모드(failure modes)를 측정하세요. 온도=1.5(고확률적 온도)에서 추측 디코딩(speculative decoding)을 실행하세요. `α`가 붕괴되고 초안 오버헤드로 인해 알고리즘이 일반 디코딩보다 느려지는 것을 보여주세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|----------|
| 타겟 모델(Target model) | "큰 모델" | 샘플링을 원하는 느리고 고품질인 모델 (p 분포) |
| 드래프트 모델(Draft model) | "추측자" | 작고 빠른 예측 모델 (q 분포); 5-30배 더 작음 |
| K / 드래프트 길이(Draft length) | "미리 보기" | 검증 한 번당 추측하는 토큰 수 |
| α / 수용률(Acceptance rate) | "적중률" | 드래프트 제안이 수용될 토큰별 확률 |
| 정확한 거부 규칙(Exact rejection rule) | "수용 테스트" | 타겟 분포를 보존하는 p/q 비교 (r < p/q) |
| 잔여 분포(Residual distribution) | "수정된 p-q" | (p - q)+ / ||(p - q)+||_1, 거부 시 샘플링할 분포 |
| 트리 드래프팅(Tree drafting) | "분기 추측" | 드래프트가 후보 트리를 출력하고, 트리 구조 어텐션 마스크로 한 번에 검증 |
| 트리 어텐션 마스크(Tree attention mask) | "위상 마스크" | 각 노드가 조상만 참조하도록 트리 위상을 인코딩한 인과적 마스크 |
| 메두사 헤드(Medusa heads) | "병렬 헤드" | 타겟 자체에 추가된 K개의 예측 헤드; 별도 드래프트 모델 없음 |
| 이글 특징 재사용(EAGLE feature reuse) | "은닉 상태 드래프트" | 드래프트 입력이 원시 토큰이 아닌 타겟의 마지막 은닉 상태; 드래프트 축소 |
| 테스트 시간 시뮬레이션 손실(Test-time simulation loss) | "이글-3 훈련" | 교사 강제가 아닌 타겟의 테스트 시간 분포와 일치하는 출력으로 드래프트 훈련 |

## 추가 자료

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — 정확한 거부 규칙 및 이론적 속도 향상 분석
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind의 동시적 추측 샘플링 논문
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — 초안 모델 대신 병렬 디코딩 헤드 사용
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — 특징 재사용 및 트리 초안
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — 동적 트리 토폴로지
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — 훈련-테스트 시간 매칭
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — 추측기 없는 대안인 야코비/룩어헤드 디코딩