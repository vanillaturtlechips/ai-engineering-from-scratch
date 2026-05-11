# 정책 경사(Policy Gradient) — REINFORCE 처음부터 구현하기

> 가치 추정을 멈추세요. 정책을 직접 파라미터화하고, 기대 보상의 기울기를 계산한 후 상승 방향으로 업데이트하세요. Williams (1992)가 하나의 정리로 정리했습니다. 이것이 PPO, GRPO, 그리고 모든 LLM 강화학습 루프가 존재하는 이유입니다.

**유형:** 구현
**언어:** Python
**선수 지식:** Phase 3 · 03 (역전파), Phase 9 · 03 (몬테카를로), Phase 9 · 04 (TD 학습)
**소요 시간:** ~75분

## 문제

Q-러닝과 DQN은 *가치* 함수를 파라미터화합니다. `argmax Q`로 액션을 선택합니다. 이는 이산 액션과 이산 상태에서는 괜찮습니다. 하지만 액션이 연속적일 때(예: 10차원 토크에 대한 `argmax`?) 또는 확률적 정책을 원할 때(구성상 `argmax`는 결정론적임) 문제가 발생합니다.

정책 경사(Policy Gradient)는 대신 *정책*을 파라미터화합니다. `π_θ(a | s)`는 액션에 대한 분포를 출력하는 신경망입니다. 이를 샘플링하여 액션을 실행합니다. `θ`에 대한 기대 보상의 기울기를 계산합니다. 기울기를 따라 상승합니다. `argmax`도, 벨만 재귀도 필요 없습니다. 단지 `J(θ) = E_{π_θ}[G]`에 대한 경사 상승법만 수행합니다.

REINFORCE 정리(Williams 1992)는 이 기울기가 계산 가능함을 보여줍니다: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. 에피소드를 실행합니다. 보상(G)을 계산합니다. 모든 단계에서 `∇ log π_θ(a | s)`에 보상을 곱합니다. 평균을 구합니다. 경사 상승법을 적용합니다. 완료.

2026년의 모든 LLM-RL 알고리즘 — PPO, DPO, GRPO — 은 REINFORCE의 개선판입니다. 이를 완벽히 이해하는 것이 이 단계의 전제 조건이며, 10·07 단계(RLHF 구현)와 10·08 단계(DPO)를 위한 필수 지식입니다.

## 개념

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**정책 경사 정리(Policy gradient theorem).** 파라미터 `θ`로 파라미터화된 모든 정책 `π_θ`에 대해:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

여기서 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}`는 단계 `t`부터의 할인된 누적 보상(discounted return)입니다. 기댓값은 `π_θ`에서 샘플링된 전체 궤적 `τ`에 대해 계산됩니다.

**증명은 간결합니다.** `J(θ) = Σ_τ P(τ; θ) G(τ)`를 기댓값 하에서 미분합니다. `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (로그-미분 트릭)을 사용합니다. `log P(τ; θ) = Σ log π_θ(a_t | s_t) + θ에 의존하지 않는 환경 항`으로 분해합니다. 환경 항은 사라집니다. 두 줄의 대수 연산으로 정리를 얻을 수 있습니다.

**분산 감소 기법(Variance reduction tricks).** 바닐라 REINFORCE는 살인적인 분산을 가집니다 — 누적 보상(return)은 노이즈가 많고, `∇ log π`도 노이즈가 많으며, 그 곱은 매우 노이즈가 많습니다. 두 가지 표준 해결법:

1. **기준선 감산(Baseline subtraction).** `G_t`를 `G_t - b(s_t)`로 대체합니다. 여기서 `b(s_t)`는 `a_t`에 의존하지 않는 임의의 기준선입니다. `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`이므로 불편향입니다. 일반적인 선택: `b(s_t) = V̂(s_t)` (비평가(critic)로 학습) → 액터-크리틱(actor-critic) (레슨 07).
2. **보상-투-고(Reward-to-go).** `Σ_t G_t · ∇ log π_θ(a_t | s_t)`를 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`로 대체합니다. 주어진 행동에 대해 미래 누적 보상만 중요합니다 — 과거 보상은 0-평균 노이즈를 기여합니다.

이 둘을 결합하면:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

를 얻으며, 이는 기준선을 가진 REINFORCE입니다 — A2C(레슨 07)와 PPO(레슨 08)의 직계 조상입니다.

**소프트맥스 정책 파라미터화(Softmax policy parameterization).** 이산 행동에 대한 표준 선택:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

여기서 `f_θ`는 행동별 점수를 출력하는 임의의 신경망입니다. 그래디언트는 깔끔한 형태를 가집니다:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

즉, 취한 행동의 점수에서 정책 하의 기대값을 뺀 값입니다.

**연속 행동을 위한 가우시안 정책(Gaussian policy for continuous actions).** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)`는 닫힌 형태를 가집니다. 이것이 Phase 9 · 07의 SAC에 필요한 전부입니다.

## 구축 방법

### 1단계: softmax 정책 네트워크

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

테이블 형식 환경에는 선형 정책(행동당 하나의 가중치 벡터)을 사용합니다. Atari의 경우 CNN으로 교체하고 softmax 헤드는 유지합니다.

### 2단계: 샘플링 및 로그 확률

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 3단계: 로그 확률 기록 롤아웃

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 4단계: REINFORCE 업데이트

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

`∇ log π(a|s) = e_a - π(·|s)` (행동 `a`의 원-핫 벡터에서 확률들을 뺀 값)는 softmax 정책 그래디언트의 핵심입니다. 반드시 암기하세요.

### 5단계: 베이스라인

최근 에피소드들의 `G`에 대한 이동 평균은 4×4 GridWorld를 실행하기에 충분한 분산 감소 효과를 제공합니다. 수렴에는 약 500개의 에피소드가 필요합니다. 베이스라인을 학습된 `V̂(s)`로 업그레이드하면 액터-크리틱을 구현할 수 있습니다.

## 함정(Pitfalls)

- **폭발하는 기울기(Exploding gradients).** 보상(Returns)이 매우 클 수 있습니다. 항상 배치(Batch)에 걸쳐 `G`를 `~N(0, 1)`로 정규화한 후 `∇ log π`와 곱하세요.
- **엔트로피 붕괴(Entropy collapse).** 정책이 너무 일찍 결정적 행동(near-deterministic action)으로 수렴하여 탐색을 멈추고 갇힙니다. 해결책: 목적 함수에 엔트로피 보너스 `β · H(π(·|s))`를 추가하세요.
- **높은 분산(High variance).** 기본 REINFORCE는 수천 개의 에피소드가 필요합니다. 비평가 기준선(Critic baseline, 레슨 07)이나 TRPO/PPO의 신뢰 영역(Trust region, 레슨 08)이 표준 해결책입니다.
- **샘플 비효율성(Sample inefficiency).** 온-폴리시(On-policy)는 한 번의 업데이트 후 모든 전이(transition)를 버립니다. 중요도 샘플링(Importance sampling)을 통한 오프-폴리시 수정은 데이터를 재사용하지만 분산 증가라는 비용이 발생합니다(PPO의 비율은 클리핑된 IS 가중치입니다).
- **비정상적 기울기(Non-stationary gradients).** 100회 전에 얻은 동일한 기울기는 오래된 `π`를 사용합니다. 온-폴리시 방법은 이 이유로 몇 번의 롤아웃(Rollout)마다 업데이트합니다.
- **신용 할당(Credit assignment).** 보상-투-고(Reward-to-go)가 없으면 과거 보상이 노이즈로 작용합니다. 항상 보상-투-고를 사용하세요.

## 사용 사례

2026년에는 REINFORCE가 직접 실행되는 경우는 거의 없지만, 그 기울기 공식은 모든 곳에 존재합니다:

| 사용 사례 | 파생 방법 |
|----------|---------------|
| 연속 제어 | 가우시안 정책을 사용한 PPO / SAC |
| LLM RLHF | 토큰 수준 정책에서 실행되는 KL 페널티가 있는 PPO |
| LLM 추론 (DeepSeek) | GRPO — 그룹 상대 기준선을 사용한 REINFORCE, 크리틱 없음 |
| 다중 에이전트 | 중앙 집중식 크리틱 REINFORCE (MADDPG, COMA) |
| 이산 행동 로봇공학 | A2C, A3C, PPO |
| 선호도 전용 설정 | DPO — 샘플링 없이 선호도-우도 손실로 재작성된 REINFORCE |

2026년 훈련 스크립트에서 `loss = -advantage * log_prob`를 읽을 때, 이는 기준선이 있는 REINFORCE입니다. 전체 논문(DPO, GRPO, RLOO)은 이 한 줄 위에 있는 분산 감소 기법입니다.

## Ship It

`outputs/skill-policy-gradient-trainer.md`로 저장:

```markdown
---
name: policy-gradient-trainer
description: 주어진 태스크에 대한 REINFORCE / 액터-크리틱 / PPO 학습 설정을 생성하고 분산 문제를 진단합니다.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

환경(이산/연속 액션, 시간 범위, 보상 통계)이 주어졌을 때 다음 내용을 출력합니다:

1. 정책 헤드. 소프트맥스(이산) 또는 가우시안(연속)과 파라미터 수.
2. 베이스라인. 없음(바닐라), 이동 평균, 학습된 `V̂(s)`, 또는 A2C 크리틱.
3. 분산 제어. 기본적으로 보상-투-고 활성화, 리턴 정규화, 그래디언트 클리핑 값.
4. 엔트로피 보너스. 계수 β와 감소 스케줄.
5. 배치 크기. 업데이트당 에피소드 수; 온-폴리시 데이터 신선도 계약.

시간 범위 > 500 스텝인 경우 REINFORCE-베이스라인-없음 설정을 거부합니다. 연속 액션 제어에 소프트맥스 헤드를 사용하는 경우 거부합니다. `β = 0`이면서 관측된 정책 엔트로피가 0.1 미만인 실행은 엔트로피 붕괴로 표시합니다.
```

## 연습 문제

1. **쉬움.** 선형 소프트맥스 정책을 사용하여 4×4 GridWorld에서 REINFORCE를 구현하세요. 베이스라인 없이 1,000 에피소드 동안 학습시키세요. 학습 곡선을 플롯하고, 분산(수익의 표준편차)을 측정하세요.
2. **중간.** 이동 평균 베이스라인을 추가하세요. 다시 학습시킨 후, 샘플 효율성과 분산을 기본 실행과 비교하세요. 베이스라인이 수렴까지의 단계를 얼마나 감소시키나요?
3. **어려움.** 엔트로피 보너스 `β · H(π)`를 추가하세요. `β ∈ {0, 0.01, 0.1, 1.0}`에 대해 스위핑을 수행하세요. 최종 수익과 정책 엔트로피를 플롯하세요. 이 작업에서 최적의 지점은 어디인가요?

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 정책 경사(Policy gradient) | "정책을 직접 훈련시킨다" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`; 로그-미분 트릭(log-derivative trick)에서 유도됨. |
| REINFORCE | "최초의 PG 알고리즘" | Williams (1992); 몬테카를로 반환값(Monte Carlo returns)에 로그-정책 경사(log-policy gradient)를 곱함. |
| 로그-미분 트릭(Log-derivative trick) | "스코어 함수 추정기(Score function estimator)" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; 기댓값의 경사(gradients of expectations)를 다루기 쉽게 만듦. |
| 베이스라인(Baseline) | "분산 감소(Variance reduction)" | `G`에서 빼는 임의의 `b(s)`; `E[b · ∇ log π] = 0`이므로 편향되지 않음(unbiased). |
| 리워드-투-고(Reward-to-go) | "미래 반환값만 중요하다" | 전체 `G_0` 대신 `G_t^{from t}`; 정확하며 분산이 낮음. |
| 엔트로피 보너스(Entropy bonus) | "탐색을 장려한다" | `+β · H(π(·|s))` 항은 정책이 붕괴(collapse)되는 것을 방지함. |
| 온-폴리시(On-policy) | "방금 본 데이터로 훈련한다" | 경사 기댓값은 현재 정책에 대한 것 — 이전 데이터를 직접 재사용할 수 없음. |
| 어드밴티지(Advantage) | "평균보다 얼마나 더 좋은가" | `A(s, a) = G(s, a) - V(s)`; REINFORCE-with-baseline이 곱하는 부호 있는 양(signed quantity). |

## 추가 자료

- [Williams (1992). 연결주의 강화 학습을 위한 간단한 통계적 기울기 추종 알고리즘](https://link.springer.com/article/10.1007/BF00992696) — 원본 REINFORCE 논문.
- [Sutton et al. (2000). 함수 근사를 이용한 강화 학습을 위한 정책 기울기 방법](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 함수 근사를 포함한 현대 정책 기울기 정리.
- [Sutton & Barto (2018). 13장 — 정책 기울기 방법](http://incompleteideas.net/book/RLbook2020.pdf) — 교과서 설명.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — PyTorch 코드와 함께 제공되는 명확한 교육적 설명.
- [Peters & Schaal (2008). 정책 기울기를 이용한 운동 기술 강화 학습](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 분산 감소 및 REINFORCE를 신뢰 영역 계열(TRPO, PPO)과 연결하는 자연 기울기 관점.