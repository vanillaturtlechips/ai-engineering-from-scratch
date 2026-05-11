# Actor-Critic — A2C와 A3C

> REINFORCE는 노이즈가 심합니다. `V̂(s)`를 학습하는 크리틱을 추가하고, 리턴에서 이를 빼면 동일한 기댓값을 가지지만 분산이 훨씬 낮은 어드밴티지를 얻을 수 있습니다. 이것이 바로 Actor-Critic입니다. A2C는 동기식으로 실행하고, A3C는 스레드 간에 실행합니다. 둘 다 모든 현대 딥러닝 강화학습 방법의 정신적 모델입니다.

**유형:** 구축
**언어:** Python
**선수 지식:** 9단계 · 04 (TD 학습), 9단계 · 06 (REINFORCE)
**소요 시간:** ~75분

## 문제

바닐라 REINFORCE는 작동하지만 분산이 끔찍합니다. 몬테카를로 반환값 `G_t`는 에피소드 간에 10배 이상 변동할 수 있습니다. 이 노이즈를 `∇ log π`에 곱하고 평균을 내면, 정책 이동 거리를 DQN 업데이트보다 훨씬 적은 횟수로 이동할 수 있음에도 수천 개의 에피소드가 필요한 그래디언트 추정기가 생성됩니다.

분산은 원시 반환값을 사용하기 때문에 발생합니다. 상태 함수인 베이스라인 `b(s_t)`(학습된 가치 포함)를 빼면 기대값은 변하지 않고 분산은 감소합니다. 가장 다루기 쉬운 베이스라인은 `V̂(s_t)`입니다. 이제 `∇ log π`에 곱해지는 양은 *어드밴티지*가 됩니다:

`A(s, a) = G - V̂(s)`

행동이 평균 이상의 반환값을 생성하면 좋은 행동이고, 평균 미만이면 나쁜 행동입니다. 학습된 크리틱을 사용한 REINFORCE는 *액터-크리틱*입니다. 크리틱은 액터에게 저분산 교사를 제공합니다. 이는 2015년 이후의 모든 심층 정책 방법(A2C, A3C, PPO, SAC, IMPALA)에 해당합니다.

## 개념

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**두 네트워크, 하나의 공유 손실 함수:**

- **Actor** `π_θ(a | s)`: 정책. 행동을 샘플링하는 데 사용. 정책 경사(policy gradient)로 훈련.
- **Critic** `V_φ(s)`: 상태(state)에서 기대 수익(expected return)을 추정. `(V_φ(s) - target)²`를 최소화하도록 훈련.

**이점(advantage).** 두 가지 표준 형태:

- *MC 이점:* `A_t = G_t - V_φ(s_t)`. 편향되지 않음(unbiased), 분산이 높음.
- *TD 이점:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. 편향됨(`V_φ` 사용), 분산이 훨씬 낮음. *TD 잔차(TD residual)* `δ_t`라고도 함.

**n-단계 이점.** 두 형태 사이를 보간:

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1`은 순수 TD. `n = ∞`는 MC. 대부분의 구현은 Atari에 `n = 5`, MuJoCo에서 PPO에 `n = 2048`을 사용.

**일반화된 이점 추정(GAE).** Schulman et al. (2016)은 모든 n-단계 이점에 대해 지수적으로 가중된 평균을 제안:

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

여기서 `λ ∈ [0, 1]`. `λ = 0`은 TD(낮은 분산, 높은 편향). `λ = 1`은 MC(높은 분산, 편향 없음). `λ = 0.95`는 2026년 기본값 — 편향/분산 조절이 원하는 지점에 도달할 때까지 튜닝.

**A2C: 동기식 이점 액터-크리틱.** `N`개의 병렬 환경에서 `T` 단계를 수집. 각 단계에 대한 이점 계산. 결합된 배치로 액터와 크리틱 업데이트. 반복. A3C의 더 간단하고 확장 가능한 변형.

**A3C: 비동기식 이점 액터-크리틱.** Mnih et al. (2016). `N`개의 작업자 스레드를 생성, 각각 환경 실행. 각 작업자는 자체 롤아웃에서 로컬로 그래디언트를 계산한 후 공유 파라미터 서버에 비동기적으로 적용. 리플레이 버퍼 불필요 — 작업자는 서로 다른 궤적을 실행하여 탈상관화. A3C는 CPU에서 대규모 훈련이 가능함을 입증. 2026년에는 GPU 기반 A2C(배치 병렬 환경)가 우세 — GPU는 큰 배치를 원하기 때문.

**결합된 손실 함수.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

세 가지 항: 정책 경사 손실, 가치 회귀, 엔트로피 보너스. `c_v ~ 0.5`, `c_e ~ 0.01`은 표준 시작점.

## 구축 방법

### 단계 1: 크리틱

선형 크리틱 `V_φ(s) = w · features(s)`를 MSE로 업데이트:

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

테이블 형식 환경에서 크리틱은 수백 번의 에피소드 안에 수렴합니다. Atari에서는 선형 크리틱을 공유 CNN 트렁크 + 값 헤드로 대체합니다.

### 단계 2: n-스텝 어드밴티지

길이 `T`의 롤아웃과 부트스트랩된 최종 `V(s_T)`가 주어졌을 때:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns`는 크리틱의 타깃입니다. `advantages`는 `∇ log π`에 곱해지는 값입니다.

### 단계 3: 통합 업데이트

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # 크리틱
    critic_update(w, x, target_v, lr_v)

    # 액터
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

온-폴리시, 업데이트당 하나의 롤아웃, 액터와 크리틱에 별도의 학습률 사용.

### 단계 4: 병렬화 (A3C vs A2C)

- **A3C:** `N`개의 스레드를 생성합니다. 각각은 자체 환경과 자체 순전파를 실행합니다. 주기적으로 공유 마스터에 그래디언트 업데이트를 푸시합니다. 마스터에 락이 없음 — 경주는 허용되며 노이즈를 추가합니다.
- **A2C:** 단일 프로세스에서 `N`개의 환경 인스턴스를 실행하고, 관측값을 `[N, obs_dim]` 배치로 쌓아 배치 순전파, 배치 역전파를 수행합니다. GPU 활용도가 높고 결정론적이며 추론이 용이합니다. 2026년 기준 기본 방식입니다.

우리의 토이 코드는 명확성을 위해 단일 스레드로 작성되었습니다. 배치 A2C로 재작성하는 것은 NumPy로 3줄이면 충분합니다.

## 함정

- **액터 기울기 이전의 크리틱 편향.** 크리틱이 무작위일 경우 기준선(baseline)이 정보를 제공하지 않으며 순수 노이즈로 훈련하게 됩니다. 정책 기울기(policy gradient)를 활성화하기 전에 크리틱을 수백 단계 예열하거나 느린 액터 학습률(learning rate)을 사용하세요.
- **이점(advantage) 정규화.** 배치(batch)별로 이점을 평균 0/표준편차 1로 정규화하세요. 거의 비용이 들지 않으면서 훈련을 크게 안정화합니다.
- **공유 트렁크(shared trunk).** 이미지 입력에 대해 액터와 크리틱에 공유 특징 추출기(feature extractor)를 사용하세요. 헤드는 분리합니다. 공유 특징은 두 손실(loss)에 대해 무료로 활용됩니다.
- **온-폴리시 계약(on-policy contract).** A2C는 데이터를 정확히 한 번의 업데이트에 재사용합니다. 더 많이 사용하면 기울기가 편향되며(중요도 샘플링(importance-sampling) 보정이 PPO가 추가하는 것).
- **엔트로피 붕괴(entropy collapse).** `c_e > 0`이 없으면 정책이 수백 업데이트 만에 거의 결정적(deterministic)이 되어 탐색을 멈춥니다.
- **보상 스케일.** 이점(advantage) 크기는 보상 스케일에 의존합니다. 작업 간 일관된 기울기 크기를 위해 보상을 정규화하세요(예: 실행 표준편차(running-std)로 나누기).

## 사용 방법

A2C/A3C는 2026년 현재 최종 선택되는 경우는 드물지만, 이후 모든 방법이 개선하는 기본 아키텍처입니다:

| 방법 | A2C와의 관계 |
|--------|----------------|
| PPO | 다중 에폭 업데이트를 위한 클리핑된 중요도 비율(importance ratio)이 추가된 A2C |
| IMPALA | V-trace 오프-폴리시 보정이 추가된 A3C |
| SAC (9단계 · 07) | 소프트-값(soft-value) 크리틱을 사용하는 오프-폴리시 A2C (다음 강의에서 설명) |
| GRPO (9단계 · 12) | 크리틱(critic) 없이 그룹-상대 어드밴티지(group-relative advantage)를 사용하는 A2C |
| DPO | 선호도 순위(preference-ranking) 손실로 축소된 A2C, 샘플링 없음 |
| AlphaStar / OpenAI Five | 리그 훈련(league training) + 모방 사전 훈련(imitation pre-training)이 추가된 A2C |

2026년 논문에서 "어드밴티지(advantage)"를 본다면 액터-크리틱(actor-critic)을 떠올리세요.

## Ship It

저장 위치: `outputs/skill-actor-critic-trainer.md`

```markdown
---
name: actor-critic-trainer
description: 주어진 환경에 대해 어드밴티지 추정 및 손실 가중치가 지정된 A2C / A3C / GAE 구성을 생성합니다.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

환경과 컴퓨팅 예산을 고려하여 다음을 출력합니다:

1. 병렬화. A2C (GPU 배치) vs A3C (CPU 비동기) 및 워커 수.
2. 롤아웃 길이 T. 업데이트당 환경별 단계 수.
3. 어드밴티지 추정기. n-step 또는 GAE(λ); λ 값 지정.
4. 손실 가중치. `c_v` (값 함수), `c_e` (엔트로피), 그래디언트 클리핑.
5. 학습률. 액터와 크리틱 (별도 사용 시 분리).

단일 워커 A2C를 지평선(horizon) > 1000인 환경에 적용하는 것을 거부합니다(너무 온-폴리시, 너무 느림). 어드밴티지 정규화 없이 배송을 거부합니다. `c_e = 0`이면서 관측된 엔트로피가 0.1 미만인 실행을 엔트로피 붕괴(entropy-collapsed)로 표시합니다.
```

## 연습 문제

1. **쉬움.** 4×4 GridWorld에서 MC 어드밴티지(`G_t - V(s_t)`)를 사용하는 액터-크리틱을 학습시켜 보세요. Lesson 06의 이동 평균 기준선을 사용한 REINFORCE와 샘플 효율성을 비교해 보세요.
2. **중간.** TD-잔차 어드밴티지(`r + γ V(s') - V(s)`)로 전환해 보세요. 어드밴티지 배치의 분산을 측정해 보세요. 얼마나 감소하나요?
3. **어려움.** GAE(λ)를 구현해 보세요. `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}` 범위에서 실험을 진행하고 최종 보상과 샘플 효율성을 그래프로 그려 보세요. 이 작업에서 편향/분산 트레이드오프의 최적 지점은 어디인가요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| Actor | "정책 네트워크" | `π_θ(a|s)`, 정책 기울기(policy gradient)로 업데이트됨. |
| Critic | "가치 네트워크" | `V_φ(s)`, 리턴/시간차(TD) 타겟에 대한 MSE(평균 제곱 오차) 회귀로 업데이트됨. |
| Advantage | "평균보다 얼마나 나은지" | `A(s, a) = Q(s, a) - V(s)` 또는 그 추정치. `∇ log π`의 승수(multiplier). |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`; 1단계 어드밴티지(advantage) 추정치. |
| GAE | "보간 조절 장치" | `λ`로 파라미터화된 n-단계 어드밴티지의 지수 가중 합. |
| A2C | "동기식 액터-크리틱" | 환경들 간 배치 처리; 롤아웃(rollout)당 1회 그래디언트 단계. |
| A3C | "비동기식 액터-크리틱" | 워커 스레드들이 공유 파라미터 서버에 그래디언트를 푸시. 원본 논문 방식; 2026년 기준 덜 일반적. |
| Bootstrap | "지평선에서 V 사용" | 롤아웃을 절단하고, 합계를 닫기 위해 `γ^n V(s_{t+n})` 추가. |

## 추가 자료

- [Mnih et al. (2016). 비동기 심층 강화 학습 방법](https://arxiv.org/abs/1602.01783) — A3C, 원본 비동기 액터-크리틱 논문.
- [Schulman et al. (2016). 일반화된 어드밴티지 추정을 활용한 고차원 연속 제어](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — 액터-크리틱 방법](http://incompleteideas.net/book/RLbook2020.pdf) — 기초 이론; 크리틱이 신경망일 경우 함수 근사에 대한 Ch. 9와 함께 참조.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — V-trace 오프-폴리시 보정을 활용한 확장 가능한 분산 액터-크리틱.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — A2C/PPO 구현체 참고용 프로덕션 코드.
- [Konda & Tsitsiklis (2000). 액터-크리틱 알고리즘](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 2-타임스케일 액터-크리틱 분해에 대한 기초 수렴 결과.