# 시간적 차분 — Q-러닝 & SARSA

> 몬테카를로는 에피소드가 끝날 때까지 기다립니다. TD는 다음 가치 추정을 부트스트랩하여 매 단계 후 업데이트합니다. Q-러닝은 오프-정책이며 낙관적이고, SARSA는 온-정책이며 신중합니다. 둘 다 코드 한 줄입니다. 둘 다 이 단계의 모든 심층 강화학습 방법의 기반이 됩니다.

**유형:** 구축  
**언어:** Python  
**선수 지식:** 9단계 · 01 (MDP), 9단계 · 02 (동적 계획법), 9단계 · 03 (몬테카를로)  
**소요 시간:** ~75분

## 문제 정의

몬테 카를로(Monte Carlo)는 작동하지만 두 가지 비용이 큰 요구사항이 있습니다. 종료되는 에피소드가 필요하며, 최종 반환값(return)이 들어온 후에야 업데이트가 이루어집니다. 만약 에피소드가 1,000단계라면 몬테 카를로는 1,000단계를 기다린 후에야 어떤 업데이트도 수행하지 않습니다. 이는 높은 분산(high-variance)과 낮은 편향(low-bias)을 가지지만 실제로는 느린 방법입니다.

동적 프로그래밍(Dynamic Programming)은 정반대의 특성을 가집니다 — 분산이 없는 부트스트랩 백업(zero-variance bootstrapped backups) — 하지만 알려진 모델(model)이 필요합니다.

시간차(Temporal Difference, TD) 학습은 이 둘의 중간을 선택합니다. 단일 전이 `(s, a, r, s')`에서 `r + γ V(s')`라는 1단계 목표(one-step target)를 형성하고 `V(s)`를 이 목표 쪽으로 조정합니다. 모델이 필요 없으며, 완전한 에피소드도 필요하지 않습니다. RHS에서 근사된 `V`를 사용함으로써 편향(bias)이 발생하지만, 몬테 카를로보다 분산이 극적으로 낮고 첫 단계부터 온라인 업데이트가 가능합니다.

이것이 현대 강화학습(RL)의 모든 것 — DQN, A2C, PPO, SAC — 이 돌아가는 중심축입니다. 이 강의의 9단계에서 다룰 나머지 내용은 이 레슨에서 구현할 1단계 TD 업데이트를 기반으로 한 함수 근사(function approximation)와 다양한 기법들의 계층입니다.

## 개념

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V에 대한 TD(0) 업데이트:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

대괄호 안의 값은 TD 오차 `δ = r + γ V(s') - V(s)`입니다. 이는 MC에서의 `G_t - V(s_t)`와 온라인 유사 개념입니다. 수렴을 위해서는 `α`가 Robbins-Monro 조건(`Σ α = ∞`, `Σ α² < ∞`)을 만족하고 모든 상태가 무한히 자주 방문되어야 합니다.

**Q-learning.** 제어(control)를 위한 off-policy TD 방법:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max`는 에이전트가 실제로 취하는 행동과 무관하게 `s'` 이후 *탐욕적(greedy) 정책*이 따를 것이라고 가정합니다. 이 분리(decoupling)로 Q-learning은 `Q*`를 학습하는 반면 에이전트는 ε-greedy로 탐색합니다. Mnih et al. (2015)는 이를 Atari용 심층 Q-learning으로 변환했습니다(레슨 05).

**SARSA.** on-policy TD 방법:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

이름은 튜플 `(s, a, r, s', a')`에서 유래합니다. SARSA는 탐욕적 `argmax`가 아닌 에이전트가 실제로 다음에 취하는 행동 `a'`를 사용합니다. 실행 중인 ε-greedy `π`에 대한 `Q^π`로 수렴하며, `ε → 0`일 때 `Q*`가 됩니다.

**절벽 걷기(cliff-walking) 차이.** 고전적인 절벽 걷기 과제(절벽 추락 시 보상 -100)에서 Q-learning은 절벽 가장자리를 따라 최적 경로를 학습하지만 탐색 중 가끔 페널티를 받습니다. SARSA는 탐색 노이즈를 Q-값에 반영하므로 절벽에서 한 걸음 떨어진 안전한 경로를 학습합니다. 학습이 완료되면 둘 다 `ε → 0`에서 최적에 도달합니다. 실제로 중요합니다: 배포 시 탐색이 실제로 발생할 때 SARSA의 행동은 더 보수적입니다.

**Expected SARSA.** `Q(s', a')`를 `π` 하의 기대값으로 대체:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

SARSA보다 분산이 낮고(`a'` 샘플링 없음), 동일한 on-policy 타겟을 가집니다. 현대 교과서에서 종종 기본값으로 사용됩니다.

**n-step TD 및 TD(λ).** 부트스트랩 전 `n` 단계를 기다려 TD(0)와 MC를 보간합니다. `n=1`은 TD, `n=∞`는 MC입니다. TD(λ)는 모든 `n`에 대해 기하 가중 `(1-λ)λ^{n-1}`로 평균을 냅니다. 대부분의 심층 강화학습은 `n`을 3에서 20 사이로 사용합니다.

## 구축

### 단계 1: ε-탐욕 정책에서의 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

8줄. Q-러닝과의 *유일한* 차이는 타겟 라인입니다.

### 단계 2: Q-러닝

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max`는 타겟과 행동을 분리합니다. 이 하나의 기호가 온-정책과 오프-정책의 차이입니다.

### 단계 3: 학습 곡선

100에피소드당 평균 보상 추적. Q-러닝은 단순한 결정적 GridWorld에서 더 빠르게 수렴하며, SARSA는 절벽 걷기(cliff-walking)에서 더 보수적입니다. `code/main.py`의 4×4 GridWorld에서는 두 방법 모두 약 2,000 에피소드 후 `α=0.1, ε=0.1` 설정으로 최적에 근접합니다.

### 단계 4: DP 진실과 비교

가치 반복(Lesson 02)을 실행하여 `Q*`를 구합니다. `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`를 확인하세요. 건강한 테이블 기반 TD 에이전트는 10,000 에피소드 후 4×4 GridWorld에서 약 `~0.5` 이내로 수렴합니다.

## 함정(Pitfalls)

- **초기 Q 값(Q values)의 중요성.** 낙관적 초기화(Optimistic init)는 탐색(Exploration)을 촉진합니다. 예를 들어, 음수 보상(Negative-reward) 작업에서 `Q = 0`으로 초기화하면 탐색이 장려됩니다. 반면 비관적 초기화(Pessimistic init)는 탐욕적 정책(Greedy policy)을 영원히 갇히게 할 수 있습니다.
- **α(알파) 스케줄.** 상수 `α`는 비정상성(Non-stationary) 문제에 적합합니다. 이론적으로 수렴을 보장하는 `α_n = 1/n`은 실제로는 너무 느립니다. `α`를 `[0.05, 0.3]` 범위로 고정하고 학습 곡선을 모니터링하세요.
- **ε(엡실론) 스케줄.** 높은 값(`ε=1.0`)으로 시작해 `ε=0.05`로 감소시킵니다. "GLIE"(Greedy in the Limit with Infinite Exploration)는 수렴 조건입니다.
- **Q-러닝의 최대 편향(Max bias).** `Q` 값이 노이즈가 있을 때 `max` 연산자는 상향 편향(Upward bias)을 유발합니다. 이는 과대추정(Overestimation)으로 이어집니다. Hasselt의 Double Q-러닝(Lesson 05의 DDQN에서 사용)은 두 개의 Q 테이블로 이를 해결합니다.
- **비종료 에피소드(Non-terminating episodes).** TD는 종료 상태(Terminal) 없이도 학습할 수 있지만, 단계 수를 제한하거나 제한 지점에서 부트스트랩(Bootstrap)을 올바르게 처리해야 합니다. 표준 접근 방식: 제한 지점을 비종료 상태로 간주하고 부트스트랩을 계속합니다.
- **상태 해싱(State hashing).** 상태가 튜플(Tuple) 또는 텐서(Tensor)인 경우 해시 가능한 키(Hashable key)를 사용하세요. 리스트(List)가 아닌 튜플(Tuple), 원시 값이 아닌 반올림된 부동 소수점(Rounded floats)의 튜플을 사용합니다.

## 사용 방법

2026년 TD(시간 차) 환경 개요:

| 작업 | 방법 | 이유 |
|------|--------|--------|
| 소규모 테이블 형식 환경 | Q-러닝(Q-learning) | 최적의 정책을 직접 학습합니다. |
| 온-정책(On-policy) 안전-중요(Safety-critical) | SARSA / 기대 SARSA(Expected SARSA) | 탐색 중 보수적입니다. |
| 고차원 상태 | DQN(Phase 9 · 05) | 재생 버퍼(replay buffer)와 타깃 네트워크(target net)를 사용한 신경망 Q-함수. |
| 연속 행동 | SAC / TD3(Phase 9 · 07) | Q-네트워크에 대한 TD 업데이트; 정책 네트워크(policy net)가 행동을 출력합니다. |
| LLM 강화학습(보상 모델 기반) | PPO / GRPO(Phase 9 · 08, 12) | GAE(Generalized Advantage Estimation)를 통한 TD 스타일 어드밴티지(advantage)를 사용한 액터-크리틱(actor-critic). |
| 오프라인 강화학습 | CQL / IQL(Phase 9 · 08) | 보수적 정규화(conservative regularization)를 적용한 Q-러닝. |

2026년 논문에서 접하는 "강화학습"의 90%는 Q-러닝 또는 SARSA의 변형입니다. 더 깊이 읽기 전에 테이블 형식 업데이트를 손에 익혀두세요.

## Ship It

`outputs/skill-td-agent.md`로 저장:

```markdown
---
name: td-agent
description: 테이블 형식 또는 소규모 특징 RL 작업을 위해 Q-러닝, SARSA, Expected SARSA 중 선택.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

테이블 형식 또는 소규모 특징 환경이 주어졌을 때 다음을 출력:

1. 알고리즘. Q-러닝 / SARSA / Expected SARSA / n-스텝 변형. 온-정책 대 오프-정책 및 분산과 연결된 한 문장 설명.
2. 하이퍼파라미터. α(학습률), γ(할인율), ε(탐욕 정책 임계값), 감쇠 스케줄.
3. 초기화. Q_0 값(낙관적 vs 0) 및 근거.
4. 수렴 진단. 목표 학습 곡선, DP 가능 시 `|Q - Q*|` 확인.
5. 배포 시 주의 사항. 추론 시 탐색은 어떻게 동작할 것인가? SARSA의 보수적 접근이 필요한가?

상태 공간이 10⁶을 초과하는 경우 테이블 형식 TD 적용을 거부. 최대 편향 주의 사항 없이 Q-러닝 에이전트 배포를 거부. ε이 1.0으로 고정된 상태에서 훈련된 에이전트(활용 단계 없음)를 플래그 처리.
```

## 연습 문제

1. **쉬움.** 4×4 GridWorld에서 Q-learning과 SARSA를 구현하세요. 2,000 에피소드 동안 학습 곡선(100 에피소드당 평균 리턴)을 플롯하세요. 어떤 방법이 더 빠르게 수렴하나요?
2. **중간.** 절벽 걷기 환경(4×12, 마지막 행은 보상 -100의 절벽이며 시작 지점으로 리셋)을 구축하세요. Q-learning과 SARSA의 최종 정책을 비교하세요. 각 방법이 취하는 경로를 스크린샷으로 찍으세요. 어떤 방법이 절벽에 더 가까운 경로를 선택하나요?
3. **어려움.** Double Q-learning을 구현하세요. 노이즈가 있는 보상 GridWorld(단계별 보상에 가우시안 노이즈 σ=5 추가)에서 Q-learning이 `V*(0,0)`을 의미 있는 수준으로 과대평가하는 반면 Double Q-learning은 그렇지 않음을 보여주세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| TD 오차 | "업데이트 신호" | `δ = r + γ V(s') - V(s)`, 부트스트랩 잔차. |
| TD(0) | "1-스텝 TD" | 다음 상태 추정치만 사용하여 매 전이 후 업데이트. |
| Q-러닝 | "오프-정책 RL 101" | 다음 상태 행동에 대한 `max`를 사용한 TD 업데이트; 행동 정책과 무관하게 `Q*`를 학습. |
| SARSA | "온-정책 Q-러닝" | 실제 다음 행동을 사용한 TD 업데이트; 현재 ε-탐욕적 정책 π에 대한 `Q^π`를 학습. |
| 기대 SARSA | "분산이 낮은 SARSA" | 샘플링된 `a'`를 π 하의 기댓값으로 대체. |
| GLIE | "올바른 탐색 일정" | 무한 탐색 한계에서 탐욕적(Greedy in the Limit with Infinite Exploration); Q-러닝 수렴에 필요. |
| 부트스트래핑 | "목표에 현재 추정치 사용" | TD와 MC를 구분하는 요소. 편향의 원인이지만 분산 감소 효과가 큼. |
| 최대화 편향 | "Q-러닝은 과대추정" | 노이즈가 있는 추정치에 대한 `max`는 상향 편향됨; 더블 Q-러닝으로 해결.

## 추가 자료

- [Watkins & Dayan (1992). Q-러닝(Q-learning)](https://link.springer.com/article/10.1007/BF00992698) — 원본 논문 및 수렴 증명.
- [Sutton & Barto (2018). Ch. 6 — 시간차 학습(Temporal-Difference Learning)](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-러닝, 기대 SARSA(Expected SARSA).
- [Hasselt (2010). 더블 Q-러닝(Double Q-learning)](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 최대화 편향(maximization bias) 해결 방법.
- [Seijen, Hasselt, Whiteson, Wiering (2009). 기대 SARSA에 대한 이론적 및 실증적 분석(A Theoretical and Empirical Analysis of Expected SARSA)](https://ieeexplore.ieee.org/document/4927542) — 기대 SARSA(Expected SARSA) 동기.
- [Rummery & Niranjan (1994). 연결주의 시스템을 이용한 온라인 Q-러닝(On-line Q-learning using connectionist systems)](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — SARSA(당시에는 "수정된 연결주의 Q-러닝"으로 불림")라는 용어를 처음 사용한 논문.
- [Sutton & Barto (2018). Ch. 7 — n-단계 부트스트래핑(n-step Bootstrapping)](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)를 TD(n)로 일반화, Q-러닝에서 적격 추적(eligibility traces) 및 이후 PPO의 GAE로의 발전 경로.