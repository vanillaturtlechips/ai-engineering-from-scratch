# 몬테 카를로 방법 — 완전한 에피소드로부터 학습

> 동적 프로그래밍은 모델이 필요합니다. 몬테 카를로는 에피소드만 있으면 됩니다. 정책을 실행하고, 수익을 관찰하고, 평균을 내세요. 강화학습에서 가장 간단한 아이디어이자 이후 모든 것을 열어주는 방법.

**유형:** 구축
**언어:** Python
**선수 지식:** 9단계 · 01 (MDP), 9단계 · 02 (동적 프로그래밍)
**소요 시간:** ~75분

## 문제 정의

동적 계획법(Dynamic Programming, DP)은 우아하지만, 모든 상태와 행동에 대해 `P(s' | s, a)`를 쿼리할 수 있다고 가정합니다. 실제 세계에서는 거의 아무것도 이런 식으로 작동하지 않습니다. 로봇은 관절 토크 이후 카메라 픽셀 분포를 해석적으로 계산할 수 없습니다. 가격 결정 알고리즘은 모든 가능한 고객 반응을 적분할 수 없습니다. LLM(Large Language Model)은 토큰 이후 가능한 모든 연속을 열거할 수 없습니다.

환경으로부터 *샘플링*할 수 있는 능력만 있으면 되는 방법이 필요합니다. 정책을 실행하고, 궤적 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`를 얻은 후, 이를 사용해 값을 추정해야 합니다. 그것이 바로 몬테카를로(Monte Carlo, MC) 방법입니다.

DP에서 MC로의 전환은 철학적으로 중요합니다: *알려진 모델 + 정확한 백업*에서 *샘플된 롤아웃 + 평균화된 리턴*으로 이동합니다. 분산은 증가하지만 적용 가능성은 폭발적으로 확장됩니다. 이 강의 이후의 모든 강화학습 알고리즘 — TD, Q-러닝, REINFORCE, PPO, GRPO — 은 본질적으로 몬테카를로 추정기이며, 때로는 부트스트래핑이 추가되기도 합니다.

## 개념

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**핵심 아이디어, 한 줄로:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)` 여기서 `G^{(i)}(s)`는 정책 `π` 하에서 상태 `s`를 방문한 후 관측된 보상입니다.

**첫 방문(First-visit) vs 매 방문(Every-visit) MC.** 상태 `s`를 여러 번 방문하는 에피소드가 주어졌을 때, 첫 방문 MC는 첫 번째 방문에서의 보상만 카운트하고, 매 방문 MC는 모든 방문을 카운트합니다. 둘 다 극한에서 편향되지 않습니다. 첫 방문은 분석이 더 간단합니다(iid 샘플). 매 방문은 에피소드당 더 많은 데이터를 사용하며 일반적으로 실제에서 더 빠르게 수렴합니다.

**증분 평균(Incremental mean).** 모든 보상을 저장하는 대신 실행 평균을 업데이트합니다:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

재구성: `V_new = V_old + α · (target - V_old)` 여기서 `α = 1/n`. `1/n`을 상수 스텝 사이즈 `α ∈ (0, 1)`로 바꾸면 `π`의 변화를 추적하는 비정상(non-stationary) MC 추정기가 됩니다. 이 변화가 MC에서 TD로, 그리고 모든 현대 RL 알고리즘으로 가는 도약입니다.

**탐색이 이제 문제가 됩니다.** DP는 열거로 모든 상태를 처리했습니다. MC는 정책이 방문하는 상태만 봅니다. `π`가 결정론적이면 상태 공간의 전체 영역이 샘플링되지 않고, 해당 가치 추정은 영원히 0으로 유지됩니다. 역사적 순서대로 세 가지 해결책:

1. **탐색 시작(Exploring starts).** 각 에피소드를 무작위 `(s, a)` 쌍에서 시작합니다. 커버리지를 보장하지만 실제로는 비현실적입니다(로봇을 임의의 상태로 "리셋"할 수 없음).
2. **ε-탐욕(ε-greedy).** 현재 `Q`에 대해 탐욕적으로 행동하지만, 확률 `ε`로 무작위 행동을 선택합니다. 모든 상태-행동 쌍이 점근적으로 샘플링됩니다.
3. **오프-정책 MC(Off-policy MC).** 행동 정책 `μ` 하에서 데이터를 수집하고, 중요도 샘플링을 통해 목표 정책 `π`에 대해 학습합니다. 분산이 높지만 DQN과 같은 리플레이 버퍼 방법으로 가는 교두보입니다.

**몬테 카를로 제어(Monte Carlo Control).** 정책 반복과 마찬가지로 평가 → 개선 → 평가를 반복하지만, 평가는 샘플링 기반입니다:

1. `π`를 실행하여 에피소드를 얻습니다.
2. 관측된 보상으로 `Q(s, a)`를 업데이트합니다.
3. `Q`에 대해 `π`를 ε-탐욕으로 만듭니다.
4. 반복합니다.

약한 조건 하에서(모든 쌍이 무한히 자주 방문되고, `α`가 Robbins-Monro 조건을 만족) 확률 1로 `Q*`와 `π*`로 수렴합니다.

## 구축 방법

### 단계 1: 롤아웃 → (s, a, r) 목록

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

모델 없이 `env.reset()`과 `env.step(s, a)`만 사용. Gym 환경과 동일한 인터페이스지만 간소화됨.

### 단계 2: 반환값 계산 (역방향 스윕)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

한 번의 패스로 `O(T)` 시간 복잡도. 역방향 재귀 `G_t = r_{t+1} + γ G_{t+1}`를 통해 재계산 방지.

### 단계 3: 첫 방문 몬테카를로 평가

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

핵심 3줄: 첫 방문 시 상태 표시, 카운트 증가, 이동 평균 업데이트.

### 단계 4: ε-탐욕적 몬테카를로 제어 (온-폴리시)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 단계 5: DP 기준선과 비교

`V^π`에 대한 몬테카를로 추정값은 에피소드 수 → ∞일 때 Lesson 02의 DP 결과와 일치해야 함. 실제로 4×4 GridWorld에서 50,000 에피소드 실행 시 DP 정답과 `~0.1` 이내 오차 발생.

## 함정

- **무한한 에피소드.** MC는 에피소드가 *종료*되어야 합니다. 정책이 영원히 반복될 수 있다면 `max_steps`를 제한하고 해당 제한을 암묵적 실패로 처리하세요. 무작위 정책을 사용하는 GridWorld는 종종 시간 초과됩니다 — 이는 정상적인 현상이며, 단지 올바르게 카운트하기만 하면 됩니다.
- **분산.** MC는 전체 리턴을 사용합니다. 긴 에피소드에서는 분산이 매우 큽니다 — 끝에서 한 번의 불운한 보상이 `V(s_0)`를 동일한 양만큼 이동시킵니다. TD 방법(레슨 04)은 부트스트래핑을 통해 이를 줄입니다.
- **상태 커버리지.** 동률 상태에서 탐욕적 MC는 단 하나의 행동만 시도합니다. 반드시 탐색(ε-탐욕적, 탐색 시작, UCB)해야 합니다.
- **비정적 정책.** `π`가 변경되면(예: MC 제어에서) 오래된 리턴은 다른 정책에서 온 것입니다. 상수-α MC는 이를 처리하지만, 샘플 평균 MC는 처리하지 못합니다.
- **오프-정책 중요도 샘플링.** 가중치 `π(a|s)/μ(a|s)`는 궤적을 따라 곱해집니다. 분산은 지평선과 함께 폭발적으로 증가합니다. 결정별 가중치 IS로 제한하거나 TD로 전환하세요.

## 사용 사례

몬테 카를로 방법의 2026년 역할:

| 사용 사례 | MC 사용 이유 |
|----------|--------------|
| 단기 게임(블랙잭, 포커) | 에피소드가 자연스럽게 종료됨; 리턴 값이 명확함. |
| 기록된 정책 오프라인 평가 | 저장된 궤적에 대한 평균 할인 리턴 계산. |
| 몬테 카를로 트리 탐색(AlphaZero) | 트리 리프에서 MC 롤아웃이 선택을 안내함. |
| LLM 강화학습 평가 | 주어진 정책에 대해 샘플링된 완성본들의 평균 보상 계산. |
| PPO에서 베이스라인 추정 | 어드밴티지 타겟 `A_t = G_t - V(s_t)`는 MC `G_t`를 사용함. |
| 강화학습 교육 | 실제로 작동하는 가장 간단한 알고리즘 — 부트스트래핑을 제거하여 핵심 개념 학습. |

현대 딥러닝 강화학습 알고리즘(PPO, SAC)은 `n`-단계 리턴 또는 GAE(Generalized Advantage Estimation)를 통해 순수 MC(전체 리턴)와 순수 TD(1-단계 부트스트랩) 사이를 보간합니다. 두 끝점은 동일한 추정기의 인스턴스입니다.

## Ship It

`outputs/skill-mc-evaluator.md`로 저장:

```markdown
---
name: mc-evaluator
description: 몬테 카를로 롤아웃을 통해 정책을 평가하고, 사용 가능한 경우 DP-비교를 포함한 수렴 보고서를 생성합니다.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

환경(에피소딕, reset+step API 지원)과 정책이 주어졌을 때 다음 내용을 출력합니다:

1. 방법. 첫 방문(first-visit) vs 매 방문(every-visit) MC. 선택 이유.
2. 에피소드 예산. 목표 에피소드 수, 분산 진단, 예상 표준 오차.
3. 탐색 계획. ε 스케줄(필요한 경우) 또는 탐색 시작(exploring starts).
4. 골드 스탠더드 비교. 테이블 형식인 경우 DP-최적 V*; 그 외에는 Q-러닝 / PPO 기준선의 경계값.
5. 종료 조건. 최대 단계 제한, 타임아웃, 비종료 궤적 처리 방법.

유한한 시간 범위 제한이 없는 비에피소딕 작업에서는 MC 실행을 거부합니다. 테이블 형식 작업의 경우 상태당 100개 미만의 에피소드로 V^π 추정치를 보고하지 않습니다. 제로 분산(zero-variance) 액션을 가진 정책은 탐색 위험으로 표시합니다.
```

## 연습 문제

1. **쉬움.** 4×4 GridWorld에서 균일 무작위 정책(uniform-random policy)에 대한 첫 방문 MC 평가(first-visit MC evaluation)를 구현하세요. 10,000개의 에피소드를 실행하세요. 에피소드 횟수에 따른 `V(0,0)` 값을 DP 정답과 비교하여 그래프로 나타내세요.
2. **중간.** `ε ∈ {0.01, 0.1, 0.3}`인 ε-탐욕적(epsilon-greedy) MC 제어(control)를 구현하세요. 20,000 에피소드 후 평균 보상(mean return)을 비교하세요. 곡선은 어떤 모양인가요? 편향-분산 트레이드오프(bias-variance tradeoff)는 어디에 존재하나요?
3. **어려움.** 중요도 샘플링(importance sampling)을 사용한 *오프-정책*(off-policy) MC를 구현하세요: 균일 무작위 정책 `μ` 하에서 데이터를 수집하고, 결정적 최적 정책 `π`에 대한 `V^π`를 추정하세요. 일반 중요도 샘플링(plain IS) vs 결정별 중요도 샘플링(per-decision IS) vs 가중치 중요도 샘플링(weighted IS)을 비교하세요. 어떤 방법이 분산이 가장 낮은가요?

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 몬테 카를로(Monte Carlo) | "무작위 샘플링" | 분포에서 iid 샘플을 평균화하여 기댓값을 추정합니다. |
| 리턴 `G_t` | "미래 보상" | 단계 `t`부터 에피소드 종료까지의 할인된 보상 합계: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| 첫 방문 몬테 카를로(First-visit MC) | "각 상태를 한 번만 계산" | 에피소드 내 첫 번째 방문만 가치 추정에 기여합니다. |
| 모든 방문 몬테 카를로(Every-visit MC) | "모든 방문 사용" | 모든 방문이 기여하며, 약간 편향되지만 샘플 효율성이 더 높습니다. |
| ε-탐욕적(ε-greedy) | "탐색 노이즈" | 확률 `1-ε`로 탐욕적 행동을 선택하고, 확률 `ε`로 무작위 행동을 선택합니다. |
| 중요도 샘플링(Importance sampling) | "잘못된 분포 샘플링 보정" | `π(a|s)/μ(a|s)` 곱으로 리턴을 재가중하여 `μ` 데이터로 `V^π`를 추정합니다. |
| 온-폴리시(On-policy) | "내 데이터로 학습" | 목표 정책 = 행동 정책. 기본 몬테 카를로, PPO, SARSA. |
| 오프-폴리시(Off-policy) | "다른 사람의 데이터로 학습" | 목표 정책 ≠ 행동 정책. 중요도 샘플링 몬테 카를로, Q-러닝, DQN.

## 추가 자료

- [Sutton & Barto (2018). Ch. 5 — 몬테카를로 방법](http://incompleteideas.net/book/RLbook2020.pdf) — 표준적인 설명.
- [Singh & Sutton (1996). 교체 적격 추적(Replacing Eligibility Traces)을 이용한 강화 학습](https://link.springer.com/article/10.1007/BF00114726) — 첫 방문(first-visit) vs 매 방문(every-visit) 분석.
- [Precup, Sutton, Singh (2000). 오프-정책 정책 평가를 위한 적격 추적(Eligibility Traces)](http://incompleteideas.net/papers/PSS-00.pdf) — 오프-정책 몬테카를로 및 분산 제어.
- [Mahmood et al. (2014). 오프-정책 학습을 위한 가중 중요도 샘플링(Weighted Importance Sampling)](https://arxiv.org/abs/1404.6362) — 현대적인 저분산 중요도 샘플링(IS) 추정기.
- [Tesauro (1995). TD-개먼(TD-Gammon), 자기 학습 백개먼 프로그램](https://dl.acm.org/doi/10.1145/203330.203343) — 몬테카를로/시간차(TD) 자기 대국(self-play)이 초인적 플레이로 수렴하는 첫 대규모 실증 사례; 이 단계의 후반부 모든 강의의 개념적 선구자.