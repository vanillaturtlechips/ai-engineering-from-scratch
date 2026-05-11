# 동적 계획법 — 정책 반복 & 가치 반복

> 동적 계획법은 치트키를 사용한 강화학습입니다. 이미 전이 및 보상 함수를 알고 있으며, `V` 또는 `π`가 수렴할 때까지 벨만 방정식을 반복적으로 적용합니다. 이는 모든 샘플링 기반 방법이 도달하려고 하는 벤치마크입니다.

**유형:** 구축
**언어:** Python
**선수 지식:** Phase 9 · 01 (MDP)
**소요 시간:** ~75분

## 문제 정의

알려진 모델(MDP)을 가지고 있습니다: 임의의 상태-행동 쌍에 대해 `P(s' | s, a)`와 `R(s, a, s')`를 조회할 수 있습니다. 재고 관리자는 수요 분포를 알고 있습니다. 보드 게임은 결정적 전이를 가집니다. 그리드월드는 파이썬 4줄로 구현됩니다. 당신은 *모델*을 가지고 있습니다.

모델 없는 강화학습(Q-러닝, PPO, REINFORCE)은 모델이 없는 경우를 위해 발명되었습니다 — 환경에서 샘플링만 가능합니다. 하지만 모델을 가지고 있을 때는 더 빠르고 더 나은 방법인 동적 프로그래밍(Dynamic Programming)이 있습니다. 벨만은 1957년에 이를 설계했습니다. 이는 여전히 정확성을 정의합니다: 사람들이 "이 MDP에 대한 최적 정책"이라고 말할 때, 이는 DP가 반환할 정책을 의미합니다.

2026년에도 이 방법이 필요한 세 가지 이유가 있습니다. 첫째, RL 연구의 모든 표 형식 환경(GridWorld, FrozenLake, CliffWalking)은 DP로 해결되어 금표준(gold-standard) 정책을 생성합니다. 둘째, 정확한 값은 샘플링 방법을 *디버깅*할 수 있게 합니다: Q-러닝의 `V*(s_0)` 추정치가 DP 결과와 30% 차이가 난다면 Q-러닝에 버그가 있는 것입니다. 셋째, 현대 오프라인 RL 및 계획 방법(MCTS, AlphaZero의 탐색, Phase 9 · 10의 모델 기반 RL)은 모두 학습된 모델 또는 주어진 모델에 대해 벨만 백업을 반복합니다.

## 개념

![정책 반복과 가치 반복, 나란히](../assets/dp.svg)

**두 알고리즘, 모두 벨만 고정점 반복입니다.**

**정책 반복.** 정책이 변하지 않을 때까지 두 단계를 번갈아 수행합니다.

1. *평가:* 주어진 정책 `π`에 대해 `V^π`를 반복적으로 계산하여 수렴할 때까지 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`를 적용합니다.
2. *개선:* 주어진 `V^π`에 대해 `V^π`에 대해 탐욕적으로 정책을 만듭니다: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

수렴이 보장되는 이유는 (a) 각 개선 단계에서 `π`를 그대로 유지하거나 일부 상태에 대해 `V^π`를 엄격하게 증가시키고, (b) 결정적 정책 공간이 유한하기 때문입니다. 일반적으로 큰 상태 공간에서도 ~5–20회의 외부 반복으로 수렴합니다.

**가치 반복.** 평가와 개선을 한 번의 스위프로 통합합니다. 벨만 *최적성* 방정식을 적용합니다:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

`max_s |V_{new}(s) - V(s)| < ε`가 될 때까지 반복합니다. 마지막에 탐욕적 행동을 선택하여 정책을 추출합니다. 반복당 계산량은 더 적지만(일반적으로 내부 평가 루프 없음), 일반적으로 수렴까지 더 많은 반복이 필요합니다.

**일반화된 정책 반복(GPI).** 통합 프레임워크입니다. 가치 함수와 정책은 상호 개선 루프에 갇혀 있으며, 둘 모두를 상호 일관성(비동적 가치 반복, 수정된 정책 반복, Q-러닝, 액터-크리틱, PPO)으로 이끄는 모든 방법은 GPI의 인스턴스입니다.

**`γ < 1`이 중요한 이유.** 벨만 연산자는 sup-노름에서 `γ`-축소 매핑입니다: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. 축소 매핑은 유일한 고정점과 기하급수적 수렴을 보장합니다. `γ < 1`을 생략하면 보장이 사라집니다 — 유한 시간 범위 또는 흡수 종료 상태가 필요합니다.

## 구축 방법

### 1단계: GridWorld MDP 모델 구축

레슨 01과 동일한 4×4 GridWorld를 사용합니다. 확률적 변형을 추가합니다: 확률 `0.1`로 에이전트가 무작위 수직 방향으로 미끄러집니다.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)`는 `(s', r, p)` 리스트를 반환합니다. 이것이 전체 모델입니다.

### 2단계: 정책 평가

정책 `π(s) = {action: prob}`이 주어졌을 때, 벨만 방정식을 반복하여 `V`가 수렴할 때까지 계산합니다:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 3단계: 정책 개선

`V`에 대해 탐욕적 정책으로 `π`를 대체합니다. `π`가 변경되지 않으면 최적 정책에 도달한 것입니다.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 4단계: 통합

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # 임의의 시작 정책
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 그리드에서 일반적인 수렴 속도: 4–6회 외부 반복. 출력값 `V*(0,0) ≈ -6`과 단계 수를 엄격히 감소시키는 정책을 반환합니다.

### 5단계: 가치 반복 (단일 루프 버전)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

동일한 고정점, 더 간결한 코드.

## 함정(Pitfalls)

- **터미널 상태(terminal state) 처리 누락.** 흡수 상태(absorbing state)에 벨만 방정식(Bellman equation)을 적용하면 아무 변화도 없는 "최적 행동(best action)"을 선택할 수 있습니다. `if s == terminal: V[s] = 0`으로 방어하세요.
- **sup-노름(sup-norm) vs L2 수렴.** 평균(average)이 아닌 `max |V_new - V|`를 사용하세요. 이론적 보장은 sup-노름에 기반합니다.
- **제자리(in-place) vs 동기식(synchronous) 업데이트.** `V[s]`를 제자리 업데이트(가우스-자이델, Gauss-Seidel)하면 별도의 `V_new` 딕셔너리(야코비, Jacobi)보다 수렴이 빠릅니다. 프로덕션 코드는 제자리 업데이트를 사용합니다.
- **정책 동률(policy ties).** 두 행동의 Q-값이 동일할 때 `argmax`가 반복마다 동률을 다르게 처리하면 "정책 안정(policy stable)" 검사가 진동할 수 있습니다. 안정적인 동률 처리(첫 번째 행동을 고정 순서로 선택)를 사용하세요.
- **상태 공간 폭발(state-space explosion).** DP는 한 스윕(sweep)당 `O(|S| · |A|)`입니다. 최대 ~10⁷ 상태까지 작동합니다. 그 이상은 함수 근사(function approximation, Phase 9 · 05 이후)가 필요합니다.

## 사용 사례

2026년에는 DP(Dynamic Programming)가 정확성 기준이자 플래너의 내부 루프로 사용됩니다:

| 사용 사례 | 방법 |
|----------|--------|
| 작은 표 형식 MDP(Markov Decision Process)를 정확히 풀기 | 가치 반복(value iteration, 더 간단) 또는 정책 반복(policy iteration, 외부 단계 수 감소) |
| Q-러닝(Q-learning) / PPO(Proximal Policy Optimization) 구현 검증 | 장난감 환경에서 DP-최적 V*(V-star)와 비교 |
| 모델 기반 RL(Phase 9 · 10) | 학습된 전이 모델에 대한 벨만 백업(Bellman backup) |
| AlphaZero / MuZero에서의 계획 수립 | 몬테카를로 트리 탐색(Monte Carlo Tree Search) = 비동기 벨만 백업 |
| 오프라인 RL(CQL, IQL) | 보수적 Q-반복(Conservative Q-iteration) — OOD(Out-Of-Distribution) 행동에 페널티 적용 DP |

누군가가 "최적의 가치 함수"라고 언급할 때마다, 이는 "DP 고정점"을 의미합니다. 논문에서 `V*` 또는 `Q*`를 보면 이 루프를 떠올리세요.

## Ship It

`outputs/skill-dp-solver.md`로 저장:

```markdown
---
name: dp-solver
description: 정책 반복(policy iteration) 또는 가치 반복(value iteration)을 통해 작은 표 형식 MDP를 정확히 해결. 수렴 동작 보고.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

알려진 모델(model)을 가진 MDP가 주어졌을 때, 다음을 출력:

1. 선택. 정책 반복 vs 가치 반복. |S|(상태 공간 크기), |A|(행동 공간 크기), γ(감가율)에 기반한 이유.
2. 초기화. V_0(초기 가치 함수), 시작 정책. 수렴 민감도.
3. 종료 조건. Sup-norm 허용 오차 ε. 예상 반복 횟수.
4. 검증. 정확히 계산된 V*(s_0)(최적 가치 함수). 탐욕 정책 추출.
5. 활용. 이 베이스라인이 샘플링 기반 방법 디버깅/평가에 어떻게 사용될지.

상태 공간이 10⁷을 초과하는 경우 DP 실행을 거부. Sup-norm 검증 없이 수렴을 주장하지 않음. 무한 시간 과제에서 γ ≥ 1인 경우 보장 위반으로 플래그 지정.
```

## 연습 문제

1. **쉬움.** `γ ∈ {0.9, 0.99}`인 4×4 GridWorld에서 가치 반복(value iteration)을 실행하세요. `max |ΔV| < 1e-6`이 될 때까지 몇 번의 스위프(sweep)가 필요한지 확인하세요. `V*`를 4×4 그리드 형태로 출력하세요.
2. **중간.** *확률적* GridWorld(미끄러짐 확률 `0.1`)에서 정책 반복(policy iteration)과 가치 반복(value iteration)을 비교하세요. 스위프 횟수, 벽시계 시간(wall-clock time), 최종 `V*(0,0)` 값을 기록하세요. 반복 횟수 기준으로 어떤 방법이 더 빠르게 수렴하나요? 벽시계 시간 기준으로는 어떤가요?
3. **어려움.** 수정된 정책 반복(modified policy iteration)을 구현하세요: 평가 단계에서 수렴할 때까지 반복하는 대신 `k`번의 스위프만 실행하세요. `k ∈ {1, 2, 5, 10, 50}`에 대해 `V*(0,0)` 오차(error)를 `k`에 따라 그래프로 그리세요. 이 곡선은 평가(evaluation)와 개선(improvement) 간의 절충(tradeoff)에 대해 어떤 통찰을 제공하나요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 정책 반복(Policy iteration) | "DP 알고리즘" | 정책 평가(`V^π`)와 개선(최적 `π` w.r.t. `V^π`)을 번갈아 수행하며 정책이 더 이상 변하지 않을 때까지 반복. |
| 가치 반복(Value iteration) | "더 빠른 DP" | 벨만 최적성 백업(Bellman optimality backup)을 한 번의 스윕(sweep)에 적용; `V*`로 기하급수적으로 수렴. |
| 벨만 연산자(Bellman operator) | "재귀식" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; sup-노름에서 `γ`-수축(contraction) 연산자. |
| 수축(Contraction) | "DP가 수렴하는 이유" | `||T x - T y|| ≤ γ ||x - y||`를 만족하는 모든 연산자 `T`는 유일한 고정점을 가짐. |
| 일반화된 정책 반복(GPI) | "모든 것은 DP" | 일반화된 정책 반복(Generalized Policy Iteration): `V`와 `π`를 상호 일관성(mutual consistency)으로 유도하는 모든 방법. |
| 동기식 업데이트(Synchronous update) | "야코비 방식" | 스윕(sweep) 동안 이전 `V`를 계속 사용; 분석은 간단하지만 느림. |
| 인플레이스 업데이트(In-place update) | "가우스-자이델 방식" | 업데이트 중인 `V`를 즉시 사용; 실제로 더 빠르게 수렴.

## 추가 학습 자료

- [Sutton & Barto (2018). Ch. 4 — 동적 프로그래밍](http://incompleteideas.net/book/RLbook2020.pdf) — 정책 반복(policy iteration)과 가치 반복(value iteration)의 표준적 설명.
- [Bertsekas (2019). 강화 학습과 최적 제어](http://www.athenasc.com/rlbook.html) — 수축 사상(contraction-mapping) 논증의 엄밀한 처리.
- [Puterman (2005). 마르코프 결정 과정](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 수정 정책 반복(modified policy iteration) 및 그 수렴 분석.
- [Howard (1960). 동적 프로그래밍과 마르코프 과정](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 원본 정책 반복 논문.
- [Bertsekas & Tsitsiklis (1996). 신경-동적 프로그래밍](http://www.athenasc.com/ndpbook.html) — DP에서 근사-DP / 심층 강화 학습(deep RL)으로의 연결고리, 이후 모든 강의에서 사용됨.