# 다중 에이전트 강화 학습(Multi-Agent RL)

> 단일 에이전트 강화 학습(Single-agent RL)은 환경이 정상성(stationary)을 가진다고 가정합니다. 같은 세계에 두 개의 학습 에이전트를 배치하면 이 가정이 깨집니다: 각 에이전트는 다른 에이전트의 환경의 일부이며, 둘 다 변화합니다. 다중 에이전트 강화 학습은 마르코프 가정이 더 이상 성립하지 않을 때 학습을 수렴시키기 위한 기법들의 집합입니다.

**유형:** 구현(Build)  
**언어:** Python  
**선수 지식:** 9단계 · 04 (Q-러닝), 9단계 · 06 (REINFORCE), 9단계 · 07 (Actor-Critic)  
**소요 시간:** ~45분

## 문제 정의

방 안에서 길을 찾는 로봇은 단일 에이전트 강화학습(RL) 문제입니다. 그러나 축구 팀은 그렇지 않습니다. AlphaStar 대 스타크래프트 상대도, 입찰 에이전트 시장도, 4-way 정차 구간에서 협상하는 두 대의 자동차도 그렇지 않습니다. 많은 실제 다대다 문제는 단일 에이전트 문제가 아닙니다.

모든 다중 에이전트 환경에서, 어떤 한 에이전트의 관점에서 다른 에이전트들은 환경의 일부입니다. 다른 에이전트들이 학습하고 행동을 변경함에 따라 환경은 비정상(non-stationary) 상태가 됩니다. "다음 상태는 현재 상태와 내 행동에만 의존한다"는 마르코프 성질(Markov property)은 위반됩니다. 왜냐하면 다음 상태는 다른 에이전트들이 선택한 행동과 그들의 정책(policy)에 따라 달라지기 때문입니다. 그리고 이 정책들은 움직이는 표적입니다.

이는 테이블 기반 수렴 증명(Q-러닝의 보장은 정적 환경을 가정)을 무효화합니다. 또한 순진한 딥러닝 기반 RL도 실패합니다: 에이전트들이 서로를 쫓는 루프에 빠져 안정적인 정책으로 수렴하지 못합니다. 다중 에이전트 전용 기법이 필요합니다: 중앙집중식 훈련/분산 실행(centralized training / decentralized execution), 반사실적 기준(counterfactual baselines), 리그 플레이(league play), 자기 대국(self-play) 등.

2026년 적용 분야: 로봇 군집(robot swarms), 교통 라우팅(traffic routing), 자율 주행 차량 군집(autonomous vehicle fleets), 시장 시뮬레이터(market simulators), 다중 에이전트 LLM 시스템(Phase 16), 그리고 하나 이상의 지능형 플레이어가 있는 모든 게임.

## 개념

![네 가지 MARL 체제: 독립형, 중앙 집중형 비평가, 자기 대국, 리그](../assets/marl.svg)

**형식화: 마르코프 게임.** MDP의 일반화: 상태 `S`, 공동 행동 `a = (a_1, …, a_n)`, 전이 `P(s' | s, a)`, 그리고 에이전트별 보상 `R_i(s, a, s')`. 각 에이전트 `i`는 자신의 정책 `π_i` 하에서 자신의 수익을 최대화합니다. 보상이 동일하면 **완전 협력**입니다. 제로섬이면 **적대적**입니다. 혼합이면 **일반합**입니다.

**핵심 과제:**

- **비정상성(Non-stationarity).** 에이전트 `i`의 관점에서 `P(s' | s, a_i)`는 변화하는 `π_{-i}`에 의존합니다.
- **신용 할당(Credit assignment).** 공유 보상이 있을 때, 어떤 에이전트가 이를 발생시켰는가?
- **탐색 조정(Exploration coordination).** 에이전트들은 동일한 상태를 중복 탐색하지 않고 상호 보완적인 전략을 탐색해야 합니다.
- **확장성(Scalability).** 공동 행동 공간은 `n`에 따라 기하급수적으로 증가합니다.
- **부분 관측성(Partial observability).** 각 에이전트는 자신의 관측만 볼 수 있으며, 전역 상태는 숨겨져 있습니다.

**네 가지 주요 체제:**

**1. 독립 Q-러닝 / 독립 PPO (IQL, IPPO).** 각 에이전트는 다른 에이전트들을 환경의 일부로 취급하며 자신의 Q-함수 또는 정책을 학습합니다. 단순하며, 때로는 작동합니다(특히 경험 재생이 에이전트 모델링 트릭으로 작용할 때). 이론적 수렴: 없음. 실제 적용: 느슨하게 결합된 작업에는 적합하지만, 강하게 결합된 작업에는 부적합합니다.

**2. 중앙 집중형 훈련, 분산 실행 (CTDE).** 가장 일반적인 현대 패러다임입니다. 각 에이전트는 배포 시 로컬 관측 `o_i`에 조건부인 자체 정책 `π_i`를 가집니다. *훈련* 중에는 중앙 집중형 비평가 `Q(s, a_1, …, a_n)`이 전체 전역 상태와 공동 행동에 조건부가 됩니다. 예시:
- **MADDPG** (Lowe et al. 2017): 중앙 집중형 비평을 가진 DDPG.
- **COMA** (Foerster et al. 2017): 반사실적 기준선 — "내가 행동 `a'`를 취했다면 보상은 어땠을까?"라고 질문하여 나의 기여도를 분리합니다.
- **MAPPO** / **IPPO** with 공유 비평가 (Yu et al. 2022): 중앙 집중형 가치 함수를 가진 PPO. 2026년 협력형 MARL의 주류.
- **QMIX** (Rashid et al. 2018): 가치 분해 — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`에 단조 혼합을 적용합니다.

**3. 자기 대국(Self-play).** 동일한 에이전트의 두 복사본이 서로 대국합니다. 상대방의 정책은 과거 스냅샷의 내 정책입니다. 알파고 / 알파제로 / 뮤제로. 오픈AI 파이브. 제로섬 게임에서 가장 잘 작동하며, 훈련 신호는 대칭적입니다.

**4. 리그 플레이(League play).** 일반합 / 적대적 환경으로 자기 대국을 확장한 것: 과거 및 현재 정책 집단을 유지하고, 리그에서 상대방을 샘플링하여 훈련합니다. 착취자(현재 최고를 이기도록 특화)와 주요 착취자(착취자를 이기도록 특화)를 추가합니다. 알파스타(스타크래프트 II). 게임에 "가위-바위-보" 전략 사이클이 있을 때 필요합니다.

**통신(Communication).** 에이전트들이 서로에게 학습된 메시지 `m_i`를 보낼 수 있도록 허용합니다. 협력 환경에서 작동합니다. Foerster et al. (2016)은 미분 가능한 에이전트 간 통신이 종단 간 훈련될 수 있음을 보였습니다. 오늘날의 LLM 기반 다중 에이전트 시스템(16단계)은 본질적으로 자연어로 통신합니다.

## 빌드하기

이 레슨은 두 협력 에이전트가 있는 6×6 GridWorld를 사용합니다. 에이전트는 반대쪽 모서리에서 시작하며 공유된 목표에 도달해야 합니다. 공유 보상: 에이전트가 움직이는 동안 매 스텝마다 `-1`, 둘 다 도착하면 `+10`. `code/main.py` 참조.

### 단계 1: 다중 에이전트 환경

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # 두 에이전트

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*공동* 행동 공간은 `|A|² = 16`입니다. 글로벌 상태는 두 위치입니다.

### 단계 2: 독립 Q-러닝

각 에이전트는 공동 상태를 키로 하는 자체 Q-테이블을 실행합니다. 각 스텝에서: 둘 다 ε-탐욕적 행동을 선택하고, 공동 전이를 수집하며, 공유 보상으로 각자의 Q를 업데이트합니다.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

보상이 밀집되어 있고 정렬되어 있기 때문에 이 작업에서 작동합니다. 긴밀하게 결합된 작업(예: 한 에이전트가 다른 에이전트를 *기다려야* 하는 경우)에서는 실패합니다.

### 단계 3: 분해된 가치 업데이트를 사용한 중앙 집중식 Q

공동 행동 `Q(s, a_1, a_2)`에 대한 하나의 Q를 사용합니다. 공유 보상으로 업데이트합니다. 실행 시 분산화를 위해 주변화: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. 지수적 공동 행동 공간을 *정확한* 글로벌 뷰와 교환합니다.

### 단계 4: 간단한 자기 대전 (적대적 2-에이전트)

동일한 에이전트, 두 역할. 에이전트 A를 에이전트 B와 대전시켜 훈련; `K` 에피소드 후 A의 가중치를 B에 복사합니다. 대칭 훈련, 일관된 진행. AlphaZero 레시피의 소형 버전입니다.

## 함정(Pitfalls)

- **비정상성 리플레이(Non-stationary replay).** 독립 에이전트를 사용한 경험 리플레이는 단일 에이전트보다 성능이 낮음. 오래된 전이(transition)는 현재 사용되지 않는 상대에 의해 생성되었기 때문. 해결: 최신성(recency)에 따라 레이블 재지정 또는 가중치 부여.
- **신용 할당 모호성(Credit assignment ambiguity).** 긴 에피소드 후 공유 보상 발생; 어떤 에이전트가 기여했는지 명확히 알 수 없음. 해결: 반사실적 기준선(counterfactual baselines, COMA) 또는 에이전트별 보상 형성(reward shaping).
- **정책 표류/추적(Policy drift / chasing).** 각 에이전트의 최적 응답은 다른 에이전트의 업데이트에 따라 변함. 해결: 중앙 집중형 비평가(centralized critic), 느린 학습률(learning rate), 또는 한 번에 하나씩 동결(freeze-one-at-a-time).
- **협조를 통한 보상 조작(Reward hacking via coordination).** 에이전트가 설계자가 예상하지 못한 협조적 악용 방법을 찾음. 경매 에이전트가 0으로 입찰하는 것으로 수렴. 해결: 신중한 보상 설계, 행동 제약 조건.
- **탐색 중복(Exploration redundancy).** 두 에이전트가 동일한 상태-행동 쌍을 탐색. 해결: 에이전트별 엔트로피 보너스(entropy bonuses) 또는 역할 조건부(role-conditioning).
- **리그 사이클(League cycles).** 순수 자기 플레이(self-play)는 지배 사이클에 갇힐 수 있음. 해결: 다양한 상대와 리그 플레이(league play).
- **샘플 폭발(Sample explosion).** `n` 에이전트 × 상태 공간 × 공동 행동. 해결: 함수 근사(function approximation)로 근사화; 요소별 행동 공간(factored action spaces, 에이전트당 하나의 정책 출력 헤드).

## 사용 방법

2026년 MARL(다중 에이전트 강화 학습) 응용 분야 지도:

| 도메인 | 방법 | 참고 사항 |
|--------|--------|-------|
| 협력 탐색 / 조작 | MAPPO / QMIX | CTDE(중앙화 훈련-분산 실행); 공유 크리틱 + 분산 액터. |
| 2인 게임(체스, 바둑, 포커) | MCTS(몬테카를로 트리 탐색) 기반 자기 대국(AlphaZero) | 제로섬; 대칭적 훈련. |
| 복잡한 다중 플레이어(도타, 스타크래프트) | 리그 플레이 + 모방 사전 훈련 | OpenAI Five, AlphaStar. |
| 자율 주행 차량 군집 | 어텐션 기반 CTDE MAPPO / PPO | 부분 관측; 가변 팀 규모. |
| 경매 시장 | 게임 이론적 균형 + RL | `n` → ∞일 때 평균장 RL. |
| LLM 다중 에이전트 시스템(16단계) | 자연어 통신 + 역할 조건화 | 에이전트 계획 계층에서 RL 루프. |

2026년 MARL에서 가장 빠르게 성장하는 분야는 LLM 기반입니다: 언어 모델 에이전트 군집이 협상, 토론, 소프트웨어 구축을 수행합니다. RL은 토큰 수준이 아닌 *트래젝터리 수준* 출력에 대한 선호 최적화로 나타납니다(16단계 · 03).

## Ship It

`outputs/skill-marl-architect.md`로 저장:

```markdown
---
name: marl-architect
description: 주어진 태스크에 적합한 다중 에이전트 강화학습(MARL) 방식(IPPO, CTDE, 셀프플레이, 리그)을 선택합니다.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

에이전트 수 `n`이 주어진 태스크에 대해 다음을 출력합니다:

1. **레짐 분류**. 협동(cooperative) / 적대적(adversarial) / 일반합(general-sum). 근거 제시.
2. **알고리즘**. IPPO / MAPPO / QMIX / 셀프플레이 / 리그. 결합 강도(coupling tightness) 및 보상 구조와 연계한 이유.
3. **정보 접근**. 중앙집중식 훈련(크리틱에 제공되는 글로벌 정보는 무엇인가)? 분산 실행?
4. **신용 할당**. 반사실적 기준(counterfactual baseline), 가치 분해(value decomposition), 또는 보상 형성(reward shaping).
5. **탐색 계획**. 에이전트별 엔트로피(entropy), 인구 기반 훈련(population-based training), 또는 리그.

**강결합 협동 태스크에 독립 Q-러닝을 거부합니다**. **순환 위험(cycle risks)이 있는 일반합 게임에 셀프플레이를 권장하지 않습니다**. **고정 상대 평가(fixed-opponent eval)가 없는 MARL 파이프라인을 플래그 처리합니다**(체리피킹된 셀프플레이 수치가 흔함).
```

## 연습 문제

1. **쉬움.** 2-에이전트 협동 GridWorld에서 독립적인 Q-러닝(Q-learning)을 학습시켜 보세요. 평균 보상(mean return)이 0보다 커질 때까지 몇 개의 에피소드가 필요한가요? 공동 학습 곡선(joint learning curve)을 그려보세요.
2. **중간.** "협력(coordination)" 작업을 추가하세요: 목표는 두 에이전트가 동시에 같은 턴에 목표 지점에 올라설 때만 달성됩니다. 독립적인 Q-러닝이 여전히 수렴하나요? 무엇이 문제인가요?
3. **어려움.** MAPPO 스타일 학습을 위한 중앙 집중형 비평가(centralized critic)를 구현하고, 협력 작업에서 독립적인 PPO와 수렴 속도를 비교해 보세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 마르코프 게임(Markov game) | "다중 에이전트 MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; 각 에이전트는 고유한 보상 함수를 가짐. |
| CTDE | "중앙 집중식 훈련, 분산 실행" | 훈련 시 공동 비평가(joint critic) 사용; 각 에이전트의 정책은 로컬 관측값만 활용. |
| IPPO | "독립 PPO" | 각 에이전트가 별도로 PPO 실행. 간단한 베이스라인; 종종 과소평가됨. |
| MAPPO | "다중 에이전트 PPO" | 전역 상태(global state)에 조건화된 중앙 집중식 가치 함수를 사용하는 PPO. |
| QMIX | "단조 가치 분해" | `Q_tot = f_monotone(Q_1, …, Q_n)`을 통해 분산 실행 시 argmax 가능. |
| COMA | "반사실적 다중 에이전트" | 어드밴티지(advantage) = 내 Q값 - 내 행동을 주변화(marginalizing)한 기대 Q값. |
| 셀프 플레이(Self-play) | "에이전트 vs 과거 자신" | 단일 에이전트, 두 역할; 제로섬 게임의 표준 접근법. |
| 리그 플레이(League play) | "인구 기반 훈련" | 과거 정책 캐싱, 풀에서 상대 샘플링; 전략 순환 문제 처리. |

## 추가 자료

- [Lowe et al. (2017). 혼합 협동-경쟁 환경을 위한 다중 에이전트 액터-크리틱 (MADDPG)](https://arxiv.org/abs/1706.02275) — 중앙 집중식 크리틱을 활용한 CTDE.
- [Foerster et al. (2017). 반사실적 다중 에이전트 정책 경사 (COMA)](https://arxiv.org/abs/1705.08926) — 신용 할당을 위한 반사실적 기준선.
- [Rashid et al. (2018). QMIX: 단조성 가치 함수 분해](https://arxiv.org/abs/1803.11485) — 단조성을 활용한 가치 분해.
- [Yu et al. (2022). 협동 다중 에이전트 게임에서 PPO의 놀라운 효과성 (MAPPO)](https://arxiv.org/abs/2103.01955) — MARL에서 PPO의 강력한 성능.
- [Vinyals et al. (2019). 다중 에이전트 강화 학습을 이용한 스타크래프트 II 그랜드마스터 수준 (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 대규모 리그 플레이.
- [Silver et al. (2017). 인간 지식 없이 바둑 게임 정복 (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 제로섬 게임에서의 순수 자기 대국.
- [Sutton & Barto (2018). Ch. 15 — 신경과학 & Ch. 17 — 개척 분야](http://incompleteideas.net/book/RLbook2020.pdf) — 다중 에이전트 설정과 CTDE가 해결하려는 비정상성 문제에 대한 교과서적 접근 포함.
- [Zhang, Yang & Başar (2021). 다중 에이전트 강화 학습: 선택적 개요](https://arxiv.org/abs/1911.10635) — 수렴 결과를 포함한 협동, 경쟁, 혼합 MARL을 다루는 조사 논문.