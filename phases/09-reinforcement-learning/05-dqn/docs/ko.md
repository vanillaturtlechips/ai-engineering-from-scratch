# 딥 Q-네트워크(DQN)

> 2013: Mnih는 원시 픽셀에 대해 하나의 Q-러닝 네트워크를 훈련시켜 7개의 아타리 게임에서 모든 고전적 RL 에이전트를 능가했습니다. 2015: 49개 게임으로 확장되어 Nature에 게재되었고, 딥 강화학습 시대를 촉발시켰습니다. DQN은 함수 근사를 안정화시키는 세 가지 트릭이 추가된 Q-러닝입니다.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** Phase 3 · 03 (역전파), Phase 9 · 04 (Q-러닝, SARSA)
**소요 시간:** ~75분

## 문제

테이블 기반 Q-러닝은 모든 (상태, 행동) 쌍에 대해 별도의 Q-값이 필요합니다. 체스 보드는 약 10⁴³개의 상태를 가집니다. 아타리 프레임은 210×160×3 = 100,800개의 특징(feature)을 가집니다. 테이블 기반 강화학습은 수천 개의 상태에서 이미 실패하는데, 수십억 개의 상태에서는 더욱 불가능합니다.

해결책은 사후적으로 명확합니다: Q-테이블을 신경망 `Q(s, a; θ)`로 대체하는 것입니다. 하지만 이 명확한 해결책은 수십 년이 걸렸습니다. Q-러닝에 순진한 함수 근사를 적용하면 "치명적 삼중주"(function approximation + 부트스트래핑 + 오프-정책 학습) 하에서 발산합니다. Mnih et al. (2013, 2015)는 학습을 안정화하는 세 가지 엔지니어링 기법을 확인했습니다:

1. **경험 리플레이(Experience replay)**는 전이(transition) 간의 상관관계를 제거합니다.
2. **타겟 네트워크(Target network)**는 부트스트래핑 목표를 고정합니다.
3. **보상 클리핑(Reward clipping)**은 그래디언트 크기를 정규화합니다.

아타리에서의 DQN은 단일 하이퍼파라미터 세트로 원시 픽셀(raw pixels)에서 수십 개의 제어 문제를 해결한 최초의 단일 아키텍처였습니다. 이후 개발된 모든 "딥 강화학습"(DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 등)은 이 세 가지 기법 위에 구축되었습니다.

## 개념

![DQN 훈련 루프: 환경, 리플레이 버퍼, 온라인 네트워크, 타겟 네트워크, 벨만 TD 손실](../assets/dqn.svg)

**목표.** DQN은 신경망 Q-함수에 대한 1단계 TD 손실을 최소화합니다:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = 온라인 네트워크, 경사 하강법으로 매 단계 업데이트. `θ^-` = 타겟 네트워크, 주기적으로 `θ`에서 복사됨 (~10,000단계마다). `D` = 과거 전이들의 리플레이 버퍼.

**중요도 순서대로 세 가지 트릭:**

**경험 리플레이.** 약 `~10⁶`개의 전이를 저장하는 링 버퍼. 각 훈련 단계에서 미니배치를 균일하게 무작위 샘플링합니다. 이는 시간적 상관관계(연속 프레임이 거의 동일함)를 제거하고, 네트워크가 드문 보상 전이로부터 여러 번 학습할 수 있게 하며, 연속적인 경사 업데이트를 비연관화합니다. 이 트릭이 없으면 신경망 기반 온-폴리시 TD는 Atari에서 발산합니다.

**타겟 네트워크.** 벨만 방정식의 양쪽에 동일한 네트워크 `Q(·; θ)`를 사용하면 매 업데이트마다 목표가 이동 — "자신의 꼬리를 쫓는" 현상이 발생합니다. 해결책: 가중치가 고정된 두 번째 네트워크 `Q(·; θ^-)`를 유지합니다. `C`단계마다 `θ → θ^-`로 복사합니다. 이는 수천 번의 경사 업데이트 동안 회귀 목표를 안정화합니다. 소프트 업데이트 `θ^- ← τ θ + (1-τ) θ^-`(DDPG, SAC에서 사용)는 더 부드러운 변형입니다.

**보상 클리핑.** Atari 보상 크기는 1에서 1000+까지 다양합니다. `{-1, 0, +1}`로 클리핑하면 단일 게임이 경사를 지배하는 것을 방지합니다. 보상 크기가 중요한 경우에는 부적절하지만, 부호만 중요한 Atari에는 적합합니다.

**더블 DQN.** Hasselt (2016)은 최대화 편향을 수정합니다: 온라인 네트워크로 액션을 *선택*하고, 타겟 네트워크로 *평가*합니다.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

즉시 적용 가능한 개선이며, 일관되게 더 나은 성능을 보입니다. 기본적으로 사용하세요.

**기타 개선 사항 (Rainbow, 2017):** 우선순위 리플레이(높은 TD 오차 전이 더 많이 샘플링), 듀얼링 아키텍처(`V(s)`와 어드밴티지 헤드 분리), 노이즈 네트워크(학습된 탐색), n-단계 반환, 분포형 Q(C51/QR-DQN), 다중 단계 부트스트래핑. 각각은 몇 %의 성능 향상을 제공하며, 이득은 대략 가산적입니다.

## 구축 방법

여기의 코드는 표준 라이브러리만 사용하며 NumPy가 없습니다 — 작은 연속 GridWorld에서 단일 은닉층을 가진 MLP를 직접 구현했으며, 모든 훈련 단계는 마이크로초 단위로 실행됩니다. 이 알고리즘은 확장 시 Atari DQN과 동일합니다.

### 1단계: 리플레이 버퍼

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari의 경우 ~50,000 용량, 우리의 장난감 환경에서는 5,000으로 충분합니다.

### 2단계: 작은 Q-네트워크(수동 MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

순전파: 선형 → ReLU → 선형. 이것이 전체 네트워크입니다.

### 3단계: DQN 업데이트

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

이 구조는 레슨 04의 Q-러닝과 두 가지 차이점이 있습니다: (a) 테이블 인덱싱 대신 미분 가능한 `Q(·; θ)`를 역전파하고, (b) 타겟은 `Q(·; θ^-)`를 사용합니다.

### 4단계: 외부 루프

각 에피소드에서 `Q(·; θ)`에 대해 ε-탐욕적 행동을 하고, 전이(transition)를 버퍼에 푸시하고, 미니배치를 샘플링하고, 경사 하강 단계를 수행하고, 주기적으로 `θ^- ← θ`를 동기화합니다. 패턴은 다음과 같습니다:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

16차원 원-핫 상태를 가진 작은 GridWorld에서 에이전트는 약 500 에피소드 만에 거의 최적의 정책을 학습합니다. Atari에서는 이 구조를 2억 프레임으로 확장하고 CNN 특징 추출기를 추가합니다.

## 함정(Pitfalls)

- **치명적 삼중주(Deadly triad).** 함수 근사(function approximation) + 오프-정책(off-policy) + 부트스트래핑(bootstrapping) 조합은 발산할 수 있음. DQN은 타겟 네트워크(target net) + 리플레이 버퍼(replay buffer)로 완화함; 둘 중 하나라도 제거하지 말 것.
- **탐색(Exploration).** ε은 감소해야 하며, 일반적으로 전체 학습의 처음 ~10% 동안 1.0에서 0.01로 감소. 초기 탐색이 충분하지 않으면 Q-네트워크가 지역 최적점(local basin)에 수렴함.
- **과대평가(Overestimation).** 노이즈가 있는 Q 값에 대한 `max` 연산은 상향 편향됨. 프로덕션 환경에서는 항상 더블 DQN(Double DQN)을 사용할 것.
- **보상 스케일(Reward scale).** 보상을 클리핑(clip)하거나 정규화(normalize)할 것; 그래디언트 크기는 보상 크기에 비례함.
- **리플레이 버퍼 콜드 스타트(Replay buffer coldstart).** 버퍼에 수천 개의 전이(transition)가 쌓이기 전까지는 학습하지 말 것. ~20개 샘플에 대한 초기 그래디언트는 과적합(overfitting) 발생.
- **타겟 동기화 빈도(Target sync frequency).** 너무 빈번하면 ≈ 타겟 네트워크 없음; 너무 드물면 ≈ 오래된 타겟. Atari DQN은 10,000 환경 단계(env steps) 사용. 경험법: 학습 지평선(training horizon)의 ~1/100마다 동기화.
- **관측 전처리(Observation preprocessing).** Atari DQN은 상태(state)를 마르코프(Markov)하게 만들기 위해 4프레임을 스택(stack)함. 속도 정보(velocity info)가 있는 환경은 프레임 스택(frame-stacking) 또는 순환 상태(recurrent state)가 필요함.

## 사용 사례

2026년 현재 DQN은 최첨단 기술은 아니지만 여전히 참조되는 오프-정책 알고리즘으로 남아 있습니다:

| 작업 | 선호 방법 | DQN을 사용하지 않는 이유 |
|------|------------------|--------------|
| 이산 행동 Atari 유사 | Rainbow DQN 또는 Muesli | 동일한 프레임워크, 더 많은 트릭 적용. |
| 연속 제어 | SAC / TD3 (9단계 · 07) | DQN에는 정책 네트워크가 없음. |
| 온-정책 / 고처리량 | PPO (9단계 · 08) | 재생 버퍼 없음; 확장성이 더 쉬움. |
| 오프라인 RL | CQL / IQL / Decision Transformer | 보수적인 Q-타겟, 부트스트래핑 폭주 방지. |
| 대규모 이산 행동 공간 (추천 시스템) | 행동 임베딩이 있는 DQN, 또는 IMPALA | 적절함; 장식 요소가 중요. |
| LLM 강화학습 | PPO / GRPO | 단계 수준이 아닌 시퀀스 수준; 다른 손실 함수. |

이 교훈들은 여전히 유효합니다. 재생 버퍼와 타겟 네트워크는 SAC, TD3, DDPG, SAC-X, AlphaZero의 자기 대국 버퍼, 그리고 모든 오프라인 RL 방법에 등장합니다. 보상 클리핑은 PPO의 어드밴티지 정규화로 계승됩니다. 이 아키텍처는 청사진 역할을 합니다.

## Ship It

저장 위치: `outputs/skill-dqn-trainer.md`

```markdown
---
name: dqn-trainer
description: 이산 행동 공간(discrete-action) 강화 학습(RL) 작업을 위한 DQN 학습 구성(버퍼, 타겟 동기화, ε 스케줄, 보상 클리핑)을 생성합니다.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

이산 행동 환경(관측 형태, 행동 수, 시간 범위, 보상 스케일)이 주어졌을 때 다음 내용을 출력합니다:

1. 네트워크. 아키텍처(MLP / CNN / Transformer), 특징 차원(feature dim), 깊이(depth).
2. 리플레이 버퍼. 용량(capacity), 미니배치 크기(minibatch size), 워밍업 크기(warmup size).
3. 타겟 네트워크. 동기화 전략(C 단계마다 하드 동기화 또는 소프트 τ).
4. 탐색. ε 시작값/종료값/스케줄 길이.
5. 손실. Huber vs MSE, 그래디언트 클리핑 값, 보상 클리핑 규칙.
6. 더블 DQN. 명시적 비활성화 사유가 없는 한 기본 활성화.

타겟 네트워크 없음, 리플레이 버퍼 없음, ε이 1로 고정된 DQN은 거부합니다. 연속 행동 공간 작업은 SAC / TD3로 라우팅합니다. 단계별 평균 대비 보상 범위가 10배 초과하는 경우 클리핑 또는 스케일 정규화 필요성을 표시합니다.
```

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 에피소드별 수익 곡선을 플롯하세요. 이동 평균이 -10을 초과하기까지 몇 개의 에피소드가 필요한가요?
2. **중간.** 타겟 네트워크를 비활성화하세요 (벨만 타겟의 양쪽에 온라인 네트워크를 사용). 훈련 불안정성을 측정하세요 — 수익이 진동하거나 발산하나요?
3. **어려움.** Double DQN을 추가하세요: 온라인 네트워크로 `argmax a'`를 선택하고, 타겟 네트워크로 평가하세요. 노이즈가 있는 보상 GridWorld에서 1,000 에피소드 후 Double DQN 사용 여부에 따른 `Q(s_0, best_a)`와 실제 `V*(s_0)`의 편향을 비교하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| DQN | "Deep Q-learning" | 신경망 Q-함수, 리플레이 버퍼, 타겟 네트워크를 사용한 Q-러닝. |
| 경험 리플레이(Experience replay) | "Shuffled transitions" | 매 경사 하강 단계마다 균일하게 샘플링되는 링 버퍼; 데이터 간 상관관계를 감소시킵니다. |
| 타겟 네트워크(Target network) | "Frozen bootstrap" | 벨만 타겟에 사용되는 Q의 주기적 복사본; 학습을 안정화합니다. |
| 치명적 삼중주(Deadly triad) | "Why RL diverges" | 함수 근사 + 부트스트래핑 + 오프-정책 = 수렴 보장 없음. |
| 더블 DQN(Double DQN) | "Fix for maximization bias" | 온라인 네트워크가 액션을 선택하고, 타겟 네트워크가 이를 평가합니다. |
| 듀얼링 DQN(Dueling DQN) | "V and A heads" | Q = V + A - mean(A)로 분해; 동일한 출력, 더 나은 그래디언트 흐름. |
| 레인보우(Rainbow) | "All the tricks" | DDQN + PER + 듀얼링 + n-스텝 + 노이즈 + 분포적 RL을 하나로 통합. |
| PER | "Prioritized Replay" | TD-오차 크기에 비례하여 전이(transition)를 샘플링합니다.

## 추가 자료

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 딥 강화 학습(Deep RL)을 시작한 2013년 NeurIPS 워크숍 논문.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 논문, 49게임 DQN.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — 듀얼링 DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 개선 기법들을 결합한 논문.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 명확한 현대식 설명.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — DQN의 타깃 네트워크와 리플레이 버퍼가 해결하려는 "치명적 삼중주"(함수 근사 + 부트스트래핑 + 오프-정책)에 대한 교과서적 설명.
- [CleanRL DQN 구현](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 제거 실험(ablation studies)에 사용된 단일 파일 DQN 참조 구현; 이 강의의 처음부터 구현한 버전과 함께 읽기 좋음.