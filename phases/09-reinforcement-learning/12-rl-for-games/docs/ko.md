# RL for Games — AlphaZero, MuZero, and the LLM-Reasoning Era

> 1992: TD-Gammon이 순수 TD로 백개먼에서 인간 챔피언을 이겼다. 2016: AlphaGo가 이세돌 9단을 이겼다. 2017: AlphaZero가 체스, 쇼기, 바둑에서 처음부터 압도했다. 2024: DeepSeek-R1이 GRPO가 PPO를 대체한 동일한 레시피가 추론에도 작동함을 입증했다. 게임은 이 단계의 모든 돌파구를 이끄는 벤치마크다.

**Type:** 구축(Build)
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120분

## 문제 정의

게임은 강화 학습(RL)이 원하는 모든 요소를 갖추고 있습니다. 명확한 보상(승/패). 무한한 에피소드(자기 대결로 리셋 가능). 완벽한 시뮬레이션(게임 자체가 시뮬레이터). 이산적 또는 소규모 연속 행동 공간. 적대적 강건성을 강제하는 다중 에이전트 구조.

그리고 게임은 모든 주요 강화 학습 돌파구가 검증된 분야입니다. TD-Gammon(백개먼, 1992). Atari-DQN(2013). AlphaGo(2016). AlphaZero(2017). OpenAI Five(도타 2, 2019). AlphaStar(스타크래프트 II, 2019). MuZero(학습된 모델, 2019). AlphaTensor(행렬 곱셈, 2022). AlphaDev(정렬 알고리즘, 2023). DeepSeek-R1(수학 추론, 2025) — 게임-RL 기법이 텍스트 분야에서도 작동함을 보여주는 최신 사례입니다.

이 캡스톤 프로젝트는 **자기 대결 + 탐색 + 정책 개선**이라는 단일 통합 렌즈를 통해 AlphaZero, MuZero, GRPO라는 세 가지 랜드마크 아키텍처를 조사합니다. 각각은 이전 모델을 일반화하며, 특히 GRPO는 LLM 추론에 AlphaZero의 레시피를 적용한 사례로, 토큰을 행동으로, 수학적 검증을 승신호로 활용합니다.

## 개념

![AlphaZero ↔ MuZero ↔ GRPO: 동일한 루프, 다른 환경](../assets/rl-games.svg)

**통합 루프.**

```
while True:
    trajectory = self_play(current_policy, search)     # 자기 자신과의 게임 플레이
    policy_target = search.improved_policy(trajectory) # 검색을 통해 원시 정책 개선
    policy_net.update(policy_target, value_target)     # 검색 출력에 대한 지도 학습
```

**AlphaZero (2017).** Silver et al. 규칙이 알려진 게임(체스, 쇼기, 바둑)이 주어졌을 때:

- 정책-가치 네트워크: 하나의 타워 `f_θ(s) → (p, v)`. `p`는 합법적 이동에 대한 사전 확률. `v`는 예상 게임 결과.
- 몬테카를로 트리 탐색(MCTS): 각 이동 시 가능한 연속성의 트리를 확장. `(p, v)`를 사전 + 부트스트랩으로 사용. UCB(PUCT)로 노드 선택: `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- 자기 자신과의 게임: 에이전트 대 에이전트로 게임 플레이. 이동 `t`에서 MCTS 방문 분포 `π_t`가 정책 훈련 타겟이 됨.
- 손실: `L = (v - z)² - π · log p + c · ||θ||²`. `z`는 게임 결과(+1 / 0 / -1).

인간의 지식 제로. 수작업 휴리스틱 제로. 각각 수천만 번의 자기 자신과의 게임 후 체스, 쇼기, 바둑을 마스터한 단일 레시피.

**MuZero (2019).** Schrittwieser et al. 규칙이 알려져야 한다는 요구사항 제거.

- 고정된 환경 대신 *잠재 동역학 모델* `(h, g, f)` 학습:
  - `h(s)`: 관측을 잠재 상태로 인코딩.
  - `g(s_latent, a)`: 다음 잠재 상태 + 보상 예측.
  - `f(s_latent)`: 정책 사전 + 가치 예측.
- MCTS는 *학습된 잠재 공간*에서 실행. 동일한 검색, 동일한 훈련 루프.
- 바둑, 체스, 쇼기 *및* 아타리에서도 작동 — 하나의 알고리즘, 규칙 지식 불필요.

**확률적 MuZero (2022).** 확률적 동역학 및 기회 노드 추가; 백게임몬 클래스 게임으로 확장.

**Muesli, Gumbel MuZero (2022-2024).** 샘플 효율성과 결정적 검색 개선.

**GRPO (2024-2025).** DeepSeek-R1 레시피. 동일한 AlphaZero 형태의 루프를 언어 모델 추론에 적용:

- "게임": 수학/코딩/추론 문제 해결. "승리" = 검증기(테스트 케이스 통과, 수치적 답변 일치)가 1 반환.
- 정책: LLM. 행동: 토큰. 상태: 프롬프트 + 응답 진행 상황.
- 비평가 없음(PPO 스타일 V_φ). 대신 각 프롬프트에 대해 정책에서 `G`개의 완성본을 샘플링. 각각에 대한 보상 계산. **그룹 상대 이점** `A_i = (r_i - mean_r) / std_r`를 REINFORCE 스타일 업데이트 신호로 사용.
- 참조 정책에 대한 KL 페널티로 드리프트 방지(RLHF와 유사).
- 전체 손실:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

보상 모델, 비평가, MCTS 없음. 그룹 상대 기준선이 이 세 가지를 대체. 추론 벤치마크에서 PPO-RLHF 품질을 계산 리소스의 일부로 일치 또는 초과.

**R1 레시피 전체.** DeepSeek-R1(DeepSeek 2025)은 한 논문에 두 모델:

- **R1-Zero.** DeepSeek-V3 기본 모델에서 시작. SFT 없음. 두 가지 보상 구성 요소로 GRPO 직접 적용: *정확도 보상*(규칙 기반 — 최종 답변이 올바른 숫자로 파싱되었는지 / 코드가 단위 테스트 통과 여부) 및 *포맷 보상*(완성본이 `…<>` 태그로 사고 체인을 감싸는지). 수천 단계 동안 평균 응답 길이가 ~100에서 ~10,000 토큰으로 증가하고 수학 벤치마크 점수가 o1-preview 수준에 근접. 모델이 처음부터 추론을 학습. 단점: 사고 체인이 종종 읽을 수 없고, 언어를 혼합하며, 스타일적 세련미가 부족.
- **R1.** R1-Zero의 가독성 문제 해결을 위한 4단계 파이프라인:
  1. **콜드 스타트 SFT.** 깨끗한 포맷의 수천 개의 긴 CoT 데모를 수집. 기본 모델을 이에 대해 지도 미세 조정. 가독성 있는 시작점 제공.
  2. **추론 지향 GRPO.** 정확도+포맷 보상 및 *언어 일관성* 보상을 추가하여 GRPO 적용(코드 스위칭 방지).
  3. **거부 샘플링 + SFT 2라운드.** RL 체크포인트에서 ~600K 추론 궤적 샘플링, 올바른 최종 답변과 가독성 있는 CoT만 유지, ~200K 비추론 SFT 예제(작성, QA, 자기 인지)와 결합. 기본 모델 재미세 조정.
  4. **전체 스펙트럼 GRPO.** 추론(규칙 기반 보상)과 일반 정렬(도움/무해 선호도 기반 보상)을 모두 포함하는 추가 RL 라운드.

결과: AIME 및 MATH-500에서 o1과 일치하는 오픈 웨이트, 충분히 작아 증류 가능. 동일한 논문은 R1의 추론 트레이스에 대해 SFT를 수행하여 6개의 증류된 밀집 모델(Qwen-1.5B에서 Llama-70B까지)도 출시 — 학생 모델에는 RL 없음. 강력한 RL 교사의 증류는 학생 규모에서 처음부터 RL을 일관되게 능가.

**추론에서 PPO 대신 GRPO를 사용하는 이유.** DeepSeekMath 논문(2024년 2월)의 세 가지 이유: (1) 훈련할 가치 네트워크 없음, 메모리 절반 감소; (2) 그룹 기준선은 추론 작업에서 발생하는 궤적 끝의 희소 보상을 자연스럽게 처리; (3) 프롬프트별 정규화는 난이도가 크게 다른 문제 간에 이점을 비교 가능하게 만들며, PPO의 단일 비평가는 이를 수행할 수 없음.

**검색 없음 vs 검색 기반.** 게임의 분기:

- *완벽한 정보 게임 + 긴 지평선*(바둑, 체스): 여전히 검색 기반. AlphaZero / MuZero가 우세.
- *LLM 추론*: 아직 생산 환경에 MCTS 없음; GRPO를 전체 롤아웃에 적용, 추론 계산을 위한 best-of-N. 프로세스 보상 모델(PRM)은 단계별 검색 추가를 암시.

## 구축 방법

`code/main.py`의 코드는 **소규모 GRPO**를 구현합니다 — 여러 그룹의 샘플을 가진 밴딧(bandit)입니다. 알고리즘은 LLM과 동일하지만 정책과 환경이 더 간단합니다. 이 코드는 *손실(loss)*과 *그룹 상대 이점(group-relative advantage)*을 학습하는데, 이는 2025년 혁신 기술입니다.

### 1단계: 초소형 검증 환경

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

실제 GRPO에서는 검증기가 단위 테스트를 실행하거나 수학적 등식을 확인합니다.

### 2단계: 정책: 프롬프트당 K개의 답변 토큰에 대한 소프트맥스

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

프롬프트에 조건화된 LLM의 최종 레이어 출력과 동등합니다.

### 3단계: 그룹 샘플링 및 그룹 상대 이점

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL 페널티: theta를 참조값으로 끌어당김
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

그룹 상대 이점은 2024년 DeepSeek의 기법입니다. 크리틱(critic)이 필요 없습니다. "기준선(baseline)"은 그룹 평균이며, 정규화는 그룹 표준편차(std)를 사용합니다.

### 4단계: REINFORCE 기준선과 비교 (가치 함수 없음)

동일한 설정, 동일한 계산량, 일반 REINFORCE. GRPO는 더 빠르고 안정적으로 수렴합니다.

### 5단계: 엔트로피 및 KL 관찰

RLHF와 동일한 진단: 참조값에 대한 평균 KL, 정책 엔트로피, 시간에 따른 보상. 이들이 안정화되면 훈련이 완료됩니다.

## 함정

- **검증자 조작을 통한 보상 해킹.** GRPO는 RLHF의 위험성을 그대로 물려받습니다: 검증자가 잘못되었거나 조작 가능한 경우 LLM은 해당 취약점을 악용합니다. 견고한 검증자(여러 테스트 케이스, 형식적 증명)가 중요합니다.
- **그룹 크기 너무 작음.** 그룹 기준선의 분산은 `1/√G`에 비례합니다. `G = 4` 미만에서는 이점 신호가 노이즈가 심합니다; 표준 선택값은 `G = 8`에서 `64`입니다.
- **길이 편향.** 다른 길이의 LLM 생성 출력은 다른 로그 확률을 가집니다. 토큰 수로 정규화하거나, 시퀀스 수준 로그 확률을 사용하거나, 최대 길이로 잘라내세요.
- **순수 자기 대국 순환.** AlphaZero 스타일 훈련은 일반합 게임에서 지배 순환(dominance loop)에 갇힐 수 있습니다. 다양한 상대 풀(리그 플레이, 레슨 10)로 완화됩니다.
- **탐색-정책 불일치.** AlphaZero는 탐색 출력을 모방하도록 정책을 훈련시킵니다. 정책 네트워크가 탐색의 분포를 표현하기에 너무 작으면 훈련이 정체됩니다.
- **계산 자원 바닥.** MuZero / AlphaZero는 막대한 계산 자원이 필요합니다. 단일 실험(ablation)만으로도 수백 GPU 시간이 소요됩니다. 학습을 위한 소형 데모가 존재합니다(예: 커넥트 포의 AlphaZero).
- **검증자 커버리지.** 버그가 있는 솔루션에 대해 통과하는 단위 테스트는 해당 버그를 강화합니다. 엣지 케이스를 포착하는 검증자를 설계하세요.

## 사용 사례

2026년 게임-RL 분야 현황, 도메인별 분류:

| 도메인 | 주요 방법론 |
|--------|-----------------|
| 2인 영합 보드 게임(바둑, 체스, 쇼기) | AlphaZero / MuZero / KataGo |
| 불완전 정보 카드 게임(포커) | CFR + 딥러닝(DeepStack, Libratus, Pluribus) |
| 아타리/픽셀 게임 | Muesli / MuZero / IMPALA-PPO |
| 대규모 멀티플레이어 전략(도타, 스타크래프트) | PPO + 자기 대국(self-play) + 리그(OpenAI Five, AlphaStar) |
| LLM 수학/코드 추론 | GRPO(DeepSeek-R1, Qwen-RL, 오픈소스 재현) |
| LLM 정렬(alignment) | DPO / RLHF-PPO(GRPO 아님; 검증자는 선호도는 검증 불가) |
| 로보틱스 | PPO + DR(게임-RL은 아니지만 동일한 정책-경사 도구 사용) |
| 조합 최적화 문제 | AlphaZero 변형(AlphaTensor, AlphaDev) |

*레시피* — 자기 대국, 탐색 강화 개선, 정책 증류 — 는 텍스트, 픽셀, 물리적 제어 분야를 아우릅니다. GRPO는 가장 최근에 등장한 사례이며, 더 많은 사례가 등장할 예정입니다.

## Ship It

`outputs/skill-game-rl-designer.md`로 저장:

```markdown
---
name: game-rl-designer
description: 주어진 도메인(게임-RL 또는 추론-RL 훈련 파이프라인(AlphaZero / MuZero / GRPO))을 설계합니다.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

목표(완전 정보 게임 / 불완전 정보 / Atari / LLM 추론 / 조합적)가 주어졌을 때 다음을 출력합니다:

1. 환경 적합성. 알려진 규칙? 마르코프성? 확률적? 다중 에이전트? AlphaZero vs MuZero vs GRPO 선택 기준.
2. 탐색 전략. MCTS(학습된 사전 확률을 가진 PUCT), Gumbel 샘플링, best-of-N 또는 없음.
3. 자기 대국 계획. 대칭 자기 대국 / 리그 / 오프라인 데이터 / 검증기 생성.
4. 목표 신호. 게임 결과 / 검증기 보상 / 선호도 / 학습된 모델. 강건성 계획 포함.
5. 진단. 기준 대비 승률, ELO 곡선, 검증기 통과율, 참조 모델과의 KL 발산.

불완전 정보 게임에 AlphaZero를 적용하지 마세요(CFR로 유도). 신뢰할 수 있는 검증기 없이 GRPO를 거부하세요. 고정된 기준 상대 집합 없이 게임-RL 파이프라인을 거부하세요(그렇지 않으면 자기 대국 ELO는 보정되지 않음).
```

## 연습 문제

1. **쉬움.** `code/main.py`에 GRPO 밴딧을 구현하세요. 2개의 프롬프트 × 각각 4개의 답변 토큰으로 학습시키세요. `G=8`로 1,000번 미만의 업데이트로 수렴시키세요.
2. **중간.** PPO(클리핑 적용)와 바닐라 REINFORCE를 통합하세요. 동일한 밴딧에서 GRPO와 비교하여 샘플 효율성과 보상 분산을 분석하세요.
3. **어려움.** 길이 2의 "추론 체인"으로 확장하세요: 에이전트가 두 토큰을 출력하고 검증기가 쌍에 대한 보상을 제공합니다. GRPO가 2단계 시퀀스 전반에 걸쳐 신용 할당을 어떻게 처리하는지 측정하세요. (힌트: *전체 시퀀스*당 그룹 어드밴티지를 계산하고, 두 토큰 위치 모두에 전파하세요.)

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| MCTS | "학습된 네트워크를 사용한 트리 탐색" | 몬테카를로 트리 탐색(Monte Carlo Tree Search); 학습된 `(p, v)` 사전 확률을 이용한 UCB1/PUCT 선택. |
| AlphaZero | "자기 대국 + MCTS" | MCTS 방문 횟수와 게임 결과에 맞춰 훈련된 정책-가치 네트워크(policy-value net). |
| MuZero | "학습된 모델을 사용하는 AlphaZero" | 학습된 동역학(learned dynamics)을 통해 잠재 공간(latent space)에서 동일한 루프 실행. |
| GRPO | "비평가 없는 PPO" | 그룹 상대 정책 최적화(Group Relative Policy Optimization); 그룹 평균 기준선(group-mean baseline) + KL을 사용한 REINFORCE. |
| PUCT | "AlphaZero의 UCB" | `Q + c · p · √N / (1 + N_a)` — 가치 추정과 사전 확률을 균형 있게 조정. |
| 자기 대국(Self-play) | "에이전트 vs 과거 자신" | 제로섬(zero-sum) 게임의 표준; 대칭적인 훈련 신호(symmetric training signal). |
| 리그 플레이(League play) | "개체군 기반 자기 대국" | 과거, 현재, 악용자(exploiters)를 샘플링하여 상대편으로 사용. |
| 검증자 보상(Verifier reward) | "검증 가능한 RL" | 결정론적 검증자(deterministic checker)에서 보상 발생(테스트 통과, 답변 일치). |
| 과정 보상(Process reward) | "PRM" | 최종 답변뿐 아니라 각 추론 단계(reasoning step)에 점수 부여.

## 추가 자료

- [Silver et al. (2017). 인간 지식 없이 바둑을 마스터하기 (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). 자기 대전을 통해 체스, 쇼기, 바둑을 마스터하는 일반 강화 학습 알고리즘 (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). 학습된 모델로 계획하여 아타리, 바둑, 체스, 쇼기 마스터하기 (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). 스타크래프트 II 그랜드마스터 수준 (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: 오픈 언어 모델에서 수학적 추론의 한계 확장 (GRPO)](https://arxiv.org/abs/2402.03300) — GRPO와 그룹-상대 기준선을 소개한 논문.
- [DeepSeek-AI (2025). DeepSeek-R1: 강화 학습을 통한 LLM의 추론 능력 강화](https://arxiv.org/abs/2501.12948) — 4단계 R1 레시피와 R1-Zero 제거 실험.
- [Brown et al. (2019). 멀티플레이어 포커를 위한 초인적 AI (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 대규모 CFR + 딥러닝.
- [Tesauro (1995). 시간차 학습 및 TD-게임몬](https://dl.acm.org/doi/10.1145/203330.203343) — 모든 것의 시작점.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 사용자 정의 보상 함수로 GRPO 적용을 위한 프로덕션 참조.
- [Qwen Team (2024). Qwen2.5-Math — GRPO 복제](https://github.com/QwenLM/Qwen2.5-Math) — 다양한 규모에서 R1 레시피의 오픈 복제.
- [Sutton & Barto (2018). Ch. 17 — 강화 학습의 최전선](http://incompleteideas.net/book/RLbook2020.pdf) — 자기 대전, 탐색, "설계된 보상"에 대한 교과서적 프레임워크로, R1이 LLM 규모에서 구현한 내용.