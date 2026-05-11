# 보상 모델링 & RLHF

> 인간은 "좋은 어시스턴트 응답"에 대한 보상 함수를 직접 작성할 수 없지만, 두 응답을 비교하여 더 나은 것을 선택할 수 있습니다. 이러한 비교를 통해 보상 모델을 학습시킨 후, 언어 모델을 이 보상 모델에 대해 강화 학습합니다. Christiano 2017. InstructGPT 2022. GPT-3를 ChatGPT로 변환한 레시피입니다. 2026년에는 대부분 DPO로 대체되고 있지만, 이 멘탈 모델은 여전히 유효합니다.

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** Phase 5 · 05 (감성 분석), Phase 9 · 08 (PPO)
**소요 시간:** ~45분

## 문제

다음 토큰 예측 목표(next-token-prediction objective)로 언어 모델을 훈련시켰습니다. 이 모델은 문법적으로 올바른 영어를 생성합니다. 하지만 거짓말을 하고, 장황하게 말하며, 거절을 거부합니다. 더 많은 사전 훈련(pretraining)으로는 이 문제를 해결할 수 없습니다 — 웹 텍스트(web text)가 문제이지 해결책이 아닙니다.

"지시문 X에 대해 응답 A가 응답 B보다 더 좋다"는 *스칼라 보상(scalar reward)*을 원합니다. 이 보상 함수를 직접 작성하는 것은 불가능합니다. "도움이 되는 정도(helpfulness)"는 토큰에 대한 닫힌 형식의 표현(closed-form expression)이 아닙니다. 하지만 인간은 두 출력을 비교하고 선호도를 표시할 수 있습니다. 이는 대규모로 수집하기에 비용이 적게 듭니다.

RLHF(Christiano et al. 2017; Ouyang et al. 2022)는 선호도를 보상 모델(reward model)로 변환한 다음, 해당 보상에 대해 PPO를 사용하여 언어 모델(LM)을 최적화합니다. 3단계로 진행됩니다: SFT(초기 훈련) → RM(보상 모델) → PPO. 이 방법은 ChatGPT, Claude, Gemini 및 2023–2025년에 출시된 모든 정렬된 대형 언어 모델(aligned-LLM)에 적용된 레시피입니다.

2026년에는 PPO 단계가 대부분 DPO(Phase 10 · 08)로 대체되었습니다. DPO는 비용이 적게 들고 정렬 튜닝(alignment tuning)에 거의 동일한 성능을 보이기 때문입니다. 하지만 *보상 모델* 부분은 여전히 모든 Best-of-N 샘플러, 모든 검증 가능한 보상 기반 강화 학습(RL-from-verifiable-rewards) 파이프라인, 그리고 프로세스 보상 모델(process reward model)을 사용하는 모든 추론 모델의 기반이 됩니다. RLHF를 이해하면 전체 정렬 스택(alignment stack)을 이해하는 것입니다.

## 개념

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**1단계: 지도 미세 조정(SFT, Supervised Fine-Tuning).** 사전 훈련된 기본 모델에서 시작합니다. 목표 행동(지시 따르기 응답, 도움이 되는 답변 등)에 대한 인간이 작성한 데모를 사용해 미세 조정합니다. 결과: *좋은 행동에 편향*되었지만 여전히 무한한 행동 공간을 가진 모델 `π_SFT`.

**2단계: 보상 모델 훈련.**

- 프롬프트 `x`에 대한 응답 쌍 `(y_+, y_-)`을 수집합니다. 인간이 "y_+가 y_-보다 선호된다"고 라벨링합니다.
- 보상 모델 `R_φ(x, y)`를 훈련시켜 `y_+`에 더 높은 점수를 할당합니다.
- 손실 함수: **Bradley-Terry 쌍별 로지스틱**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ는 시그모이드 함수입니다. 보상 차이는 선호도의 로그 오즈를 의미합니다. BT는 1952년(Bradley-Terry) 이후 표준이 되었으며 현대 RLHF에서 가장 널리 사용되는 선택입니다.

- `R_φ`는 일반적으로 스칼라 헤드가 추가된 SFT 모델에서 초기화됩니다. 동일한 트랜스포머 백본; 단일 선형 레이어가 보상을 출력합니다.

**3단계: KL 페널티가 있는 PPO를 통한 RM 최적화.**

- 훈련 가능한 정책 `π_θ`를 `π_SFT`에서 초기화합니다. 고정된 *참조* `π_ref = π_SFT`를 유지합니다.
- 응답 `y` 종료 시 보상:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL 페널티는 `π_θ`가 `π_SFT`에서 임의로 벗어나는 것을 방지합니다. *규제자* 역할을 하며, 하드 신뢰 영역이 아닙니다. `β`는 일반적으로 `0.01`-`0.05`입니다.
- 이 보상으로 PPO(레슨 08)를 실행합니다. 어드밴티지는 토큰 수준 궤적에서 계산되지만, RM은 전체 응답만 점수화합니다.

**KL 페널티가 필요한 이유?** KL 페널티가 없으면 PPO는 보상 해킹 전략을 쉽게 찾습니다. RM은 분포 내 완성본만으로 훈련되었기 때문입니다. 분포 외 응답이 인간이 작성한 어떤 응답보다 높은 점수를 받을 수 있습니다. KL 페널티는 `π_θ`를 RM이 훈련된 매니폴드 근처에 유지합니다. RLHF에서 가장 중요한 조절 장치입니다.

**2026년 현황:**

- **DPO**(Rafailov 2023): 닫힌 형식의 대수적 접근으로 2+3단계를 단일 지도 손실 함수로 축소합니다. RM, PPO 불필요. 정렬 벤치마크에서 동일한 품질을 계산 비용의 일부로 달성합니다. 10단계·08에서 다룹니다.
- **GRPO**(DeepSeek 2024–2025): 크리틱 대신 그룹 상대 기준선을 사용하는 PPO, 인간이 훈련한 RM 대신 *검증자*(코드 실행/수학 정답 일치)로부터 보상을 받습니다. 추론 모델에서 주류입니다. 9단계·12에서 다룹니다.
- **프로세스 보상 모델(PRMs):** 부분 해(추론 단계 각각)를 점수화하며, 추론용 RLHF 및 GRPO 변형에서 사용됩니다.
- **헌법적 AI/RLAIF:** 인간 대신 정렬된 LLM을 사용해 선호도를 생성합니다. 선호도 예산을 확장합니다.

## 구축 방법

이 레슨은 문자열로 표현된 작은 합성 "프롬프트"와 "응답"을 사용합니다. RM은 토큰 백(bag-of-tokens) 표현에 대한 선형 점수 모델입니다. 실제 LLM은 사용하지 않습니다 — 파이프라인의 *형태*가 중요하며 규모는 중요하지 않습니다. `code/main.py`를 참조하세요.

### 1단계: 합성 선호 데이터

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

실제 RLHF에서는 인간 라벨러가 이 부분을 대체합니다. 형태 — `(프롬프트, 선호 응답, 거부 응답)` — 는 동일합니다.

### 2단계: Bradley-Terry 보상 모델

선형 점수: `R(x, y) = w · bag(y)`. BT 쌍별 로그 손실 최소화를 위해 훈련:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

수백 번의 업데이트 후, `w`는 좋은 단어 토큰에 양의 가중치를, 나쁜 단어 토큰에 음의 가중치를 할당합니다.

### 3단계: RM 위의 PPO 유사 정책

우리의 장난감 정책은 어휘에서 단일 토큰을 생성합니다. RM 하에서 토큰을 점수화하고, `log π_θ(토큰 | 프롬프트)`를 계산하며, 참조에 대한 KL 페널티를 추가하고, 클리핑된 PPO 대리 함수를 적용합니다.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # 보상을 반환값으로 취급하여 theta에 대한 ppo 스타일 업데이트
    ...
```

### 4단계: KL 모니터링

업데이트마다 평균 `KL(π_θ || π_ref)`를 추적합니다. 값이 `~5-10`을 넘어서면 정책이 `π_SFT`에서 멀리 떨어진 것입니다 — `β`가 상승하거나 보상 해킹이 시작되고 있을 수 있습니다. 이는 실제 RLHF에서 가장 중요한 진단 지표입니다.

### 5단계: TRL을 사용한 프로덕션 레시피

장난감 파이프라인을 이해한 후, 실제 라이브러리 사용자가 작성하는 동일한 루프를 소개합니다. Hugging Face의 [TRL](https://huggingface.co/docs/trl)은 참조 구현입니다 — 2단계는 `RewardTrainer`, 3단계는 (KL-to-reference가 내장된) `PPOTrainer`를 사용합니다.

```python
# 2단계: 쌍별 선호도로부터 보상 모델
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# 데이터셋 행: {"prompt", "chosen", "rejected"} — Bradley-Terry 형식
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# 3단계: SFT 참조에 대한 KL 페널티와 함께 RM에 대한 PPO
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # 동결

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats에는 다음이 포함: mean_kl, clip_frac, value_loss — 세 가지 PPO 진단 지표
```

라이브러리가 자동으로 처리하는 세 가지 사항. `adap_kl_ctrl=True`는 적응형-β 스케줄을 구현합니다: 관측된 KL이 `target_kl`을 초과하면 β가 2배가 되고, 절반 미만이면 β가 절반으로 줄어듭니다. 참조 모델은 관례적으로 동결됩니다 — `policy`와 파라미터를 실수로 공유해서는 안 됩니다. 그리고 값 헤드는 정책과 동일한 백본(`AutoModelForCausalLMWithValueHead`는 스칼라 MLP 헤드를 부착)에 있기 때문에 TRL은 `policy/kl`과 `value/loss`를 별도로 보고합니다.

## 함정(Pitfalls)

- **과도한 최적화 / 보상 해킹(reward hacking).** RM은 불완전합니다. `π_θ`는 높은 점수를 받지만 실제로는 나쁜 적대적 완성(adversarial completions)을 찾습니다. 증상: 보상(reward)이 무한히 증가하는 반면 인간 평가 점수는 정체되거나 하락합니다. 해결: 조기 중단, `β` 증가, RM 훈련 데이터 확장.
- **길이 해킹(length hacking).** 도움이 되는 응답으로 훈련된 RM은 종종 암묵적으로 길이(length)를 보상합니다. 정책이 응답을 패딩(padding)하는 방법을 학습합니다. 해결: 길이 정규화 보상(length-normalized reward) 또는 길이 인식 RM을 사용한 RLAIF.
- **너무 작은 RM.** RM은 정책(policy)만큼 커야 합니다. 작은 RM은 정책의 출력을 충실히 평가할 수 없습니다.
- **KL 조정(KL tuning).** 너무 낮은 `β` → 드리프트(drift) 및 보상 해킹. 너무 높은 `β` → 정책이 거의 변하지 않음. 표준 기법은 *적응형(adaptive)* `β`로, 단계당 고정 KL을 대상으로 합니다.
- **선호 데이터 노이즈.** 인간 라벨의 ~30%는 노이즈가 있거나 모호합니다. 동의 필터링(agreement-filtered) 데이터로 RM을 훈련하거나 BT에 온도(temperature)를 적용하여 보정(calibrate)합니다.
- **오프-정책 문제(off-policy problems).** PPO 데이터는 첫 번째 에포크 이후 약간 오프-정책입니다. 레슨 08과 같이 클리프 비율(clip fraction)을 모니터링합니다.

## 사용 방법

2026년의 RLHF는 계층적으로 구성됩니다:

| 레이어 | 대상 | 방법 |
|-------|--------|--------|
| 지시 따르기, 유용성, 무해성 | 정렬 | RLHF-PPO보다 DPO(Phase 10 · 08)가 선호됨. |
| 추론 정확성(수학, 코드) | 능력 | 검증기 보상(GRPO with verifier reward, Phase 9 · 12)을 사용한 GRPO. |
| 장기 다단계 작업 | 에이전트성 | 단계별 프로세스 보상 모델을 활용한 PPO / GRPO. |
| 안전 / 거부 행동 | 안전 | 별도의 안전 보상 모델(RM)을 사용한 RLHF-PPO 또는 Constitutional AI. |
| 추론 시 Best-of-N | 빠른 정렬 | 디코딩 시 RM 사용; 정책 훈련 불필요. |
| 보상 증류 | 추론 연산 | 고정된 언어 모델(LM) 위에 작은 "보상 헤드"를 훈련. |

RLHF는 2022–2024년에 *주요* 방법이었습니다. 2026년에는 프로덕션 정렬 파이프라인이 DPO를 우선하고, 보상 모델(RM)이 많이 필요하거나 안전-중요 단계에서만 PPO를 사용합니다.

## Ship It

`outputs/skill-rlhf-architect.md`로 저장:

```markdown
---
name: rlhf-architect
description: 언어 모델을 위한 RLHF / DPO / GRPO 정렬 파이프라인 설계, RM, KL, 데이터 전략 포함.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

기본 LM, 목표 행동(정렬 / 추론 / 거부 / 에이전트), 선호도 또는 검증자 예산이 주어졌을 때 다음을 출력:

1. 단계. SFT? RM? DPO? GRPO? 근거 포함.
2. 선호도 또는 검증자 소스. 인간, AI 피드백, 규칙 기반, 단위 테스트 통과, 또는 보상 증류.
3. KL 전략. 고정 β, 적응형 β, 또는 DPO(암묵적 KL).
4. 진단. 평균 KL, 보상 안정성, 과최적화 방지(홀드아웃 인간 평가).
5. 안전 게이트. 레드 팀 세트, 거부율, 안전성 RM과 유용성 RM 분리.

KL 모니터 없이 RLHF-PPO를 출시하지 마세요. 목표 정책보다 작은 RM 사용을 거부하세요. 길이만 고려한 보상을 거부하세요. 블라인드 인간 평가 세트를 보유하지 않은 파이프라인은 과최적화 방지 기능이 부족하다고 표시하세요.
```

## 연습 문제

1. **쉬움.** `code/main.py`에서 Bradley-Terry 보상 모델을 500개의 합성 선호 쌍으로 학습시키세요. 100개의 홀드아웃 쌍에 대해 쌍별 정확도를 측정하세요. 90%를 초과해야 합니다.
2. **중간.** `β ∈ {0.0, 0.1, 1.0}`로 장난감 PPO-RLHF 루프를 실행하세요. 각각에 대해 업데이트별 RM 점수 대 참조 KL을 그래프로 그리세요. 어떤 실행에서 보상 해킹이 발생하나요?
3. **어려움.** 동일한 선호 데이터에 대해 DPO(닫힌 형식의 선호-우도 손실)를 구현하고 RLHF-PPO 파이프라인과 사용된 계산량 및 최종 RM 점수를 비교하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| RLHF | "정렬 RL" | 3단계 SFT + RM + PPO 파이프라인 (Christiano 2017, Ouyang 2022). |
| 보상 모델(RM) | "점수 매기는 네트워크" | Bradley-Terry를 통해 쌍별 선호도에 적합된 학습된 스칼라 함수. |
| Bradley-Terry | "쌍별 로지스틱 손실" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; 표준 RM 목적 함수. |
| KL 페널티 | "참조 모델 근처 유지" | 보상 내 `β · KL(π_θ || π_ref)`; 보상 해킹 방지 정규화 항. |
| 보상 해킹 | "Goodhart의 법칙" | 정책이 RM 결함을 악용; 증상: 보상 증가, 인간 평가 정체. |
| RLAIF | "AI 라벨링 선호도" | 라벨이 인간이 아닌 다른 언어 모델(LM)에서 나오는 RLHF. |
| PRM | "과정 보상 모델" | 부분 추론 단계 점수 매김; 추론 파이프라인에서 사용. |
| 헌법 AI | "Anthropic의 방법" | 명시적 규칙에 따라 AI 생성 선호도.

## 추가 읽기 자료

- [Christiano et al. (2017). 인간 선호로부터 심층 강화 학습(Deep Reinforcement Learning from Human Preferences)](https://arxiv.org/abs/1706.03741) — RLHF를 시작한 논문.
- [Ouyang et al. (2022). 지시 따르기 언어 모델 학습(InstructGPT — Training language models to follow instructions with human feedback)](https://arxiv.org/abs/2203.02155) — ChatGPT의 레시피.
- [Stiennon et al. (2020). 인간 피드백을 통한 요약 학습(Learning to summarize with human feedback)](https://arxiv.org/abs/2009.01325) — 요약 작업을 위한 초기 RLHF.
- [Rafailov et al. (2023). 직접 선호 최적화(Direct Preference Optimization)](https://arxiv.org/abs/2305.18290) — DPO; 2026년 이후 RLHF의 기본 접근법.
- [Bai et al. (2022). 헌법 AI: AI 피드백으로부터의 무해성(Constitutional AI: Harmlessness from AI Feedback)](https://arxiv.org/abs/2212.08073) — RLAIF와 자기 비판 루프.
- [Anthropic RLHF 논문 (Bai et al. 2022). 도움이 되고 무해한 조교 학습(Training a Helpful and Harmless Assistant)](https://arxiv.org/abs/2204.05862) — HH 논문.
- [Hugging Face TRL 라이브러리](https://huggingface.co/docs/trl) — 프로덕션용 `RewardTrainer` 및 `PPOTrainer`. 적응형 KL 및 가치 헤드 세부 정보는 트레이너 소스 코드를 참조.
- [Hugging Face — 인간 피드백으로부터 강화 학습 설명(Illustrating Reinforcement Learning from Human Feedback)](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — 다이어그램과 함께 3단계 파이프라인을 설명하는 표준 가이드.
- [von Werra et al. (2020). TRL: 트랜스포머 강화 학습(Transformer Reinforcement Learning)](https://github.com/huggingface/trl) — 라이브러리; `examples/`에는 Llama, Mistral, Qwen용 엔드투엔드 RLHF 스크립트가 포함됨.
- [Sutton & Barto (2018). Ch. 17.4 — 보상 신호 설계(Designing Reward Signals)](http://incompleteideas.net/book/RLbook2020.pdf) — 보상 가설 관점; 보상 해킹 사고 필수 전제 조건.