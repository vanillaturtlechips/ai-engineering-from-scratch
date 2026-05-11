# Proximal Policy Optimization (PPO)

> A2C는 각 롤아웃을 한 번 업데이트한 후 버립니다. PPO는 클리핑된 중요도 비율(clipped importance ratio)로 정책 경사(Policy Gradient)를 감싸서 동일한 데이터로 10+ 에포크를 수행해도 정책이 폭발하지 않도록 합니다. Schulman et al. (2017). 2026년 현재까지도 기본 정책 경사 알고리즘입니다.

**Type:** Build  
**Languages:** Python  
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)  
**Time:** ~75 minutes

## 문제 정의

A2C (레슨 07)는 온-폴리시(on-policy) 방식입니다: 기울기 `E_{π_θ}[A · ∇ log π_θ]`는 *현재* `π_θ`에서 샘플링된 데이터를 필요로 합니다. 한 번의 업데이트 후 `π_θ`가 변경되면, 사용한 데이터는 오프-폴리시(off-policy)가 됩니다. 이를 재사용하면 기울기가 편향됩니다.

롤아웃(rollout)은 비용이 많이 듭니다. Atari에서 8개 환경 × 128 스텝 = 1024 트랜지션과 수십 초의 환경 시간이 소요됩니다. 한 번의 기울기 단계 후 이를 버리는 것은 낭비입니다.

Trust Region Policy Optimization (TRPO, Schulman 2015)은 첫 번째 해결책이었습니다: 각 업데이트를 제약하여 이전 정책과 새 정책 간의 KL 발산(KL divergence)이 `δ` 이하로 유지되도록 합니다. 이론적으로는 깔끔하지만, 업데이트마다 공액 기울기(conjugate-gradient) 계산이 필요합니다. 2026년에는 아무도 TRPO를 실행하지 않습니다.

PPO (Schulman et al. 2017)는 엄격한 신뢰 영역 제약을 간단한 클리핑(clipped) 목적 함수로 대체합니다. 코드 한 줄 추가만으로 구현 가능합니다. 롤아웃당 10에폭(epoch) 학습이 가능하며, 공액 기울기 계산이 필요 없습니다. 충분한 이론적 보장을 제공하면서도, MuJoCo부터 RLHF까지 9년이 지난 지금까지도 기본 정책 기울기 알고리즘으로 사용됩니다.

## 개념

![PPO 클리핑된 대리 목적 함수: 1 ± ε에서의 비율 클리핑](../assets/ppo.svg)

**중요도 비율(importance ratio).**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

이는 새로운 정책과 데이터를 수집한 정책의 우도 비율입니다. `r_t = 1`은 변화가 없음을 의미합니다. `r_t = 2`는 새로운 정책이 이전 정책보다 `a_t`를 선택할 확률이 2배 높음을 의미합니다.

**클리핑된 대리 목적 함수(clipped surrogate).**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

두 가지 경우:

- 만약 이점(advantage) `A_t > 0`이고 비율이 `1 + ε`을 넘어서려고 하면, 클리핑은 그래디언트를 평탄화합니다 — 좋은 행동을 이전 확률보다 `+ε` 이상 더 밀어붙이지 않습니다.
- 만약 이점 `A_t < 0`이고 비율이 `1 - ε`을 넘어서려고 하면(클리핑된 감소 대비 나쁜 행동을 더 가능하게 만드는 경우), 클리핑은 그래디언트를 제한합니다 — 나쁜 행동을 `-ε` 이하로 밀어붙이지 않습니다.

`min`은 반대 방향을 처리합니다: 비율이 *유익한* 방향으로 이동했다면, 그래디언트를 그대로 얻습니다(해를 입히는 방향의 클리핑은 없음).

일반적인 `ε = 0.2`입니다. `r_t`의 함수로 목적 함수를 플롯하면, "좋은 쪽"에 평평한 지붕과 "나쁜 쪽"에 평평한 바닥이 있는 조각별 선형 함수가 됩니다.

**전체 PPO 손실 함수(full PPO loss).**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

A2C와 동일한 액터-크리틱 구조입니다. 세 가지 계수가 있으며, 일반적으로 `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`입니다.

**훈련 루프(training loop).**

1. `N`개의 병렬 환경에서 각각 `T` 스텝 동안 `N × T` 전이(transition)를 수집합니다.
2. 이점(GAE)을 계산하고 상수로 고정합니다.
3. 현재 `π_θ`의 스냅샷으로 `π_{θ_old}`를 고정합니다.
4. `K` 에포크 동안, 각 미니배치 `(s, a, A, V_target, log π_old(a|s))`에 대해:
   - `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`를 계산합니다.
   - `L^{CLIP}` + 가치 손실 + 엔트로피를 적용합니다.
   - 그래디언트 스텝을 수행합니다.
5. 롤아웃을 폐기하고 1단계로 돌아갑니다.

`K = 10`과 64 크기의 미니배치는 표준 하이퍼파라미터 세트입니다. PPO는 강건합니다: 정확한 수치는 ±50% 범위 내에서 거의 중요하지 않습니다.

**KL-페널티 변형(KL-penalty variant).** 원본 논문에서는 적응형 KL 페널티를 사용하는 대안을 제안했습니다: `L = L^{PG} - β · KL(π_θ || π_old)`로, `β`는 관측된 KL에 따라 조정됩니다. 클리핑 버전이 주로 사용되게 되었으며, KL 변형은 RLHF에서 살아남았습니다(참조 정책에 대한 KL은 항상 별도로 원하는 제약 조건이기 때문입니다).

## 구축 방법

### 단계 1: 롤아웃 시 `log π_old(a | s)` 캡처

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

스냅샷은 롤아웃 시 한 번만 캡처되며, 업데이트 에포크 동안 변경되지 않습니다.

### 단계 2: GAE(Generalized Advantage Estimation) 어드밴티지 계산 (레슨 07)

A2C와 동일합니다. 배치 단위로 정규화합니다.

### 단계 3: 클리핑된 서러게이트 업데이트

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # -surrogate 역전파, 가치 손실 추가, 엔트로피 감소
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # 클리핑됨
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"클리핑 → 제로 그래디언트" 패턴이 PPO의 핵심입니다. 새로운 정책이 이미 유익한 방향으로 너무 멀리 이동했다면 업데이트가 중단됩니다.

### 단계 4: 가치 및 엔트로피

A2C와 동일하게 크리틱에 표준 MSE를 추가하고 액터에 엔트로피 보너스를 적용합니다.

### 단계 5: 진단

업데이트마다 확인해야 할 세 가지 항목:

- **평균 KL(Kullback-Leibler) 발산** `E[log π_old - log π_θ]`. `[0, 0.02]` 범위에 유지되어야 합니다. `0.1`을 초과하면 `K_EPOCHS` 또는 `LR`을 줄이세요.
- **클리핑 비율** — 비율이 `[1-ε, 1+ε]` 범위를 벗어나는 샘플 비율. `~0.1-0.3`이 적당합니다. `~0`이면 클리핑이 작동하지 않음 → `LR` 또는 `K_EPOCHS`를 높이세요. `~0.5+`이면 롤아웃에 과적합됨 → 낮추세요.
- **설명된 분산** `1 - Var(V_target - V_pred) / Var(V_target)`. 크리틱 품질 지표입니다. 크리틱이 학습됨에 따라 1에 가까워져야 합니다.

## 함정(Pitfalls)

- **클리핑 계수(clip coefficient) 미조정.** `ε = 0.2`가 사실상의 표준입니다. `0.1`로 설정하면 업데이트가 너무 소극적으로 이루어지고, `0.3+`는 불안정성을 초래합니다.
- **너무 많은 에포크(epochs).** `K > 20`은 정책이 `π_old`에서 너무 멀어지기 때문에 일반적으로 불안정해집니다. 특히 대규모 네트워크의 경우 에포크 수를 제한하세요.
- **보상 정규화(reward normalization) 누락.** 큰 보상 스케일은 클리핑 범위를 침식합니다. 어드밴티지(advantage) 계산 전에 보상을 정규화하세요(실행 표준편차 기준).
- **어드밴티지 정규화(advantage normalization) 생략.** 배치별 평균 0/표준편차 1 정규화가 표준입니다. 이를 생략하면 대부분의 벤치마크에서 PPO가 실패합니다.
- **학습률(learning rate) 감소 미적용.** PPO는 선형 학습률 감소를 0까지 적용하는 것이 유리합니다. 상수 학습률은 종종 성능이 떨어집니다.
- **중요도 비율(importance ratio) 계산 오류.** 수치적 안정성을 위해 항상 `exp(log_new - log_old)`를 사용하세요. `new / old` 방식은 사용하지 마세요.
- **잘못된 그래디언트 부호.** 대리 목적함수(surrogate)를 최대화하려면 `-L^{CLIP}`을 *최소화*해야 합니다. 부호 반전은 가장 흔한 PPO 버그입니다.

## 사용 사례

PPO는 2026년 기준 놀라운 수의 도메인에서 기본 RL 알고리즘으로 사용됩니다:

| 사용 사례 | PPO 변형 |
|----------|-------------|
| MuJoCo / 로봇 제어 | 가우시안 정책(Gaussian policy), GAE(0.95)를 사용한 PPO |
| Atari / 이산 게임 | 범주형 정책(categorical policy), 128-스텝 롤아웃(rolling 128-step rollouts)을 사용한 PPO |
| LLM을 위한 RLHF | 참조 모델(reference model)에 대한 KL 페널티(KL penalty), 응답 종료 시 RM으로부터의 보상(reward)을 사용한 PPO |
| 대규모 게임 에이전트 | IMPALA + PPO (AlphaStar, OpenAI Five) |
| 추론 LLM | GRPO (레슨 12) — 크리틱(critic)이 없는 PPO 변형 |
| 선호도 전용 데이터 | DPO — PPO+KL의 폐쇄형 축소(closed-form collapsing), 온라인 샘플링 없음 |

PPO *손실 형태(loss shape)* — 클리핑된 대리 손실(clipped surrogate) + 가치(value) + 엔트로피(entropy) — 는 DPO, GRPO 및 거의 모든 RLHF 파이프라인의 기본 구조입니다.

## Ship It

`outputs/skill-ppo-trainer.md`로 저장:

```markdown
---
name: ppo-trainer
description: 주어진 환경에 대한 PPO 훈련 설정(config)과 진단 계획을 생성합니다.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

환경과 훈련 예산(training budget)이 주어졌을 때 다음을 출력합니다:

1. 롤아웃 크기. `N` 환경 × `T` 스텝.
2. 업데이트 일정. `K` 에포크, 미니배치 크기, 학습률(LR) 스케줄.
3. 대리(surrogate) 파라미터. `ε` (클리핑), `c_v`, `c_e`, 어드밴티지 정규화(advantage normalization) 활성화.
4. 어드밴티지. GAE(`λ`)에 명시적 `γ` (할인율)와 `λ` (GAE 파라미터) 적용.
5. 진단 계획. KL 발산, 클리핑 비율, 설명 분산(explained variance) 임계값과 경고(alerts).

`K > 30` 또는 `ε > 0.3` (불안정한 신뢰 영역)은 거부합니다. 어드밴티지 정규화나 KL/클리핑 모니터링이 없는 PPO 실행은 거부합니다. 클리핑 비율이 0.4 이상으로 지속될 경우 드리프트(drift)로 표시합니다.
```

## 연습 문제

1. **쉬움.** 4×4 GridWorld에서 `ε=0.2, K=4`로 PPO를 실행하세요. 환경 단계(env steps)를 일치시킨 상태에서 A2C(롤아웃당 1에폭)와 샘플 효율성을 비교하세요.
2. **중간.** `K ∈ {1, 4, 10, 30}`에 대해 스위프(sweep)를 수행하세요. 환경 단계 대비 보상(return)과 업데이트당 평균 KL(Kullback-Leibler) 발산을 추적하세요. 이 작업에서 KL이 발산하는 `K` 값은 얼마인가요?
3. **어려움.** 클리핑된 대리 목적 함수(clipped surrogate)를 적응형 KL 페널티로 교체하세요(`KL > 2·target`이면 `β`를 2배, `KL < target/2`이면 `β`를 1/2로 조정). 최종 보상(return), 안정성, 클리핑 비사용(clip-free) 여부를 비교하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 중요도 비율 | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`; 데이터를 수집한 정책과의 편차. |
| 클리핑된 대리 함수 | "PPO의 주요 기법" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; 유익한 측면에서 클리핑 후 평탄한 기울기. |
| 트러스트 리전 | "TRPO / PPO 의도" | 각 업데이트의 KL 발산을 제한하여 단조 개선 보장. |
| KL 페널티 | "소프트 트러스트 리전" | 대체 PPO: `L - β · KL(π_θ || π_old)`. 적응형 `β`. |
| 클리핑 비율 | "클리핑이 발생하는 빈도" | 진단 지표 — 0.1-0.3이어야 함; 이 범위 밖은 튜닝 오류 의미. |
| 다중 에포크 학습 | "데이터 재사용" | 각 롤아웃에 K 에포크 적용; 분산 비용 대신 샘플 효율성 확보. |
| 온-폴리시-ish | "대부분 온-폴리시" | PPO는 명목상 온-폴리시지만 K>1 에포크에서는 약간 오프-폴리시 데이터를 안전하게 사용. |
| PPO-KL | "다른 PPO" | KL 페널티 변형; RLHF에서 참조 정책 대비 KL이 이미 제약 조건인 경우 사용.

## 추가 자료

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 논문 원본.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, PPO의 전신.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 모든 PPO 하이퍼파라미터에 대한 실험.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; RLHF에서 PPO 적용 레시피.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — PyTorch를 활용한 깔끔한 현대식 설명.
- [CleanRL PPO 구현](https://github.com/vwxyzjn/cleanrl) — 많은 논문에서 참조하는 단일 파일 PPO 구현체.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 언어 모델용 PPO 프로덕션 레시피; 레슨 09(RLHF)와 함께 참고.
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37가지 코드 수준 최적화" 논문; PPO 트릭 중 어떤 것이 핵심이고 어떤 것이 경험적 지식인지 분석.