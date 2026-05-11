# 직접 선호 최적화(Direct Preference Optimization) 계열

> Rafailov et al. (2023)는 RLHF의 최적점이 선호 데이터 측면에서 닫힌 형태(closed form)를 가짐을 보였으며, 이를 통해 명시적인 보상 모델(reward model)을 생략하고 정책을 직접 최적화할 수 있습니다. 이 통찰은 DPO의 실패 모드를 각각 해결하는 IPO, KTO, SimPO, ORPO, BPO와 같은 계열을 탄생시켰습니다. 2026년에는 직접 정렬 알고리즘(DAAs)이 PPO보다 더 많은 프론티어 사후 훈련(post-training) 실행을 제공합니다. 하지만 2강에서 다룬 과적합 곡선은 여전히 적용됩니다: DAAs는 Goodhart의 법칙을 피하지 못하며, 단지 문제가 발생하는 위치만 바꿉니다.

**유형:** 학습
**언어:** Python (표준 라이브러리, six-variant 선호 손실 비교기)
**선수 지식:** 18단계 · 01 (InstructGPT), 18단계 · 02 (보상 해킹), 10단계 · 08 (DPO 기초)
**소요 시간:** ~75분

## 학습 목표

- RLHF-with-KL 최적점(optimal point)으로부터 DPO 폐쇄형(closed form)을 유도해 내라.
- IPO, KTO, SimPO, ORPO, BPO 각각이 DPO에서 해결하는 실패 모드(failure mode)를 설명하라.
- "암묵적 보상 차이(implicit reward gap)"와 "선호 강도(preference strength)"를 구분하고, IPO의 항등 사상(identity mapping)이 중요한 이유를 설명하라.
- Rafailov et al. (NeurIPS 2024)이 명시적 보상 모델(RM)이 없음에도 DAAs(Direct Preference Optimization Algorithms)가 과최적화(over-optimize)하는 것을 증명하는 이유를 설명하라.

## 문제 정의

RLHF 목적 함수(레슨 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

는 알려진 최적해를 가집니다:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

따라서 보상은 최적 정책과 참조 정책의 비율로 암묵적으로 정의됩니다:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

이를 Bradley-Terry 선호 가능도에 대입하면 분할 함수 `Z(x)`는 `x`에만 의존하므로 소거됩니다. 남은 것은 정책 파라미터만을 포함하는 손실 함수입니다 — 보상 모델이 필요 없습니다. 이것이 바로 DPO입니다.

문제점: 이 유도는 최적해에 도달 가능, 선호 데이터가 분포 내에 존재, 참조 정책이 진정한 기준점이라는 가정을 전제로 합니다. 이 중 어느 것도 정확히 성립하지 않습니다. 각 변형 방법(패밀리 멤버)은 서로 다른 위반된 가정을 수정합니다.

## 개념

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

잠재적 문제점:

- 암시적 보상 간격 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)`은 무제한입니다. 작은 선호도 차이가 임의로 큰 간격을 생성할 수 있습니다.
- 손실 함수는 선택된 응답과 거부된 응답의 로그 확률을 반대 방향으로 유도합니다. 거부된 응답의 로그 확률이 더 빠르게 감소하는 한 선택된 응답의 절대 로그 확률을 낮출 수 있습니다. 이는 "저하된 선택 응답(Degraded Chosen Response)" 현상입니다.
- 분포 외 선호도(드문 쌍 vs 드문 쌍)는 임의의 암시적 보상을 생성합니다.

### IPO (Azar et al., 2024)

아이덴티티 선호도 최적화(Identity Preference Optimization)는 로그-시그모이드 함수를 선호도 확률에 대한 아이덴티티 매핑으로 대체합니다. 손실 함수는 유계된 목표에 대한 제곱 오차 형태로 변환됩니다:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

마진은 `1/(2 beta)`로 유계됩니다. 선호도 강도와 암시적 보상 간격은 비례하며, 발산 현상이 없습니다.

### KTO (Ethayarajh et al., 2024)

카네만-트버스키 최적화(Kahneman-Tversky Optimization)는 쌍별 구조를 완전히 제거합니다. 단일 레이블된 출력과 이진 "바람직함" 또는 "바람직하지 않음" 신호를 사용하여 전망 이론(prospect-theory) 유틸리티로 매핑합니다:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

이익과 손실에 서로 다른 가중치(손실 회피)를 적용합니다. 이점: 훨씬 더 풍부한 비쌍별 데이터를 사용할 수 있습니다.

### SimPO (Meng et al., 2024)

단순 선호도 최적화(Simple Preference Optimization)는 훈련 신호를 생성과 정렬합니다. 참조 정책을 완전히 제거하고 로그 확률을 길이로 정규화합니다:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

안정성을 위한 마진 `gamma`를 추가합니다. 길이 정규화는 DPO의 길이 편향 실패 모드(더 긴 `y_w`가 구조적으로 더 큰 로그 확률 간격을 생성)를 제거합니다.

### ORPO (Hong et al., 2024)

오즈 비율 선호도 최적화(Odds-Ratio Preference Optimization)는 표준 지도 학습(SFT) 음의 로그 우도에 선호도 항을 추가합니다:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

참조 정책이 없으며, SFT 항이 정규화 역할을 합니다. 기본 모델에서 정렬된 모델까지 단일 단계로 훈련하며, 별도의 SFT 체크포인트가 필요 없습니다.

### BPO (ICLR 2026 제출, OpenReview id=b97EwMUWu7)

"저하된 선택 응답" 문제를 식별합니다: DPO는 `y_w > y_l` 순위를 유지하지만 `y_w`의 절대 로그 확률이 감소할 수 있습니다. BPO는 선택된 응답의 하향 이동을 페널티하는 단일 라인 수정 사항을 추가합니다. Llama-3.1-8B-Instruct에서 수학 추론 정확도가 DPO 대비 +10.1% 향상되었다고 보고됩니다.

### 보편적 결과: DAA는 여전히 과최적화합니다

Rafailov et al. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024)는 여러 데이터셋에서 KL 예산에 따라 DPO, IPO, SLiC로 정책을 훈련했습니다. 골드-보상 대 KL 곡선은 Gao et al.의 피크-붕괴 형태를 보입니다. 암시적 보상 쿼리는 훈련 중 분포 외 샘플을 대상으로 하며, KL 정규화는 이를 안정화하지 못합니다.

DAA는 Goodhart의 법칙을 벗어나지 못합니다. 과최적화가 발생하는 표면을 "보상 모델 과최적화"에서 "참조 정책 비율 과최적화"로 변경할 뿐입니다. 보편적인 해결책인 더 나은 데이터, 앙상블, 조기 중지는 둘 모두에 적용됩니다.

### 선택 방법 (2026)

- 대규모 쌍별 선호도 데이터가 있는 경우: 보수적인 `beta`를 사용한 DPO, 길이 편향이 뚜렷한 경우 SimPO.
- 비쌍별 이진 피드백이 있는 경우: KTO.
- 기본 모델부터 단일 단계 파이프라인을 원하는 경우: ORPO.
- DPO 로그에서 저하된 선택 로그 확률이 관찰되는 경우: BPO.
- 선호도 강도가 크게 변하고 DPO가 포화되는 경우: IPO.

모든 연구실은 5가지 방법을 모두 테스트하고 작업별로 승자를 선택합니다. 수학 추론과 안전에 대해 최적점이 동일할 이유가 없습니다.

## 사용 방법

`code/main.py`는 쌍별 실제 선호 강도가 변하는 장난감 선호 데이터셋에서 6가지 손실 함수(DPO, IPO, KTO, SimPO, ORPO, BPO)를 비교합니다. 각 손실 함수는 작은 소프트맥스 정책을 사용하여 동일한 500개 쌍 샘플에 대해 최적화됩니다. 플롯은 방법별 최종 승률, 선택된 로그 확률 변화, 암시적 보상 분산을 보여줍니다.

## Ship It

이 레슨은 `outputs/skill-preference-loss-selector.md`를 생성합니다. 데이터셋 통계(쌍(pair) vs 비쌍(unpaired), 가변(variable) vs 균일(uniform) 선호도 강도, 길이 분포)와 목표(단일 단계(single-stage) 또는 SFT-then-선호도)를 고려하여 선호도 손실 함수를 추천하고, 해당 손실 함수가 보호하는 실패 모드(failure mode)를 보고합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. DPO와 BPO의 최종 선택된 로그 확률 감소(chosen-log-prob drop)를 보고하세요. BPO는 더 높은 선택된 절대 확률을 유지해야 합니다 — 이를 확인하세요.

2. 모든 쌍의 선호도가 동일한 강도를 갖도록 선호도 데이터를 수정하세요. 6가지 방법 중 가장 강건한 방법은 무엇인가요? 어떤 방법이 성능이 저하되나요? IPO의 장점을 설명하세요.

3. 거부된 응답(rejected responses)을 선택된 응답(chosen responses)보다 평균적으로 2배 길게 만드세요. 다른 것은 변경하지 않고, DPO의 길이 편향(length exploitation)을 수치적으로 보여주고 SimPO의 해결 방법을 설명하세요.

4. Rafailov et al. (NeurIPS 2024)은 DAA(직접 선호 최적화)가 과도한 최적화를 한다고 주장합니다. 단일 포인트 버전을 재현하세요: 선택된-거부된 KL 발산(chosen-minus-rejected KL divergence)을 플롯하고 큰 베타(beta)에서 DPO의 과도한 최적화를 관찰하세요.

5. BPO 논문 초록(OpenReview b97EwMUWu7)을 읽으세요. BPO가 DPO에 추가하는 한 줄짜리 수정 사항을 작성하세요. `code/main.py`의 구현과 대조하여 확인하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| DPO | "보상 모델 없는 RLHF" | 닫힌 형식의 RLHF 최적값에서 유도된 손실; 정책 파라미터만 사용 |
| 암시적 보상 | "로그 비율" | `beta * log(pi(y|x) / pi_ref(y|x))` — DPO에서 암시된 보상 |
| IPO | "제한된 DPO" | 로그-시그모이드를 항등 함수로 대체; 암시적 보상 격차는 `1/(2 beta)`로 제한 |
| KTO | "비쌍 DPO" | 손실 회피가 있는 단일 레이블에 대한 프로스펙트 이론 효용 |
| SimPO | "참조 정책 없는 DPO" | 길이 정규화된 로그 우도 + 마진; 참조 정책 없음 |
| ORPO | "단일 단계 DPO" | NLL + 승산 비율 선호도 항; 기본 모델부터 한 번에 훈련 |
| BPO | "선택 응답 보존 DPO" | DPO에 선택 응답의 절대 로그 확률 감소 페널티 추가 |
| 열화된 선택 | "선택 응답 감소" | DPO는 거부 응답보다 빠르게 감소하는 한 선택 로그 확률을 감소시킴 |
| DAA | "직접 정렬 알고리즘" | 명시적 RM을 생략하는 모든 선호도 손실 방법 |

## 추가 자료

- [Rafailov et al. — 직접 선호 최적화(Direct Preference Optimization, NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — 인간 선호로부터 학습을 이해하기 위한 일반적인 이론적 패러다임(A General Theoretical Paradigm to Understand Learning from Human Preferences, AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: 전망 이론 최적화로서의 모델 정렬(KTO: Model Alignment as Prospect Theoretic Optimization, arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO(단순 선호 최적화, NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO(최적 응답 정책 최적화, EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — 행동 보존 최적화(Behavior Preservation Optimization, ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — DAA에서의 RM 과적합에 대한 확장 법칙(Scaling Laws for RM Overoptimization in DAAs, NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)