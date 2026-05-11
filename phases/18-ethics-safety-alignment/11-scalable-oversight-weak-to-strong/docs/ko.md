# 확장 가능한 감독과 약-강 일반화

> Burns et al. (OpenAI Superalignment, "Weak-to-Strong Generalization", 2023)는 초정렬(superalignment) 문제에 대한 대리 접근법을 제안했습니다: 더 약한 모델이 생성한 레이블을 사용하여 강력한 모델을 파인튜닝(fine-tuning)하는 것입니다. 만약 강력한 모델이 불완전한 약한 감독으로부터 올바르게 일반화한다면, 현재의 인간 수준 정렬 방법이 초인간 시스템으로 확장될 수 있습니다. 확장 가능한 감독(scalable oversight)과 W2SG(Weak-to-Strong Generalization)는 상호 보완적입니다. 확장 가능한 감독(토론, 재귀적 보상 모델링, 작업 분해)은 감독자의 효과적 능력을 증가시켜 감독 대상 모델을 따라잡을 수 있게 합니다. W2SG는 강력한 모델이 감독자가 제공하는 어떤 불완전한 감독으로부터도 올바르게 일반화하도록 보장합니다. "Debate Helps W2SG" (arXiv:2501.13124, 2025년 1월)는 이 둘을 결합합니다.

**유형:** 학습
**언어:** Python (표준 라이브러리, W2SG 격차 시뮬레이터)
**선수 지식:** 18단계 · 01 (지시 따르기), 18단계 · 10 (AI 제어), 09단계 (강화 학습 기초)
**소요 시간:** ~60분

## 학습 목표

- 확장 가능한 감독(scalable oversight)과 약-강 일반화(weak-to-strong generalization)를 정의하고, 이들이 어떻게 상호 보완적인지 설명하라.
- Burns et al. 2023 실험 설정을 설명하라: GPT-2의 레이블을 사용하여 GPT-4를 파인튜닝(fine-tuning)하기.
- 성능 회복 격차(PGR, performance gap recovered) 메트릭과 그것이 측정하는 바를 설명하라.
- 세 가지 주요 확장 가능한 감독 메커니즘(토론(debate), 재귀적 보상 모델링(recursive reward modeling), 작업 분해(task decomposition))과 각각의 강점 하나를 제시하라.

## 문제 정의

18단계(Phase 18)까지의 모든 정렬(alignment) 기법은 감독자(overseer)가 모델의 행동을 평가할 수 있다고 가정합니다. 그러나 모델이 초인적인(superhuman) 능력을 갖추게 되면 감독자가 가장 약한 고리가 됩니다. 초정렬(superalignment) 문제의 핵심은 다음과 같습니다: 더 약한 감독자가 더 강력하고 정렬된 모델을 안정적으로 생성할 수 있는가?

Burns 등은 이 문제를 다음과 같은 실증적 설정으로 축소합니다: 약한 감독자가 강한 모델을 감독할 때, 강한 모델의 능력 중 얼마나 많은 부분이 약한 감독을 거쳐도 살아남는지 측정합니다. 이는 초정렬 문제의 해결책이 아니라, 진행 상황을 측정할 수 있는 대리 지표(proxy)입니다.

## 개념

### W2SG: 번스 등(Burns et al.)의 설정

- 약한 모델: GPT-2 클래스.
- 강한 모델: GPT-4 클래스.
- 목표: 작업에서 강한 GPT-4의 상한선(ceiling) 달성.

절차:
1. 작업에서 약한 모델의 제로샷(zero-shot) 예측 획득.
2. 약한 라벨이 지정된 데이터로 강한 모델 파인튜닝.
3. 파인튜닝된 강한 모델의 정확도 측정.

기준선: 약한 모델 자체의 정확도, 골드 라벨에 대한 강한 모델의 지도 학습 상한선.

격차 지표: 성능 격차 회복도(PGR) = (파인튜닝 - 약한 모델) / (상한선 - 약한 모델). PGR 1.0은 약한 감독이 격차를 완전히 해소함을 의미; PGR 0은 약한 감독이 도움이 되지 않았음을 의미.

### 번스 등(Burns et al.)의 실증적 발견

NLP, 체스 퍼즐, 보상 모델링 작업에서 PGR은 일관되게 양수: 강한 모델은 약한 감독자의 실수를 넘어 일반화됨. 작업별 PGR은 ~20%에서 ~80%까지 다양. 강한 모델은 불완전한 라벨에서 의도된 작업을 "이해"하며 사전 학습된 사전 확률(prior)로 채움.

번스 등이 지적한 한계:
- 여기서 약한-강한 격차는 능력이 아닌 정렬(alignment) 문제. 약한 모델은 정렬되어 있음; 진정한 초인적 모델은 인간에 의해 정렬될 수 있지만 훨씬 큰 능력 격차가 존재.
- 강한 모델의 일반화는 실제 의도 회복보다 작업 사전 확률에 더 기인할 수 있음.
- 강한 모델의 잠재적 지식을 끌어내는 것이 진짜 문제; PGR은 특정 운영화(operationalization)를 측정.

### 확장 가능한 감독: 세 가지 메커니즘

- **토론(Debate, Irving et al. 2018).** U의 두 인스턴스가 반대 입장을 주장; 약한 판사가 결정. 가정: 설득력 있는 참 답을 찾는 것이 설득력 있는 거짓 답을 찾는 것보다 쉬움. 최근 2024-2025 연구(Khan et al., Michael et al.)는 토론이 도움이 될 때도 있고 아닐 때도 있으며, 도움 여부는 작업 구조에 달렸음을 보여줌.
- **재귀적 보상 모델링(Recursive Reward Modeling, Leike et al. 2018).** U가 인간이 U+1의 보상 모델을 훈련하도록 도움. 감독자의 효과적 능력은 모델과 함께 성장.
- **작업 분해(Task Decomposition, Christiano, Shlegeris, Amodei 2018).** 어려운 작업을 인간이 확인할 수 있는 하위 작업으로 분해, 재귀적 적용. 분해 가능성 가정.

각 메커니즘은 작업 구조나 중간 구성 요소의 정렬에 대해 특정 가정을 함.

### 확장 가능한 감독과 W2SG의 상호보완성

확장 가능한 감독은 감독자의 효과적 신호 품질을 증가시킴.
W2SG는 감독자가 제공할 수 있는 어떤 불완전한 신호로부터의 격차를 해소.

Lang et al. — "Debate Helps Weak-to-Strong Generalization" (arXiv:2501.13124)은 이들을 결합: 토론 프로토콜이 더 나은 약한 라벨을 제공하고, 강한 모델은 해당 라벨로 훈련됨. NLP 작업에서 PGR 향상 보고.

### 조직적 드라마

OpenAI의 슈퍼정렬(Superalignment) 팀은 2024년 5월 얀 라이크(Jan Leike)의 앤트로픽(Anthropic) 이적 후 해체. 의제(확장 가능한 감독, W2SG, 자동화된 정렬 연구)는 앤트로픽과 학술 연구실(MATS(레슨 28), 레드우드(Redwood, 레슨 10), 아폴로(Apollo, 레슨 8), METR(레슨 28))에서 계속됨. 조직 구조는 변경; 연구 질문은 그대로.

### 18단계에서의 위치

레슨 6-10은 U가 신뢰할 수 없다는 가정 하에 위협과 방어 패러다임을 설명. 레슨 11은 공격 패러다임: U의 정렬을 검증할 수 있을 만큼 감독자를 강하게 만듦. 레슨 12-16은 적대적 평가의 실용적 도구로 전환.

## 사용 방법

`code/main.py`는 합성 작업에서 W2SG 미세 조정(fine-tuning)을 시뮬레이션합니다. 약한 라벨러(weak labeler)는 70% 정확도에 구조화된 오류를 가지며, 강한 모델(strong model)은 골드 라벨(gold label)에서 95% 상한선을 가집니다. 약한 라벨로 강한 모델을 미세 조정하고, PGR(Progressive Gain Ratio)을 측정한 후, 골드 라벨 기반 강한 모델(strong-on-gold)과 약한 모델 단독(weak-alone) 성능과 비교합니다.

## Ship It

이 레슨은 `outputs/skill-w2sg-pgr.md`를 생성합니다. 감독 설정 설명이 주어졌을 때, 약한 감독자(weak supervisor), 강한 모델(strong model), 감독 품질(supervision quality)을 식별하고, PGR(Progressive Generalization Rate)을 계산(또는 요청)합니다. 또한 주장이 "약한 감독자가 강한 모델을 감독할 수 있다"인지 또는 "약한 감독자 + 감독 메커니즘이 강한 모델을 감독할 수 있다"인지 여부를 표시합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. weak_accuracy = 0.60, 0.70, 0.80에 대한 PGR을 보고하세요. PGR 곡선의 형태를 설명하세요.

2. 약한 라벨러(weak labeler)를 수정하여 구조화된 오류(예: 특정 입력 클래스에서 항상 오답)를 가지도록 만드세요. PGR은 증가, 감소, 또는 유지되나요? 설명하세요.

3. Burns et al. 2023 Section 4.3(NLP 작업)을 읽으세요. "신뢰도 보조 손실(confidence auxiliary loss)" 직관을 재현하세요: 강한 모델이 약한 라벨보다 더 확신하는 경우, 누가 승리하나요?

4. 소프트웨어 엔지니어링 작업을 위해 토론(debate)과 작업 분해(task decomposition)를 결합한 확장 가능한 감독 프로토콜을 설계하세요. 각 구성 요소의 실패 모드 하나를 명시하고, 이 결합이 각 실패 모드를 어떻게 해결하거나 해결하지 못하는지 설명하세요.

5. "약한-강한 일반화(weak-to-strong generalization)가 초정렬(superalignment)로 가는 실행 가능한 경로"라는 주장을 반증할 수 있는 조건을 명확히 하세요. 관찰해야 할 경험적 징후에 대해 구체적으로 설명하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 확장 가능한 감독(scalable oversight) | "감독자 강화" | 더 강력한 모델을 평가할 수 있는 감독자의 능력을 높이는 메커니즘 |
| W2SG(weak-to-strong gradient) | "약한 것이 강한 것을 감독" | 약한 레이블로 강력한 모델을 파인튜닝(fine-tuning)하고 복구된 능력을 측정 |
| PGR(performance gap recovered) | "성능 격차 회복" | (파인튜닝 - 약한) / (상한 - 약한); 1.0 = 완전히 회복, 0 = 도움 없음 |
| 토론(debate) | "두 U 인스턴스가 논쟁" | 약한 심판자가 두 U 방어자 중 선택하는 확장 가능한 감독 메커니즘 |
| RRM(recursive reward modeling) | "재귀적 보상 모델링" | U가 U+1의 보상 모델 훈련을 지원; 감독 능력은 U를 추적 |
| 작업 분해(task decomposition) | "인간이 확인하는 하위 작업" | 어려운 작업을 인간이 검증할 수 있는 하위 작업으로 분할, 재귀적으로 적용 |
| 초정렬(superalignment) | "초인적 AI 정렬" | 인간이 직접 평가할 수 없는 모델 정렬과 관련된 연구 계획 |

## 추가 자료

- [Burns et al. — 약-강 일반화(Weak-to-Strong Generalization, OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — W2SG 논문
- [Irving, Christiano, Amodei — 토론을 통한 AI 안전성(AI safety via debate, arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — 토론 메커니즘
- [Leike et al. — 보상 모델링을 통한 확장 가능한 에이전트 정렬(Scalable agent alignment via reward modeling, arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — 재귀적 보상 모델링
- [Khan et al. — 더 설득력 있는 LLM과의 토론이 더 진실된 답변으로 이어짐(Debating with More Persuasive LLMs Leads to More Truthful Answers, arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024년 강력한 토론자와의 토론 실증 연구
- [Lang et al. — 토론이 약-강 일반화를 돕는다는 연구(Debate Helps Weak-to-Strong Generalization, arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025년 토론 + W2SG 결합 연구