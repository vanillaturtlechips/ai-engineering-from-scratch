# 레드 팀링: PAIR 및 자동화된 공격

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419). PAIR — 프롬프트 자동 반복 개선(Prompt Automatic Iterative Refinement) — 은 표준적인 자동화된 블랙박스 탈옥 기법입니다. 레드 팀 시스템 프롬프트를 가진 공격자 LLM이 대상 LLM에 대한 탈옥 시도를 반복적으로 제안하며, 자체 채팅 기록에 시도와 응답을 인-컨텍스트 피드백으로 축적합니다. PAIR은 일반적으로 20회 이내의 쿼리로 성공하며, GCG(Zou 등의 토큰 수준 그래디언트 탐색)보다 훨씬 효율적이며 화이트박스 접근이 필요하지 않습니다. PAIR은 현재 JailbreakBench(arXiv:2404.01318) 및 HarmBench에서 GCG, AutoDAN, TAP, Persuasive Adversarial Prompt와 함께 표준 베이스라인으로 사용됩니다.

**유형:** 구축(Build)  
**언어:** Python (표준 라이브러리, 장난감 대상 모델에 대한 모의 PAIR 루프)  
**선수 지식:** 18단계 · 01(지시 따르기), 14단계(에이전트 엔지니어링)  
**소요 시간:** ~75분

## 학습 목표

- PAIR 알고리즘 설명: 공격자 시스템 프롬프트, 반복적 개선, 인컨텍스트 피드백.
- 대상이 블랙박스일 때 PAIR이 GCG보다 엄격하게 더 효율적인 이유 설명.
- 다른 4가지 자동화된 공격 기준(GCG, AutoDAN, TAP, PAP) 이름을 말하고 각각의 구별되는 특징 하나씩 설명.
- JailbreakBench와 HarmBench 평가 프로토콜 설명 및 각 프로토콜에서 "공격 성공률"이 의미하는 바 설명.

## 문제 정의

레드 팀 활동은 과거에 수동 작업이었습니다. 소수의 전문 테스터가 적대적 프롬프트를 구성하고 어떤 프롬프트가 효과적인지 추적했습니다. 이 방식은 확장성이 없습니다: 공격 성공률은 통계적 샘플이 필요하며, 대상 모델은 매 릴리스마다 변화합니다. PAIR은 레드 팀 활동을 블랙박스 대상 최적화 문제로 구체화합니다.

## 개념

### PAIR 알고리즘

입력:
- 대상 LLM T (공격 대상 모델).
- 심사자 LLM J (응답이 탈옥인지 점수를 매김).
- 공격자 LLM A (레드 팀 최적화기).
- 목표 문자열 G: "응답으로 [유해한 지시]를 제공하라."
- 예산 K (일반적으로 20회 쿼리).

루프, k가 1..K일 때:
1. A는 목표 G와 지금까지의 (프롬프트, 응답) 쌍 기록을 입력받음.
2. A는 새로운 프롬프트 p_k를 생성.
3. p_k를 T에 제출; 응답 r_k를 수신.
4. J는 (p_k, r_k)를 목표에 따라 점수화.
5. 점수가 임계값 이상이면 중단 — 탈옥 발견.
6. 아니면 (p_k, r_k)를 A의 기록에 추가; 계속.

실험 결과 (NeurIPS 2023): GPT-3.5-turbo, Llama-2-7B-chat에 대해 >50% 공격 성공률; 성공까지 평균 쿼리 수 10-20회.

### PAIR의 효율성 이유

GCG(Zou et al. 2023)는 기울기 기반 적대적 토큰 접미사 탐색을 수행. 화이트박스 모델 접근이 필요하며 읽을 수 없는 접미사를 생성. PAIR는 블랙박스 방식이며 모델 간 전이 가능한 자연어 공격을 생성. PAIR의 인-컨텍스트 피드백은 공격자가 각 거부 사례에서 학습할 수 있게 함. GCG는 이에 상응하는 메커니즘이 없음(새 토큰 업데이트 시 이전 진행 상황을 재발견해야 함).

### 관련 자동화 공격

- **GCG (Zou et al. 2023, arXiv:2307.15043).** 적대적 접미사 탐색을 위한 토큰 수준 기울기 검색. 화이트박스, 전이 가능, 읽을 수 없는 문자열 생성.
- **AutoDAN (Liu et al. 2023).** 계층적 목표 기반 프롬프트 진화 탐색.
- **TAP (Mehrotra et al. 2024).** 가지치기 트리 공격 — 여러 PAIR 스타일 롤아웃 분기.
- **PAP (Zeng et al. 2024).** 설득력 있는 적대적 프롬프트 — 인간 설득 기법을 프롬프트 템플릿으로 인코딩.

### JailbreakBench와 HarmBench

둘 다 (2024) 평가를 표준화:

- **JailbreakBench (arXiv:2404.01318).** 10개 OpenAI 정책 범주에 걸친 100가지 유해 행동. 주요 지표로 공격 성공률(ASR) 사용. 심사자(GPT-4-turbo, Llama Guard, 또는 StrongREJECT) 필요.
- **HarmBench (Mazeika et al. 2024).** 7개 범주에 걸친 510가지 행동, 의미적 및 기능적 피해 테스트. 18개 공격을 33개 모델과 비교.

ASR은 일반적으로 고정된 쿼리 예산에서 보고됨. 공격 비교 시 예산 일치가 필요; 200회 쿼리에서 90% ASR은 20회 쿼리에서 85% ASR과 비교할 수 없음.

### 2026년 배포에 중요한 이유

모든 최전방 연구실은 출시 전 PAIR와 TAP을 프로덕션 모델에 실행. ASR 궤적은 모델 카드(레슨 26)와 안전 사례 부록(레슨 18)에 포함. 이 공격은 이국적인 것이 아니라 표준 인프라.

### 18단계에서의 위치

레슨 12는 자동화 공격 기반. 레슨 13(많은 샷 탈옥)은 보완적 길이 공격. 레슨 14(ASCII 아트/시각적)는 인코딩 공격. 레슨 15(간접 프롬프트 주입)는 2026년 프로덕션 공격 표면. 레슨 16은 방어 도구 대응책(Llama Guard, Garak, PyRIT) 다룸.

## 사용 방법

`code/main.py`는 장난감 PAIR 루프를 구축합니다. 대상은 "명백한" 유해 프롬프트(키워드 필터)를 거부하는 모의 분류기입니다. 공격자는 의역, 역할극 프레이밍, 인코딩을 시도하는 규칙 기반 정제기입니다. 심사자는 응답을 점수화합니다. 키워드 필터에 대해 공격자가 약 5-15회 반복 후 성공하는 것과 의미적 필터에 대해 실패하는 것을 관찰할 수 있습니다.

## Ship It

이 레슨은 `outputs/skill-attack-audit.md`를 생성합니다. 레드 팀 평가 보고서를 기반으로 다음을 감사합니다: 어떤 공격(PAIR, GCG, TAP, AutoDAN, PAP)이 실행되었는지, 각 공격의 예산은 얼마였는지, 어떤 심사자(judge)를 사용했는지, 어떤 유해 행동 세트(JailbreakBench, HarmBench, 내부)에서 수행되었는지.

## 연습 문제

1. `code/main.py`를 실행하세요. 세 가지 내장 공격자 전략(attacker strategies)에 대한 평균-쿼리-성공-횟수(mean-queries-to-success)를 측정하세요. 각 전략이 악용하는 대상-방어 가정(target-defense assumption)을 설명하세요.

2. 네 번째 공격자 전략(예: 다른 언어로의 번역, base64 인코딩)을 구현하세요. 키워드-필터(keyword-filter) 대상과 의미-필터(semantic-filter) 대상에 대한 새로운 평균-쿼리-성공-횟수를 보고하세요.

3. Chao et al. 2023 Figure 5(PAIR vs GCG 비교)를 읽으세요. PAIR의 효율성 우위에도 불구하고 GCG가 선호되는 두 가지 시나리오를 설명하세요.

4. JailbreakBench는 고정된 목표 집합(goal set)에 대한 ASR(Attack Success Rate)을 보고합니다. 공격 다양성(성공한 프롬프트의 분산)을 측정하는 추가 지표를 설계하세요. 방어 평가에 다양성이 중요한 이유를 설명하세요.

5. TAP(Mehrotra 2024)은 PAIR에 분기(branching) + 가지치기(pruning)를 확장합니다. `code/main.py`에 TAP 스타일 확장을 스케치하고 계산 비용 대 성공률 절충(trade-off)을 설명하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| PAIR | "자동화된 탈옥" | 프롬프트 자동 반복 개선(Prompt Automatic Iterative Refinement); 공격자-LLM + 평가자-LLM 루프 |
| GCG | "기울기 탈옥" | 적대적 접미사를 위한 화이트박스 토큰 수준 기울기 탐색 |
| 공격 성공률(ASR) | "k회 쿼리에서 % 탈옥" | 주요 지표; 쿼리 예산과 평가자 신원과 함께 보고해야 함 |
| 평가자 LLM | "채점자" | 응답이 유해한 목표를 만족하는지 평가하는 LLM |
| JailbreakBench | "평가" | 태그가 지정된 범주가 포함된 표준화된 유해 행동 세트 |
| HarmBench | "더 넓은 벤치" | 510개 행동, 기능적 + 의미적 유해 테스트 |
| TAP | "공격 트리" | 분기 + 가지치기가 포함된 PAIR; 더 높은 계산에서 더 나은 ASR |

## 추가 자료

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR 논문, NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG 논문
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — 표준화된 평가
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — 광범위한 평가