# 정렬 연구 생태계 — MATS, 레드우드, 아폴로, METR

> 5개 조직이 2026년 비실험실 정렬 연구 계층을 정의합니다. MATS(ML 정렬 및 이론 학자): 2021년 말부터 527명 이상의 연구자, 180편 이상의 논문, 10,000회 이상의 인용, h-지수 47; 2024년 여름 코호트는 501(c)(3) 비영리 단체로 편입되었으며 약 90명의 학자와 40명의 멘토를 보유; 2025년 이전 졸업생의 80%는 안전/보안 분야에서 근무하며 200명 이상이 Anthropic, DeepMind, OpenAI, 영국 AISI, RAND, 레드우드, METR, 아폴로에서 활동 중입니다. 레드우드 리서치: 버크 슐레저리스가 설립한 응용 정렬 연구실; AI 제어(Lesson 10) 소개; 영국 AISI와 협력하여 제어 안전 사례 연구 진행. 아폴로 리서치: 프론티어 연구실을 위한 사전 배치 계획 평가; In-Context Scheming(Lesson 8) 및 AI 계획 안전 사례 연구(Towards Safety Cases for AI Scheming) 저술. METR(모델 평가 및 위협 연구): 작업 기반 능력 평가, 자율 작업 시간 범위 연구; "Common Elements of Frontier AI Safety Policies"는 연구실 프레임워크 비교. 엘레오스 AI 리서치: 모델 복지 사전 배치 평가(Lesson 19); Claude Opus 4 복지 평가 수행.

**유형:** 학습  
**언어:** 없음  
**선수 지식:** 18단계 · 01-27 (이전 18단계 레슨)  
**소요 시간:** ~45분

## 학습 목표

- 비-실험실 정렬 연구 생태계의 5개 조직과 각 조직의 핵심 산출물을 식별한다.
- MATS의 규모(연구자, 논문, h-지수)와 인재 파이프라인 역할을 설명한다.
- Redwood의 AI 제어 의제와 UK AISI와의 협력 관계를 설명한다.
- METR의 작업 기반 평가 방법론을 설명한다.

## 문제 정의

프론티어 랩(레슨 18)은 내부적으로 안전성 평가를 수행하고 선별된 결과를 공개합니다. 랩 외부의 생태계는 평가 결과가 검증되는 장소이며, 새로운 실패 모드가 처음 발견되는 곳이고, 인재가 양성되는 곳입니다. 생태계를 이해하면 어떤 연구 결과가 누구에게 신뢰받는지 해석하는 데 도움이 됩니다.

## 개념

### MATS (ML 정렬 및 이론 학자 프로그램)

2021년 말에 시작. 연구 멘토링 프로그램; 학자들은 10-12주 동안 선임 연구자와 함께 특정 정렬 문제를 연구합니다.

규모 (2026년 기준):
- 창립 이후 527명 이상의 연구자.
- 180편 이상의 논문 발표.
- 10,000회 이상의 인용.
- h-지수 47.
- 2024년 여름: 90명의 학자 + 40명의 멘토; 501(c)(3) 비영리 단체로 편입.

경력 결과: 2025년 이전 졸업생의 약 80%가 안전/보안 분야에서 근무 중. Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo 등에서 200명 이상 근무.

### Redwood Research

응용 정렬 연구실. Buck Shlegeris가 설립. AI 제어 어젠다(레슨 10) 도입. UK AISI와 협력하여 제어 안전 사례 연구 수행. DeepMind와 Anthropic에 평가 설계 자문 제공.

대표 논문: Greenblatt, Shlegeris 외, "AI 제어"(arXiv:2312.06942, ICML 2024); 정렬 위조(Greenblatt, Denison, Wright 외, arXiv:2412.14093, Anthropic과 공동).

스타일: 특정 위협 모델, 최악의 경우 적대적 시나리오, 스트레스 테스트 가능한 구체적 프로토콜.

### Apollo Research

프론티어 연구실을 위한 사전 배치 계획 평가 수행. "컨텍스트 내 계획"(레슨 8, arXiv:2412.04984) 저술. 2025년 OpenAI 반계획 훈련 협력 파트너. "AI 계획에 대한 안전 사례 연구"(2024) 제작.

스타일: 속임수가 발생할 수 있는 에이전트 설정 평가; 3가지 핵심 요소 분해(정렬 실패, 목표 지향성, 상황 인식).

### METR (모델 평가 및 위협 연구)

작업 기반 능력 평가. 자율 작업 완료 시간 범위 연구. "프론티어 AI 안전 정책의 공통 요소"(metr.org/common-elements, 2025)에서 연구실 프레임워크 비교.

Apollo와 공동으로 AI 계획 안전 사례 스케치 논문 공동 저술.

스타일: 장기 작업 평가, 경험적 능력 측정, 프레임워크 통합.

### Eleos AI Research

모델 복지 사전 배치 평가 수행. 시스템 카드 5.3절에 문서화된 Claude Opus 4 복지 평가 수행. 레슨 19의 복지 관련 주장에 대한 외부 방법론 검증 제공.

### 흐름

MATS는 연구자를 양성합니다. 졸업생은 Anthropic, DeepMind, OpenAI(연구실 안전 팀) 또는 Redwood, Apollo, METR, Eleos(외부 평가 기관)로 진출합니다. 외부 평가 기관은 연구실과 UK AISI/CAISI와 협력합니다. 출판물은 다음 기수를 위해 MATS로 다시 피드백됩니다.

### 이 계층이 중요한 이유

단일 출처 평가는 신뢰할 수 없음: 연구실이 자체 모델을 평가할 때는 구조적 이해 상충이 존재합니다. 외부 평가자는 연구실이 과소보고할 수 있는 실패 모드를 발견하고 검증할 수 있습니다. 2024년 "슬리퍼 에이전트" 논문(레슨 7)은 Anthropic + Redwood, "정렬 위조"는 Anthropic + Redwood, "컨텍스트 내 계획"은 Apollo, "반계획"은 Apollo + OpenAI가 수행했습니다. 다중 조직 구조가 품질 관리 역할을 합니다.

### 이 내용이 18단계에 어떻게 부합하는가

레슨 7-11은 Redwood와 Apollo 연구를 참조; 레슨 18은 METR의 프레임워크 비교를 참조; 레슨 19는 Eleos를 참조. 레슨 28은 나머지 18단계가 의존하는 생태계에 대한 명시적 조직 지도입니다.

## 활용 방법

코드 없음. METR의 "Common Elements of Frontier AI Safety Policies"를 외부 종합이 실험실 내부 정책 작업에 가치를 더하는 사례로 참고하시오.

## Ship It

이 레슨은 `outputs/skill-ecosystem-map.md`를 생성합니다. 정렬 주장(alignment claim) 또는 평가(evaluation)가 주어졌을 때, 해당 조직(organisation), 출판 장소(publication venue), 방법론 스타일(methodological style)을 식별하고, 알려진 대응 조직(known-counterpart organisation)과 상호 대조(cross-check)합니다.

## 연습 문제

1. 7-15강 중 한 논문을 선택하고 관련된 조직들을 식별하세요. 저자를 MATS 동문과 현재 생태계 소속과 대조해 확인하세요.

2. METR의 "프론티어 AI 안전 정책의 공통 요소"를 읽으세요. 그들이 강조하는 세 가지 연구실 간 수렴점과 두 가지 가장 큰 차이점을 식별하세요.

3. MATS 진로 결과는 ~80%가 안전/보안 분야입니다. 이 선택 압력이 적응적(분야를 훈련시킴)인지 아니면 편향적(이단적 입장을 걸러냄)인지 논증하세요.

4. Redwood와 Apollo는 모두 제어/계획(scheming) 작업을 하지만 다른 스타일을 가집니다. 실패 모드 하나를 선택하고 각 조직이 이를 어떻게 조사할지 설명하세요.

5. Eleos AI는 유일한 순수 모델-복지 조직입니다. 다른 복지 관련 질문(인지적 자유, 로봇 체화 등)에 초점을 맞춘 가상의 두 번째 조직을 설계하고 그 방법론을 명확히 하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| MATS | "멘토십 프로그램" | ML Alignment & Theory Scholars; 2021년 이후 527명 이상의 연구자 |
| Redwood Research | "제어 연구실" | 응용 정렬(applied alignment); AI Control 논문 저자; 영국 AISI 파트너 |
| Apollo Research | "계획 평가 연구실" | 프론티어 연구실을 위한 사전 배치 계획 평가(scheming evaluations) |
| METR | "작업-범위 평가" | 작업 기반 능력 평가(task-based capability evaluations); 프레임워크 통합 |
| Eleos AI | "복지 연구실" | 모델 복지 사전 배치 평가(model-welfare pre-deployment evaluations) |
| 인재 파이프라인 | "MATS → 연구실" | MATS 졸업생들이 Anthropic, DM, OpenAI, Redwood, Apollo, METR로 진출 |
| 외부 평가 | "비-연구실 검증" | 모델 제작자가 아닌 제3자가 수행하는 평가; 신뢰성 추가 |

## 추가 자료

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — 멘토링 프로그램
- [Redwood Research](https://www.redwoodresearch.org/) — AI 제어 논문
- [Apollo Research](https://www.apolloresearch.ai/) — 평가 체계 연구
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — 프레임워크 비교
- [Eleos AI Research](https://www.eleosai.org/research) — 모델 복지 방법론