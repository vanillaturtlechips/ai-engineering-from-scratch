# HTN과 진화 탐색을 활용한 계획 수립

> 심볼릭 계획 수립은 계획이 증명 가능하게 정확한 경우를 처리합니다. 진화 코드 탐색은 적합도 함수가 기계 검증 가능한 경우를 처리합니다. ChatHTN(2025)과 AlphaEvolve(2025)는 각각이 LLM과 결합될 때 어떤 가능성을 열어주는지 보여줍니다.

**유형:** 구축(Build)
**언어:** Python (표준 라이브러리)
**선수 지식:** 14단계 · 02 (ReWOO 및 계획-실행)
**소요 시간:** ~75분

## 학습 목표

- 계층적 작업 네트워크(Hierarchical Task Networks, HTN) 설명: 작업(tasks), 방법(methods), 연산자(operators), 선행 조건(preconditions), 효과(effects).
- ChatHTN의 하이브리드 루프 설명 — 기호적 검색(symbolic search)과 LLM 폴백 분해(LLM fallback decomposition).
- AlphaEvolve의 진화론적 루프(evolutionary loop) 설명 및 프로그램적 평가자(programmatic evaluator)와만 작동하는 이유.
- 표준 라이브러리(stdlib)를 이용한 장난감 HTN 플래너(toy HTN planner) 및 장난감 진화 검색(toy evolutionary search) 구현.

## 문제 정의

ReWOO(레슨 02), Plan-and-Execute, ReAct는 대부분의 에이전트 계획 방식을 다룹니다. 하지만 다음 두 가지 경우는 잘 처리하지 못합니다:

1. **증명 가능한 정확성을 가진 계획.** 스케줄링, 비행 경로 계획, 규정 준수 워크플로우 — 계획은 구조적으로 타당해야 합니다. 때때로 단계를 환각(hallucination)하는 유창한 LLM 계획은 허용되지 않습니다.
2. **기계 검증 가능한 적합도 함수를 가진 최적화.** 행렬 곱셈, 스케줄링 휴리스틱, 컴파일러 패스 — 목표는 "올바른 계획"이 아니라 "최적의 계획"입니다.

HTN 계획과 AlphaEvolve는 서로 다른 두 문제를 해결합니다. 둘 다 LLM을 대체재가 아닌 증폭기로 사용합니다.

## 개념

### 계층적 작업 네트워크(HTN)

HTN은 다음과 같이 구성됩니다:

- **작업(Tasks)** — 복합 작업(분해 대상) 및 원시 작업(직접 실행 가능).
- **메서드(Methods)** — 전제 조건을 갖춘 복합 작업을 하위 작업으로 분해하는 방법.
- **연산자(Operators)** — 전제 조건과 효과를 가진 원시 동작.
- **상태(State)** — 사실(facts)의 집합.

계획 수립: 목표 작업과 초기 상태가 주어졌을 때, 순차적으로 전제 조건이 충족되는 원시 연산자들로의 분해 경로를 찾습니다.

HTN은 LLM보다 오래된 기술이며, 여전히 증명 가능한 정확한 계획을 위한 기준입니다.

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN (arXiv:2505.11814)은 기호적 HTN과 LLM 쿼리를 교차 실행합니다:

1. 기존 메서드로 현재 복합 작업을 분해 시도.
2. 적용 가능한 메서드가 없으면 LLM에 질문: "상태 `s`에서 `작업`을 어떻게 분해하겠는가?"
3. LLM 응답을 후보 하위 작업으로 변환.
4. 연산자 스키마와 대조하여 검증; 유효하지 않은 분해 거부.
5. 재귀 실행.

논문의 주요 주장: 생성된 모든 계획은 LLM 제안이 후보 분해로만 입력되고 직접적인 계획 수정으로는 사용되지 않기 때문에 증명 가능한 정확성을 가집니다. 기호 계층이 정확성을 소유하며, LLM은 메서드 라이브러리를 확장합니다.

온라인 메서드 학습(OpenReview `gwYEDY9j2x`, 2025 후속 연구)은 회귀를 통해 LLM 생성 분해를 일반화하는 학습자를 추가하여 LLM 쿼리 빈도를 최대 75%까지 줄입니다.

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, 2025년 6월)는 다른 접근 방식입니다: Gemini 2.0 Flash/Pro 앙상블이 조율하는 진화적 코드 검색.

루프:

1. 시드 프로그램 + 프로그램적 평가자(적합도 점수 반환)로 시작.
2. LLM 앙상블이 변이(mutation) 제안.
3. 변이들을 평가자에 통과시킴.
4. 최고 성능 유지; 다시 변이 생성.

발표된 성과:

- 56년 만에 4x4 복소수 행렬 곱셈에서 Strassen을 능가하는 첫 개선(48개의 스칼라 곱셈).
- Borg 스케줄링 휴리스틱을 통해 Google 컴퓨팅 자원 0.7% 회수.
- 프론티어 워크로드에서 FlashAttention 속도 32% 향상.

하드 제약: 적합도 함수는 기계로 검증 가능해야 합니다. 산문형 답변에 대한 진화적 검색은 수렴하지 않습니다.

### 어떤 것을 사용할 것인가

| 문제 유형 | 사용 기술 | 이유 |
|---------------|-----|-----|
| 하드 제약이 있는 스케줄링 | HTN + ChatHTN | 증명 가능한 정확성 |
| 컴파일러 최적화 | AlphaEvolve | 기계로 검증 가능한 적합도 |
| 다단계 작업 실행 | ReAct / ReWOO | LLM이 루프에 참여, 형식적 보장 없음 |
| 테스트가 있는 코드 개선 | AlphaEvolve | 테스트가 평가자 역할 |
| 정책 기반 자동화 | HTN | 전제 조건이 정책 인코딩 |

### 이 패턴이 실패하는 경우

- **연산자가 없는 HTN.** 전제 조건/효과 스키마가 없으면 정확성 주장이 무너집니다. ChatHTN의 "LLM이 분해 제안"은 스키마가 무효한 이동을 거부해야 합니다.
- **실제 평가자가 없는 AlphaEvolve.** "LLM에 코드가 더 나은지 물어보기"는 적합도 함수가 아닙니다. 평가자는 결정론적이고 빨라야 합니다.
- **과도한 설계.** 대부분의 에이전트 작업은 둘 다 필요 없습니다. 먼저 ReAct 또는 ReWOO를 고려하세요.

## 빌드하기

`code/main.py`는 다음 두 가지 토이(toy)를 구현합니다:

- 연산자(operators), 방법(methods), 선행 조건(preconditions), 효과(effects)를 갖춘 표준 라이브러리 HTN 플래너와, 복합 작업(compound task)에 일치하는 방법이 없을 때 작동하는 `LLMFallback`(LLM 대체 전략). "LLM"은 스크립트된 분해기(decomposer)로, 플래너가 오프라인에서 실행됩니다.
- 산술 프로그램(arithmetic programs)에 대한 표준 라이브러리 진화 탐색(evolutionary search): 테스트 세트에서 출력값이 `|f(x) - target|`을 최소화하는 표현식을 성장시킵니다. 평가기(evaluator)는 결정론적입니다(deterministic).

실행 방법:

```
python3 code/main.py
```

트레이스(trace)는 HTN 플래너가 복합 작업을 분해(중간에 LLM 대체 전략 사용)하는 과정과 진화 루프가 목표 표현식에 수렴하는 과정을 보여줍니다.

## 사용 방법

- **HTN 플래너** — `pyhop`, `SHOP3` 또는 도메인별 정책 강화를 위해 직접 구축.
- **ChatHTN** — 연구용 코드; (기호적 + LLM 폴백) 패턴은 모든 HTN 플래너에 깔끔하게 이식 가능.
- **AlphaEvolve** — DeepMind 논문; (앙상블 + 평가기) 패턴은 재현 가능. OpenEvolve 및 유사 오픈소스 포크가 등장 중.
- **에이전트 프레임워크** — 아직 HTN 또는 AlphaEvolve를 기본 제공하지 않음. 서브에이전트 또는 백그라운드 작업자로 구축.

## Ship It

`outputs/skill-hybrid-planner.md`는 LLM 역할이 명시적으로 범위가 지정된 하이브리드 플래너(HTN 또는 진화형) 스캐폴드를 생성합니다.

## 연습 문제

1. HTN 플래너에 백트래킹 기능 추가: 런타임에 연산자의 사후 조건이 실패할 경우, 롤백하고 다음 방법을 시도하도록 확장.
2. ChatHTN에 LLM-메서드 캐시 추가: LLM이 상태 패턴 `P`에서 작업 `T`를 분해할 때 결과를 저장. 다음 호출 시 먼저 메서드 라이브러리를 재확인.
3. 진화 탐색 평가기를 실제 테스트 스위트로 교체: 20개 테스트 케이스를 통과하는 정렬 함수 진화; 수렴까지 세대 수 보고.
4. AlphaEvolve의 평가기 설계 노트 읽기. 관심 있는 도메인(SQL 쿼리 최적화, 테스트-스위트 최소화, 배포 YAML)에 대한 평가기 설계.
5. 통합: HTN을 사용해 복합 작업을 하위 작업으로 분해한 후, 각 하위 작업의 원시 연산자에 대해 진화 탐색 적용. 어디서 빛을 발하고, 어디서 과도한 엔지니어링이 발생하는가?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| HTN | "계층적 계획기" | 작업 분해(operators, 사전 조건, 효과 포함) |
| Method | "분해 규칙" | 복합 작업을 하위 작업으로 분할하는 방법 |
| Operator | "기본 동작" | 사전 조건과 효과를 가진 구체적 단계 |
| ChatHTN | "LLM + HTN" | 심볼릭 계획기가 일치하는 방법이 없을 때 LLM에 요청 |
| AlphaEvolve | "진화적 코드 검색" | 앙상블 LLM이 코드 변형; 결정적 평가기가 선택 |
| Fitness function | "평가기" | 출력물에 대한 결정적, 기계 검증 가능 점수 |
| Online method learning | "캐시된 LLM 분해" | LLM 계획 저장 및 일반화하여 쿼리 비용 절감 |

## 추가 자료

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — 심볼릭(symbolic) + LLM 하이브리드 플래너(planner)  
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — LLM 변이(mutations)를 활용한 진화형 코드 검색(evolutionary code search)  
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 플래너(planner) vs 단순 루프(simple loop) 선택 기준