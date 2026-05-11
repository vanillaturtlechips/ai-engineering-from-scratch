# 에이전트 관측 가능성: Langfuse, Phoenix, Opik

> 2026년을 주도하는 세 가지 오픈소스 에이전트 관측 가능성 플랫폼. Langfuse (MIT) — 월 600만+ 설치, 트레이싱 + 프롬프트 관리 + 평가 + 세션 재생. Arize Phoenix (Elastic 2.0) — 심층 에이전트 특화 평가, RAG 관련성 분석, OpenInference 자동 계측. Comet Opik (Apache 2.0) — 자동화된 프롬프트 최적화, 가드레일, LLM-판단기 환각 탐지.

**유형:** 학습
**언어:** Python (표준 라이브러리)
**선수 조건:** 14단계 · 23 (OTel GenAI)
**소요 시간:** ~45분

## 학습 목표

- 상위 3개 오픈소스 에이전트 관측 가능성 플랫폼과 해당 라이선스를 명명할 수 있다.
- 각 플랫폼의 강점 분야를 구분할 수 있다: Langfuse (프롬프트 관리 + 세션), Phoenix (RAG + 자동 계측), Opik (최적화 + 가드레일).
- 2026년까지 89%의 조직이 에이전트 관측 가능성을 도입한 이유를 설명할 수 있다.
- LLM-판단자 평가를 포함한 stdlib 트레이스-대시보드 파이프라인을 구현할 수 있다.

## 문제

OTel GenAI(레슨 23)는 스키마를 제공합니다. 하지만 여전히 스팬(span)을 수집하고, 평가를 실행하며, 프롬프트 버전을 저장하고, 회귀(regression)를 표면화하는 플랫폼이 필요합니다. 세 가지 주요 후보들은 각각 라이프사이클의 다른 부분을 강조합니다.

## 개념

### Langfuse (MIT)

- 월간 600만+ SDK 설치, GitHub 스타 19k+.
- 기능: 트레이싱, 버전 관리 + 플레이그라운드가 포함된 프롬프트 관리, 평가(LLM-as-judge, 사용자 피드백, 커스텀), 세션 재생.
- 2025년 6월: 이전 상용 모듈(LLM-as-a-judge, 어노테이션 큐, 프롬프트 실험, 플레이그라운드)이 MIT 라이선스로 오픈 소스화됨.
- 강점: 타이트한 프롬프트 관리 루프와 함께 제공되는 엔드투엔드 관측 가능성.

### Arize Phoenix (Elastic License 2.0)

- 에이전트별 심층 평가: 트레이스 클러스터링, 이상 감지, RAG를 위한 검색 관련성.
- 네이티브 OpenInference 자동 계측.
- 프로덕션용 관리형 Arize AX와 연동.
- 프롬프트 버전 관리 없음 — 더 넓은 플랫폼과 함께 드리프트/행동 회귀 도구로 포지셔닝.
- 강점: RAG 관련성, 행동 드리프트, 이상 감지.

### Comet Opik (Apache 2.0)

- A/B 실험을 통한 자동화된 프롬프트 최적화.
- 가드레일(PII 삭제, 주제 제약).
- LLM-판사 기반 환각 감지.
- Comet 자체 측정 기준 벤치마크: Opik 로깅 + 평가 23.44초 vs Langfuse 327.15초(~14배 차이) — 벤더 벤치마크는 방향성만 참고.
- 강점: 최적화 루프, 자동화된 실험, 가드레일 강제 적용.

### 업계 데이터

Maxim(2026년 현장 분석)에 따르면: 89%의 조직이 에이전트 관측 가능성을 도입했으며, 품질 문제가 프로덕션의 최대 장벽(응답자 32%가 언급).

### 선택 기준

| 필요 사항 | 선택 |
|------|------|
| 프롬프트 관리가 포함된 올인원 | Langfuse |
| 심층 RAG 평가 + 드리프트 | Phoenix |
| 자동화된 최적화 + 가드레일 | Opik |
| 오픈 라이선스, ELv2 없음 | Langfuse (MIT) 또는 Opik (Apache 2.0) |
| Datadog / New Relic 통합 | 모두 — OTel을 모두 내보냄 |

### 이 패턴이 실패하는 경우

- **평가 전략 부재.** 평가 없는 트레이싱은 비용이 많이 드는 로깅에 불과.
- **근거 없는 자체 LLM-판사.** CRITIC 패턴(레슨 05) 적용 — 판사는 사실 검증을 위한 외부 도구 필요.
- **트레이스와 연결되지 않은 프롬프트 버전.** 프로덕션에서 회귀가 발생할 때, 원인을 제공한 프롬프트로 이분 탐색 불가.

## 빌드하기

`code/main.py`는 표준 라이브러리 추적 수집기 + LLM-판단자 평가기를 구현합니다:

- GenAI 형태의 스팬(span)을 수집합니다.
- 세션별로 그룹화하고, 실패한 실행(안전장치 트리거, 낮은 신뢰도 평가)을 태그합니다.
- 평가 기준(rubric)에 따라 에이전트 응답을 점수화하는 스크립트 기반 LLM-판단자.
- 대시보드 형식의 요약: 실패율, 주요 실패 원인, 평가 점수 분포.

실행 방법:

```
python3 code/main.py
```

출력: Langfuse/Phoenix/Opik에서 보여주는 것과 일치하는 세션별 평가 점수 및 실패 분류 결과.

## 사용 방법

- **Langfuse** 자체 호스팅 또는 클라우드; OTel 또는 자체 SDK로 연동.
- **Arize Phoenix** 자체 호스팅; OpenInference 자동 계측.
- **Comet Opik** 자체 호스팅 또는 클라우드; 자동화된 최적화 루프.
- **Datadog LLM Observability** Datadog을 이미 사용 중인 혼합 운영+ML 팀을 위한 솔루션.

## Ship It

`outputs/skill-obs-platform-wiring.md`는 플랫폼을 선택하고 트레이스(traces) + 평가(evals) + 프롬프트 버전(prompt versions)을 기존 에이전트에 연결합니다.

## 연습 문제

1. OTel 트레이스 1주 분량을 Langfuse 클라우드(무료 티어)로 내보내세요. 어떤 세션이 실패했나요? 이유는 무엇인가요?
2. 도메인별 LLM-판사 평가 기준(factual correctness, tone, scope adherence)을 작성하세요. 50개 트레이스로 테스트해 보세요.
3. Langfuse 프롬프트 버전 관리와 Phoenix의 트레이스 클러스터링을 비교하세요. 어떤 것이 더 빠르게 문제 원인을 알려주나요?
4. Opik의 가드레일 문서를 읽고, 에이전트 실행 중 하나에 PII 삭제 가드레일을 연결하세요.
5. 세 가지 방법을 자체 코퍼스로 벤치마킹하세요. 벤더 제공 수치는 무시하고 직접 측정하세요.  

> **번역 참고사항**  
> - "factual correctness" → "사실적 정확성" (도메인 용어 보존)  
> - "PII" → "개인 식별 정보(PII)" (약어 유지)  
> - "vendor-published numbers" → "벤더 제공 수치" (의미 중심 번역)

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 트레이싱(Tracing) | "스팬 수집기(Spans collector)" | OpenTelemetry(OTel) / SDK 스팬 수집; 세션별 인덱싱 |
| 프롬프트 관리(Prompt management) | "프롬프트 CMS" | 트레이스와 연결된 버전 관리 프롬프트 |
| LLM-as-judge | "자동 평가(Automated eval)" | 별도의 LLM이 평가 기준에 따라 에이전트 출력 점수 부여 |
| 세션 재생(Session replay) | "트레이스 재생(Trace playback)" | 디버깅을 위한 과거 실행 단계 추적 |
| RAG 관련성(RAG relevancy) | "검색 품질(Retrieval quality)" | 검색된 컨텍스트가 쿼리와 일치하는가 |
| 트레이스 클러스터링(Trace clustering) | "행동 그룹화(Behavioral grouping)" | 유사 실행 클러스터링으로 드리프트 감지 |
| 가드레일 강제(Guardrail enforcement) | "로그 시 정책(Policy at log time)" | 로그된 콘텐츠에 대한 PII/유독성/범위 검사 |

## 추가 자료

- [Langfuse 문서](https://langfuse.com/) — 트레이싱(tracing), 평가(evals), 프롬프트 관리(prompt mgmt)  
- [Arize Phoenix 문서](https://docs.arize.com/phoenix) — 자동 계측(auto-instrumentation), 드리프트(drift)  
- [Comet Opik](https://www.comet.com/site/products/opik/) — 최적화(optimization) + 가드레일(guardrails)  
- [OpenTelemetry GenAI 시맨틱 컨벤션](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 세 도구 모두 사용하는 스키마(schema)