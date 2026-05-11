# Agno와 Mastra: 프로덕션 런타임

> Agno(Python)와 Mastra(TypeScript)는 2026년 프로덕션 런타임 쌍입니다. Agno는 마이크로초 단위의 에이전트 인스턴스화와 상태 없는 FastAPI 백엔드를 목표로 합니다. Mastra는 에이전트, 도구, 워크플로우, 통합 모델 라우팅, 복합 스토리지를 Vercel AI SDK 기반에서 제공합니다.

**유형:** 학습  
**언어:** Python, TypeScript  
**사전 요구 사항:** Phase 14 · 01 (에이전트 루프), Phase 14 · 13 (LangGraph)  
**소요 시간:** ~45분

## 학습 목표

- Agno의 성능 목표와 해당 목표가 중요한 시점을 식별한다.
- Mastra의 세 가지 기본 구성 요소인 **에이전트(Agents)**, **툴(Tools)**, **워크플로우(Workflows)**와 지원되는 서버 어댑터를 명명한다.
- 무상태 세션 범위 FastAPI 백엔드가 Agno 프로덕션 환경에서 권장되는 이유를 설명한다.
- 주어진 스택(파이썬 우선 vs 타입스크립트 우선)에 따라 Agno와 Mastra 중 적절한 것을 선택한다.

## 문제

LangGraph, AutoGen, CrewAI는 프레임워크에 크게 의존합니다. "에이전트 루프만 빠르게 내 런타임에서 실행하고 싶다"는 팀은 Agno(Python) 또는 Mastra(TypeScript)를 선택합니다. 두 도구 모두 프레임워크가 제공하는 일부 기본 요소를 희생하는 대신 원시 속도와 주변 스택과의 더 긴밀한 통합을 추구합니다.

## 개념

### Agno

- Python 런타임, 이전 명칭 Phi-data.
- "그래프, 체인, 복잡한 패턴 없이 — 순수한 파이썬만."
- 문서 기준 성능 목표: ~2μs 에이전트 인스턴스화, ~3.75 KiB 에이전트당 메모리, ~23 모델 제공자.
- 프로덕션 경로: 상태 없는 세션 범위 FastAPI 백엔드. 각 요청은 새 에이전트를 시작하며, 세션 상태는 DB에 저장.
- 네이티브 멀티모달(텍스트, 이미지, 오디오, 비디오, 파일) 및 에이전트 RAG 지원.

속도 목표는 초당 수천 개의 단명 에이전트(채팅 팬인, 평가 파이프라인)가 있을 때 중요. 10분 동안 실행되는 단일 에이전트일 때는 덜 중요.

### Mastra

- TypeScript, Vercel AI SDK 기반.
- 세 가지 기본 요소: **에이전트**, **툴**(Zod 타입), **워크플로우**.
- 통합 모델 라우터 — 94개 제공자에 걸친 3,300+ 모델(2026년 3월 기준).
- 복합 저장소: 메모리, 워크플로우, 관측 가능성을 다양한 백엔드에 연결; 대규모 관측 가능성에는 ClickHouse 권장.
- Apache 2.0 라이선스, 단 `ee/` 디렉터리는 소스 공개 엔터프라이즈 라이선스 적용.
- Express, Hono, Fastify, Koa용 서버 어댑터; Next.js 및 Astro 통합 지원.
- 디버깅용 Mastra Studio(localhost:4111) 제공.
- 22k+ GitHub 스타, 1.0 버전(2026년 1월) 기준 주간 npm 다운로드 300k+.

### 포지셔닝

둘 다 LangGraph가 되려 하지 않음. 경쟁 요소는 다음과 같음:

- **언어 적합성.** Agno는 Python 우선 팀용; Mastra는 TypeScript 우선 팀용.
- **런타임 편의성.** Agno = 거의 제로 오버헤드; Mastra = Vercel 생태계와 통합.
- **관측 가능성.** 둘 다 Langfuse/Phoenix/Opik(레슨 24)과 통합되지만, Mastra Studio는 자체 제공.

### 각 도구 선택 시기

- **Agno** — Python 백엔드, 많은 단명 에이전트, 강력한 성능 요구사항, FastAPI 사용 시.
- **Mastra** — TypeScript 백엔드, Next.js / Vercel 배포, 통합 멀티 제공자 모델 라우팅, Zod 타입 툴.
- **LangGraph**(레슨 13) — 지속 상태와 명시적 그래프 추론이 순수 속도보다 중요할 때.
- **OpenAI / Claude 에이전트 SDK** — 제공자의 제품화된 형태를 원할 때(레슨 16–17).

### 이 패턴이 실패하는 경우

- **성능을 위한 성능.** 요청당 하나의 느린 에이전트 호출만 있는 워크로드에서 "2μs"라는 수치만 보고 Agno를 선택. 오버헤드는 병목 지점이 아님.
- **생태계 종속.** Mastra의 Vercel 특화 통합은 Vercel에서는 장점이지만, 다른 곳에서는 단점.
- **엔터프라이즈 라이선스 혼동.** Mastra의 `ee/` 디렉터리는 소스 공개 상태이며, Apache 2.0 라이선스가 아님. 포크 계획이 있다면 라이선스 확인 필수.

## 구축 방법

이 레슨은 주로 비교에 초점을 맞추고 있습니다. 단일 코드 아티팩트로는 두 프레임워크에 대한 공정한 비교가 어렵습니다. `code/main.py`를 참조하면 나란히 비교할 수 있는 간단한 예제가 있습니다. 이 예제는 "에이전트 실행, 출력 스트리밍, 세션 저장"이라는 최소한의 흐름을 Agno와 Mastra 두 가지 방식으로 각각 구현한 것입니다.

실행 방법:

```
python3 code/main.py
```

구조적으로 다르지만 기능적으로는 동등한 두 가지 실행 결과를 확인할 수 있습니다.

## 사용 방법

- **Agno** — 속도와 FastAPI 형태가 필요한 Python 백엔드.
- **Mastra** — 많은 프로바이더와 워크플로우 기본 요소를 갖춘 TypeScript 백엔드.
- 둘 다 1급 관측 가능성(observability) 훅을 제공합니다. 둘 다 Langfuse와 통합됩니다.

## Ship It

`outputs/skill-runtime-picker.md`는 스택, 지연 시간 예산, 운영 형태에 따라 Agno, Mastra, LangGraph 또는 공급자 SDK를 선택합니다.

## 연습 문제

1. Agno 문서를 읽으세요. stdlib ReAct 루프(레슨 01)를 Agno로 이식하세요. 무엇이 사라졌나요? 무엇이 유지되었나요?
2. Mastra 문서를 읽으세요. 동일한 루프를 Mastra로 이식하세요. 도구 타이핑에서 무엇이 변경되었나요(Zod vs 없음)?
3. 벤치마크: 사용 중인 스택에서 에이전트 인스턴스화 지연 시간을 측정하세요. Agno의 2μs가 작업 부하에 중요한가요?
4. 마이그레이션 설계: Python에서 CrewAI를 실행 중이었다면, Agno로 이동할 때 무엇이 깨지나요?
5. Mastra의 `ee/` 라이선스 조건을 읽으세요. 오픈 소스 포크에 영향을 미치는 제한 사항은 무엇인가요?

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Agno | "빠른 Python 에이전트" | 세션 범위의 상태 없는 에이전트 런타임 |
| Mastra | "Vercel AI SDK의 TypeScript 에이전트" | 에이전트 + 도구 + 워크플로우 + 모델 라우터 |
| 통합 모델 라우터 | "멀티 프로바이더 접근" | 94개 프로바이더에 걸친 3,300+ 모델을 위한 단일 클라이언트 |
| 복합 저장소 | "여러 백엔드" | 메모리/워크플로우/관측 가능성을 각각 다른 저장소에 저장 |
| Mastra Studio | "로컬 디버거" | 에이전트 분석을 위한 localhost:4111 UI |
| 소스 사용 가능 | "OSS가 아님" | 라이선스는 소스 코드 읽기를 허용하지만 상업적 사용은 제한 |

## 추가 자료

- [Agno Agent Framework 문서](https://www.agno.com/agent-framework) — 성능 목표, FastAPI 통합
- [Mastra 문서](https://mastra.ai/docs) — 기본 요소, 서버 어댑터, 모델 라우터
- [LangGraph 개요](https://docs.langchain.com/oss/python/langgraph/overview) — 상태 기반 그래프 대안
- [Comet Opik](https://www.comet.com/site/products/opik/) — Mastra 통합에서 인용한 관측 가능성 비교