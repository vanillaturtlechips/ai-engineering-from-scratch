# 캡스톤 — 완전한 도구 생태계 구축하기

> 13단계에서는 모든 구성 요소를 가르쳤습니다. 이 캡스톤 프로젝트에서는 MCP 서버, 도구 + 리소스 + 프롬프트 + 작업 + UI, 엣지에서의 OAuth 2.1, RBAC 게이트웨이, 멀티 서버 클라이언트, A2A 서브 에이전트 호출, OTel 트레이싱을 통한 수집기 연동, CI 환경에서의 도구 오염 감지, AGENTS.md + SKILL.md 번들을 하나의 프로덕션 형태의 시스템으로 통합합니다. 프로젝트 완료 시 모든 아키텍처 설계 선택을 방어할 수 있게 됩니다.

**유형:** 구축
**언어:** Python (표준 라이브러리, 엔드투엔드 생태계 하니스)
**선수 조건:** 13단계 · 01부터 21까지
**소요 시간:** ~120분

## 학습 목표

- `ui://` 앱과 함께 도구, 리소스, 프롬프트, 작업을 노출하는 MCP 서버를 구성합니다.
- RBAC(Role-Based Access Control)와 고정된 해시(pinned hashes)를 강제하는 OAuth 2.1 게이트웨이로 서버를 프론팅합니다.
- OTel GenAI 속성을 종단간(end-to-end)으로 추적하는 멀티 서버 클라이언트를 작성합니다.
- A2A(agent-to-agent) 서브 에이전트에게 작업 부하의 일부를 위임하고, 불투명성(opacity) 보존을 검증합니다.
- 다른 에이전트가 구동할 수 있도록 AGENTS.md + SKILL.md로 전체 스택을 패키징합니다.

## 문제 정의

"연구 및 보고서" 시스템 구축:

- 사용자 요청: "에이전트 프로토콜에 관한 2026년 가장 많이 인용된 arXiv 논문 3편을 요약해 주세요."
- 시스템 동작: MCP를 통해 arXiv 검색 → A2A를 통해 전문 작성 에이전트에게 논문 요약 위임 → 결과 통합 → MCP Apps `ui://` 리소스로 대화형 보고서 렌더링 → 모든 단계를 OTel에 로깅

Phase 13의 모든 기본 요소들이 등장합니다. 이는 단순한 데모가 아닙니다. 2026년 Anthropic(Claude Research 제품), OpenAI(Apps SDK를 갖춘 GPTs) 및 타사에서 출시한 실제 연구 보조 시스템들이 바로 이 구조를 가지고 있습니다.

## 개념

### 아키텍처

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP 도구: arxiv_search (pure)
                                                      +- MCP 리소스: notes://recent
                                                      +- MCP 프롬프트: /research_topic
                                                      +- MCP 작업: generate_report (long)
                                                      +- MCP 앱 UI: ui://report/current
                                                      +- A2A 호출: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### 트레이스 계층 구조

```
agent.invoke_agent
 ├── llm.chat (시작)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── 작업 전환 (불투명 내부)
 ├── mcp.call -> tools/call generate_report (작업 증강)
 │    └── 작업/상태 폴링
 │    └── 작업/결과 (완료, ui:// 리소스 반환)
 └── llm.chat (최종 합성)
```

하나의 트레이스 ID. 모든 스팬은 올바른 `gen_ai.*` 속성을 가짐.

### 보안 체계

- OAuth 2.1 + PKCE로 리소스 지시자를 게이트웨이에 고정.
- 게이트웨이가 업스트림 자격 증명을 보유; 사용자는 이를 볼 수 없음.
- RBAC: `alice`는 `research:read`, `research:write` 권한을 가지며 모든 도구를 호출할 수 있음. `bob`은 `research:read` 권한만 가지며 `generate_report`를 호출할 수 없음.
- 고정된 설명 매니페스트: 도구 해시가 변경된 서버는 모두 제거.
- Rule of Two 감사: 어떤 도구도 신뢰할 수 없는 입력, 민감한 데이터, 중대한 작업을 결합하지 않음.

### 렌더링

최종 `generate_report` 작업은 콘텐츠 블록과 `ui://report/current` 리소스를 반환. 클라이언트의 호스트(Claude Desktop 등)는 샌드박스 iframe에서 대화형 대시보드를 렌더링. 대시보드에는 정렬된 논문 목록, 인용 횟수, 사용자가 클릭한 논문에 대해 `host.callTool('summarize_paper', {arxiv_id})`를 호출하는 버튼이 포함됨.

### 패키징

전체 시스템은 다음과 같이 제공됨:

```
research-system/
  AGENTS.md                     # 프로젝트 규칙
  skills/
    run-research/
      SKILL.md                  # 최상위 워크플로우
  servers/
    research-mcp/               # MCP 서버
      pyproject.toml
      src/
  agents/
    writer/                     # A2A 에이전트
  gateway/
    config.yaml                 # RBAC + 고정된 매니페스트
```

사용자는 `docker compose up`으로 배포. Claude Code, Cursor, Codex, opencode 사용자는 `run-research` 스킬을 호출하여 시스템을 구동할 수 있음.

### 각 13단계 레슨이 기여한 내용

| 레슨 | 캡스톤에서 사용하는 내용 |
|--------|------------------------|
| 01-05 | 도구 인터페이스, 공급자 이식성, 병렬 호출, 스키마, 린팅 |
| 06-10 | MCP 기본 요소, 서버, 클라이언트, 전송 계층, 리소스 + 프롬프트 |
| 11-14 | 샘플링, 루트 + 유도, 비동기 작업, `ui://` 앱 |
| 15-17 | 도구 오염 방지, OAuth 2.1, 게이트웨이 + 레지스트리 |
| 18 | A2A 하위 에이전트 위임 |
| 19 | OTel GenAI 트레이싱 |
| 20 | LLM 계층을 위한 라우팅 게이트웨이 |
| 21 | SKILL.md + AGENTS.md 패키징 |

## 사용 방법

`code/main.py`는 이전 레슨의 패턴들을 하나의 실행 가능한 데모로 통합합니다. 모든 표준 라이브러리(stdlib)를 사용하며, 인프로세스(in-process) 방식으로 구현되어 처음부터 끝까지 읽을 수 있습니다. 이 데모는 연구 및 보고서 시나리오의 전체 흐름을 실행합니다: 게이트웨이와의 핸드셰이크, OAuth 2.1 시뮬레이션, 도구 목록 병합, 작업(task)으로서의 `generate_report`, 작성자(writer)로의 A2A 호출, `ui://` 리소스 반환, OTel 스팬(span) 발행.

주목할 부분:

- 모든 홉(hop)에서 하나의 트레이스 ID(trace id)가 유지됩니다.
- 게이트웨이 정책(policy)이 두 번째 사용자의 쓰기를 차단합니다.
- 작업 수명 주기(task lifecycle)는 `working` → `completed` 상태로 전환되며 텍스트와 `ui://` 콘텐츠를 모두 반환합니다.
- A2A 호출의 내부 상태(inner state)는 오케스트레이터(orchestrator)에게 불투명(opaque)합니다.
- `AGENTS.md`와 `SKILL.md`는 다른 에이전트가 워크플로우를 재현하는 데 필요한 유일한 파일입니다.

## Ship It

이 레슨은 `outputs/skill-ecosystem-blueprint.md`를 생성합니다. 제품 요구사항(연구, 요약, 자동화)이 주어지면, 스킬은 전체 아키텍처를 생성합니다: 어떤 MCP 프리미티브, 어떤 게이트웨이 제어, 어떤 A2A 호출, 어떤 텔레메트리, 어떤 패키징이 필요한지.

## 연습 문제

1. `code/main.py`를 실행해 보세요. 단일 트레이스 ID와 스팬(span)이 중첩되는 방식을 확인하세요. 데모가 Phase 13의 몇 가지 기본 요소(primitive)를 사용하는지 세어 보세요.

2. 데모 확장: 두 번째 백엔드 MCP 서버(예: `bibliography`)를 추가하고 게이트웨이가 동일한 네임스페이스에 해당 도구를 병합하는지 확인하세요.

3. 가짜 A2A(agent-to-agent) 기록 에이전트(writer agent)를 Lesson 19 하네스(harness)를 사용하는 실제 서브프로세스로 교체하세요.

4. 오케스트레이터(orchestrator)와 LLM 사이에 라우팅 게이트웨이에 PII(개인 식별 정보) 삭제 단계를 추가하세요. 사용자 쿼리의 이메일 주소가 제거되는지 확인하세요.

5. 이 시스템을 유지보수할 팀원을 위한 AGENTS.md를 작성하세요. 5분 이내에 읽을 수 있어야 하며, Cursor 또는 Codex에서 캡스톤(capstone)을 구동하는 데 필요한 모든 정보를 제공해야 합니다.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 캡스톤(Capstone) | "Phase-13 통합 데모" | 모든 기본 요소를 사용하는 엔드-투-엔드 시스템 |
| 연구 및 보고(Research and report) | "시나리오" | 검색, 요약, 패턴 렌더링 |
| 생태계(Ecosystem) | "모든 구성 요소 통합" | 서버 + 클라이언트 + 게이트웨이 + 서브 에이전트 + 원격 분석 + 패키지 |
| 추적 계층 구조(Trace hierarchy) | "단일 추적 ID" | 모든 홉의 스팬이 추적을 공유; 스팬 ID를 통한 부모-자식 관계 |
| 게이트웨이 발급 토큰(Gateway-issued token) | "전이적 인증" | 클라이언트는 게이트웨이의 토큰만 확인; 게이트웨이가 상위 자격 증명 보유 |
| 병합된 네임스페이스(Merged namespace) | "모든 도구를 단일 평면 목록에" | 게이트웨이에서 다중 서버 병합, 충돌 시 접두사 추가 |
| 불투명성 경계(Opacity boundary) | "A2A 호출은 내부 숨김" | 서브 에이전트의 추론 과정이 오케스트레이터에 보이지 않음 |
| 3계층 스택(Three-layer stack) | "AGENTS.md + SKILL.md + MCP" | 프로젝트 컨텍스트 + 워크플로우 + 도구 |
| 다층 방어(Defense-in-depth) | "다중 보안 계층" | 고정된 해시, OAuth, RBAC, Rule of Two, 감사 로그 |
| 사양 준수 매트릭스(Spec compliance matrix) | "사양이 요구하는 우리가 제공하는 것" | 2025-11-25 요구사항과 납품물을 매핑하는 체크리스트 |

## 추가 자료

- [MCP — 사양 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 통합 참조 문서
- [MCP 블로그 — 2026 로드맵](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 프로토콜의 발전 방향
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 참조 문서
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 표준 추적 규칙
- [Anthropic — Claude Agent SDK 개요](https://code.claude.com/docs/en/agent-sdk/overview) — 프로덕션 에이전트 런타임 패턴