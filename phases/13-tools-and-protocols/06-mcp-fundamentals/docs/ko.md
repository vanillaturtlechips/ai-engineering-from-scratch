# MCP 기본 사항 — 프리미티브, 라이프사이클, JSON-RPC 기반

> MCP 이전의 모든 통합은 일회성 작업이었습니다. 2024년 11월 Anthropic에서 처음 출시하고 현재 Linux Foundation의 Agentic AI Foundation에서 관리하는 모델 컨텍스트 프로토콜(MCP)은 발견 및 호출을 표준화하여 모든 클라이언트가 모든 서버와 통신할 수 있도록 합니다. 2025-11-25 사양은 6개의 프리미티브(서버 3개, 클라이언트 3개), 3단계 라이프사이클, JSON-RPC 2.0 와이어 포맷을 정의합니다. 이 내용을 학습하면 이 단계의 MCP 장 나머지 부분은 읽기만 하면 됩니다.

**유형:** 학습  
**언어:** Python (표준 라이브러리, JSON-RPC 파서)  
**선수 지식:** 13단계 · 01~05 (도구 인터페이스 및 함수 호출)  
**소요 시간:** ~45분

## 학습 목표

- 6가지 MCP 기본 요소(서버 측: 도구, 리소스, 프롬프트; 클라이언트 측: 루트, 샘플링, 추출)의 이름을 모두 말하고, 각각에 대한 사용 사례 하나씩 제시.
- 3단계 라이프사이클(초기화, 운영, 종료)을 설명하고, 각 단계에서 누가 어떤 메시지를 보내는지 명시.
- JSON-RPC 2.0 요청, 응답, 알림(envelope) 구조를 파싱 및 생성.
- `initialize` 단계에서의 기능 협상(capability negotiation)이 무엇인지 설명하고, 이것이 없을 때 발생하는 문제 설명.

## 문제

MCP 이전에는 모든 도구 사용 에이전트가 자체 프로토콜을 가지고 있었습니다. Cursor는 MCP와 유사하지만 호환되지 않는 도구 시스템을 가지고 있었습니다. Claude Desktop은 다른 프로토콜을 탑재하고 출시되었습니다. VS Code의 Copilot 확장 프로그램은 세 번째 프로토콜을 사용했습니다. "Postgres 쿼리" 도구를 구축한 팀은 각 호스트의 API에 맞춰 동일한 도구를 세 번 작성해야 했습니다. 재사용을 위해서는 코드를 복사해야 했습니다.

결과적으로 일회성 통합이 폭발적으로 증가했고 생태계 발전 속도가 정체되었습니다.

MCP는 와이어 포맷을 표준화하여 이 문제를 해결합니다. 단일 MCP 서버는 모든 MCP 클라이언트에서 작동합니다: Claude Desktop, ChatGPT, Cursor, VS Code, Gemini, Goose, Zed, Windsurf, 2026년 4월 기준 300개 이상의 클라이언트. 월간 SDK 다운로드는 1억 1천만 건, 공개 서버는 10,000개 이상입니다. 리눅스 재단은 2025년 12월 새로운 에이전트 AI 재단(Agentic AI Foundation) 아래 관리권을 인수했습니다.

이 단계에서 사용되는 사양 개정판은 **2025-11-25**입니다. 여기에는 비동기 작업(SEP-1686), URL 모드 유도(SEP-1036), 도구를 이용한 샘플링(SEP-1577), 점진적 범위 동의(SEP-835), OAuth 2.1 리소스 지시자 시맨틱스가 추가되었습니다. 13단계 · 09~16은 이러한 확장 기능을 다룹니다. 이 레슨은 기본 사항까지만 설명합니다.

## 개념

### 세 가지 서버 기본 요소

1. **도구(Tools).** 호출 가능한 액션. 13단계 · 01과 동일한 4단계 루프.
2. **리소스(Resources).** 노출된 데이터. URI로 주소 지정 가능한 읽기 전용 콘텐츠: `file:///path`, `db://query/...`, 사용자 정의 스킴.
3. **프롬프트(Prompts).** 재사용 가능한 템플릿. 호스트 UI의 슬래시 명령어; 서버가 템플릿을 제공하고 클라이언트가 인수를 채움.

### 세 가지 클라이언트 기본 요소

4. **루트(Roots).** 서버가 접근할 수 있는 URI 집합. 클라이언트가 선언; 서버는 이를 존중.
5. **샘플링(Sampling).** 서버가 클라이언트의 모델에 완료를 요청. 서버 측 API 키 없이 서버 호스팅 에이전트 루프 가능.
6. **유도(Elicitation).** 서버가 클라이언트 사용자에게 중간 구조 입력을 요청. 양식 또는 URL(SEP-1036).

MCP의 모든 기능은 이 6가지 중 하나에 정확히 속함. 13단계 · 10부터 14까지는 각각을 심층적으로 다룸.

### 와이어 형식: JSON-RPC 2.0

모든 메시지는 다음 필드를 가진 JSON 객체:

- 요청: `{jsonrpc: "2.0", id, method, params}`.
- 응답: `{jsonrpc: "2.0", id, result | error}`.
- 알림: `{jsonrpc: "2.0", method, params}` — `id` 없음, 응답 기대 안 함.

기본 사양에는 ~15개의 메서드가 있으며 기본 요소별로 그룹화됨. 중요한 것들:

- `initialize` / `initialized` (핸드셰이크)
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage` (서버→클라이언트)
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### 세 단계 라이프사이클

**1단계: initialize.**

클라이언트가 `capabilities`와 `clientInfo`를 포함한 `initialize`를 보냄. 서버는 자체 `capabilities`, `serverInfo`, 사용하는 사양 버전으로 응답. 클라이언트는 응답을 소화한 후 `notifications/initialized`를 보냄. 이후 협상된 기능에 따라 양측이 요청을 보낼 수 있음.

**2단계: 운영.**

양방향. 클라이언트는 `tools/list`를 호출하여 발견한 후 `tools/call`로 실행. 서버는 해당 기능을 선언한 경우 `sampling/createMessage`를 보낼 수 있음. 서버는 도구 집합이 변경될 때 `notifications/tools/list_changed`를 보낼 수 있음. 사용자가 루트 범위를 변경할 때 클라이언트는 `notifications/roots/list_changed`를 보낼 수 있음.

**3단계: 종료.**

양측 중 하나가 전송을 닫음. MCP에는 구조화된 종료 메서드가 없음; 전송(stdio 또는 스트리밍 HTTP, 13단계 · 09)이 연결 종료 신호를 전달.

### 기능 협상

`initialize` 핸드셰이크의 `capabilities`는 계약. 서버 예시:

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

서버는 `tools/list_changed` 알림을 방출할 수 있고 `resources/subscribe`를 지원한다고 선언. 클라이언트는 자체 선언으로 동의:

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

클라이언트가 `sampling`을 선언하지 않으면 서버는 `sampling/createMessage`를 호출해서는 안 됨. 대칭적: 서버가 `resources.subscribe`를 선언하지 않으면 클라이언트는 구독을 시도해서는 안 됨.

이것이 생태계 드리프트를 방지. 샘플링을 지원하지 않는 클라이언트는 여전히 유효한 MCP 클라이언트; `sampling`을 호출하지 않는 서버는 여전히 유효한 MCP 서버. 단지 해당 기능을 함께 사용하지 않을 뿐.

### 구조화된 콘텐츠와 오류 형태

`tools/call`은 `text`, `image`, `resource`와 같은 타입 블록인 `content` 배열을 반환. 13단계 · 14는 MCP 앱(`ui://` 대화형 UI)을 이 목록에 추가.

오류는 JSON-RPC 오류 코드를 사용. 사양 정의 추가 사항: `-32002` "리소스 없음", `-32603` "내부 오류", `error.data`로 MCP 특정 오류 데이터.

### 클라이언트 기능 vs 도구 호출 세부 사항

흔한 혼동: `capabilities.tools`는 클라이언트가 도구 목록 변경 알림을 지원하는지 여부. 특정 도구를 호출할지는 모델의 런타임 선택 사항이며 기능 플래그가 아님. 기능 플래그는 사양 수준 계약. 모델의 선택은 독립적.

### REST가 아닌 JSON-RPC를 선택한 이유

JSON-RPC 2.0(2010)은 가벼운 양방향 프로토콜. REST는 클라이언트 시작. MCP는 서버 시작 메시지(샘플링, 알림)가 필요했으므로 대칭적 요청/응답 형태의 JSON-RPC가 자연스럽게 적합. JSON-RPC는 stdio 및 WebSocket/스트리밍 HTTP 위에서 HTTP 요청 형태를 재창조하지 않고 깔끔하게 조합됨.

## 사용 방법

`code/main.py`는 최소한의 JSON-RPC 2.0 파서와 생성기를 제공하며, `initialize` → `tools/list` → `tools/call` → `shutdown` 시퀀스를 수동으로 실행하면서 모든 메시지를 출력합니다. 실제 전송 계층은 없으며, 메시지 구조만 확인할 수 있습니다. 추가 자료 섹션에 링크된 사양과 비교하여 각 메시지 구조를 검증해 보세요.

확인해야 할 사항:

- `initialize`는 양방향 기능을 선언하며, 응답에는 `serverInfo`와 `protocolVersion: "2025-11-25"`가 포함됩니다.
- `tools/list`는 `tools` 배열을 반환하며, 각 항목에는 `name`, `description`, `inputSchema`가 있습니다.
- `tools/call`은 `params.name`과 `params.arguments`를 사용합니다.
- 응답 `content`는 `{type, text}` 블록들의 배열입니다.

## Ship It

이 레슨은 `outputs/skill-mcp-handshake-tracer.md`를 생성합니다. MCP 클라이언트-서버 상호작용의 pcap 스타일 트랜스크립트가 주어지면, 스킬은 각 메시지에 대해 어떤 프리미티브(primitive), 어떤 라이프사이클 단계(lifecycle phase), 그리고 어떤 기능(capability)에 의존하는지 주석을 추가합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 기능 협상(capability negotiation)이 발생하는 줄을 식별하고, 서버가 `tools.listChanged`를 선언하지 않을 경우 어떤 변화가 있을지 설명하세요.

2. 파서를 확장하여 `notifications/progress`를 처리하도록 수정하세요. 메시지 구조: `{method: "notifications/progress", params: {progressToken, progress, total}}`. 장시간 실행되는 `tools/call` 작업 중에 이 메시지를 발생시키고, 클라이언트 핸들러가 진행률 표시줄을 표시하는지 확인하세요.

3. MCP 2025-11-25 명세서를 처음부터 끝까지 읽으세요(전체 약 80페이지). 대부분의 서버가 **필요하지 않은** 하나의 기능 플래그(capability flag)를 식별하세요. 힌트: 리소스 구독(resource subscription)과 관련이 있습니다.

4. 가상의 "크론 잡(cron job)" 기능이 속할 원시(primitive)를 종이에 스케치하세요. (힌트: 서버는 클라이언트가 예약된 시간에 이를 호출하기를 원합니다. 현재 6가지 원시에는 해당되지 않습니다.) MCP의 2026 로드맵에는 이를 위한 SEP 초안이 있습니다.

5. GitHub의 공개 MCP 서버에서 세션 로그 하나를 파싱하세요. 요청(request) vs 응답(response) vs 알림(notification) 메시지 수를 세고, 트래픽 중 라이프사이클(lifecycle) 대 작업(operation) 비율을 계산하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| MCP | "Model Context Protocol" | 모델-툴 간 발견 및 호출을 위한 개방형 프로토콜 |
| 서버 프리미티브 | "서버가 노출하는 것" | 툴(액션), 리소스(데이터), 프롬프트(템플릿) |
| 클라이언트 프리미티브 | "클라이언트가 서버가 사용하도록 허용하는 것" | 루트(범위), 샘플링(LLM 콜백), 유도(사용자 입력) |
| JSON-RPC 2.0 | "전송 형식" | 대칭형 요청/응답/알림 봉투 |
| `initialize` 핸드셰이크 | "기능 협상" | 첫 번째 메시지 쌍; 서버와 클라이언트가 지원하는 기능을 선언 |
| `tools/list` | "발견" | 클라이언트가 서버에 현재 툴 세트를 요청 |
| `tools/call` | "호출" | 클라이언트가 서버에 인수와 함께 툴 실행을 요청 |
| `notifications/*_changed` | "변경 이벤트" | 서버가 클라이언트에게 프리미티브 목록이 변경되었음을 알림 |
| 콘텐츠 블록 | "타입이 지정된 결과" | 툴 결과에서 `{type: "text" | "image" | "resource" | "ui_resource"}` |
| SEP | "명세 진화 제안" | 명명된 초안 제안 (예: 비동기 태스크를 위한 SEP-1686) |

## 추가 자료

- [모델 컨텍스트 프로토콜 — 사양 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 공식 사양 문서
- [모델 컨텍스트 프로토콜 — 아키텍처 개념](https://modelcontextprotocol.io/docs/concepts/architecture) — 6가지 기본 요소 정신 모델
- [앤트로픽 — 모델 컨텍스트 프로토콜 소개](https://www.anthropic.com/news/model-context-protocol) — 2024년 11월 출시 포스트
- [MCP 블로그 — 첫 MCP 기념일](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 1주년 회고 및 2025-11-25 사양 변경 사항
- [워크OS — MCP 2025-11-25 사양 업데이트](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686, 1036, 1577, 835, 1724 요약