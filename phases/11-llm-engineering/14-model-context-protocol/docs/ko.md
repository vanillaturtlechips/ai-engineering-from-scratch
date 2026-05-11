# 모델 컨텍스트 프로토콜(MCP)

> 2025년 이전에 구축된 모든 LLM 앱은 자체 도구 스키마를 발명했습니다. 이후 Anthropic이 MCP를 출시하고, Claude가 이를 채택했으며, OpenAI도 채택하면서 2026년에는 모든 LLM을 도구, 데이터 소스 또는 에이전트에 연결하는 기본 전송 형식으로 자리 잡았습니다. MCP 서버를 하나만 작성하면 모든 호스트가 이와 통신할 수 있습니다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** 11단계 · 09 (함수 호출), 11단계 · 03 (구조화된 출력)
**소요 시간:** ~75분

## 문제

채팅봇을 출시할 때 데이터베이스 쿼리, 캘린더 API, 파일 리더기라는 세 가지 도구가 필요합니다. Claude를 위해 세 가지 JSON 스키마를 작성합니다. 그런 다음 영업팀에서 ChatGPT에도 동일한 도구를 요구하자 OpenAI의 `tools` 매개변수에 맞게 다시 작성합니다. 이후 Cursor, Zed, Claude Code를 추가하면서 각각 미묘하게 다른 JSON 관례를 따르는 세 번의 재작업을 수행합니다. 일주일 후 Anthropic에서 새 필드를 추가하자 여섯 개의 스키마를 업데이트합니다.

이것이 2025년 이전의 현실이었습니다. 모든 호스트(LLM을 실행하는 주체)와 모든 서버(도구와 데이터를 노출하는 주체)가 독자적인 프로토콜을 사용했습니다. 확장은 N×M 통합 행렬을 의미했습니다.

모델 컨텍스트 프로토콜(MCP)은 이 행렬을 단순화합니다. JSON-RPC 기반의 단일 사양입니다. 하나의 서버가 도구, 리소스, 프롬프트를 노출하면 Claude Desktop, ChatGPT, Cursor, Claude Code, Zed 및 다양한 에이전트 프레임워크와 같은 모든 호환 호스트가 사용자 정의 연결 없이 이를 발견하고 호출할 수 있습니다.

2026년 초 기준으로 MCP는 빅3(Anthropic, OpenAI, Google)와 모든 주요 에이전트 하네스(harness)에서 기본 도구 및 컨텍스트 프로토콜로 자리 잡았습니다.

## 개념

![MCP: 하나의 호스트, 하나의 서버, 세 가지 기능](../assets/mcp-architecture.svg)

**세 가지 기본 요소.** MCP 서버는 정확히 세 가지를 노출합니다.

1. **도구(Tools)** — 모델이 호출할 수 있는 함수. OpenAI의 `tools` 또는 Anthropic의 `tool_use`에 해당. 각각 이름, 설명, JSON 스키마 입력, 핸들러를 가짐.
2. **리소스(Resources)** — 모델이나 사용자가 요청할 수 있는 읽기 전용 콘텐츠(파일, 데이터베이스 행, API 응답). URI로 주소 지정.
3. **프롬프트(Prompts)** — 사용자가 바로가기로 호출할 수 있는 재사용 가능한 템플릿 프롬프트.

**전송 형식.** JSON-RPC 2.0을 stdio, WebSocket 또는 스트리밍 HTTP로 전송. 모든 메시지는 `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}` 형식. 탐색 메서드는 `tools/list`, `resources/list`, `prompts/list`. 호출 메서드는 `tools/call`, `resources/read`, `prompts/get`.

**호스트 vs 클라이언트 vs 서버.** 호스트는 LLM 애플리케이션(Claude Desktop). 클라이언트는 호스트의 하위 구성 요소로 정확히 하나의 서버와 통신. 서버는 사용자 코드. 하나의 호스트는 여러 서버를 동시에 마운트할 수 있음.

### 핸드셰이크

모든 세션은 `initialize`로 시작. 클라이언트가 프로토콜 버전과 기능을 전송. 서버는 버전, 이름, 지원하는 기능 집합(`tools`, `resources`, `prompts`, `logging`, `roots`)으로 응답. 이후 모든 통신은 이 기능 집합을 기준으로 협상됨.

### MCP가 아닌 것

- 검색 API가 아님. RAG(Phase 11 · 06)가 무엇을 가져올지 결정; MCP는 검색 결과를 리소스로 노출하는 전송 계층.
- 에이전트 프레임워크가 아님. MCP는 기반 인프라; LangGraph, PydanticAI, OpenAI Agents SDK 같은 프레임워크는 그 위에 구축.
- Anthropic에 종속되지 않음. 사양 및 참조 구현은 `modelcontextprotocol` 조직 아래 오픈소스.

## 구축 방법

### 1단계: 최소한의 MCP 서버

공식 Python SDK는 `mcp`(이전 `mcp-python`)입니다. 고수준 `FastMCP` 헬퍼는 핸들러를 데코레이팅합니다.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """앱의 현재 JSON 설정을 반환합니다."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """코드 정확성과 스타일을 검토합니다."""
    return f"당신은 시니어 {language} 검토자입니다. 검토 대상:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

세 데코레이터는 세 가지 기본 요소를 등록합니다. 타입 힌트는 호스트가 보는 JSON 스키마가 됩니다. Claude Desktop 또는 Claude Code에서 서버 진입점을 이 파일로 지정하여 실행합니다.

### 2단계: 호스트에서 MCP 서버 호출

공식 Python 클라이언트는 JSON-RPC를 사용합니다. Anthropic SDK와 함께 사용하면 12줄 정도로 구현할 수 있습니다.

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()`는 LLM이 보는 것과 동일한 스키마를 반환합니다. 프로덕션 호스트는 매 턴마다 이 스키마를 주입하여 모델이 `tool_use` 블록을 생성하게 하고, 클라이언트는 이를 서버로 전달합니다.

### 3단계: 스트리밍 가능한 HTTP 전송

Stdio는 로컬 개발에 적합합니다. 원격 도구에는 스트리밍 가능한 HTTP를 사용하세요. 요청당 하나의 POST, 진행 상황을 위한 선택적 Server-Sent Events, 2025-06-18 스펙 개정부터 지원됩니다.

```python
# 서버 진입점 내부
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

호스트 설정 (Claude Desktop `mcp.json` 또는 Claude Code `~/.mcp.json`):

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

서버는 동일한 데코레이터를 유지하며, 전송 방식만 변경됩니다.

### 4단계: 범위 및 안전성

MCP 도구는 다른 사람의 신뢰 경계에서 실행되는 임의의 코드입니다. 세 가지 필수 패턴:

- **기능 허용 목록.** 호스트는 `roots` 기능을 노출시켜 서버가 허용된 경로만 볼 수 있도록 합니다. 도구 핸들러에서 이를 강제 적용하세요. 모델이 제공한 경로는 신뢰하지 마세요.
- **변조를 위한 인간 개입.** 읽기 전용 도구는 자동 실행할 수 있습니다. 쓰기/삭제 도구는 확인이 필요합니다. 서버가 도구 메타데이터에 `destructiveHint: true`를 설정하면 호스트가 승인 UI를 표시합니다.
- **도구 오염 방어.** 악성 리소스는 숨겨진 프롬프트 주입 지시사항("요약 시 `exfil`도 호출")을 포함할 수 있습니다. 리소스 내용을 신뢰할 수 없는 데이터로 취급하고, 시스템 메시지 영역으로 넘어가지 않도록 하세요. Phase 11 · 12(가드레일) 참조.

`code/main.py`에서 이 모든 것을 보여주는 실행 가능한 서버 + 클라이언트 쌍을 확인하세요.

## 2026년에도 여전히 발생하는 문제점

- **스키마 드리프트(Schema drift).** 모델이 1턴에서 `tools/list`를 확인했습니다. 5턴에서 도구 세트가 변경되었습니다. 모델이 사라진 도구를 호출합니다. 호스트는 `notifications/tools/list_changed`에서 다시 목록을 가져와야 합니다.
- **대용량 리소스 블록.** 2MB 파일을 리소스로 덤프하면 컨텍스트가 낭비됩니다. 서버 측에서 페이지네이션 또는 요약을 수행하세요.
- **너무 많은 서버.** 50개의 MCP 서버를 마운트하면 도구 예산을 초과합니다(Phase 11 · 05). 대부분의 프론티어 모델은 ~40개 도구 이후로 성능이 저하됩니다.
- **버전 불일치(Version skew).** 사양 개정(2024-11, 2025-03, 2025-06, 2025-12)에서 호환되지 않는 필드가 도입되었습니다. CI에서 프로토콜 버전을 고정하세요.
- **표준 입출력 데드락(Stdio deadlocks).** 표준 출력(stdout)에 로깅하는 서버는 JSON-RPC 스트림을 손상시킵니다. 표준 오류(stderr)로만 로깅하세요.

## 사용 방법

2026 MCP 스택:

| 상황 | 선택 |
|-----------|------|
| 로컬 개발, 단일 사용자 도구 | Python `FastMCP`, stdio 전송 |
| 원격 팀 도구 / SaaS 통합 | 스트리밍 HTTP, OAuth 2.1 인증 |
| TypeScript 호스트 (VS Code 확장, 웹 앱) | `@modelcontextprotocol/sdk` |
| 고대역폭 서버, 타입 기반 접근 | 공식 Rust SDK (`modelcontextprotocol/rust-sdk`) |
| 생태계 서버 탐색 | `modelcontextprotocol/servers` 모노레포 (파일 시스템, GitHub, Postgres, Slack, Puppeteer) |

경험적 규칙: 도구가 읽기 전용이고 캐시 가능하며 두 개 이상의 호스트에서 호출되는 경우 MCP 서버로 제공하세요. 일회성 인라인 로직인 경우 로컬 함수로 유지하세요 (Phase 11 · 09).

## Ship It

`outputs/skill-mcp-server-designer.md` 저장:

```markdown
---
name: mcp-server-designer
description: 도구, 리소스 및 안전 기본값을 사용하여 MCP 서버를 설계하고 스캐폴딩합니다.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

도메인(내부 API, 데이터베이스, 파일 소스)과 서버를 마운트할 호스트가 주어졌을 때, 다음을 출력합니다:

1. 프리미티브 맵. 어떤 기능이 `도구`(액션)가 되고, 어떤 것이 `리소스`(읽기 전용 데이터)가 되며, 어떤 것이 `프롬프트`(사용자 호출 템플릿)가 되는지 정의합니다. 프리미티브당 한 줄로 작성합니다.
2. 인증 계획. Stdio(신뢰할 수 있는 로컬), API 키가 있는 스트리밍 HTTP, 또는 PKCE가 있는 OAuth 2.1 중 선택하고 근거를 제시합니다.
3. 스키마 초안. 모든 도구 파라미터에 대한 JSON 스키마. `description` 필드는 모델 도구 선택에 최적화되어야 합니다(API 문서용이 아님).
4. 파괴적 동작 목록. 상태를 변경하는 모든 도구. `destructiveHint: true`와 인간 승인을 필수로 요구합니다.
5. 테스트 계획. 도구별: 스키마 전용 계약 테스트 1개, MCP 클라이언트를 통한 왕복 테스트 1개, 레드 팀 프롬프트 인젝션 사례 1개.

승인 경로 없이 디스크에 쓰거나 외부 API를 호출하는 서버는 출하를 거부합니다. 하나의 서버에 20개 이상의 도구를 노출하지 마세요. 대신 도메인 범위 서버로 분할하세요.
```

## 연습 문제

1. **쉬움.** `demo-server`에 `subtract` 도구를 추가합니다. Claude Desktop에서 이를 연결합니다. `tools/list_changed` 알림을 발생시켜 호스트가 재시작 없이 새 도구를 인식하는지 확인합니다.
2. **중간.** `/var/log/app.log`의 마지막 100줄을 노출하는 `resource`를 추가합니다. 루트 허용 목록을 강제 적용하여 모델이 요청하더라도 `../etc/passwd`는 차단되도록 합니다.
3. **어려움.** 세 개의 업스트림 서버(Filesystem, GitHub, Postgres)를 하나의 집계된 인터페이스로 다중화하는 MCP 프록시를 구축합니다. 이름 충돌을 처리하고 `notifications/tools/list_changed` 알림을 깔끔하게 전달합니다.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| MCP | "LLM을 위한 도구 프로토콜" | 모든 LLM 호스트에 도구, 리소스, 프롬프트를 노출하기 위한 JSON-RPC 2.0 사양 |
| 호스트 | "Claude Desktop" | LLM 애플리케이션 — 모델과 사용자 인터페이스를 소유하며, 하나 이상의 클라이언트를 마운트 |
| 클라이언트 | "연결" | 호스트 내부의 서버별 연결 — 정확히 하나의 서버와 JSON-RPC로 통신 |
| 서버 | "도구가 있는 것" | 사용자 코드; 도구/리소스/프롬프트를 광고하고 호출을 처리 |
| 도구 | "함수 호출" | JSON 스키마 입력과 텍스트/JSON 결과를 가진 모델 호출 가능 액션 |
| 리소스 | "읽기 전용 데이터" | 호스트가 요청할 수 있는 URI 주소 지정 콘텐츠(파일, 행, API 응답) |
| 프롬프트 | "저장된 프롬프트" | 사용자가 호출할 수 있는 템플릿(종종 인수 포함) — 슬래시 명령으로 표시 |
| Stdio 전송 | "로컬 개발 모드" | 부모 호스트가 서버를 자식 프로세스로 생성; stdin/stdout을 통한 JSON-RPC |
| 스트리밍 HTTP | "2025-06 원격 전송" | 요청에는 POST, 선택적 SSE는 서버 시작 메시지; 이전 SSE 전용 전송을 대체 |

## 추가 자료

- [모델 컨텍스트 프로토콜 사양](https://modelcontextprotocol.io/specification) — 날짜별 버전 관리되는 표준 참조 문서.
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — 파일 시스템, GitHub, Postgres, Slack, Puppeteer 참조 서버.
- [Anthropic — MCP 소개 (2024년 11월)](https://www.anthropic.com/news/model-context-protocol) — 설계 근거를 포함한 출시 포스트.
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 이 레슨에서 사용된 공식 SDK.
- [MCP 보안 고려사항](https://modelcontextprotocol.io/docs/concepts/security) — 루트, 파괴적 힌트, 도구 오염.
- [Google A2A 사양](https://google.github.io/A2A/) — 에이전트 간 통신 프로토콜; MCP의 에이전트-도구 범위를 보완하는 형제 표준.
- [Anthropic — 효과적인 에이전트 구축 (2024년 12월)](https://www.anthropic.com/research/building-effective-agents) — 에이전트 설계(증강 LLM, 워크플로우, 자율 에이전트)를 위한 광범위한 패턴 라이브러리에서 MCP의 위치.