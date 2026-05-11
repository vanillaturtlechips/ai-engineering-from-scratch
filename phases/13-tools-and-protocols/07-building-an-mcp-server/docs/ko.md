# MCP 서버 구축하기 — Python + TypeScript SDKs

> 대부분의 MCP 튜토리얼은 stdio "hello-world" 예제만 보여줍니다. 실제 서버는 도구, 리소스, 프롬프트를 노출하고, 기능 협상(capability negotiation)을 처리하며, 구조화된 오류를 발생시키고, SDK 간에 동일하게 작동해야 합니다. 이 레슨에서는 노트 서버를 처음부터 끝까지 구축합니다: stdlib stdio 전송, JSON-RPC 디스패치, 세 가지 서버 기본 요소, 그리고 Python SDK의 FastMCP 또는 TypeScript SDK로 전환할 때 바로 적용할 수 있는 순수 함수 스타일.

**유형:** 구축(Build)
**언어:** Python (stdlib, stdio MCP 서버)
**선수 조건:** 13단계 · 06 (MCP 기초)
**소요 시간:** ~75분

## 학습 목표

- `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get` 메서드 구현.
- stdin에서 JSON-RPC 메시지를 읽고 stdout에 응답을 쓰는 디스패치 루프 작성.
- JSON-RPC 2.0 스펙 및 MCP 추가 코드에 따른 구조화된 오류 응답 발생.
- 도구 로직을 재작성하지 않고 stdlib 구현을 FastMCP(Python SDK) 또는 TypeScript SDK로 전환.

## 문제

원격 전송(Phase 13 · 09) 또는 인증 계층(Phase 13 · 16)을 사용하기 전에 깨끗한 로컬 서버가 필요합니다. 로컬은 stdio를 의미합니다: 서버는 클라이언트에 의해 자식 프로세스로 생성되며, 메시지는 stdin/stdout을 통해 개행 문자로 구분된 형태로 흐릅니다.

2025-11-25 사양에서는 stdio 메시지가 명시적 `\n` 구분자를 가진 JSON 객체로 인코딩되도록 규정합니다. 여기에는 SSE(Server-Sent Events)가 없습니다; SSE는 이전 원격 모드였으며 2026년 중반에 제거될 예정입니다(Atlassian의 Rovo MCP 서버는 2026년 6월 30일에, Keboola는 2026년 4월 1일에 사용 중단). stdio의 경우 한 줄에 하나의 JSON 객체가 전체 와이어 포맷입니다.

노트 서버는 세 가지 서버 기본 기능을 모두 테스트하기 때문에 좋은 형태입니다. 도구들은 변형 작업(`notes_create`)을 수행합니다. 리소스는 데이터(`notes://{id}`)를 노출합니다. 프롬프트는 템플릿(`review_note`)을 제공합니다. 이 레슨의 구조는 모든 도메인에 일반화될 수 있습니다.

## 개념

### 디스패치 루프

```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

세 가지 규칙:

- JSON-RPC 봉투가 아닌 내용은 stdout에 출력하지 마세요. 디버그 로그는 stderr로 보냅니다.
- 모든 요청은 동일한 `id`를 가진 응답과 매칭되어야 합니다.
- 알림에는 응답하지 마세요.

### `initialize` 구현

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

지원하는 기능만 선언하세요. 클라이언트는 기능 집합을 기반으로 기능을 제어합니다.

### `tools/list` 및 `tools/call` 구현

`tools/list`는 `{tools: [...]}`를 반환하며, 각 항목에는 `name`, `description`, `inputSchema`가 포함됩니다. `tools/call`은 `{name, arguments}`를 입력으로 받아 `{content: [blocks], isError: bool}`을 반환합니다.

콘텐츠 블록은 타입이 지정됩니다. 가장 일반적인 타입:

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

도구 오류는 두 가지 형태로 나타납니다. 프로토콜 수준 오류(알 수 없는 메서드, 잘못된 파라미터)는 JSON-RPC 오류입니다. 도구 수준 오류(유효한 호출이지만 도구가 실패한 경우)는 `{content: [...], isError: true}`로 반환됩니다. 이를 통해 모델은 실패 내용을 컨텍스트에서 확인할 수 있습니다.

### 리소스 구현

리소스는 기본적으로 읽기 전용입니다. `resources/list`는 매니페스트를 반환하고, `resources/read`는 콘텐츠를 반환합니다. URI는 `file://...`, `http://...` 또는 `notes://`와 같은 사용자 정의 스키마를 사용할 수 있습니다.

데이터를 도구가 아닌 리소스로 노출할 때:

- 모델은 이를 "호출"하지 않습니다. 클라이언트는 사용자 요청 시 컨텍스트에 주입할 수 있습니다.
- 구독을 통해 리소스가 변경될 때 서버가 업데이트를 푸시할 수 있습니다(Phase 13 · 10).
- Phase 13 · 14에서는 `ui://`를 통해 대화형 리소스를 확장합니다.

### 프롬프트 구현

프롬프트는 이름이 지정된 인수를 가진 템플릿입니다. 호스트는 이를 슬래시 명령으로 표시합니다. `review_note` 프롬프트는 `note_id` 인수를 받아 클라이언트가 모델에 제공하는 다중 메시지 프롬프트 템플릿을 생성할 수 있습니다.

### Stdio 전송 세부 사항

- 개행으로 구분된 JSON. 길이 접두사 프레임 없음.
- 버퍼링하지 마세요. 각 쓰기 후 `sys.stdout.flush()`를 호출하세요.
- 클라이언트가 수명을 제어합니다. stdin이 닫히면(EOF) 정상적으로 종료하세요.
- SIGPIPE를 조용히 처리하지 마세요. 로그를 남기고 종료하세요.

### 어노테이션

각 도구는 안전 속성을 설명하는 `annotations`를 가질 수 있습니다:

- `readOnlyHint: true` — 순수 읽기, 재시도해도 안전합니다.
- `destructiveHint: true` — 되돌릴 수 없는 부작용; 클라이언트는 확인을 요청해야 합니다.
- `idempotentHint: true` — 동일한 입력에 대해 동일한 출력을 생성합니다.
- `openWorldHint: true` — 외부 시스템과 상호작용합니다.

클라이언트는 이를 사용하여 UX(확인 대화상자, 상태 표시기)와 라우팅(Phase 13 · 17)을 결정합니다.

### 졸업 경로

`code/main.py`의 stdlib 서버는 약 180줄입니다. FastMCP(Python)는 동일한 로직을 데코레이터 스타일로 축소합니다:

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK도 동일한 구조를 가집니다. 졸업 경로는 준비가 되면 바로 사용할 수 있습니다. 개념(기능, 디스패치, 콘텐츠 블록)은 동일합니다.

## 사용 방법

`code/main.py`는 stdio와 stdlib만을 사용하는 완전한 Notes MCP 서버입니다. `initialize`, `tools/list`, `tools/call`(세 가지 도구: `notes_list`, `notes_search`, `notes_create` 지원), 각 노트에 대한 `resources/list` 및 `resources/read`, 그리고 `review_note` 프롬프트를 처리합니다. JSON-RPC 메시지를 파이핑하여 구동할 수 있습니다:

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

주목할 부분:

- 디스패처는 메서드 이름을 키로 하는 `dict[str, Callable]`입니다.
- 모든 도구 실행기는 단순한 문자열이 아닌 콘텐츠 블록 목록을 반환합니다.
- 실행기에서 예외가 발생하면 `isError: true`가 설정됩니다.

## Ship It

이 레슨은 `outputs/skill-mcp-server-scaffolder.md`를 생성합니다. 도메인(노트, 티켓, 파일, 데이터베이스)이 주어지면, 스킬은 적절한 도구/리소스/프롬프트 분할 및 SDK 졸업 경로를 갖춘 MCP 서버를 스캐폴딩합니다.

## 연습 문제

1. `code/main.py`를 실행하고 수작업으로 만든 JSON-RPC 메시지로 구동해 보세요. `notes_create`를 실행한 다음, `resources/read`를 사용하여 새로 생성된 노트를 검색해 보세요.

2. `annotations: {destructiveHint: true}`를 포함하는 `notes_delete` 도구를 추가해 보세요. 클라이언트가 확인 대화상자를 표시하는지 검증해 보세요(실제 호스트가 필요함; Claude Desktop 사용 가능).

3. `resources/subscribe`를 구현하여 서버가 노트가 수정될 때마다 `notifications/resources/updated`를 푸시하도록 해 보세요. keepalive 작업도 추가해 보세요.

4. 서버를 FastMCP로 이식해 보세요. Python 파일은 80줄 미만으로 줄어들어야 합니다. 와이어 동작은 동일해야 하며, 동일한 JSON-RPC 테스트 하네스로 검증해야 합니다.

5. 명세서의 `server/tools` 섹션을 읽고 이 레슨의 서버에서 구현되지 않은 도구 정의의 필드 하나를 식별해 보세요. (힌트: 여러 가지가 있음; 하나를 골라 추가해 보세요.)

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| MCP 서버 | "도구를 노출하는 것" | stdio 또는 HTTP를 통해 MCP JSON-RPC를 사용하는 프로세스 |
| stdio 전송 | "자식 프로세스 모델" | 클라이언트가 서버를 생성; stdin/stdout을 통해 통신 |
| 디스패처 | "메서드 라우터" | JSON-RPC 메서드 이름과 핸들러 함수 간의 매핑 |
| 콘텐츠 블록 | "도구 결과 청크" | 도구 응답의 `content` 배열에 있는 타입이 지정된 요소 |
| `isError` | "도구 수준 실패" | 도구 실패를 신호; JSON-RPC 오류와 구분 |
| 어노테이션 | "안전성 힌트" | readOnly / destructive / idempotent / openWorld 플래그 |
| FastMCP | "Python SDK" | MCP 프로토콜 기반의 데코레이터 기반 상위 프레임워크 |
| 리소스 URI | "주소 지정 가능한 데이터" | `file://`, `db://` 또는 사용자 정의 스키마로 식별되는 리소스 |
| 프롬프트 템플릿 | "슬래시 명령 요약" | 호스트 UI를 위한 인수 슬롯이 포함된 서버 제공 템플릿 |
| 기능 선언 | "기능 토글" | `initialize`에서 선언된 기본 요소별 플래그 |

## 추가 자료

- [모델 컨텍스트 프로토콜 — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — 참조 Python 구현
- [모델 컨텍스트 프로토콜 — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 병렬 TS 구현
- [FastMCP — 서버 프레임워크](https://gofastmcp.com/) — MCP 서버를 위한 데코레이터 스타일 Python API
- [MCP — 빠른 시작 서버 가이드](https://modelcontextprotocol.io/quickstart/server) — 어느 SDK를 사용한 엔드-투-엔드 튜토리얼
- [MCP — 서버 도구 사양](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* 메시지에 대한 완전한 참조