# MCP 리소스 및 프롬프트 — 도구를 넘어선 컨텍스트 노출

> 도구는 MCP 관심의 90%를 차지합니다. 나머지 두 서버 기본 요소는 다른 문제를 해결합니다. 리소스는 데이터를 읽을 수 있도록 노출하고, 프롬프트는 재사용 가능한 템플릿을 슬래시 명령으로 노출합니다. 많은 서버는 도구에서 읽기 작업을 래핑하는 대신 리소스를 사용하고, 클라이언트 프롬프트에 워크플로우를 하드코딩하는 대신 프롬프트를 사용해야 합니다. 이 레슨에서는 결정 규칙을 명명하고 `resources/*` 및 `prompts/*` 메시지를 설명합니다.

**유형:** 구축
**언어:** Python (표준 라이브러리, 리소스 + 프롬프트 핸들러)
**선수 조건:** 13단계 · 07 (MCP 서버)
**소요 시간:** ~45분

## 학습 목표

- 주어진 도메인에 대해 기능을 도구(tool), 리소스(resource), 프롬프트(prompt) 중 어떤 형태로 노출할지 결정한다.
- `resources/list`, `resources/read`, `resources/subscribe`를 구현하고 `notifications/resources/updated`를 처리한다.
- 인수 템플릿(argument templates)을 사용하여 `prompts/list`와 `prompts/get`을 구현한다.
- 호스트가 프롬프트를 슬래시 명령어(slash-commands)로 노출하는지 vs 자동 주입된 컨텍스트(auto-injected context)로 노출하는지 구분한다.

## 문제

노트 앱용 순진한 MCP 서버는 모든 것을 도구(tool)로 노출합니다: `notes_read`, `notes_list`, `notes_search`. 이는 모든 데이터 접근을 모델 기반 도구 호출로 래핑하는 방식입니다. 결과:

- 모델은 컨텍스트에서 이점을 얻을 수 있는 모든 쿼리에 대해 `notes_read`를 호출할지 여부를 결정해야 합니다.
- 읽기 전용 콘텐츠는 구독하거나 호스트의 사이드 패널로 스트리밍할 수 없습니다.
- 클라이언트 UI(Claude Desktop의 리소스 첨부 패널, Cursor의 "파일 포함" 선택기 등)가 데이터를 표시할 수 없습니다.

올바른 분리 방식: 데이터를 리소스로 노출하고, 변형 또는 계산 작업을 도구로 노출하며, 재사용 가능한 다단계 워크플로우를 프롬프트로 노출합니다. 각 기본 요소에는 UX 지원 방식과 접근 패턴이 있습니다.

## 개념

### 도구 vs 리소스 vs 프롬프트 — 결정 규칙

| 기능 | 원시 요소 |
|------------|-----------|
| 사용자가 데이터를 검색, 필터링, 또는 변환하려는 경우 | 도구 |
| 사용자가 호스트가 이 데이터를 컨텍스트로 포함하기를 원하는 경우 | 리소스 |
| 사용자가 다시 실행할 수 있는 템플릿화된 워크플로우를 원하는 경우 | 프롬프트 |

지침: 모델이 모든 관련 쿼리에서 호출 시 이점을 얻는다면 도구입니다. 사용자가 대화에 첨부 시 이점을 얻는다면 리소스입니다. 전체 다단계 워크플로우가 사용자가 재사용하려는 단위라면 프롬프트입니다.

### 리소스

`resources/list`는 `{resources: [{uri, name, mimeType, description?}]}`를 반환합니다. `resources/read`는 `{uri}`를 입력으로 받아 `{contents: [{uri, mimeType, text | blob}]}`를 반환합니다.

URI는 주소 지정 가능한 모든 것이 될 수 있습니다:

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14` (커스텀 스킴)
- `memory://session-2026-04-22/recent` (서버 전용)

`contents[]`는 텍스트와 바이너리 모두 지원합니다. 바이너리는 `blob`을 base64 인코딩된 문자열과 함께 사용하며 `mimeType`을 포함합니다.

### 리소스 구독

기능에서 `{resources: {subscribe: true}}`를 선언합니다. 클라이언트는 `resources/subscribe {uri}`를 호출합니다. 서버는 리소스가 변경될 때 `notifications/resources/updated {uri}`를 전송합니다. 클라이언트는 다시 읽습니다.

사용 사례: 디스크 상의 파일을 리소스로 사용하는 노트 서버; 파일 감시자가 업데이트 알림을 트리거; Claude Desktop은 호스트 외부에서 편집된 파일을 컨텍스트로 다시 불러옵니다.

### 리소스 템플릿 (2025-11-25 추가)

`resourceTemplates`를 통해 파라미터화된 URI 패턴을 노출할 수 있습니다: `notes://{id}`에서 `id`를 완성 대상으로 지정. 클라이언트는 리소스 선택기에서 `id`를 자동 완성할 수 있습니다.

### 프롬프트

`prompts/list`는 `{prompts: [{name, description, arguments?}]}`를 반환합니다. `prompts/get`은 `{name, arguments}`를 입력으로 받아 `{description, messages: [{role, content}]}`를 반환합니다.

프롬프트는 호스트가 모델에 제공하는 메시지 목록으로 채워지는 템플릿입니다. 예를 들어, `code_review` 프롬프트는 `file_path` 인수를 받아 시스템 메시지, 파일 본문이 포함된 사용자 메시지, 추론 템플릿이 있는 어시스턴트 시작 메시지로 구성된 3단계 시퀀스를 반환합니다.

### 호스트와 프롬프트

Claude Desktop, VS Code, Cursor는 채팅 UI에서 프롬프트를 슬래시 명령으로 노출합니다. 사용자는 `/code_review`를 입력하고 폼에서 인수를 선택합니다. 서버의 프롬프트는 "사용자 단축키"와 "모델에 전송되는 전체 프롬프트" 간의 계약입니다.

모든 클라이언트가 프롬프트를 지원하는 것은 아닙니다 — 기능 협상을 확인하세요. 프롬프트 기능을 선언했지만 프롬프트 지원이 없는 클라이언트는 슬래시 명령을 볼 수 없습니다.

### "목록 변경" 알림

리소스와 프롬프트 모두 집합이 변경될 때 `notifications/list_changed`를 방출합니다. 20개의 새 노트를 방금 가져온 노트 서버는 `notifications/resources/list_changed`를 방출; 클라이언트는 추가 사항을 반영하기 위해 `resources/list`를 다시 호출합니다.

### 콘텐츠 유형 규칙

텍스트: `mimeType: "text/plain"`, `text/markdown`, `application/json`.
바이너리: `image/png`, `application/pdf`, 그리고 `blob` 필드.
MCP 앱(레슨 14): `ui://` URI에서 `text/html;profile=mcp-app`.

### 동적 리소스

리소스 URI는 정적 파일에 대응할 필요가 없습니다. `notes://recent`는 읽을 때마다 최신 5개 노트를 반환할 수 있습니다. `db://query/users/active`는 파라미터화된 쿼리를 실행할 수 있습니다. 서버는 콘텐츠를 동적으로 계산할 수 있습니다.

규칙: 클라이언트가 URI로 캐싱할 수 있다면 URI는 안정적이어야 합니다. 계산이 일회성이라면 URI에 타임스탬프나 난수를 포함시켜 클라이언트 캐시가 오래되지 않도록 해야 합니다.

### 구독 vs 폴링

구독 가능한 클라이언트는 `notifications/resources/updated`를 통해 서버 푸시를 받습니다. 구독 전 클라이언트나 지원하지 않는 호스트는 다시 읽어 폴링합니다. 둘 다 사양 준수입니다. 서버의 기능 선언이 클라이언트에게 지원 여부를 알려줍니다.

구독의 비용: 서버 측 세션별 상태(누가 무엇을 구독 중인지). 구독 집합을 유계로 유지; 연결이 끊긴 클라이언트는 타임아웃되어야 합니다.

### 프롬프트 vs 시스템 프롬프트

MCP의 프롬프트는 시스템 프롬프트가 아닙니다. 호스트의 시스템 프롬프트(자체 운영 지침)와 MCP 프롬프트(사용자가 호출하는 서버 제공 템플릿)는 나란히 존재합니다. 잘 동작하는 클라이언트는 서버 프롬프트가 자체 시스템 프롬프트를 덮어쓰지 않도록 계층화합니다.

## 사용 방법

`code/main.py`는 Lesson 07의 노트 서버를 다음과 같이 확장합니다:

- `resources/subscribe` 지원을 포함한 개별 노트 리소스(`notes://note-1` 등)
- 3개의 메시지 템플릿으로 렌더링되는 `review_note` 프롬프트
- 노트 수정 시 `notifications/resources/updated`를 발행하는 파일 감시 시뮬레이션
- 항상 최신 5개의 노트를 반환하는 동적 리소스 `notes://recent`

데모를 실행하여 전체 흐름을 확인하세요.

## Ship It

이 레슨은 `outputs/skill-primitive-splitter.md`를 생성합니다. 제안된 MCP 서버가 주어졌을 때, 스킬은 각 기능을 도구(tool)/자원(resource)/프롬프트(prompt)로 분류하고 근거를 제시합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 초기 리소스 목록을 관찰한 후, 노트 편집을 트리거하고 `notifications/resources/updated` 이벤트가 발생하는지 확인하세요.

2. `resources/list_changed` 발신자(emitter)를 추가하세요: 새 노트가 생성될 때 클라이언트가 다시 발견할 수 있도록 알림을 전송합니다.

3. GitHub MCP 서버를 위한 다음 세 가지 프롬프트를 설계하세요: `summarize_pr`, `triage_issue`, `release_notes`. 각각에 인수 스키마를 포함하고, 프롬프트 본문은 추가 편집 없이 실행 가능해야 합니다.

4. Lesson 07 서버의 기존 도구를 선택하여, 해당 도구가 그대로 유지되어야 하는지 아니면 리소스 + 도구 쌍으로 분리되어야 하는지 분류하세요. 한 문장으로 정당화하세요.

5. 사양의 `server/resources` 및 `server/prompts` 섹션을 읽으세요. `resources/read`에서 거의 채워지지 않지만 사양에서 지원되는 필드를 식별하세요. 힌트: 리소스 콘텐츠의 `_meta`를 확인하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 리소스 | "노출된 데이터" | 호스트가 읽을 수 있는 URI 주소 지정 가능 콘텐츠 |
| 리소스 URI | "데이터 포인터" | 스키마 접두사가 붙은 식별자 (`file://`, `notes://` 등) |
| `resources/subscribe` | "변경 사항 감시" | 특정 URI에 대한 클라이언트 선택형 서버 푸시 업데이트 |
| `notifications/resources/updated` | "리소스 변경됨" | 구독한 리소스에 새 콘텐츠가 있음을 클라이언트에 알리는 신호 |
| 리소스 템플릿 | "매개변수화된 URI" | 호스트 선택기를 위한 완성 힌트가 포함된 URI 패턴 |
| 프롬프트 | "슬래시 명령어 템플릿" | 인수 슬롯이 있는 명명된 다중 메시지 템플릿 |
| 프롬프트 인수 | "템플릿 입력" | 호스트가 렌더링 전에 수집하는 타입이 지정된 매개변수 |
| `prompts/get` | "템플릿 렌더링" | 서버가 채워진 메시지 목록을 반환 |
| 콘텐츠 블록 | "타입이 지정된 청크" | `{type: text | image | resource | ui_resource}` |
| 슬래시 명령어 UX | "사용자 단축키" | 호스트가 `/`로 시작하는 명령으로 프롬프트를 표시 |

## 추가 자료

- [MCP — 개념: 리소스](https://modelcontextprotocol.io/docs/concepts/resources) — 리소스 URI, 구독, 템플릿
- [MCP — 개념: 프롬프트](https://modelcontextprotocol.io/docs/concepts/prompts) — 프롬프트 템플릿 및 슬래시 명령 통합
- [MCP — 서버 리소스 사양 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 전체 `resources/*` 메시지 참조
- [MCP — 서버 프롬프트 사양 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 전체 `prompts/*` 메시지 참조
- [MCP — 프로토콜 정보 사이트: 리소스](https://modelcontextprotocol.info/docs/concepts/resources/) — 공식 문서 확장 커뮤니티 가이드