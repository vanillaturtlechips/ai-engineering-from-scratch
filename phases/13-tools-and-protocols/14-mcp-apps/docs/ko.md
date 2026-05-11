# MCP 앱 — `ui://`를 통한 대화형 UI 리소스

> 텍스트만 출력하는 도구 결과는 에이전트가 보여줄 수 있는 내용을 제한합니다. MCP 앱(SEP-1724, 2026년 1월 26일 공식 발표)은 도구가 Claude Desktop, ChatGPT, Cursor, Goose, VS Code에서 인라인으로 렌더링되는 샌드박스 처리된 대화형 HTML을 반환할 수 있게 합니다. 대시보드, 폼, 지도, 3D 장면 등 모든 것을 하나의 확장으로 구현할 수 있습니다. 이 강의에서는 `ui://` 리소스 체계, `text/html;profile=mcp-app` MIME 타입, iframe-sandbox postMessage 프로토콜, 그리고 서버가 HTML을 렌더링할 때 발생하는 보안 영역에 대해 설명합니다.

**유형:** 빌드  
**언어:** Python(표준 라이브러리, UI 리소스 발신기), HTML(샘플 앱)  
**선수 조건:** 13단계 · 07(MCP 서버), 13단계 · 10(리소스)  
**소요 시간:** ~75분

## 학습 목표

- 도구 호출에서 `ui://` 리소스를 반환하고 올바른 MIME 및 메타데이터를 설정합니다.
- `_meta.ui.resourceUri`, `_meta.ui.csp`, `_meta.ui.permissions`로 도구의 관련 UI를 선언합니다.
- UI-호스트 통신을 위한 iframe 샌드박스 `postMessage` JSON-RPC를 구현합니다.
- UI 기반 공격에 대응하는 CSP 및 권한 정책 기본값을 적용합니다.

## 문제

2025년 시대의 `visualize_timeline` 도구는 "다음은 시간순으로 정리된 14개의 노트입니다: ..."라는 문장을 반환할 수 있습니다. 이는 단순한 단락입니다. 사용자들은 실제로 인터랙티브한 타임라인을 원합니다. MCP Apps 이전에는 다음과 같은 옵션만 존재했습니다: 클라이언트별 위젯 API(Claude 아티팩트, OpenAI 커스텀 GPT HTML) 또는 UI가 전혀 없는 경우.

MCP Apps(SEP-1724, 2026년 1월 26일 출시)는 계약을 표준화했습니다. 도구 결과에는 `ui://...` URI를 가지며 MIME이 `text/html;profile=mcp-app`인 `resource`가 포함됩니다. 호스트는 제한된 CSP(Content Security Policy)와 명시적 권한이 없는 한 네트워크 접근 없이 샌드박싱된 iframe에 이를 렌더링합니다. iframe 내부의 UI는 작은 postMessage JSON-RPC 방언을 통해 호스트에 메시지를 게시합니다.

모든 호환 클라이언트(Claude Desktop, ChatGPT, Goose, VS Code)는 동일한 `ui://` 리소스를 동일한 방식으로 렌더링합니다. 하나의 서버, 하나의 HTML 번들, 범용 UI.

## 개념

### `ui://` 리소스 스킴

도구가 반환하는 내용:

```json
{
  "content": [
    {"type": "text", "text": "여기 노트 타임라인이 있습니다:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

호스트는 `ui://notes/timeline` URI에 대해 `resources/read`를 호출하고 다음 응답을 받습니다:

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### iframe 샌드박스

호스트는 다음 설정으로 샌드박스 처리된 `<iframe>` 내부에 HTML을 렌더링합니다:

- `sandbox="allow-scripts allow-same-origin"` (또는 서버 선언에 따라 더 엄격한 설정)
- 서버 선언 CSP가 응답 헤더를 통해 적용됩니다.
- 호스트 오리진의 쿠키, localStorage 없음.
- 네트워크 접근은 CSP의 `connectSrc`로 제한됩니다.

### postMessage 프로토콜

iframe은 `window.postMessage`를 통해 호스트와 통신합니다. JSON-RPC 2.0의 간단한 변형:

항상 `targetOrigin`을 피어의 정확한 오리진으로 고정하고, 수신 측에서는 `event.origin`을 허용 목록과 비교한 후 페이로드를 처리해야 합니다. 이 채널의 양쪽에서 `"*"`는 절대 사용하지 마세요 — 본문에는 도구 호출과 리소스 읽기가 포함됩니다.

```js
// iframe → 호스트 (호스트 오리진으로 고정)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// 호스트 → iframe (iframe 오리진으로 고정)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// 양쪽 수신기
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // event.data 처리 안전
});
```

UI가 호출할 수 있는 호스트 측 메서드:

- `host.callTool(name, arguments)` — 서버 도구를 실행합니다.
- `host.readResource(uri)` — MCP 리소스를 읽습니다.
- `host.getPrompt(name, arguments)` — 프롬프트 템플릿을 가져옵니다.
- `host.close()` — UI를 닫습니다.

모든 호출은 여전히 MCP 프로토콜을 통과하며 서버의 권한을 상속받습니다.

### 권한

`_meta.ui.permissions` 목록은 추가 기능을 요청합니다:

- `camera` — 사용자 카메라 접근 (문서 스캔 UI에 사용).
- `microphone` — 음성 입력.
- `geolocation` — 위치 정보.
- `network:*` — `connectSrc` 단독 허용보다 넓은 네트워크 접근.

각 권한은 UI 렌더링 전에 사용자가 보는 프롬프트입니다.

### 보안 위험

iframe 내 HTML은 여전히 HTML입니다. 새로운 공격 표면:

- **UI를 통한 프롬프트 인젝션.** 악성 서버 UI는 시스템 메시지처럼 보이는 텍스트를 표시하고 사용자를 속일 수 있습니다. 호스트 렌더링은 서버 UI와 호스트 UI를 시각적으로 구분해야 합니다.
- **`connectSrc`를 통한 데이터 유출.** CSP가 `connect-src: *`를 허용하면 UI가 데이터를 아무 곳으로나 보낼 수 있습니다. 기본값은 엄격해야 합니다.
- **클릭재킹.** UI가 호스트 크롬을 오버레이합니다. 호스트는 z-index 조작을 방지하고 불투명도 규칙을 강제해야 합니다.
- **포커스 탈취.** UI가 키보드 포커스를 가져가고 다음 메시지를 캡처합니다. 호스트는 이를 가로채야 합니다.

13단계 · 15단계에서 MCP 보안의 일부로 자세히 다룹니다. 이 레슨에서는 소개만 합니다.

### `ui/initialize` 핸드셰이크

iframe 로드 후 postMessage로 `ui/initialize`를 전송합니다:

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

호스트는 기능과 세션 토큰으로 응답합니다. UI는 이후 모든 호스트 호출에 세션 토큰을 사용합니다.

### AppRenderer / AppFrame SDK 기본 요소

ext-apps SDK는 두 가지 편의 기능을 노출합니다:

- `AppRenderer` (서버 측) — React / Vue / Solid 컴포넌트를 래핑하고 올바른 MIME 및 메타데이터가 있는 `ui://` 리소스를 방출합니다.
- `AppFrame` (클라이언트 측) — 리소스를 수신하고 iframe을 마운트하며 postMessage를 중개합니다.

이들을 사용하거나 HTML과 JSON-RPC를 직접 구현할 수 있습니다.

### 생태계 현황

MCP Apps는 2026년 1월 26일에 출시되었습니다. 2026년 4월 기준 클라이언트 지원 현황:

- **Claude Desktop.** 2026년 1월부터 전체 지원.
- **ChatGPT.** Apps SDK를 통한 전체 지원 (동일한 기본 MCP Apps 프로토콜).
- **Cursor.** 베타; 설정에서 활성화.
- **VS Code.** 인사이더 빌드만.
- **Goose.** 전체 지원.
- **Zed, Windsurf.** 로드맵에 있음.

프로덕션 서버: 대시보드, 지도 시각화, 데이터 테이블, 차트 빌더, 샌드박스 IDE 미리보기.

## 사용 방법

`code/main.py`는 `visualize_timeline` 도구를 추가하여 노트 서버를 확장하며, 이 도구는 `ui://notes/timeline` 리소스를 반환합니다. 또한 해당 URI에 대한 `resources/read` 핸들러는 작지만 완전한 HTML 번들을 반환하며, 여기에는 SVG 타임라인이 포함됩니다. HTML은 stdlib 템플릿으로 작성되었으며 — 빌드 시스템이 필요 없습니다. `postMessage`는 stdlib이 브라우저를 구동할 수 없기 때문에 JS 주석에 개략적으로 설명되어 있습니다.

확인할 사항:

- 도구 응답의 `_meta.ui`에는 `resourceUri`, `CSP`, `permissions`가 포함됩니다.
- HTML은 네트워크 접근 없이 렌더링됩니다. 모든 데이터는 인라인 처리됩니다.
- JS는 `window.parent.postMessage`를 통해 `host.callTool`을 호출합니다(이 stdlib 데모에서는 문서화되었지만 비활성 상태입니다).

## Ship It

이 레슨은 `outputs/skill-mcp-apps-spec.md`를 생성합니다. 대화형 UI의 이점을 얻을 수 있는 도구가 주어졌을 때, 스킬은 MCP Apps 계약 전체를 생성합니다: `ui://` URI, CSP(Content Security Policy), 권한(permissions), postMessage 진입점(entrypoints), 그리고 보안 체크리스트(security checklist)를 포함합니다.

## 연습 문제

1. `code/main.py`를 실행하고 생성된 HTML을 검사하세요. 브라우저에서 HTML을 직접 열어 SVG가 렌더링되는지 확인하세요. 그런 다음 UI가 `host.callTool("notes_update", ...)`를 호출하는 데 사용할 `postMessage` 계약을 스케치해 보세요.

2. CSP(콘텐츠 보안 정책)를 강화하세요: `'unsafe-inline'`을 제거하고 nonce 기반 스크립트 정책을 사용하세요. HTML 생성 코드에서 어떤 변경이 필요한가요?

3. 두 번째 UI 리소스 `ui://notes/editor`를 추가하세요. 이 리소스는 노트를 인라인으로 편집할 수 있는 폼을 제공합니다. 사용자가 제출하면 iframe이 `host.callTool("notes_update", ...)`를 호출해야 합니다.

4. UI의 공격 표면을 감사하세요. 악성 서버가 콘텐츠를 주입할 수 있는 위치는 어디인가요? iframe 샌드박스는 무엇을 방어하고 무엇을 방어하지 못하나요?

5. SEP-1724 사양을 읽고 이 간단한 구현에서 사용하지 않는 MCP Apps SDK의 기능 하나를 식별하세요. (힌트: 컴포넌트 수준 상태 동기화)

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| MCP Apps | "인터랙티브 UI 리소스" | 2026-01-26에 출시된 SEP-1724 확장 |
| `ui://` | "앱 URI 체계" | UI 번들을 위한 리소스 체계 |
| `text/html;profile=mcp-app` | "MIME" | MCP 앱 HTML의 콘텐츠 유형 |
| Iframe sandbox | "렌더링 컨테이너" | CSP 및 권한으로 UI를 브라우저에서 샌드박싱 |
| postMessage JSON-RPC | "UI-호스트 간 통신" | 호스트 호출을 위한 postMessage 기반 JSON-RPC 변형 |
| `_meta.ui` | "툴-UI 바인딩" | 툴 결과와 UI 리소스를 연결하는 메타데이터 |
| CSP | "Content-Security-Policy" | 스크립트, 네트워크, 스타일에 허용되는 소스 선언 |
| AppRenderer | "서버 SDK 기본 요소" | 프레임워크 컴포넌트를 `ui://` 리소스로 변환 |
| AppFrame | "클라이언트 SDK 기본 요소" | postMessage를 중재하는 iframe 마운트 도우미 |
| `ui/initialize` | "핸드셰이크" | UI에서 호스트로 보내는 첫 번째 postMessage |

## 추가 자료

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — 참조 구현 및 SDK
- [MCP Apps 사양 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 공식 사양 문서
- [MCP — Apps 확장 개요](https://modelcontextprotocol.io/extensions/apps/overview) — 고수준 문서
- [MCP 블로그 — MCP Apps 출시](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026년 1월 출시 공지
- [MCP Apps API 참조](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 스타일 SDK 참조