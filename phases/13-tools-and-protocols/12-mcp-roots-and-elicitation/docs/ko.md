# 루트와 엘리시테이션 — 범위 지정 및 실행 중 사용자 입력

> 하드코딩된 경로는 사용자가 다른 프로젝트를 여는 순간 깨집니다. 미리 채워진 도구 인수는 사용자가 불완전하게 지정할 때 실패합니다. 루트는 서버를 사용자 제어 URI 집합으로 범위 지정하며, 엘리시테이션은 도구 호출 중간에 일시 중지하여 양식이나 URL을 통해 구조화된 입력을 사용자에게 요청합니다. 두 가지 클라이언트 기본 기능, 두 가지 일반적인 MCP 실패 모드 해결 방법. SEP-1036 (URL 모드 엘리시테이션, 2025-11-25)은 2026년 H1까지 실험적입니다 — 의존하기 전에 SDK 버전을 확인하세요.

**유형:** 빌드  
**언어:** Python (표준 라이브러리, 루트 + 엘리시테이션 데모)  
**사전 요구 사항:** 13단계 · 07 (MCP 서버)  
**소요 시간:** ~45분

## 학습 목표

- `roots`를 선언하고 `notifications/roots/list_changed`에 응답하기.
- 서버 파일 작업을 선언된 루트 집합 내부의 URI로 제한하기.
- `elicitation/create`를 사용하여 도구 호출 중간에 사용자에게 확인 또는 구조화된 입력 요청하기.
- 폼 모드(form-mode)와 URL 모드(URL-mode) 중 선택 (후자는 실험적 기능; 드리프트 위험 주의).

## 문제

프로덕션 환경에서 MCP 서버가 겪는 두 가지 구체적인 실패 사례.

**잘못된 경로 가정.** 서버는 `~/notes`를 기준으로 작성되었습니다. `~/Documents/Notes`에 노트를 저장한 다른 사용자의 경우 도구 호출이 실패(파일을 찾을 수 없음)하거나 더 나쁜 경우 잘못된 위치에 작성됩니다.

**사용자가 알 수 있는 누락된 인수.** 사용자가 "오래된 TPS 보고서 노트 삭제"라고 요청합니다. 모델은 `notes_delete(title: "TPS report")`를 호출하지만 2023년, 2024년, 2025년 버전의 세 가지 일치하는 노트가 존재합니다. 도구가 추측할 수 없습니다. "모호함"으로 실패하는 것은 귀찮고, 세 개 모두에 실행하는 것은 치명적입니다.

Roots는 첫 번째 문제를 해결합니다: 클라이언트는 `initialize`에서 서버가 접근할 수 있는 URI 집합을 선언합니다. Elicitation은 두 번째 문제를 해결합니다: 서버는 도구 호출을 일시 중지하고 `elicitation/create`를 보내 사용자에게 어떤 노트를 선택할지 묻습니다.

## 개념

### 루트

클라이언트는 `initialize`에서 루트 목록을 선언합니다:

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

서버는 이후 `roots/list`를 호출할 수 있습니다:

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

서버는 루트를 경계로 취급해야 합니다: 루트 집합 외부의 파일 읽기/쓰기는 거부됩니다. 이는 클라이언트에서 강제하지 않지만(서버는 여전히 사용자가 신뢰하는 코드), 사양 준수 서버는 이를 준수합니다.

사용자가 루트를 추가하거나 제거할 때, 클라이언트는 `notifications/roots/list_changed`를 전송합니다. 서버는 `roots/list`를 다시 호출하고 경계를 업데이트합니다.

### 루트가 클라이언트 프리미티브인 이유

루트는 클라이언트의 동의 모델을 나타내기 때문에 클라이언트가 선언합니다. 사용자는 "이 두 디렉터리에 대한 접근 권한을 이 노트 서버에 부여하라"고 Claude Desktop에 지시합니다. 서버는 이 범위를 확장할 수 없습니다.

### 추출: 폼 모드 기본값

`elicitation/create`는 폼 스키마와 자연어 프롬프트를 받습니다:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "'TPS 보고서' 삭제? 여러 노트가 일치합니다. 하나를 선택하세요.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

클라이언트는 폼을 렌더링하고 사용자 답변을 수집한 후 다음을 반환합니다:

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

가능한 세 가지 액션: `accept`(사용자가 입력 완료), `decline`(사용자가 창 닫음), `cancel`(사용자가 전체 도구 호출 중단).

폼 스키마는 평탄합니다 — v1에서는 중첩 객체가 지원되지 않습니다. SDK는 일반적으로 단일 계층보다 복잡한 것을 거부합니다.

### 추출: URL 모드 (SEP-1036, 실험적)

2025-11-25에 새로 추가됨. 스키마 대신 서버가 URL을 전송합니다:

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "GitHub에 로그인",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

클라이언트는 브라우저에서 URL을 열고 완료를 기다린 후 사용자가 돌아오면 반환합니다. OAuth 흐름, 결제 승인, 문서 서명과 같이 폼이 부적합한 경우에 유용합니다.

드리프트 위험 주의: SEP-1036 응답 형식은 아직 확정되지 않았습니다. 일부 SDK는 콜백 URL을 반환하고, 다른 SDK는 완료 토큰을 반환합니다. 프로덕션에서 URL 모드를 사용하기 전에 SDK의 릴리스 노트를 읽으세요.

### 추출이 적합한 경우

- 파괴적 작업 전 사용자 확인(파괴적 힌트 + 추출).
- 모호성 해소(N개 일치 항목 중 하나 선택).
- 첫 실행 설정(API 키, 디렉터리, 환경 설정).
- OAuth 스타일 흐름(URL 모드).

### 추출이 부적합한 경우

- 모델이 일반 텍스트로 요청할 수 있는 도구의 필수 인수 채우기. 추출 대화상자 대신 일반 재프롬프트를 사용하세요.
- 고주파 호출. 추출은 대화를 중단시킵니다. 루프 내부에서 실행하지 마세요.
- 서버가 사후에 검증할 수 있는 모든 것. 검증 후 오류를 반환하고, 모델이 텍스트로 사용자에게 요청하도록 하세요.

### 인간-루프 브리지

추출과 샘플링을 함께 사용하면 MCP의 "인간-루프" 모델을 구현할 수 있습니다. 서버의 에이전트 루프는 사용자 입력(추출) 또는 모델 추론(샘플링)을 위해 일시 중지할 수 있습니다. 13단계·11은 샘플링을 다루었고, 이 레슨은 추출을 다룹니다. 둘을 결합하여 전체 루프 중간 제어를 구현하세요.

## 사용 방법

`code/main.py`는 다음 기능을 추가하여 노트 서버를 확장합니다:

- `roots/list` 응답으로, 서버가 `root-list-changed` 알림 후 재조회합니다.
- 여러 노트가 일치할 때 `elicitation/create`를 사용하여 명확화하는 `notes_delete` 도구.
- URL 모드 명확화를 사용하여 첫 실행 구성 페이지를 여는(모의) `notes_setup` 도구.
- 선언된 루트 외부의 URI에 대한 작업을 거부하는 경계 검사.

데모는 세 가지 시나리오를 실행합니다: 행복한 경로(하나 일치), 명확화(세 개 일치, 명확화 발동), 루트 외부 쓰기(거부됨).

## Ship It

이 레슨은 `outputs/skill-elicitation-form-designer.md`를 생성합니다. 사용자 확인 또는 명확화가 필요한 도구가 주어졌을 때, 스킬은 정보 추출 양식 스키마와 메시지 템플릿을 설계합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 모호성 해소 경로를 트리거하고, 시뮬레이션된 사용자 응답이 툴로 다시 라우팅되는지 확인하세요.

2. 매번 확인(파괴적 힌트)이 필요한 새로운 툴 `notes_archive`를 추가하세요. UX를 확인하세요: 모델이 텍스트로 다시 묻는 것과 비교했을 때 어떤 차이가 있나요?

3. 첫 실행 OAuth 흐름을 위한 URL-모드 확인 구현을 추가하세요. 드리프트 위험을 기록하고 SDK 버전 가드를 추가하세요.

4. `roots/list` 처리를 확장하세요: 알림이 도착하면 서버는 이제 범위를 벗어날 수 있는 열린 파일 핸들을 원자적으로 다시 읽고 재검사해야 합니다.

5. GitHub의 SEP-1036 이슈 토론 스레드를 읽으세요. 서버가 URL-모드 콜백을 처리하는 방식에 영향을 미치는 미해결 질문 하나를 식별하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Root | "동의 경계" | 클라이언트가 서버가 접근하도록 허용한 URI |
| `roots/list` | "서버가 범위 요청" | 클라이언트가 현재 루트 집합을 반환 |
| `notifications/roots/list_changed` | "사용자가 범위 변경" | 클라이언트가 루트 집합이 변경되었음을 신호 |
| Elicitation | "통화 중 사용자 요청" | 구조화된 사용자 입력을 위한 서버 시작 요청 |
| `elicitation/create` | "메서드" | Elicitation 요청을 위한 JSON-RPC 메서드 |
| Form mode | "스키마 기반 폼" | 클라이언트 UI에서 폼으로 렌더링되는 평면 JSON 스키마 |
| URL mode | "브라우저 리디렉션" | SEP-1036 실험 기능; URL을 열고 대기 |
| `accept` / `decline` / `cancel` | "사용자 응답 결과" | 서버가 처리하는 세 가지 분기 |
| Disambiguation | "하나 선택" | 도구가 N개의 후보를 가질 때 일반적인 Elicitation 사용 사례 |
| Flat form | "최상위 속성만" | Elicitation 스키마는 중첩 불가 |

## 추가 자료

- [MCP — 클라이언트 루트 사양](https://modelcontextprotocol.io/specification/draft/client/roots) — 표준 루트 참조
- [MCP — 클라이언트 추출 사양](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 표준 추출 참조
- [Cisco — MCP 추출, 구조화 콘텐츠, OAuth 개선 사항](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 추가 기능 설명
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL 모드 추출 제안 (실험적, 드리프트 위험)
- [The New Stack — 추출 기능이 AI 도구에 인간 개입 방식을 도입하는 방법](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX 워크스루