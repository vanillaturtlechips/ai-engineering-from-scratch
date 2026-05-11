# 비동기 작업(SEP-1686) — 장기 실행 작업을 위한 "지금 호출, 나중에 가져오기"

> 실제 에이전트 작업은 수분에서 수 시간이 소요됩니다: CI 실행, 심층 연구 종합, 배치 내보내기 등. 동기식 도구 호출은 연결 끊김, 시간 초과 또는 UI 차단을 유발합니다. 2025-11-25에 병합된 SEP-1686은 작업(Tasks) 기본 요소를 추가합니다: 모든 요청은 작업으로 확장될 수 있으며, 결과는 나중에 가져오거나 상태 알림을 통해 스트리밍될 수 있습니다. 드리프트 위험 주의: 작업은 2026년 H1까지 실험적 단계입니다; SDK 표면은 여전히 사양 중심으로 설계 중입니다.

**유형:** 빌드  
**언어:** Python (표준 라이브러리, 비동기 작업 상태 머신)  
**사전 요구 사항:** 13단계 · 07 (MCP 서버), 13단계 · 09 (전송 계층)  
**소요 시간:** ~75분

## 학습 목표

- 도구 상태를 동기식에서 작업 증강(task-augmented)으로 승격할 시점을 식별한다(서버 측 작업 >30초).
- 작업 수명 주기 추적: `working` → `input_required` → `completed` / `failed` / `cancelled`.
- 충돌 시 진행 중인 작업이 손실되지 않도록 작업 상태를 지속(persist)한다.
- `tasks/status` 폴링 및 `tasks/result` 가져오기를 올바르게 수행한다.

## 문제

`generate_report` 도구는 수 분 동안 실행되는 추출 파이프라인을 실행합니다. 동기식 모델 하의 옵션:

1. 3분 동안 연결을 유지합니다. 원격 전송이 중단되고; 클라이언트가 타임아웃되며; UI가 멈춥니다.
2. 즉시 플레이스홀더(placeholder)를 반환합니다. 클라이언트가 사용자 정의 엔드포인트를 폴링(polling)해야 합니다. MCP(Multi-Client Protocol) 일관성을 깨뜨립니다.
3. 실행 후 결과를 반환하지 않습니다(Fire-and-forget).

모두 좋은 옵션이 아닙니다. SEP-1686은 네 번째 옵션인 **태스크 증강(task augmentation)**을 추가합니다. 모든 요청(일반적으로 `tools/call`)을 태스크로 태그할 수 있습니다. 서버는 즉시 태스크 ID를 반환합니다. 클라이언트는 `tasks/status`를 폴링하고 완료 시 `tasks/result`를 가져옵니다. 서버 측 상태는 재시작 후에도 유지됩니다.

## 개념

### 작업 확장(Task augmentation)

요청은 `params._meta.task.required: true` (또는 `optional: true`, 서버가 결정)를 설정하여 작업이 됩니다. 서버는 즉시 다음과 같이 응답합니다:

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl`은 서버가 상태를 유지할 것을 약속하는 시간입니다. ttl 이후에는 작업 결과가 폐기됩니다.

### 도구별 선택적 참여(Per-tool opt-in)

도구 주석은 작업 지원을 선언할 수 있습니다:

- `taskSupport: "forbidden"` — 이 도구는 항상 동기적으로 실행됩니다. 빠른 도구에 안전합니다.
- `taskSupport: "optional"` — 클라이언트가 작업 확장을 요청할 수 있습니다.
- `taskSupport: "required"` — 클라이언트는 반드시 작업 확장을 사용해야 합니다.

`generate_report` 도구는 `required`가 될 수 있습니다. `notes_search` 도구는 `forbidden`이 될 수 있습니다.

### 상태(States)

```
working  -> input_required -> working  (루프 via elicitation)
working  -> completed
working  -> failed
working  -> cancelled
```

상태 머신은 append-only입니다: `completed`, `failed`, 또는 `cancelled`가 되면 작업은 종료 상태가 됩니다.

### 메서드

- `tasks/status {taskId}` — 현재 상태와 진행률 힌트를 반환합니다.
- `tasks/result {taskId}` — 작업이 완료되지 않았으면 차단하거나 404를 반환합니다.
- `tasks/cancel {taskId}` — 멱등성(idempotent)을 가집니다. 종료 상태는 무시합니다.
- `tasks/list` — 선택 사항; 활성 및 최근 완료된 작업을 열거합니다.

### 스트리밍 상태 변경

서버가 지원하는 경우, 클라이언트는 상태 알림을 구독할 수 있습니다:

```
server -> notifications/tasks/updated {taskId, state, progress?}
```

폴링 대신 스트리밍을 사용하는 클라이언트는 더 나은 UX를 얻습니다. 폴링은 항상 최소한의 기능으로 지원됩니다.

### 영구 상태(Durable state)

명세서는 작업 지원을 선언한 서버가 상태를 영구히 저장할 것을 요구합니다. 충돌 시 ttl 내의 완료된 결과가 손실되어서는 안 됩니다. 저장소는 SQLite에서 Redis, 파일 시스템까지 다양합니다. Lesson 13 하네스는 파일 시스템을 사용합니다.

### 취소 의미론(Cancellation semantics)

`tasks/cancel`은 멱등성(idempotent)을 가집니다. 작업이 실행 중이면 서버는 중지를 시도합니다(실행기-협력적 취소 확인). 이미 종료 상태면 요청은 무시됩니다.

### 충돌 복구(Crash recovery)

서버 프로세스가 재시작할 때:

1. 모든 영구 작업 상태를 로드합니다.
2. 프로세스가 종료된 `working` 작업을 `failed` 상태로 표시하고 오류 `CRASH_RECOVERY`를 추가합니다.
3. `completed` / `failed` / `cancelled` 상태는 ttl 동안 보존합니다.

### 비동기 작업 및 샘플링

작업은 자체적으로 `sampling/createMessage`를 호출할 수 있습니다. 이는 장기 실행 연구 작업이 작동하는 방식입니다: 서버의 작업 스레드는 필요에 따라 클라이언트의 모델을 샘플링하고, 클라이언트 UI는 작업을 `working` 상태로 표시하며 주기적인 진행률 업데이트를 보여줍니다.

### 실험적 이유

SEP-1686은 2025-11-25에 출시되었지만, 광범위한 로드맵에서는 세 가지 미해결 문제를 지적하고 있습니다: 영구 구독 기본 요소, 하위 작업(부모-자식 작업 관계), 결과-TTL 표준화. 2026년까지 명세서가 진화할 것으로 예상됩니다. 프로덕션 코드는 일반적인 경우에만 작업을 안정적이라고 간주하고 하위 작업에 대한 향후 SDK 변경에 대비해야 합니다.

## 사용 방법

`code/main.py`는 영구 작업 저장소(파일 시스템 기반)와 백그라운드 스레드에서 실행되는 `generate_report` 도구를 구현합니다. 클라이언트는 도구를 호출하고 즉시 작업 ID를 받은 후, 작업자가 진행 상황을 업데이트하는 동안 `tasks/status`를 폴링하고 완료 시 `tasks/result`를 가져옵니다. 취소 기능이 작동하며, 작업자 스레드를 종료하고 상태를 다시 로드하여 충돌 복구를 시뮬레이션합니다.

확인할 사항:

- 작업 상태 JSON이 `/tmp/lesson-13-tasks/<id>.json`에 지속됩니다.
- 작업자 스레드가 `progress` 필드를 업데이트하며, 폴링 시 진행 상황이 표시됩니다.
- 클라이언트 측에서 취소 요청 시 이벤트가 설정되고, 작업자가 이를 확인하여 조기 종료합니다.
- "충돌" 시 상태 재로드가 진행 중인 작업을 `failed` 상태로 표시하고 `CRASH_RECOVERY` 원인을 기록합니다.

## Ship It

이 레슨은 `outputs/skill-task-store-designer.md`를 생성합니다. 장기간 실행되는 도구(연구, 빌드, 내보내기)가 주어졌을 때, 스킬은 작업 저장소(상태 형태, TTL, 내구성)를 설계하고, 적절한 `taskSupport` 플래그를 선택하며, 진행 상태 알림을 스케치합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. `generate_report` 작업을 시작하고, 상태를 폴링한 다음 결과를 가져오세요.

2. 실행 중에 `tasks/cancel` 호출을 추가하세요. 작업자가 이를 준수하는지 확인하고 상태가 `cancelled`로 변경되는지 검증하세요.

3. 충돌 복구 시뮬레이션: 작업자 스레드를 종료하고, 로더를 재시작한 후 `CRASH_RECOVERY` 실패 모드를 관찰하세요.

4. 저장소를 SQLite로 확장하세요. 지속성 이점은 동일하지만, 쿼리 옵션이 확장됩니다(예: 세션 X의 모든 작업 목록 조회).

5. 2026년 MCP 로드맵 게시물을 읽고, 향후 1년 내 SDK API 설계에 가장 큰 영향을 미칠 가능성이 높은 작업 관련 미해결 이슈를 식별하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Task | "장기 실행 도구 호출" | 비동기 실행을 위해 `_meta.task`로 보강된 요청 |
| SEP-1686 | "태스크 명세" | 2025-11-25에 태스크를 추가한 명세 진화 제안(Spec Evolution Proposal) |
| `_meta.task` | "태스크 봉투" | id, 상태(state), ttl을 포함하는 요청별 메타데이터 |
| taskSupport | "도구 플래그" | 도구별 `forbidden` / `optional` / `required` |
| `tasks/status` | "폴링 메서드" | 현재 상태와 선택적 진행률 힌트를 조회 |
| `tasks/result` | "결과 가져오기" | 완료된 페이로드 반환 또는 아직 완료되지 않았을 경우 404 반환 |
| `tasks/cancel` | "중지 요청" | 멱등성(idempotent)을 가진 취소 요청 |
| ttl | "보존 예산" | 서버가 태스크 상태를 유지하기로 약속한 밀리초 단위 시간 |
| `notifications/tasks/updated` | "상태 푸시" | 서버 발신의 상태 변경 이벤트 |
| Durable store | "크래시 안전 상태" | 파일시스템 / SQLite / Redis 지속성(persistence) 계층 |

## 추가 자료

- [MCP — GitHub SEP-1686 이슈](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 제안 및 전체 논의 원문
- [WorkOS — AI 에이전트 워크플로우를 위한 MCP 비동기 작업](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 설계 근거와 함께 하는 워크스루
- [DeepWiki — MCP 작업 시스템과 비동기 연산](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — 메커니즘과 상태 머신
- [FastMCP — 작업(Tasks)](https://gofastmcp.com/servers/tasks) — SDK 수준 작업 구현 패턴
- [MCP 블로그 — 2026 로드맵](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 하위 작업 포함 오픈 이슈 및 2026 우선순위