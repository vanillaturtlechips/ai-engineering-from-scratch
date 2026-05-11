# Claude 에이전트 SDK: 서브에이전트 및 세션 저장소

> Claude 에이전트 SDK는 Claude 코드 하네스의 라이브러리 형태입니다. 내장 도구, 컨텍스트 격리를 위한 서브에이전트, 훅, W3C 추적 전파, 세션 저장소 패리티를 제공합니다. Claude 관리형 에이전트는 장시간 비동기 작업을 위한 호스팅 대안입니다.

**유형:** 학습 + 구축  
**언어:** Python (표준 라이브러리)  
**사전 요구 사항:** 14단계 · 01 (에이전트 루프), 14단계 · 10 (스킬 라이브러리)  
**소요 시간:** ~75분

## 학습 목표

- Anthropic Client SDK(원시 API)와 Claude Agent SDK(하네스 형태)의 차이점을 설명합니다.
- 서브에이전트(병렬화 및 컨텍스트 격리)와 사용 시기를 설명합니다.
- Python SDK의 세션 저장소 표면(`append`, `load`, `list_sessions`, `delete`, `list_subkeys`)과 `--session-mirror`의 역할을 설명합니다.
- 내장 도구, 격리된 컨텍스트를 사용한 서브에이전트 생성, 라이프사이클 훅, 세션 저장소를 갖춘 표준 라이브러리 하네스를 구현합니다.

## 문제 정의

원시 LLM API는 단일 왕복(round-trip)만 제공합니다. 반면 프로덕션 환경에서는 에이전트 운영을 위해 도구 실행, MCP 서버, 라이프사이클 훅, 서브에이전트 생성, 세션 지속성, 트레이스 전파 등이 필요합니다. Claude Agent SDK는 이러한 기능을 라이브러리 형태로 제공하며, Claude Code에서 사용하는 동일한 하네스를 커스텀 에이전트에 적용할 수 있도록 공개합니다.

## 개념

### 클라이언트 SDK vs 에이전트 SDK

- **클라이언트 SDK (`anthropic`).** Raw Messages API. 루프, 도구, 상태를 직접 관리합니다.
- **에이전트 SDK (`claude-agent-sdk`).** 내장 도구 실행, MCP 연결, 훅, 서브에이전트 생성, 세션 저장소. Claude Code 루프를 라이브러리로 제공합니다.

### 내장 도구

SDK는 10개 이상의 도구를 기본 제공합니다: 파일 읽기/쓰기, 셸, grep, glob, 웹 가져오기 등. 사용자 정의 도구는 표준 도구-스키마 인터페이스를 통해 등록됩니다.

### 서브에이전트

Anthropic에서 문서화한 두 가지 목적:

1. **병렬화.** 독립적인 작업을 동시에 실행합니다. "이 20개 모듈에 대한 테스트 파일을 각각 찾아라"는 20개의 병렬 서브에이전트 작업입니다.
2. **컨텍스트 격리.** 서브에이전트는 자체 컨텍스트 윈도우를 사용하며, 결과만 오케스트레이터로 반환됩니다. 오케스트레이터의 예산이 보존됩니다.

Python SDK 최신 추가 기능: `list_subagents()`, `get_subagent_messages()`로 서브에이전트 대화 기록을 읽을 수 있습니다.

### 세션 저장소

TypeScript와 프로토콜 호환:

- `append(session_id, message)` — 대화 추가.
- `load(session_id)` — 대화 복원.
- `list_sessions()` — 세션 목록.
- `delete(session_id)` — 서브에이전트 세션에 대한 캐스케이드 삭제 포함.
- `list_subkeys(session_id)` — 서브에이전트 키 목록.

`--session-mirror` (CLI 플래그)는 스트리밍 중 대화 기록을 외부 파일에 미러링하여 디버깅을 지원합니다.

### 훅

등록할 수 있는 라이프사이클 훅:

- `PreToolUse`, `PostToolUse` — 도구 호출 제어 또는 감사.
- `SessionStart`, `SessionEnd` — 설정 및 정리.
- `UserPromptSubmit` — 모델이 사용자 입력을 처리하기 전에 동작.
- `PreCompact` — 컨텍스트 압축 전 실행.
- `Stop` — 에이전트 종료 시 정리.
- `Notification` — 사이드 채널 알림.

훅은 프로-워크플로우(Phase 14 커리큘럼 참조) 및 유사 시스템이 횡단적 동작을 추가하는 방식입니다.

### W3C 트레이스 컨텍스트

호출자에서 활성화된 OTel 스팬은 W3C 트레이스 컨텍스트 헤더를 통해 CLI 하위 프로세스로 전파됩니다. 전체 다중 프로세스 트레이스가 백엔드에서 하나의 트레이스로 표시됩니다.

### Claude 관리형 에이전트

호스팅 대안(베타 헤더 `managed-agents-2026-04-01`). 장기 비동기 작업, 내장 프롬프트 캐싱, 내장 압축. 관리형 인프라와 제어 권한을 교환합니다.

### 이 패턴이 실패하는 경우

- **서브에이전트 과생성.** 100개의 작은 작업을 위해 100개의 서브에이전트를 생성. 오버헤드가 지배적입니다. 대신 배치 처리.
- **훅 증가.** 모든 팀이 훅을 추가하면 시작 시간이 급증. 분기별로 훅 검토.
- **세션 과다.** 세션이 누적되고 크기가 증가. `list_sessions` + 만료 정책 사용.

## 빌드하기

`code/main.py`는 stdlib에 SDK 형태를 구현합니다:

- `Tool`, `ToolRegistry`에 내장된 `read_file`, `write_file`, `list_dir`.
- `Subagent` — 개인 컨텍스트, 격리된 실행, 결과 반환.
- `SessionStore` — 추가, 로드, 목록, 삭제, 하위 키 목록.
- `Hooks` — `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- 데모: 메인 에이전트가 3개의 서브에이전트를 병렬로 생성(각각 격리), 결과 집계, 세션 지속.

실행 방법:

```
python3 code/main.py
```

트레이스에는 서브에이전트 컨텍스트 격리(오케스트레이터 컨텍스트 크기 제한 유지), 훅 실행, 세션 지속성이 표시됩니다.

## 사용 방법

- **Claude 에이전트 SDK**는 Claude-first 제품에서 Claude 코드 하네스 형태를 원하는 경우 사용합니다.  
- **Claude 관리형 에이전트**는 호스팅되는 장기 비동기 작업을 위해 사용합니다.  
- **OpenAI 에이전트 SDK** (레슨 16)는 OpenAI-first 대응 제품을 위해 사용합니다.  
- **LangGraph + 커스텀 도구**는 그래프 형태의 상태 머신을 원하는 경우 사용합니다.

## Ship It

`outputs/skill-claude-agent-scaffold.md`는 서브에이전트(subagents), 훅(hooks), 세션 저장소(session store), MCP 서버 연결(MCP server attachment), W3C 추적 전파(W3C trace propagation)를 포함한 Claude Agent SDK 앱을 위한 스캐폴드(scaffold)를 제공합니다.

## 연습 문제

1. 20개의 작업을 5개의 병렬 하위 에이전트 그룹으로 배치 처리하는 하위 에이전트 생성기를 추가하세요. 작업당 1개 대비 오케스트레이터 컨텍스트 크기를 측정하세요.
2. `write_file` 호출을 분당 세션당 5회로 제한하는 `PreToolUse` 훅을 구현하세요. 동작을 추적하세요.
3. `list_subkeys`를 하위 에이전트 트리 렌더링에 연결하세요. 깊은 중첩은 어떤 모습인가요?
4. 이 예제를 실제 `claude-agent-sdk` Python 패키지로 이식하세요. 도구 등록에 어떤 변화가 발생하나요?
5. Claude 관리형 에이전트 문서를 읽으세요. 자체 호스팅에서 관리형으로 언제 전환해야 할까요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Agent SDK | "Claude Code as a library" | 하니스 형태: 도구(tools), MCP, 훅(hooks), 서브에이전트(subagents), 세션 저장소(session store) |
| Subagent | "Child agent" | 별도의 컨텍스트, 자체 예산; 결과가 상위로 전달됨 |
| Session store | "Conversation DB" | 서브에이전트 캐스케이드와 함께 턴(persist, load, list, delete) 관리 |
| Hook | "Lifecycle callback" | 도구/세션/프롬프트 제출 전/후, 압축(compact), 중지(stop) 시 콜백 |
| W3C trace context | "Cross-process trace" | 부모 스팬(span)이 CLI 하위 프로세스로 전파 |
| Managed Agents | "Hosted harness" | Anthropic 호스팅 장기 실행 비동기 작업 |
| `--session-mirror` | "Transcript mirror" | 세션 턴을 스트리밍되는 대로 외부 파일에 기록 |
| MCP server | "Tool surface" | 에이전트에 연결된 외부 도구/리소스 소스 |

## 추가 자료

- [Claude 에이전트 SDK 개요](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude Code의 라이브러리 형태
- [Anthropic, Claude 에이전트 SDK로 에이전트 구축하기](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 프로덕션 패턴
- [Claude 관리형 에이전트 개요](https://platform.claude.com/docs/en/managed-agents/overview) — 호스팅 대안
- [OpenAI 에이전트 SDK](https://openai.github.io/openai-agents-python/) — 대응 제품