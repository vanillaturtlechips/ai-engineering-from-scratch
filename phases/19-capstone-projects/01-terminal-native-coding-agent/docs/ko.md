# 캡스톤 01 — 터미널 네이티브 코딩 에이전트

> 2026년까지 코딩 에이전트의 형태가 정착됩니다. TUI 하니스, 상태 유지 계획, 샌드박싱된 도구 표면, 계획-실행-관찰-복구 루프. Claude Code, Cursor 3, OpenCode는 50피트 거리에서 모두 동일하게 보입니다. 이 캡스톤은 CLI 입력에서 풀 리퀘스트 출력까지 종단 간 시스템을 구축하고, SWE-bench Pro에서 mini-swe-agent 및 Live-SWE-agent와 비교 측정하도록 요청합니다. 모델 호출이 아닌 도구 루프, 샌드박스, 50턴 실행의 비용 한계가 진정한 난제임을 배우게 될 것입니다.

**유형:** 캡스톤  
**언어:** TypeScript / Bun(하니스), Python(평가 스크립트)  
**선수 조건:** 11단계(LLM 엔지니어링), 13단계(도구 및 프로토콜), 14단계(에이전트), 15단계(자율 시스템), 17단계(인프라)  
**연습 단계:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18  
**소요 시간:** 35시간

## 문제

2026년에 코딩 에이전트(coding agent)가 지배적인 AI 응용 분야 카테고리가 되었습니다. Claude Code(Anthropic), Cursor 3 with Composer 2 및 Agent Tabs(Cursor), Amp(Sourcegraph), OpenCode(112k 스타), Factory Droids, Google Jules는 모두 동일한 아키텍처의 변형을 제공합니다: 터미널 하니스(terminal harness), 권한 부여된 도구 표면(permissioned tool surface), 샌드박스(sandbox), 그리고 프론티어 모델(frontier model)을 중심으로 구축된 계획-실행-관찰(plan-act-observe) 루프. 프론티어는 좁지만 — Live-SWE-agent는 Opus 4.5로 SWE-bench Verified에서 79.2%를 달성했습니다 — 엔지니어링 기술은 광범위합니다. 대부분의 실패 모드는 모델 오류가 아닙니다. 도구 루프 불안정성(tool-loop instability), 컨텍스트 오염(context poisoning), 토큰 비용 폭주(runaway token cost), 파괴적인 파일 시스템 작업(destructive filesystem operations)이 주요 원인입니다.

이러한 에이전트를 외부에서 추론할 수 없습니다. 직접 구축한 후, ripgrep이 8MB의 일치 결과를 반환할 때 47번째 턴에서 루프가 충돌하는 것을 관찰하고, 잘림(truncation) 계층을 재구축해야 합니다. 이것이 이 캡스톤 프로젝트의 목적입니다.

## 개념

하네스에는 네 가지 표면이 있습니다. **Plan**은 모델이 매 턴마다 재작성하는 TodoWrite 스타일의 상태 객체를 유지합니다. **Act**는 도구 호출(읽기, 편집, 실행, 검색, git)을 디스패치합니다. **Observe**는 stdout/stderr/종료 코드를 캡처하고, 잘라낸 후 요약본을 다시 제공합니다. **Recover**는 컨텍스트 창을 날리거나 무한 루프를 발생시키지 않으면서 도구 오류를 처리합니다. 2026 형태에는 **훅(hooks)**이라는 한 가지 요소가 추가됩니다. `PreToolUse`, `PostToolUse`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`, `Notification`, `Stop`, `PreCompact` — 운영자가 정책, 원격 분석 및 가드레일을 주입하는 구성 가능한 확장 지점입니다.

샌드박스는 E2B 또는 Daytona입니다. 각 작업은 읽기-쓰기 권한이 있는 git 작업 트리가 마운트된 새로운 개발 컨테이너에서 실행됩니다. 하네스는 호스트 파일 시스템을 절대 건드리지 않습니다. 작업 트리는 성공 또는 실패 시 철거됩니다. 비용 제어는 세 계층에서 강제됩니다: 턴당 토큰 상한, 세션당 달러 예산, 그리고 하드 턴 제한(일반적으로 50). 관측 가능성 계층은 GenAI 시맨틱 규약을 갖춘 OpenTelemetry 스팬으로, 자체 호스팅된 Langfuse로 전송됩니다.

## 아키텍처

```
  사용자 CLI  ->  하니스 (Bun + Ink TUI)
                  |
                  v
           계획/실행/관찰 루프  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (OpenRouter 경유, 모델-불가지론적)
                  v
           툴 디스패처 (MCP StreamableHTTP 클라이언트)
                  |
     +------------+------------+----------+
     v            v            v          v
  읽기/편집    ripgrep     tree-sitter   git/실행
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona 샌드박스  (작업 트리 격리)
                  |
                  v
           훅: 사전/사후, 세션, 프롬프트, 컴팩트
                  |
                  v
           OpenTelemetry -> Langfuse (스팬, 토큰, $)
                  |
                  v
           GitHub 앱을 통한 PR
```

## 스택

- **하네스 런타임**: Bun 1.2 + Ink 5 (React-in-terminal)
- **모델 접근**: OpenRouter 통합 API와 Claude Sonnet 4.7, GPT-5.4-Codex, Gemini 3 Pro, Opus 4.5 (가장 어려운 작업용)
- **도구 전송**: 모델 컨텍스트 프로토콜 스트리밍 HTTP (MCP 2026 개정판)
- **샌드박스**: E2B 샌드박스 (JS SDK) 또는 Daytona 개발 컨테이너
- **코드 검색**: ripgrep 서브프로세스, 17개 언어용 tree-sitter 파서 (사전 컴파일됨)
- **격리**: 작업별 `git worktree add`, 성공/실패 시 정리
- **평가 하네스**: SWE-bench Pro (검증된 부분 집합) + Terminal-Bench 2.0 + 자체 30개 작업 홀드아웃
- **관측 가능성**: OpenTelemetry SDK와 `gen_ai.*` semconv → 자체 호스팅 Langfuse
- **PR 게시**: 세분화된 토큰을 가진 GitHub 앱, 대상 리포지토리로 범위 제한

## 빌드하기

1. **TUI 및 명령 루프.** Ink로 Bun 프로젝트 스캐폴드를 구축합니다. `agent run <repo> "<task>"`를 수용합니다. 분할 뷰를 출력합니다: 계획 패널(상단), 도구 호출 스트림(중간), 토큰 예산(하단). Ctrl-C로 취소 시 `SessionEnd` 훅을 실행한 후 종료합니다.

2. **계획 상태.** 타입이 지정된 TodoWrite 스키마(미결/진행 중/완료 항목과 메모)를 정의합니다. 모델은 매 턴마다 전체 상태를 도구 호출로 재작성합니다 — 증분 변경을 허용하지 않습니다. 충돌 시 재개할 수 있도록 계획을 `.agent/state.json`에 지속합니다.

3. **도구 표면.** 6가지 도구를 정의합니다: `read_file`, `edit_file`(diff 미리보기 포함), `ripgrep`, `tree_sitter_symbols`, `run_shell`(타임아웃 포함), `git`(상태/차이/커밋/푸시). MCP StreamableHTTP를 통해 노출하여 하네스가 전송 계층과 무관하도록 합니다. 모든 도구는 잘린 출력을 반환합니다(호출당 4k 토큰으로 제한).

4. **샌드박스 래핑.** 각 작업은 E2B 샌드박스를 생성합니다. `git worktree add -b agent/$TASK_ID`로 새 브랜치를 추가합니다. 모든 도구 호출은 샌드박스 내에서 실행됩니다. 호스트 파일 시스템은 접근할 수 없습니다.

5. **훅.** 2026년 8가지 훅 유형을 모두 구현합니다. 최소 4개의 사용자 작성 훅을 연결합니다: (a) 작업 트리 외부 `rm -rf`를 차단하는 `PreToolUse` 파괴적 명령 가드, (b) `PostToolUse` 토큰 회계, (c) `SessionStart` 예산 초기화, (d) `Stop` 최종 트레이스 번들 작성.

6. **평가 루프.** SWE-bench Pro Python의 30개 이슈 하위 집합을 복제합니다. 각 이슈에 대해 하네스를 실행합니다. mini-swe-agent(최소 기준선)와 pass@1, 작업당 턴 수, 작업당 $-비용을 비교합니다. 결과를 `eval/results.jsonl`에 기록합니다.

7. **비용 제어.** 하드 컷오프: 50턴, 200k 컨텍스트, 작업당 $5. `PreCompact` 훅은 150k 마크에서 이전 턴을 이전 상태 블록으로 요약하여 새로운 관측치를 위한 공간을 확보하면서 계획을 잃지 않습니다.

8. **PR 게시.** 성공 시 최종 단계는 `git push` + 계획과 diff 요약을 본문에 포함한 PR을 여는 GitHub API 호출입니다.

## 사용 방법

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[계획]  1 worker.rs 위치 확인 및 mutex 사용 목록 열거
        2 경합 상태의 공유 자원 식별
        3 수정안 제안, 테스트 검증
[도구]  ripgrep mutex.*lock -t rust           (44개 일치, 생략됨)
[도구]  read_file src/worker.rs 120..180
[도구]  edit_file src/worker.rs (+8 -3)
[도구]  run_shell cargo test worker::          (성공)
[계획]  1 완료 · 2 완료 · 3 완료
[완료]  PR 생성됨: #482   턴=9   토큰=38k   비용=$0.41
```

## Ship It

결과물은 `outputs/skill-terminal-coding-agent.md`에 있습니다. 리포지토리 경로와 작업 설명을 입력으로 받아 샌드박스 내에서 전체 계획-실행-관찰 루프를 실행하고 PR URL과 트레이스 번들을 반환합니다. 이 캡스톤의 평가 기준은 다음과 같습니다:

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 vs 베이스라인 | 30개의 일치하는 Python 작업에서 귀하의 하니스 vs mini-swe-agent |
| 20 | 아키텍처 명확성 | 계획/실행/관찰 분리, 훅 표면, 도구 스키마 — Live-SWE-agent 레이아웃과 대조 검토 |
| 20 | 안전성 | 샌드박스 탈출 테스트, 권한 프롬프트, 파괴적 명령어 가드 통과 레드팀 평가 |
| 20 | 관측 가능성 | 트레이스 완성도(도구 호출 100% 스팬), 턴당 토큰 회계 |
| 15 | 개발자 UX | 콜드 스타트 < 2초, 충돌 복구 시 계획 재개, Ctrl-C로 도구 실행 중단 시 깔끔하게 취소 |
| **100** | | |

## 연습 문제

1. 백엔드 모델을 Claude Sonnet 4.7에서 vLLM에서 제공되는 Qwen3-Coder-30B로 교체하세요. pass@1과 $-per-task를 비교하세요. 오픈 모델이 성능이 떨어지는 부분을 보고하세요.

2. PR 게시 전 diff를 읽고 수정 루프를 요청할 수 있는 `reviewer` 서브 에이전트를 추가하세요. 거짓 긍정 리뷰가 SWE-bench 통과율을 단일 에이전트 기준보다 낮추는지 측정하세요 (힌트: 일반적으로 그렇습니다).

3. 샌드박스 스트레스 테스트: 외부 URL에 `curl`을 시도하는 작업과 작업 디렉터리 외부에 쓰기를 시도하는 작업을 작성하세요. PreToolUse 훅에 의해 둘 다 차단되는지 확인하세요. 시도 기록을 남기세요.

4. 더 작은 모델(Haiku 4.5)로 `PreCompact` 요약 기능을 구현하세요. 3배 압축 시 계획 충실도가 얼마나 손실되는지 측정하세요.

5. MCP StreamableHTTP 전송 계층을 stdio로 교체하세요. 콜드 스타트 및 호출당 지연 시간을 벤치마킹하세요. 로컬 전용 사용 시 어떤 것을 선택할지 결정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 하니스(Harness) | "에이전트 루프" | 모델을 둘러싼 코드로, 도구를 디스패치하고 계획 상태를 유지하며 예산을 강제 적용 |
| 훅(Hook) | "에이전트 이벤트 리스너" | 하니스가 8가지 라이프사이클 이벤트 중 하나에서 실행하는 사용자 작성 스크립트 |
| 워크트리(Worktree) | "Git 샌드박스" | 별도의 경로에 있는 연결된 Git 체크아웃; 메인 클론을 건드리지 않고 폐기 가능 |
| 투두라이트(TodoWrite) | "계획 상태" | 모델이 매 턴마다 재작성하는 보류/진행 중/완료 항목의 타입이 지정된 목록 |
| 스트리밍 HTTP(StreamableHTTP) | "MCP 전송" | 2026 MCP 개정판: 양방향 스트리밍이 가능한 장기 HTTP 연결; SSE 대체 |
| 토큰 상한(Token ceiling) | "컨텍스트 예산" | 턴당 또는 세션당 입력+출력 토큰 상한; 압축 또는 종료 트리거 |
| 패스앳원(pass@1) | "단일 시도 통과율" | SWE-bench 작업 중 재시도나 테스트셋 피킹 없이 첫 실행에서 해결된 비율

## 추가 자료

- [Claude Code 문서](https://docs.anthropic.com/en/docs/claude-code) — Anthropic의 참조 하니스
- [Cursor 3 변경 로그](https://cursor.com/changelog) — 에이전트 탭 및 컴포저 2 제품 노트
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 하니스 비교를 위한 최소 기준선
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — Opus 4.5로 SWE-bench 검증 79.2% 달성
- [OpenCode](https://opencode.ai) — 오픈 하니스, 112k 스타
- [SWE-bench Pro 리더보드](https://www.swebench.com) — 이 캡스톤 프로젝트의 평가 대상
- [모델 컨텍스트 프로토콜 2026 로드맵](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 스트림 가능한 HTTP, 기능 메타데이터
- [OpenTelemetry GenAI 시맨틱 규약](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 도구 호출 및 토큰 사용량을 위한 스팬 스키마