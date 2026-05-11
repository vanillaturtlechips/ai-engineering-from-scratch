# OpenAI Agents SDK: 핸드오프, 가드레일, 트레이싱

> OpenAI Agents SDK는 Responses API 위에 구축된 경량 멀티 에이전트 프레임워크입니다. 5가지 기본 요소: 에이전트(Agent), 핸드오프(Handoff), 가드레일(Guardrail), 세션(Session), 트레이싱(Tracing). 핸드오프는 `transfer_to_<agent>`로 명명된 도구입니다. 가드레일은 입력 또는 출력 시 트리거됩니다. 트레이싱은 기본적으로 활성화되어 있습니다.

**유형:** 학습 + 구축  
**언어:** Python (표준 라이브러리)  
**선수 지식:** 14단계 · 01 (에이전트 루프), 14단계 · 06 (툴 사용)  
**소요 시간:** ~75분

## 학습 목표

- OpenAI Agents SDK의 5가지 기본 요소(primitives)를 나열할 수 있다.
- 핸드오프(handoffs)에 대해 설명할 수 있다: 왜 도구(tool)로 모델링되는지, 모델이 보는 이름 형태(name shape)는 무엇인지, 컨텍스트 전달 방식은 어떻게 이루어지는지.
- 입력 가드레일(input guardrails), 출력 가드레일(output guardrails), 도구 가드레일(tool guardrails)을 구분하고, `run_in_parallel`과 블로킹(blocking) 모드의 차이점을 설명할 수 있다.
- 핸드오프 + 가드레일 + 스팬 스타일(span-style) 추적이 포함된 stdlib 런타임을 구현할 수 있다.

## 문제

깔끔하게 위임할 수 없는 에이전트는 모든 것을 하나의 프롬프트에 쑤셔 넣게 됩니다. 가드레일이 없는 에이전트는 개인 식별 정보(PII), 정책 위반 출력물을 전달하거나 영원히 루프에 갇히게 됩니다. OpenAI의 SDK는 다중 에이전트 작업을 실행 가능하게 만드는 세 가지 기본 요소를 체계화했습니다.

## 개념

### 다섯 가지 기본 요소

1. **에이전트(Agent).** LLM + 지시문 + 도구 + 핸드오프.
2. **핸드오프(Handoff).** 다른 에이전트에게 위임. 모델에는 `transfer_to_<에이전트_이름>`이라는 도구로 표현됨.
3. **가드레일(Guardrail).** 입력(첫 번째 에이전트만), 출력(마지막 에이전트만), 또는 도구 호출(함수 도구별)에 대한 검증.
4. **세션(Session).** 턴 간 자동 대화 기록 관리.
5. **트레이싱(Tracing).** LLM 생성, 도구 호출, 핸드오프, 가드레일을 위한 내장 스팬.

### 도구로서의 핸드오프

모델은 도구 목록에서 `transfer_to_billing_agent`를 확인. 이를 호출하면 런타임에 다음을 신호:

1. 대화 컨텍스트 복사(또는 `nest_handoff_history` 베타를 통해 축소).
2. 대상 에이전트를 해당 지시문으로 초기화.
3. 대상 에이전트로 실행 계속.

이것은 감독자 패턴(Lesson 13 / Lesson 28)의 제품화 버전.

### 가드레일

세 가지 유형:

- **입력 가드레일.** 첫 번째 에이전트의 입력에서 실행. LLM 호출 전 안전하지 않거나 범위를 벗어난 요청 거부.
- **출력 가드레일.** 마지막 에이전트의 출력에서 실행. PII 유출, 정책 위반, 잘못된 응답 감지.
- **도구 가드레일.** 함수 도구별로 실행. 인수 검증, 권한 확인, 실행 감사.

모드:

- **병렬(기본값).** 가드레일 LLM이 주 LLM과 동시에 실행. 꼬리 지연 시간 감소. 트리거 발생 시 주 LLM 작업 폐기(토큰 낭비).
- **차단(`run_in_parallel=False`).** 가드레일 LLM이 먼저 실행. 트리거 발생 시 주 호출에 토큰 낭비 없음.

트리거 발생 시 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered` 발생.

### 트레이싱

기본적으로 활성화. 모든 LLM 생성, 도구 호출, 핸드오프, 가드레일이 스팬을 방출. `OPENAI_AGENTS_DISABLE_TRACING=1`로 비활성화. `add_trace_processor(processor)`로 OpenAI와 함께 자체 백엔드로 스팬 전달.

### 세션

`Session`은 백엔드(SQLite, Redis, 커스텀)에 대화 기록을 저장. `Runner.run(agent, input, session=session)`은 자동 로드 및 추가.

### 이 패턴이 실패하는 경우

- **핸드오프 드리프트.** 에이전트 A가 에이전트 B로 핸드오프한 후 다시 에이전트 A로 돌아옴. 홉 카운터 추가.
- **가드레일 우회.** 도구 가드레일은 함수 도구에서만 발동; 내장 도구(파일 리더, 웹 페치)는 별도 정책 필요.
- **과도한 트레이싱.** 스팬에 민감한 콘텐츠 포함. OTel GenAI 콘텐츠 캡처 규칙(Lesson 23)과 결합 — 외부 저장, ID로 참조.

## 빌드하기

`code/main.py`는 stdlib에서 SDK 형태를 구현합니다:

- `Agent`, `FunctionTool`, `Handoff` (전송 시맨틱스를 가진 함수 툴로 구현).
- 입력/출력/툴 가드레일, 핸드오프 디스패치, 홉 카운터를 포함한 `Runner`.
- 트레이스 형태를 보여주는 간단한 스팬 이미터.
- 사용자의 쿼리에 따라 청구(billing) 또는 지원(support)으로 핸드오프하는 트리아지 에이전트; 하나의 입력에서 가드레일 트립 발생.

실행 방법:

```
python3 code/main.py
```

트레이스에는 두 번의 성공적인 핸드오프, 한 번의 입력 가드레일 트립, 실제 SDK에서 발생하는 것과 유사한 스팬 트리가 표시됩니다.

## 사용 방법

- **OpenAI 에이전트 SDK**를 OpenAI 중심 제품에 사용.
- **Claude 에이전트 SDK** (레슨 17)를 Claude 중심 제품에 사용.
- **LangGraph** (레슨 13)를 명시적 상태 관리와 지속 가능한 재개 기능이 필요할 때 사용.
- **커스텀** 구현을 음성 처리, 다중 공급자 통합, 연합 배포 등 정확한 제어가 필요할 때 사용.

## Ship It

`outputs/skill-agents-sdk-scaffold.md`는 트라이아지 에이전트(triage agent), 핸드오프(handoffs), 입력/출력/툴 가드레일(input/output/tool guardrails), 세션 저장소(session store), 트레이스 프로세서(trace processor)가 포함된 Agents SDK 앱을 위한 스캐폴드를 제공합니다.

## 연습 문제

1. 핸드오프 홉 카운터 추가: N회 전송 후 거부. 동작 추적.
2. `nest_handoff_history` 옵션 구현 — 전송 전 이전 메시지를 하나의 요약으로 축소.
3. 블로킹 출력 가드레일 작성. 트리거되는 프롬프트와 통과하는 프롬프트의 지연 시간 비교.
4. `add_trace_processor`를 JSON 로거에 연결. 스팬당 어떤 형태의 데이터를 출력하는가?
5. SDK 문서 읽기. stdlib 토이를 `openai-agents-python`으로 이식. 어떤 부분을 잘못 모델링했는가?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 에이전트(Agent) | "LLM + 지시사항" | SDK의 에이전트 유형; 도구(tool)와 핸드오프(handoff)를 소유 |
| 핸드오프(Handoff) | "전달" | 모델이 다른 에이전트에게 위임하기 위해 호출하는 도구 |
| 가드레일(Guardrail) | "정책 검사" | 입력/출력/도구 호출에 대한 검증 |
| 트립와이어(Tripwire) | "가드레일 위반" | 가드레일이 거부할 때 발생하는 예외 |
| 세션(Session) | "히스토리 저장소" | 실행 간 지속되는 대화 메모리 |
| 트레이싱(Tracing) | "스팬(Spans)" | LLM + 도구 + 핸드오프 + 가드레일에 대한 내장 관측 가능성 |
| 블로킹 가드레일(Blocking guardrail) | "순차적 검사" | 가드레일이 먼저 실행; 트립 시 토큰 낭비 없음 |
| 병렬 가드레일(Parallel guardrail) | "동시 검사" | 가드레일이 병렬로 실행; 낮은 지연 시간, 트립 시 토큰 낭비 발생 |

## 추가 자료

- [OpenAI Agents SDK 문서](https://openai.github.io/openai-agents-python/) — 프리미티브(primitives), 핸드오프(handoffs), 가드레일(guardrails), 트레이싱(tracing)
- [Claude Agent SDK 개요](https://platform.claude.com/docs/en/agent-sdk/overview) — Claude 전용 대응 기능
- [Anthropic, 효과적인 에이전트 구축](https://www.anthropic.com/research/building-effective-agents) — 핸드오프를 언제 활용해야 하는가
- [OpenTelemetry GenAI 시맨틱 컨벤션](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 표준 에이전트 SDK 스팬 매핑 대상