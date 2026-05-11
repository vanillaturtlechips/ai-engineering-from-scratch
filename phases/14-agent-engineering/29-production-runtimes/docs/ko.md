# 프로덕션 런타임: 큐, 이벤트, 크론

> 프로덕션 에이전트는 요청-응답, 스트리밍, 내구성 실행, 큐 기반 백그라운드, 이벤트 기반, 예약된 총 6가지 런타임 형태에서 실행됩니다. 프레임워크를 선택하기 전에 형태를 먼저 선택하세요. 모든 형태에서 관측 가능성(observability)은 핵심 요소입니다.

**유형:** 학습
**언어:** Python (표준 라이브러리)
**선수 지식:** 14단계 · 13 (LangGraph), 14단계 · 22 (음성)
**소요 시간:** ~60분

## 학습 목표

- 6가지 프로덕션 런타임 형태(runtime shape)를 명명하고 각각을 프레임워크/제품 패턴과 매칭할 수 있다.
- 장기 작업(long-horizon tasks)에 대해 지속 실행(durable execution, LangGraph)이 중요한 이유를 설명할 수 있다.
- 이벤트 기반 런타임(event-driven runtime)과 Claude 관리형 에이전트(Claude Managed Agents)가 적합한 경우를 설명할 수 있다.
- 다단계 에이전트(multi-step agents)에 대한 "관측 가능성(observability)이 구조적 하중을 견딘다(load-bearing)"는 주장을 설명할 수 있다.

## 문제

프로덕션 에이전트는 Jupyter 노트북에서는 나타나지 않는 방식으로 실패합니다: 37단계에서 네트워크 타임아웃 발생, 사용자가 음성 통화 중간에 끊기, 머신 재부팅 시 크론 작업(cron job) 종료, 백그라운드 작업자(background worker)의 메모리 부족. 런타임 형태(runtime shape)는 어떤 실패가 복구 가능한지 결정합니다.

## 개념

### 요청-응답

- 동기식 HTTP. 사용자는 완료까지 대기.
- 짧은 작업(<30초)에만 적합.
- 스택: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- 관측 가능성: 표준 HTTP 액세스 로그 + OTel 스팬.

### 스트리밍

- 점진적 출력을 위한 SSE 또는 WebSocket.
- LiveKit은 WebRTC로 음성/영상 확장 (레슨 22).
- 스택: 스트리밍 지원 프레임워크 + SSE/WS를 처리하는 프론트엔드.
- 관측 가능성: 청크별 타이밍, 첫 토큰 지연 시간, 꼬리 지연 시간.

### 지속적 실행

- 모든 단계 후 상태 체크포인트 저장; 실패 시 자동 재개.
- AutoGen v0.4 액터 모델은 실패를 단일 에이전트로 격리 (레슨 14).
- LangGraph의 핵심 차별점 (레슨 13).
- 단계 수가 불확실하고 복구 비용이 높을 때 필수.

### 큐 기반 / 백그라운드

- 작업이 큐에 진입, 워커가 처리, 결과는 웹훅 또는 pub/sub으로 반환.
- 긴 작업 범위 에이전트(Anthropic의 컴퓨터 사용 발표에 따른 작업당 수십~수백 단계)에 필수.
- 스택: Celery (Python), BullMQ (Node), SQS + Lambda (AWS), 커스텀.
- 관측 가능성: 큐 깊이, 작업별 지연 분포, DLQ 크기.

### 이벤트 기반

- 에이전트는 트리거에 구독: 새 이메일, PR 열림, 크론 실행.
- Claude Managed Agents는 기본 제공 (레슨 17).
- CrewAI Flows (레슨 15)는 이벤트 기반 결정적 워크플로우 구조화.
- 관측 가능성: 트리거 소스, 이벤트-시작 지연, 에이전트 지연.

### 예약형

- 주기적으로 실행되는 크론 형태 에이전트.
- 지속적 실행과 결합하여 실패한 야간 작업이 다음 주기에서 재개되도록 함.
- 스택: Kubernetes CronJob + 지속적 프레임워크; 호스팅 (Render cron, Vercel cron).

### 2026년 배포 패턴

- **CrewAI Flows** 이벤트 기반 프로덕션용.
- **Agno** Python 마이크로서비스용 상태 비저장 FastAPI.
- **Mastra** 임베딩용 서버 어댑터 (Express, Hono, Fastify, Koa).
- **Pipecat Cloud / LiveKit Cloud** 관리형 음성용 (레슨 22).
- **Claude Managed Agents** 호스팅된 장기 비동기 작업용.

### 관측 가능성은 핵심 인프라

OpenTelemetry GenAI 스팬(레슨 23)과 Langfuse/Phoenix/Opik 백엔드(레슨 24) 없이는 40단계에서 실패한 다단계 에이전트를 디버깅할 수 없음. 프로덕션에서는 선택 사항이 아님. "빠르게 디버깅" vs "더 많은 로깅으로 처음부터 재실행"의 차이.

### 프로덕션 런타임 실패 사례

- **잘못된 형태 선택.** 5분 작업에 요청-응답 선택. 사용자 연결 끊김; 작업자 대기열 증가; 재시도 누적.
- **DLQ 미구현.** 데드레터 없이 큐 작업자 실행. 실패한 작업 사라짐.
- **불투명한 백그라운드 작업.** 트레이스 내보내기 없이 백그라운드 에이전트 실행. 사용자 보고 전까지 실패 감지 불가.
- **지속적 상태 건너뛰기.** 재시작 비용을 감당할 수 없는 30초 이상 작업은 지속적 실행 필요.

## 빌드하기

`code/main.py`는 stdlib 다중 형태 데모입니다:

- 요청-응답 엔드포인트(일반 함수).
- 스트리밍 핸들러(제너레이터).
- DLQ가 있는 큐 기반 워커.
- 이벤트 트리거 레지스트리.
- 크론 형태의 스케줄러.

실행 방법:

```bash
python3 code/main.py
```

출력: 동일한 작업에 대한 각 형태의 동작을 보여주는 5개의 트레이스. 동일한 에이전트 로직, 다른 외부 셸. 영구 실행(여섯 번째 형태)은 LangGraph 체크포인팅과 함께 레슨 13에서 의도적으로 다룹니다.

## 사용 방법

- **요청-응답**을 통한 채팅 스타일 UX.
- **스트리밍**을 통한 점진적 응답.
- **지속성**을 활용한 장기 작업.
- **큐**를 이용한 배치/비동기/장시간 작업.
- **이벤트** 기반 에이전트 반응성.
- **크론**을 활용한 유지보수 작업 (메모리 통합, 평가, 비용 보고서).

## Ship It

`outputs/skill-runtime-shape.md`는 작업에 대한 런타임 형태(runtime shape)를 선택하고 관측 가능성(observability) 요구 사항을 연결합니다.

## 연습 문제

1. 레슨 01의 ReAct 루프를 스택 내 6개 모든 도형에 적용하세요. 어떤 도형이 어떤 제품 표면에 적합한지 분석하세요.  
2. 큐 기반 데모에 DLQ(Dead Letter Queue)를 추가하세요. 10% 작업 실패를 시뮬레이션하고 DLQ 크기를 확인하세요.  
3. 매일 밤 상위 20개 트레이스를 대상으로 실행되는 cron 트리거 평가 에이전트를 작성하세요.  
4. 백프레셔(backpressure)가 적용된 스트리밍을 구현하세요: 클라이언트가 느릴 경우 에이전트를 일시 중지하세요. 이것이 턴 예산(turn budget)과 어떻게 상호작용하는지 설명하세요.  
5. Claude 관리형 에이전트 문서를 읽으세요. 자체 호스팅 장기 에이전트(long-horizon agent)를 관리형으로 전환해야 하는 경우는 언제인가요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Request-response | "동기식" | 사용자가 대기; 짧은 작업만 가능 |
| Streaming | "SSE / WS" | 점진적 출력; 더 나은 UX; 청크별 지연 시간 관측 가능 |
| Durable execution | "실패 시 재개" | 체크포인트된 상태; 마지막 단계에서 재시작 |
| Queue-based | "백그라운드 작업" | 프로듀서 / 워커 풀 / DLQ |
| Event-driven | "트리거 기반" | 에이전트가 외부 이벤트에 반응 |
| DLQ | "데드-레터 큐" | 실패한 작업을 위한 대기 공간 |
| Claude Managed Agents | "호스팅된 하니스" | Anthropic 호스팅 장기 비동기 실행 + 캐싱 + 압축 |

## 추가 자료

- [LangGraph 개요](https://docs.langchain.com/oss/python/langgraph/overview) — 지속적 실행 세부 정보
- [Claude 관리형 에이전트 개요](https://platform.claude.com/docs/en/managed-agents/overview) — 호스팅된 장기 비동기 처리
- [Anthropic, 컴퓨터 사용 소개](https://www.anthropic.com/news/3-5-models-and-computer-use) — "작업당 수십~수백 단계"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 액터 모델 결함 격리