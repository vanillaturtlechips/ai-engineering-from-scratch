# A2A — 에이전트 간 프로토콜(Agent-to-Agent Protocol)

> MCP는 에이전트-툴 간 프로토콜입니다. A2A(Agent2Agent)는 에이전트 간 프로토콜로, 서로 다른 프레임워크로 구축된 불투명한 에이전트들이 협업할 수 있도록 하는 개방형 프로토콜입니다. 2025년 4월 Google에서 출시되었으며, 2025년 6월 Linux Foundation에 기부되었고, 2026년 4월 AWS, Cisco, Microsoft, Salesforce, SAP, ServiceNow를 포함한 150개 이상의 지지자들과 함께 v1.0에 도달했습니다. IBM의 ACP를 흡수하고 AP2 결제 확장 기능을 추가했습니다. 이 강의에서는 에이전트 카드, 태스크 라이프사이클, 그리고 두 가지 전송 바인딩을 다룹니다.

**유형:** 구축(Build)
**언어:** Python (표준 라이브러리, 에이전트 카드 + 태스크 하니스)
**선수 지식:** 13단계 · 06 (MCP 기초), 13단계 · 08 (MCP 클라이언트)
**소요 시간:** ~75분

## 학습 목표

- 에이전트-도구(MCP)와 에이전트-에이전트(A2A) 사용 사례를 구분합니다.
- 기술(skills) 및 엔드포인트 메타데이터를 포함한 에이전트 카드(Agent Card)를 `/.well-known/agent.json`에 게시합니다.
- 작업(Task) 수명 주기(제출됨 → 작업 중 → 입력 필요 → 완료/실패/취소/거부)를 관리합니다.
- 텍스트, 파일, 데이터를 포함한 파트(parts)와 아티팩트(Artifacts)를 출력으로 사용하는 메시지(Messages)를 활용합니다.

## 문제 정의

고객 서비스 에이전트는 보고서 작성을 전문 작가 에이전트에게 위임해야 합니다. A2A(agent-to-agent) 이전의 옵션:

- **커스텀 REST API**. 작동은 하지만 모든 페어링이 일회성입니다.
- **공유 코드베이스**. 두 에이전트가 동일한 프레임워크를 실행해야 합니다.
- **MCP(Multi-Action-Tool Calling Protocol)**. 적합하지 않음: MCP는 도구 호출을 위한 프로토콜이며, 각 에이전트의 불투명한 내부 추론을 유지하면서 두 에이전트가 협업하는 경우에는 적용할 수 없습니다.

A2A는 이러한 간극을 메웁니다. A2A는 한 에이전트가 다른 에이전트에게 **Task(작업)**를 보내는 방식으로 상호작용을 모델링하며, 여기에는 라이프사이클, 메시지, 아티팩트가 포함됩니다. 호출받은 에이전트의 내부 상태는 불투명하게 유지되며, 호출자는 작업 상태 전환과 최종 출력만 확인할 수 있습니다.

A2A는 "서로 다른 프레임워크에 있는 에이전트들이 서로 통신할 수 있게 하는" 프로토콜입니다. MCP를 대체하지 않으며, 두 프로토콜은 상호 보완적입니다.

## 개념

### 에이전트 카드

모든 A2A 호환 에이전트는 `/.well-known/agent.json`에 카드를 게시합니다:

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "학술 논문을 요약하고 인용문을 작성합니다.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "논문 요약",
      "description": "논문 PDF를 읽고 3단락 요약을 생성합니다.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

발견은 URL 기반입니다: 카드를 가져오고, A2A 엔드포인트 URL을 학습하며, 기술을 열거합니다.

### 서명된 에이전트 카드 (AP2)

AP2 확장(2025년 9월)은 에이전트 카드에 암호화 서명을 추가합니다. 발행자는 JWT로 자신의 카드에 서명하고, 소비자는 이를 검증합니다. 이는 사칭을 방지합니다.

### 작업 수명 주기

```
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (메시지를 통한 루프)
```

클라이언트는 `tasks/send`로 작업을 시작합니다. 호출된 에이전트는 상태를 전환하며, 클라이언트는 SSE를 통해 상태 업데이트를 구독하거나 폴링합니다.

### 메시지와 파트

메시지는 하나 이상의 파트를 포함합니다:

- `text` — 일반 콘텐츠.
- `file` — mimeType이 있는 base64 blob.
- `data` — 타입이 지정된 JSON 페이로드(호출된 에이전트를 위한 구조화된 입력).

예시:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "이 논문을 요약해 주세요."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 단락"}}
  ]
}
```

### 아티팩트

출력은 원시 문자열이 아닌 아티팩트입니다. 아티팩트는 이름이 지정된 타입 출력입니다:

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

아티팩트는 청크 단위로 스트리밍될 수 있습니다. 호출자는 이를 누적합니다.

### 두 가지 전송 바인딩

1. **HTTP 위의 JSON-RPC.** `/a2a` 엔드포인트, 요청에는 POST, 스트리밍에는 선택적 SSE. 기본 바인딩.
2. **gRPC.** gRPC가 네이티브인 엔터프라이즈 환경용.

두 바인딩 모두 동일한 논리적 메시지 형태를 전달합니다.

### 불투명성 보존

핵심 설계 원칙: 호출된 에이전트의 내부 상태는 불투명합니다. 호출자는 작업 상태와 아티팩트만 볼 수 있습니다. 호출된 에이전트의 사고 과정, 도구 호출, 하위 에이전트 위임 등은 모두 보이지 않습니다. 이는 도구 호출이 투명한 MCP와 다릅니다.

근거: A2A는 경쟁자들이 내부 구조를 노출하지 않고 협업할 수 있게 합니다. A2A는 "이 고객 서비스 에이전트를 호출"할 수 있지만, 호출자는 해당 에이전트가 서비스를 어떻게 구현하는지 알 수 없습니다.

### 타임라인

- **2025-04-09.** Google이 A2A를 발표.
- **2025-06-23.** Linux Foundation에 기부.
- **2025-08.** IBM의 ACP를 통합.
- **2025-09.** AP2 확장(에이전트 결제) 출시.
- **2026-04.** 150개 이상의 지원 조직과 함께 v1.0 출시.

### MCP와의 관계

| 차원 | MCP | A2A |
|-----------|-----|-----|
| 사용 사례 | 에이전트-도구 | 에이전트-에이전트 |
| 불투명성 | 투명한 도구 호출 | 불투명한 내부 추론 |
| 일반적인 호출자 | 에이전트 런타임 | 다른 에이전트 |
| 상태 | 도구 호출 결과 | 수명 주기가 있는 작업 |
| 인증 | OAuth 2.1 (Phase 13 · 16) | JWT 서명된 에이전트 카드 (AP2) |
| 전송 | Stdio / 스트리밍 가능한 HTTP | HTTP 위의 JSON-RPC / gRPC |

특정 도구를 호출하려면 MCP를 사용합니다. 전체 작업을 다른 에이전트에게 위임하려면 A2A를 사용합니다. 많은 프로덕션 시스템은 둘 다 사용합니다: 에이전트는 도구 계층에 MCP를, 협업 계층에 A2A를 사용합니다.

## 사용 방법

`code/main.py`는 최소한의 A2A(하네스) 구현을 포함합니다: 연구 에이전트가 자신의 카드(agent card)를 게시하고, 작성자 에이전트가 PDF와 텍스트 지시사항을 포함하는 파트를 가진 `tasks/send`를 수신한 후, 작업 상태(working → input_required → working → completed)를 전환하며 텍스트 아티팩트를 반환합니다. 모든 표준 라이브러리(stdlib)를 사용하며, 메시지 형태에 집중하기 위해 메모리 내 전송(in-memory transport)을 활용합니다.

확인해야 할 사항:

- 에이전트 카드(Agent Card) JSON 형식.
- 작업 ID 할당 및 상태 전이.
- 혼합형 타입(parts)을 포함한 메시지.
- 작업 중간(input-required) 분기 처리.
- 완료 시 아티팩트 반환.

## Ship It

이 레슨은 `outputs/skill-a2a-agent-spec.md`를 생성합니다. 다른 에이전트에서 호출할 수 있어야 하는 새로운 에이전트가 주어졌을 때, 스킬은 에이전트 카드 JSON, 스킬 스키마, 엔드포인트 블루프린트를 생성합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 호출된 에이전트가 명확화를 요청하는 입력 대기 단계를 포함하여 전체 작업(Task) 생명주기를 추적하세요.

2. 서명된 에이전트 카드(Agent Card)를 추가하세요. 카드의 정규화된 JSON에 대해 HMAC으로 서명하세요. 검증기를 작성하고 변조된 카드에서 실패하는지 확인하세요.

3. 작업 스트리밍(task streaming)을 구현하세요: 작성 에이전트(writer agent)가 SSE를 통해 3개의 증분적 아티팩트 청크를 방출하고 호출자가 이를 누적하도록 하세요.

4. MCP 서버를 래핑하는 A2A 에이전트를 설계하세요. 각 MCP 도구를 A2A 스킬에 매핑하세요. 트레이드오프를 분석하세요 — 어떤 불투명성(opacity)이 손실되는지 설명하세요.

5. A2A v1.0 발표 문서를 읽고 2026년 4월 기준으로 어떤 프레임워크에서도 아직 구현되지 않은 하나의 기능을 식별하세요. (힌트: 다중 홉 작업 위임(multi-hop task delegation)과 관련이 있습니다.)

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| A2A | "에이전트 간 프로토콜(Agent-to-Agent protocol)" | 불투명한 에이전트 협업을 위한 개방형 프로토콜 |
| 에이전트 카드(Agent Card) | "`.well-known/agent.json`" | 에이전트의 스킬과 엔드포인트를 설명하는 공개된 메타데이터 |
| 스킬(Skill) | "호출 가능한 단위(A callable unit)" | 에이전트가 지원하는 명명된 작업(MCP 도구와 유사) |
| 태스크(Task) | "위임 단위(Unit of delegation)" | 라이프사이클과 최종 산출물이 있는 작업 항목 |
| 메시지(Message) | "태스크 입력(Task input)" | 파트(Part, 텍스트/파일/데이터)를 전달 |
| 파트(Part) | "타입화된 청크(Typed chunk)" | 메시지의 `text` / `file` / `data` 요소 |
| 아티팩트(Artifact) | "태스크 출력(Task output)" | 완료 시 반환되는 이름/타입이 지정된 출력물 |
| AP2 | "에이전트 결제 프로토콜(Agent Payments Protocol)" | 신뢰 및 결제를 위한 서명된 에이전트 카드 확장 |
| 불투명성(Opacity) | "블랙박스 협업(Black-box collaboration)" | 호출자에게 호출된 에이전트의 내부 구조가 숨겨짐 |
| 입력 필요(Input-required) | "태스크 일시 정지(Task pause)" | 에이전트가 추가 정보를 필요로 하는 라이프사이클 상태 |

## 추가 자료

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — 공식 A2A(Agent-to-Agent) 프로토콜 사양
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 참조 구현 및 SDK
- [Linux Foundation — A2A 출시 보도 자료](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025년 6월 거버넌스 이전
- [Google Cloud — A2A 프로토콜 업그레이드](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 로드맵 및 파트너 동향
- [Google Dev — A2A 1.0 마일스톤](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 릴리스 노트 및 하위 호환성 가이드