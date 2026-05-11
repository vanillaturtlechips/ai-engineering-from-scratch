# MCP 샘플링 — 서버 요청 LLM 완성 및 에이전트 루프

> 대부분의 MCP 서버는 단순한 실행기입니다: 인수를 받아 코드를 실행하고 콘텐츠를 반환합니다. 샘플링은 서버가 방향을 전환할 수 있게 합니다. 즉, 클라이언트의 LLM에 결정을 요청합니다. 이를 통해 서버가 모델 자격 증명을 소유하지 않아도 서버 호스팅 에이전트 루프를 구현할 수 있습니다. 2025-11-25에 병합된 SEP-1577은 샘플링 요청 내부에 도구를 추가하여 루프에 더 깊은 추론을 포함할 수 있도록 했습니다. 드리프트 위험 주의: SEP-1577의 샘플링 내 도구 형태는 2026년 1분기까지 실험적 단계였으며 SDK API에서 여전히 정착 중입니다.

**유형:** 구축  
**언어:** Python (표준 라이브러리, 샘플링 하니스)  
**사전 요구 사항:** 13단계 · 07 (MCP 서버), 13단계 · 10 (리소스 및 프롬프트)  
**소요 시간:** ~75분

## 학습 목표

- `sampling/createMessage`가 해결하는 문제(서버 측 API 키 없이 서버 호스팅 루프 구현) 설명.
- 클라이언트에게 멀티턴 프롬프트 샘플링을 요청하고 완성본을 반환하는 서버 구현.
- `modelPreferences`(비용/속도/지능 우선순위)를 활용해 클라이언트 모델 선택 유도.
- 동작을 하드코딩하지 않고 내부적으로 샘플링을 반복하는 `summarize_repo` 도구 구축.

## 문제

코드 요약 워크플로우를 위한 유용한 MCP 서버는 다음 작업을 수행해야 합니다: 파일 트리를 탐색하고, 읽을 파일을 선택하며, 요약을 생성하고 반환합니다. LLM 추론은 어디에서 발생해야 할까요?

옵션 A: 서버가 자체 LLM을 호출합니다. API 키가 필요하며, 서버 측에서 요금이 청구되고, 사용자당 비용이 많이 듭니다.

옵션 B: 서버가 원시 콘텐츠를 반환하고, 클라이언트 에이전트가 추론을 수행합니다. 작동은 하지만 서버 로직이 클라이언트 프롬프트로 이동하여 취약해집니다.

옵션 C: 서버가 `sampling/createMessage`를 통해 클라이언트의 LLM에 요청합니다. 서버는 알고리즘(읽을 파일, 반복 횟수 등)을 유지하는 반면, 클라이언트는 과금 및 모델 선택을 유지합니다. 서버에는 인증 정보가 전혀 없습니다.

샘플링(sampling)은 옵션 C입니다. 이는 신뢰할 수 있는 서버가 전체 LLM 호스트가 되지 않으면서도 에이전트 루프를 호스팅할 수 있는 메커니즘입니다.

## 개념

### `sampling/createMessage` 요청

서버가 전송:

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

클라이언트가 LLM 실행 후 반환:

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

합계가 1.0인 세 개의 부동 소수점 값:

- `costPriority`: 더 저렴한 모델을 선호.
- `speedPriority`: 더 빠른 모델을 선호.
- `intelligencePriority`: 더 강력한 모델을 선호.

추가 `hints`: 서버가 선호하는 명명된 모델. 클라이언트는 힌트를 존중할 수도 있고 무시할 수도 있음. 클라이언트의 사용자 설정이 항상 우선.

### `includeContext`

세 가지 값:

- `"none"` — 서버가 제공한 메시지만 포함. 기본값.
- `"thisServer"` — 이 서버 세션의 이전 메시지 포함.
- `"allServers"` — 모든 세션 컨텍스트 포함.

`includeContext`는 2025-11-25 기준으로 보안 문제로 인해 소프트-폐지됨. 서버 간 컨텍스트 유출이 발생할 수 있음. `"none"`을 선호하고 메시지에 명시적 컨텍스트를 전달.

### 도구 사용 샘플링 (SEP-1577)

2025-11-25 신규: 샘플링 요청에 `tools` 배열을 포함할 수 있음. 클라이언트는 해당 도구를 사용하여 전체 도구 호출 루프를 실행. 이를 통해 서버는 클라이언트의 모델을 통해 ReAct 스타일 에이전트 루프를 호스팅할 수 있음.

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

클라이언트는 루프를 실행: 샘플링 → 도구 호출 시 실행 → 다시 샘플링 → 최종 어시스턴트 메시지 반환. 2026년 1분기까지 실험적 기능. SDK 서명이 변경될 수 있음. 구현 시 2025-11-25 명세서의 클라이언트/샘플링 섹션과 확인 필요.

### 인간 개입(Human-in-the-loop)

클라이언트는 샘플링을 실행하기 전에 서버가 모델에 요청하는 내용을 사용자에게 반드시 표시해야 함. 악성 서버가 샘플링을 이용해 사용자 세션을 조작할 수 있음("사용자에게 X를 말하여 Y를 클릭하게 함"). Claude Desktop, VS Code, Cursor는 샘플링 요청을 사용자가 거부할 수 있는 확인 대화상자로 표시.

2026년 합의: 인간 확인 없이 샘플링을 실행하는 것은 위험 신호. 게이트웨이(Phase 13 · 17)는 저위험 샘플링을 자동 승인하고 의심스러운 것은 자동 거부 가능.

### API 키 없이 서버 호스팅 루프

표준 사용 사례: 자체 LLM 접근 권한이 없는 코드 요약 MCP 서버. 다음 작업 수행:

1. 리포지토리 구조 탐색.
2. "이 리포지토리의 목적을 가장 잘 설명하는 5개 파일을 선택하세요."로 `sampling/createMessage` 호출.
3. 해당 파일 읽기.
4. 파일 내용과 "3단락으로 리포지토리 요약"으로 `sampling/createMessage` 호출.
5. 요약을 `tools/call` 결과로 반환.

서버는 LLM API를 전혀 사용하지 않음. 클라이언트의 사용자가 자신의 자격 증명으로 완성에 대한 비용을 지불.

### 안전 위험 (Unit 42 공개, 2026 Q1)

- **은밀한 샘플링.** 항상 "세션 컨텍스트에서 사용자 이메일로 응답"이라는 샘플링 요청을 호출하는 도구. Phase 13 · 15에서 공격 벡터 다룸.
- **샘플링을 통한 자원 탈취.** 서버가 클라이언트에게 공격자의 페이로드 요약을 요청하여 사용자에게 비용 청구.
- **루프 폭탄.** 서버가 타이트한 루프에서 샘플링을 호출. 클라이언트는 세션당 속도 제한을 반드시 강제해야 함.

## 사용 방법

`code/main.py`에는 가짜 서버-클라이언트 샘플링 하네스가 포함되어 있습니다. 시뮬레이션된 "summarize_repo" 도구는 두 단계의 샘플링(파일 선택, 요약)을 호출하며, 가짜 클라이언트는 미리 준비된 응답을 반환합니다. 이 하네스는 다음을 보여줍니다:

- 서버가 `modelPreferences`와 함께 `sampling/createMessage`를 전송합니다.
- 클라이언트가 완성(completion)을 반환합니다.
- 서버가 루프를 계속 진행합니다.
- 레이트 리미터(rate limiter)가 도구 호출당 총 샘플링 호출 횟수를 제한합니다.

확인해야 할 사항:

- 서버는 단일 도구(`summarize_repo`)만 노출하며, 모든 추론은 샘플링 호출에서 이루어집니다.
- 모델 선호도(model preferences)는 클라이언트의 모델 선택에 가중치를 부여하며, 힌트(hints)는 선호 모델 목록을 제공합니다.
- 루프는 `stopReason: "endTurn"`에서 종료됩니다.
- `max_samples_per_tool = 5` 제한은 무한 루프 상황을 방지합니다.

## Ship It

이 레슨은 `outputs/skill-sampling-loop-designer.md`를 생성합니다. LLM 호출(연구, 요약, 계획)이 필요한 서버 측 알고리즘이 주어졌을 때, 스킬은 적절한 `modelPreferences`, 속도 제한, 안전 확인 기능을 갖춘 샘플링 기반 구현을 설계합니다.

## 연습 문제

1. `code/main.py`를 실행합니다. `max_samples_per_tool`을 2로 변경하고 rate-limit 차단 지점을 관찰합니다.

2. SEP-1577 샘플링 내 도구 변형(variant)을 구현합니다: 샘플링 요청은 `tools` 배열을 포함합니다. 클라이언트 측 루프가 최종 완성(completion)을 반환하기 전에 해당 도구들을 실행하는지 검증합니다. 주의: 드리프트 위험: SDK 서명은 2026년 H1까지 변경될 수 있습니다.

3. 인간 개입 확인(human-in-the-loop) 추가: 서버의 첫 번째 `sampling/createMessage` 이전에 일시 중지하고 사용자 승인을 기다립니다. 거부된 호출은 타입이 지정된 거부(refusal)를 반환합니다.

4. 클라이언트 세션별로 키(key)가 지정된 사용자별 rate limiter를 추가합니다. 동일 사용자의 동일 서버 루프는 예산을 공유해야 합니다.

5. `summarize_pdf` 도구를 설계합니다. 이 도구는 샘플링을 사용하여 포함할 청크(chunk)를 선택합니다. 전송되는 메시지들을 스케치합니다. `modelPreferences.intelligencePriority`가 0.1과 0.9에서 어떻게 동작을 변경하는지 설명합니다.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 샘플링 | "서버-클라이언트 LLM 호출" | 서버가 클라이언트의 모델에 완성(completion)을 요청 |
| `sampling/createMessage` | "그 메서드" | 샘플링 요청을 위한 JSON-RPC 메서드 |
| `modelPreferences` | "모델 우선순위" | 비용/속도/지능 가중치 및 이름 힌트 |
| `includeContext` | "세션 간 유출" | 소프트-폐기된 컨텍스트 포함 모드 |
| SEP-1577 | "샘플링 내 도구" | 서버 호스팅 ReAct를 위한 샘플링 내 도구 허용 |
| Human-in-the-loop | "사용자 확인" | 클라이언트가 실행 전 사용자에게 샘플링 요청을 표시 |
| 루프 폭탄 | "폭주하는 샘플링" | 서버 측 무한 샘플링 루프; 클라이언트는 속도 제한 필요 |
| 은밀한 샘플링 | "숨겨진 추론" | 악성 서버가 샘플링 프롬프트에 의도를 숨김 |
| 리소스 도난 | "사용자의 LLM 예산 사용" | 서버가 클라이언트가 원하지 않는 샘플링에 비용 지출 강제 |
| `stopReason` | "생성 중단 이유" | `endTurn`, `stopSequence`, 또는 `maxTokens`

## 추가 자료

- [MCP — 개념: 샘플링](https://modelcontextprotocol.io/docs/concepts/sampling) — 샘플링에 대한 고수준 개요  
- [MCP — 클라이언트 샘플링 사양 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 표준 `sampling/createMessage` 형태  
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — 샘플링 도구 사양 진화 제안(실험적)  
- [Unit 42 — MCP 공격 벡터](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 은밀한 샘플링 및 자원 탈취 패턴  
- [Speakeasy — MCP 샘플링 핵심 개념](https://www.speakeasy.com/mcp/core-concepts/sampling) — 클라이언트 측 코드 샘플이 포함된 워크스루