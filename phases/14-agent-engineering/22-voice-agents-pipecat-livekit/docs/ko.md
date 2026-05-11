# 보이스 에이전트: Pipecat과 LiveKit

> 2026년에는 보이스 에이전트가 주요 프로덕션 카테고리로 자리잡습니다. Pipecat은 Python 프레임 기반 파이프라인(VAD → STT → LLM → TTS → 전송)을 제공합니다. LiveKit Agents는 WebRTC를 통해 AI 모델을 사용자와 연결합니다. 프리미엄 스택의 경우 엔드투엔드 생산 지연 시간 목표는 450–600ms입니다.

**유형:** 학습
**언어:** Python (표준 라이브러리)
**선수 지식:** 14단계 · 01 (에이전트 루프), 14단계 · 12 (워크플로우 패턴)
**소요 시간:** ~60분

## 학습 목표

- Pipecat의 프레임 기반 파이프라인 설명: **DOWNSTREAM**(소스→싱크) 및 **UPSTREAM**(제어).
- 표준 음성 파이프라인 단계와 Pipecat이 지원하는 전송(transport) 방식 명명.
- **LiveKit Agents**의 두 음성 에이전트 클래스(**MultimodalAgent**, **VoicePipelineAgent**) 설명 및 각 사용 사례.
- 2026년 프로덕션 지연 시간(latency) 기대치 요약 및 아키텍처 선택에 미치는 영향 설명.

## 문제 정의

음성 에이전트(voice agent)는 TTS(Text-to-Speech)가 단순히 결합된 텍스트 루프가 아닙니다. 대기 시간 예산(latency budget)은 매우 엄격하며(~600ms), 부분 오디오(partial audio)가 기본값이고, 턴 감지(turn detection)는 모델 기반이며, 전송 방식(transport)은 전화 SIP(telephony SIP)부터 WebRTC까지 다양합니다. 프레임 기반 파이프라인(Pipecat)을 구축하거나 플랫폼(LiveKit)에 의존해야 합니다.

## 개념

### Pipecat (pipecat-ai/pipecat)

- Python 프레임 기반 파이프라인 프레임워크.
- `Frame` → `FrameProcessor` 체인.
- 두 가지 흐름 방향:
  - **DOWNSTREAM** — 소스 → 싱크 (오디오 입력, TTS 출력).
  - **UPSTREAM** — 피드백 및 제어 (취소, 메트릭, 바지인).
- `PipelineTask`는 이벤트(`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`)와 메트릭/트레이싱/RTVI 관측자를 통해 라이프사이클을 관리.

전형적인 파이프라인:

```
VAD (Silero) → STT → LLM (컨텍스트가 사용자/어시스턴트 간 전환) → TTS → 전송
```

전송 방식: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows는 구조화된 대화(상태 머신)를 추가. Pipecat Cloud는 관리형 런타임.

### LiveKit Agents (livekit/agents)

- WebRTC를 통해 AI 모델을 사용자에게 연결.
- 주요 개념: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- 두 가지 음성 에이전트 클래스:
  - **MultimodalAgent** — OpenAI Realtime 또는 유사 서비스를 통한 직접 오디오.
  - **VoicePipelineAgent** — STT → LLM → TTS 캐스케이드; 텍스트 수준 제어 제공.
- 트랜스포머 모델을 통한 의미적 턴 감지.
- 네이티브 MCP 통합.
- SIP를 통한 전화 통신.
- LiveKit Inference를 통해 API 키 없이 50+ 모델 지원; 플러그인을 통해 200+ 추가 모델 지원.

### 상용 플랫폼

Vapi(최적화된 프리미엄 스택에서 ~450–600ms)와 Retell(180회 테스트 통화에서 ~600ms 종단 간)은 이 위에 구축됨. WebRTC 팀 없이 관리형 음성 스택을 원할 때 플랫폼 선택.

### 이 패턴이 실패하는 경우

- **바지인 처리 없음.** 사용자가 중단해도 에이전트는 계속 말함. Pipecat에서는 UPSTREAM 취소 프레임, LiveKit에서는 동등한 기능 필요.
- **STT 신뢰도 무시.** 낮은 신뢰도 트랜스크립트가 LLM에 절대 진리처럼 전달됨. 신뢰도 기준 적용 또는 확인 요청 필요.
- **TTS 중간 문장 절단.** 파이프라인이 발화 중간에 취소될 때 TTS가 인지하거나 오디오를 잘라야 함.
- **지연 예산 무시.** 모든 구성 요소가 50–200ms 추가. 출시 전 체인 합계 계산.

### 2026년 일반적인 지연 시간

- VAD: 20–60ms
- STT 부분: 100–250ms
- LLM 첫 토큰: 150–400ms
- TTS 첫 오디오: 100–200ms
- 전송 RTT: 30–80ms

종단 간 450–600ms는 프리미엄. 800–1200ms는 일반적. 1500ms 이상은 고장난 느낌.

## 빌드하기

`code/main.py`는 다음과 같은 프레임 기반 장난감 파이프라인입니다:

- `Frame` 유형(오디오, 전사, 텍스트, tts_오디오, 제어).
- `process(frame)`을 가진 `Processor` 인터페이스.
- 스크립트된 프로세서로 구성된 5단계 파이프라인(VAD → STT → LLM → TTS → 전송).
- 바지인(barge-in) 데모를 위한 UPSTREAM 취소 프레임.

실행 방법:

```
python3 code/main.py
```

트레이스에는 정상 흐름과 발화 중간에 TTS를 중단하는 바지인 취소 사례가 표시됩니다.

## 사용 방법

- **Pipecat** — 전체 제어 가능: 사용자 정의 프로세서, Python 우선, 플러그형 공급자.
- **LiveKit Agents** — WebRTC 우선 배포 및 전화 통신.
- **Vapi / Retell** — WebRTC 팀 없이 호스팅되는 음성 에이전트.
- **OpenAI Realtime / Gemini Live** — 직접 오디오 입력/출력 (MultimodalAgent).

## Ship It

`outputs/skill-voice-pipeline.md`는 VAD(Voice Activity Detection) + STT(Speech-to-Text) + LLM(Large Language Model) + TTS(Text-to-Speech) + 전송(transport) 기능과 함께 바지인(barge-in) 처리 기능을 포함한 Pipecat 형태의 음성 파이프라인을 생성합니다.

## 연습 문제

1. 장난감 파이프라인에 메트릭스 관찰자 추가: 단계별 초당 프레임 수(count frames per stage per second)를 측정하세요. 레이턴시(latency)는 어디에서 누적되나요?
2. 신뢰도 기반 STT 구현: 임계값(threshold) 미만일 경우 "다시 한 번 말씀해 주시겠어요?"(could you repeat that?)를 요청하세요.
3. 의미 기반 턴 감지 추가: 간단한 규칙 — 전사(transcript)가 "?"로 끝나면 턴 종료(end of turn)로 판단합니다.
4. Pipecat의 전송(transport) 문서 읽기. 표준 라이브러리 전송(stdlib transport)을 SmallWebRTCTransport 설정(stub)으로 교체하세요.
5. 동일한 쿼리(query)에 대해 OpenAI Realtime과 STT+LLM+TTS 캐스케이드(cascade)의 성능 측정. 텍스트 수준 제어(text-level control)가 어떤 레이턴시 비용(latency cost)을 발생시키나요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 프레임(Frame) | "이벤트(Event)" | 파이프라인 내 데이터 유형 단위 (오디오, 전사, 텍스트, 제어) |
| 프로세서(Processor) | "파이프라인 단계(Pipeline stage)" | `process(frame)` 메서드를 가진 핸들러 |
| 다운스트림(DOWNSTREAM) | "순방향 흐름(Forward flow)" | 소스 → 싱크: 오디오 입력, 음성 출력 |
| 업스트림(UPSTREAM) | "피드백 흐름(Feedback flow)" | 제어: 취소, 메트릭, 바지인(barge-in) |
| VAD(Voice Activity Detection) | "음성 활동 감지" | 사용자가 말하는 시점을 감지 |
| 의미적 턴 감지(Semantic turn detection) | "스마트 턴 종료(Smart end-of-turn)" | 사용자가 말을 마쳤다는 모델 기반 결정 |
| 멀티모달 에이전트(MultimodalAgent) | "직접 오디오 에이전트" | 오디오 입력, 오디오 출력; 중간에 텍스트 없음 |
| 보이스파이프라인 에이전트(VoicePipelineAgent) | "캐스케이드 에이전트" | STT + LLM + TTS; 텍스트 수준 제어 |

## 추가 자료

- [Pipecat 문서](https://docs.pipecat.ai/getting-started/introduction) — 프레임 기반 파이프라인, 프로세서, 전송 계층
- [LiveKit Agents 문서](https://docs.livekit.io/agents/) — WebRTC + 음성 기본 요소
- [Vapi](https://vapi.ai/) — 관리형 음성 플랫폼
- [Retell AI](https://www.retellai.com/) — 관리형 음성, 지연 시간 벤치마크 제공