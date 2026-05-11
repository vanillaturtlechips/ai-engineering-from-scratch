# 스트리밍 음성-음성 변환 — 모시, 히비키, 그리고 전이중 대화

> 2024-2026년은 음성 AI를 재정의했습니다. 모시는 200ms 지연 시간으로 동시에 듣고 말하는 단일 모델을 출시했습니다. 히비키는 청크 단위로 음성-음성 번역을 수행합니다. 둘 다 ASR → LLM → TTS 파이프라인을 버리고 미미 코덱 토큰 기반의 통합 전이중 아키텍처를 채택했습니다. 이것이 새로운 레퍼런스 설계입니다.

**유형:** 학습
**언어:** Python
**사전 요구 사항:** 6단계 · 13 (신경 오디오 코덱), 6단계 · 11 (실시간 오디오), 7단계 · 05 (풀 트랜스포머)
**소요 시간:** ~75분

## 문제 정의

레슨 11 + 12로 구축된 모든 음성 에이전트는 300-500ms 정도의 근본적인 지연 시간 한계를 가집니다: VAD 발화 감지, STT 음성-텍스트 변환, LLM 추론, TTS 텍스트-음성 변환. 각 단계마다 최소 지연 시간이 존재하며, 튜닝과 병렬화를 통해 개선할 수 있지만 파이프라인 구조가 최종 한계를 결정합니다.

Moshi(Kyutai, 2024-2026)는 다른 질문을 던집니다: 파이프라인이 없다면 어떨까? 하나의 모델이 오디오를 입력받아 텍스트를 필수 단계가 아닌 중간 "내적 독백"으로 사용하며, 직접 연속적으로 오디오를 출력한다면?

그 해답은 **전이중 음성-음성 변환(full-duplex speech-to-speech)**입니다. 이론적 지연 시간은 160ms(80ms Mimi 프레임 + 80ms 음향 지연)이며, 단일 L4 GPU에서의 실제 지연 시간은 200ms입니다. 이는 최고 수준의 파이프라인 음성 에이전트가 달성하는 지연 시간의 절반에 해당합니다.

## 개념

![Moshi 아키텍처: 두 개의 병렬 Mimi 스트림 + 내부 독백 텍스트](../assets/moshi-hibiki.svg)

## Moshi 아키텍처

**입력.** 12.5Hz × 8 코드북을 사용하는 두 개의 Mimi 코덱 스트림:

- 스트림 1: 사용자 오디오 (Mimi 인코딩, 지속적으로 수신)
- 스트림 2: Moshi 자체 오디오 (Moshi가 생성)

**트랜스포머.** 7B-파라미터 Temporal Transformer가 두 스트림과 텍스트 "내부 독백" 스트림을 처리합니다. 각 80ms 단계에서 다음을 수행합니다:

1. 최신 사용자 Mimi 토큰(8 코드북) 소비
2. 가장 최근 Moshi Mimi 토큰(생성된 8 코드북) 소비
3. 다음 Moshi 텍스트 토큰(내부 독백) 생성
4. 다음 Moshi Mimi 토큰(소형 Depth Transformer를 통한 8 코드북) 생성

사용자 오디오, Moshi 오디오, Moshi 텍스트 세 스트림은 모두 병렬로 실행됩니다. Moshi는 말하는 동안 사용자 음성을 들을 수 있으며, 사용자가 방해할 때 자기 자신을 중단할 수 있고, 주요 발화를 끊지 않으면서 백채널("음")을 보낼 수 있습니다.

**Depth Transformer.** 프레임 내에서 8 코드북은 병렬로 예측되지 않습니다 — 코드북 간 의존성이 존재합니다. 소형 2-레이어 "Depth Transformer"가 80ms 내에 순차적으로 예측합니다. 이는 AR 코덱 LM의 표준 분해 방식(VALL-E, VibeVoice에서도 사용)입니다.

## 내부 독백 텍스트가 도움이 되는 이유

명시적 텍스트가 없으면 모델은 음향 스트림에서 언어를 암묵적으로 모델링해야 합니다. Moshi의 통찰: 오디오와 함께 텍스트 토큰을 강제로 출력하게 합니다. 텍스트 스트림은 Moshi가 말하는 내용의 전사와 같습니다. 이는 의미적 일관성을 개선하고, 언어 모델 헤드를 교체하기 쉽게 하며, 전사를 무료로 제공합니다.

## Hibiki: 스트리밍 음성-음성 번역

동일한 아키텍처로, 번역 쌍으로 훈련됩니다. 소스 오디오 입력, 대상 언어 오디오 출력, 지속적으로. Hibiki-Zero(2026년 2월)는 단어 수준 정렬 훈련 데이터 필요성을 제거 — 문장 수준 데이터 + GRPO 강화 학습을 사용한 지연 최적화.

초기 4개 언어 쌍 지원; 약 1000시간으로 새 언어에 적응 가능.

## 광범위한 Kyutai 스택(2026)

- **Moshi** — 전이중 대화(프랑스어 우선, 영어 잘 지원)
- **Hibiki / Hibiki-Zero** — 동시 음성 번역
- **Kyutai STT** — 스트리밍 ASR(500ms 또는 2.5s 선행)
- **Kyutai Pocket TTS** — CPU에서 실행되는 100M-파라미터 TTS(2026년 1월)
- **Unmute** — 공개 서버에서 이들을 결합한 전체 파이프라인

L40S GPU 처리량: 3× 실시간 속도로 64개 동시 세션.

## Sesame CSM — 사촌 모델

Sesame CSM(2025)은 유사한 아이디어 — Llama-3 백본에 Mimi 코덱 헤드를 사용합니다. 하지만 CSM은 전이중이 아닌 단방향(컨텍스트 + 텍스트 입력, 음성 출력)입니다. 시장에서 최고의 "음성 존재감" TTS이지만, Moshi의 전이중 기능과는 다릅니다.

## 2026 성능 수치

| 모델 | 지연 시간 | 사용 사례 | 라이선스 |
|-------|---------|----------|---------|
| Moshi | 200ms (L4) | 전이중 영어/프랑스어 대화 | CC-BY 4.0 |
| Hibiki | 12.5Hz 프레임률 | 프랑스어 ↔ 영어 스트리밍 번역 | CC-BY 4.0 |
| Hibiki-Zero | 동일 | 5개 언어 쌍, 정렬 데이터 불필요 | CC-BY 4.0 |
| Sesame CSM-1B | 200ms TTFA | 컨텍스트 조건부 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300ms | 폐쇄형, OpenAI API | 상용 |
| Gemini 2.5 Live | ~350ms | 폐쇄형, Google API | 상용 |

## 빌드하기

## 1단계: 인터페이스

Moshi는 80ms 청크의 Mimi 인코딩 오디오를 받아 80ms 청크의 Mimi 인코딩 오디오를 반환하는 WebSocket 서버를 노출합니다. 양방향입니다. 지속적으로.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

## 2단계: 전이중(Full-Duplex) 루프

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

양방향은 동시에 실행됩니다. Python asyncio 또는 Rust futures가 표준 전송 계층입니다.

## 3단계: 학습 목표(개념적)

모든 80ms 프레임 `t`에 대해:

- 입력: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- 예측: `moshi_text[t]`, 그 다음 `moshi_mimi[t, codebook_0..7]`

텍스트는 오디오보다 먼저 예측됩니다(내적 독백). 오디오는 깊이 변환기 내에서 코드북 순차적 방식으로 예측됩니다.

## 4단계: Moshi의 강점과 약점

Moshi의 강점:

- 저렴한 하드웨어에서 250ms 미만의 종단간 지연.
- 자연스러운 백채널 및 대화 중단 처리.
- 파이프라인 글루 코드 불필요.

Moshi의 약점:

- 툴 호출(훈련되지 않음; 별도의 LLM 경로 필요).
- 긴 추론(Moshi는 8B급 대화 모델, Claude/GPT-4 아님).
- 니치 주제에 대한 사실적 정확도.
- 대부분의 프로덕션 엔터프라이즈 사용 사례(2026년에도 파이프라인 사용).

## 사용 방법

| 상황 | 선택 |
|-----------|------|
| 최저 지연 시간 음성 동반자 | Moshi |
| 실시간 번역 통화 | Hibiki |
| 음성 데모 / 연구 | Moshi, CSM |
| 도구 활용 기업용 에이전트 | Pipeline (레슨 12), Moshi 제외 |
| 컨텍스트 내 맞춤형 음성 TTS | Sesame CSM |
| 모든 언어 간 음성-음성 변환 | GPT-4o Realtime 또는 Gemini 2.5 Live (상용) |

## 함정

- **제한된 도구 호출.** Moshi는 대화 모델이지 에이전트 프레임워크가 아닙니다. 도구 사용을 위해 파이프라인과 결합하세요.
- **특정 음성 조건화.** Moshi는 단일 훈련된 페르소나를 사용합니다. 클론 생성은 별도의 훈련 실행이 필요합니다.
- **언어 커버리지.** 프랑스어 + 영어는 우수하지만 다른 언어는 제한적입니다. Hibiki-Zero가 도움이 되지만 여전히 훈련 데이터가 필요합니다.
- **리소스 비용.** 전체 Moshi 세션은 GPU 슬롯을 점유합니다. 저렴한 공유 테넌트 배포 패턴이 아닙니다.

## Ship It

`outputs/skill-duplex-pipeline.md`로 저장하세요. 음성 에이전트 워크로드에 대해 파이프라인(pipeline) vs 풀-듀플렉스(full-duplex) 아키텍처를 선택하고 그 이유를 설명합니다.

## 선택: 파이프라인 아키텍처

1. **지연 시간 최적화**  
   파이프라인 아키텍처는 입력 → 처리 → 출력의 단계적 흐름으로 설계되며, 음성 인식(ASR) → NLU → 응답 생성(TTS) 단계를 순차적으로 처리합니다. 이는 음성 에이전트가 사용자 발화를 완전히 인식한 후 응답을 생성하는 **하프-듀플렉스(half-duplex)** 통신에 적합하며, 실시간 양방향 대화보다 **낮은 지연 시간**이 요구되는 시나리오에 효과적입니다.

2. **리소스 효율성**  
   풀-듀플렉스 아키텍처는 동시 음성 입력/출력 처리를 위해 추가 버퍼링과 충돌 감지 메커니즘이 필요합니다. 반면 파이프라인 방식은 각 단계를 독립적으로 최적화할 수 있어 **계산 리소스 사용이 효율적**이며, 에지 디바이스(edge device) 배포 시 유리합니다.

3. **워크로드 특성 부합**  
   음성 에이전트 작업(예: 예약 확인, 정보 조회)은 일반적으로 **단문 대화**로 구성됩니다. 풀-듀플렉스의 동시성 이점은 장시간 연속 대화(예: 회의 통역)에서 두드러지지만, 단문 기반 워크로드에서는 파이프라인으로도 충분한 UX를 제공할 수 있습니다.

4. **디버깅 및 모니터링 용이성**  
   파이프라인은 각 단계(ASR/NLU/TTS)의 입력/출력을 명확히 분리해 **오류 추적**이 용이합니다. 풀-듀플렉스는 중첩된 스트림 처리로 인해 문제 진단이 복잡할 수 있습니다.

## 예외 고려 사항
- **연속 음성 상호작용**이 주요 요구사항인 경우(예: 실시간 통역), 풀-듀플렉스 아키텍처를 재평가해야 합니다.
- 하이브리드 접근 방식(예: 파이프라인 기반 + 오버랩 스트리밍)도 검토할 수 있으나, 현재 요구사항에서는 단순성과 안정성이 우선시됩니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 이 스크립트는 두 스트림 + 내적 독백 아키텍처를 기호적으로 시뮬레이션합니다.
2. **중간.** HuggingFace에서 Moshi를 다운로드하고, 서버를 실행한 후 한 번의 대화를 테스트하세요. 사용자 발화 종료부터 Moshi 응답 시작까지의 벽시계 지연 시간(wall-clock latency)을 측정하세요.
3. **어려움.** Lesson 12 파이프라인 에이전트를 가져와 20개의 매칭된 테스트 발화에서 P50 지연 시간을 Moshi와 비교하세요. 파이프라인 아키텍처가 구조적으로 우위를 점하는 경우를 분석 보고서에 작성하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| Full-duplex | 동시에 듣고 말하기 | 동일한 모델에서 두 오디오 스트림이 동시에 활성화됩니다. |
| Inner monologue | 모델의 텍스트 스트림 | Moshi는 오디오 출력과 함께 텍스트 토큰을 방출합니다. |
| Depth transformer | 인터-코덱북 예측기 | 하나의 80ms 프레임 내에서 8개 코덱북을 예측하는 소형 트랜스포머입니다. |
| Mimi | Kyutai의 코덱 | 12.5Hz × 8 코덱북; 시맨틱+음향; Moshi의 기반 기술입니다. |
| Streaming S2S | 오디오 → 실시간 오디오 | 청크별 번역/대화, 파이프라인 단계 없음. |
| Back-channeling | "음" 반응 | Moshi는 자신의 턴을 끊지 않고 작은 확인 신호를 방출할 수 있습니다.

## 추가 자료

- [Défossez et al. (2024). Moshi — 음성-텍스트 기반 모델](https://arxiv.org/html/2410.00037v2) — 논문.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 정렬된 데이터 없이 스트리밍 번역.
- [Sesame (2025). 음성의 언캐니 밸리 극복](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 사양.
- [Kyutai — Moshi 저장소](https://github.com/kyutai-labs/moshi) — 설치 + 서버.
- [OpenAI — 실시간 API](https://platform.openai.com/docs/guides/realtime) — 폐쇄형 상용 경쟁 제품.
- [Kyutai — 지연 스트림 모델링](https://github.com/kyutai-labs/delayed-streams-modeling) — 내부 STT/TTS 프레임워크.