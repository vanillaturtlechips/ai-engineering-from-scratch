# 캡스톤 03 — 실시간 음성 어시스턴트 (ASR에서 LLM을 거쳐 TTS까지)

> 자연스러운 느낌을 주는 음성 에이전트는 종단 간 지연 시간이 800ms 미만이며, 사용자가 말을 멈췄을 때를 인지하고, 바지-인(barge-in)을 처리하며, 중단 없이 도구를 호출할 수 있어야 합니다. Retell, Vapi, LiveKit Agents, Pipecat은 모두 2026년에 이 기준을 충족했습니다. 이들은 동일한 구조로 구현됩니다: 스트리밍 ASR, 턴 감지기(turn-detector), 스트리밍 LLM, 스트리밍 TTS가 모든 홉(hop)에서 공격적인 지연 예산을 가지고 WebRTC를 통해 연결됩니다. 이를 직접 구축하고, WER(단어 오류율)과 MOS(평균 의견 점수), 잘못된 절단율(false-cutoff rate)을 측정한 후 패킷 손실 조건에서 실행해 보세요.

**유형:** 캡스톤  
**언어:** Python(에이전트 + 파이프라인), TypeScript(웹 클라이언트)  
**선수 과목:** 6단계(음성 및 오디오), 7단계(트랜스포머), 11단계(LLM 엔지니어링), 13단계(도구), 14단계(에이전트), 17단계(인프라)  
**연습 단계:** P6 · P7 · P11 · P13 · P14 · P17  
**소요 시간:** 30시간

## 문제

음성은 2025-2026년 동안 가장 빠르게 발전한 AI UX 분야입니다. 기술적 진입 장벽은 매분기마다 낮아졌습니다. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0, Pipecat 0.0.70은 모두 800ms 미만의 첫 오디오 출력을 가능하게 했습니다. 기준은 단순히 지연 시간만이 아닙니다. 상호작용 경험입니다: 사용자를 중간에 끊지 않기, 자신이 끊기지 않기, 문장 중간 중단에서 복구하기, 대화 중 오디오 지연 없이 도구 호출하기, 불안정한 모바일 네트워크에서도 생존력 유지하기.

REST 호출 3개를 조합해서는 이 수준에 도달할 수 없습니다. 아키텍처는 종단 간 파이프라인 스트리밍이어야 합니다. 이를 구축하면 실패 모드가 드러납니다: 전화 음성용으로 튜닝된 VAD(음성 활동 감지)가 배경 TV 소리에 반응하는 문제, 마침표를 기다리지만 결코 오지 않는 턴 감지기, 400ms를 버퍼링한 후 출력하는 TTS(텍스트 음성 변환). 최종 목표는 부하 상태에서 이러한 문제들을 하나씩 해결하고 지연 시간-품질 보고서를 공개하는 것입니다.

## 개념

파이프라인은 다섯 가지 스트리밍 단계로 구성됩니다: **오디오 입력** (브라우저의 WebRTC 또는 PSTN), **ASR** (Deepgram Nova-3 또는 faster-whisper에서 스트리밍되는 부분 전사), **턴 감지** (부분 전사를 읽어 완료 신호를 판단하는 VAD 및 소형 턴 감지 모델), **LLM** (턴이 완료된 것으로 판단되면 즉시 토큰 스트리밍), **TTS** (첫 LLM 토큰 이후 ~200ms 이내 스트리밍 오디오 출력).

세 가지 주요 고려 사항. **바지인(Barge-in)**: 에이전트가 말하는 중에 사용자가 말을 시작하면 TTS가 취소되고 ASR이 즉시 재개됩니다. **도구 사용**: 대화 중 기능 호출(날씨, 캘린더 등)은 오디오를 지연시키지 않고 사이드 채널에서 실행되어야 합니다. 에이전트는 지연 시간이 300ms를 초과하면 확인 토큰("잠시만 기다려 주세요...")을 미리 채웁니다. **백프레셔**: 패킷 손실 시 부분 전사가 보류되고, VAD는 음성 게이트 임계값을 높이며, 에이전트는 미확인 메시지 위에 말을 하지 않습니다.

측정 기준은 정량적입니다. 15dB SNR에서 Hamming VAD 벤치마크 기준 WER 8% 미만. 100회 측정 통화에서 첫 오디오 출력 p50 800ms 미만. 오절단율 3% 미만. TTS MOS 4.2 이상. 단일 g5.xlarge 인스턴스에서 50개 동시 통화 처리. 이 수치들이 납품 기준입니다.

## 아키텍처

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP 엣지
        |
        v
  LiveKit Agents 1.0  (또는 Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     사이드 채널
(Deepgram         (Silero)          (LiveKit)        도구
 Nova-3 /         speech-gate    완료 점수    (날씨,
 Whisper-v3)      20ms당        부분 입력 기준)        캘린더)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (스트리밍)
     GPT-4o-realtime / Gemini 2.5 Flash /
     캐스케이드 Claude Haiku 4.5
            |
            v
        TTS 스트리밍
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     발신자에게 오디오 반환
            |
            v
   OpenTelemetry 음성 추적 -> Langfuse
```

## 스택

- **전송**: LiveKit Agents 1.0 (WebRTC) + Twilio PSTN 게이트웨이; 대체 프레임워크로 Pipecat 0.0.70
- **ASR**: Deepgram Nova-3 (스트리밍, 첫 부분 결과 300ms 미만) 또는 자체 호스팅 faster-whisper Whisper-v3-turbo
- **VAD**: Silero VAD v5 + LiveKit turn-detector (부분 트랜스크립트를 읽는 소형 트랜스포머)
- **LLM**: 긴밀한 통합을 위한 OpenAI GPT-4o-realtime, Gemini 2.5 Flash Live, 또는 캐스케이드형 Claude Haiku 4.5 (스트리밍 완성, 별도 오디오 경로)
- **TTS**: Cartesia Sonic-2 (최초 바이트 지연 최소화), ElevenLabs Flash v3, 또는 자체 호스팅 오픈소스 Orpheus
- **도구**: 날씨/캘린더/예약용 FastMCP 사이드 채널; 도구 처리 시간 300ms 초과 시 에이전트 사전 필러 발행
- **관측 가능성**: OpenTelemetry 음성 스팬, 오디오 재생 기능이 있는 Langfuse 음성 트레이스
- **배포**: 자체 호스팅 Whisper + Orpheus용 단일 g5.xlarge (24GB VRAM); 최저 지연 시간용 호스팅 API

## 빌드 방법

1. **WebRTC 세션.** LiveKit 룸과 마이크 오디오를 스트리밍하는 웹 클라이언트를 구축합니다. 서버에서 룸에 참여하는 에이전트 워커를 연결합니다.

2. **ASR 스트리밍.** 20ms PCM 프레임을 Deepgram Nova-3(또는 GPU 기반 faster-whisper)로 전달합니다. 부분 및 최종 전사본을 구독합니다. 부분 전사별 지연 시간을 기록합니다.

3. **VAD 및 턴 감지기.** 프레임 스트림에서 Silero VAD v5를 실행합니다. 음성 종료 이벤트 시, 최신 부분 전사본에 대해 LiveKit 턴 감지기를 트리거합니다. VAD가 500ms 동안 무음 상태를 보고하고 턴 감지기 점수가 0.6 이상일 때만 "턴 완료"로 확정합니다.

4. **LLM 스트리밍.** 턴 완료 시, 실행 중인 대화 내용과 최종 전사본을 포함한 LLM 호출을 시작합니다. 토큰을 스트리밍합니다. 첫 번째 토큰이 도착하면 TTS로 전환합니다.

5. **TTS 스트리밍.** Cartesia Sonic-2가 오디오 청크를 스트리밍합니다. 첫 번째 청크는 첫 LLM 토큰 이후 200ms 이내에 서버를 떠나야 합니다. 청크를 LiveKit 룸으로 전송하면 클라이언트가 WebRTC 지터 버퍼를 통해 재생합니다.

6. **대화 끼어들기(Barge-in).** TTS 재생 중 VAD가 새로운 사용자 음성을 감지하면 즉시 TTS 스트림을 취소하고, 남은 LLM 출력을 폐기한 후 ASR을 재시작합니다. `tts_canceled` 스팬을 게시합니다.

7. **도구 사이드 채널.** 날씨 및 캘린더를 함수 호출 도구로 등록합니다. 호출 시 동시에 실행합니다. 300ms 이내에 해결되지 않으면 LLM이 "잠시만요, 확인해 보겠습니다"라는 필러 문구를 출력하도록 합니다. 도구 응답 후 재개합니다.

8. **평가 하네스.** 100건의 통화를 기록합니다. WER(보류된 전사본 대비), 잘못된 종료율(사용자가 문장 중간에 TTS가 취소된 경우), 첫 오디오 출력 p50, TTS MOS(인간 평가 또는 NISQA), 지터 손실 테스트(패킷 3% 손실)를 측정합니다.

9. **부하 테스트.** 합성 발신자로 단일 g5.xlarge 인스턴스에서 50건의 동시 통화를 구동합니다. 지속적인 첫 오디오 출력 p95를 측정합니다.

## 사용 예시

```
caller: "도쿄의 내일 날씨는 어때요"
[asr  ] 부분 결과 @280ms: "도쿄의"
[asr  ] 부분 결과 @540ms: "도쿄의 내일"
[turn ] 완료 점수 0.82 @820ms; 확정
[llm  ] 첫 토큰 @960ms
[tool ] weather.tokyo tomorrow -> 68/52 구름 조금 @1140ms
[tts  ] 첫 오디오 출력 @1040ms: "도쿄의 내일 날씨는 구름이 조금 낀..."
턴 지연 시간: 1040ms 사용자 중단 -> 오디오 출력
```

## Ship It

`outputs/skill-voice-agent.md`가 산출물입니다. 도메인(고객 지원, 일정 관리, 키오스크)이 주어지면 측정 기준에 맞춰 조정된 ASR/VAD/LLM/TTS 파이프라인을 갖춘 LiveKit 에이전트를 구축합니다. 평가 기준:

| 가중치 | 평가 항목 | 측정 방법 |
|:-:|---|---|
| 25 | 종단 간 지연 시간 | 100개의 녹음된 통화에서 p50 첫 오디오 출력 시간 800ms 미만 |
| 20 | 대화 차례 품질 | Hamming VAD 벤치마크에서 오류 차단율 3% 미만 |
| 20 | 도구 사용 정확성 | 오디오 지연 없이 올바른 데이터를 반환하는 중간 대화 도구 호출 |
| 20 | 패킷 손실 시 신뢰성 | 3% 패킷 손실 주입 시 WER 및 대화 차례 안정성 |
| 15 | 평가 하니스 완성도 | 공개 구성으로 재현 가능한 측정 |
| **100** | | |

## 연습 문제

1. g5.xlarge에서 Deepgram Nova-3를 faster-whisper v3 turbo로 교체해 보세요. 지연 시간(latency)과 WER(word error rate) 차이를 측정하고, CPU 대 GPU 결정이 중요한 지점을 식별하세요.

2. 중단-중재 정책(interruption-arbitration policy)을 추가하세요: 사용자가 도구 호출(tool call) 중에 끼어들 때 에이전트는 어떻게 반응해야 할까요? 세 가지 정책(강제 취소(hard cancel), 도구 완료 후 중지(finish-tool-then-stop), 다음 턴 대기열 추가(queue next turn))을 비교하세요.

3. 적대적 턴 감지기(adversarial turn-detector) 테스트를 실행하세요: 사용자가 문장 중간에 긴 침묵을 주는 상황을 테스트하세요. VAD(voice activity detection) 침묵 임계값과 턴 감지기 점수 임계값을 조정하여 900ms를 초과하지 않으면서 최소 오절단(false-cutoff)을 달성하세요.

4. Twilio를 통해 PSTN에 동일한 에이전트를 배포하세요. PSTN 첫 오디오 출력(first-audio-out)과 WebRTC를 비교하세요. 지터 버퍼(jitter-buffer)와 코덱(codec) 차이점을 설명하세요.

5. 비영어 언어(일본어, 스페인어)에 대한 음성 활동 감지(voice activity detection)를 추가하세요. Silero VAD v5의 오작동(false-trigger) 비율을 언어별 파인튜닝(fine-tuning)과 비교 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 턴 감지(Turn detection) | "발화 종료" | VAD 무음 구간과 부분 전사본을 입력으로 받아 사용자가 말을 마쳤는지 판단하는 분류기 |
| 바지인(Barge-in) | "중단 처리" | VAD가 새로운 사용자 발화를 감지하면 재생 중인 TTS를 취소하는 기능 |
| 첫 오디오 출력(First-audio-out) | "지연 시간" | 사용자가 말을 멈춘 시점부터 서버에서 첫 오디오 패킷이 전송되기까지의 시간 |
| VAD(Voice Activity Detection) | "음성 게이트" | 오디오 프레임을 음성 vs 무음으로 분류하는 모델; Silero VAD v5가 2026년 기본값 |
| 지터 버퍼(Jitter buffer) | "오디오 평활화" | 네트워크 변동성을 흡수하기 위해 패킷을 잠시 보관하는 클라이언트 측 버퍼 |
| 필러(Filler) | "확인 토큰" | 도구가 느릴 때 에이전트가 침묵을 피하기 위해 출력하는 짧은 문구 |
| MOS(Mean Opinion Score) | "평균 의견 점수" | 지각적 음성 품질 평가 점수; NISQA가 자동화된 대체 지표 |

## 추가 자료

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 참조 WebRTC 에이전트 프레임워크
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 대체 Python 우선 스트리밍 에이전트 프레임워크
- [OpenAI 실시간 API](https://platform.openai.com/docs/guides/realtime) — 통합 음성 모델 참조
- [Deepgram Nova-3 문서](https://developers.deepgram.com/docs) — 스트리밍 ASR 참조
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 참조 모델
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 저지연 TTS 참조
- [Retell AI 아키텍처](https://docs.retellai.com) — 프로덕션 음성 에이전트 아키텍처
- [Vapi.ai 프로덕션 스택](https://docs.vapi.ai) — 대체 프로덕션 참조