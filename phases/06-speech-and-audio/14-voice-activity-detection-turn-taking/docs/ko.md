# 음성 활동 감지 & 발화 전환 — Silero, Cobra, 그리고 Flush 트릭

> 모든 음성 에이전트는 두 가지 결정에 따라 성공하거나 실패합니다: 사용자가 지금 말하고 있는가, 그리고 말을 끝냈는가? VAD(Voice Activity Detection)는 첫 번째 질문에 답합니다. 발화 전환 감지(VAD + 무음 후처리 + 의미적 종료점 모델)는 두 번째 질문에 답합니다. 둘 중 하나라도 잘못되면 어시스턴트는 사용자를 중간에 자르거나 영원히 말을 멈추지 않게 됩니다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** 6단계 · 11 (실시간 오디오), 6단계 · 12 (음성 어시스턴트)
**소요 시간:** ~45분

## 문제 정의

음성 에이전트가 매 20ms 단위마다 내리는 세 가지 주요 결정:

1. **이 프레임에 음성이 있는가?** — VAD(Voice Activity Detection). 이진 분류, 프레임 단위.
2. **사용자가 새로운 발화를 시작했는가?** — 발화 시작 감지(onset detection).
3. **사용자가 발화를 마쳤는가?** — 발화 종료 감지(end-pointing, 턴 종료).

순진한 접근 방식(에너지 임계값)은 교통 소음, 키보드 소리, 군중 소음 등 어떤 배경 소음에도 실패합니다. 2026년형 해결책: **Silero VAD**(오픈소스, 딥러닝 기반) + **턴 감지 모델**(의미 기반 종료 감지) + **VAD 보정 무음 유지 시간**.

> **참고**:  
> - VAD: Voice Activity Detection  
> - Silero VAD: [공식 문서](https://github.com/snakers4/silero-vad)  
> - 턴 감지 모델: 발화 의도/의미를 고려한 종료 예측  
> - 무음 유지 시간: VAD 결과에 기반한 적응형 침묵 지속 시간 보정

## 개념

![VAD 캐스케이드: 에너지 → Silero → 턴 감지기 → 플러시 트릭](../assets/vad-turn-taking.svg)

### 3단계 VAD 캐스케이드

**1단계: 에너지 게이트.** 가장 저렴함. -40 dBFS 임계값 RMS. 명백한 무음을 필터링하지만 임계값 이상의 모든 소음에 반응.

**2단계: Silero VAD** (2020-2026, MIT). 100만 개의 파라미터. 6000개 이상의 언어로 학습. 단일 CPU 스레드에서 30ms 청크당 약 1ms 실행. 5% FPR에서 87.7% TPR. 오픈소스 기본값.

**3단계: 의미론적 턴 감지기.** LiveKit의 턴 감지 모델(2024-2026) 또는 자체 소형 분류기. "문장 중간 일시정지"와 "말하기 완료"를 구분. 무음뿐만 아니라 언어적 맥락(억양 + 최근 단어) 사용.

### 주요 파라미터 및 기본값

- **임계값.** Silero는 확률을 출력; 0.5 이상(기본값) 또는 0.3 이상(민감)에서 음성으로 분류. 낮은 임계값 = 첫 단어 클리핑 감소, 더 많은 오탐.
- **최소 음성 지속 시간.** 250ms 미만의 음성 거부 — 일반적으로 기침 또는 의자 소음.
- **무음 후행 시간(엔드 포인팅).** VAD가 0으로 복귀한 후 500-800ms 대기 후 턴 종료 선언. 너무 짧으면 → 사용자 방해. 너무 길면 → 느린 느낌.
- **프리로드 버퍼.** VAD 발동 전 300-500ms 오디오 유지. "헤이"가 클리핑되는 것 방지.

### 플러시 트릭 (Kyutai 2025)

스트리밍 STT 모델은 미리 보기 지연 시간(Kyutai STT-1B의 경우 500ms, STT-2.6B의 경우 2.5초)을 가짐. 일반적으로 음성 종료 후 그 시간만큼 기다린 후 전사 결과를 얻음. 플러시 트릭: VAD가 음성 종료를 감지하면 **STT에 플러시 신호를 전송**하여 즉시 출력을 강제. STT는 약 4배 실시간 속도로 처리하므로 500ms 버퍼가 약 125ms 내에 완료.

엔드투엔드: 125ms VAD + 플러시 STT = 대화형 지연 시간.

### 2026 VAD 비교

| VAD | 5% FPR에서 TPR | 지연 시간 | 라이선스 |
|-----|----------------|-----------|----------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | 상용 |
| pyannote 분할 | 95% | ~10 ms | MIT 유사 |

Silero는 기본값으로 적합. Cobra는 규정 준수/정확도 업그레이드 옵션. 에너지 전용 VAD는 2026년 프로덕션 환경에 부적합.

## 빌드하기

### 단계 1: 에너지 게이트

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 단계 2: Python에서 Silero VAD 사용

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 단계 3: 턴 종료 상태 머신

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 단계 4: 플러시 트릭 기본 구조

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT(Kyutai, Deepgram, AssemblyAI)는 이 기능이 작동하려면 플러시를 지원해야 합니다. Whisper 스트리밍은 블록 기반이며 항상 청크를 기다리기 때문에 이 방식을 지원하지 않습니다.

## 사용 방법

| 상황 | VAD 선택 |
|-----------|-----------|
| 개방형, 빠른, 일반적 | Silero VAD |
| 상용 콜센터 | Cobra VAD |
| 온디바이스(휴대폰) | Silero VAD ONNX |
| 연구 / 화자 분리 | pyannote segmentation |
| 종속성 없는 대체 솔루션 | WebRTC VAD (레거시) |
| 턴 종료 품질 필요 | Silero + LiveKit turn-detector 계층화 |

경험적 규칙: 다른 옵션이 정말 없는 경우가 아니라면 에너지 기반 VAD만 단독으로 사용하지 마세요.

## 함정(Pitfalls)

- **고정된 임계값(Fixed threshold).** 조용한 환경에서는 작동하지만 시끄러운 환경에서는 실패합니다. 장치 내 보정(calibration)을 수행하거나 Silero로 전환하세요.
- **너무 짧은 무음 후처리(Too-short silence hangover).** 에이전트가 문장 중간에 말을 끊습니다. 대화형 음성에는 500-800ms가 적절합니다.
- **너무 긴 후처리(Too-long hangover).** 반응이 느려집니다. 대상 사용자와 A/B 테스트를 수행하세요.
- **사전 롤 버퍼 없음(No pre-roll buffer).** 사용자 음성의 처음 200-300ms가 손실됩니다. 항상 롤링 사전 롤 버퍼를 유지하세요.
- **의미적 종료점 무시(Ignoring semantic endpointing).** "음, 잠시 생각해 볼게요..."와 같은 문장에는 긴 일시정지가 포함됩니다. 사용자는 생각이 끊기는 것을 싫어합니다. LiveKit의 턴 감지기(turn-detector) 또는 유사 기술을 사용하세요.

## Ship It

`outputs/skill-vad-tuner.md`로 저장. 워크로드에 맞는 VAD 모델, 임계값(threshold), 행오버(hangover), 프리롤(pre-roll), 턴 감지 전략(turn-detection strategy)을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 이 스크립트는 음성 + 무음 + 음성 + 기침 시퀀스를 시뮬레이션하고 세 가지 VAD(Voice Activity Detection) 계층을 테스트합니다.
2. **중간.** `silero-vad`를 설치하고, 5분 길이의 녹음 파일을 처리하세요. 첫 단어 클리핑과 오탐지를 최소화하도록 임계값을 조정하세요. 정밀도(precision)와 재현율(recall)을 보고하세요.
3. **어려움.** 미니 턴 감지기(turn-detector)를 구축하세요: Silero VAD + 마지막 10개 단어의 임베딩(embedding)에 대한 3층 MLP(Multi-Layer Perceptron)를 결합합니다(sentence-transformers 사용). 수작업으로 레이블링된 턴 종료(turn-end) 데이터셋으로 훈련시키세요. Silero 단독 대비 F1 점수 10% 향상을 달성하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| VAD | 음성 감지기 | 프레임별 이진 분류: 이 프레임에 음성이 있는가? |
| Turn detection | 발화 종료 감지 | VAD + 묵음 유지 시간 + 의미적 종료점. |
| Silence hangover | 발화 후 대기 시간 | 발화 종료 선언 전 대기 시간; 500-800ms. |
| Pre-roll | 발화 전 버퍼 | VAD가 트리거되기 전 300-500ms 오디오 유지. |
| Flush trick | 큐타이 핵 | VAD → STT 플러시 → 500ms 대신 125ms 지연. |
| Semantic endpoint | "의도적으로 멈췄는가?" | 묵음이 아닌 단어를 분석하는 ML 분류기. |
| TPR @ FPR 5% | ROC 지점 | 표준 VAD 벤치마크; Silero 87.7%, WebRTC 50%. |

## 추가 자료

- [Silero VAD](https://github.com/snakers4/silero-vad) — 참조용 오픈소스 음성 활동 감지(VAD).
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 상용 정확도 리더.
- [Kyutai — Unmute + flush 트릭](https://kyutai.org/stt) — 200ms 미만 엔지니어링 트릭.
- [LiveKit — 턴 감지](https://docs.livekit.io/agents/logic/turns/) — 프로덕션 환경의 의미 기반 엔드포인팅.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 레거시 기준선.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 화자 분리(diarization) 등급 세분화.