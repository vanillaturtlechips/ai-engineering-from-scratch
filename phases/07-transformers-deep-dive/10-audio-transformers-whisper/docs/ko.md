# 오디오 트랜스포머 — Whisper 아키텍처

> 오디오는 시간에 따른 주파수의 이미지입니다. Whisper는 멜 스펙트로그램을 입력으로 받아 음성을 출력하는 ViT입니다.

**유형:** 학습
**언어:** Python
**사전 요구 사항:** 7단계 · 05 (풀 트랜스포머), 7단계 · 08 (인코더-디코더), 7단계 · 09 (ViT)
**소요 시간:** ~45분

## 문제 정의

Whisper(OpenAI, Radford et al. 2022) 이전에는 최첨단 자동 음성 인식(ASR) 기술이 wav2vec 2.0과 HuBERT — 자기 지도 학습 기반 특징 추출기와 파인튜닝된 헤드 — 를 의미했습니다. 고품질이지만 비용이 많이 드는 데이터 파이프라인, 도메인에 취약한 문제. 다국어 음성 인식은 언어 계열별로 별도의 모델이 필요했습니다.

Whisper는 세 가지 핵심 전략을 채택했습니다:

1. **모든 데이터로 학습.** 97개 언어에 걸쳐 인터넷에서 수집한 680,000시간의 약한 레이블이 달린 오디오 데이터. 정제된 학술 코퍼스 없음. 음소 레이블 없음.
2. **멀티태스크 단일 모델.** 전사, 번역, 음성 활동 감지, 언어 식별, 타임스탬핑을 작업 토큰을 통해 동시에 학습하는 하나의 디코더.
3. **표준 인코더-디코더 트랜스포머.** 인코더는 로그-멜 스펙트로그램을 처리. 디코더는 텍스트 토큰을 자기회귀적으로 생성. 보코더 없음, CTC 없음, HMM 없음.

결과: Whisper large-v3는 억양, 노이즈, 정제된 레이블 데이터가 전혀 없는 언어에서도 견고하게 작동합니다. 2026년 기준 모든 오픈소스 음성 어시스턴트와 대부분의 상용 음성 어시스턴트의 기본 음성 프론트엔드입니다.

## 개념

![Whisper 파이프라인: 오디오 → 멜 → 인코더 → 디코더 → 텍스트](../assets/whisper.svg)

### 단계 1 — 리샘플링 + 윈도우

16 kHz의 오디오. 30초로 자르기/패딩. 로그-멜 스펙트로그램 계산: 80 멜 빈, 10 ms 스트라이드 → ~3,000 프레임 × 80 특징. 이것이 Whisper가 보는 "입력 이미지"입니다.

### 단계 2 — 합성곱 스템

커널 3과 스트라이드 2를 가진 두 개의 Conv1D 레이어가 3,000 프레임을 1,500으로 줄입니다. 많은 파라미터를 추가하지 않으면서 시퀀스 길이를 절반으로 줄입니다.

### 단계 3 — 인코더

1,500 타임스텝에 대한 24-레이어(대형 모델 기준) 트랜스포머 인코더. 사인 곡선형 위치 인코딩, 셀프 어텐션, GELU FFN. 1,500 × 1,280 은닉 상태를 생성합니다.

### 단계 4 — 디코더

24-레이어 트랜스포머 디코더. GPT-2의 BPE 어휘 집합을 확장한 오디오 특화 특수 토큰을 포함하는 BPE 어휘에서 토큰을 자동회귀적으로 생성합니다.

### 단계 5 — 작업 토큰

디코더 프롬프트는 모델에 수행할 작업을 지시하는 제어 토큰으로 시작합니다:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

또는

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

모델은 이 규약으로 학습되었습니다. 접두사로 작업을 제어합니다. 2026년식 인스트럭션 튜닝의 동등물이지만 음성에 적용되었습니다.

### 단계 6 — 출력

로그-확률 임계값을 사용한 빔 서치(너비 5). `<|notimestamps|>` 토큰이 없을 때 오디오의 0.02초마다 타임스탬프가 예측됩니다.

### Whisper 모델 크기

| 모델 | 파라미터 | 레이어 | d_model | 헤드 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1,024 | 16 | ~5 GB |
| Large | 1,550M | 32 | 1,280 | 20 | ~10 GB |
| Large-v3 | 1,550M | 32 | 1,280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1,280 | 20 | ~6 GB (4-레이어 디코더) |

Large-v3-turbo(2024)는 디코더를 32레이어에서 4레이어로 줄였습니다. 8배 빠른 디코딩으로 <1 WER 포인트 회귀. 이 디코딩 속도 향상이 2026년 실시간 음성 에이전트의 기본값으로 Whisper-turbo가 사용되는 이유입니다.

### Whisper가 하지 않는 것

- 화자 분리(diarization) 없음. pyannote와 함께 사용.
- 네이티브 실시간 스트리밍 없음 — 30초 윈도우 고정. 현대 래퍼(`faster-whisper`, `WhisperX`)는 VAD + 오버랩을 통해 스트리밍을 추가.
- 외부 청킹 없이는 30초 이상의 장문 컨텍스트 없음. 인간 음성은 전사에 장거리 컨텍스트가 거의 필요 없어 실제로 잘 작동.

### 2026년 현황

| 작업 | 모델 | 비고 |
|------|-------|-------|
| 영어 ASR | Whisper-turbo, Moonshine | Moonshine은 엣지에서 4배 빠름 |
| 다국어 ASR | Whisper-large-v3 | 97개 언어 |
| 스트리밍 ASR | faster-whisper + VAD | 150ms 지연 시간 목표 달성 가능 |
| TTS | Piper, XTTS-v2, Kokoro | 인코더-디코더 패턴, Whisper 형태 |
| 오디오 + 언어 | AudioLM, SeamlessM4T | 텍스트 토큰 + 오디오 토큰을 하나의 트랜스포머에 통합 |

## 빌드하기

`code/main.py`를 참조하세요. 우리는 Whisper를 훈련시키지 않습니다 — 로그-멜 스펙트로그램 파이프라인 + 작업 토큰 프롬프트 포맷터를 구축합니다. 이 부분들이 실제로 프로덕션에서 다루게 되는 부분입니다.

### 1단계: 오디오 합성

16kHz로 샘플링된 440Hz 사인파 1초 분량을 생성합니다. 16,000개 샘플입니다.

### 2단계: 로그-멜 스펙트로그램 (간소화 버전)

전체 멜 스펙트로그램에는 FFT가 필요합니다. 우리는 `librosa` 없이도 파이프라인을 보여줄 수 있는 간소화된 프레임 분할 + 프레임별 에너지 버전을 구현합니다:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

프레임 = 25ms, 홉 = 10ms. Whisper의 윈도우 설정과 일치합니다. 교육용으로는 프레임별 에너지가 멜 빈(mel bin)을 대체합니다.

### 3단계: 30초 길이로 패딩

Whisper는 항상 30초 청크를 처리합니다. 스펙트로그램을 3,000프레임으로 패딩(또는 자르기)합니다.

### 4단계: 프롬프트 토큰 구성

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

이것이 전체 작업 제어 인터페이스입니다. 4토큰 접두사로 구성됩니다.

## 사용 방법

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

더 빠른, OpenAI 호환:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026년에 Whisper를 선택해야 하는 경우:**

- 단일 모델로 다국어 ASR(Automatic Speech Recognition) 지원.
- 노이즈가 많고 다양한 오디오의 강력한 전사.
- 연구/프로토타입 ASR — 가장 빠른 시작점.

**다른 것을 선택해야 하는 경우:**

- 엣지에서의 초저지연 스트리밍 — Moonshine이 동일 품질에서 Whisper를 능가.
- <200ms가 필요한 실시간 대화형 AI — 전용 스트리밍 ASR.
- 화자 분할(diarization) — Whisper는 지원하지 않음; pyannote를 추가.

## Ship It

`outputs/skill-asr-configurator.md`를 참조하세요. 이 스킬은 새로운 음성 애플리케이션을 위해 ASR 모델, 디코딩 파라미터, 전처리 파이프라인을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. 16 kHz 샘플링 레이트에서 10 ms 홉(hop) 크기의 1초 신호에 대한 프레임 수가 ~100 프레임인지 확인합니다. 30초 신호의 경우: ~3,000 프레임.
2. **중간.** `numpy.fft`를 사용하여 전체 로그-멜 스펙트로그램(log-mel spectrogram)을 구축합니다. 80개의 멜 빈(mel bins)이 `librosa.feature.melspectrogram(n_mels=80)`과 수치 오차 범위 내에서 일치하는지 검증합니다.
3. **어려움.** 스트리밍 추론(streaming inference)을 구현합니다: 2초 오버랩(overlap)으로 오디오를 10초 윈도우로 분할하고, 각 청크에 대해 Whisper를 실행한 후 전사 결과를 병합합니다. 5분 팟캐스트 샘플에 대해 단일 패스(single-pass) 대비 단어 오류율(word-error rate)을 측정합니다.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| Mel 스펙트로그램 | "오디오 이미지" | 2D 표현: 한 축은 주파수 빈, 다른 축은 시간 프레임; 셀당 로그 스케일 에너지. |
| Log-mel | "Whisper가 보는 것" | Mel 스펙트로그램에 로그를 적용한 것; 인간의 음량 인식을 근사화. |
| 프레임 | "한 시간 조각" | 25ms 샘플 창; 10ms 스트라이드로 중첩. |
| 태스크 토큰 | "음성을 위한 프롬프트 접두사" | 디코더 프롬프트의 `<|transcribe|>` / `<|translate|>` 같은 특수 토큰. |
| 음성 활동 감지(VAD) | "음성 찾기" | ASR 전 묵음 제거 게이트; 비용을 대폭 절감. |
| CTC | "Connectionist Temporal Classification" | 정렬 없는 훈련을 위한 클래식 ASR 손실 함수; Whisper는 사용하지 않음. |
| Whisper-turbo | "소형 디코더, 전체 인코더" | large-v3 인코더 + 4층 디코더; 8배 빠른 디코딩. |
| Faster-whisper | "프로덕션 래퍼" | CTranslate2 재구현; int8 양자화; OpenAI 레퍼런스 대비 4배 빠름.

## 추가 자료

- [Radford et al. (2022). 대규모 약한 감독을 통한 강건한 음성 인식](https://arxiv.org/abs/2212.04356) — Whisper 논문.
- [OpenAI Whisper 저장소](https://github.com/openai/whisper) — 참조 코드 + 모델 가중치. `whisper/model.py`를 읽어보면 Conv1D 스템(stem) + 인코더(encoder) + 디코더(decoder) 전체 구조를 ~400줄로 확인할 수 있습니다.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 5~6단계에서 설명한 빔 서치(beam search) + 태스크 토큰(task-token) 로직이 여기에 있습니다. 500줄 분량으로 완전히 읽을 수 있습니다.
- [Baevski et al. (2020). wav2vec 2.0: 음성 표현 자기 감독 학습 프레임워크](https://arxiv.org/abs/2006.11477) — 선행 연구; 일부 설정에서 여전히 SOTA(state-of-the-art) 특징 제공.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 프로덕션용 래퍼, 참조 구현보다 4배 빠름.
- [Jia et al. (2024). Moonshine: 실시간 자막 및 음성 명령용 음성 인식](https://arxiv.org/abs/2410.15608) — 2024년 엣지 최적화 ASR, Whisper 형태지만 더 작음.
- [HuggingFace 블로그 — "🤗 Transformers로 다국어 ASR용 Whisper 파인튜닝하기"](https://huggingface.co/blog/fine-tune-whisper) — 멜 스펙트로그램 전처리기와 토큰-타임스탬프 처리를 포함한 표준 파인튜닝 레시피.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 인코더(encoder), 디코더(decoder), 크로스 어텐션(cross-attention), 생성(generation)을 포함한 전체 구현체로, 본 강의의 아키텍처 다이어그램과 일치합니다.