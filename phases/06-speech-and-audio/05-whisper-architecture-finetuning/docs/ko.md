# Whisper — 아키텍처 & 파인튜닝

> Whisper는 30초 윈도우 기반 트랜스포머 인코더-디코더로, 680k 시간의 다국어 약감독 오디오-텍스트 쌍으로 학습되었습니다. 하나의 아키텍처로 여러 작업 수행, 99개 언어에서 강건성. 2026년 기준 ASR(Automatic Speech Recognition) 참조 모델.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** Phase 6 · 04 (ASR), Phase 5 · 10 (어텐션(attention)), Phase 7 · 05 (풀 트랜스포머(full transformer))
**소요 시간:** ~75분

## 문제 정의

OpenAI가 2022년 9월에 출시한 Whisper는 최초의 상용 ASR(Automatic Speech Recognition) 모델이었습니다. 오디오를 붙여넣기만 하면 텍스트를 얻을 수 있으며, 99개 언어를 지원하고, 잡음에 강하며, 노트북에서도 실행 가능합니다. 2024년에는 Large-v3 및 Turbo 버전이 출시되었고, 2026년에는 팟캐스트 전사부터 음성 비서, YouTube 자막 생성까지 모든 분야의 기본 베이스라인으로 자리 잡았습니다.

하지만 Whisper를 영원히 블랙박스로 취급할 수 있는 파이프라인은 아닙니다. 도메인 변화(domain shift)가 성능을 저하시킵니다. 기술 용어, 화자의 억양, 고유 명사, 짧은 클립, 묵음 등이 문제가 됩니다. 다음 사항을 알아야 합니다:

1. Whisper의 내부 구조.
2. 청크(chunk) 단위, 스트리밍, 장문 오디오를 올바르게 처리하는 방법.
3. 파인튜닝(fine-tuning)이 필요한 시기와 방법.

## 개념

![Whisper 인코더-디코더, 작업, 청크 추론, 파인튜닝](../assets/whisper.svg)

**아키텍처.** 표준 트랜스포머 인코더-디코더 구조.

- **입력:** 30초 로그-멜 스펙트로그램, 80개 멜, 10ms 홉 → 3000 프레임. 짧은 클립은 0으로 패딩, 긴 클립은 청크 처리.
- **인코더:** 컨볼루션 다운샘플링(스트라이드 2) + `N`개의 트랜스포머 블록. Large-v3의 경우 32층, 1280차원, 20개 헤드.
- **디코더:** 인과적 자기 주의(causal self-attn) + 인코더 출력에 대한 교차 주의(cross-attn)를 가진 `N`개의 트랜스포머 블록. 인코더와 동일한 크기.
- **출력:** 51,865개 토큰 어휘집(BPE 토큰) 기반.

Large-v3는 15.5B 파라미터를 가짐. Turbo는 4층 디코더(32층 → 4층)를 사용해 지연 시간을 8배 줄이며 WER(단어 오류율)은 1% 미만 증가.

**프롬프트 형식.** Whisper는 디코더 프롬프트의 특수 토큰으로 제어되는 멀티태스크 모델:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world. 
```

- `<|en|>` — 언어 태그. 번역-전사 동작 강제.
- `<|transcribe|>` 또는 `<|translate|>` — 모든 언어 입력에서 영어 출력으로 번역하거나 그대로 전사.
- `<|notimestamps|>` — 단어 수준 타임스탬프 생략(더 빠름).

프롬프트를 변경하면 하나의 모델로 여러 작업 수행 가능. `<|en|>`을 `<|fr|>`로 변경하면 프랑스어 전사.

**30초 윈도우.** 모든 것이 30초에 고정. 긴 클립은 청크 처리 필요, 짧은 클립은 패딩. 윈도우는 네이티브 스트리밍 미지원 — WhisperX, Whisper-Streaming, faster-whisper가 존재하는 이유.

**로그-멜 정규화.** `(log_mel - 평균) / 표준편차`로, 통계는 Whisper 훈련 코퍼스에서 추출. *반드시* Whisper의 전처리(`whisper.audio.log_mel_spectrogram`)를 사용해야 함. `librosa.feature.melspectrogram`은 사용 불가.

## 2026년 변형 모델

| 변형 모델 | 파라미터 | 지연 시간(A100) | WER(LibriSpeech-clean) |
|----------|---------|----------------|------------------------|
| Tiny | 39M | 1× 실시간 | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming(2024) | 1.55B | 스트리밍 | 2.0% |

## 파인튜닝

2026년 표준 워크플로우:

1. 목표 도메인 오디오 10–100시간 수집 + 정렬된 전사본.
2. `transformers.Seq2SeqTrainer` 실행 + `generate_with_loss` 콜백.
3. 파라미터 효율적: 어텐션 레이어의 `q_proj`, `k_proj`, `v_proj`에 LoRA 적용 시 GPU 메모리 4배 절약, WER 0.3% 미만 증가.
4. 10시간 미만 데이터 시 인코더 동결. 디코더만 튜닝.
5. Whisper의 토크나이저와 프롬프트 형식 사용. 토크나이저 교체 금지.

커뮤니티 결과: 20시간 의학 딕테이션 데이터로 Medium 파인튜닝 시 의학 어휘 WER 12% → 4.5%. 4시간 아이슬란드어 데이터로 Turbo 파인튜닝 시 WER 18% → 6%.

## 빌드하기

## 1단계: 기본 Whisper 실행

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # 반복 방지
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

항상 재정의해야 하는 주요 기본값: `temperature=0.0` (샘플링은 0.0 → 0.2 → 0.4 … 폴백 체인), `condition_on_previous_text=False` (연쇄적 환각 문제 방지), `no_speech_threshold=0.6` (무음 감지).

## 2단계: 청크 기반 장문 처리

```python
# whisperx는 2026년 기준 단어 수준 타임스탬프를 지원하는 장문 처리 레퍼런스
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX는 (1) Silero VAD 게이팅, (2) wav2vec 2.0 기반 단어 수준 정렬, (3) `pyannote.audio`를 통한 화자 분리 기능을 추가합니다. 2026년 프로덕션 트랜스크립션의 핵심 도구.

## 3단계: LoRA로 파인튜닝

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M 학습 가능 / 809M 총 파라미터
```

이후 표준 Trainer 루프를 실행합니다. 1000단계마다 체크포인트 저장. 홀드아웃 데이터로 WER(단어 오류율) 평가.

## 4단계: 각 레이어가 학습하는 내용 분석

```python
# 디코딩 중 크로스 어텐션 가중치를 추출하여 디코더가 주목하는 부분 확인
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: 레이어 × 헤드 × 단계 × 소스 길이
```

히트맵으로 시각화하면 디코더 단계가 인코더 프레임을 스캔할 때 대각선 정렬이 나타납니다. 이 대각선이 Whisper의 단어 타임스탬프 개념입니다.

## 사용 방법

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 일반 영어, 오프라인 | `whisperx`를 통한 Large-v3-turbo |
| 모바일 / 엣지 | 양자화된(int8) Whisper-Tiny 또는 Moonshine |
| 다국어 장문 | `whisperx` + 화자 분리(diarization)를 통한 Large-v3 |
| 저자원 언어 | LoRA로 Medium 또는 Turbo 파인튜닝(fine-tuning) |
| 스트리밍(2초 지연) | Whisper-Streaming 또는 Parakeet-TDT |
| 단어 수준 타임스탬프 | WhisperX (wav2vec 2.0을 통한 강제 정렬) |

`faster-whisper`(CTranslate2 백엔드)는 2026년 기준 가장 빠른 CPU+GPU 추론 런타임으로, 동일한 출력 대비 기본 버전보다 4배 빠릅니다.

## 2026년에도 여전히 발생하는 문제점

- **무음 구간에서의 환각 텍스트.** 캡션으로 학습된 Whisper는 "시청해 주셔서 감사합니다!", "구독하세요!", 노래 가사 등을 포함합니다. 항상 Whisper를 호출하기 전에 VAD(Voice Activity Detection)로 게이트 처리하세요.
- **`condition_on_previous_text` 캐스케이드.** 하나의 환각 텍스트가 이후 창(window)을 오염시킵니다. 청크 간 유창성이 필요하지 않은 경우 `False`로 설정하세요.
- **짧은 클립 패딩.** 2초 클립을 30초로 패딩하면 후속 무음 구간에서 환각이 발생할 수 있습니다. `pad=False`를 사용하거나 VAD로 게이트 처리하세요.
- **잘못된 멜 스펙트로그램 통계.** Whisper의 `whisper.audio.log_mel_spectrogram` 대신 librosa의 멜 스케일을 사용하면 거의 무작위 출력이 생성됩니다. Whisper의 함수를 사용하세요.

## Ship It

`outputs/skill-whisper-tuner.md`로 저장. 주어진 도메인에 대한 Whisper 파인튜닝(fine-tuning) 또는 추론(inference) 파이프라인을 설계합니다.

## 1. 목표 정의
- **도메인 식별**: 의료, 법률, 금융 등 특정 도메인 선택
- **요구 사항 수집**: 
  - 지원 언어 (예: 한국어, 영어)
  - 음성 품질 (잡음 환경, 화자 다양성)
  - 출력 형식 (텍스트, 타임스탬프, 번역)

## 2. 데이터 준비

## 2.1 데이터 수집
- **도메인 특화 음성 데이터** 확보
  - 공개 데이터셋 (예: 의료 대화 녹음)
  - 자체 수집 데이터 (고객 통화 기록 등)
- **메타데이터** 포함: 화자 ID, 도메인 태그, 언어 정보


## 2.2 데이터 전처리
```python
# 예시: Whisper 호환 포맷 변환
import whisper
model = whisper.load_model("base")
model.transcribe("audio.mp3", language="ko")
```


## 2.3 데이터 분할
- 80% 학습 / 10% 검증 / 10% 테스트
- 도메인별 계층적 분할 (stratified split)

## 3. 모델 선택
- **기본 모델**: `whisper-tiny`(저사양) ~ `whisper-large-v3`(고성능)
- **사전 학습 모델**: HuggingFace에서 도메인 유사 모델 검색
  ```bash
  huggingface-cli login
  transformers-cli model list openai whisper
  ```

## 4. 파인튜닝 전략

## 4.1 전이 학습
- **단계적 언프리징**(Gradual Unfreezing):
  ```mermaid
  graph TD
    A[1단계: 어텐션 헤드 고정] --> B[2단계: 인코더 일부 해제]
    B --> C[3단계: 전체 디코더 학습]
  ```


## 4.2 하이퍼파라미터
- **학습률(learning rate)**: 1e-5 ~ 1e-4
- **배치 크기**: GPU 메모리 한계 내 최대값
- **에포크**: 5-10 (과적합 모니터링)

## 5. 평가 메트릭
| 메트릭 | 설명 | 목표 값 |
|---------|------|---------|
| WER(Word Error Rate) | 단어 오류율 | <15% |
| CER(Character Error Rate) | 문자 오류율 | <10% |
| 도메인 정확도 | 전문 용어 인식률 | >90% |

## 6. 배포 파이프라인
1. **모델 변환**: TorchScript 또는 ONNX 형식
2. **서빙**: FastAPI + NVIDIA Triton
3. **모니터링**: 
   - 실시간 WER 추적
   - 사용자 피드백 수집 시스템

## 7. 유지보수
- **지속적 학습**: 새 데이터 20% 추가 시 재학습
- **A/B 테스트**: 새 모델 vs 기존 모델 비교

> **주의**: 의료 도메인에서는 HIPAA 준수 필수. 법률 도메인은 개인정보 마스킹 필요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. Whisper 스타일의 프롬프트를 토크나이징하고, 디코딩된 형태 예산을 계산하며, 10분 클립에 대한 청크 스케줄을 출력합니다.
2. **중간.** `faster-whisper`를 설치하고, 10분 분량의 팟캐스트를 전사하세요. 인간 전사본과 WER(Word Error Rate)을 비교하세요. `language="auto"`와 강제 `language="en"`을 각각 시도해 보세요.
3. **어려움.** HF `datasets`를 사용하여 Whisper가 어려움을 겪는 언어(예: 우르두어)를 선택하고, 2시간 분량의 데이터로 2에폭 동안 LoRA를 사용해 Medium 모델을 파인튜닝(fine-tuning)하세요. WER(Word Error Rate) 변화량을 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 30-sec window | Whisper의 제한 | 하드 입력 제한; 더 긴 오디오는 청크 처리. |
| SOT | Start-of-transcript | `<|startoftranscript|>`가 디코더 프롬프트를 시작합니다. |
| Timestamps token | 시간 정렬 | 0.02초 간격의 오프셋이 51k 어휘 집합의 특수 토큰입니다. |
| Turbo | 빠른 변형 | 4개의 디코더 레이어, 8배 빠름, <1% WER(단어 오류율) 회귀. |
| WhisperX | 장문 처리 래퍼 | VAD(음성 활동 감지) + Whisper + wav2vec 정렬 + 화자 분리. |
| LoRA fine-tune | 효율적 튜닝 | 어텐션에 저랭크 어댑터 추가; 파라미터의 ~0.3%만 학습. |
| Hallucination | 무음 실패 | Whisper가 잡음/무음에서 유창한 영어를 생성합니다. |

## 추가 자료

- [Radford et al. (2022). Whisper 논문](https://arxiv.org/abs/2212.04356) — 원본 아키텍처 및 학습 레시피.
- [OpenAI (2024). Whisper Large-v3-turbo 출시](https://github.com/openai/whisper/discussions/2363) — 4계층 디코더, 8배 속도 향상.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 장문, 단어 정렬, 화자 분리.
- [Systran — faster-whisper 저장소](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 기반, 4배 더 빠름.
- [HuggingFace — Whisper 파인튜닝 튜토리얼](https://huggingface.co/blog/fine-tune-whisper) — 표준 LoRA / 전체 파라미터 튜닝 가이드.