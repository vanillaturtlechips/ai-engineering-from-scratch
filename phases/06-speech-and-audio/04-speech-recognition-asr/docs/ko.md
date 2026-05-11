# 음성 인식(ASR) — CTC, RNN-T, 어텐션(Attention)

> 음성 인식은 매 타임스텝에서의 오디오 분류이며, 영어와 묵음(silence)을 아는 시퀀스 모델로 결합됩니다. CTC, RNN-T, 어텐션은 이를 수행하는 세 가지 방법입니다. 하나를 선택하고 그 이유를 이해하세요.

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** Phase 6 · 02 (스펙트로그램 & Mel), Phase 5 · 08 (텍스트용 CNN & RNN), Phase 5 · 10 (어텐션)
**소요 시간:** ~45분

## 문제 정의

10초 길이의 16kHz 오디오 클립이 있습니다. 목표는 "turn on the kitchen lights"라는 문자열을 얻는 것입니다. 문제는 구조적 측면에서 발생합니다: 오디오 프레임이 문자와 1:1로 정렬되지 않습니다. "okay"라는 단어는 200ms 또는 1200ms가 소요될 수 있습니다. 침묵이 발화를 구분하며, 일부 음소(phoneme)는 다른 음소보다 길게 나타납니다. 출력 토큰의 수는 미리 알 수 없습니다.

이 문제를 해결하는 세 가지 접근 방식은 다음과 같습니다:

1. **CTC(Connectionist Temporal Classification).** 특수 *blank* 토큰을 포함한 프레임별 토큰 확률을 출력합니다. 디코딩 시 반복과 blank를 제거합니다. 비자기회귀적(non-autoregressive)이며 빠릅니다. wav2vec 2.0, MMS에서 사용됩니다.
2. **RNN-T(Recurrent Neural Network Transducer).** 인코더 프레임과 이전 토큰을 기반으로 다음 토큰을 예측하는 결합 네트워크(joint network)를 사용합니다. 스트리밍이 가능합니다. Google의 온디바이스 ASR, NVIDIA Parakeet에서 사용됩니다.
3. **어텐션 인코더-디코더(Attention encoder-decoder).** 인코더가 오디오를 은닉 상태(hidden states)로 압축하고, 디코더는 교차 어텐션(cross-attention)을 통해 토큰을 자기회귀적(autoregressively)으로 생성합니다. Whisper, SeamlessM4T에서 사용됩니다.

2026년 기준 LibriSpeech test-clean에서 SOTA 단어 오류율(WER)은 1.4%(Parakeet-TDT-1.1B, NVIDIA)와 1.58%(Whisper-Large-v3-turbo)입니다. 성능 차이는 미미하지만, 배포 측면의 차이는 매우 큽니다.

## 개념

![세 가지 ASR 공식: CTC, RNN-T, 어텐션-인코더-디코더](../assets/asr-formulations.svg)

**CTC 직관.** 인코더 출력이 `T` 프레임 수준의 `V+1` 토큰(V 문자 + 공백)에 대한 분포라고 하자. 길이 `U < T`인 목표 문자열 `y`에 대해, `y`로 축소되는 모든 프레임 정렬이 허용된다. CTC 손실은 이러한 모든 정렬에 대해 합산한다. 추론: 프레임별 argmax, 반복 축소, 공백 제거.

장점: 비자기회귀적, 스트리밍 가능, 선행 정보 불필요. 단점: *조건부 독립 가정* — 각 프레임 예측은 다른 프레임과 독립적이므로 내부 언어 모델이 없다. 빔 서치 또는 얕은 융합을 통해 외부 LM으로 보완.

**RNN-T 직관.** 토큰 히스토리를 임베딩하는 *예측기* 네트워크와 예측기 상태와 인코더 프레임을 결합하여 `V+1`(여기서 `+1`은 널/출력 없음)에 대한 결합 분포를 생성하는 *조인터*를 추가한다. CTC가 무시한 조건부 의존성을 명시적으로 모델링한다. 각 단계가 과거 프레임과 과거 토큰에만 조건화되므로 스트리밍 가능.

장점: 스트리밍 가능 + 내부 LM. 단점: 훈련이 더 복잡하고 메모리 집약적(3D 손실 격자); RNN-T 손실 커널은 별도의 라이브러리 범주.

**어텐션 인코더-디코더.** 로그-멜 프레임에 대한 인코더(6-32 트랜스포머 레이어). 디코더(6-32 트랜스포머 레이어)는 인코더 출력에 교차 어텐션하여 토큰을 자기회귀적으로 생성. 정렬 제약 없음 — 어텐션은 오디오의 모든 위치를 참조할 수 있음. 어텐션을 제한하지 않으면(예: 청크화된 Whisper-Streaming, 2024) 스트리밍 불가.

장점: 오프라인 ASR에서 최고 품질, 표준 시퀀스-투-시퀀스 툴로 쉽게 훈련. 단점: 자기회귀 지연은 출력 길이에 비례; 엔지니어링 없이는 스트리밍 불가.

### WER: 하나의 숫자

**단어 오류율**(WER) = `(S + D + I) / N`, 여기서 S=대체, D=삭제, I=삽입, N=참조 단어 수. 단어 수준에서 레벤슈타인 편집 거리와 일치. 낮을수록 좋음. 20% 이상의 WER은 일반적으로 사용 불가; 5% 미만은 읽기 음성에서 인간 수준. 2026년 표준 벤치마크 결과:

| 모델 | LibriSpeech test-clean | LibriSpeech test-other | 크기 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B 파라미터 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

이 모든 모델은 인코더-디코더 또는 RNN-T 기반. 순수 CTC 시스템(wav2vec 2.0)은 test-clean에서 약 1.8–2.1% 수준.

## 구축 방법

### 1단계: 탐욕적 CTC 디코딩

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: 프레임별 확률 벡터 리스트
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

두 가지 규칙: 연속된 반복 축소, 공백 제거. 예시: `a a _ _ a b b _ c` → `a a b c`.

### 2단계: 빔 서치 CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (토큰, 로그 확률)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

실제 구현에서는 접두사 트리 빔 서치와 언어 모델(LM) 결합을 사용; 이 코드는 개념적 구조만 제공.

### 3단계: WER(단어 오류율)

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 4단계: Whisper 추론

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026년 기준 가장 강력한 일반 ASR 원라이너. 24GB GPU에서 ~20배 실시간 속도로 실행.

### 5단계: Parakeet 또는 wav2vec 2.0을 이용한 스트리밍

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

스트리밍 ASR은 청크 기반 인코더 어텐션과 상태 유지가 필요; 지원 라이브러리 사용(Parakeet의 NeMo, `transformers` 파이프라인에 `chunk_length_s` 추가).

## 사용 방법

2026 스택:

| 상황 | 선택 |
|-----------|------|
| 영어, 오프라인, 최대 품질 | Whisper-large-v3-turbo |
| 다국어, 강건성 | SeamlessM4T v2 |
| 스트리밍, 저지연 | Parakeet-TDT-1.1B 또는 Riva |
| 엣지, 모바일, <500ms 지연 | Whisper-Tiny 양자화 또는 Moonshine (2024) |
| 장문 | VAD 기반 청킹(WhisperX)을 사용한 Whisper |
| 도메인 특화(의료, 법률) | wav2vec 2.0 파인튜닝 + 도메인 언어 모델 융합 |

## 2026년에도 여전히 발생하는 문제점

- **VAD 미사용.** 무음 구간에 Whisper를 실행하면 환각 현상("시청해 주셔서 감사합니다!")이 발생합니다. 항상 VAD로 필터링하세요.
- **문자 vs 단어 vs 서브워드 WER.** 정규화(소문자 변환, 구두점 제거) *후* 단어 수준 WER을 보고하세요.
- **언어 ID 드리프트.** Whisper의 자동 LID는 잡음이 많은 클립을 일본어 또는 웨일스어로 잘못 분류합니다. 알고 있는 경우 `language="en"`을 강제 적용하세요.
- **청킹 없이 긴 클립 처리.** Whisper는 30초 윈도우를 가집니다. 더 긴 클립에는 `chunk_length_s=30, stride=5`를 사용하세요.

## Ship It

`outputs/skill-asr-picker.md`로 저장. 주어진 배포 대상에 대해 모델, 디코딩 전략, 청킹(chunking), 언어 모델(LM) 퓨전(fusion)을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. 이는 수작업으로 만든 CTC 출력을 탐욕적으로 디코딩하고 참조에 대한 WER을 계산합니다.
2. **중간.** 2단계의 접두사-트리 빔 서치를 올바르게 구현합니다(블랭크 병합 규칙 고려). 10개 예시 합성 데이터셋에서 탐욕적 디코딩과 비교합니다.
3. **어려움.** [LibriSpeech test-clean](https://www.openslr.org/12) 데이터셋에 `whisper-large-v3-turbo`를 사용합니다. 처음 100개 발화에서 WER을 계산하고 공개된 수치와 비교합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| CTC | 공백 토큰 손실 | 모든 프레임-토큰 정렬에 대한 주변 확률; 비-자기회귀(Non-AR). |
| RNN-T | 스트리밍 손실 | CTC + 다음 토큰 예측기; 단어 순서 처리 가능. |
| 어텐션 인코더-디코더 | Whisper 스타일 | 인코더 + 교차 어텐션 디코더; 오프라인 품질 최고. |
| WER | 보고하는 숫자 | 단어 수준에서 `(치환+삭제+삽입)/참조사(N)`. |
| 공백 | 공허함 | CTC에서 "이 프레임에 출력 없음"을 나타내는 특수 토큰. |
| 언어 모델 융합 | 외부 언어 모델 | 빔 서치 중 가중치 적용된 언어 모델 로그 확률 추가. |
| VAD | 묵음 게이트 | 음성 활동 감지기; 비음성 구간 제거. |

## 추가 자료

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 논문.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 논문.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022년 기준 논문; 2024년 v3-turbo 확장.
- [NVIDIA NeMo — Parakeet-TDT 카드](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026년 Open ASR 리더보드 1위.
- [Hugging Face — Open ASR 리더보드](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25+ 모델 실시간 벤치마크.