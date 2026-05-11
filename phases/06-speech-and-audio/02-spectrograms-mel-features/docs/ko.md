# 스펙트로그램, 멜 스케일 & 오디오 특징

> 신경망은 원시 파형을 잘 소비하지 않습니다. 스펙트로그램을 소비합니다. 멜 스펙트로그램을 더 잘 소비합니다. 2026년의 모든 ASR(Automatic Speech Recognition), TTS(Text-to-Speech), 오디오 분류기는 이 단일 전처리 선택에 따라 성공하거나 실패합니다.

**유형:** 구축
**언어:** Python
**선수 지식:** 6단계 · 01 (오디오 기초)
**소요 시간:** ~45분

## 문제 정의

10초 길이의 16kHz 클립을 예로 들어보자. 이는 160,000개의 부동소수점(floats)으로 구성되며, 모든 값은 `[-1, 1]` 범위에 있다. 이 데이터는 "개 짖는 소리" 또는 "cat이라는 단어"라는 레이블과 거의 완벽하게 상관관계가 없다. 원시 파형(raw waveform)에는 정보가 있지만 모델이 쉽게 추출할 수 없는 형태로 존재한다. 100ms 간격으로 발음된 두 개의 동일한 음소(phoneme)는 완전히 다른 원시 샘플을 가진다.

스펙트로그램(spectrogram)은 이 문제를 해결한다. 인간 지각이 무시하는 시간적 세부 사항(마이크로초 단위의 지터)을 압축하고, 지각이 주목하는 구조(어떤 주파수 대역이 에너지적인지, ~10–25ms 시간 창에 걸쳐)를 보존한다.

멜 스펙트로그램(Mel spectrogram)은 더 나아간다. 인간은 피치를 로그(log) 스케일로 지각한다: 100Hz와 200Hz의 차이는 1000Hz와 2000Hz의 차이와 "같은 거리"로 느껴진다. 멜 스케일은 주파수 축을 이 지각에 맞춰 왜곡한다. 멜 스케일된 스펙트로그램은 2010년부터 2026년까지 음성 ML에서 가장 중요한 단일 특징(feature)이다.

## 개념

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform).** 파형을 겹치는 프레임으로 분할합니다 (일반적: 25 ms 윈도우, 10 ms 홉 = 16 kHz에서 400 샘플 / 160 샘플). 각 프레임에 윈도우 함수(Hann이 기본값; Hamming은 약간 다른 트레이드오프)를 곱합니다. 각 프레임을 FFT 처리합니다. 크기 스펙트럼을 `(n_frames, n_freq_bins)` 형태의 행렬로 쌓습니다. 이것이 스펙트로그램입니다.

**로그-크기(Log-magnitude).** 원시 크기는 5-6 오더(10^5~10^6) 범위를 가집니다. `log(|X| + 1e-6)` 또는 `20 * log10(|X|)`를 사용하여 동적 범위를 압축합니다. 모든 프로덕션 파이프라인은 원시 크기가 아닌 로그-크기를 사용합니다.

**멜 스케일(Mel scale).** Hz 단위의 주파수 `f`는 `m = 2595 * log10(1 + f / 700)`로 멜 `m`에 매핑됩니다. 이 매핑은 1 kHz 이하에서는 대략 선형, 이상에서는 대략 로그 스케일입니다. 0–8 kHz를 커버하는 80개의 멜 빈이 표준 ASR 입력입니다.

**멜 필터뱅크(Mel filterbank).** 멜 스케일에 균등하게 배치된 삼각형 필터 집합입니다. 각 필터는 인접 FFT 빈의 가중 합입니다. STFT 크기에 필터뱅크 행렬을 곱하면 행렬 곱셈 하나로 멜 스펙트로그램을 얻습니다.

**로그-멜 스펙트로그램(Log-mel spectrogram).** `log(mel_spec + 1e-10)`. Whisper의 입력. Parakeet의 입력. SeamlessM4T의 입력. 2026년 오디오 프론트엔드의 표준.

**MFCCs (Mel-Frequency Cepstral Coefficients).** 로그-멜 스펙트로그램에 DCT (타입 II)를 적용하고 처음 13개 계수를 유지합니다. 특징을 디코릴레이션하고 추가 압축합니다. 2015년경까지 주요 특징이었으나, 이후 CNN/Transformer가 원시 로그-멜을 직접 처리하게 되었습니다. 화자 인식(x-vectors, ECAPA)에서는 여전히 사용됩니다.

**해상도 트레이드오프(Resolution trade).** 더 큰 FFT는 주파수 해상도는 향상되지만 시간 해상도는 저하됩니다. 25 ms / 10 ms는 오디오-ML 기본값, 50 ms / 12.5 ms는 음악용, 5 ms / 2 ms는 트랜젠트 감지(드럼 히트, 플로서브)용입니다.

## 구축 방법

### 1단계: 파형 프레임 분할

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

`frame_len=400, hop=160`으로 10초 16kHz 클립을 처리하면 998개의 프레임이 생성됩니다.

### 2단계: 한(Hann) 윈도우

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT 전에 요소별 곱셈을 수행합니다. 비영점(non-zero) 끝점에서 발생하는 스펙트럼 누설(spectral leakage)을 제거합니다.

### 3단계: STFT 크기 계산

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

실제 구현에서는 `torch.stft` 또는 `librosa.stft`(FFT 기반, 벡터화)를 사용합니다. 여기의 루프는 교육용이며, `code/main.py`의 짧은 클립에서 실행됩니다.

### 4단계: 멜 필터뱅크

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

`n_fft=400`으로 0–8kHz 범위의 80개 멜 스케일은 `(80, 201)` 행렬을 생성합니다. `(n_frames, 201)` STFT 크기에 전치 행렬을 곱하여 `(n_frames, 80)` 멜 스펙트로그램을 얻습니다.

### 5단계: 로그-멜 변환

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

일반적인 대안: `librosa.power_to_db`(참조 정규화된 dB), `10 * log10(power + eps)`. Whisper는 더 복잡한 클리핑 및 정규화 루틴을 사용합니다(Whisper의 `log_mel_spectrogram` 참조).

### 6단계: MFCC 추출

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

각 로그-멜 프레임에 DCT를 적용하고 처음 13개 계수를 유지합니다. 이것이 MFCC 행렬입니다. 첫 번째 계수는 일반적으로 생략됩니다(전반적인 에너지 정보를 인코딩하기 때문).

## 사용 방법

2026 스택:

| 작업 | 특징 |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels, 10 ms hop, 25 ms window |
| TTS 음향 모델 (VITS, F5-TTS, Kokoro) | 80 mels, 5–12 ms hop for fine temporal control |
| 오디오 분류 (AST, PANNs, BEATs) | 128 log-mels, 10 ms hop |
| 화자 임베딩 (ECAPA-TDNN, WavLM) | 80 log-mels or raw-waveform SSL |
| 음악 (MusicGen, Stable Audio 2) | EnCodec discrete tokens (not mels) |
| 키워드 스팟팅 | 40 MFCCs for tiny devices |

경험적 규칙: **음악 작업을 하지 않는다면 80 log-mels로 시작하세요.** 다른 방식을 사용할 경우 그 타당성을 입증해야 합니다.

## 2026년에도 여전히 발생하는 함정

- **멜 개수 불일치.** 학습 시 80개 멜, 추론 시 128개 멜 사용. 무음 실패 발생. 양쪽 끝의 특징(feature) 형태를 로그에 기록하세요.
- **샘플링 레이트 불일치(상류).** 22.05kHz에서 계산된 멜은 16kHz와 다르게 보입니다. 특징 추출(featurization) 전에 샘플링 레이트(SR)를 수정하세요.
- **dB vs 로그(log).** Whisper는 dB-멜이 아닌 로그-멜(log-mel)을 기대합니다. 일부 HF 파이프라인은 자동 감지하지만, 사용자 정의 코드는 그렇지 않습니다.
- **정규화 드리프트.** 학습 시 발화별(per-utterance) 정규화, 추론 시 전역(global) 정규화. WER(단어 오류율)을 두 배로 늘리는 프로덕션 버그입니다.
- **패딩으로 인한 정보 누출.** 클립 끝에 제로 패딩(zero-padding)을 적용하면 후행 프레임에 평탄한 스펙트럼이 생성됩니다. 대칭 패딩(symmetric padding) 또는 복제(replication) 방식을 사용하세요.

## Ship It

`outputs/skill-feature-extractor.md`로 저장하세요. 이 스킬은 주어진 모델 대상에 대해 특징 유형(feature type), 멜 계수 개수(mel count), 프레임/호프(frame/hop), 정규화(normalization)를 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 이 스크립트는 치르프 신호(주파수가 200 → 4000Hz로 스윕되는 신호)를 합성하고 프레임별 최대값 멜 빈(argmax mel bin)을 출력합니다. (선택 사항) 플롯을 그려 스윕과 일치하는지 확인하세요.
2. **중간.** `n_mels`를 `{40, 80, 128}` 중 하나로, `frame_len`을 `{200, 400, 800}` 중 하나로 변경해 다시 실행하세요. 시간 축을 따라 날카로운 피크 대역폭을 측정하세요. 어떤 조합이 치르프 신호를 가장 잘 분해하나요?
3. **어려움.** `power_to_db`를 구현하고, AudioMNIST에서 (a) raw log-mel, (b) `ref=max`를 사용한 dB-mel, (c) MFCC-13 + 델타 + 델타-델타를 특징으로 하는 소형 CNN 분류기의 ASR 정확도를 비교하세요. Top-1 정확도를 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|----------|
| 프레임(Frame) | A slice | 하나의 FFT에 입력되는 25ms 파형 청크. |
| 홉(Hop) | Stride | 연속된 프레임 사이의 샘플 수; 10ms가 ASR 기본값. |
| 윈도우(Window) | Hann/Hamming 것 | 프레임 가장자리를 0으로 점진적으로 줄이는 점별 승수. |
|
| STFT(Short-Time Fourier Transform) | 스펙트로그램 생성기 | 프레임화 + 윈도우 적용된 FFT; 시간 × 주파수 행렬을 생성. |
| 멜(Mel) | 왜곡된 주파수 | 로그-지각 척도; `m = 2595·log10(1 + f/700)`. |
| 필터뱅크(Filterbank) | 행렬 | STFT를 멜 빈으로 투영하는 삼각형 필터. |
| 로그-멜(Log-mel) | Whisper의 입력 | `log(mel_spec + eps)`; 2026년 표준화. |
| MFCC(Mel-Frequency Cepstral Coefficients) | 구식 특징 | 로그-멜의 DCT; 13개 계수, 상관성 제거. |

## 추가 자료

- [Davis, Mermelstein (1980). 단음절 단어 인식을 위한 파라미터 표현 비교](https://ieeexplore.ieee.org/document/1163420) — MFCC 논문.
- [Stevens, Volkmann, Newman (1937). 심리적 음고 측정 척도](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 원본 멜 스케일.
- [OpenAI — Whisper 소스, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 참조 구현 읽기.
- [librosa 특징 추출 문서](https://librosa.org/doc/main/feature.html) — `mfcc`, `melspectrogram`, hop/window에 대한 참조.
- [NVIDIA NeMo — 오디오 전처리](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 모델을 위한 프로덕션 규모 파이프라인.