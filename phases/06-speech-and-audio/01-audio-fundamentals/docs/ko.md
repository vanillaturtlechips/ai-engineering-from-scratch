# 오디오 기초 — 파형, 샘플링, 푸리에 변환

> 파형은 원시 신호입니다. 스펙트로그램은 표현 방식입니다. 멜 특징은 ML 친화적인 형태입니다. 모든 현대 음성 인식(ASR) 및 음성 합성(TTS) 파이프라인은 이 단계를 거치며, 첫 번째 단계는 샘플링과 푸리에 변환을 이해하는 것입니다.

**유형:** 학습  
**언어:** Python  
**선수 지식:** Phase 1 · 06 (벡터와 행렬), Phase 1 · 14 (확률 분포)  
**소요 시간:** ~45분

## 문제

마이크는 압력-시간 신호를 생성합니다. 신경망은 텐서를 입력으로 받습니다. 그 사이에 관습의 스택이 존재하며, 이를 위반하면 소리 없는 버그가 발생합니다: 모델은 잘 학습되지만 WER(단어 오류율)이 두 배로 증가하거나, TTS(텍스트-음성 변환) 시스템이 잡음을 출력하거나, 음성 복제 시스템이 화자 대신 마이크를 암기하는 등의 문제가 발생합니다.

음성 시스템의 모든 버그는 다음 세 가지 질문 중 하나로 귀결됩니다:

1. 데이터는 어떤 샘플링 레이트로 기록되었으며, 모델은 어떤 값을 예상하는가?
2. 신호가 에일리어싱(aliased)되었는가?
3. 원시 샘플(raw samples)을 다루고 있는가, 아니면 주파수 표현(frequency representation)을 다루고 있는가?

이 세 가지를 정확히 이해하면 Phase 6의 나머지 부분은 해결 가능합니다. 반대로 잘못 처리하면 Whisper-Large-v4와 같은 모델도 무의미한 결과를 생성합니다.

## 개념

![파형, 샘플링, DFT, 주파수 빈 시각화](../assets/audio-fundamentals.svg)

**파형(Waveform).** `[-1.0, 1.0]` 범위의 1차원 부동소수점 배열. 샘플 번호로 인덱싱. 초 단위로 변환하려면 샘플 레이트로 나눔: `t = n / sr`. 16kHz에서 10초 클립은 160,000개의 부동소수점 배열.

**샘플 레이트(sr).** 초당 샘플 수. 2026년 기준 일반적인 레이트:

| 레이트 | 용도 |
|------|-----|
| 8 kHz | 전화 통신, 레거시 VOIP. 나이퀴스트 4kHz로 자음 손실. ASR에는 피할 것. |
| 16 kHz | ASR 표준. Whisper, Parakeet, SeamlessM4T v2 모두 16kHz 입력 사용. |
| 22.05 kHz | 구형 모델용 TTS 보코더 학습. |
| 24 kHz | 현대 TTS (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD 오디오, 음악. |
| 48 kHz | 영화, 프로 오디오, 고품질 TTS (VALL-E 2, NaturalSpeech 3). |

**나이퀴스트-섀넌(Nyquist-Shannon).** `sr` 샘플 레이트는 최대 `sr/2` 주파수까지 명확히 표현 가능. `sr/2` 경계는 *나이퀴스트 주파수*. 나이퀴스트 이상 에너지는 *에일리어싱*되어 저주파로 접히며 신호 손상. 다운샘플링 전 항상 저역통과 필터 적용.

**비트 깊이(Bit depth).** 16비트 PCM(부호 있는 int16, 범위 ±32,767)은 범용 교환 형식. 음악용 24비트, 내부 DSP용 32비트 부동소수점. `soundfile` 같은 라이브러리는 int16을 읽지만 `[-1, 1]` 범위의 float32 배열을 노출.

**푸리에 변환(Fourier Transform).** 유한 신호는 다양한 주파수의 사인파 합. 이산 푸리에 변환(DFT)은 `N` 샘플에 대해 `N`개의 복소수 계수 계산 — 주파수 빈당 하나. `빈 k`는 `k · sr / N` Hz 주파수에 매핑. 크기는 해당 주파수 진폭, 각도는 위상.

**FFT.** 고속 푸리에 변환: `N`이 2의 거듭제곱일 때 DFT를 `O(N log N)`으로 계산. 모든 오디오 라이브러리는 내부적으로 FFT 사용. 16kHz에서 1024샘플 FFT는 0–8kHz 범위를 15.6Hz 해상도로 512개 사용 가능 주파수 빈 제공.

**프레임 분할 + 윈도우.** 전체 클립을 FFT하지 않음. 일반적으로 25ms 프레임(10ms 홉)으로 분할하고, 각 프레임에 윈도우 함수(한, 해밍) 곱해 경계 불연속성 제거 후 FFT 적용. 이것이 단시간 푸리에 변환(STFT). 02강에서 이어감.

## 구축 방법

### 1단계: 클립 읽기 및 파형 플롯

`code/main.py`는 데모의 종속성을 없애기 위해 stdlib `wave` 모듈만 사용합니다. 프로덕션 환경에서는 `soundfile` 또는 `torchaudio.load`(둘 다 `(waveform, sr)` 튜플 반환)를 사용합니다:

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 2단계: 기본 원리부터 사인파 합성

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16kHz에서 1초 동안 440Hz 사인파(콘서트 A)는 16,000개의 부동 소수점 값입니다. 16비트 PCM 인코딩을 사용하여 `wave.open(..., "wb")`로 작성합니다.

### 3단계: 직접 DFT 계산

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — `N=256`에서 정확성 확인에는 적합하지만 실제 오디오에는 비효율적입니다. 실제 코드에서는 `numpy.fft.rfft` 또는 `torch.fft.rfft`를 호출합니다.

### 4단계: 우세한 주파수 찾기

크기 피크 인덱스 `k_star`는 주파수 `k_star * sr / N`에 매핑됩니다. 440Hz 사인파에 대해 실행하면 `440 * N / sr` 빈에서 피크가 나타나야 합니다.

### 5단계: 에일리어싱 시연

10kHz로 샘플링된 7kHz 사인파(나이퀴스트 = 5kHz)를 사용합니다. 7kHz 톤은 나이퀴스트 주파수보다 높아 `10 − 7 = 3kHz`로 접힙니다. FFT 피크는 3kHz에서 나타납니다. 이는 고전적인 에일리어싱 데모이며 모든 DAC/ADC에 벽돌 필터 형태의 저역 통과 필터가 포함되는 이유입니다.

## 사용 방법

2026년에 실제로 사용할 스택:

| 작업 | 라이브러리 | 이유 |
|------|---------|-----|
| WAV/FLAC/OGG 읽기/쓰기 | `soundfile` (libsndfile 래퍼) | 가장 빠르고 안정적이며 float32를 반환합니다. |
| 리샘플링 | `torchaudio.transforms.Resample` 또는 `librosa.resample` | 올바른 앤티-에일리어싱이 내장되어 있습니다. |
| STFT / Mel | `torchaudio` 또는 `librosa` | GPU 친화적; PyTorch 생태계. |
| 실시간 스트리밍 | `sounddevice` 또는 `pyaudio` | 크로스플랫폼 PortAudio 바인딩. |
| 파일 검사 | `ffprobe` 또는 `soxi` | CLI, 빠르고 샘플링 레이트/채널/코덱을 보고합니다. |

결정 규칙: **다른 어떤 것보다 먼저 샘플링 레이트를 일치시키세요**. Whisper는 16kHz 모노 float32를 기대합니다. 44.1kHz 스테레오를 전달하면 모델 버그처럼 보이는 쓰레기 결과가 나올 것입니다.

## Ship It

`outputs/skill-audio-loader.md`로 저장하세요. 이 스킬은 오디오 입력이 다운스트림 모델의 기대치와 일치하는지 확인하고, 일치하지 않을 경우 올바르게 리샘플링되는지 검증하는 데 도움을 줍니다.

## 연습 문제

1. **쉬움.** 16 kHz 샘플링 레이트에서 220 Hz + 440 Hz + 880 Hz의 1초 혼합 신호를 합성하세요. DFT를 실행하고 예상된 빈(bin)에서 세 개의 피크를 확인하세요.
2. **중간.** 48 kHz로 3초 분량의 음성 WAV 파일을 녹음하세요. `torchaudio.transforms.Resample`(안티-에일리어싱 적용)을 사용해 16 kHz로 다운샘플링한 후, 나이브 데시메이션(매 3번째 샘플 추출)으로 다시 16 kHz로 변환하세요. 두 신호에 대해 FFT를 수행하세요. 에일리어싱은 어디에서 발생하나요?
3. **어려움.** `math` 라이브러리와 3단계의 DFT만을 사용해 STFT를 직접 구현하세요. 프레임 크기 400, 홉 길이 160, Hann 윈도우를 적용하세요. `matplotlib.pyplot.imshow`로 크기(magnitude)를 시각화하세요. 이는 레슨 02의 스펙트로그램입니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 샘플 레이트(Sample rate) | 초당 샘플 수 | ADC가 신호를 측정하는 Hz 단위의 주파수. |
| 나이퀴스트(Nyquist) | 표현할 수 있는 최대 주파수 | `sr/2`; 그 이상의 에너지는 낮은 빈으로 에일리어싱됨. |
| 비트 깊이(Bit depth) | 각 샘플의 해상도 | `int16` = 65,536 단계; `float32` = `[-1, 1]`에서 24비트 정밀도. |
| DFT(Discrete Fourier Transform) | 시퀀스에 대한 푸리에 변환 | `N` 샘플 → `N` 개의 복소수 주파수 계수. |
| FFT(Fast Fourier Transform) | 빠른 DFT | `O(N log N)` 알고리즘, `N`은 2의 거듭제곱 필요. |
| 빈(Bin) | 주파수 열 | `k · sr / N` Hz; 해상도 = `sr / N`. |
| STFT(Short-Time Fourier Transform) | 스펙트로그램의 내부 동작 | 시간 축에서 프레임화 + 윈도우 적용된 FFT. |
| 에일리어싱(Aliasing) | 이상한 주파수 유령 | 나이퀴스트 이상의 에너지가 낮은 빈으로 반사됨.

## 추가 자료

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 샘플링 정리의 기반이 된 논문.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 무료 DSP 표준 교재.
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 코드를 활용한 실용적인 튜토리얼.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 실제 오디오가 깨끗한 사인파가 아닌 이유를 설명하는 참고서.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10분 안에 이해하는 주파수 빈 직관.