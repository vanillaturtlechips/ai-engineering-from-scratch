# 신경망 오디오 코덱 — EnCodec, SNAC, Mimi, DAC 및 의미-음향 분할

> 2026년 오디오 생성은 거의 모두 토큰 기반입니다. EnCodec, SNAC, Mimi, DAC는 연속적인 파형을 트랜스포머가 예측할 수 있는 이산 시퀀스로 변환합니다. 의미 토큰 대 음향 토큰 분할 — 첫 번째 코덱북을 의미 토큰으로, 나머지를 음향 토큰으로 사용 — 은 오디오 분야에서 트랜스포머 이후 가장 중요한 아키텍처 변화입니다.

**유형:** 학습
**언어:** Python
**사전 요구 사항:** 6단계 · 02 (스펙트로그램), 10단계 · 11 (양자화), 5단계 · 19 (서브워드 토큰화)
**소요 시간:** ~60분

## 문제 정의

언어 모델은 이산 토큰(discrete token) 위에서 작동합니다. 오디오는 연속적(continuous)입니다. 음성/음악용 LLM 스타일 모델 — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — 을 구축하려면 먼저 **신경 오디오 코덱(neural audio codec)** 이 필요합니다. 이는 오디오를 작은 어휘(vocabulary)의 토큰으로 이산화(discretize)하는 학습된 인코더(encoder)와 파형(waveform)을 재구성하는 매칭 디코더(decoder)로 구성됩니다.

두 가지 계열이 등장했습니다:

1. **재구성 우선 코덱(reconstruction-first codecs)** — EnCodec, DAC. 지각적 오디오 품질(perceptual audio quality)을 최적화합니다. 토큰은 "음향적(acoustic)"입니다 — 화자 식별(speaker identity), 음색(timbre), 배경 잡음(background noise)을 포함한 모든 것을 포착합니다.
2. **의미 우선 코덱(semantic-first codecs)** — Mimi (Kyutai), SpeechTokenizer. 첫 번째 코덱북(codebook)이 언어적/음성적 내용(linguistic / phonetic content)을 인코딩하도록 강제합니다 (종종 WavLM에서 증류(distillation)하여 구현). 이후 코덱북은 음향적 세부 사항(acoustic detail)을 담습니다.

2024-2026년의 통찰: **순수 재구성 코덱은 텍스트로부터 음성을 생성할 때 흐릿한 결과를 냅니다.** 코덱 토큰 위의 LLM은 동일한 코덱북에서 언어 구조(language structure)와 음향 구조(acoustic structure)를 모두 학습해야 하는데, 이는 확장되지 않습니다. 이들을 분리하는 것 — 의미 코덱북 0, 음향 코덱북 1-N — 이 Moshi와 Sesame CSM의 작동 원리입니다.

## 개념

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 핵심 기법: 잔차 벡터 양자화(Residual Vector Quantization, RVQ)

고품질을 위해 수백만 개의 코드가 필요한 단일 대형 코덱북 대신, 모든 현대 오디오 코덱은 **RVQ**를 사용합니다. RVQ는 작은 코덱북들의 캐스케이드입니다. 첫 번째 코덱북은 인코더 출력을 양자화하고, 두 번째 코덱북은 잔차를 양자화합니다. 각 코덱북은 1024개의 코드를 가지며, 8개의 코덱북은 1024^8 = 10^24의 유효 어휘 크기를 제공합니다.

추론 시 디코더는 프레임당 선택된 모든 코드를 합산하여 재구성합니다.

### 2026년에 중요한 네 가지 코덱

**EnCodec (Meta, 2022).** 기준 모델입니다. 파형 위의 인코더-디코더, RVQ 병목 구조를 사용합니다. 24 kHz, 최대 32개 코덱북, 기본 4개 코덱북 @ 1.5 kbps. `1D 컨볼루션 + 트랜스포머 + 1D 컨볼루션` 아키텍처를 사용합니다. MusicGen에서 사용됩니다.

**DAC (Descript, 2023).** L2 정규화된 코덱북, 주기적 활성화 함수, 개선된 손실 함수를 갖춘 RVQ. 12개 코덱북으로 원본 음성과 구별 불가능한 최고 재구성 충실도를 제공합니다. 44.1 kHz 풀밴드.

**SNAC (Hubert Siuzdak, 2024).** 다중 스케일 RVQ — 거친 코덱북은 세부 코덱북보다 낮은 프레임 레이트에서 작동합니다. 계층적으로 오디오를 모델링합니다: ~12 Hz의 거친 "스케치"와 50 Hz의 세부 사항. 계층적 구조가 LM 기반 생성에 잘 매핑되기 때문에 Orpheus-3B에서 사용됩니다.

**Mimi (Kyutai, 2024).** 2026년의 게임 체인저. 12.5 Hz 프레임 레이트(극히 낮음), 8개 코덱북 @ 4.4 kbps. 코덱북 0은 **WavLM에서 증류**되었으며, WavLM의 음성 콘텐츠 특징을 예측하도록 훈련되었습니다. 코덱북 1-7은 음향 잔차입니다. 이 분할은 Moshi(레슨 15)와 Sesame CSM을 구동합니다.

### 언어 모델링을 위한 프레임 레이트의 중요성

낮은 프레임 레이트 = 짧은 시퀀스 = 더 빠른 LM.

| 코덱 | 프레임 레이트 | 1초 = N 프레임 | 적합한 용도 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 음악, 일반 오디오 |
| DAC-44.1k | 86 Hz | 86 | 고충실도 음악 |
| SNAC-24k (거친) | ~12 Hz | 12 | AR-LM 효율적 |
| Mimi | 12.5 Hz | 12.5 | 스트리밍 음성 |

12.5 Hz에서 10초 발화는 125개의 코덱 프레임에 불과합니다. 트랜스포머가 쉽게 예측할 수 있습니다.

### 의미적 vs 음향적 토큰

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **의미적 토큰 (Mimi의 코덱북 0).** 발음된 내용(음소, 단어, 콘텐츠)을 인코딩합니다. 보조 예측 손실을 통해 WavLM에서 증류되었습니다.
- **음향적 토큰 (코덱북 1-7).** 음색, 화자 신원, 억양, 배경 소음, 세부 사항을 인코딩합니다.

AR LM은 의미적 토큰을 먼저 예측한 후(텍스트에 조건화됨), 음향적 토큰을 예측합니다(의미적 + 화자 참조에 조건화됨). 이 분해는 현대 TTS가 제로샷으로 음성을 복제할 수 있는 이유입니다: 의미적 모델은 콘텐츠를 처리하고, 음향적 모델은 음색을 처리합니다.

### 2026년 재구성 품질 (초당 비트 수, 낮은 비트레이트가 더 좋음)

| 코덱 | 비트레이트 | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Opus와 같은 전통적 코덱은 비트당 지각적 품질에서 여전히 우세합니다. 신경망 코덱은 **이산 토큰**(Opus는 생성하지 않음)과 **생성 모델 품질**(LM이 토큰으로 수행할 수 있는 작업)에서 승리합니다.

## 구축 방법

### 1단계: EnCodec으로 인코딩

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps에서 `n_codebooks=8`입니다. 각 코드는 0-1023(10비트) 범위입니다.

### 2단계: 디코딩 및 재구성 측정

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 3단계: 의미-음향 분리 (Mimi 스타일)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

의미 코드북 0은 WavLM과 정렬됩니다. 텍스트-의미 변환기를 훈련시킬 수 있습니다 — 오디오 직접 생성보다 훨씬 작은 어휘 집합입니다. 이후 별도의 음향-파형 디코더가 화자 참조에 조건부로 작동합니다.

### 4단계: 코덱 토큰에 대한 AR 언어 모델 작동 원리

Mimi의 12.5Hz × 8 코드북에서 10초 음성 클립의 경우:

```
N_tokens = 10 * 12.5 * 8 = 1000 토큰
```

1000 토큰은 변환기에게 사소한 컨텍스트입니다. 256M 파라미터 변환기는 현대 GPU에서 밀리초 단위로 10초 음성을 생성할 수 있습니다.

## 사용 방법

문제 영역 → 코덱 매핑:

| 작업 | 코덱 |
|------|-------|
| 일반 음악 생성 | EnCodec-24k |
| 최고 충실도 복원 | DAC-44.1k |
| 음성 기반 AR 언어 모델(TTS) | SNAC 또는 Mimi |
| 스트리밍 전이중 음성 | Mimi (12.5Hz) |
| 텍스트 기반 사운드 효과 라이브러리 | EnCodec + T5 조건 |
| 세밀한 오디오 편집 | DAC + 인페인팅 |

경험적 규칙: **생성 모델을 구축한다면 Mimi 또는 SNAC로 시작하세요. 압축 파이프라인을 구축한다면 Opus를 사용하세요.**

## 함정(Pitfalls)

- **너무 많은 코드북(codebook).** 코드북 추가는 충실도(fidelity)를 선형적으로 증가시키지만 LM 시퀀스 길이도 선형적으로 증가시킵니다. 8-12개에서 멈추세요.
- **프레임 레이트(frame-rate) 불일치.** 12.5Hz Mimi로 LM을 훈련한 후 50Hz EnCodec으로 파인튜닝(fine-tuning)하면 소리 없이 실패합니다.
- **모든 코드북이 동일하다고 가정.** Mimi에서 코드북 0은 콘텐츠(content)를 담고 있습니다. 이를 잃으면 명료성(intelligibility)이 완전히 파괴됩니다. 코드북 7을 잃는 것은 거의 눈에 띄지 않습니다.
- **재구성 품질(reconstruction quality)을 유일한 메트릭으로 사용.** 코덱(codec)은 재구성 품질이 뛰어나더라도 의미 구조(semantic structure)가 나쁘면 LM 기반 생성에는 쓸모없을 수 있습니다.

## Ship It

`outputs/skill-codec-picker.md`로 저장. 주어진 생성(generative) 또는 압축(compression) 작업에 적합한 코덱(codec)을 선택하세요.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 이 스크립트는 장난감 스칼라 + 잔차 양자화기를 구현하고 코드북을 추가할 때 재구성 오류를 측정합니다.
2. **중간.** `encodec`를 설치하고 보류된 음성 클립에서 1, 4, 8, 32 코드북을 비교하세요. PESQ 또는 MSE 대 비트레이트를 플롯하세요.
3. **어려움.** Mimi를 로드하세요. 클립을 인코딩하세요. 코드북 0을 무작위 정수로 교체한 후 디코딩하세요. 그런 다음 코드북 7을 유사하게 교체하세요. 두 가지 손상 효과를 비교하세요 — 코드북 0 손상은 인식 가능성을 완전히 파괴해야 하며, 코드북 7 손상은 거의 영향을 주지 않아야 합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|----------|
| RVQ | 잔차 양자화(Residual quantization) | 작은 코덱북(codebook)들의 연쇄; 각각 이전 잔차를 양자화함. |
| 프레임 레이트(Frame rate) | 코덱 속도(Codec speed) | 초당 토큰-프레임 수. 낮을수록 빠른 언어 모델(LM). |
| 의미 코덱북(Semantic codebook) | 코덱북 0(Mimi) | SSL(자체 감독 학습) 특징에서 추출된 코덱북; 콘텐츠(contents)를 인코딩함. |
| 음향 코덱북(Acoustic codebooks) | 그 외 모든 것 | 음색(timbre), 운율(prosody), 노이즈, 세부 디테일. |
| PESQ / ViSQOL | 지각 품질(Perceptual quality) | MOS(평균 의견 점수)와 상관관계가 있는 객관적 지표. |
| EnCodec | 메타 코덱(Meta codec) | RVQ 기준 모델; MusicGen에서 사용됨. |
| Mimi | 큐타이 코덱(Kyutai codec) | 12.5Hz 프레임 레이트; 의미-음향 분할; Moshi의 기반. |

## 추가 자료

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — RVQ(Residual Vector Quantization) 베이스라인.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — 최고 음질의 오픈소스.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — 다중 스케일(multi-scale) RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — 의미-음향(semantic-acoustic) 분리, WavLM(웨이브엘엠) 지식 증류.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — 2단계 의미/음향 패러다임.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 최초의 스트리밍 가능한 RVQ 코덱.