# 오디오 생성

> 오디오는 16-48kHz의 1-D 신호입니다. 5초 클립은 80-240k 샘플입니다. 어떤 트랜스포머도 해당 시퀀스를 직접 처리하지 않습니다. 2026년 모든 프로덕션 오디오 모델의 해결 방법은 동일합니다: 신경망 코덱(Encodec, SoundStream, DAC)이 오디오를 50-75Hz의 이산 토큰으로 압축하고, 트랜스포머 또는 확산 모델이 토큰을 생성합니다.

**유형:** 구축(Build)  
**언어:** Python  
**선수 지식:** Phase 6 · 02 (오디오 특징), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)  
**소요 시간:** ~45분

## 문제 정의

세 가지 오디오 생성 작업:

1. **텍스트-음성 변환(Text-to-speech).** 텍스트를 입력으로 받아 음성을 생성합니다. 깨끗한 음성은 좁은 대역을 가지며 강한 음운 구조를 갖습니다 — 토큰 기반 트랜스포머로 잘 해결됩니다. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **음악 생성(Music generation).** 프롬프트(텍스트, 멜로디, 코드 진행, 장르)를 입력으로 받아 음악을 생성합니다. 훨씬 더 넓은 분포를 가집니다. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **오디오 효과/사운드 디자인(Audio effects / sound design).** 프롬프트를 입력으로 받아 환경음 또는 폴리(Foley) 사운드를 생성합니다. AudioGen, AudioLDM 2, Stable Audio Open.

이 세 가지 모두 동일한 기반에서 실행됩니다: 신경망 오디오 코덱(neural audio codec) + 토큰-AR 또는 확산(diffusion) 생성기.

## 개념

![오디오 생성: 코덱 토큰 + 트랜스포머 또는 디퓨전](../assets/audio-generation.svg)

### 신경망 오디오 코덱

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). 컨볼루션 인코더는 웨이브폼을 시간 단계별 벡터로 압축하며, 잔차 벡터 양자화(RVQ)는 각 벡터를 K개의 코덱 인덱스 캐스케이드로 변환합니다. 디코더는 이를 역변환합니다. 24kHz 오디오를 2kbps로 압축하기 위해 75Hz에서 8개의 RVQ 코덱을 사용하면 초당 600토큰이 생성됩니다.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → 75Hz에서 인덱스
                     ├─ RVQ layer 2 → 75Hz에서 인덱스
                     ├─ ...
                     └─ RVQ layer 8
```

### 두 가지 생성 패러다임

**토큰-자기회귀적(Token-autoregressive).** RVQ 토큰을 시퀀스로 평탄화한 후 디코더 전용 트랜스포머를 실행합니다. MusicGen은 "지연된 병렬" 방식을 사용하여 K개의 코덱 스트림을 병렬로 생성하며, 각 스트림에는 오프셋이 적용됩니다. VALL-E는 텍스트 프롬프트 + 3초 음성 샘플을 기반으로 음성 토큰을 생성합니다.

**잠재 디퓨전(Latent diffusion).** 코덱 토큰을 연속 잠재 변수로 패킹하거나 범주형 디퓨전으로 모델링합니다. Stable Audio 2.5는 연속 오디오 잠재 변수에 플로우 매칭을 사용합니다. AudioLDM 2는 텍스트-멜-오디오 디퓨전을 적용합니다.

2024-2026 트렌드: 플로우 매칭은 음악 생성에서 우세합니다(빠른 추론, 깨끗한 샘플). 반면 토큰-AR은 자연스러운 인과성과 스트리밍 특성으로 인해 음성 생성 분야에서 여전히 강세를 보이고 있습니다.

## 프로덕션 환경 개요

| 시스템 | 작업 | 백본 | 지연 시간 |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + 신경망 보코더 | ~300ms 첫 토큰 |
| OpenAI GPT-4o audio | 전이중 음성 | 엔드투엔드 멀티모달 AR | ~200ms |
| NaturalSpeech 3 | TTS | 잠재 흐름 매칭 | 비스트리밍 |
| Stable Audio 2.5 | 음악 / SFX | DiT + 오디오 잠재 공간 흐름 매칭 | ~10초(1분 클립 기준) |
| Suno v4 | 전체 곡 | 미공개; Token-AR 추정 | ~30초(곡당) |
| Udio v1.5 | 전체 곡 | 미공개 | ~30초(곡당) |
| MusicGen 3.3B | 음악 | Encodec 32kHz 기반 Token-AR | 실시간 |
| AudioCraft 2 | 음악 + SFX | 흐름 매칭 | ~5초(5초 클립 기준) |
| Riffusion v2 | 음악 | 스펙트로그램 디퓨전 | ~10초 |

## 구축 방법

`code/main.py`는 핵심 아이디어를 시뮬레이션합니다: 두 가지 서로 다른 "스타일"(스타일 A는 낮은 토큰과 높은 토큰이 번갈아 나타남, 스타일 B는 단조 증가하는 램프)에서 생성된 합성 "오디오 토큰" 시퀀스에 대해 소형 다음 토큰 예측 트랜스포머를 학습시키고, 스타일을 조건으로 하여 샘플링합니다.

### 1단계: 합성 오디오 토큰 생성

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "음성 유사": 번갈아 나타남
        return [i % vocab_size for i in range(length)]
    # "음악 유사": 램프
    return [(i * 3) % vocab_size for i in range(length)]
```

### 2단계: 소형 토큰 예측기 학습

스타일을 조건으로 하는 바이그램(bigram) 스타일 예측기입니다. 핵심은 패턴입니다: 코덱 토큰 → 교차 엔트로피(cross-entropy) 학습 → 자기회귀적 샘플링.

### 3단계: 조건부 샘플링

스타일 토큰과 시작 토큰이 주어졌을 때, 예측된 분포에서 다음 토큰을 샘플링합니다. 20-40개 토큰에 대해 이 과정을 반복합니다.

## 함정

- **코덱 품질이 출력 품질을 제한합니다.** 코덱이 소리를 충실히 표현할 수 없다면 생성기 품질이 아무리 높아도 도움이 되지 않습니다. DAC는 현재 공개된 최고의 코덱입니다.
- **RVQ 오차 누적.** 각 RVQ 레이어는 이전 레이어의 잔차를 모델링합니다. 레이어 1의 오차가 전파됩니다. 상위 레이어에서 온도 0으로 샘플링하면 도움이 됩니다.
- **음악 구조.** 30초 토큰은 75Hz에서 20,000개 이상의 토큰입니다. 트랜스포머에게는 어렵습니다. MusicGen은 슬라이딩 윈도우 + 프롬프트 연속화를 사용하며, Stable Audio는 짧은 클립 + 크로스페이딩을 사용합니다.
- **경계면 아티팩트.** 생성된 클립 간 크로스페이딩에는 신중한 오버랩-추가 기법이 필요합니다.
- **정제된 데이터 요구량.** 음악 생성기는 수만 시간 분량의 라이선스 음악이 필요합니다. Suno / Udio의 RIAA 소송(2024)이 이 문제를 부각시켰습니다.
- **음성 복제 윤리 문제.** 3초 샘플과 텍스트 프롬프트만으로 VALL-E / XTTS / ElevenLabs가 음성을 복제할 수 있습니다. 모든 프로덕션 모델은 악용 탐지 + 옵트아웃 목록이 필요합니다.

## 사용 방법

| 작업 | 2026 스택 |
|------|------------|
| 상업용 TTS | ElevenLabs, OpenAI TTS, 또는 Azure Neural |
| 음성 복제 (동의 확인 완료) | XTTS v2 (오픈) 또는 ElevenLabs Pro |
| 배경 음악, 빠른 생성 | Stable Audio 2.5 API, Suno, 또는 Udio |
| 가사가 있는 음악 | Suno v4 또는 Udio v1.5 |
| 사운드 효과 / 폴리 | AudioCraft 2, ElevenLabs SFX, 또는 Stable Audio Open |
| 실시간 음성 에이전트 | GPT-4o 실시간 또는 Gemini Live |
| 오픈 웨이트 음악 연구 | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| 더빙 / 번역 | HeyGen, ElevenLabs Dubbing |

## Ship It

`outputs/skill-audio-brief.md`를 저장하세요. 이 스킬은 오디오 브리프(작업, 지속 시간, 스타일, 목소리, 라이선스)를 입력으로 받아 다음을 출력합니다: 모델 + 호스팅, 프롬프트 형식(장르 태그, 스타일 설명자, 구조적 마커), 코덱 + 생성기 + 보코더 체인, 시드 프로토콜, 그리고 평가 계획(MOS / CLAP 점수 / TTS용 CER / 사용자 A/B 테스트).

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하고 스타일을 명시적으로 설정하세요. 생성된 시퀀스가 스타일 패턴과 일치하는지 확인하세요.
2. **중간.** 지연 병렬 디코딩 추가: 1단계 오프셋을 유지해야 하는 2개의 토큰 스트림을 시뮬레이션하세요. 공동 예측기를 학습시키세요.
3. **어려움.** HuggingFace transformers를 사용하여 MusicGen-small을 로컬에서 실행하세요. 서로 다른 3개의 프롬프트로 10초 클립을 생성하고 스타일 준수 여부를 A/B 테스트하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 코덱(Codec) | "신경망 압축" | 오디오용 인코더/디코더; 일반적인 출력은 50-75 Hz 토큰입니다. |
| RVQ | "잔차 VQ" | K개의 양자화기 캐스케이드; 각각은 이전 양자화기의 잔차를 모델링합니다. |
| 토큰(Token) | "하나의 코덱 심볼" | 코드북 내 이산 인덱스; 일반적으로 1024 또는 2048입니다. |
| 지연 병렬(Delayed parallel) | "오프셋 코덱" | 시퀀스 길이 감소를 위해 K개의 토큰 스트림을 오프셋으로 발행합니다. |
| 플로우 매칭(Flow matching) | "2024년 오디오 분야 승자" | 확산 모델의 직선 경로 대안; 더 빠른 샘플링이 가능합니다. |
| 음성 프롬프트(Voice prompt) | "3초 샘플" | 복제된 음성을 유도하는 화자 임베딩 또는 토큰 접두사입니다. |
| 멜 스펙트로그램(Mel spectrogram) | "시각적 표현" | 로그-크기 지각 스펙트로그램; 많은 TTS 시스템에서 사용됩니다. |
| 보코더(Vocoder) | "멜에서 웨이브로" | 멜 스펙트로그램을 오디오로 변환하는 신경망 구성 요소입니다. |

## 프로덕션 노트: 오디오는 스트리밍 문제

오디오는 사용자가 *생성되는 대로* 받기를 기대하는 유일한 출력 방식입니다. 프로덕션 측면에서 이는 TPOT(Time Per Output Token)가 중요함을 의미합니다. 사용자의 청취 속도가 목표 처리량이 되기 때문입니다. 16kHz 오디오가 초당 약 75토큰(Encodec 기준)으로 토큰화되는 경우, 서버는 재생 부드러움을 유지하기 위해 사용자당 초당 75토큰 이상을 생성해야 합니다.

두 가지 아키텍처 결과:

- **플로우 매칭 오디오 모델은 간단히 스트리밍할 수 없습니다.** Stable Audio 2.5와 AudioCraft 2는 한 번의 패스로 고정 클립 길이를 렌더링합니다. 스트리밍을 위해서는 클립을 청크로 나누고 경계를 겹쳐야 합니다. 슬라이딩 윈도우 디퓨전(sliding-window diffusion)을 생각해 보면, 코덱 AR 모델에 비해 100-300ms의 지연 오버헤드가 추가됩니다.

제품이 "라이브 음성 채팅" 또는 "실시간 음악 이어하기"라면 코덱 AR 경로를 선택하세요. "제출 시 30초 클립 렌더링"이라면 플로우 매칭이 품질과 총 지연 시간 측면에서 우수합니다.

## 추가 자료

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — 코덱 표준.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 최초로 널리 사용된 신경망 오디오 코덱.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — 2025년 텍스트-음악 생성(flow matching).