# 음악 생성 — MusicGen, Stable Audio, Suno, 그리고 라이선스 지진

> 2026년 음악 생성: Suno v5와 Udio v4가 상용 시장을 주도; MusicGen, Stable Audio Open, ACE-Step이 오픈소스 분야를 선도. 기술적 문제는 대부분 해결됨. 법적 문제(Warner Music 5억 달러 합의, UMG 합의)가 2025-2026년 업계 판도를 재편).

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** 6단계 · 02(스펙트로그램), 4단계 · 10(확산 모델)
**소요 시간:** ~75분

## 문제 정의

텍스트 → 30초에서 4분 길이의 음악 클립 생성, 가사, 보컬, 구조 포함. 세 가지 하위 문제:

1. **악기 연주 생성.** "따뜻한 키(key)가 있는 로파이 힙합 드럼"과 같은 텍스트 → 오디오. MusicGen, Stable Audio, AudioLDM.
2. **노래 생성 (보컬 + 가사 포함).** "비 오는 텍사스 밤에 관한 컨트리 노래" → 완성된 노래. Suno, Udio, YuE, ACE-Step.
3. **조건부/제어 가능 생성.** 기존 클립 확장, 브릿지 재생성, 장르 변경, 스템 분리, 인페인팅. Udio의 인페인팅 + 스템 분리는 2026년 목표 기능.

## 개념

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

## 신경망 코덱 토큰 기반 토큰 언어 모델

Meta의 **MusicGen** (2023, MIT) 및 많은 파생 모델: 텍스트/멜로디 임베딩에 조건화, EnCodec 토큰(32kHz, 4개 코덱북)을 자동회귀적으로 예측, EnCodec으로 디코딩. 300M - 3.3B 파라미터. 강력한 베이스라인; 30초 이상 생성 시 어려움.

**ACE-Step** (오픈소스, 4B XL 2026년 4월 출시)는 전체 곡 가사 조건 생성을 위해 이를 확장. 오픈 커뮤니티에서 Suno에 가장 근접한 모델.

## 멜 또는 잠재 공간 기반 확산 모델

**Stable Audio (2023)** 및 **Stable Audio Open (2024)**: 압축된 오디오에 대한 잠재 확산. 루프, 사운드 디자인, 앰비언트 텍스처 생성에 강점. 구조화된 전체 곡 생성에는 부적합.

**AudioLDM / AudioLDM2**: T2I(텍스트-이미지) 스타일 잠재 확산을 통한 텍스트-오디오 생성, 음악, 효과음, 음성으로 일반화.

## 하이브리드(프로덕션) — Suno, Udio, Lyria

폐쇄형 가중치. AR 코덱 언어 모델 + 확산 기반 보코더에 전문 음성/드럼/멜로디 헤드 결합 가능성. Suno v5(2026)는 ELO 1293 품질 기준 선두. Udio v4는 인페인팅 + 스템 분리(베이스, 드럼, 보컬 별도 다운로드) 추가.

## 평가

- **FAD(Fréchet Audio Distance).** VGGish 또는 PANNs 특징을 사용한 생성 오디오와 실제 오디오 분포 간 임베딩 수준 거리. 낮을수록 우수. MusicGen small: MusicCaps에서 4.5 FAD; SOTA ~3.0.
- **음악성(주관적).** 인간 선호도. Suno v5 ELO 1293이 선두.
- **텍스트-오디오 정렬도.** 프롬프트와 출력 간 CLAP 점수.
- **음악성 아티팩트.** 박자 불일치 전환, 보컬 프레이즈 드리프트, 30초 이후 구조 손실.

## 2026 모델 맵

| 모델 | 파라미터 | 길이 | 보컬 | 라이선스 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30초 | 없음 | MIT |
| Stable Audio Open | 1.2B | 47초 | 없음 | Stability 비상업적 |
| ACE-Step XL (2026년 4월) | 4B | 2분 이상 | 있음 | Apache-2.0 |
| YuE | 7B | 2분 이상 | 있음, 다국어 | Apache-2.0 |
| Suno v5 (폐쇄형) | ? | 4분 | 있음, ELO 1293 | 상용 |
| Udio v4 (폐쇄형) | ? | 4분 | 있음 + 스템 | 상용 |
| Google Lyria 3 (폐쇄형) | ? | 실시간 | 있음 | 상용 |
| MiniMax Music 2.5 | ? | 4분 | 있음 | 상용 API |

## 법적 환경 (2025-2026)

- **Warner Music vs Suno 합의.** $500M. WMG는 이제 Suno에서 AI-유사성, 음악 권리, 사용자 생성 트랙에 대한 감독 권한을 가짐. Udio에서도 유사한 UMG 합의 발생.
- **EU AI 법** + **캘리포니아 SB 942**: AI 생성 음악은 반드시 공개되어야 함.
- **Riffusion / MusicGen**은 MIT 라이선스로 준수 부담이 없지만 상업적 보컬도 없음.

안전한 배포 패턴:

1. 인스트루멘탈만 생성 (MusicGen, Stable Audio Open, MIT/CC0 출력물).
2. 상용 API 사용 (Suno, Udio, ElevenLabs Music) + 생성별 라이선스.
3. 소유 또는 라이선스를 받은 카탈로그로 학습 (대부분의 기업이 이 방식으로 전환).
4. 생성물에 워터마크 + 메타데이터 태그 추가.

## 구축 방법

## 1단계: MusicGen으로 생성

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

세 가지 크기: `small` (300M, 빠름), `medium` (1.5B), `large` (3.3B). "아이디어가 통하는지" 확인하려면 small로 충분합니다.

## 2단계: 멜로디 조건화

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody는 크로마그램을 받아 음색을 바꾸면서 멜로디를 보존합니다. "이 멜로디를 현악 사중주로 연주해 줘"와 같은 요청에 유용합니다.

## 3단계: FAD 평가

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

VGGish 임베딩 거리를 계산합니다. 장르 수준의 회귀 테스트에 유용하지만, 인간 평가자를 대체할 수는 없습니다.

## 4단계: LLM-음악 워크플로우에 추가

레슨 7-8의 아이디어와 결합:

```python
prompt = "30초 재즈 루프를 작성해. 드럼, 베이스, 피아노 보이싱을 설명해."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 사용 방법

| 목표 | 스택 |
|------|-------|
| 악기 사운드 디자인 | Stable Audio Open |
| 게임 / 적응형 음악 | Google Lyria RealTime (폐쇄형) |
| 보컬이 포함된 풀 송 (상용) | Suno v5 또는 Udio v4 (명시적 라이선스) |
| 보컬이 포함된 풀 송 (오픈) | ACE-Step XL 또는 YuE |
| 짧은 광고 징글 | MusicGen (허밍 참조 멜로디 조건부 생성) |
| 뮤직비디오 배경 | MusicGen + Stable Video Diffusion

## 2026년에도 여전히 발생하는 문제점

- **저작권 세탁 프롬프트.** "테일러 스위프트 스타일의 노래" — 상용 Suno/Udio는 현재 이를 필터링하지만, 오픈 모델은 그렇지 않습니다. 자체 필터 목록을 추가하세요.
- **30초 이후 반복/표류.** AR 모델은 루프를 형성합니다. 여러 생성물을 크로스페이드하거나, 구조적 일관성을 위해 ACE-Step을 사용하세요.
- **템포 표류.** 모델이 BPM에서 벗어납니다. 프롬프트에 BPM 태그를 사용하고, librosa의 `beat_track`으로 후처리 필터링하세요.
- **보컬 가독성.** Suno는 우수하지만, 오픈 모델은 단어 표현이 흐릿한 경우가 많습니다. 가사가 중요하다면 상용 API를 사용하거나 파인튜닝(fine-tuning)하세요.
- **모노 출력.** 오픈 모델은 모노 또는 가짜 스테레오를 생성합니다. 적절한 스테레오 재구성(ezst, Cartesia의 스테레오 디퓨전)으로 업그레이드하세요.

## Ship It

`outputs/skill-music-designer.md`로 저장합니다. 음악 생성 모델 배포를 위한 모델 선택, 라이선스 전략, 길이/구조 계획, 공개 메타데이터를 결정합니다.

## 1. 모델 선택
- **기본 모델**: [MusicGen](https://github.com/facebookresearch/audiocraft) (Meta)  
  - 텍스트/멜로디 입력 → 고품질 음악 생성 지원
  - 1.5B 파라미터 모델로 실시간 생성 가능
- **대체 모델**: [MusicLM](https://github.com/google-research/google-research/tree/master/musiclm) (Google)  
  - 복잡한 텍스트 설명 기반 음악 생성 특화
  - 리소스 요구량이 높아 고사양 서버 필요

## 2. 라이선스 전략
- **오픈소스 컴포넌트**:  
  - MusicGen 모델 가중치: [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (비상업적 사용만 허용)
  - 커스텀 프론트엔드 코드: [MIT License](https://opensource.org/licenses/MIT)
- **상용 API**:  
  - 분당 생성 비용 $0.01 (10,000자 한도)
  - 엔터프라이즈 계약 시 월 $499 무제한 플랜 제공

## 3. 길이/구조 계획
| 트랙 유형 | 최대 길이 | 구조 템플릿 |
|----------|-----------|-------------|
| 로파이    | 3:30      | Intro(30s) → Verse(1m) → Chorus(45s) ×2 |
| EDM      | 4:00      | Build-up(1m) → Drop(30s) → Breakdown(45s) |
| 클래식   | 2:30      | Theme(45s) → Variation(1m) → Coda(45s) |

## 4. 공개 메타데이터
```json
{
  "model_card": {
    "model_name": "Skill-Music-Designer",
    "created": "2023-11-20",
    "authors": [
      {"name": "AI Music Labs", "role": "training"},
      {"name": "SoundCraft", "role": "evaluation"}
    ],
    "training_data": {
      "sources": ["FMA-Large", "LICENSED_ARTISTS_2023"],
      "license": "CC-BY-SA 3.0"
    },
    "ethics": {
      "bias_mitigation": "다양성 샘플링 + 장르 균형 가중치 적용",
      "disclaimer": "생성된 음악은 상업적 사용 전 저작권 검토 필요"
    }
  },
  "deployment": {
    "endpoint": "/v1/generate",
    "rate_limit": "100 requests/minute",
    "supported_formats": ["wav", "mp3"]
  }
}
```

## 5. 배포 체크리스트
- [ ] 모델 서빙: PyTorch Serve + NVIDIA T4 GPU
- [ ] 모니터링: Prometheus + Grafana 대시보드
- [ ] 로깅: ELK 스택 (Elasticsearch, Logstash, Kibana)
- [ ] 보안: API 키 인증 + IP 화이트리싱

> **주의**: 배포 전 반드시 [모델 카드 표준](https://www.modelcardtools.github.io/modelcards/) 검증 필수!

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. ASCII 기호로 표현된 "생성형" 코드 진행 + 드럼 패턴이 생성됩니다 — 음악 생성 카툰입니다. 원한다면 MIDI 렌더러를 통해 재생할 수 있습니다.
2. **중간.** `audiocraft`를 설치하고, MusicGen-small로 4개의 장르 프롬프트에 걸쳐 10초 클립을 생성한 후, 참조 장르 세트에 대해 FAD(Fréchet Audio Distance)를 측정하세요.
3. **어려움.** ACE-Step(또는 MusicGen-melody)을 사용하여 동일한 멜로디에 대해 서로 다른 음색 프롬프트로 3가지 변형을 생성하세요. 프롬프트와의 일치도를 검증하기 위해 CLAP 유사도를 계산하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| FAD | Audio FID | 실제 및 생성된 오디오 임베딩 분포 간 프레셰(Fréchet) 거리. |
| Chromagram | 멜로디를 음높이로 표현 | 프레임당 12차원 벡터; 멜로디 조건화(conditioning) 입력. |
| Stems | 악기 트랙 | 분리된 베이스/드럼/보컬/멜로디 WAV 파일. |
| Inpainting | 구간 재생성 | 시간 창 마스킹; 모델이 해당 부분만 재생성. |
| CLAP | 텍스트-오디오 CLIP | 대조 학습(contrastive learning) 기반 오디오-텍스트 임베딩; 텍스트-오디오 정렬 평가. |
| EnCodec | 음악 코덱 | MusicGen에서 사용하는 Meta의 신경망 코덱; 32kHz, 4개 코덱북(codebook). |

## 추가 자료

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 오픈 오토리그레시브 벤치마크.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 사운드 디자인 기본 모델.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 오픈 4B 풀송 생성기, 2026년 4월.
- [Suno v5 플랫폼 문서](https://suno.com) — 상업적 품질 선두주자.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 음악 및 사운드 효과를 위한 잠재 확산 모델.
- [WMG-Suno 합의 내용](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025년 11월 선례.