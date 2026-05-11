# 캡스톤 12 — 비디오 이해 파이프라인 (장면, QA, 검색)

> Twelve Labs는 Marengo + Pegasus를 제품화했습니다. VideoDB는 비디오용 CRUD API를 출시했습니다. AI2의 Molmo 2는 오픈 VLM 체크포인트를 공개했습니다. Gemini 장문 컨텍스트는 비디오 수시간을 네이티브로 처리합니다. TimeLens-100K는 대규모 시간적 그라운딩을 정의했습니다. 2026년 파이프라인은 확정되었습니다: 장면 분할, 장면별 캡션 + 임베딩, 트랜스크립트 정렬, 멀티벡터 인덱스, 그리고 (시작, 종료) 타임스탬프와 프레임 미리보기로 답변하는 쿼리. 캡스톤은 100시간 분량을 수집하며, 공개 벤치마크를 실행하고, 개수 세기 및 행동 관련 질문에 대한 환각 현상을 측정합니다.

**유형:** 캡스톤  
**언어:** Python (파이프라인), TypeScript (UI)  
**선수 조건:** 4단계 (CV), 6단계 (음성), 7단계 (트랜스포머), 11단계 (LLM 엔지니어링), 12단계 (멀티모달), 17단계 (인프라)  
**관련 단계:** P4 · P6 · P7 · P11 · P12 · P17  
**소요 시간:** 30시간

## 문제

장형 비디오 QA는 2026년 규모에서 가장 대역폭을 많이 소모하는 멀티모달 문제입니다. Gemini 2.5 Pro는 2시간 분량의 비디오를 네이티브로 읽을 수 있지만, 100시간 분량의 비디오를 쿼리 가능한 코퍼스로 수집하려면 여전히 장면 수준의 인덱스가 필요합니다. 프로덕션 형태는 장면 분할(TransNetV2 또는 PySceneDetect), VLM(Gemini 2.5, Qwen3-VL-Max 또는 Molmo 2)을 이용한 장면별 캡션 생성, 타임스탬프가 포함된 전사 정렬(Whisper-v3-turbo), 그리고 캡션, 프레임 임베딩, 전사를 나란히 저장하는 멀티벡터 인덱스로 구성됩니다. 쿼리 파이프라인은 (시작, 종료) 타임스탬프와 프레임 미리보기로 답변합니다.

벤치마크는 공개(ActivityNet-QA, NeXT-GQA) 및 자체 100개 쿼리 커스텀 세트로 구성됩니다. 개수 세기 및 동작 유형 질문에 대한 환각(hallucination)은 알려진 어려운 실패 사례이며, 캡스톤에서는 이를 명시적으로 측정합니다.

## 개념

세 개의 파이프라인이 수집 단계에서 병렬로 실행됩니다. **장면 분할(scene segmentation)**은 비디오를 장면 단위로 분할합니다. **VLM 캡셔닝(VLM captioning)**은 각 장면에 대한 캡션과 키프레임에서 추출한 프레임 임베딩을 생성합니다. **ASR 정렬(ASR alignment)**은 단어 수준의 타임스탬프를 생성합니다. 세 스트림은 (scene_id, 시간 범위)로 결합됩니다. 각 장면은 멀티벡터 인덱스(Qdrant)에 세 가지 벡터 유형(캡션 임베딩, 키프레임 임베딩, 트랜스크립트 임베딩)을 저장합니다.

쿼리 시 자연어로 된 질문은 세 벡터 모두에 대해 실행되며, 결과는 RRF(Reciprocal Rank Fusion)로 통합됩니다. 시간 기반 어댑터(TimeLens 스타일)는 상위 장면 내 (시작, 종료) 시간 범위를 정제합니다. VLM 합성기(Gemini 2.5 Pro 또는 Qwen3-VL-Max)는 쿼리 + 상위 장면 + 잘린 프레임을 입력으로 받아 인용 타임스탬프와 프레임 미리보기를 포함한 답변을 생성합니다.

환각(hallucination) 측정은 중요합니다. "방에 몇 명이 들어왔나요?"와 같은 개수 세기 질문이나 "요리사(stirring 전에 pour를 하나요?"와 같은 행동 유형 질문은 신뢰도가 낮은 것으로 알려져 있습니다. 설명형 질문과 구분하여 정확도를 별도로 보고해야 합니다.

## 아키텍처

```
비디오 파일 / URL
      |
      v
PySceneDetect / TransNetV2  (장면 분할)
      |
      +--- 장면별 키프레임 --- VLM 캡션 + 프레임 임베딩
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- 오디오 채널 --- Whisper-v3-turbo ASR + 단어 타임스탬프
      |
      v
멀티벡터 Qdrant: {캡션_임베딩, 키프레임_임베딩, 트랜스크립트_임베딩}
      |
쿼리:
  세 가지 모두에 대한 밀집 검색 -> RRF 병합 -> 상위-k 장면
      |
      v
TimeLens / VideoITG 시간적 근거 (장면 내 시작/종료 정제)
      |
      v
VLM 합성: 쿼리 + 상위 장면 + 프레임 미리보기
      |
      v
답변 + (시작, 종료) 타임스탬프 + 프레임 썸네일 + 인용
```

## 스택

- **장면 분할**: TransNetV2 (state-of-the-art 2024-26) 또는 PySceneDetect  
- **ASR**: Whisper-v3-turbo를 faster-whisper로 구동 (단어 타임스탬프 포함)  
- **VLM 캡션 생성기 + 답변기**: Gemini 2.5 Pro 또는 Qwen3-VL-Max 또는 Molmo 2  
- **시간적 위치 지정**: TimeLens-100K로 훈련된 어댑터 또는 VideoITG  
- **인덱싱**: 멀티벡터 지원 Qdrant (캡션 / 프레임 / 음성 텍스트)  
- **UI**: HTML5 비디오 플레이어 및 장면 썸네일이 포함된 Next.js 15  
- **평가**: ActivityNet-QA, NeXT-GQA, 직접 제작한 100문항 수동 라벨링 세트  
- **환각 벤치마크**: 수동 라벨이 포함된 개수 세기 및 동작 유형 하위 집합

## 빌드하기

1. **비디오 수집기.** YouTube URL 또는 로컬 MP4 파일을 입력받음. 필요 시 720p로 다운스케일. `{video_id, file_path}` 형식으로 저장.

2. **장면 분할.** TransNetV2 또는 PySceneDetect 실행하여 `[{scene_id, start_ms, end_ms, keyframe_path}]` 생성. 100시간 기준 약 6,000~8,000개 장면 목표.

3. **음성 인식(ASR) 처리.** 오디오에 Whisper-v3-turbo 실행; 단어 단위 타임스탬프 추출; 장면별 자막 조각으로 분할.

4. **VLM 캡셔닝.** 각 장면마다 키프레임과 짧은 캡션 템플릿을 Gemini 2.5 Pro(또는 Qwen3-VL-Max)에 전달. 캡션 + 프레임 임베딩 생성.

5. **멀티 벡터 인덱스.** 3개의 명명된 벡터를 가진 Qdrant 컬렉션. 페이로드: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **쿼리 처리.** 자연어 질문이 3개의 밀집 벡터 쿼리를 실행; 상호 순위 융합(RRF)으로 통합; 상위 k=5개 장면 반환.

7. **시간적 위치 지정.** TimeLens 스타일의 어댑터를 상위 장면에 적용하여 장면 내 (시작, 종료) 윈도우를 세분화.

8. **VLM 합성.** 쿼리 + 상위 3개 장면 클립(이미지 또는 짧은 동영상) + 자막을 Gemini 2.5 Pro에 전달. `(video_id, start_ms, end_ms)` 인용 필수.

9. **평가.** ActivityNet-QA와 NeXT-GQA 실행. 100개 쿼리 커스텀 세트 구축. 전체 정확도 + 클래스별 분석(계수, 동작, 설명) 보고.

## 사용 방법

```
$ video-qa ask --url=https://youtube.com/watch?v=X "첫 1분 동안 교차로를 지나는 차량은 몇 대인가요?"
[scene]    23개 장면 감지됨
[asr]      음성 텍스트 변환 완료, 4분 12초
[index]    69개 벡터 생성 (23개 장면 × 3)
[query]    상위 장면: 장면 3 [01:32-01:54], 신뢰도 0.84
[ground]   세부 구간: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4초
답변:      00:12부터 00:58까지 교차로를 지나는 차량은 5대입니다.
출처:      [장면 3: 00:12-00:58]
          [프레임 미리보기: 00:14, 00:27, 00:44, 00:51, 00:57]
```

## Ship It

`outputs/skill-video-qa.md`가 결과물입니다. YouTube URL 또는 업로드된 동영상을 입력하면 파이프라인이 장면을 인덱싱하고 타임스탬프 인용과 함께 질문에 답변합니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 시간적 근거 IoU | 홀드아웃 근거 세트에서의 교집합-합집합 비율 |
| 20 | QA 정확도 | NeXT-GQA 및 커스텀 100개 쿼리 |
| 20 | 수집 처리량 | 1달러당 처리 가능한 동영상 시간 |
| 20 | UI 및 인용 UX | 타임스탬프 링크, 썸네일 스트립, 프레임 이동 |
| 15 | 환각 생성률 | 카운팅 및 동작 유형 정확도 별도 측정 |
| **100** | | |

## 연습 문제

1. 캡션 생성 단계에서 Gemini 2.5 Pro를 Qwen3-VL-Max로 교체해 보세요. 인간 평가된 50개 장면 샘플에 대해 캡션 품질 변화량(delta)을 보고하세요.

2. 장면당 프레임 임베딩을 다중 벡터 대신 하나의 풀링 벡터로 축소하세요. 검색 성능(retrieval) 회귀(regression)를 측정하세요.

3. "엄격한 카운팅" 모드 구축: 합성기가 각 카운팅된 인스턴스를 타임스탬프와 함께 추출하고 사용자가 클릭하여 검증하도록 합니다. 사용자 검증이 환각(hallucination) 감소에 기여하는지 측정하세요.

4. 처리 비용 벤치마크: 세 가지 VLM(Vision-Language Model) 선택 사항에 대해 "달러당 처리 가능한 비디오 시간"을 비교하세요. 최적의 균형점(sweet spot)을 선택하세요.

5. 화자 구분(diarized) 트랜스크립트 추가: 오디오에 pyannote 화자 구분(speaker diarization)을 실행하고 화자별 트랜스크립트를 임베딩하세요. "앨리스가 X에 대해 뭐라고 말했나요?"와 같은 쿼리를 시연해 보세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 장면 분할(Scene segmentation) | "샷 감지(Shot detection)" | 샷 경계에서 비디오를 장면으로 분할 |
| 다중 벡터 인덱스(Multi-vector index) | "캡션 + 프레임 + 대본(Caption + frame + transcript)" | 표현별 명명된 벡터를 포함한 Qdrant 컬렉션 |
| 시간적 근거(Temporal grounding) | "정확히 언제 발생했는가(When exactly did it happen)" | 쿼리 답변에 대한 (시작, 종료) 윈도우 개선 |
| 프레임 임베딩(Frame embedding) | "시각적 표현(Visual representation)" | 키프레임의 벡터 임베딩; 장면-시각적 유사도 계산에 사용 |
| RRF 융합(RRF fusion) | "상호 순위 융합(Reciprocal rank fusion)" | 여러 순위 목록 간 병합 전략; 고전적인 하이브리드 검색 기법 |
| 개수 환각(Counting hallucination) | "개수 오류(Miscount)" | "X의 개수는 몇 개인가" 질문에 대한 VLM의 알려진 실패 모드 |
| ActivityNet-QA | "비디오 QA 벤치마크(Video-QA benchmark)" | 장문 비디오 QA 정확도 벤치마크

## 추가 자료

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 오픈 VLM 체크포인트
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — 대규모 시간적 그라운딩
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — 호스팅된 참조
- [VideoDB](https://videodb.io) — 비디오용 CRUD API 참조
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 상용 참조
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 장면 분할 모델
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 클래식 오픈 대안
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 참조 평가 벤치마크