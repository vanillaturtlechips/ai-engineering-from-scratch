# 캡스톤 04 — 멀티모달 문서 QA (비전-퍼스트 PDF, 표, 차트)

> 2026년 문서-QA 최전선은 OCR-then-text에서 벗어나 비전-퍼스트 후기 상호작용으로 이동했습니다. ColPali, ColQwen2.5, ColQwen3-omni는 각 PDF 페이지를 이미지로 처리하고, 멀티벡터 후기 상호작용으로 임베딩하며, 쿼리가 패치에 직접 어텐션(attention)할 수 있게 합니다. 금융 10-K 문서, 과학 논문, 필기 노트에서 이 패턴은 OCR-퍼스트보다 큰 격차로 우수합니다. 10,000페이지 규모의 파이프라인을 엔드 투 엔드로 구축하고 OCR-then-text와 비교한 결과를 공개하세요.

**유형:** 캡스톤  
**언어:** Python (파이프라인), TypeScript (뷰어 UI)  
**선수 조건:** 4단계(컴퓨터 비전), 5단계(NLP), 7단계(트랜스포머), 11단계(LLM 엔지니어링), 12단계(멀티모달), 17단계(인프라)  
**적용 단계:** P4 · P5 · P7 · P11 · P12 · P17  
**소요 시간:** 30시간

## 문제

기업들은 OCR 파이프라인이 망가뜨리는 PDF들을 보유하고 있습니다: 회전된 표가 포함된 스캔된 10-K 문서, 수식이 빽빽한 과학 논문, 이미지로만 의미 있는 차트, 수기 주석 등. 이들을 텍스트 중심으로 처리하면 신호의 절반을 잃게 됩니다. 2026년의 해결책은 원시 페이지 이미지에 대한 후기 상호작용 다중 벡터 검색입니다. ColPali(Illuin Tech)가 이를 도입했으며, ColQwen2.5-v0.2와 ColQwen3-omni는 정확도를 향상시켰습니다. ViDoRe v3에서 비전 우선 검색은 OCR-후 텍스트 검색보다 의미 있는 차이로 높은 점수를 기록했으며, 차트, 표, 수기 내용에서 이 격차는 더욱 벌어집니다.

트레이드오프는 저장 공간과 지연 시간입니다. ColQwen 임베딩은 페이지당 단일 1024차원 벡터가 아닌 약 2048개의 패치 벡터입니다. 원시 저장 공간이 급증합니다. DocPruner(2026)는 측정 가능한 정확도 손실 없이 50% 가지치기를 제공합니다. 10,000페이지를 인덱싱하고, ViDoRe v3 nDCG@5를 측정하며, 2초 이내에 응답을 제공하고, OCR-후 텍스트 베이스라인과 직접 비교해야 합니다.

## 개념

지연 상호작용(late interaction)은 모든 쿼리 토큰(query token)이 모든 패치 토큰(patch token)과 점수를 매기고, 쿼리 토큰당 최대 점수를 합산하는 방식을 의미합니다. 단일 풀링 벡터(pooled vector)가 필요 없이 세밀한 매칭(fine-grained matching)을 얻을 수 있습니다. 멀티 벡터 인덱스(multi-vector index, 예: Vespa, Qdrant 멀티 벡터, AstraDB)는 패치별 임베딩(per-patch embeddings)을 저장하고 검색 시 MaxSim을 실행합니다.

답변자(answerer)는 쿼리와 상위 k개 검색된 페이지를 이미지로 입력받아 증거 영역(bounding boxes 또는 페이지 참조)을 포함한 답변을 작성하는 비전-언어 모델(vision-language model)입니다. Qwen3-VL-30B, Gemini 2.5 Pro, InternVL3가 2026년 최신 선택지입니다. 수식 및 과학 표기법(science notation)의 경우 OCR 폴백(fallback, 예: Nougat, dots.ocr)이 선택적 텍스트 채널로 통합됩니다.

평가는 2차원 행렬로 진행됩니다. 한 축은 콘텐츠 유형(평문 단락, 밀집 표, 막대/선 그래프, 필기 노트, 수식)입니다. 다른 축은 검색 접근법(비전 우선 지연 상호작용 vs OCR-후 텍스트 vs 하이브리드)입니다. 각 셀에는 nDCG@5와 답변 정확도가 기록됩니다. 보고서가 최종 산출물입니다.

## 아키텍처

```
PDFs -> 페이지 렌더러 (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 임베딩 (페이지당 멀티벡터, ~2048 패치)
           |
           +------> DocPruner 50% 압축
           |
           v
   멀티벡터 인덱스 (Vespa 또는 Qdrant 멀티벡터)
           |
쿼리 ----+----> 상위-k 페이지 검색 (MaxSim)
           |
           v
  VLM 답변기: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    입력: 쿼리 + 상위-k 페이지 이미지 + 선택적 OCR 텍스트
           |
           v
  인용된 페이지 번호 + 증거 영역이 포함된 답변
           |
           v
  Streamlit / Next.js 뷰어: 소스 페이지의 강조 표시된 박스
```

## 스택

- 페이지 렌더링: PyMuPDF (fitz) 180 DPI, 세로-정규화
- 후기 상호작용 모델: ColQwen2.5-v0.2 또는 ColQwen3-omni (Hugging Face의 vidore 팀)
- 인덱스: 다중 벡터 필드가 있는 Vespa, 또는 Qdrant 다중 벡터, 또는 MaxSim이 있는 AstraDB
- 프루닝: DocPruner 2026 정책 (고분산 패치 유지, 50% 압축 시 < 0.5% 정확도 손실)
- OCR 대체(방정식/밀집 테이블): dots.ocr 또는 Nougat
- VLM 답변자: 자체 호스팅 Qwen3-VL-30B 또는 호스팅된 Gemini 2.5 Pro; 대체용 InternVL3
- 평가: ViDoRe v3 벤치마크, 다단계 추론을 위한 M3DocVQA
- 뷰어 UI: 증거 영역 캔버스 오버레이가 있는 Next.js 15

## 구축 단계

1. **데이터 수집.** 10-K 문서, 과학 논문, 스캔 문서 등 10,000페이지 분량의 코퍼스를 처리합니다. 각 페이지를 1536x2048 PNG 이미지로 렌더링합니다. `{doc_id, page_num, image_path}` 형식으로 저장합니다.

2. **임베딩 생성.** 각 페이지 이미지에 ColQwen2.5-v0.2 모델을 실행합니다. 출력 형태는 2048개 패치의 128차원 임베딩입니다. DocPruner를 적용하여 신호 강도가 높은 상위 50% 패치만 유지합니다. Vespa 멀티벡터 필드 또는 Qdrant 멀티벡터 저장소에 기록합니다.

3. **검색.** 들어오는 각 쿼리에 대해 쿼리 타워(단어 수준 임베딩)로 임베딩을 생성합니다. 인덱스에 대해 MaxSim을 실행합니다: 모든 쿼리 토큰에 대해 페이지 패치 임베딩과의 최대 내적값을 계산한 후 합산합니다. 상위 k개 페이지를 반환합니다.

4. **합성.** 쿼리와 상위 5개 페이지 이미지를 Qwen3-VL-30B에 입력합니다. 프롬프트: "제공된 페이지만 사용하여 답변하세요. 각 주장을 (doc_id, 페이지)로 인용하고 영역 유형(그림, 표, 단락)을 명시하세요."

5. **증거 영역 추출.** 답변에서 인용된 영역을 후처리합니다. VLM이 바운딩 박스를 출력하면(Qwen3-VL은 가능), 뷰어에 오버레이로 표시합니다.

6. **OCR 대체 경로.** 이미지 분산도(heuristic)로 방정식 밀집 페이지로 식별된 경우, Nougat 또는 dots.ocr을 실행하고 OCR 텍스트를 이미지 채널과 함께 추가합니다.

7. **평가.** ViDoRe v3(검색 nDCG@5)와 M3DocVQA(다중 페이지 QA 정확도)를 실행합니다. 동일한 코퍼스와 합성기로 OCR-텍스트 파이프라인도 실행하고, 콘텐츠 유형 × 접근법 행렬을 생성합니다.

8. **UI.** 초기 프로토타입은 Streamlit으로 개발, 프로덕션 뷰어는 페이지 단위 증거 영역 오버레이가 있는 Next.js 15로 구현합니다.

## 사용 방법

```
$ doc-qa ask "EMEA 부문의 2024년 운영 마진 변화는 무엇이었나요?"
[retrieve]   320ms 내 상위 5개 페이지 검색 (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4초 소요, 인용 (form-10k-2024, p. 88) + (..., p. 92)
답변:
  EMEA 운영 마진은 18.2%에서 16.8%로 140bp 하락했습니다.
  인용: 10-K-2024.pdf p.88 (표 4, 부문별 운영 마진)
        10-K-2024.pdf p.92 (MD&A, 운영 성과)
[viewer]     p.88 표 4에 강조된 경계 상자가 겹쳐진 상태로 열기
```

## Ship It

`outputs/skill-doc-qa.md`는 결과물을 설명합니다: 특정 코퍼스에 맞춰 조정된 비전 우선 멀티모달 문서 QA 시스템이며, ViDoRe v3에서 OCR-텍스트 기준선과 비교하여 평가됩니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 정확도 | OCR-텍스트 기준선 및 공개된 리더보드 대비 벤치마크 수치 |
| 20 | 증거 영역 그라운딩 | 인용된 영역 중 실제 답변 스팬을 포함하는 비율 |
| 20 | 저장 및 지연 시간 엔지니어링 | DocPruner 압축 비율, 인덱스 p95, 답변 p95 |
| 20 | 다중 페이지 추론 | 수작업으로 라벨링된 100개 질문 다중 페이지 세트 정확도 |
| 15 | 소스 검사 UX | 뷰어 명확성, 오버레이 정확도, 나란히 비교 도구 |
| **100** | | |

## 연습 문제

1. 동일한 코퍼스에서 ColQwen2.5-v0.2와 ColQwen3-omni를 비교 측정하세요. 어떤 페이지에서 하나는 정답을 내고 다른 하나는 놓치는지 확인하세요. 인덱스에 "콘텐츠 클래스" 태그를 추가하여 유형별로 라우팅할 수 있게 하세요.

2. 임베딩을 공격적으로 프루닝(75%, 90%)하세요. 압축 한계점(compression cliff)을 찾으세요: ViDoRe nDCG@5가 OCR 기준선 아래로 떨어지는 지점입니다.

3. 하이브리드 시스템을 구축하세요: OCR-텍스트와 ColQwen을 병렬로 실행하고 RRF(Reciprocal Rank Fusion)로 융합한 후 크로스-인코더로 재순위화하세요. 하이브리드가 단독 시스템보다 성능이 우수한지 확인하세요. 어디에서 가장 도움이 되는지 분석하세요.

4. Qwen3-VL-30B를 더 작은 VLM(Qwen2.5-VL-7B)으로 교체하세요. 정확도-비용 곡선을 측정하세요.

5. 필기 노트 지원을 추가하세요. 필기 코퍼스를 렌더링하고 ColQwen으로 임베딩한 후 검색 성능을 측정하세요. 필기 OCR 파이프라인과 비교하세요.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | "ColPali-style retrieval" | 쿼리 토큰이 페이지 패치와 독립적으로 점수를 매김; MaxSim이 집계 |
| Multi-vector | "Per-patch embedding" | 각 문서는 하나의 풀링된 벡터가 아닌 여러 벡터를 가짐 |
| MaxSim | "Late-interaction scoring" | 모든 쿼리 토큰에 대해 문서 벡터 중 최대 유사도를 선택; 합산 |
| DocPruner | "Patch compression" | 2026년 정확도 손실 없이 패치의 50%를 유지하는 프루닝 기법 |
| ViDoRe v3 | "Document-retrieval benchmark" | 시각적 문서 검색 측정을 위한 2026년 표준 |
| Evidence region | "Cited bounding box" | 답변 범위를 지역화하는 소스 페이지의 경계 상자(bbox) |
| OCR fallback | "Equation channel" | 방정식 또는 표가 많은 페이지를 위해 비전과 함께 사용되는 텍스트 파이프라인 |

## 추가 자료

- [ColPali (Illuin Tech) 저장소](https://github.com/illuin-tech/colpali) — 레퍼런스 후기 상호작용 문서 검색
- [ColPali 논문 (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — 기초 방법론 논문
- [Hugging Face의 ColQwen 패밀리](https://huggingface.co/vidore) — 프로덕션 준비 완료 체크포인트
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — 멀티페이지 멀티모달 RAG 베이스라인
- [Vespa 멀티벡터 튜토리얼](https://docs.vespa.ai/en/colpali.html) — 레퍼런스 서빙 스택
- [Qdrant 멀티벡터 지원](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 대체 인덱스
- [AstraDB 멀티벡터](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 대체 관리형 인덱스
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 수식 인식 OCR 대체 솔루션