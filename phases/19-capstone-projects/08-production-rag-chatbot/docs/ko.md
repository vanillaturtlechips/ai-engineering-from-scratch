# 캡스톤 08 — 규제 대상 산업을 위한 프로덕션 RAG 챗봇

> Harvey, Glean, Mendable, LlamaCloud는 모두 2026년에 동일한 프로덕션 형태를 운영합니다. 문서 수집에는 docling 또는 Unstructured를 사용하고, 시각적 데이터에는 ColPali를 활용합니다. 하이브리드 검색 방식을 채택하며, bge-reranker-v2-gemma로 재랭킹합니다. 프롬프트 캐싱(60-80% 히트율)을 사용하여 Claude Sonnet 4.7로 합성합니다. Llama Guard 4와 NeMo Guardrails로 보호합니다. Langfuse와 Phoenix로 모니터링하며, 200개 질문 골든 세트에서 RAGAS로 평가합니다. 규제 도메인(법률, 임상, 보험)에서 구축하고, 골든 세트 통과, 레드 팀 테스트, 드리프트 대시보드 검증을 완료하면 캡스톤이 완료됩니다.

**유형:** 캡스톤  
**언어:** Python(파이프라인 + API), TypeScript(챗 UI)  
**선수 과목:** 5단계(NLP), 7단계(transformers), 11단계(LLM 엔지니어링), 12단계(멀티모달), 17단계(인프라), 18단계(안전성)  
**적용 단계:** P5 · P7 · P11 · P12 · P17 · P18  
**소요 시간:** 30시간

## 문제

규제 분야 RAG(법적 계약서, 임상시험 프로토콜, 보험 정책)는 2026년 가장 많이 배포된 프로덕션 형태일 것입니다. ROI(투자 수익률)가 명확하고 이해관계자가 구체적이기 때문입니다. Harvey(Allen & Overy)는 법률 분야를 위해 이를 구축했습니다. Mendable은 개발자 문서 버전을 제공합니다. Glean은 기업 검색을 다룹니다. 패턴은 다음과 같습니다: 고정밀 데이터 수집, 하이브리드 검색 및 재랭킹, 인용 강제 및 프롬프트 캐싱 기반 합성, 다중 안전 계층 보호, 지속적인 드리프트 모니터링.

어려운 부분은 모델이 아니라 다음과 같습니다:  
- 관할권 인식 준수(HIPAA, GDPR, SOC2)  
- 인용 수준 감사 가능성  
- 비용 제어(프롬프트 캐싱은 적중률이 높을 때 60-90% 할인 효과)  
- RAGAS 충실도 기반 환각 탐지  
- 소스 문서가 업데이트되지만 인덱스가 따라잡지 못할 때 드리프트 탐지  

이 캡스톤 프로젝트에서는 200개 질문 골든 세트와 레드 팀 스위트를 함께 사용하여 이 모든 것을 구현해야 합니다.

## 개념

파이프라인은 두 가지 측면을 가집니다. **수집(Ingestion)**: docling 또는 Unstructured가 구조화된 문서를 파싱하고, ColPali는 시각적으로 풍부한 문서를 처리합니다. 청크(chunk)는 요약, 태그, 역할 기반 접근 제어 레이블을 부여받습니다. 벡터는 pgvector + pgvectorscale(50M 벡터 미만) 또는 Qdrant Cloud에 저장되며, 희소(sparse) BM25가 함께 실행됩니다. **대화(Conversation)**: LangGraph가 메모리와 다중 턴(turn)을 처리하며, 각 쿼리는 하이브리드 검색을 실행하고 bge-reranker-v2-gemma-2b로 재랭킹하며, Claude Sonnet 4.7(프롬프트 캐싱 적용)로 종합합니다. 출력은 Llama Guard 4와 NeMo Guardrails를 거쳐 인용(anchored) 기반 응답을 생성합니다.

평가 스택은 4계층으로 구성됩니다. **골든 세트**(200개의 레이블이 지정된 Q/A 및 인용문)로 정확성을 검증합니다. **레드 팀**(탈옥 시도, PII 추출 시도, 도메인 외 질문)으로 안전성을 테스트합니다. **RAGAS**는 매 턴마다 충실도/답변 관련성/컨텍스트 정밀도를 자동 평가합니다. **드리프트 대시보드**(Arize Phoenix)는 검색 품질과 환각(hallucination) 점수를 주간 단위로 모니터링합니다.

프롬프트 캐싱은 비용 절감의 핵심 요소입니다. Claude 4.5+와 GPT-5+는 시스템 프롬프트 + 검색된 컨텍스트를 캐싱할 수 있습니다. 60-80%의 캐시 적중률에서 쿼리당 비용이 3-5배 감소합니다. 파이프라인은 안정적인 접두사(시스템 프롬프트 + 재랭킹된 컨텍스트 우선)를 설계해 높은 캐시 적중률을 달성해야 합니다.

## 아키텍처

```
문서 (계약서, 프로토콜, 정책)
      |
      v
DocLing / 비구조화 파싱 + ColPali (시각 자료 처리)
      |
      v
청크 + 요약 + 역할 라벨 + 관할권 태그
      |
      v
pgvector + pgvectorscale + BM25 (Tantivy)
      |
쿼리 + 역할 + 관할권
      |
      v
LangGraph 대화형 에이전트
   +--- 검색 (하이브리드)
   +--- 역할 + 관할권 기준 필터링
   +--- 재순위화 (bge-reranker-v2-gemma-2b 또는 Voyage rerank-2)
   +--- 종합 (Claude Sonnet 4.7, 프롬프트 캐싱)
   +--- 보호 (Llama Guard 4 + NeMo Guardrails + Presidio 출력 PII 제거)
   +--- 출처 표기 + 반환
      |
      v
평가:
  RAGAS 충실도 / 답변 관련성 / 컨텍스트 정확도 (온라인)
  Langfuse 주석 큐 (샘플링)
  Arize Phoenix 드리프트 (주간)
  레드 팀 테스트 (출시 전)
```

## 스택

- **수집(Ingestion)**: 구조화 문서에는 Unstructured.io 또는 docling; 시각적으로 풍부한 PDF에는 ColPali  
- **벡터 DB**: 50M 벡터 미만 시 pgvector + pgvectorscale; 그 외 경우 Qdrant Cloud  
- **희소(Sparse)**: 필드 가중치를 적용한 Tantivy BM25  
- **오케스트레이션**: 수집 시 LlamaIndex Workflows, 대화 시 LangGraph  
- **재순위기(Re-ranker)**: 자체 호스팅 bge-reranker-v2-gemma-2b 또는 호스팅된 Voyage rerank-2  
- **LLM**: 프롬프트 캐싱이 적용된 Claude Sonnet 4.7; 대체 모델로 자체 호스팅 Llama 3.3 70B  
- **평가(Eval)**: 온라인 RAGAS 0.2, 환각(hallucination) 및 탈옥(jailbreak) 테스트용 DeepEval  
- **관측 가능성(Observability)**: 주석 큐가 있는 자체 호스팅 Langfuse; 드리프트(drift) 감지용 Arize Phoenix  
- **가드레일(Guardrails)**: 입력/출력 분류기 Llama Guard 4, 정책 NeMo Guardrails v0.12, PII 제거 Presidio  
- **규정 준수(Compliance)**: 청크에 역할 기반 접근 제어 레이블; GDPR/HIPAA를 위한 관할권 태그

## 빌드 방법

1. **데이터 수집.** Unstructured 또는 docling으로 코퍼스(1000-10000 문서)를 파싱합니다. 스캔된 문서나 시각적 요소가 많은 페이지의 경우 ColPali를 통해 라우팅합니다. 요약, 역할 레이블, 관할권 태그가 포함된 청크를 생성합니다.

2. **인덱싱.** Voyage-3 또는 Nomic-embed-v2를 사용한 밀집 임베딩을 pgvector + pgvectorscale에 저장합니다. Tantivy를 통해 BM25 사이드 인덱스를 구축합니다. 역할 및 관할권 필터는 페이로드로 처리합니다.

3. **하이브리드 검색.** 역할+관할권으로 먼저 필터링한 후, 밀집 검색과 BM25를 병렬로 실행합니다. 상호 순위 융합(reciprocal rank fusion)으로 결과를 병합하고, 상위 20개 결과를 재순위 지정기로 전달합니다. 상위 5개 결과를 합성 단계로 넘깁니다.

4. **프롬프트 캐싱을 통한 합성.** 시스템 프롬프트와 정적 정책을 캐시 헤더에 포함시킵니다. 재순위 지정된 컨텍스트를 캐시 확장으로 사용하고, 사용자 질문을 캐시되지 않은 접미사로 처리합니다. 정상 상태에서 60-80%의 캐시 적중률을 목표로 합니다.

5. **가드레일.** 입력 단계에서는 Llama Guard 4를 사용합니다. NeMo Guardrails는 도메인 외 질문이나 정책 금지 주제를 차단합니다. Presidio는 출력에서 실수로 노출된 PII(개인 식별 정보)를 제거합니다. 인용 강제(post-filter)를 적용합니다.

6. **골든 세트.** 도메인 전문가가 (답변, 인용)과 함께 레이블을 지정한 200개의 Q/A 쌍을 구성합니다. 에이전트를 정확한 인용 일치, 답변 정확성, 충실도(RAGAS)로 평가합니다.

7. **레드 팀 테스트.** 50개의 적대적 프롬프트(PAIR, TAP 등 탈옥 시도, PII 유출 시도, 도메인 외 질문, 관할권 간 정보 유출)를 테스트합니다. 통과/실패 및 심각도 기준으로 점수를 매깁니다.

8. **드리프트 대시보드.** Arize Phoenix를 사용해 검색 품질(nDCG, 인용 충실도)을 주간 단위로 추적합니다. 5% 하락 시 경고를 발송합니다.

9. **비용 보고서.** Langfuse로 프롬프트 캐싱 적중률, 쿼리당 토큰 수, 단계별 $/쿼리 내역을 분석합니다.

## 사용 방법

```
$ chat --role=analyst --jurisdiction=GDPR
> 우리 계약 하에서 EU 사용자 프로필에 대한 데이터 보존 의무는 무엇인가요?
[retrieve]  하이브리드 상위 20개 결과 중 GDPR + 분석가 역할 필터 적용
[rerank]    상위 5개 유지
[synth]     claude-sonnet-4.7, 캐시 적중률 74%, 0.8초
답변:
  계약(2024-03-11자 마스터 서비스 계약 제12.4조)은 GDPR 제17조에 따라 EU 사용자 프로필을
  계약 종료 후 30일 이내에 삭제할 것을 의무화하고 있습니다. DPA 개정안(DPA-v2.1 제5조)은
  "제한" 범주 데이터에 대해 이 기간을 14일로 연장합니다.
  인용: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Ship It

`outputs/skill-production-rag.md`는 결과물을 설명합니다. 규정 준수 라벨이 포함된 규제 도메인 챗봇이 루브릭을 통과했으며, 실시간 드리프트 모니터링으로 관찰되었습니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | RAGAS 충실도 + 답변 관련성 | 골든 세트(200개 Q/A)에서의 온라인 점수 |
| 20 | 인용 정확성 | 검증 가능한 소스 앵커가 포함된 답변 비율 |
| 20 | 가드레일 적용 범위 | Llama Guard 4 통과율 + 탈옥 시도 결과 |
| 20 | 비용/지연 시간 엔지니어링 | 프롬프트 캐시 적중률, p95 지연 시간, $/쿼리 |
| 15 | 드리프트 모니터링 대시보드 | 주간 검색 품질 추세를 포함한 Phoenix 실시간 대시보드 |
| **100** | | |

## 연습 문제

1. 다른 관할권(예: GDPR과 함께 HIPAA) 아래에서 두 번째 코퍼스 슬라이스를 구축하세요. 20개 질문으로 구성된 관할권 간 탐색에서 역할+관할권 필터링이 교차 유출을 방지하는 것을 증명하세요.

2. 일주일 간의 프로덕션 트래픽에서 프롬프트 캐시 적중률을 측정하세요. 캐시 접두사를 무효화하는 쿼리를 식별하고 재구성하세요.

3. 10k-토큰 요약 버퍼와 함께 다중 턴 메모리를 추가하세요. 대화가 길어질수록 충실도(faithfulness)가 감소하는지 측정하세요.

4. Claude Sonnet 4.7을 자체 호스팅 Llama 3.3 70B로 교체하세요. $/쿼리와 충실도 변화량(delta)을 측정하세요.

5. "불확실" 모드를 추가하세요: 상위 재순위 점수가 임계값 미만인 경우 에이전트가 답변 대신 "신뢰할 수 있는 인용이 없습니다"라고 응답하게 합니다. 잘못된 자신감(false-confidence) 감소를 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 프롬프트 캐싱 | "캐시된 시스템 + 컨텍스트" | Claude/OpenAI 기능: 캐시된 접두사 토큰이 히트 시 60-90% 할인 적용 |
| RAGAS | "RAG 평가기" | 충실도, 답변 관련성, 컨텍스트 정확도에 대한 자동 평가 |
| 골든 세트 | "레이블된 평가" | 인용이 포함된 200개 이상의 전문가 레이블 Q/A; 정답 데이터 |
| 관할권 태그 | "컴플라이언스 레이블" | 청크에 첨부된 GDPR/HIPAA/SOC2 범위; 검색 필터로 강제 적용 |
| 인용 충실도 | "근거 기반 답변 비율" | 검색 가능한 소스 스팬으로 뒷받침되는 주장의 비율 |
| 드리프트 | "검색 품질 저하" | nDCG 또는 인용 점수의 주간 변화; 경고 임계값 5% |
| 레드 팀 | "적대적 평가" | 출시 전 탈옥 시도, PII 추출, 도메인 외 프로브 테스트 |

## 추가 자료

- [Harvey AI](https://www.harvey.ai) — 참조 법률 생산 스택
- [Glean 기업용 검색](https://www.glean.com) — 대규모 기업용 RAG 참조
- [Mendable 문서](https://mendable.ai) — 개발자 문서 RAG 참조
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 관리형 수집
- [Anthropic 프롬프트 캐싱](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 비용 최적화 참조
- [RAGAS 0.2 문서](https://docs.ragas.io/) — 표준 RAG 평가 프레임워크
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 참조 드리프트 관측 가능성
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026년 안전 분류기
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 정책 레일 프레임워크