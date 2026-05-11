# 캡스톤 02 — 코드베이스 기반 RAG(교차 리포지토리 시맨틱 검색)

> 2026년의 모든 진지한 엔지니어링 조직은 문자열이 아닌 의미를 이해하는 내부 코드 검색 시스템을 운영합니다. Sourcegraph Amp, Cursor의 코드베이스 답변, Augment의 엔터프라이즈 그래프, Aider의 리포지토리 맵, Pinterest의 내부 MCP — 모두 같은 형태입니다. 여러 리포지토리를 수집하고, tree-sitter로 파싱하며, 함수 및 클래스 수준의 청크를 임베딩하고, 하이브리드 검색, 재순위 지정, 인용을 포함한 답변을 제공합니다. 이 캡스톤에서는 10개 리포지토리에 걸쳐 200만 줄의 코드를 처리하고, 모든 git 푸시 시 증분 재색인화를 견딜 수 있는 시스템을 구축하도록 요청합니다.

**유형:** 캡스톤  
**언어:** Python(데이터 수집), TypeScript(API + UI)  
**선수 조건:** 5단계(NLP 기초), 7단계(트랜스포머), 11단계(LLM 엔지니어링), 13단계(도구), 17단계(인프라)  
**연습 단계:** P5 · P7 · P11 · P13 · P17  
**소요 시간:** 30시간

## 문제

2026년까지 모든 프론티어 코딩 에이전트는 코드베이스 검색 레이어를 탑재하게 될 것입니다. 왜냐하면 컨텍스트 윈도우만으로는 크로스-리포지토리(cross-repo) 질문을 해결할 수 없기 때문입니다. Claude의 1M-토큰 컨텍스트는 도움이 되지만, 순위 기반 검색(ranked retrieval)의 필요성을 없애지는 않습니다. 원시 청크(raw chunks)에 대한 순진한 코사인 유사도 검색은 생성된 코드, 모노레포 중복(monorepo duplication), 그리고 드물게 임포트되는 심볼의 긴 꼬리(long tail) 문제에서 결과를 오염시킵니다. 프로덕션 수준의 해결책은 재순위화기(re-ranker)를 갖춘 AST-인식 청크(AST-aware chunks)에 대한 하이브리드(밀집 + BM25) 검색이며, 이는 심볼 참조 그래프(symbol reference graph)에 의해 뒷받침됩니다.

이를 실제 플릿(fleet) — 튜토리얼용 리포지토리가 아닌 — 을 인덱싱하고 MRR@10, 인용 충실도(citation faithfulness), 점진적 신선도(incremental freshness)를 측정함으로써 학습합니다. 실패 모드는 인프라적 문제입니다: 100k-파일 모노레포, 파일의 절반을 수정하는 푸시(push), 그리고 네 개의 리포지토리를 교차해야 정확히 답변할 수 있는 쿼리입니다.

## 개념

AST-aware 인제스션 파이프라인은 tree-sitter로 각 파일을 파싱하고, 함수 및 클래스 노드를 추출하며, 고정된 토큰 윈도우가 아닌 노드 경계에서 청킹을 수행합니다. 각 청크는 세 가지 표현을 얻습니다: 밀집 임베딩(Voyage-code-3 또는 nomic-embed-code), 희소 BM25 용어, 짧은 자연어 요약. 요약은 세 번째 검색 가능한 모달리티를 추가합니다 — 사용자가 "X는 어떻게 승인되나요?"라고 질문하면 코드에는 `check_permission`만 있더라도 요약에 "authz"가 언급됩니다.

검색은 하이브리드 방식입니다. 쿼리는 밀집 검색과 BM25 검색을 모두 실행하고, 상위 k개를 병합한 후 교차 인코더 재랭커(Cohere rerank-3 또는 bge-reranker-v2-gemma-2b)에 전달합니다. 재랭킹된 목록은 장문 컨텍스트 합성기(Claude Sonnet 4.7과 프롬프트 캐싱, 또는 Llama 3.3 70B 자체 호스팅)로 전달되며, 모든 주장을 파일 및 라인 범위로 인용하라는 지시가 포함됩니다. 인용이 없는 답변은 후처리 필터에 의해 거부됩니다.

증분 신선도(incremental freshness)는 인프라 문제입니다. Git 푸시는 diff를 트리거합니다: 어떤 파일이 변경되었는지, 어떤 심볼이 변경되었는지. 영향을 받은 청크만 재임베딩됩니다. 영향을 받은 파일 간 심볼 엣지(임포트, 메서드 호출)는 재계산됩니다. 인덱스는 커밋마다 2M 라인을 재처리하지 않고도 일관성을 유지합니다.

## 아키텍처

```
git push --> 웹훅 --> 인제스트 워커 (LlamaIndex 워크플로우)
                           |
                           v
             tree-sitter 파싱 + AST 청크
                           |
            +--------------+----------------+
            v              v                v
          밀집 임베딩      BM25 인덱스       요약 (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      심볼 그래프 (Neo4j / kuzu)
                            |
  쿼리 --> LangGraph 에이전트 (검색 -> 재정렬 -> 합성)
                            |
                            v
                 Claude Sonnet 4.7 1M 컨텍스트
                            |
                            v
                 답변 + 파일:라인 인용
```

## 스택

- **파싱**: 17개 언어 문법(Python, TS, Rust, Go, Java, C++ 등)을 지원하는 tree-sitter
- **밀집 임베딩**: Voyage-code-3(호스팅) 또는 nomic-embed-code-v1.5(자체 호스팅), bge-code-v1 대체
- **희소 인덱스**: BM25F를 사용하는 Tantivy(Rust), 심볼 이름 대 본문에 가중치 적용
- **벡터 DB**: 하이브리드 검색 지원 Qdrant 1.12, 또는 50M 벡터 미만 팀을 위한 pgvector + pgvectorscale
- **청크 요약 모델**: 프롬프트 캐싱된 Claude Haiku 4.5 또는 Gemini 2.5 Flash
- **재랭커**: Cohere rerank-3 또는 자체 호스팅 bge-reranker-v2-gemma-2b
- **오케스트레이션**: 인제스션을 위한 LlamaIndex Workflows, 쿼리 에이전트를 위한 LangGraph
- **합성기**: 프롬프트 캐싱된 Claude Sonnet 4.7(1M 컨텍스트)
- **심볼 그래프**: 임포트 및 호출 엣지를 위한 Neo4j(관리형) 또는 kuzu(임베디드)
- **관측 가능성**: 검색 + 합성 단계별 Langfuse 스팬

## 빌드 방법

1. **인제스션 워커.** 모든 푸시 훅에서 git 히스토리를 반복합니다. 변경된 파일을 수집합니다. 각 파일에 대해 tree-sitter로 파싱하고, 전체 소스 스팬과 함께 함수 및 클래스 노드를 추출합니다. 청크 레코드 `{repo, path, start_line, end_line, symbol, body}`를 출력합니다.

2. **청크 요약기.** 시스템 프리앰블에 프롬프트 캐싱을 사용하여 Haiku 4.5 호출로 청크를 배치 처리합니다. 프롬프트: "이 함수를 한 문장으로 요약하세요. 공개 계약과 부작용을 명시하세요." 요약본을 청크와 함께 저장합니다.

3. **임베딩 풀.** 두 개의 병렬 큐: 밀집(Dense, Voyage-code-3 배치 128) 및 요약(동일 모델, 요약 문자열 대상). 페이로드 `{repo, path, start_line, end_line, symbol, kind}`와 함께 Qdrant에 벡터를 기록합니다.

4. **BM25 인덱스.** 필드 가중치 Tantivy 인덱스: 심볼 이름 가중치 4, 심볼 본문 가중치 1, 요약 가중치 2. "X라는 이름의 함수 찾기"와 "X를 수행하는 함수 찾기" 쿼리를 지원합니다.

5. **심볼 그래프.** 각 청크에 대해 엣지를 기록합니다: 임포트(이 파일이 리포지토리 Z의 심볼 Y를 사용), 호출(이 함수가 클래스 C의 메서드 M을 호출), 상속. kuzu에 저장합니다. 쿼리 시 리포지토리 경계를 넘어 검색을 확장하는 데 사용됩니다.

6. **쿼리 에이전트.** 세 개의 노드가 있는 LangGraph. `retrieve`는 밀집 + BM25를 병렬로 실행하고 (repo, path, symbol)로 중복을 제거합니다. `rerank`는 상위 50개에 대해 교차 인코더를 실행하고 상위 10개를 유지합니다. `synth`는 재순위화된 청크를 컨텍스트로 사용하여 Claude Sonnet 4.7을 호출하고 시스템 프롬프트를 캐시하며 파일:라인 인용을 요구합니다.

7. **인용 강제.** 모델 출력을 파싱합니다. `(repo/path:start-end)` 앵커가 없는 주장은 재요청 플래그 처리되거나 삭제됩니다. 인용된 답변만 사용자에게 반환합니다.

8. **증분 재인덱싱.** 각 웹훅에서 심볼 수준 차이를 계산합니다. 텍스트가 변경된 청크만 재임베딩합니다. 임포트가 변경된 청크에 대해 심볼 엣지를 재계산합니다. 측정: 2M-LOC 플릿에서 50개 파일 푸시가 60초 이내에 재인덱싱됩니다.

9. **평가.** 100개의 크로스 리포지토리 질문에 대해 골드 파일:라인 답변을 라벨링합니다. MRR@10, nDCG@10, 인용 충실도(검증 가능한 앵커가 있는 주장의 비율), p50/p99 지연 시간을 측정합니다.

## 사용 방법

```
$ code-rag ask "S3 멀티파트 중단(multipart abort)이 재시도 예산에 어떻게 통합되어 있나요?"
[검색]  밀집 검색 12개 청크 + BM25 검색 7개 청크, 중복 제거 후 16개 고유 청크
[재순위]  상위 5개 유지 (cohere rerank-3)
[합성]  claude-sonnet-4.7, 캐시 적중률 68%, 2.1초
답변:
  멀티파트 중단은 `AbortMultipartOnFail`에 의해 트리거되며,
  services/uploader/retry.go:122-148에 구현되어 있습니다. 이는
  config/budgets.yaml:34-51에 정의된 버킷별 재시도 예산을 감소시킵니다...
  인용: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Ship It

제공할 기술 `outputs/skill-codebase-rag.md`. 주어진 리포지토리 코퍼스를 기반으로 인제스션 파이프라인, 하이브리드 인덱스, 쿼리 에이전트를 구축하고, 리포지토리 간 질문에 대한 인용된 답변을 반환합니다. 평가 기준:

| 가중치 | 평가 항목 | 측정 방법 |
|:-:|---|---|
| 25 | 검색 품질 | 100개 질문 홀드아웃 세트에서 MRR@10 및 nDCG@10 |
| 20 | 인용 충실도 | 검증 가능한 파일:라인 앵커가 있는 답변 주장의 비율 |
| 20 | 지연 시간 및 확장성 | 인덱싱된 코퍼스 크기에서 10k QPS 시 p95 쿼리 지연 시간 |
| 20 | 증분 인덱싱 정확성 | 50개 파일 커밋에서 git 푸시부터 검색 가능까지의 시간 |
| 15 | UX 및 답변 포맷팅 | 인용 클릭 가능성, 스니펫 미리보기, 후속 조치 가능성 |
| **100** | | |

## 연습 문제

1. Voyage-code-3를 nomic-embed-code 자체 호스팅 버전으로 교체하세요. MRR@10 변화량을 측정하세요. 재순위 지정(re-ranking) 활성화 시 격차가 줄어드는지 보고하세요.

2. 코퍼스에 20% 생성된 코드(LLM 생성 보일러플레이트)를 주입하고 재평가를 수행하세요. 검색 오염(retrieval poisoning) 현상을 관찰하세요. 페이로드에 "generated" 플래그를 추가하고 해당 검색 결과의 가중치를 낮추세요.

3. Qdrant 하이브리드 검색과 pgvector + pgvectorscale을 코퍼스 크기에서 벤치마크하세요. 배치 크기 1에서 p99(99번째 백분위 수) 성능을 보고하세요.

4. 샘플링 기반 드리프트 검사 추가: 매주 100개 질문 평가를 재실행하세요. MRR@10 하락이 5% 초과 시 알림을 발생시키세요.

5. 크로스 언어 심볼 해석으로 확장: gRPC를 통해 Go 서비스를 호출하는 Python 함수를 구현하세요. 심볼 그래프를 사용하여 이들을 연결하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| AST-aware chunking | "함수 수준 분할" | 고정된 토큰 윈도우 대신 tree-sitter 노드 경계에서 코드 분할 |
| Hybrid search | "밀집 + 희소" | BM25와 벡터 검색을 병렬로 실행, 상위 k개 병합 후 재순위 지정 |
| Cross-encoder rerank | "2단계 순위 지정" | (쿼리, 후보) 쌍에 점수를 매기는 모델, 코사인 유사도보다 정확도 높음 |
| Prompt caching | "캐시된 시스템 프롬프트" | 반복되는 접두사 토큰을 최대 90%까지 할인하는 2026년 Claude/OpenAI 기능 |
| Symbol graph | "코드 그래프" | 파일 및 저장소 간 임포트, 호출, 상속 관계를 나타내는 엣지 |
| Citation faithfulness | "근거 기반 답변 비율" | 사용자가 앵커를 클릭해 참조된 스팬을 읽고 확인할 수 있는 주장의 비율 |
| Incremental re-index | "푸시-검색 시간" | git 푸시부터 변경된 심볼이 검색 가능해질 때까지의 벽시계 시간 |

## 추가 자료

- [Sourcegraph Amp](https://ampcode.com) — 프로덕션 크로스-리포지토리 코드 인텔리전스
- [Sourcegraph Cody RAG 아키텍처](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 이 캡스톤 프로젝트의 참조 심층 분석
- [Aider 리포지토리 맵](https://aider.chat/docs/repomap.html) — 트리-시터(Tree-sitter) 기반 리포지토리 뷰
- [Augment Code 엔터프라이즈 그래프](https://www.augmentcode.com) — 상용 심볼-그래프 RAG
- [Qdrant 하이브리드 검색 문서](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 참조 구현
- [Voyage AI 코드 임베딩](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 상세 정보
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 크로스-인코더(Cross-Encoder) 참조
- [Pinterest MCP 내부 검색](https://medium.com/pinterest-engineering) — 내부 플랫폼 참조