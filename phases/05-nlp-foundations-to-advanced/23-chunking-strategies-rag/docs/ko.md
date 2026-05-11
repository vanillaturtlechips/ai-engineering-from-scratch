# RAG를 위한 청킹 전략

> 청킹 구성은 임베딩 모델 선택만큼 검색 품질에 영향을 미칩니다(Vectara NAACL 2025). 청킹을 잘못하면 재순위 지정으로도 해결할 수 없습니다.

**유형:** 구축  
**언어:** Python  
**사전 요구 사항:** Phase 5 · 14 (정보 검색), Phase 5 · 22 (임베딩 모델)  
**소요 시간:** ~60분

## 문제

50페이지 분량의 계약서를 RAG 시스템에 입력했습니다. 사용자가 "해지 조항은 무엇인가요?"라고 질문합니다. 검색기가 표지 페이지를 반환합니다. 왜 그럴까요? 모델이 512토큰 단위로 학습되었고, 해지 조항은 20페이지 내에 위치하며 페이지 구분으로 분할되어 있으며, 쿼리와 연결되는 로컬 키워드가 없기 때문입니다.

해결책은 "더 나은 임베딩 모델을 구입하는 것"이 아닙니다. 해결책은 청킹(chunking)입니다. 청크 크기는 얼마로? 중복(overlap)은 얼마나? 분할 위치는? 주변 컨텍스트는 어떻게?

2026년 2월 벤치마크에서 놀라운 결과가 나타났습니다:

- Vectara의 2026년 연구: 재귀적 512토큰 청킹이 의미론적 청킹보다 69% → 54% 정확도로 우세했습니다.
- SPLADE + Mistral-8B on Natural Questions: 중복은 측정 가능한 이점을 전혀 제공하지 않았습니다.
- 컨텍스트 절벽: 2,500토큰 컨텍스트 주변에서 응답 품질이 급격히 하락했습니다.

"당연한" 해결책(의미론적 청킹, 20% 중복, 1000토큰)은 종종 틀립니다. 이 레슨은 6가지 전략에 대한 직관을 구축하고 어떤 상황에서 어떤 전략을 사용해야 하는지 알려줍니다.

## 개념

![하나의 구절에 시각화된 6가지 청킹 전략](../assets/chunking.svg)

**고정 청킹(Fixed chunking).** 매 N 문자 또는 토큰마다 분할. 가장 간단한 기준선. 문장 중간에서 분할 발생. 압축률은 좋으나 일관성 저하.

**재귀적 청킹(Recursive).** LangChain의 `RecursiveCharacterTextSplitter`. 먼저 `\n\n`으로 분할 시도, 다음 `\n`, `.`, 공백 순으로 시도. 깔끔하게 폴백. 2026년 기본값.

**의미론적 청킹(Semantic).** 각 문장을 임베딩. 인접 문장 간 코사인 유사도 계산. 유사도가 임계값 아래로 떨어지는 지점에서 분할. 주제 일관성 유지. 느림; 때로는 검색에 방해되는 40토큰 미만의 작은 조각 생성.

**문장 청킹(Sentence).** 문장 경계에서 분할. 청크당 하나의 문장 또는 N개 문장 윈도우. 약 5k 토큰까지 의미론적 청킹과 유사한 성능, 비용은 일부.

**부모-문서 청킹(Parent-document).** 검색을 위한 작은 자식 청크와 맥락 제공을 위한 큰 부모 청크 저장. 자식으로 검색, 부모 반환. 우아한 성능 저하: 나쁜 자식 청크도 합리적인 부모 반환.

**지연 청킹(Late chunking, 2024).** 먼저 문서 전체를 토큰 수준에서 임베딩한 후, 토큰 임베딩을 청크 임베딩으로 풀링. 청크 간 맥락 보존. 장문 맥락 임베더(BGE-M3, Jina v3)와 호환. 높은 계산 비용.

**맥락 인식 검색(Contextual retrieval, Anthropic, 2024).** 각 청크 앞에 문서 내 위치에 대한 LLM 생성 요약 추가("이 청크는 해지 조항의 3.2절입니다..."). Anthropic 자체 벤치마크에서 35-50% 검색 성능 향상. 인덱싱 비용 높음.

### 모든 기본값을 능가하는 규칙

청크 크기를 쿼리 유형에 맞추기:

| 쿼리 유형 | 청크 크기 |
|------------|-----------|
| 사실형("CEO 이름은?") | 256-512 토큰 |
| 분석형 / 다중 홉 | 512-1024 토큰 |
| 전체 섹션 이해 | 1024-2048 토큰 |

NVIDIA의 2026년 벤치마크. 청크는 답변과 지역 맥락을 포함할 만큼 충분히 커야 하며, 검색기의 상위-K 결과가 맥락 노이즈보다 답변에 집중하도록 충분히 작아야 함.

## 구축 방법

### 1단계: 고정 및 재귀적 청킹

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 2단계: 의미 기반 청킹

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

`threshold`를 도메인에 맞게 조정하세요. 너무 높으면 → 단편화. 너무 낮으면 → 하나의 거대한 청크.

### 3단계: 부모-문서

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

핵심 통찰: 부모 문서 중복 제거. 여러 자식 청크가 동일한 부모로 매핑될 수 있으며, 모두 반환하면 컨텍스트가 낭비됩니다.

### 4단계: 컨텍스트 기반 검색 (Anthropic 패턴)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

컨텍스트가 추가된 청크를 인덱싱하세요. 쿼리 시, 추가 주변 신호로부터 검색 이점을 얻을 수 있습니다.

### 5단계: 평가

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

항상 벤치마크를 수행하세요. 코퍼스에 대한 "최적" 전략이 블로그 게시물과 일치하지 않을 수 있습니다.

## 함정(Pitfalls)

- **사실형 쿼리(factoid queries)에서만 평가된 청킹(chunking).** 다중 홉 쿼리(multi-hop queries)는 완전히 다른 결과를 보여줍니다. 쿼리 유형-계층화 평가 세트(query-type-stratified eval set)를 사용하세요.
- **최소 크기(minimum size) 없는 의미론적 청킹(semantic chunking).** 40토큰 조각들을 생성하여 검색(retrieval)에 악영향을 줍니다. 항상 `min_tokens`를 강제하세요.
- **미신적 중복(overlap as cargo cult).** 2026개 연구에서 중복(overlap)은 종종 아무런 이점이 없으며 인덱스 비용을 2배로 늘립니다. 가정하지 말고 측정하세요.
- **최소/최대 크기 제한 없음.** 5토큰 또는 5000토큰 청크(chunk) 모두 검색을 방해합니다. 크기를 제한하세요(clamp).
- **문서 간 청킹(cross-doc chunking).** 청크가 두 문서에 걸치지 않도록 하세요. 항상 문서별 청킹 후 병합하세요.

## 사용 방법

2026 스택:

| 상황 | 전략 |
|-----------|----------|
| 첫 구축, 알 수 없는 코퍼스 | 재귀적, 512 토큰, 중복 없음 |
| 사실형 QA | 재귀적, 256-512 토큰 |
| 분석형 / 다중 홉 | 재귀적, 512-1024 토큰 + 부모 문서 |
| 다량의 상호 참조 (계약서, 논문) | 후기 청킹 또는 컨텍스트 기반 검색 |
| 대화형 / 대화 코퍼스 | 턴 수준 청크 + 화자 메타데이터 |
| 짧은 발화 (트윗, 리뷰) | 하나의 문서 = 하나의 청크 |

재귀적 512로 시작하세요. 50개 쿼리 평가 세트에서 recall@5를 측정하세요. 이후 조정하세요.

## Ship It

`outputs/skill-chunker.md`로 저장:

```markdown
---
name: chunker
description: 주어진 코퍼스와 쿼리 분포에 대해 청킹 전략, 크기, 중첩을 선택합니다.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

코퍼스(문서 유형, 평균 길이, 도메인)와 쿼리 분포(사실형/분석형/다중 홉)가 주어졌을 때 다음 내용을 출력합니다:

1. 전략. 재귀적/문장/의미론적/상위 문서/지연/문맥 기반. 근거.
2. 청크 크기. 토큰 수. 쿼리 유형과 연관된 근거.
3. 중첩. 기본값 0; 0보다 클 경우 정당화.
4. 최소/최대 강제 적용. `min_tokens`, `max_tokens` 보호 장치.
5. 평가 계획. 50개 쿼리 계층화 평가 세트(사실형, 분석형, 다중 홉)에서 Recall@5.

최소/최대 청크 크기 강제 적용이 없는 청킹 전략은 거부합니다. 20% 이상의 중첩은 도움이 된다는 실험 결과(ablation study) 없이는 거부합니다. 최소 토큰 기준이 없는 의미론적 청킹 권장 사항은 플래그를 표시합니다.
```

## 연습 문제

1. **쉬움.** 고정(512, 0), 재귀(512, 0), 재귀(512, 100) 방식으로 20페이지 문서 하나를 청크 분할하세요. 청크 개수와 경계 품질을 비교하세요.
2. **중간.** 5개 문서에 대해 30개 쿼리 평가 세트를 구축하세요. 재귀, 의미 기반, 부모 문서 방식의 recall@5를 측정하세요. 어떤 방식이 가장 좋은 성능을 내나요? 블로그 포스트 결과와 일치하나요?
3. **어려움.** 컨텍스트 기반 검색 기능을 구현하세요. 기준선 재귀 방식 대비 MRR(Mean Reciprocal Rank) 개선도를 측정하세요. 인덱스 비용(LLM 호출 횟수) 대비 정확도 향상률을 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 청크(Chunk) | 문서의 일부 | 임베딩, 인덱싱, 검색이 되는 하위 문서 단위 |
| 오버랩(Overlap) | 안전 마진 | 인접한 청크 간 공유되는 N개의 토큰; 2026년 벤치마크에서는 종종 쓸모없음 |
| 시맨틱 청킹(Semantic chunking) | 스마트 청킹 | 인접 문장 임베딩 유사도가 떨어지는 지점에서 분할 |
| 부모 문서(Parent-document) | 2단계 검색 | 작은 자식 청크를 검색한 후 더 큰 부모 문서를 반환 |
| 레이트 청킹(Late chunking) | 임베딩 후 청킹 | 토큰 수준에서 전체 문서를 임베딩한 후 청크 벡터로 풀링 |
| 컨텍스트 검색(Contextual retrieval) | Anthropic의 트릭 | 인덱싱 전 각 청크 앞에 LLM 생성 요약문을 추가 |
| 컨텍스트 절벽(Context cliff) | 2500토큰 벽 | RAG에서 2.5k 컨텍스트 토큰 주변에서 관찰되는 품질 저하 (2026년 1월 기준) |

## 추가 자료

- [Yepes et al. / LangChain — 재귀적 문자 분할 문서](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 프로덕션 환경의 기본 설정.
- [Vectara (2024, NAACL 2025). 청킹 구성 분석](https://arxiv.org/abs/2410.13070) — 임베딩 선택만큼 청킹이 중요.
- [Jina AI — 장문 컨텍스트 임베딩 모델의 지연 청킹 (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — 지연 청킹 논문.
- [Anthropic — 컨텍스트 기반 검색](https://www.anthropic.com/news/contextual-retrieval) — LLM 생성 컨텍스트 접두어로 검색 성능 35-50% 향상.
- [NVIDIA 2026 청크 크기 벤치마크 — Premai 요약](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 쿼리 유형별 청크 크기.