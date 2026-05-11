# 챗봇 — 규칙 기반에서 신경망, LLM 에이전트로

> ELIZA는 패턴 매칭으로 응답했습니다. DialogFlow는 의도(intent)를 매핑했습니다. GPT는 가중치(weights)로부터 답변했습니다. Claude는 도구를 실행하고 검증합니다. 각 시대는 이전 시대의 가장 큰 실패를 해결했습니다.

**유형:** 학습
**언어:** Python
**선수 지식:** 5단계 · 13(질의 응답), 5단계 · 14(정보 검색)
**소요 시간:** ~75분

## 문제 정의

사용자가 "항공편을 변경하고 싶어요."라고 말합니다. 시스템은 사용자가 원하는 것, 누락된 정보, 정보를 얻는 방법, 그리고 작업을 완료하는 방법을 파악해야 합니다. 그런 다음 사용자가 "잠깐, 취소하면 어떻게 되나요?"라고 말하면 시스템은 컨텍스트를 기억하고 작업을 전환하며 상태를 유지해야 합니다.

ML 시스템에게 대화는 어렵습니다. 입력은 개방적입니다. 출력은 여러 차례에 걸쳐 일관성 있어야 합니다. 시스템은 실제 세계에 작용해야 할 수 있습니다(항공편 변경, 카드 결제). 모든 잘못된 단계는 사용자에게 노출됩니다.

챗봇 아키텍처는 네 가지 패러다임을 순환해 왔으며, 각각은 이전 패러다임이 너무 눈에 띄게 실패했기 때문에 도입되었습니다. 이 강의에서는 순서대로 설명합니다. 2026년 프로덕션 환경은 마지막 두 가지의 혼합 형태입니다.

## 개념

![챗봇 진화: 규칙 기반 → 검색 기반 → 신경망 → 에이전트](../assets/chatbot.svg)

**규칙 기반(ELIZA, AIML, DialogFlow).** 수동으로 작성된 패턴이 사용자 입력과 매칭되어 응답을 생성합니다. 의도 분류기(intent classifier)는 미리 정의된 흐름으로 라우팅합니다. 슬롯 채우기(slot-filling) 상태 머신은 필요한 정보를 수집합니다. 설계된 좁은 범위 내에서는 뛰어난 성능을 발휘하지만, 그 범위를 벗어나면 즉시 실패합니다. 환각(hallucination)이 허용되지 않는 안전-중요 도메인(은행 인증, 항공사 예약)에서는 여전히 사용됩니다.

**검색 기반(Retrieval-based).** FAQ 스타일의 시스템입니다. 모든 (발화, 응답) 쌍을 인코딩합니다. 런타임 시 사용자 메시지를 인코딩하고 가장 유사한 저장된 응답을 검색합니다. Zendesk의 "유사 문서" 기능을 생각해 보세요. 규칙보다 다양한 표현을 더 잘 처리합니다. 생성이 없으므로 환각이 발생하지 않습니다.

**신경망(seq2seq).** 대화 로그로 훈련된 인코더-디코더 모델입니다. 처음부터 응답을 생성합니다. 유창하지만 일반적인 출력("모르겠어요")과 사실적 오류(factual drift)가 발생하기 쉽습니다. 주제를 일관되게 유지하지 못합니다. 2016-2019년 Google, Facebook, Microsoft의 실망스러운 챗봇 실패 원인입니다.

**LLM 에이전트.** 계획 수립, 도구 호출, 결과 검증을 반복하는 루프로 언어 모델을 감싼 시스템입니다. 긴 프롬프트를 가진 챗봇이 아닙니다. 에이전트 루프: 계획 → 도구 호출 → 결과 관찰 → 다음 단계 결정. 검색 기반 근거(RAG)는 환각을 방지합니다. 도구 호출을 통해 실제 작업을 수행할 수 있습니다. 2026년 표준 아키텍처입니다.

네 가지 패러다임은 순차적 대체 관계가 아닙니다. 2026년 프로덕션 챗봇은 네 가지 모두를 활용합니다: 인증과 파괴적 작업에는 규칙 기반, FAQ에는 검색 기반, 자연스러운 표현에는 신경망 생성, 모호한 개방형 질의에는 LLM 에이전트를 라우팅합니다.

## 구축 방법

### 1단계: 규칙 기반 패턴 매칭

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20줄로 구현한 ELIZA. 반사 기법("I feel sad" → "Why do you feel sad")은 1966년 바이젠바움의 정신치료 데모의 정석입니다. 여전히 교육적입니다.

### 2단계: 검색 기반 (FAQ)

이 예시 코드는 `pip install sentence-transformers`(torch 포함)가 필요합니다. 이 레슨의 실행 가능한 `code/main.py`는 외부 종속성 없이 표준 라이브러리 자카드 유사도를 사용합니다.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

임계값 기반 거절이 핵심 설계 선택입니다. 최고 일치도가 충분히 높지 않으면 `None`을 반환하고 시스템이 에스컬레이션하도록 합니다.

### 3단계: 신경망 생성 (기준선)

소규모 지시 튜닝된 인코더-디코더(FLAN-T5) 또는 파인튜닝된 대화형 모델을 사용합니다. 2026년에는 단독으로는 프로덕션에 부적합하지만(모순, 주제 이탈, 사실적 오류), 자연스러운 표현을 위해 하이브리드 시스템에 탑재됩니다. DialoGPT 스타일의 디코더 전용 모델은 일관된 응답을 생성하기 위해 명시적인 턴 구분자와 EOS 처리가 필요합니다. FLAN-T5 텍스트2텍스트 파이프라인은 교육용 예제로 바로 작동합니다.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 4단계: LLM 에이전트 루프

2026년 프로덕션 형태:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            result = tools[tool_name](**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

세 가지 핵심 요소. 도구는 LLM이 호출할 수 있는 함수입니다. 루프는 LLM이 도구 호출 대신 최종 답변을 반환할 때 종료됩니다. 단계 예산은 모호한 작업에서 무한 루프를 방지합니다.

실제 프로덕션에는 다음이 추가됩니다: 검색 기반 근거(매 LLM 호출 전 관련 문서 주입), 가드레일(확인 없이 파괴적 작업 거부), 관측 가능성(모든 단계 로깅), 평가(에이전트 동작이 사양에 맞는지 자동 확인).

### 5단계: 하이브리드 라우팅

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

패턴: 파괴적 작업에는 결정적 규칙, FAQ에는 검색, 그 외 모든 작업에는 LLM 에이전트. 2026년 고객 지원 시스템에 탑재되는 방식입니다.

## 사용 방법

2026 스택:

| 사용 사례 | 아키텍처 |
|---------|---------------|
| 예약, 결제, 인증 | 규칙 기반 상태 머신 + 슬롯 채우기 |
| 고객 지원 FAQ | 선별된 답변에 대한 검색(retrieval) |
| 개방형 도움말 채팅 | RAG + 도구 호출(tool calls)을 사용하는 LLM 에이전트 |
| 내부 도구 / IDE 어시스턴트 | 도구 호출(검색, 읽기, 쓰기)을 사용하는 LLM 에이전트 |
| 동반자 / 캐릭터 챗봇 | 페르소나 시스템 프롬프트가 적용된 튜닝된 LLM, 지식 검색(retrieval) |

프로덕션 환경에서는 항상 하이브리드 라우팅(hybrid routing)을 사용하세요. 단일 아키텍처로는 모든 요청을 효과적으로 처리할 수 없습니다. 라우팅 계층 자체는 일반적으로 작은 의도 분류기(intent classifier)로 구성됩니다.

## 여전히 발생하는 실패 모드

- **자신감 있는 허위 생성.** LLM 에이전트가 수행하지 않은 작업을 완료했다고 주장합니다. 완화 방법: 결과 확인, 도구 호출 로깅, 성공적인 도구 반환 없이 LLM이 작업을 수행했다고 주장하지 않도록 합니다.
- **프롬프트 인젝션.** 사용자가 시스템 프롬프트를 덮어쓰는 텍스트를 삽입합니다. OWASP Top 10 for LLM Applications 2025에서 LLM01로 분류되었습니다. 두 가지 유형: 직접 인젝션(채팅에 붙여넣기)과 간접 인젝션(에이전트가 읽는 문서, 이메일 또는 도구 출력에 숨겨짐).

  공격률은 시나리오에 따라 다릅니다. 일반 도구 사용 및 코딩 벤치마크에서 프론티어 모델 기준 성공률은 ~0.5-8.5%로 측정되었습니다. 특정 고위험 설정(AI 코딩 에이전트에 대한 적응형 공격, 취약한 오케스트레이션)에서는 ~84%에 달했습니다. 실제 CVE 사례에는 EchoLeak(CVE-2025-32711, CVSS 9.3)이 포함됩니다. 이는 공격자가 제어하는 이메일로 트리거되는 Microsoft 365 Copilot의 제로클릭 데이터 유출 결함입니다.

  완화 방법: 루프 전체에서 사용자 입력을 신뢰하지 않음; 도구 호출 전 입력값 검증; 도구 출력을 메인 프롬프트와 격리; 에이전트가 먼저 계획을 수립한 후 해당 계획에 따라 각 작업을 검증한 후 실행하는 Plan-Verify-Execute(PVE) 패턴 사용(이 방식은 도구 결과가 새로운 계획되지 않은 작업을 주입하는 것을 차단); 파괴적 작업에 대한 사용자 확인 요구; 도구 범위에 최소 권한 적용.

  프롬프트 엔지니어링만으로는 이 위험을 완전히 제거할 수 없습니다. 외부 런타임 방어 계층(LLM Guard, 허용 목록 검증, 의미적 이상 감지)이 필요합니다.
- **범위 확장.** 도구 호출이 관련 없는 정보를 반환하여 에이전트가 작업을 벗어납니다. 완화 방법: 도구 계약 범위 축소; 시스템 프롬프트를 집중 유지; 작업 이탈률 평가 추가.
- **무한 루프.** 에이전트가 동일한 도구를 계속 호출합니다. 완화 방법: 단계 예산 설정, 도구 호출 중복 제거, "진전 중인가?"에 대한 LLM 판단.
- **컨텍스트 창 고갈.** 긴 대화로 인해 초기 대화 내용이 컨텍스트에서 벗어납니다. 완화 방법: 이전 대화 요약, 유사도 기반 관련 과거 대화 검색, 또는 장문 컨텍스트 모델 사용.

## Ship It

`outputs/skill-chatbot-architect.md`로 저장:

```markdown
---
name: chatbot-architect
description: 주어진 사용 사례에 대한 챗봇 스택을 설계합니다.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

제품 컨텍스트(사용자 요구, 규정 준수 제약, 사용 가능한 도구, 데이터 양)가 주어졌을 때 다음을 출력합니다:

1. 아키텍처. 규칙 기반, 검색 기반, 신경망, LLM 에이전트 또는 하이브리드(어떤 경로에 어떤 방식을 적용할지 명시).
2. 적용 가능한 경우 LLM 선택. 모델 패밀리 이름(Claude, GPT-4, Llama-3.1, Mixtral)을 지정합니다. 도구 사용 품질과 비용에 맞게 매칭.
3. 그라운딩 전략. RAG 소스, 검색 방법(레슨 14 참조), 도구 계약.
4. 평가 계획. 작업 성공률, 도구 호출 정확성, 작업 외 요청률, 홀드아웃 대화에서의 환각률.

구조화된 확인 흐름 없이 파괴적 행동(결제, 계정 삭제, 데이터 수정)에 대해 순수 LLM 에이전트를 추천하는 것을 거부합니다. 에이전트가 어떤 것에 쓰기 접근 권한이 있는 경우 프롬프트 인젝션 감사를 건너뛰는 것을 거부합니다.
```

## 연습 문제

1. **쉬움.** 커피숍 주문 봇을 위한 10가지 패턴 기반 응답 시스템을 구현하세요. 테스트 케이스: 중복 주문, 수정 요청, 취소, 불명확한 의도.
2. **중간.** 하이브리드 FAQ + LLM 폴백 시스템을 구축하세요. SaaS 제품용 50개의 사전 정의된 FAQ 항목과 문서 사이트 검색 기반 LLM 폴백을 구현하세요. 100개의 실제 지원 질문에 대해 거부율과 정확도를 측정하세요.
3. **어려움.** 세 가지 도구(검색, 사용자 데이터 읽기, 이메일 전송)를 사용하는 에이전트 루프를 구현하세요. 프롬프트 인젝션 시도를 포함한 50개의 테스트 시나리오로 평가를 실행하세요. 작업 이탈률, 실패한 작업률, 인젝션 성공 여부를 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 인텐트(Intent) | 사용자가 원하는 것 | 범주형 라벨 (book_flight, reset_password). 핸들러로 라우팅됨. |
| 슬롯(Slot) | 정보 조각 | 봇이 필요로 하는 파라미터 (날짜, 목적지). 슬롯 채우기(slot filling)는 질문 시퀀스의 연속. |
| RAG(Retrieval-Augmented Generation) | 검색 + 생성 | 관련 문서를 검색한 후 LLM의 응답을 근거화(ground)함. |
| 툴 호출(Tool call) | 함수 호출 | LLM이 이름과 인수를 포함한 구조화된 호출을 방출. 런타임이 실행하고 결과를 반환. |
| 에이전트 루프(Agent loop) | 계획, 실행, 검증 | 작업 완료까지 LLM 호출과 툴 호출을 교차 실행하는 컨트롤러. |
| 프롬프트 인젝션(Prompt injection) | 사용자 공격 프롬프트 | 시스템 프롬프트를 덮어쓰려는 악성 입력. |

## 추가 자료

- [Weizenbaum (1966). ELIZA — 자연어 의사소통 연구를 위한 컴퓨터 프로그램](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 최초의 규칙 기반 챗봇 논문.
- [Thoppilan et al. (2022). LaMDA: 대화 응용 프로그램을 위한 언어 모델](https://arxiv.org/abs/2201.08239) — LLM 에이전트 등장 직전 구글의 신경망 기반 챗봇 논문.
- [Yao et al. (2022). ReAct: 언어 모델에서 추론과 행동 통합](https://arxiv.org/abs/2210.03629) — 에이전트 루프 패턴을 명명한 논문.
- [Anthropic의 효과적인 에이전트 구축 가이드](https://www.anthropic.com/research/building-effective-agents) — 2024년 제작 가이드라인으로 2026년에도 유효.
- [Greshake et al. (2023). 당신이 동의한 것이 아님: 간접 프롬프트 주입을 통한 실제 LLM 통합 애플리케이션 침해](https://arxiv.org/abs/2302.12173) — 프롬프트 주입 논문.
- [OWASP LLM 애플리케이션 Top 10 2025 — LLM01 프롬프트 주입](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 프롬프트 주입을 최고 보안 문제로 지정한 순위.
- [AWS — 간접 프롬프트 주입으로부터 Amazon Bedrock 에이전트 보호](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — Plan-Verify-Execute 및 사용자 확인 흐름을 포함한 실용적인 오케스트레이션 계층 방어.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 간접 프롬프트 주입으로 인한 표준 제로클릭 데이터 유출 CVE. 쓰기 접근 에이전트에게 런타임 방어가 필요한 이유를 보여주는 사례.