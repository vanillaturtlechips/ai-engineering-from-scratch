# 왜 멀티 에이전트인가?

> 한 에이전트가 벽에 부딪혔을 때, 현명한 해결책은 더 큰 에이전트가 아니라 더 많은 에이전트입니다.

**유형:** 학습
**언어:** TypeScript
**선수 지식:** 14단계 (에이전트 엔지니어링)
**소요 시간:** ~60분

## 학습 목표

- 단일 에이전트 한계(컨텍스트 오버플로우, 혼합 전문성, 순차적 병목)를 식별하고, 여러 에이전트로 분할해야 하는 적절한 시점을 설명
- 오케스트레이션 패턴(파이프라인, 병렬 팬아웃, 감독자, 계층적)을 비교하고, 주어진 작업 구조에 적합한 패턴 선택
- 명확한 역할 경계, 공유 상태, 통신 계약을 갖춘 다중 에이전트 시스템 설계
- 다중 에이전트 복잡성(지연 시간, 비용, 디버깅 난이도)과 단일 에이전트 단순성 간의 트레이드오프 분석

## 문제

14단계에서 단일 에이전트를 구축했습니다. 그것은 작동합니다. 파일을 읽고, 명령을 실행하며, API를 호출하고, 결과를 추론할 수 있습니다. 그런 다음 실제 코드베이스(200개 파일, 3개 언어, 인프라에 의존하는 테스트, 코드 작성 전 외부 API 연구 필요)를 대상으로 지정합니다.

에이전트가 제대로 작동하지 않습니다. LLM이 무능해서가 아니라, 작업이 하나의 에이전트 루프로 처리할 수 있는 범위를 초과하기 때문입니다. 컨텍스트 창은 파일 내용으로 가득 찹니다. 에이전트는 40번의 도구 호출 전에 읽은 내용을 잊어버립니다. 연구자, 코더, 리뷰어의 역할을 동시에 수행하려고 하며, 세 가지 모두 제대로 해내지 못합니다.

이것이 단일 에이전트 한계입니다. 다음 조건을 요구하는 작업마다 이 한계에 부딪힙니다:

- **하나의 창에 담을 수 있는 컨텍스트 초과** - 50개 파일 읽기는 200k 토큰을 초과합니다
- **단계별 다른 전문성 요구** - 연구에는 코드 생성과 다른 프롬프트 방식이 필요합니다
- **병렬 처리 가능한 작업** - 왜 3개 파일을 순차적으로 읽을 때 동시에 읽을 수 있는데?

## 개념

### 단일 에이전트 한계

단일 에이전트는 하나의 루프, 하나의 컨텍스트 윈도우, 하나의 시스템 프롬프트입니다. 다음과 같이 상상해 보세요:

```
┌─────────────────────────────────────────┐
│            단일 에이전트                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         컨텍스트 윈도우            │  │
│  │                                   │  │
│  │  연구 노트                       │  │
│  │  + 코드 파일                     │  │
│  │  + 테스트 출력                   │  │
│  │  + 리뷰 피드백                   │  │
│  │  + API 문서                      │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ███████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  하나의 시스템 프롬프트가 연구 + 코딩 + 리뷰 + 테스트를 모두 처리하려 함       │
│                                         │
│  결과: 모든 면에서 평범한 성능         │
└─────────────────────────────────────────┘
```

세 가지 문제가 발생합니다:

1. **컨텍스트 포화** - 도구 결과가 계속 쌓입니다. 30턴이 되면 에이전트는 파일 내용, 명령어 출력, 이전 추론으로 150k 토큰을 소비합니다. 5턴에서 나온 중요한 세부 사항이 사라집니다.

2. **역할 혼란** - "당신은 연구자, 코더, 리뷰어, 테스터입니다"라는 시스템 프롬프트는 반쯤 연구하고, 반쯤 코딩하며, 리뷰는 절대 끝내지 못하는 에이전트를 생성합니다.

3. **순차적 병목 현상** - 에이전트는 파일 A를 읽고, 파일 B를 읽고, 파일 C를 읽습니다. 세 번의 직렬 LLM 호출. 세 번의 직렬 도구 실행. 병렬 처리 없음.

### 다중 에이전트 솔루션

작업을 분할하세요. 각 에이전트에게 하나의 작업, 하나의 컨텍스트 윈도우, 해당 작업에 맞춰 조정된 하나의 시스템 프롬프트를 제공하세요:

```
┌──────────────────────────────────────────────────────────┐
│                    오케스트레이터                          │
│                                                          │
│  "사용자 관리를 위한 REST API를 구축하세요"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │연구자     │ │  코더   │ │ 리뷰어   │ │  테스터  │  │
│   │          │ │          │ │          │ │          │  │
│   │ 문서를    │ │ 코드를   │ │ 코드를    │ │ 테스트를 │  │
│   │ 읽고,     │ │ 작성하며 │ │ 확인하고,│ │ 실행하고,│  │
│   │ 패턴을    │ │ 연구 +   │ │ 버그를    │ │ 결과를   │  │
│   │ 찾습니다   │ │ 사양을   │ │ 찾습니다   │ │ 보고합니다│  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     결과 병합                        │
└──────────────────────────────────────────────────────────┘
```

각 에이전트는 다음을 가집니다:
- 집중된 시스템 프롬프트("당신은 코드 리뷰어입니다. 버그를 찾는 것이 유일한 업무입니다.")
- 다른 에이전트의 작업으로 오염되지 않은 자체 컨텍스트 윈도우
- 명확한 입출력 계약(연구 노트를 받아 코드를 출력)

### 실제 시스템 사례

**Claude Code 서브에이전트** - Claude Code가 `Task`로 서브에이전트를 생성하면 범위가 지정된 작업을 가진 자식 에이전트가 생성됩니다. 부모는 컨텍스트를 깨끗하게 유지합니다. 자식은 집중된 작업을 수행하고 요약본을 반환합니다.

**Devin** - 플래너 에이전트, 코더 에이전트, 브라우저 에이전트를 실행합니다. 플래너는 작업을 단계로 나눕니다. 코더는 코드를 작성합니다. 브라우저는 문서를 조사합니다. 각각 별도의 컨텍스트를 가집니다.

**다중 에이전트 코딩 팀 (SWE-bench)** - SWE-bench에서 최고 성능을 내는 시스템은 코드베이스를 읽는 연구자, 수정 사항을 설계하는 플래너, 이를 구현하는 코더를 사용합니다. 단일 에이전트 시스템은 점수가 낮습니다.

**ChatGPT 심층 연구** - 여러 검색 에이전트를 병렬로 생성하여 각각 다른 각도를 탐색한 후 결과를 종합합니다.

### 스펙트럼

다중 에이전트는 이진법이 아닙니다. 스펙트럼입니다:

```
단순 ──────────────────────────────────────────── 복잡

 단일        서브        파이프라인      팀         군집
 에이전트   에이전트

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │공유 │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ 상태 │
             └───┘          └───┘───┘  │  메시지│    └───────┘
                                       │  버스   │
 1 루프      부모 +      단계별        │       │    N 개의 동료,
 1 컨텍스트   자식 작업   처리       └───────┘    출현적
                                       명시적      행동
                                       역할
```

**단일 에이전트** - 하나의 루프, 하나의 프롬프트. 간단한 작업에 적합합니다.

**서브에이전트** - 부모가 집중된 하위 작업을 위해 자식을 생성합니다. 부모는 계획을 유지합니다. 자식은 보고합니다. Claude Code가 이 방식을 사용합니다.

**파이프라인** - 에이전트가 순차적으로 실행됩니다. 에이전트 A의 출력이 에이전트 B의 입력이 됩니다. 단계별 워크플로에 적합합니다: 연구 -> 코드 -> 리뷰 -> 테스트.

**팀** - 공유 메시지 버스를 통해 병렬로 실행되는 에이전트. 각각 역할이 있습니다. 오케스트레이터가 조정합니다. 다양한 기술이 동시에 필요할 때 적합합니다.

**군집** - 공유 상태를 가진 동일하거나 거의 동일한 다수의 에이전트. 고정된 오케스트레이터가 없습니다. 에이전트는 큐에서 작업을 선택합니다. 높은 처리량의 병렬 작업에 적합합니다.

### 네 가지 다중 에이전트 패턴


#### 패턴 1: 파이프라인

```
입력 ──▶ 에이전트 A ──▶ 에이전트 B ──▶ 에이전트 C ──▶ 출력
          (연구)  (코드)      (리뷰)
```

각 에이전트는 데이터를 변환하고 다음 단계로 전달합니다. 추론이 간단합니다. 한 단계의 실패는 나머지 단계를 차단합니다.


#### 패턴 2: 팬아웃 / 팬인

```
                ┌──▶ 에이전트 A ──┐
                │              │
입력 ──▶ 분할 ├──▶ 에이전트 B ──├──▶ 병합 ──▶ 출력
                │              │
                └──▶ 에이전트 C ──┘
```

병렬 에이전트로 작업을 분할한 후 결과를 병합합니다. 독립적인 하위 작업으로 분해되는 작업에 적합합니다.


#### 패턴 3: 오케스트레이터-워커

```
                    ┌──────────┐
                    │  오케스트│
                    └──┬───┬───┘
                  작업 │   │ 작업
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ 워커 A   │   │ 워커 B   │
           └──────────┘   └──────────┘
```

스마트 오케스트레이터가 작업을 결정하고 워커에게 위임한 후 결과를 종합합니다. 오케스트레이터 자체가 워커를 생성하는 도구를 가진 에이전트입니다.


#### 패턴 4: 피어 군집

```
         ┌───┐ ◄──── 메시지 ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      메시지 │    ┌───────────┐     │ 메시지
           └───▶│  공유       │◄────┘
                │  상태       │
           ┌───▶│  / 큐       │◄────┐
           │    └───────────┘     │
      메시지 │                      │ 메시지
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── 메시지 ────▶ │ D │
         └───┘                  └───┘
```

중앙 오케스트레이터가 없습니다. 에이전트는 P2P로 통신합니다. 결정은 상호작용에서 나타납니다. 디버깅은 어렵지만 많은 에이전트로 확장 가능합니다.

### 다중 에이전트를 사용하지 말아야 할 때

다중 에이전트는 복잡성을 추가합니다. 에이전트 간 모든 메시지는 잠재적 실패 지점입니다. 디버깅은 "하나의 대화 읽기"에서 "다섯 에이전트 간 메시지 추적"으로 바뀝니다.

**단일 에이전트를 유지할 때:**
- 작업이 하나의 컨텍스트 윈도우에 적합할 때(작업 데이터 ~100k 토큰 이하)
- 다른 단계에 다른 시스템 프롬프트가 필요하지 않을 때
- 순차적 실행이 충분히 빠를 때
- 작업을 분할하는 것이 가치보다 오버헤드를 더 추가할 정도로 단순할 때

**복잡성 비용:**
- 모든 에이전트 경계는 손실 압축 단계입니다: 에이전트 A의 전체 컨텍스트가 에이전트 B를 위한 메시지로 요약됩니다
- 조정 논리(누가 무엇을, 언제, 어떤 순서로 할지)는 버그 발생 원인이 됩니다
- 지연 증가: N 에이전트는 최소 N개의 직렬 LLM 호출을 의미하며, 상호 작용이 필요하면 더 증가합니다
- 비용 증가: 각 에이전트는 독립적으로 토큰을 소모합니다

경험적 규칙: 작업이 20개 미만의 도구 호출로 완료되고 100k 토큰에 적합하면 단일 에이전트를 유지하세요.

## 구축 방법

### 1단계: 과부하 단일 에이전트

다음은 모든 작업을 수행하려는 단일 에이전트입니다. 하나의 거대한 시스템 프롬프트와 연구, 코드, 리뷰를 보관하는 하나의 컨텍스트 윈도우를 가지고 있습니다:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `당신은 풀스택 개발자입니다. 다음을 수행해야 합니다:
1. 요구사항 연구
2. 코드 작성
3. 코드 버그 검토
4. 테스트 작성
이 모든 작업을 단일 대화에서 수행하십시오.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `연구: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `다음 연구 내용을 기반으로:\n${contextWindow.join("\n")}\n\n이제 ${task}에 대한 코드를 작성하십시오.`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `모든 이전 컨텍스트를 기반으로:\n${contextWindow.join("\n")}\n\n코드를 검토하십시오.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

이 접근 방식의 문제점:
- 컨텍스트 윈도우는 모든 단계에서 증가합니다. 검토 단계에서는 연구 노트, 코드, 이전 추론 내용이 모두 포함됩니다.
- 시스템 프롬프트가 일반적입니다. 각 단계에 맞게 조정할 수 없습니다.
- 아무것도 병렬로 실행되지 않습니다.

### 2단계: 전문 에이전트

이제 분할합니다. 각 에이전트는 하나의 작업을 수행합니다:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "당신은 기술 연구원입니다. 문서를 읽고 패턴을 찾아 결과를 요약하십시오. 구현에 필요한 사실만 출력하십시오."
);

const coder = createSpecialist(
  "coder",
  "당신은 시니어 TypeScript 개발자입니다. 요구사항과 연구 노트를 기반으로 깨끗하고 테스트된 코드를 작성하십시오. 그 외 다른 작업은 하지 마십시오."
);

const reviewer = createSpecialist(
  "reviewer",
  "당신은 코드 검토자입니다. 버그, 보안 문제, 논리 오류를 찾으십시오. 구체적으로 설명하십시오. 라인 번호를 인용하십시오."
);
```

각 전문 에이전트는 집중된 프롬프트를 가지고 있습니다. 각 에이전트는 필요한 입력만 포함된 깨끗한 컨텍스트 윈도우를 받습니다.

### 3단계: 메시지를 통한 조정

명시적 메시지 전달로 전문 에이전트들을 연결합니다:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[${m.from}로부터]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[${m.from}로부터]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

각 에이전트는 자신에게 전달된 메시지만 받습니다. 컨텍스트 오염이 없습니다. 연구자의 50k 토큰 분량의 문서 읽기는 검토자의 컨텍스트에 절대 들어가지 않습니다.

### 4단계: 비교

```typescript
async function compare() {
  const task = "Express.js API를 위한 레이트 리미터 미들웨어를 구축하십시오.";

  console.log("=== 단일 에이전트 ===");
  const single = await singleAgentApproach(task);
  console.log(`토큰: ${single.tokensUsed}`);
  console.log(`툴 호출: ${single.toolCalls}`);

  console.log("\n=== 다중 에이전트 ===");
  const multi = await multiAgentApproach(task);
  console.log(`토큰: ${multi.tokensUsed}`);
  console.log(`툴 호출: ${multi.toolCalls}`);
}
```

다중 에이전트 버전은 총 토큰 사용량이 더 많지만(세 에이전트, 세 번의 별도 LLM 호출), 각 에이전트의 컨텍스트는 깨끗하게 유지됩니다. 시스템 프롬프트가 특화되어 있기 때문에 각 단계의 품질이 향상됩니다.

## 사용 방법

이 레슨은 멀티에이전트 접근 방식을 언제 사용할지 결정하기 위한 재사용 가능한 프롬프트를 생성합니다. `outputs/prompt-multi-agent-decision.md`를 참조하세요.

## 연습 문제

1. 네 번째 전문가 추가: 코더로부터 코드를 받고 리뷰어로부터 피드백을 받은 후 테스트를 작성하는 "테스터" 에이전트 추가  
2. 파이프라인 수정: 리뷰어가 코더에게 피드백을 되돌려 최대 2라운드까지 수정 루프 가능하도록 구현  
3. 순차적 파이프라인 변환: 연구자와 "요구사항 분석가" 에이전트를 병렬로 실행한 후, 출력물을 병합하여 코더에게 전달하는 팬아웃 구조로 변경  

> **번역 규칙 적용 사항**  
> - "tester" → "테스터" (직역보다 역할 설명 강조)  
> - "review feedback" → "리뷰 피드백" (복합 명사 처리)  
> - "revision loop" → "수정 루프" (기술 용어 보존)  
> - "fan-out" → "팬아웃" (시스템 설계 용어 원어 유지)  
> - "merge their outputs" → "출력물을 병합" (동사 중심 번역)

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| 스웜(Swarm) | "AI 에이전트들의 집단 지성" | 공유 상태를 가지며 고정된 리더가 없는 동료 에이전트 집합. 행동은 지역적 상호작용에서 발생한다. |
| 오케스트레이터(Orchestrator) | "보스 에이전트" | 다른 에이전트를 생성하고 관리하는 도구를 가진 에이전트. 계획을 수립하고 작업을 위임하지만 실제 작업을 수행하지 않을 수 있다. |
| 코디네이터(Coordinator) | "교통 경찰" | (LLM이 아닌) 비에이전트 구성 요소(종종 코드). 규칙에 따라 에이전트 간 메시지를 라우팅한다. |
| 합의(Consensus) | "에이전트들이 동의했다" | 여러 에이전트가 진행 전 합의에 도달해야 하는 프로토콜. 상충되는 출력 해결이 필요할 때 사용된다. |
| 창발적 행동(Emergent behavior) | "에이전트들이 스스로 해결했다" | 에이전트 상호작용에서 발생하지만 명시적으로 프로그래밍되지 않은 시스템 수준의 패턴. 유용할 수도 있고 해로울 수도 있다. |
| 팬아웃/팬인(Fan-out / fan-in) | "에이전트용 맵-리듀스" | 작업을 병렬 에이전트들에 분배(팬아웃)한 후 그 결과를 통합(팬인)하는 것. |
| 메시지 전달(Message passing) | "에이전트들이 서로 대화한다" | 에이전트 간 통신 메커니즘: 공유 컨텍스트 창을 대체하는 구조화된 데이터를 한 에이전트에서 다른 에이전트로 전송한다. |

## 추가 자료

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - 다중 에이전트 패턴 조사
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft의 다중 에이전트 대화 프레임워크
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code의 Task 기반 위임 방식
- [CrewAI documentation](https://docs.crewai.com/) - 역할 기반 다중 에이전트 프레임워크