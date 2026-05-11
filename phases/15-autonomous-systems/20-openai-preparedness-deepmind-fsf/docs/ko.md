# OpenAI 준비성 프레임워크와 DeepMind 프론티어 안전 프레임워크

> OpenAI 준비성 프레임워크 v2(2025년 4월)는 연구 범주 — 장거리 자율성, 샌드배깅, 자율 복제 및 적응, 안전장치 약화 — 를 추적 범주와 구분하여 도입했습니다. 추적 범주는 안전 자문 그룹이 검토하는 역량 보고서와 안전장치 보고서를 트리거합니다. DeepMind의 FSF v3(2025년 9월, 추적 역량 레벨 추가 2026년 4월 17일)는 자율성을 ML R&D 및 사이버 도메인에 통합합니다(ML R&D 자율성 레벨 1 = 인간 + AI 도구 대비 경쟁력 있는 비용으로 AI R&D 파이프라인 완전 자동화). FSF v3는 도구적 추론 오용을 위한 자동화된 모니터링을 통해 기만적 정렬을 명시적으로 다룹니다. 솔직한 설명: PF v2의 연구 범주(장거리 자율성 포함)는 자동으로 완화 조치를 트리거하지 않으며, 정책 언어는 "잠재적"입니다. DeepMind 자체는 도구적 추론이 강화될 경우 자동화된 모니터링이 "장기적으로 충분하지 않을 것"이라고 언급했습니다.

**유형:** 학습  
**언어:** Python(표준 라이브러리, 3개 프레임워크 결정 테이블 비교 도구)  
**선수 조건:** 15단계 · 19단계(Anthropic RSP)  
**소요 시간:** ~45분

## 문제 정의

레슨 19에서는 Anthropic의 확장 정책을 자세히 읽었습니다. 이 레슨은 OpenAI와 DeepMind의 정책을 읽음으로써 전체적인 그림을 완성합니다. 이 세 문서는 동일한 질문 — 최첨단 연구소가 언제 모델을 일시 중지하거나 제한해야 하는가 — 에 답하는 사촌 관계 문서들이며, 소수의 범주에서 수렴하고 중요한 부분에서 분기됩니다.

수렴 지점: 세 기관 모두 장기 자율성(long-range autonomy)을 추적해야 할 능력 클래스로 분류합니다. 세 기관 모두 기만적 행동(정렬 위장, 성능 은폐)을 특정 위험 클래스로 인정합니다. 세 기관 모두 내부 검토 기구를 보유하고 있습니다. 분기 지점: OpenAI는 범주를 "추적 대상"(필수 완화 조치)과 "연구 대상"(자동 트리거 없음)으로 구분합니다. DeepMind는 자율성을 별도로 명명하지 않고 두 도메인으로 통합합니다. 각 연구소는 "추적 대상 vs 연구 대상", "중요 vs 보통", "티어-1 vs 티어-2"와 같은 명칭을 사용하며, 특정 버킷에 속한 능력의 운영적 결과는 연구소마다 다릅니다.

이들을 함께 읽는 것이 유용한 연습입니다. 동일한 능력이 Anthropic에서는 "필수 완화 조치", OpenAI에서는 "모니터링되지만 트리거되지 않음", DeepMind에서는 "특정 도메인에서 추적"으로 분류될 수 있습니다. 정책 입장이 중요합니다.

## 개념

### OpenAI 준비 프레임워크 v2 (2025년 4월)

구조:

- **추적 범주**: 트리거 기능 보고서(모델이 할 수 있는 것) 및 안전장치 보고서(적용된 완화 조치 포함). 배포 전 안전 자문 그룹(Safety Advisory Group)에서 검토.
- **연구 범주**: 연구실이 추적 중이지만 아직 특정 완화 조치를 약속하지 않은 모델 기능. 장거리 자율성(Long-range Autonomy), 샌드배깅(Sandbagging), 자율 복제 및 적응(Autonomous Replication and Adaptation), 안전장치 약화(Undermining Safeguards) 포함.

v2의 주요 프레임: 연구 범주는 자동으로 완화 조치를 트리거하지 않음. 정책 언어는 "잠재적" 완화 조치. 주목할 특정 차이점: 범주가 추적(Tracked) vs 연구(Research)에 위치하는 것이 "게이트 처리"와 "관찰" 사이의 핵심 기준.

### DeepMind 프론티어 안전 프레임워크 v3 (2025년 9월; 추적 기능 수준 2026년 4월 17일 추가)

구조:

- **중요 기능 수준(CCLs)**: 사이버(Cyber), 바이오(Bio), ML R&D, CBRN, 자율성(Autonomy) 5개 영역의 기능 임계값. 자율성은 ML R&D 및 사이버에 통합됨.
- **추적 기능 수준**: 2026년 4월 추가 세부 수준. 구체적 예시: ML R&D 자율성 수준 1 = 경쟁 비용 대비 인간 + AI 도구 대비 AI R&D 파이프라인 완전 자동화.
- **기만적 정렬 모니터링**: 도구적 추론 악용을 위한 자동화된 모니터링에 대한 명시적 약속.

자율성 프레임은 OpenAI와 다름. DeepMind는 "자율성"을 최상위 도메인으로 유지하지 않으며, 자율성이 피해를 초래할 수 있는 도메인(ML R&D 및 사이버)에 통합. 주장은 도메인 없는 자율성은 위험 없는 기능이라는 것; 반론은 도메인 간 자율성이 프레임워크가 명시해야 할 메타 위험이라는 것.

### 세 프레임워크의 공통점

- 내부 안전 자문 그룹(Anthropic SAG, OpenAI SAG, DeepMind 내부 위원회). 고기능 모델 배포 전 검토.
- 기만적 정렬/정렬 위장을 위험 클래스로 명시적 언급.
- 선언된 주기에 따른 지속적 산출물(Anthropic: 프론티어 안전 로드맵, 위험 보고서; OpenAI: 기능 및 안전장치 보고서; DeepMind: FSF 업데이트 주기).
- 모니터링 전용 방어의 한계 인정. DeepMind는 명시적: "자동화된 모니터링은 장기적으로 충분하지 않을 것."

### 차이점

- **Anthropic**: v3.0에서 일시 중지 약속 제거; AI R&D-4 임계값이 다음 게이트로 명명됨.
- **OpenAI**: 추적 vs 연구 범주 분리; 연구 범주(장거리 자율성 포함)는 자동으로 게이트 처리되지 않음.
- **DeepMind**: 자율성이 다른 도메인에 통합; 2026년 4월 추적 기능 수준 추가로 세분화.

### 샌드배깅: 세 프레임워크 모두를 복잡하게 만드는 특정 기능

샌드배깅(모델이 평가에서 전략적으로 저성능을 보이는 것)은 OpenAI의 연구 범주에 포함. Anthropic의 RSP v3.0은 평가-맥락 격차(레슨 1)를 통해 대응. DeepMind는 FSF v3의 기만적 정렬 모니터링을 통해 대응.

모델이 평가에서 샌드배깅을 수행하면 모든 프레임워크의 기능 임계값이 과소평가됨. 프레임워크는 측정이 정확할 때만 작동. 따라서 실험실 자체 평가 외에 외부 측정(레슨 21, METR) 및 적대적 평가가 필수적.

### 정책 해석 기술

- **위치 파악**: 관심 있는 모든 기능은 정책에서 찾을 수 있어야 함. 찾을 수 없다면 정책이 해당 내용을 다루지 않음.
- **분류**: 추적(완화 조치 트리거) vs 연구(추적되지만 트리거하지 않음)인가? OpenAI는 이를 명시; Anthropic과 DeepMind는 자체 기준 적용.
- **주기**: 정책이 선언된 일정에 따라 업데이트되는가, 아니면 특정 이벤트 후에만 업데이트되는가? 선언된 주기가 더 강력함.
- **독립성**: 외부 검토가 필수인가 선택인가? Anthropic은 아폴로(Apollo) 및 미국 AI 안전 연구소와 협력; OpenAI는 METR과 협력; DeepMind는 주로 내부 SAG와 협력.

## 사용 방법

`code/main.py`는 소규모 결정 테이블 비교 도구를 구현합니다. 주어진 능력(자율성, 기만적 정렬, R&D 자동화, 사이버 강화 등)에 대해 세 가지 정책이 해당 능력을 어떻게 분류하는지, 그리고 어떤 완화 조치가 발동되는지를 출력합니다. 이는 정책 도구가 아닌 읽기 보조 도구입니다.

## Ship It

`outputs/skill-cross-policy-diff.md`는 특정 기능에 대한 정책 간 비교를 생성하며, 세 가지 프레임워크를 참조로 사용합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. diff 도구의 출력이 소스 문서와 대조하여 확인할 수 있는 최소 두 가지 기능(capability)에 대한 정책과 일치하는지 확인하세요.

2. OpenAI Preparedness Framework v2를 전부 읽으세요. 각 연구 범주(Research Category)를 식별하세요. 각각에 대해 왜 해당 범주가 추적(Tracked)이 아닌 연구(Research)에 속하는지 한 문장으로 설명하세요.

3. DeepMind FSF v3와 2026년 4월 추적 기능 수준(Tracked Capability Levels) 업데이트를 전부 읽으세요. ML R&D 자율성 수준 1의 구체적인 평가 기준을 식별하세요. 이를 외부에서 어떻게 측정할 수 있을까요?

4. 샌드배깅(sandbagging)은 OpenAI의 연구 범주에 포함됩니다. 샌드배깅 모델이 실제 능력을 드러내도록 강제하는 평가 방법을 설계하세요. 레슨 1의 평가-맥락 조작(eval-context-gaming) 논의를 참조하세요.

5. 특정 기능(선택 사항)에 대해 세 정책을 비교하세요. 어떤 정책의 분류가 가장 엄격하고 어떤 것이 가장 느슨한지 명명하세요. 소스 텍스트를 인용하여 근거를 제시하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|---|---|---|
| Preparedness Framework | "OpenAI의 확장 정책" | PF v2 (2025년 4월); Tracked vs Research 범주 |
| Tracked Category | "의무적 완화 조치" | Capabilities + Safeguards 보고서 트리거; SAG 검토 |
| Research Category | "모니터링만 수행" | 추적되지만 자동 완화 조치 없음; Long-range Autonomy 포함 |
| Frontier Safety Framework | "DeepMind의 확장 정책" | FSF v3 (2025년 9월) + Tracked Capability Levels (2026년 4월) |
| CCL | "중요 역량 수준" | DeepMind의 도메인별 임계값 (Cyber, Bio, ML R&D, CBRN) |
| ML R&D 자율성 수준 1 | "R&D 자동화" | 경쟁력 있는 비용으로 AI R&D 파이프라인 완전 자동화 |
| Sandbagging | "전략적 성능 저하" | 평가 시 모델 성능 저하; OpenAI Research 범주 포함 |
| Instrumental reasoning | "수단-목적 추론" | 목표 달성 방법에 대한 추론; DeepMind 모니터링 대상 |

## 추가 자료

- [OpenAI — 준비 프레임워크 업데이트](https://openai.com/index/updating-our-preparedness-framework/) — v2 발표.
- [OpenAI — 준비 프레임워크 v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 전체 문서.
- [DeepMind — 프론티어 안전 프레임워크 강화](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 발표.
- [DeepMind — 프론티어 안전 프레임워크 업데이트 (2026년 4월)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 추적 기능 레벨 추가.
- [Gemini 3 Pro FSF 보고서](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 형식 위험 보고서 예시.