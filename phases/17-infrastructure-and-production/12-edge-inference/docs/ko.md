# 엣지 추론 — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> 핵심 엣지 제약은 연산(compute)이 아닌 메모리 대역폭입니다. 모바일 DRAM은 50-90 GB/s에 머무는 반면, 데이터센터 HBM3는 2-3 TB/s를 기록해 30-50배 격차가 발생합니다. 디코딩은 메모리 바운드 작업이므로 이 격차가 결정적입니다. 2026년에는 네 가지 방향으로 분화될 전망입니다. Apple M4/A18 Neural Engine은 통합 메모리(CPU↔NPU 복사 불필요)로 38 TOPS 성능을 제공합니다. Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon은 45 TOPS를 달성합니다. WebGPU + WebLLM은 M3 Max에서 Llama 3.1 8B(Q4)를 초당 ~41토큰 처리(네이티브 대비 70-80% 성능)하며, 17.6k GitHub 스타를 기록했고 OpenAI 호환 API를 지원하며 모바일 70-75% 커버리지를 확보했습니다. NVIDIA Jetson Orin Nano Super(8GB)는 Llama 3.2 3B / Phi-3를 구동할 수 있고, AGX Orin은 vLLM을 통해 gpt-oss-20b를 ~40 tok/s로 실행합니다. Jetson T4000(JetPack 7.1)은 AGX Orin 대비 2배 성능을 제공합니다. TensorRT Edge-LLM은 EAGLE-3, NVFP4, 청크 기반 프리필(chunked prefill)을 지원하며, 2026년 CES에서 Bosch, ThunderSoft, MediaTek이 데모를 선보였습니다.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 대역폭 바운드 디코딩 시뮬레이터)  
**선수 지식:** 17단계 · 04 (vLLM 서빙 내부 구조), 17단계 · 09 (프로덕션 양자화)  
**소요 시간:** ~60분

## 학습 목표

- 모바일 LLM 추론이 메모리 대역폭에 의존적이며 계산 성능은 부차적인 이유를 설명.
- 네 가지 엣지 타겟(Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson)을 열거하고 각각을 사용 사례와 매칭.
- 2026년 WebGPU 커버리지 격차(Firefox Android의 따라잡기)와 Safari iOS 26 출시 계획 언급.
- 타겟별 양자화 형식 선택(ANE에는 Core ML INT4 + FP16, Hexagon에는 QNN INT8/INT4, 브라우저에는 WebGPU Q4, Jetson Thor에는 NVFP4).

## 문제

고객은 온디바이스 챗봇을 원합니다: 음성 우선, 기본적으로 비공개, 오프라인 작동. MacBook Pro M3 Max에서 Llama 3.1 8B Q4는 ~55 tok/s로 실행됩니다 — 괜찮습니다. iPhone 16 Pro에서는 동일한 모델이 3 tok/s로 실행됩니다 — 문제가 있습니다. Snapdragon 8 Gen 3이 탑재된 중저가형 Android 기기에서는 7 tok/s입니다. Chrome Android v121+에서 WebGPU를 통해 브라우저에서 실행할 경우 기기에 따라 4-8 tok/s입니다.

처리량 차이는 포팅 문제가 아닙니다. 이는 대역폭 격차 × 양자화 형식 × NPU가 사용자 공간에서 접근 가능한지의 여부에 따른 것입니다. 2026년 에지 추론은 네 가지 다른 문제와 네 가지 다른 솔루션을 의미합니다.

## 개념

### 대역폭이 실제 한계

Decode는 모든 토큰에 대해 전체 가중치 세트를 읽습니다. Q4 형식의 7B 모델은 3.5GB입니다. 50GB/s 속도로 3.5GB를 읽는 데 70ms가 소요되며, 이는 이론적 한계인 ~14 tok/s를 의미합니다. 90GB/s(고급 모바일 DRAM)에서는 한계가 ~25 tok/s로 이동합니다. 이 수치 이하에서는 계산 능력이 아무리 높아도 도움이 되지 않습니다.

데이터센터 HBM3의 3TB/s 속도는 동일한 3.5GB를 1.2ms 만에 처리하며, 한계는 830 tok/s입니다. 동일한 모델, 동일한 가중치, 다른 메모리 서브시스템입니다.

### Apple Neural Engine (M4 / A18)

- 최대 38 TOPS. 통합 메모리(CPU와 ANE가 동일한 풀 공유) — 복사 오버헤드 없음.
- Core ML + `.mlmodel` 컴파일 모델 또는 PyTorch를 통한 Metal Performance Shaders(MPS)로 접근.
- Llama.cpp Metal 백엔드는 ANE를 직접 사용하지 않고 MPS를 활용; 네이티브 ANE에는 Core ML 변환 필요.
- 2026년 iOS 앱을 위한 최적의 경로: INT4 가중치 + FP16 활성화를 사용하는 Core ML.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 최대 45 TOPS. SoC 내 CPU 및 GPU와 통합되었지만 별도의 메모리 도메인.
- QNN(Qualcomm Neural Network) SDK 및 AI Hub는 PyTorch/ONNX에서 변환을 제공.
- 채팅 템플릿, Llama 3.2, Phi-3는 모두 AI Hub에서 1급 아티팩트로 제공.

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS. 소프트웨어는 Apple/Qualcomm에 비해 뒤처짐; OpenVINO는 개선 중이지만 니치.
- Windows ARM 코파일럿 앱에 최적; AMD/Intel 데스크탑에서 로컬 퍼스트 지원.

### WebGPU + WebLLM

- WebGPU 컴퓨트 셰이더를 통해 브라우저에서 모델 실행; 설치 불필요.
- M3 Max에서 Llama 3.1 8B Q4는 ~41 tok/s — 동일한 백엔드를 통한 네이티브 대비 약 70-80% 성능.
- WebLLM은 17.6k GitHub 스타; OpenAI 호환 JS API; Apache 2.0.
- 2026년 지원 범위: Chrome Android v121+, Safari iOS 26 GA, Firefox Android는 아직 따라잡는 중. 전체 모바일 커버리지 ~70-75%.

### NVIDIA Jetson 제품군

- Orin Nano Super (8GB): Llama 3.2 3B, Phi-3를 적절한 tok/s로 실행 가능.
- AGX Orin: vLLM을 통해 gpt-oss-20b를 ~40 tok/s로 실행.
- Thor / T4000 (JetPack 7.1): AGX Orin 대비 2배 성능, EAGLE-3 및 NVFP4 지원.
- TensorRT Edge-LLM(2026)은 EAGLE-3 추론 디코딩, NVFP4 가중치, 청크 프리필 지원 — 데이터센터 최적화를 엣지로 이식.

### 대상별 양자화 선택

| 대상 | 형식 | 참고 |
|--------|--------|-------|
| Apple ANE | INT4 가중치 + FP16 활성화 | Core ML 변환 경로 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 변환기 |
| WebGPU / WebLLM | Q4 MLC(q4f16_1) | `mlc_llm convert_weight` + 컴파일된 `.wasm` 사용; GGUF는 지원되지 않음 |
| Jetson Orin Nano | Q4 GGUF 또는 TRT-LLM INT4 | 메모리 바운드 |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 경로 |

### 엣지에서의 장문맥 함정

Llama 3.1의 128K 문맥은 데이터센터 기능입니다. 8GB RAM이 있는 휴대폰에서 4GB 모델 + 32K 토큰에 대한 2GB KV 캐시 + OS 오버헤드 = OOM. 엣지 배포는 공격적인 KV 양자화(Q4 KV)를 수용하지 않는 한 문맥을 4K-8K로 유지합니다.

### 음성이 킬러 앱

음성 에이전트는 지연 시간에 민감합니다(첫 토큰 < 500ms). 로컬 추론은 네트워크 지연을 완전히 제거합니다. 음성-텍스트 변환(Whisper Turbo 변형 엣지 실행)과 결합하면 엣지 추론이 프로덕션 품질의 음성 루프가 됩니다.

### 기억해야 할 수치

- Apple M4 / A18 ANE: 38 TOPS.
- Qualcomm Hexagon SD X Elite: 45 TOPS.
- WebLLM M3 Max: Llama 3.1 8B Q4에서 ~41 tok/s.
- AGX Orin: vLLM을 통해 gpt-oss-20b에서 ~40 tok/s.
- 데이터센터-엣지 대역폭 격차: 30-50배.
- WebGPU 모바일 커버리지: ~70-75%(Firefox Android는 뒤처짐).

## 사용 방법

`code/main.py`는 대역폭 제한 수학 계산을 통해 엣지 대상(edge targets)에 대한 이론적 디코딩 처리량 상한선을 계산합니다. 계산된 결과를 실제 벤치마크 관측치와 비교하며, 계산(compute)이 아닌 대역폭(bandwidth)이 병목 현상인 지점을 강조합니다.

## Ship It

이 레슨은 `outputs/skill-edge-target-picker.md`를 생성합니다. 플랫폼(iOS/Android/브라우저/Jetson), 모델, 지연 시간/메모리 예산을 고려하여 양자화 형식과 변환 파이프라인을 선택합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. Snapdragon 8 Gen 3(~77 GB/s 대역폭)에서 Q4 형식의 7B 모델에 대해 디코드 천장(decode ceiling)을 계산하세요. 관측된 6-8 tok/s와 비교 — 실행 시간이 효율적인가요?
2. Android의 WebGPU는 Chrome v121+가 필요합니다. 이전 브라우저를 위한 대체 방안 — 동일한 OpenAI 호환 API를 통한 서버 측 구현을 설계하세요.
3. iOS 앱에 4K-컨텍스트 스트리밍이 필요합니다. iPhone 16에서 활성 메모리 4GB 미만을 유지할 수 있는 모델/포맷 조합은 무엇인가요?
4. Jetson AGX Orin은 gpt-oss-20b를 40 tok/s로 실행합니다. Jetson Nano는 3B 모델만 지원합니다. 제품이 두 장치를 모두 대상으로 할 때, 추론 스택을 어떻게 통합하나요?
5. "WebLLM은 2026년에 프로덕션 준비 완료"라는 주장에 대해 찬반 입장을 제시하세요. 커버리지, 성능, Firefox Android 격차를 근거로 제시하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| ANE | "Apple neural engine" | M 시리즈 및 A 시리즈 내 온디바이스 NPU; 통합 메모리 |
| Hexagon | "Qualcomm NPU" | 스냅드래곤 NPU; QNN SDK를 통한 접근 |
| WebGPU | "browser GPU" | W3C 표준화된 브라우저 GPU API; 크롬/사파리 2026 |
| WebLLM | "browser LLM runtime" | MLC-LLM 프로젝트; 아파치 2.0 라이선스; OpenAI 호환 JS |
| Jetson | "NVIDIA edge" | 오린 나노/AGX/토르/T4000 패밀리 |
| TRT Edge-LLM | "edge TensorRT" | 2026년 엣지용 TensorRT-LLM 포트; EAGLE-3 + NVFP4 |
| Unified memory | "shared pool" | CPU와 NPU가 동일한 RAM을 참조; 복사 오버헤드 없음 |
| Bandwidth-bound | "memory limited" | 가중치 읽기 바이트/초에 의해 디코딩이 제한됨 |
| Core ML | "Apple conversion" | ANE 네이티브 모델을 위한 애플 프레임워크 |
| QNN | "Qualcomm stack" | 퀄컴 신경망 SDK |

## 추가 자료

- [온디바이스 LLM 현황 2026](https://v-chandra.github.io/on-device-llms/) — 생태계 및 벤치마크.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026년 엣지 포트 발표.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — 설계 및 벤치마크.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE 네이티브 변환.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Hexagon용 사전 변환 모델.