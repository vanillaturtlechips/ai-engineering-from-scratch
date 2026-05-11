# GPU 설정 및 클라우드

> CPU에서 학습하는 것은 학습에 적합합니다. 실제 배포를 위한 학습에는 GPU가 필요합니다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** Phase 0, Lesson 01
**소요 시간:** ~45분

## 학습 목표

- `nvidia-smi` 및 PyTorch의 CUDA API를 사용하여 로컬 GPU 가용성 확인
- 무료 클라우드 기반 실험을 위해 T4 GPU로 Google Colab 구성
- CPU vs GPU에서 행렬 곱셈 벤치마킹 및 속도 향상 측정
- fp16 경험 법칙을 사용하여 VRAM에 적합한 최대 모델 크기 추정

## 문제

1-3단계의 대부분의 레슨은 CPU에서도 잘 실행됩니다. 하지만 CNN, 트랜스포머, LLM(대규모 언어 모델)을 훈련하기 시작하는 4단계 이상에서는 GPU 가속이 필요합니다. CPU에서 8시간 걸리는 훈련 작업이 GPU에서는 10분 만에 완료됩니다.

세 가지 옵션이 있습니다: 로컬 GPU, 클라우드 GPU, 또는 Google Colab(무료).

## 개념

```
당신의 옵션:

1. 로컬 NVIDIA GPU
   비용: $0 (이미 보유 중)
   설정: CUDA + cuDNN 설치
   최적 사용처: 일반 사용, 대규모 데이터셋

2. Google Colab (무료 티어)
   비용: $0
   설정: 없음
   최적 사용처: 빠른 실험, 집에 GPU 없음

3. 클라우드 GPU (Lambda, RunPod, Vast.ai)
   비용: $0.20-2.00/시간
   설정: SSH + 설치
   최적 사용처: 본격적인 학습, 대규모 모델
```

## 빌드하기

### 옵션 1: 로컬 NVIDIA GPU

GPU 확인 방법:

```bash
nvidia-smi
```

CUDA 지원 PyTorch 설치:

```python
import torch

print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"CUDA 버전: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"메모리: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### 옵션 2: Google Colab

1. [colab.research.google.com](https://colab.research.google.com) 접속
2. Runtime > 런타임 유형 변경 > T4 GPU 선택
3. `!nvidia-smi` 실행하여 확인

이 과정의 노트북 파일을 Colab에 직접 업로드할 수 있습니다.

### 옵션 3: 클라우드 GPU

Lambda Labs, RunPod, Vast.ai 사용 시:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### GPU가 없어도 문제 없습니다.

대부분의 레슨은 CPU에서도 작동합니다. GPU가 필요한 레슨은 별도로 표시되며 Colab 링크가 포함됩니다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")
```

## Build It: GPU vs CPU 벤치마크

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 연습 문제

1. 위의 벤치마크를 실행하고 CPU와 GPU 실행 시간을 비교하세요  
2. GPU가 없는 경우 Google Colab에서 실행한 후 비교하세요  
3. 사용 가능한 GPU 메모리 용량을 확인하고 수용 가능한 가장 큰 모델 크기를 추정하세요 (경험적 규칙: fp16 파라미터당 2바이트)

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| CUDA | "GPU 프로그래밍" | NVIDIA의 병렬 컴퓨팅 플랫폼으로 GPU에서 코드를 실행할 수 있게 해줌 |
| VRAM | "GPU 메모리" | GPU에 있는 비디오 RAM으로 시스템 RAM과 별개. 모델 크기 제한 |
| fp16 | "하프 정밀도" | 16비트 부동소수점, fp32 대비 메모리 사용량 절반이며 정확도 손실 최소화 |
| Tensor Core | "고속 행렬 연산 하드웨어" | 행렬 곱셈을 위한 전용 GPU 코어, 일반 코어 대비 4-8배 빠름 |