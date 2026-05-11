# 3D 가우시안 스플래팅 기초부터 구현하기

> 장면은 수백만 개의 3D 가우시안 구름으로 구성됩니다. 각각은 위치, 방향, 크기, 불투명도, 그리고 시야 방향에 따라 달라지는 색상을 가집니다. 이들을 래스터화하고, 래스터화를 통해 역전파한 후 완료됩니다.

**유형:** 구현
**언어:** Python
**선수 지식:** 4단계 13강 (3D 비전 & NeRF), 1단계 12강 (텐서 연산), 4단계 10강 (확산 모델 기초 - 선택 사항)
**소요 시간:** ~90분

## 학습 목표

- 2026년에 3D 가우시안 스플래팅(3D Gaussian Splatting)이 뉴럴 레디언스 필드(NeRF, Neural Radiance Field)를 대체하여 사실적인 3D 재구성의 프로덕션 기본 기술로 채택된 이유를 설명
- 가우시안별 6개 파라미터(위치, 회전 쿼터니언, 스케일, 불투명도, 구면 조화 함수 색상, 선택적 특징)와 각 파라미터가 기여하는 부동소수점(float) 수 명시
- `alpha` 컴포지팅을 사용하여 2D 가우시안 스플래팅 래스터라이저를 처음부터 구현한 후, 3D 사례가 동일한 루프로 투영되는 방식 설명
- `nerfstudio`, `gsplat`, 또는 `SuperSplat`을 사용하여 20-50장의 사진으로 장면을 재구성하고, `KHR_gaussian_splatting` glTF 확장 또는 OpenUSD 26.03 `UsdVolParticleField3DGaussianSplat` 스키마로 내보내기

## 문제 정의

NeRF(Neural Radiance Fields)는 장면을 MLP(Multi-Layer Perceptron)의 가중치로 저장합니다. 렌더링되는 모든 픽셀은 광선을 따라 수백 번의 MLP 쿼리를 수행합니다. 훈련에는 수 시간이 소요되고, 렌더링에는 수 초가 걸리며, 가중치를 직접 편집할 수 없습니다 — 장면 내 의자를 이동하려면 재훈련이 필요합니다.

3D 가우시안 스플래팅(Kerbl, Kopanas, Leimkühler, Drettakis, SIGGRAPH 2023)은 이 모든 것을 대체했습니다. 장면은 명시적인 3D 가우시안 집합으로 표현됩니다. 렌더링은 100+ fps의 GPU 래스터화 방식으로 이루어집니다. 훈련에는 수 분이 소요되며, 편집은 직접 가능합니다: 가우시안 부분 집합을 이동하면 의자가 옮겨집니다. 2026년까지 Khronos 그룹은 가우시안 스플래트를 위한 glTF 확장 표준을 승인했고, OpenUSD 26.03은 가우시안 스플래트 스키마를 탑재했습니다. Zillow와 Apartments.com은 부동산 렌더링을 위해 이를 사용하며, 3D 재구성에 관한 대부분의 새로운 연구 논문은 핵심 3DGS 아이디어의 변형입니다.

정신 모델(mental model)은 간단하지만, 수학에는 충분한 구성 요소가 있어 대부분의 소개 자료는 래스터화에서 시작하고 투영(projection)과 구면 조화 함수(spherical harmonics)는 생략합니다. 이 강의에서는 전체 구조를 구축합니다 — 먼저 2D 버전을 다룬 후 3D로 확장합니다.

## 개념

### 가우시안이 담는 것

하나의 3D 가우시안은 공간에서 다음과 같은 속성을 가진 매개변수 기반 블롭(blob)입니다:

```
위치             mu         (3,)    월드 좌표계에서의 중심
회전             q          (4,)    방향을 인코딩하는 단위 쿼터니언
크기             s          (3,)    축별 로그 스케일 (렌더링 시 지수화)
투명도           alpha      (1,)    시그모이드 이후 투명도 [0, 1]
구면 조화 계수   c_lm       (3 * (L+1)^2,)   시야 의존적 색상
```

회전 + 크기는 3x3 공분산 행렬을 구성합니다: `Sigma = R S S^T R^T`. 이는 3D에서 가우시안의 형태입니다. 구면 조화 함수는 시야 방향에 따라 색상을 변화시킵니다 — 스펙큘러 하이라이트, 미묘한 광택, 시야 의존적 발광 — 하지만 시야별 텍스처를 저장하지 않습니다. SH 차수 3을 사용하면 색상 채널당 16개 계수, 가우시안당 색상만 48개의 부동소수점이 필요합니다.

일반적인 장면에는 100만~500만 개의 가우시안이 있습니다. 각각은 대략 60개의 부동소수점(3 + 4 + 3 + 1 + 48 + 기타)을 저장합니다. 500만 개 가우시안 장면의 경우 240MB로, 텍스처가 있는 포인트 클라우드보다 훨씬 작고, 고해상도로 재렌더링된 NeRF의 MLP 가중치보다 10배 작습니다.

### 레이 마칭이 아닌 래스터화

```mermaid
flowchart LR
    SCENE["수백만 개의 3D 가우시안<br/>(위치, 회전, 크기,<br/>투명도, SH 색상)"] --> PROJ["2D로 투영<br/>(카메라 외부/내부 파라미터)"]
    PROJ --> TILES["타일 할당<br/>(16x16 화면 공간)"]
    TILES --> SORT["깊이 정렬<br/>타일별"]
    SORT --> ALPHA["알파 블렌딩<br/>앞→뒤"]
    ALPHA --> PIX["픽셀 색상"]

    style SCENE fill:#dbeafe,stroke:#2563eb
    style ALPHA fill:#fef3c7,stroke:#d97706
    style PIX fill:#dcfce7,stroke:#16a34a
```

5단계로 모두 GPU 친화적입니다. 픽셀당 MLP 쿼리가 필요 없습니다. 단일 RTX 3080 Ti는 600만 개의 스플랫을 147fps로 렌더링합니다.

### 투영 단계

월드 위치 `mu`와 3D 공분산 `Sigma`를 가진 3D 가우시안은 화면 위치 `mu'`와 2D 공분산 `Sigma'`를 가진 2D 가우시안으로 투영됩니다:

```
mu' = project(mu)
Sigma' = J W Sigma W^T J^T          (2 x 2)

W = 카메라 회전 + 평행 이동을 포함한 시야 변환
J = mu'에서의 원근 투영 야코비안
```

2D 가우시안의 풋프린트는 `Sigma'`의 고유벡터가 축인 타원입니다. 이 타원 내 모든 픽셀은 가우시안의 기여를 받으며, 가중치는 `exp(-0.5 * (p - mu')^T Sigma'^-1 (p - mu'))`로 결정됩니다.

### 알파 블렌딩 규칙

하나의 픽셀에 대해 이를 덮는 가우시안들은 뒤에서 앞 순서로(또는 역순 공식으로 앞→뒤) 정렬됩니다. 색상은 1980년대 이후 반투명 래스터라이저의 동일한 방정식으로 블렌딩됩니다:

```
C_pixel = sum_i alpha_i * T_i * c_i

T_i = prod_{j < i} (1 - alpha_j)       i번째까지의 투과율
alpha_i = opacity_i * exp(-0.5 * d^T Sigma'^-1 d)   지역적 기여도
c_i = eval_SH(SH_i, view_direction)    시야 의존적 색상
```

이는 **NeRF의 체적 렌더링과 동일한 방정식**으로, 광선 따라 밀집 샘플 대신 명시적 희소 가우시안 집합에 적용됩니다. 이 동일성 때문에 렌더링 품질이 NeRF와 일치합니다 — 둘 다 동일한 복사장 방정식을 적분합니다.

### 미분 가능한 이유

투영, 타일 할당, 알파 블렌딩, SH 평가 등 모든 단계는 가우시안 매개변수에 대해 미분 가능합니다. 실제 이미지와 렌더링된 픽셀 손실을 계산한 후 래스터라이저를 통해 역전파하고, 경사 하강법으로 모든 `(mu, q, s, alpha, c_lm)`을 업데이트합니다. 약 3만 번의 반복 후 가우시안들은 올바른 위치, 크기, 색상을 찾습니다.

### 밀집화 및 가지치기

고정된 가우시안 집합만으로는 복잡한 장면을 표현할 수 없습니다. 훈련에는 두 가지 적응 메커니즘이 포함됩니다:

- **복제(Clone)**: 경사 크기는 크지만 스케일이 작은 가우시안을 현재 위치에서 복제 — 해당 영역에 더 많은 디테일이 필요합니다.
- **분할(Split)**: 경사가 큰 큰 스케일 가우시안을 두 개의 작은 가우시안으로 분할 — 하나의 큰 가우시안으로는 해당 영역을 부드럽게 표현하기 어렵습니다.
- **가지치기(Prune)**: 투명도가 임계값 아래로 떨어진 가우시안 제거 — 기여도가 없습니다.

밀집화는 N번의 반복마다 실행됩니다. 장면은 일반적으로 SfM 포인트에서 시드된 약 10만 개의 초기 가우시안에서 훈련 종료 시 100만~500만 개로 성장합니다.

### 구면 조화 함수 한 단락 설명

시야 의존적 색상은 단위 구면 상의 함수 `c(direction)`입니다. 구면 조화 함수는 구면의 푸리에 기저입니다. 차수 `L`에서 절단하면 채널당 `(L+1)^2`개의 기저 함수가 생성됩니다. 새로운 시야에 대한 색상 평가는 학습된 SH 계수와 시야 방향에서 평가된 기저 간의 내적입니다. 차수 0 = 1개 계수 = 상수 색상. 차수 3 = 16개 계수 = 램버트 셰이딩, 스펙큘러, 약한 반사 포착 가능. SD 가우시안 스플랫팅 논문에서는 기본적으로 차수 3을 사용합니다.

### 2026년 프로덕션 스택

```
1. 캡처         스마트폰 / DJI 드론 / 핸드헬드 스캐너
2. SfM / MVS    COLMAP 또는 GLOMAP로 카메라 포즈 + 희소 포인트 추출
3. 3DGS 훈련   nerfstudio / gsplat / INRIA 공식 / PostShot (~10-30분, RTX 4090 기준)
4. 편집         SuperSplat / SplatForge (플로터 제거, 세그먼트)
5. 내보내기     .ply -> glTF KHR_gaussian_splatting 또는 .usd (OpenUSD 26.03)
6. 뷰어         Cesium / Unreal / Babylon.js / Three.js / Vision Pro
```

### 4D 및 생성형 변형

- **4D 가우시안 스플랫팅** — 가우시안이 시간의 함수; 볼륨메트릭 비디오(Superman 2026, A$AP Rocky의 "Helicopter")에 사용.
- **생성형 스플랫** — 전체 장면을 환각하는 텍스트-투-스플랫 모델(World Labs의 Marble).
- **3D 가우시안 무향 변환** — 자율주행 시뮬레이션을 위한 NVIDIA NuRec의 변형.

## 빌드하기

### 단계 1: 2D 가우시안

우리는 먼저 2D 래스터라이저를 구축합니다. 3D 경우는 투영 후 이 2D로 축소됩니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_2d_gaussian(means, covs, points):
    """
    means:  (G, 2)      중심점
    covs:   (G, 2, 2)   공분산 행렬
    points: (H, W, 2)   픽셀 좌표
    returns: (G, H, W)  모든 픽셀에 대한 각 가우시안의 밀도
    """
    G = means.size(0)
    H, W, _ = points.shape
    flat = points.view(-1, 2)
    inv = torch.linalg.inv(covs)
    diff = flat[None, :, :] - means[:, None, :]
    d = torch.einsum("gpi,gij,gpj->gp", diff, inv, diff)
    density = torch.exp(-0.5 * d)
    return density.view(G, H, W)
```

`einsum`은 모든 (가우시안, 픽셀) 쌍에 대해 이차 형식 `diff^T Sigma^-1 diff`를 수행합니다.

### 단계 2: 2D 스플래팅 래스터라이저

앞에서부터 뒤로 알파 합성. 2D에서 깊이는 의미가 없으므로 순서를 위해 학습된 가우시안별 스칼라 값을 사용합니다.

```python
def rasterise_2d(means, covs, colours, opacities, depths, image_size):
    """
    means:     (G, 2)
    covs:      (G, 2, 2)
    colours:   (G, 3)
    opacities: (G,)     [0, 1] 범위
    depths:    (G,)     순서 결정에 사용되는 가우시안별 스칼라
    image_size: (H, W)
    returns:   (H, W, 3) 렌더링된 이미지
    """
    H, W = image_size
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=means.device),
        torch.arange(W, dtype=torch.float32, device=means.device),
        indexing="ij",
    )
    points = torch.stack([xx, yy], dim=-1)

    densities = eval_2d_gaussian(means, covs, points)
    alphas = opacities[:, None, None] * densities
    alphas = alphas.clamp(0.0, 0.99)

    order = torch.argsort(depths)
    alphas = alphas[order]
    colours_sorted = colours[order]

    T = torch.ones(H, W, device=means.device)
    out = torch.zeros(H, W, 3, device=means.device)
    for i in range(means.size(0)):
        a = alphas[i]
        out += (T * a)[..., None] * colours_sorted[i][None, None, :]
        T = T * (1.0 - a)
    return out
```

빠르지는 않지만 — 실제 구현은 타일 기반 CUDA 커널을 사용 — 정확한 수학 연산과 완전한 미분 가능성을 제공합니다.

### 단계 3: 학습 가능한 2D 스플랫 장면

```python
class Splats2D(nn.Module):
    def __init__(self, num_splats=128, image_size=64, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        H, W = image_size, image_size
        self.means = nn.Parameter(torch.rand(num_splats, 2, generator=g) * torch.tensor([W, H]))
        self.log_scale = nn.Parameter(torch.ones(num_splats, 2) * math.log(2.0))
        self.rot = nn.Parameter(torch.zeros(num_splats))  # 2D에서 단일 각도
        self.colour_logits = nn.Parameter(torch.randn(num_splats, 3, generator=g) * 0.5)
        self.opacity_logit = nn.Parameter(torch.zeros(num_splats))
        self.depth = nn.Parameter(torch.rand(num_splats, generator=g))

    def covs(self):
        s = torch.exp(self.log_scale)
        c, si = torch.cos(self.rot), torch.sin(self.rot)
        R = torch.stack([
            torch.stack([c, -si], dim=-1),
            torch.stack([si, c], dim=-1),
        ], dim=-2)
        S = torch.diag_embed(s ** 2)
        return R @ S @ R.transpose(-1, -2)

    def forward(self, image_size):
        covs = self.covs()
        colours = torch.sigmoid(self.colour_logits)
        opacities = torch.sigmoid(self.opacity_logit)
        return rasterise_2d(self.means, covs, colours, opacities, self.depth, image_size)
```

`log_scale`, `opacity_logit`, `colour_logits`는 모두 렌더링 시 적절한 활성화 함수를 통해 매핑되는 제약 없는 매개변수입니다. 이는 모든 3DGS 구현의 표준 패턴입니다.

### 단계 4: 대상 이미지에 2D 가우시안 적합

```python
import math
import numpy as np

def make_target(size=64):
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    img = np.zeros((size, size, 3), dtype=np.float32)
    # 빨간색 원
    mask = (xx - 20) ** 2 + (yy - 20) ** 2 < 10 ** 2
    img[mask] = [1.0, 0.2, 0.2]
    # 파란색 사각형
    mask = (np.abs(xx - 45) < 8) & (np.abs(yy - 40) < 8)
    img[mask] = [0.2, 0.3, 1.0]
    return torch.from_numpy(img)


target = make_target(64)
model = Splats2D(num_splats=64, image_size=64)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(200):
    pred = model((64, 64))
    loss = F.mse_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        print(f"step {step:3d}  mse {loss.item():.4f}")
```

200단계 동안 64개의 가우시안이 두 형태로 수렴합니다. 이것이 전체 아이디어입니다 — 명시적 기하학적 원시에 대한 경사 하강법.

### 단계 5: 2D에서 3D로

3D 확장은 동일한 루프를 유지합니다. 추가된 사항:

1. 가우시안별 회전은 단일 각도 대신 쿼터니언입니다.
2. 공분산은 `R S S^T R^T`이며, `R`은 쿼터니언에서 생성되고 `S = diag(exp(log_scale))`입니다.
3. 투영 `(mu, Sigma) -> (mu', Sigma')`는 카메라 외부 매개변수와 `mu`에서의 원근 투영 야코비안을 사용합니다.
4. 색상은 구면 조화 함수 확장으로 변환되며, 시야 방향에서 평가됩니다.
5. 깊이 정렬은 학습된 스칼라 대신 실제 카메라 공간 z에서 수행됩니다.

모든 프로덕션 구현(`gsplat`, `inria/gaussian-splatting`, `nerfstudio`)은 타일 기반 CUDA 커널로 GPU에서 정확히 이 작업을 수행합니다.

### 단계 6: 구면 조화 함수 평가

3차까지의 SH 기반은 채널당 16개의 항을 가집니다. 평가:

```python
def eval_sh_degree_3(sh_coeffs, dirs):
    """
    sh_coeffs: (..., 16, 3)   마지막 차원은 RGB 채널
    dirs:      (..., 3)       단위 벡터
    returns:   (..., 3)
    """
    C0 = 0.282094791773878
    C1 = 0.488602511902920
    C2 = [1.092548430592079, 1.092548430592079,
          0.315391565252520, 1.092548430592079,
          0.546274215296039]
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    x2, y2, z2 = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = C0 * sh_coeffs[..., 0, :]
    result = result - C1 * y[..., None] * sh_coeffs[..., 1, :]
    result = result + C1 * z[..., None] * sh_coeffs[..., 2, :]
    result = result - C1 * x[..., None] * sh_coeffs[..., 3, :]

    result = result + C2[0] * xy[..., None] * sh_coeffs[..., 4, :]
    result = result + C2[1] * yz[..., None] * sh_coeffs[..., 5, :]
    result = result + C2[2] * (2.0 * z2 - x2 - y2)[..., None] * sh_coeffs[..., 6, :]
    result = result + C2[3] * xz[..., None] * sh_coeffs[..., 7, :]
    result = result + C2[4] * (x2 - y2)[..., None] * sh_coeffs[..., 8, :]

    # 간결성을 위해 3차 항은 생략; 전체 16계수 버전은 코드 파일에 있음
    return result
```

학습된 `sh_coeffs`는 해당 가우시안의 "모든 방향의 색상"을 저장합니다. 렌더링 시 현재 시야 방향에 대해 평가하여 3차원 RGB 벡터를 얻습니다.

## 사용 방법

실제 3DGS 작업에는 `gsplat` (Meta) 또는 `nerfstudio`를 사용하세요:

```bash
pip install nerfstudio gsplat
ns-download-data example
ns-train splatfacto --data path/to/data
```

`splatfacto`는 `nerfstudio`의 3DGS 트레이너입니다. 일반적인 장면의 경우 RTX 4090에서 10-30분 정도 소요됩니다.

2026년에 중요한 내보내기 옵션:

- `.ply` — 원시 가우시안 클라우드 (휴대성 우수, 가장 큰 파일 크기).
- `.splat` — PlayCanvas / SuperSplat 양자화 형식.
- glTF `KHR_gaussian_splatting` — Khronos 표준, 뷰어 간 호환성 (2026년 2월 RC).
- OpenUSD `UsdVolParticleField3DGaussianSplat` — USD 네이티브, NVIDIA Omniverse 및 Vision Pro 파이프라인용.

4D / 동적 장면의 경우, `4DGS`와 `Deformable-3DGS`는 시간에 따라 변하는 평균값과 불투명도를 통해 동일한 메커니즘을 확장합니다.

## Ship It

이 레슨은 다음을 생성합니다:

- `outputs/prompt-3dgs-capture-planner.md` — 주어진 장면 유형에 대한 캡처 세션(사진 수, 카메라 경로, 조명)을 계획하는 프롬프트.
- `outputs/skill-3dgs-export-router.md` — 다운스트림 뷰어 또는 엔진에 따라 적절한 내보내기 형식(`.ply` / `.splat` / glTF / USD)을 선택하는 스킬.

## 연습 문제

1. **(쉬움)** 위의 2D 스플랫 트레이너를 다른 합성 이미지에 적용해 실행하세요. `num_splats`를 `[16, 64, 256]` 범위에서 변화시키며 각 경우에 대해 MSE 대 스텝(step) 그래프를 그리세요. 한계 효용 체감 지점을 식별하세요.
2. **(중간)** 2D 래스터라이저를 확장하여 스칼라 "시야각(view angle)"에 의존하는 가우시안별 RGB 색상을 2차 조화(harmonic)로 지원하도록 만드세요. 한 쌍의 대상 이미지에 대해 훈련하고 모델이 둘 모두를 재구성하는지 검증하세요.
3. **(어려움)** `nerfstudio`를 복제하고 20장의 사진으로 촬영된 임의의 장면(책상, 식물, 얼굴, 방 등)에 대해 `splatfacto`를 훈련하세요. glTF `KHR_gaussian_splatting` 형식으로 내보낸 후 뷰어(Three.js `GaussianSplats3D`, SuperSplat, Babylon.js V9)에서 열어보세요. 훈련 시간, 가우시안 수, 렌더링 fps를 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| 3DGS | "가우시안 스플랫" | 각 가우시안별 위치, 회전, 크기, 불투명도, SH 색상을 가진 수백만 개의 3D 가우시안으로 표현된 명시적 장면 표현 |
| 공분산(Covariance) | "가우시안의 형태" | `Sigma = R S S^T R^T`; 하나의 가우시안에 대한 방향과 비등방성 크기 |
| 알파 컴포지팅(Alpha compositing) | "뒤에서 앞으로 블렌딩" | NeRF의 체적 렌더링과 동일한 방정식, 이제 명시적 희소 집합에 적용 |
| 밀집화(Densification) | "복제 및 분할" | 재구성이 부적합한 영역에 새로운 가우시안을 적응적으로 추가 |
| 가지치기(Pruning) | "저불투명도 삭제" | 훈련 중 불투명도가 거의 0으로 수렴한 가우시안 제거 |
| 구면 조화 함수(Spherical harmonics) | "시야 의존적 색상" | 구면 상의 푸리에 기저; 시야 방향에 따른 색상 함수 저장 |
| 스플랫팩토(Splatfacto) | "nerfstudio의 3DGS" | 2026년 기준 3DGS 훈련을 위한 가장 쉬운 경로 |
| `KHR_gaussian_splatting` | "glTF 표준" | 3DGS를 뷰어 및 엔진 간에 이식 가능하게 만드는 크로노스 2026 확장 표준 |

## 추가 자료

- [3D 가우시안 스플래팅을 활용한 실시간 라디언스 필드 렌더링 (Kerbl et al., SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — 원본 논문
- [gsplat (Meta/nerfstudio)](https://github.com/nerfstudio-project/gsplat) — 프로덕션 품질의 CUDA 래스터라이저
- [nerfstudio Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) — 참조 학습 레시피
- [Khronos KHR_gaussian_splatting 확장](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_gaussian_splatting/README.md) — 2026년 휴대용 포맷
- [OpenUSD 26.03 릴리스 노트](https://openusd.org/release/) — `UsdVolParticleField3DGaussianSplat` 스키마
- [THE FUTURE 3D 가우시안 스플래팅 현황 2026](https://www.thefuture3d.com/blog-0/2026/4/4/state-of-gaussian-splatting-2026) — 업계 개요