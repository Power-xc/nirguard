# NIRGuard

> 화면 재촬영 유출 귀속을 위한 포렌식 워터마킹 검증 랩.

특허 출원 중. 검증 단계의 연구 프로젝트이며 완성된 제품이 아닙니다.

English version: [README.en.md](README.en.md)

## 문제

HDCP, Widevine, FairPlay — 모든 DRM에는 공통 사각지대가 있습니다. 화면을 카메라로
찍는 것입니다. 이것이 아날로그 홀이며, 콘텐츠가 빛이 되는 순간 디지털 보호는 복사
경로를 더 이상 추적하지 못합니다.

NIRGuard는 통제된 리뷰 워크플로(사전 공개 미디어, 기밀 문서, 디자인 리뷰, 게임
빌드, 내부 보안 리뷰)에서 재촬영된 화면 사진·영상을 리뷰 세션으로 역추적하는
포렌식 워터마킹을 탐구합니다.

## 접근법

사용자/세션별 패턴을 저진폭 R채널 조정으로 시각 신호에 삽입하고, 용의 캡처에서
패턴을 복원해 등록된 패턴들과 대조합니다. 불확실하면 지목하지 않는(abstain)
신뢰도 정책을 적용합니다.

```
User ID → SHA-256 → PRNG → 64×64 이진 패턴
→ R채널 삽입 (±4 조정)
→ PSNR로 가시 품질 영향 측정
→ 용의 캡처 업로드
→ 패턴 복원 (평균화 + Otsu 이진화)
→ 해밍 거리 + confidence/abstain 정책
→ 후보 귀속 리포트
```

현재 방식은 가시광 공간영역 워터마킹입니다. 디스플레이의 근적외선 방출을 전제로
하는 물리 주장은 하지 않으며, 패널이 유의미한 NIR 신호를 방출하는지는
docs/HARDWARE_TEST_BENCH.md에 정의된 분광 측정으로만 판정합니다.

## 검증 현황

지금까지의 모든 강인성 수치는 디지털 시뮬레이션(JPEG 압축, 가우시안 노이즈,
다운스케일)에서 나온 것이며 실물 광학 재촬영 실측이 아닙니다. 검증 하네스에서
재현되기 전까지 정확도 주장은 사용하지 않습니다.

| 영역 | 상태 |
|------|------|
| 사용자별 패턴 생성 | 구현됨 |
| 워터마크 삽입/추출 (픽셀·블록 모드) | 구현됨 |
| confidence / abstain 귀속 정책 | 구현됨 |
| 합성 강인성 하네스 | 구현됨 — 픽셀 모드는 복합 열화에서 실패, 블록 모드 튜닝 중 |
| FastAPI 데모 서버 + Next.js UI | 구현됨 |
| 실측 재촬영 복원 (스마트폰, 프로토콜 기반) | 계획 — docs/REAL_CAPTURE_PROTOCOL.md |
| 분광 전제 측정 | 계획 — docs/HARDWARE_TEST_BENCH.md |
| 제3자 검증 | 미실시 |

## 기술 스택

| 계층 | 기술 |
|------|------|
| 코어 | Python 3.11+, NumPy, OpenCV |
| API | FastAPI, Uvicorn |
| 데모 UI | Next.js 14, React, TypeScript, Tailwind CSS |
| 저장소 | SQLite |
| 테스트 | pytest |

## 저장소 구조

```
nirguard/
├── watermark/          # 코어 파이프라인: 생성·삽입·복원·판정
├── matching/           # 패턴 DB·후보 매칭
├── api/                # FastAPI 데모 서버 (로컬 전용)
├── demo/               # Next.js 데모 UI
├── tools/              # 검증 랩 CLI
├── experiments/        # 실험 매니페스트·로컬 결과
├── tests/              # 단위·강인성 테스트
└── docs/               # 검증 계획·프로토콜
```

## 시작하기

### 1. Python API 서버

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

API 서버는 로컬 데모 도구입니다. 인증 계층이 없으므로 공개망에 배포하면 안 됩니다.

### 2. 데모 UI

```bash
cd demo
npm install
npm run dev
# → http://localhost:3000
```

### 3. 테스트

```bash
python3 -m pytest tests/ -v

# 열화 조건별 BER 표
python3 tests/robustness_test.py --quick
```

## 검증 CLI

```bash
# 단일 검증 실험
python3 tools/run_experiment.py --input path/to/image.png --user-id user_003 --mode block

# 배치 매니페스트 실행
python3 tools/run_manifest.py --manifest experiments/manifests/example_batch_manifest.json

# 멀티프레임 실험
python3 tools/run_video_experiment.py --input path/to/video.mp4 --user-id user_003 --mode block

# 실측 촬영 킷 준비
python3 tools/create_capture_kit.py --input path/to/image.png --user-id user_003 --out-dir experiments/datasets/case_001

# 결과 요약 / 임계값 스윕 / 증거 산출물
python3 tools/analyze_results.py --results-dir experiments/results
python3 tools/threshold_sweep.py --results-dir experiments/results --output experiments/results/threshold_sweep.json
python3 tools/generate_report.py --result experiments/results/run.json --output experiments/results/report.pdf
python3 tools/build_verification_package.py --results-dir experiments/results --out-dir validation_package
```

결과 보고의 클레임 규칙은 docs/CLAIMS_POLICY.md, 외부 검증 프로토콜은
docs/EXTERNAL_VALIDATION_PACKAGE.md를 따릅니다.

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/embed` | 워터마크 삽입 → PSNR 반환 |
| `POST` | `/api/simulate` | 재촬영 열화 시뮬레이션 |
| `POST` | `/api/extract` | 패턴 복원 + 후보 랭킹 |
| `GET` | `/api/health` | 헬스체크 + DB 상태 |

인터랙티브 문서: `http://localhost:8000/docs`

## 로드맵

- docs/REAL_CAPTURE_PROTOCOL.md 기반 실측 데이터셋 수집과 조건별 BER 측정
- docs/HARDWARE_TEST_BENCH.md 기반 분광 전제 측정 (NIR 질문의 최종 판정)
- docs/VALIDATION_PLAN.md의 오탐·보류율 목표 대비 임계값/ECC 튜닝
- docs/EXTERNAL_VALIDATION_PACKAGE.md 기반 외부 검증 라운드

각 마일스톤은 docs/VALIDATION_PLAN.md에 정의된 지표 기준에 연결됩니다.

## 지적재산권과 라이선스

특허 출원 중 (한국 특허청). NIRGuard는 저작자의 프로젝트명입니다.

Copyright (c) 2026 KangKeun Park. All rights reserved. 이 저장소는 연구 투명성과
검증 리뷰를 위해 공개되었습니다. 명시적·묵시적 라이선스(특허 라이선스 포함)는
일절 부여되지 않습니다. LICENSE 파일을 참조하세요.

## 문의

라이선싱 · 파일럿 · 연구 협업 문의: pkk062300@naver.com
