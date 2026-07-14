"""
api/main.py — NIRGuard Secure Viewer API

엔드포인트:
  POST /api/embed    — 이미지에 워터마크 삽입 → base64 + PSNR 반환
  POST /api/simulate — 재촬영 시뮬레이션 (JPEG + 노이즈 + 다운스케일)
  POST /api/extract  — 패턴 복원 + 후보 식별 Top-3 반환
  GET  /api/health   — 헬스체크

로컬 데모/검증 전용 서버 — 인증 계층이 없으므로 공개망에 배포하지 않는다.
"""

import base64
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# 프로젝트 루트를 sys.path에 추가 (api/main.py 위치에서 실행 시)
sys.path.insert(0, str(Path(__file__).parent.parent))

from watermark import (
    ConfidencePolicy,
    MatchCandidate,
    apply_degradation,
    decide_attribution,
    embed_block_image,
    generate_pattern,
    embed_image,
    extract_block_from_image,
    extract_from_image,
)
from watermark.metrics import calculate_psnr, calculate_ber
from matching.pattern_db import PatternDB
from tools.generate_report import generate as generate_pdf_report
from tools.run_manifest import run as run_manifest
from tools.analyze_results import load_results, summarize, summarize_by_condition

logger = logging.getLogger("nirguard.api")

# ── 패턴 DB 초기화 ────────────────────────────────────────────────────────────

APP_VERSION = "1.0.0"
_db = PatternDB(db_path="matching/nirguard.db")

# 데모용 사용자 100명을 시드로 등록 (이미 있으면 건너뜀)
_DEMO_USERS = [f"user_{i:03d}" for i in range(100)]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """서버 시작 시 데모 사용자 패턴을 DB에 등록한다."""
    if _db.count() == 0:
        for uid in _DEMO_USERS:
            _db.add_pattern(uid, generate_pattern(uid))
        logger.info("패턴 DB 시드 완료: %d명 등록", _db.count())
    else:
        logger.info("패턴 DB 기존 데이터 사용: %d명", _db.count())
    yield


# ── 앱 초기화 ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="NIRGuard Secure Viewer API",
    version=APP_VERSION,
    description=(
        "Validation-stage controlled review attribution API for watermark embedding, "
        "candidate matching, confidence scoring, abstain decisions, and evidence reports."
    ),
    docs_url=None,
    redoc_url=None,
    lifespan=_lifespan,
)

# CORS: 로컬 Secure Viewer에서 호출 가능하도록
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3005",
        "http://127.0.0.1:3005",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── 유틸리티 ─────────────────────────────────────────────────────────────────

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_UPLOAD_BYTES = 50_000_000  # 50MB — 업로드 DoS 방지


def _bytes_to_image(data: bytes) -> np.ndarray:
    """업로드된 파일 bytes → OpenCV BGR ndarray."""
    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "이미지를 디코딩할 수 없습니다. PNG 또는 JPEG를 업로드하세요.")
    return image


async def _read_upload_image(file: UploadFile) -> np.ndarray:
    """업로드 파일을 검증(형식·크기)하고 BGR ndarray로 디코딩한다."""
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"지원하지 않는 파일 형식: {file.content_type}")
    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "파일 크기 초과 (최대 50MB)")
    return _bytes_to_image(raw)


def _image_to_base64(image: np.ndarray, lossless: bool = False) -> str:
    """
    OpenCV BGR ndarray → base64 문자열.

    lossless=True: PNG (워터마크 이미지 전송 필수 — JPEG는 ±4 신호 소멸)
    lossless=False: JPEG Q=92 (열화 결과, 원본 비교용 등 일반 이미지)
    """
    if lossless:
        _, encoded = cv2.imencode(".png", image)
    else:
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


# ── 응답 스키마 ───────────────────────────────────────────────────────────────

class EmbedResponse(BaseModel):
    """POST /api/embed 응답."""
    watermarked_b64: str = Field(description="워터마크 삽입된 이미지 (JPEG base64)")
    original_b64: str    = Field(description="원본 이미지 (JPEG base64)")
    psnr: float          = Field(description="삽입 후 PSNR (dB)")
    user_id: str         = Field(description="워터마크 적용된 user_id")
    strength: int        = Field(description="사용된 strength 값")
    psnr_pass: bool      = Field(description="PSNR ≥ 40dB 여부")


class SimulateResponse(BaseModel):
    """POST /api/simulate 응답."""
    degraded_b64: str    = Field(description="열화된 이미지 (JPEG base64)")
    original_b64: str    = Field(description="원본 이미지 (비교용, JPEG base64)")
    jpeg_quality: int
    noise_sigma: float
    downscale: float


class Candidate(BaseModel):
    """유출자 후보 1명."""
    user_id: str
    hamming: float  = Field(description="해밍 거리 [0, 1]")
    confidence: float = Field(description="신뢰도 (%)")


class ExtractResponse(BaseModel):
    """POST /api/extract 응답."""
    top_candidates: list[Candidate] = Field(description="유출자 후보 Top-3")
    ber: float                      = Field(description="최우선 후보 기준 BER")
    ber_pass: bool                  = Field(description="BER ≤ 5% 여부")
    identified: bool                = Field(description="confidence 정책 기준 식별 성공 여부")
    abstained: bool                 = Field(description="불확실하여 식별 보류했는지 여부")
    reason: str                     = Field(description="식별 또는 보류 판정 사유")


class HealthResponse(BaseModel):
    status: str
    db_count: int
    version: str


class CaseCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    asset_type: str = Field(default="image")
    reviewer_id: str = Field(min_length=1, max_length=256)
    notes: str = Field(default="", max_length=2000)


class CaseResponse(BaseModel):
    case_id: str
    path: str


class ManifestRunRequest(BaseModel):
    manifest: dict


class ManifestRunResponse(BaseModel):
    result_count: int
    output_dir: str


class ResultSummary(BaseModel):
    path: str
    run_id: str
    user_id: str
    condition: str
    identified: bool
    abstained: bool
    ber: float
    confidence: float


class ReportRequest(BaseModel):
    result_path: str


class ReportResponse(BaseModel):
    report_path: str
    report_b64: str


class AnalysisResponse(BaseModel):
    overall: dict
    by_condition: dict


# ── API 콘솔 페이지 ───────────────────────────────────────────────────────────

_API_PAGE_STYLE = """
:root {
  color-scheme: dark;
  --page: #10120f;
  --surface: #171a15;
  --surface-2: #1e221c;
  --line: rgba(198, 211, 187, 0.14);
  --line-strong: #343b31;
  --text: #eef2e8;
  --text-soft: #b9c2af;
  --muted: #8d9785;
  --green: #82b596;
  --green-strong: #9dd5b2;
  --amber: #d2a85b;
}
* { box-sizing: border-box; }
html, body { min-height: 100%; }
body {
  margin: 0;
  background:
    linear-gradient(135deg, rgba(130, 181, 150, 0.08), transparent 38%),
    linear-gradient(315deg, rgba(210, 168, 91, 0.08), transparent 42%),
    var(--page);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
a { color: inherit; text-decoration: none; }
.shell { width: min(1180px, calc(100% - 40px)); margin: 0 auto; padding: 34px 0 48px; }
.topbar {
  min-height: 64px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 24px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.mark {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(157, 213, 178, 0.45);
  border-radius: 8px;
  background: rgba(130, 181, 150, 0.14);
  color: var(--green-strong);
  font-weight: 800;
}
.brand h1, .hero h2, .panel h3, .card h3 { margin: 0; }
.brand h1 { font-size: 1rem; }
.brand p { margin: 2px 0 0; color: var(--muted); font-size: 0.78rem; }
.nav { display: flex; gap: 8px; flex-wrap: wrap; }
.button {
  min-height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.035);
  color: var(--text);
  padding: 0 13px;
  font-size: 0.86rem;
  font-weight: 720;
}
.button.primary { border-color: transparent; background: var(--green); color: #0e130f; }
.hero, .panel, .card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(23, 26, 21, 0.88);
  box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
  backdrop-filter: blur(14px);
}
.hero { padding: 28px; margin-bottom: 14px; }
.eyebrow {
  margin: 0 0 7px;
  color: var(--green-strong);
  font-size: 0.72rem;
  font-weight: 780;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.hero h2 { max-width: 760px; font-size: clamp(1.8rem, 4vw, 3.2rem); line-height: 1.04; letter-spacing: 0; }
.hero p { max-width: 760px; margin: 14px 0 0; color: var(--text-soft); line-height: 1.65; }
.status-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 22px; }
.status {
  min-height: 88px;
  display: grid;
  align-content: space-between;
  border: 1px solid var(--line-strong);
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.16);
  padding: 13px;
}
.status span, .method, .caption { color: var(--muted); font-size: 0.76rem; font-weight: 720; text-transform: uppercase; }
.status strong { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 1.3rem; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }
.card { padding: 16px; min-height: 154px; display: grid; gap: 10px; align-content: start; }
.card h3 { font-size: 1rem; }
.card p { margin: 0; color: var(--text-soft); line-height: 1.55; font-size: 0.9rem; }
.method { color: var(--amber); }
.endpoint {
  width: max-content;
  max-width: 100%;
  overflow-wrap: anywhere;
  border: 1px solid var(--line-strong);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.035);
  color: var(--text);
  padding: 5px 9px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.76rem;
}
.panel { margin-top: 14px; padding: 16px; }
.links { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.caption { margin-top: 14px; color: var(--muted); text-transform: none; }
@media (max-width: 820px) {
  .shell { width: min(100% - 28px, 1180px); padding-top: 20px; }
  .topbar { align-items: flex-start; flex-direction: column; }
  .status-grid, .grid { grid-template-columns: 1fr; }
}
"""


def _endpoint_card(method: str, endpoint: str, title: str, body: str) -> str:
    return (
        '<article class="card">'
        f'<span class="method">{method}</span>'
        f"<h3>{title}</h3>"
        f"<p>{body}</p>"
        f'<span class="endpoint">{endpoint}</span>'
        "</article>"
    )


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def api_index() -> HTMLResponse:
    cards = "\n".join(
        [
            _endpoint_card("POST", "/api/embed", "Embed", "Insert a reviewer/session watermark into a controlled review asset."),
            _endpoint_card("POST", "/api/simulate", "Simulate", "Apply lab recapture conditions for validation-stage testing."),
            _endpoint_card("POST", "/api/extract", "Extract", "Return top candidates with confidence and abstain-aware attribution."),
            _endpoint_card("GET", "/api/results", "Results", "List recent validation results stored by the experiment harness."),
            _endpoint_card("POST", "/api/report", "Report", "Generate a compact PDF evidence report from a result JSON."),
            _endpoint_card("GET", "/api/analysis", "Analysis", "Summarize BER, identification, and abstain behavior across runs."),
        ]
    )
    html = (
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NIRGuard Secure Viewer API</title>
  <style>$STYLE</style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="mark">NG</span>
        <div>
          <h1>NIRGuard</h1>
          <p>Secure Viewer API</p>
        </div>
      </div>
      <nav class="nav">
        <a class="button primary" href="/docs">API Docs</a>
        <a class="button" href="/openapi.json">OpenAPI JSON</a>
        <a class="button" href="http://127.0.0.1:3005">Secure Viewer</a>
      </nav>
    </header>

    <section class="hero">
      <p class="eyebrow">Validation-stage controlled review attribution</p>
      <h2>NIRGuard Secure Viewer API</h2>
      <p>
        Backend service for watermark insertion, recapture simulation, candidate attribution,
        confidence scoring, abstain decisions, validation result review, and evidence report generation.
      </p>
      <div class="status-grid">
        <div class="status"><span>Status</span><strong>$STATUS</strong></div>
        <div class="status"><span>Version</span><strong>$VERSION</strong></div>
        <div class="status"><span>Pattern DB</span><strong>$DB_COUNT</strong></div>
      </div>
    </section>

    <section class="grid">
      $CARDS
    </section>

    <section class="panel">
      <p class="eyebrow">Interface</p>
      <h3>Stable API paths for the local Secure Viewer pilot</h3>
      <div class="links">
        <a class="button primary" href="/docs">Open API Docs</a>
        <a class="button" href="/api/health">Health Check</a>
        <a class="button" href="/openapi.json">OpenAPI Schema</a>
      </div>
      <p class="caption">
        Attribution output is a technical validation signal. Use confidence, Top-N candidates,
        and abstain decisions before making an operational judgment.
      </p>
    </section>
  </main>
</body>
</html>"""
        .replace("$STYLE", _API_PAGE_STYLE)
        .replace("$STATUS", "ok")
        .replace("$VERSION", APP_VERSION)
        .replace("$DB_COUNT", str(_db.count()))
        .replace("$CARDS", cards)
    )
    return HTMLResponse(content=html)


@app.head("/", include_in_schema=False)
async def api_index_head() -> Response:
    return Response(status_code=200, media_type="text/html")


@app.get("/docs", include_in_schema=False, response_class=HTMLResponse)
async def api_docs() -> HTMLResponse:
    html = (
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NIRGuard API Docs</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
  <style>$STYLE
    body { background: var(--page); }
    .shell { width: min(1280px, calc(100% - 40px)); padding-bottom: 24px; }
    #swagger-ui {
      width: min(1280px, calc(100% - 40px));
      margin: 0 auto 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
      color: #1f2937;
      color-scheme: light;
    }
    .swagger-ui .topbar { display: none; }
    #swagger-ui .swagger-ui {
      color: #1f2937;
    }
    #swagger-ui .swagger-ui .json-schema-2020-12,
    #swagger-ui .swagger-ui .json-schema-2020-12-head,
    #swagger-ui .swagger-ui .json-schema-2020-12-body,
    #swagger-ui .swagger-ui .json-schema-2020-12-accordion,
    #swagger-ui .swagger-ui .json-schema-2020-12-accordion__children,
    #swagger-ui .swagger-ui .json-schema-2020-12-expand-deep-button {
      background: #ffffff;
      color: #1f2937;
    }
    #swagger-ui .swagger-ui .json-schema-2020-12__title,
    #swagger-ui .swagger-ui .json-schema-2020-12__attribute,
    #swagger-ui .swagger-ui .json-schema-2020-12__attribute--primary,
    #swagger-ui .swagger-ui .json-schema-2020-12-property,
    #swagger-ui .swagger-ui .json-schema-2020-12-property-name,
    #swagger-ui .swagger-ui .json-schema-2020-12-property-type,
    #swagger-ui .swagger-ui .json-schema-2020-12-description {
      color: #1f2937;
    }
    #swagger-ui .swagger-ui .json-schema-2020-12__attribute--primary {
      border-color: #b7e1c6;
      background: #ecfdf3;
      color: #166534;
    }
    #swagger-ui .swagger-ui .json-schema-2020-12-accordion svg,
    #swagger-ui .swagger-ui .json-schema-2020-12-accordion path {
      fill: #1f2937;
    }
  </style>
</head>
<body>
  <header class="shell">
    <div class="topbar">
      <div class="brand">
        <span class="mark">NG</span>
        <div>
          <h1>NIRGuard API Docs</h1>
          <p>OpenAPI schema for Secure Viewer validation workflows</p>
        </div>
      </div>
      <nav class="nav">
        <a class="button" href="/">API Home</a>
        <a class="button" href="/openapi.json">OpenAPI JSON</a>
        <a class="button primary" href="http://127.0.0.1:3005">Secure Viewer</a>
      </nav>
    </div>
  </header>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.ui = SwaggerUIBundle({
      url: "/openapi.json",
      dom_id: "#swagger-ui",
      layout: "BaseLayout",
      deepLinking: true,
      showExtensions: true,
      showCommonExtensions: true,
      presets: [
        SwaggerUIBundle.presets.apis,
        SwaggerUIBundle.SwaggerUIStandalonePreset
      ]
    });
  </script>
</body>
</html>"""
        .replace("$STYLE", _API_PAGE_STYLE)
    )
    return HTMLResponse(content=html)


@app.head("/docs", include_in_schema=False)
async def api_docs_head() -> Response:
    return Response(status_code=200, media_type="text/html")


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """서버 상태 및 DB 패턴 수를 반환한다."""
    return HealthResponse(
        status="ok",
        db_count=_db.count(),
        version=APP_VERSION,
    )


@app.post("/api/embed", response_model=EmbedResponse)
async def embed(
    file: UploadFile = File(..., description="원본 이미지 (PNG / JPEG)"),
    user_id: str = Form(..., min_length=1, max_length=256),
    strength: int = Form(default=4, ge=1, le=10),
    mode: str = Form(default="block"),
) -> EmbedResponse:
    """
    이미지에 user_id 기반 워터마크를 삽입한다.

    - user_id가 DB에 없으면 자동 등록한다.
    - 가시 품질 영향은 PSNR로 응답에 반환한다 (목표 기준 40dB).
    """
    original = await _read_upload_image(file)

    pattern = generate_pattern(user_id)
    if not _db.has_user(user_id):
        _db.add_pattern(user_id, pattern)
    watermarked = (
        embed_block_image(original, pattern, strength=strength)
        if mode == "block"
        else embed_image(original, pattern, strength=strength)
    )
    psnr = calculate_psnr(original, watermarked)

    return EmbedResponse(
        watermarked_b64=_image_to_base64(watermarked, lossless=True),  # PNG: 워터마크 신호 보존
        original_b64=_image_to_base64(original, lossless=True),
        psnr=round(psnr, 2),
        user_id=user_id,
        strength=strength,
        psnr_pass=psnr >= 40.0,
    )


@app.post("/api/simulate", response_model=SimulateResponse)
async def simulate(
    file: UploadFile = File(..., description="워터마크 삽입된 이미지"),
    jpeg_quality: int = Form(default=80, ge=1, le=100),
    noise_sigma: float = Form(default=10.0, ge=0.0, le=50.0),
    downscale: float = Form(default=0.75, ge=0.1, le=1.0),
) -> SimulateResponse:
    """
    재촬영 시뮬레이션: JPEG 압축 + 가우시안 노이즈 + 다운스케일을 순서대로 적용한다.

    기본값(Q=80, σ=10, scale=0.75)은 스마트폰 재촬영 수준의 디지털 열화를 모사.
    실물 광학 재촬영을 대체하지 않는다.
    """
    image = await _read_upload_image(file)
    original_b64 = _image_to_base64(image, lossless=True)

    degraded = apply_degradation(
        image,
        jpeg_quality=jpeg_quality,
        noise_sigma=noise_sigma,
        downscale=downscale,
    )

    # degraded 이미지도 PNG(무손실) 반환 — 이후 /api/extract에 전달 시 신호 보존
    return SimulateResponse(
        degraded_b64=_image_to_base64(degraded, lossless=True),
        original_b64=original_b64,
        jpeg_quality=jpeg_quality,
        noise_sigma=noise_sigma,
        downscale=downscale,
    )


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(
    file: UploadFile = File(..., description="패턴을 추출할 이미지 (열화된 이미지)"),
    mode: str = Form(default="block"),
) -> ExtractResponse:
    """
    이미지에서 워터마크 패턴을 복원하고 후보를 식별한다.

    DB에 등록된 모든 패턴과 해밍 거리를 비교해 Top-3 후보를 반환한다.
    BER은 Best Match 패턴과의 오류율이다.
    """
    image = await _read_upload_image(file)

    # 패턴 복원
    extracted = extract_block_from_image(image) if mode == "block" else extract_from_image(image)

    # DB 매칭
    candidates = _db.match(extracted, top_k=3)

    if not candidates:
        raise HTTPException(503, "DB에 패턴이 없습니다. /api/health로 DB 상태를 확인하세요.")

    # BER: Best Match 기준
    best = candidates[0]
    best_pattern = generate_pattern(best["user_id"])
    ber = calculate_ber(best_pattern, extracted)
    decision = decide_attribution(
        [
            MatchCandidate(
                user_id=c["user_id"],
                hamming=c["hamming"],
                confidence=c["confidence"],
            )
            for c in candidates
        ],
        ConfidencePolicy(),
    )

    return ExtractResponse(
        top_candidates=[Candidate(**c) for c in candidates],
        ber=round(ber, 4),
        ber_pass=ber <= 0.05,
        identified=decision.identified,
        abstained=decision.abstained,
        reason=decision.reason,
    )


@app.get("/api/demo-users")
async def demo_users() -> dict:
    """데모용 user_id 목록을 반환한다 (UI 드롭다운용)."""
    return {"users": _db.get_demo_users(n=10)}


@app.post("/api/cases", response_model=CaseResponse)
async def create_case(payload: CaseCreateRequest) -> CaseResponse:
    case_id = payload.title.lower().replace(" ", "-")[:80]
    case_id = "".join(ch for ch in case_id if ch.isalnum() or ch in {"-", "_"}).strip("-")
    if not case_id:
        case_id = "case"

    cases_dir = Path("experiments/cases")
    cases_dir.mkdir(parents=True, exist_ok=True)
    path = cases_dir / f"{case_id}.json"
    suffix = 1
    while path.exists():
        path = cases_dir / f"{case_id}-{suffix}.json"
        suffix += 1

    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    data["case_id"] = path.stem
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return CaseResponse(case_id=path.stem, path=str(path))


@app.get("/api/results", response_model=list[ResultSummary])
async def list_results() -> list[ResultSummary]:
    root = Path("experiments/results")
    results = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest_results.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in data:
            continue
        metrics = data["metrics"]
        best = data.get("best", {})
        condition = data.get("condition", {})
        results.append(
            ResultSummary(
                path=str(path),
                run_id=data.get("run_id", path.stem),
                user_id=data.get("user_id", ""),
                condition=condition.get("name", ""),
                identified=bool(metrics.get("identified", False)),
                abstained=bool(metrics.get("abstained", False)),
                ber=float(metrics.get("ber", 0.0)),
                confidence=float(best.get("confidence", 0.0)),
            )
        )
    return results


@app.post("/api/manifest/run", response_model=ManifestRunResponse)
async def run_manifest_endpoint(payload: ManifestRunRequest) -> ManifestRunResponse:
    manifest_dir = Path("experiments/manifests/generated")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    name = str(payload.manifest.get("name", "api_manifest"))
    safe_name = "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_"}).strip("-_") or "api_manifest"
    manifest_path = manifest_dir / f"{safe_name}.json"
    manifest_path.write_text(json.dumps(payload.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    results = run_manifest(manifest_path)
    out_dir = Path(payload.manifest.get("out_dir", "experiments/results")) / payload.manifest.get("name", "manifest")
    return ManifestRunResponse(result_count=len(results), output_dir=str(out_dir))


@app.post("/api/report", response_model=ReportResponse)
async def create_report(payload: ReportRequest) -> ReportResponse:
    result_path = Path(payload.result_path)
    if not result_path.exists() or result_path.suffix != ".json":
        raise HTTPException(404, "결과 JSON을 찾을 수 없습니다.")

    report_path = result_path.with_suffix(".pdf")
    generate_pdf_report(result_path, report_path)
    report_bytes = report_path.read_bytes()
    return ReportResponse(
        report_path=str(report_path),
        report_b64=base64.b64encode(report_bytes).decode("utf-8"),
    )


@app.get("/api/analysis", response_model=AnalysisResponse)
async def analysis() -> AnalysisResponse:
    results = load_results(Path("experiments/results"))
    return AnalysisResponse(
        overall=summarize(results),
        by_condition=summarize_by_condition(results),
    )
