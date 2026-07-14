"use client";

import { useEffect, useRef, useState } from "react";
import {
  Download,
  FileSearch,
  FileText,
  FolderOpen,
  ImageIcon,
  Microscope,
  Play,
  RefreshCw,
  Save,
  SearchCheck,
  ShieldCheck,
  SlidersHorizontal,
  UploadCloud,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface EmbedResult {
  watermarked_b64: string;
  original_b64: string;
  psnr: number;
  user_id: string;
  strength: number;
  psnr_pass: boolean;
}

interface SimulateResult {
  degraded_b64: string;
  original_b64: string;
  jpeg_quality: number;
  noise_sigma: number;
  downscale: number;
}

interface Candidate {
  user_id: string;
  hamming: number;
  confidence: number;
}

interface ExtractResult {
  top_candidates: Candidate[];
  ber: number;
  ber_pass: boolean;
  identified: boolean;
  abstained: boolean;
  reason: string;
}

interface ResultSummary {
  path: string;
  run_id: string;
  user_id: string;
  condition: string;
  identified: boolean;
  abstained: boolean;
  ber: number;
  confidence: number;
}

interface AnalysisSummary {
  overall: {
    count: number;
    avg_ber?: number;
    identified_rate?: number;
    abstain_rate?: number;
    ber_95ci?: {
      low: number;
      high: number;
    };
  };
}

type AssetType = "image" | "pdf" | "video";
type WorkflowStep = "intake" | "embed" | "capture" | "analysis";

const API = "/api";
const REVIEWERS = ["user_003", "user_007", "user_012", "user_021", "user_042", "user_056"];

async function apiPost<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API}${path}`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "API error");
  }
  return res.json();
}

async function apiJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "API error");
  }
  return res.json();
}

function StatusPill({
  tone,
  children,
}: {
  tone: "neutral" | "success" | "warning" | "danger";
  children: React.ReactNode;
}) {
  return <span className={`status-pill ${tone}`}>{children}</span>;
}

function MetricTile({
  label,
  value,
  detail,
  tone = "neutral",
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "neutral" | "success" | "warning" | "danger";
}) {
  return (
    <div className="metric-tile">
      <span className="metric-label">{label}</span>
      <strong className={`metric-number ${tone}`}>{value}</strong>
      {detail && <span className="metric-detail">{detail}</span>}
    </div>
  );
}

function StepRail({ active }: { active: WorkflowStep }) {
  const steps: Array<{ id: WorkflowStep; label: string }> = [
    { id: "intake", label: "Intake" },
    { id: "embed", label: "Embed" },
    { id: "capture", label: "Capture" },
    { id: "analysis", label: "Analysis" },
  ];
  const activeIndex = steps.findIndex((step) => step.id === active);

  return (
    <div className="step-rail">
      {steps.map((step, index) => (
        <div
          key={step.id}
          className={`step-chip ${index < activeIndex ? "done" : index === activeIndex ? "active" : ""}`}
        >
          <span>{index + 1}</span>
          {step.label}
        </div>
      ))}
    </div>
  );
}

function PreviewPane({ src, assetType }: { src: string | null; assetType: AssetType }) {
  if (!src) {
    return (
      <div className="empty-preview">
        <span>No asset selected</span>
      </div>
    );
  }

  if (assetType === "video") {
    return <video className="asset-preview" src={src} controls />;
  }

  if (assetType === "pdf") {
    return (
      <object className="asset-preview" data={src} type="application/pdf">
        <span>PDF preview unavailable</span>
      </object>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img className="asset-preview" src={src} alt="Review asset preview" />
  );
}

function Base64Image({
  src,
  label,
}: {
  src: string;
  label: string;
}) {
  return (
    <div className="compare-frame">
      <div className="frame-label">{label}</div>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`data:image/png;base64,${src}`} alt={label} />
    </div>
  );
}

export default function SecureViewerPage() {
  const [caseTitle, setCaseTitle] = useState("Pre-release review investigation");
  const [caseNote, setCaseNote] = useState(
    "Reviewer-specific watermark validation. Attach real captures to manifest runs before external review.",
  );
  const [reviewerId, setReviewerId] = useState("user_003");
  const [assetType, setAssetType] = useState<AssetType>("image");
  const [activeStep, setActiveStep] = useState<WorkflowStep>("intake");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [casePath, setCasePath] = useState<string | null>(null);
  const [embedResult, setEmbedResult] = useState<EmbedResult | null>(null);
  const [simulateResult, setSimulateResult] = useState<SimulateResult | null>(null);
  const [extractResult, setExtractResult] = useState<ExtractResult | null>(null);
  const [results, setResults] = useState<ResultSummary[]>([]);
  const [analysis, setAnalysis] = useState<AnalysisSummary | null>(null);
  const [strength, setStrength] = useState(6);
  const [jpegQ, setJpegQ] = useState(80);
  const [sigma, setSigma] = useState(0);
  const [downscale, setDownscale] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void refreshResults();
  }, []);

  async function refreshResults() {
    try {
      const [resultData, analysisData] = await Promise.all([
        apiJson<ResultSummary[]>("/results"),
        apiJson<AnalysisSummary>("/analysis"),
      ]);
      setResults(resultData.slice(-8).reverse());
      setAnalysis(analysisData);
    } catch {
      setResults([]);
      setAnalysis(null);
    }
  }

  function setSelectedFile(nextFile: File | null) {
    if (!nextFile) return;
    setFile(nextFile);
    setError(null);
    setEmbedResult(null);
    setSimulateResult(null);
    setExtractResult(null);
    setActiveStep("intake");

    if (nextFile.type === "application/pdf") setAssetType("pdf");
    else if (nextFile.type.startsWith("video/")) setAssetType("video");
    else setAssetType("image");

    const reader = new FileReader();
    reader.onload = (event) => setPreview(event.target?.result as string);
    reader.readAsDataURL(nextFile);
  }

  function createSampleAsset() {
    const canvas = document.createElement("canvas");
    canvas.width = 960;
    canvas.height = 960;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#f4f0e8";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#20251f";
    ctx.font = "600 42px sans-serif";
    ctx.fillText("CONFIDENTIAL REVIEW", 110, 220);
    ctx.font = "24px sans-serif";
    ctx.fillText("Validation asset for controlled screen-capture tests", 110, 270);
    ctx.strokeStyle = "#79916b";
    ctx.lineWidth = 3;
    ctx.strokeRect(110, 330, 740, 420);
    ctx.fillStyle = "#d8ded0";
    ctx.fillRect(140, 360, 680, 360);
    ctx.fillStyle = "#4f6f58";
    ctx.font = "32px sans-serif";
    ctx.fillText("Secure Viewer Sample", 250, 545);

    canvas.toBlob((blob) => {
      if (!blob) return;
      setSelectedFile(new File([blob], "review_asset.png", { type: "image/png" }));
    });
  }

  async function saveCase() {
    setLoading(true);
    setError(null);
    try {
      const created = await apiJson<{ case_id: string; path: string }>("/cases", {
        title: caseTitle,
        asset_type: assetType,
        reviewer_id: reviewerId,
        notes: caseNote,
      });
      setCasePath(created.path);
      await refreshResults();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save case");
    } finally {
      setLoading(false);
    }
  }

  async function embedAsset() {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Image files are required for automated embed/extract validation in this pilot build.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("user_id", reviewerId);
      form.append("strength", String(strength));
      form.append("mode", "block");
      const result = await apiPost<EmbedResult>("/embed", form);
      setEmbedResult(result);
      setActiveStep("embed");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to embed watermark");
    } finally {
      setLoading(false);
    }
  }

  async function simulateCapture() {
    if (!embedResult) return;

    setLoading(true);
    setError(null);
    try {
      const binary = atob(embedResult.watermarked_b64);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }

      const form = new FormData();
      form.append("file", new File([bytes], "watermarked.png", { type: "image/png" }));
      form.append("jpeg_quality", String(jpegQ));
      form.append("noise_sigma", String(sigma));
      form.append("downscale", String(downscale));
      const result = await apiPost<SimulateResult>("/simulate", form);
      setSimulateResult(result);
      setActiveStep("capture");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to simulate capture");
    } finally {
      setLoading(false);
    }
  }

  async function extractCandidate() {
    if (!simulateResult) return;

    setLoading(true);
    setError(null);
    try {
      const binary = atob(simulateResult.degraded_b64);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }

      const form = new FormData();
      form.append("file", new File([bytes], "suspect.png", { type: "image/png" }));
      form.append("mode", "block");
      const result = await apiPost<ExtractResult>("/extract", form);
      setExtractResult(result);
      setActiveStep("analysis");
      await refreshResults();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to extract candidate");
    } finally {
      setLoading(false);
    }
  }

  async function downloadReport(resultPath: string) {
    setLoading(true);
    setError(null);
    try {
      const report = await apiJson<{ report_path: string; report_b64: string }>("/report", {
        result_path: resultPath,
      });
      const link = document.createElement("a");
      link.href = `data:application/pdf;base64,${report.report_b64}`;
      link.download = report.report_path.split("/").pop() ?? "nirguard-report.pdf";
      link.click();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to generate report");
    } finally {
      setLoading(false);
    }
  }

  const bestCandidate = extractResult?.top_candidates[0];
  const overall = analysis?.overall;
  const quickActions = [
    { label: "Asset", icon: UploadCloud, onClick: () => fileInput.current?.click(), disabled: loading },
    { label: "Sample", icon: ImageIcon, onClick: createSampleAsset, disabled: loading },
    { label: "Embed", icon: ShieldCheck, onClick: embedAsset, disabled: loading || !file },
    { label: "Capture", icon: Play, onClick: simulateCapture, disabled: loading || !embedResult },
    { label: "Attribute", icon: SearchCheck, onClick: extractCandidate, disabled: loading || !simulateResult },
  ];

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <span className="brand-mark">
            <ShieldCheck size={19} strokeWidth={2.2} />
          </span>
          <div>
            <h1>NIRGuard</h1>
            <p>Secure Viewer</p>
          </div>
        </div>
        <nav className="side-nav">
          <span className="active"><FileSearch size={16} />Investigation</span>
          <span><Microscope size={16} />Validation Lab</span>
          <span><FileText size={16} />Evidence</span>
          <span><SlidersHorizontal size={16} />Settings</span>
        </nav>
        <div className="sidebar-note">
          <span>Mode</span>
          <strong>Block watermark</strong>
          <small>Confidence + abstain policy</small>
        </div>
      </aside>

      <section className="main-surface">
        <header className="topbar">
          <div>
            <p className="eyebrow">Controlled Review Case</p>
            <h2>{caseTitle}</h2>
          </div>
          <div className="topbar-actions">
            <Button className="button secondary" onClick={refreshResults} disabled={loading}>
              <RefreshCw size={16} />
              Refresh
            </Button>
            <Button className="button primary" onClick={saveCase} disabled={loading}>
              <Save size={16} />
              Save Case
            </Button>
          </div>
        </header>

        {error && <div className="alert">{error}</div>}

        <section className="case-strip">
          <label>
            <span>Case name</span>
            <input value={caseTitle} onChange={(event) => setCaseTitle(event.target.value)} />
          </label>
          <label>
            <span>Reviewer</span>
            <select
              value={reviewerId}
              onChange={(event) => setReviewerId(event.target.value)}
            >
              {REVIEWERS.map((reviewer) => (
                <option key={reviewer} value={reviewer}>
                  {reviewer}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Asset type</span>
            <select
              value={assetType}
              onChange={(event) => setAssetType(event.target.value as AssetType)}
            >
              <option value="image">Image</option>
              <option value="pdf">PDF</option>
              <option value="video">Video</option>
            </select>
          </label>
          <div className="record-path">
            <span>Record</span>
            <strong>{casePath ?? "unsaved"}</strong>
          </div>
        </section>

        <section className="command-dock">
          <Textarea
            value={caseNote}
            onChange={(event) => setCaseNote(event.target.value)}
            className="command-input"
            placeholder="Case note"
          />
          <div className="command-footer">
            <span className="command-label">Fast actions</span>
            <div className="quick-actions">
              {quickActions.map((action) => {
                const Icon = action.icon;
                return (
                  <Button
                    key={action.label}
                    className="quick-action"
                    onClick={action.onClick}
                    disabled={action.disabled}
                  >
                    <Icon size={15} />
                    {action.label}
                  </Button>
                );
              })}
            </div>
          </div>
        </section>

        <section className="metrics-grid">
          <MetricTile
            label="Stored runs"
            value={String(overall?.count ?? results.length)}
            detail="local validation results"
          />
          <MetricTile
            label="Average BER"
            value={overall?.avg_ber !== undefined ? `${(overall.avg_ber * 100).toFixed(1)}%` : "--"}
            detail={overall?.ber_95ci ? `CI ${overall.ber_95ci.low}-${overall.ber_95ci.high}` : "no interval"}
            tone="warning"
          />
          <MetricTile
            label="Identified"
            value={overall?.identified_rate !== undefined ? `${overall.identified_rate.toFixed(0)}%` : "--"}
            detail="accepted decisions"
            tone="success"
          />
          <MetricTile
            label="Abstain"
            value={overall?.abstain_rate !== undefined ? `${overall.abstain_rate.toFixed(0)}%` : "--"}
            detail="uncertain captures"
          />
        </section>

        <StepRail active={activeStep} />

        <section className="workspace-grid">
          <div className="panel asset-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Review Asset</p>
                <h3>{file?.name ?? "No file selected"}</h3>
              </div>
              <StatusPill tone={file ? "success" : "neutral"}>{assetType}</StatusPill>
            </div>
            <PreviewPane src={preview} assetType={assetType} />
            <div className="action-row">
              <input
                ref={fileInput}
                type="file"
                className="hidden"
                accept="image/png,image/jpeg,application/pdf,video/mp4,video/quicktime"
                onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
              />
              <Button className="button secondary" onClick={() => fileInput.current?.click()}>
                <FolderOpen size={16} />
                Select File
              </Button>
              <Button className="button secondary" onClick={createSampleAsset}>
                <ImageIcon size={16} />
                Load Sample
              </Button>
              <Button className="button primary" onClick={embedAsset} disabled={loading || !file}>
                <ShieldCheck size={16} />
                Embed
              </Button>
            </div>
          </div>

          <div className="panel control-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Capture Model</p>
                <h3>Synthetic Recapture</h3>
              </div>
              <StatusPill tone="warning">lab</StatusPill>
            </div>
            <label className="range-field">
              <span>Strength {strength}</span>
              <input
                type="range"
                min={2}
                max={8}
                value={strength}
                onChange={(event) => setStrength(Number(event.target.value))}
              />
            </label>
            <label className="range-field">
              <span>JPEG Q {jpegQ}</span>
              <input
                type="range"
                min={1}
                max={100}
                value={jpegQ}
                onChange={(event) => setJpegQ(Number(event.target.value))}
              />
            </label>
            <label className="range-field">
              <span>Noise sigma {sigma}</span>
              <input
                type="range"
                min={0}
                max={30}
                value={sigma}
                onChange={(event) => setSigma(Number(event.target.value))}
              />
            </label>
            <label className="range-field">
              <span>Scale {(downscale * 100).toFixed(0)}%</span>
              <input
                type="range"
                min={0.25}
                max={1}
                step={0.05}
                value={downscale}
                onChange={(event) => setDownscale(Number(event.target.value))}
              />
            </label>
            <div className="action-stack">
              <Button className="button primary" onClick={simulateCapture} disabled={loading || !embedResult}>
                <Play size={16} />
                Simulate Capture
              </Button>
              <Button className="button primary" onClick={extractCandidate} disabled={loading || !simulateResult}>
                <SearchCheck size={16} />
                Run Attribution
              </Button>
            </div>
          </div>
        </section>

        {embedResult && (
          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Watermarked Output</p>
                <h3>Visual comparison</h3>
              </div>
              <StatusPill tone={embedResult.psnr_pass ? "success" : "warning"}>
                PSNR {embedResult.psnr.toFixed(1)} dB
              </StatusPill>
            </div>
            <div className="compare-grid">
              <Base64Image src={embedResult.original_b64} label="Original" />
              <Base64Image src={embedResult.watermarked_b64} label="Watermarked" />
            </div>
          </section>
        )}

        {simulateResult && (
          <section className="panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Suspect Capture</p>
                <h3>Degradation preview</h3>
              </div>
              <StatusPill tone="neutral">
                Q{simulateResult.jpeg_quality} / sigma {simulateResult.noise_sigma}
              </StatusPill>
            </div>
            <div className="compare-grid">
              <Base64Image src={simulateResult.original_b64} label="Watermarked source" />
              <Base64Image src={simulateResult.degraded_b64} label="Suspect capture" />
            </div>
          </section>
        )}

        {extractResult && (
          <section className="panel investigation-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Attribution Decision</p>
                <h3>{extractResult.abstained ? "Abstained" : extractResult.identified ? "Candidate accepted" : "No attribution"}</h3>
              </div>
              <StatusPill tone={extractResult.abstained ? "warning" : extractResult.identified ? "success" : "danger"}>
                {extractResult.reason}
              </StatusPill>
            </div>
            <div className="decision-grid">
              <MetricTile
                label="Best candidate"
                value={bestCandidate?.user_id ?? "--"}
                detail="ranked by hamming distance"
                tone={extractResult.identified ? "success" : "warning"}
              />
              <MetricTile
                label="Confidence"
                value={bestCandidate ? `${bestCandidate.confidence.toFixed(1)}%` : "--"}
                detail="technical score"
                tone={extractResult.identified ? "success" : "warning"}
              />
              <MetricTile
                label="BER"
                value={`${(extractResult.ber * 100).toFixed(1)}%`}
                detail="best match"
                tone={extractResult.ber_pass ? "success" : "warning"}
              />
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>User</th>
                  <th>Hamming</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {extractResult.top_candidates.map((candidate, index) => (
                  <tr key={candidate.user_id}>
                    <td>{index + 1}</td>
                    <td className="mono">{candidate.user_id}</td>
                    <td>{(candidate.hamming * 100).toFixed(1)}%</td>
                    <td>{candidate.confidence.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        <section className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Validation Dashboard</p>
              <h3>Recent experiment results</h3>
            </div>
            <StatusPill tone="neutral">{results.length} rows</StatusPill>
          </div>
          {results.length === 0 ? (
            <div className="empty-table">No stored results</div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Condition</th>
                  <th>User</th>
                  <th>BER</th>
                  <th>Confidence</th>
                  <th>Decision</th>
                  <th>Report</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => (
                  <tr key={result.path}>
                    <td className="mono">{result.run_id.slice(-10)}</td>
                    <td>{result.condition}</td>
                    <td className="mono">{result.user_id}</td>
                    <td>{(result.ber * 100).toFixed(1)}%</td>
                    <td>{result.confidence.toFixed(1)}%</td>
                    <td>
                      <StatusPill tone={result.abstained ? "warning" : result.identified ? "success" : "danger"}>
                        {result.abstained ? "Abstain" : result.identified ? "Identified" : "Failed"}
                      </StatusPill>
                    </td>
                    <td>
                      <button className="table-button" onClick={() => downloadReport(result.path)} disabled={loading}>
                        <Download size={14} />
                        PDF
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </section>
    </main>
  );
}
