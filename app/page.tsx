"use client";

import {
  Activity,
  Check,
  ChevronDown,
  Columns2,
  Download,
  FileCheck2,
  FileUp,
  Layers3,
  LocateFixed,
  Play,
  Rows3,
  Ruler,
  Satellite,
  Settings,
  Sprout,
  TriangleAlert,
} from "lucide-react";
import type { GeoJsonObject } from "geojson";
import type {
  GeoJSON as LeafletGeoJSON,
  Map as LeafletMap,
  TileLayer,
} from "leaflet";
import { useCallback, useEffect, useRef, useState } from "react";

const MAX_FILE_BYTES = 500 * 1024 * 1024;
const API_BASE = (process.env.NEXT_PUBLIC_ANALYSIS_API_URL || "/api").replace(
  /\/$/,
  "",
);
const METHOD_LABELS = {
  fft: "Fourier / FFT",
  strip_nb: "Strip without birth",
  strip: "Strip + dynamic birth",
} as const;

type MethodKey = keyof typeof METHOD_LABELS;

type Summary = {
  n_rows: number;
  n_gap_segments: number;
  total_gap_length_m: number;
  overall_gap_pct: number;
  median_row_spacing_m: number;
};

type MethodResult = {
  method: string;
  method_label: string;
  summary: Summary;
  rows_path: string;
  gaps_path: string;
  csv_path: string;
};

type AnalysisResult = {
  default_method: MethodKey;
  bounds: [[number, number], [number, number]];
  tile_path: string;
  methods: Record<MethodKey, MethodResult>;
};

type JobStatus = {
  job_id: string;
  status: "queued" | "processing" | "succeeded" | "failed";
  progress: number;
  message: string;
  result?: AnalysisResult;
};

function apiUrl(path: string) {
  if (/^https?:\/\//.test(path)) return path;
  if (path.startsWith("/api/")) {
    const root = API_BASE.endsWith("/api") ? API_BASE.slice(0, -4) : "";
    return `${root}${path}`;
  }
  return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function Home() {
  const mapElement = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const rasterRef = useRef<TileLayer | null>(null);
  const rowsRef = useRef<LeafletGeoJSON | null>(null);
  const gapsRef = useRef<LeafletGeoJSON | null>(null);
  const leftRowsRef = useRef<LeafletGeoJSON | null>(null);
  const leftGapsRef = useRef<LeafletGeoJSON | null>(null);
  const rightRowsRef = useRef<LeafletGeoJSON | null>(null);
  const rightGapsRef = useRef<LeafletGeoJSON | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const overlayRequestRef = useRef(0);
  const methodDataCacheRef = useRef<
    Partial<Record<MethodKey, { rows: GeoJsonObject; gaps: GeoJsonObject }>>
  >({});

  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState("");
  const [gapDistance, setGapDistance] = useState("0.5");
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState<"idle" | JobStatus["status"]>("idle");
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("Ready for analysis");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [showRows, setShowRows] = useState(true);
  const [showGaps, setShowGaps] = useState(true);
  const [selectedMethod, setSelectedMethod] = useState<MethodKey>("strip");
  const [compareMode, setCompareMode] = useState(false);
  const [leftMethod, setLeftMethod] = useState<MethodKey>("fft");
  const [rightMethod, setRightMethod] = useState<MethodKey>("strip");
  const [splitPosition, setSplitPosition] = useState(50);
  const [isSplitDragging, setIsSplitDragging] = useState(false);
  const [coords, setCoords] = useState("28.613900, 77.209000");

  useEffect(() => {
    let active = true;
    async function createMap() {
      if (!mapElement.current || mapRef.current) return;
      const L = await import("leaflet");
      if (!active || !mapElement.current) return;

      const map = L.map(mapElement.current, {
        zoomControl: false,
        attributionControl: true,
        minZoom: 2,
      }).setView([28.6139, 77.209], 5);
      L.control.zoom({ position: "topright" }).addTo(map);
      L.control
        .scale({ position: "bottomleft", metric: true, imperial: false })
        .addTo(map);
      map.createPane("satellite");
      map.getPane("satellite")!.style.zIndex = "200";
      map.createPane("orthomosaic");
      map.getPane("orthomosaic")!.style.zIndex = "300";
      map.createPane("comparisonLeft");
      map.getPane("comparisonLeft")!.style.zIndex = "410";
      map.getPane("comparisonLeft")!.style.pointerEvents = "none";
      map.createPane("comparisonRight");
      map.getPane("comparisonRight")!.style.zIndex = "420";
      map.getPane("comparisonRight")!.style.pointerEvents = "none";
      L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
          attribution:
            "Satellite imagery &copy; Esri and imagery contributors",
          pane: "satellite",
          maxNativeZoom: 17,
          maxZoom: 24,
        },
      ).addTo(map);
      L.tileLayer(
        "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        {
          attribution: "Labels &copy; Esri",
          pane: "satellite",
          maxNativeZoom: 17,
          maxZoom: 24,
        },
      ).addTo(map);
      map.on("mousemove", (event) => {
        setCoords(
          `${event.latlng.lat.toFixed(6)}, ${event.latlng.lng.toFixed(6)}`,
        );
      });
      mapRef.current = map;
    }
    createMap();
    return () => {
      active = false;
      mapRef.current?.remove();
      mapRef.current = null;
    };
  }, []);

  const validateFile = useCallback((candidate: File) => {
    const suffix = candidate.name.toLowerCase();
    if (!suffix.endsWith(".tif") && !suffix.endsWith(".tiff")) {
      setFile(null);
      setFileError("Please select a GeoTIFF (.tif or .tiff).");
      return;
    }
    if (candidate.size > MAX_FILE_BYTES) {
      setFile(null);
      setFileError("The selected file exceeds the 500 MB limit.");
      return;
    }
    setFile(candidate);
    setFileError("");
    setError("");
    setResult(null);
    methodDataCacheRef.current = {};
    setSelectedMethod("strip");
    setCompareMode(false);
    setSplitPosition(50);
    setStatus("idle");
    setStatusMessage("Ready for analysis");
    setProgress(0);
  }, []);

  const clearVectorLayers = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    overlayRequestRef.current += 1;
    [
      rowsRef,
      gapsRef,
      leftRowsRef,
      leftGapsRef,
      rightRowsRef,
      rightGapsRef,
    ].forEach((layerRef) => {
      if (layerRef.current && map.hasLayer(layerRef.current)) {
        map.removeLayer(layerRef.current);
      }
      layerRef.current = null;
    });
  }, []);

  const clearMapLayers = useCallback(() => {
    const map = mapRef.current;
    if (!map) return;
    clearVectorLayers();
    if (rasterRef.current && map.hasLayer(rasterRef.current)) {
      map.removeLayer(rasterRef.current);
    }
    rasterRef.current = null;
  }, [clearVectorLayers]);

  const applyComparisonClip = useCallback((position: number) => {
    const map = mapRef.current;
    if (!map) return;
    const leftPane = map.getPane("comparisonLeft");
    const rightPane = map.getPane("comparisonRight");
    leftPane?.querySelectorAll("svg").forEach((element) => {
      element.style.clipPath = `inset(0 ${100 - position}% 0 0)`;
    });
    rightPane?.querySelectorAll("svg").forEach((element) => {
      element.style.clipPath = `inset(0 0 0 ${position}%)`;
    });
  }, []);

  useEffect(() => {
    applyComparisonClip(splitPosition);
  }, [applyComparisonClip, splitPosition]);

  const fetchMethodData = useCallback(
    async (analysis: AnalysisResult, method: MethodKey) => {
      const cached = methodDataCacheRef.current[method];
      if (cached) return cached;

      const methodResult = analysis.methods[method];
      const [rows, gaps] = await Promise.all([
        fetch(apiUrl(methodResult.rows_path)).then((response) => {
          if (!response.ok) throw new Error("Could not load row overlay");
          return response.json() as Promise<GeoJsonObject>;
        }),
        fetch(apiUrl(methodResult.gaps_path)).then((response) => {
          if (!response.ok) throw new Error("Could not load gap overlay");
          return response.json() as Promise<GeoJsonObject>;
        }),
      ]);
      const data = { rows, gaps };
      methodDataCacheRef.current[method] = data;
      return data;
    },
    [],
  );

  const displayResult = useCallback(
    async (
      analysis: AnalysisResult,
      method: MethodKey,
      fitToBounds = false,
    ) => {
      const map = mapRef.current;
      if (!map) return;
      const L = await import("leaflet");
      clearVectorLayers();
      const requestId = overlayRequestRef.current;

      if (!rasterRef.current) {
        rasterRef.current = L.tileLayer(apiUrl(analysis.tile_path), {
          minZoom: 2,
          maxZoom: 24,
          maxNativeZoom: 24,
          bounds: analysis.bounds,
          opacity: 1,
          pane: "orthomosaic",
          attribution: "Uploaded orthomosaic",
        }).addTo(map);
      }

      const { rows, gaps } = await fetchMethodData(analysis, method);
      if (requestId !== overlayRequestRef.current) return;

      rowsRef.current = L.geoJSON(rows, {
        style: { color: "#f3d34a", weight: 2, opacity: 0.92 },
      });
      gapsRef.current = L.geoJSON(gaps, {
        style: { color: "#ff4d5e", weight: 4, opacity: 1 },
      });
      if (showRows) rowsRef.current.addTo(map);
      if (showGaps) gapsRef.current.addTo(map);
      if (fitToBounds) {
        map.fitBounds(analysis.bounds, {
          padding: [26, 26],
          animate: true,
        });
      }
    },
    [clearVectorLayers, fetchMethodData, showGaps, showRows],
  );

  const displayComparison = useCallback(
    async (
      analysis: AnalysisResult,
      left: MethodKey,
      right: MethodKey,
      fitToBounds = false,
    ) => {
      const map = mapRef.current;
      if (!map) return;
      const L = await import("leaflet");
      clearVectorLayers();
      const requestId = overlayRequestRef.current;

      if (!rasterRef.current) {
        rasterRef.current = L.tileLayer(apiUrl(analysis.tile_path), {
          minZoom: 2,
          maxZoom: 24,
          maxNativeZoom: 24,
          bounds: analysis.bounds,
          opacity: 1,
          pane: "orthomosaic",
          attribution: "Uploaded orthomosaic",
        }).addTo(map);
      }

      const [leftData, rightData] = await Promise.all([
        fetchMethodData(analysis, left),
        fetchMethodData(analysis, right),
      ]);
      if (requestId !== overlayRequestRef.current) return;
      const leftRenderer = L.svg({
        pane: "comparisonLeft",
        padding: 0,
      });
      const rightRenderer = L.svg({
        pane: "comparisonRight",
        padding: 0,
      });

      leftRowsRef.current = L.geoJSON(leftData.rows, {
        style: {
          color: "#f3d34a",
          weight: 2,
          opacity: 0.92,
          pane: "comparisonLeft",
          renderer: leftRenderer,
        },
      });
      leftGapsRef.current = L.geoJSON(leftData.gaps, {
        style: {
          color: "#ff4d5e",
          weight: 4,
          opacity: 1,
          pane: "comparisonLeft",
          renderer: leftRenderer,
        },
      });
      rightRowsRef.current = L.geoJSON(rightData.rows, {
        style: {
          color: "#f3d34a",
          weight: 2,
          opacity: 0.92,
          pane: "comparisonRight",
          renderer: rightRenderer,
        },
      });
      rightGapsRef.current = L.geoJSON(rightData.gaps, {
        style: {
          color: "#ff4d5e",
          weight: 4,
          opacity: 1,
          pane: "comparisonRight",
          renderer: rightRenderer,
        },
      });

      if (showRows) {
        leftRowsRef.current.addTo(map);
        rightRowsRef.current.addTo(map);
      }
      if (showGaps) {
        leftGapsRef.current.addTo(map);
        rightGapsRef.current.addTo(map);
      }
      applyComparisonClip(splitPosition);
      if (fitToBounds) {
        map.fitBounds(analysis.bounds, {
          padding: [26, 26],
          animate: true,
        });
      }
    },
    [
      applyComparisonClip,
      clearVectorLayers,
      fetchMethodData,
      showGaps,
      showRows,
      splitPosition,
    ],
  );

  useEffect(() => {
    if (!jobId || status === "succeeded" || status === "failed") return;
    let stopped = false;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/jobs/${jobId}`, {
          cache: "no-store",
        });
        if (!response.ok) throw new Error("Could not read analysis status");
        const payload = (await response.json()) as JobStatus;
        if (stopped) return;
        setStatus(payload.status);
        setProgress(payload.progress || 0);
        setStatusMessage(payload.message);
        if (payload.status === "succeeded" && payload.result) {
          window.clearInterval(timer);
          setResult(payload.result);
          setSelectedMethod(payload.result.default_method);
          await displayResult(
            payload.result,
            payload.result.default_method,
            true,
          );
        }
        if (payload.status === "failed") {
          window.clearInterval(timer);
          setError(payload.message || "Analysis failed.");
        }
      } catch (pollError) {
        window.clearInterval(timer);
        if (!stopped) {
          setStatus("failed");
          setError(
            pollError instanceof Error ? pollError.message : "Analysis failed.",
          );
        }
      }
    }, 1800);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [displayResult, jobId, status]);

  async function changeMethod(method: MethodKey) {
    setSelectedMethod(method);
    setError("");
    if (!result) return;
    try {
      await displayResult(result, method);
    } catch (methodError) {
      setError(
        methodError instanceof Error
          ? methodError.message
          : "Could not switch analysis method.",
      );
    }
  }

  async function toggleComparison() {
    if (!result) return;
    const nextMode = !compareMode;
    setCompareMode(nextMode);
    setError("");
    try {
      if (nextMode) {
        await displayComparison(result, leftMethod, rightMethod);
      } else {
        await displayResult(result, selectedMethod);
      }
    } catch (comparisonError) {
      setError(
        comparisonError instanceof Error
          ? comparisonError.message
          : "Could not update comparison view.",
      );
    }
  }

  async function changeComparisonMethod(
    side: "left" | "right",
    method: MethodKey,
  ) {
    const nextLeft = side === "left" ? method : leftMethod;
    const nextRight = side === "right" ? method : rightMethod;
    if (side === "left") setLeftMethod(method);
    else setRightMethod(method);
    setError("");
    if (!result || !compareMode) return;
    try {
      await displayComparison(result, nextLeft, nextRight);
    } catch (comparisonError) {
      setError(
        comparisonError instanceof Error
          ? comparisonError.message
          : "Could not switch comparison method.",
      );
    }
  }

  function updateSplitPosition(clientX: number) {
    const rect = mapElement.current?.getBoundingClientRect();
    if (!rect) return;
    const position = Math.min(
      95,
      Math.max(5, ((clientX - rect.left) / rect.width) * 100),
    );
    setSplitPosition(position);
    applyComparisonClip(position);
  }

  function nudgeSplitPosition(delta: number) {
    const position = Math.min(95, Math.max(5, splitPosition + delta));
    setSplitPosition(position);
    applyComparisonClip(position);
  }

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    [rowsRef, leftRowsRef, rightRowsRef].forEach((layerRef) => {
      if (!layerRef.current) return;
      if (showRows) layerRef.current.addTo(map);
      else map.removeLayer(layerRef.current);
    });
    if (compareMode) applyComparisonClip(splitPosition);
  }, [applyComparisonClip, compareMode, showRows, splitPosition]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    [gapsRef, leftGapsRef, rightGapsRef].forEach((layerRef) => {
      if (!layerRef.current) return;
      if (showGaps) layerRef.current.addTo(map);
      else map.removeLayer(layerRef.current);
    });
    if (compareMode) applyComparisonClip(splitPosition);
  }, [applyComparisonClip, compareMode, showGaps, splitPosition]);

  async function runAnalysis() {
    if (!file) return;
    const minGap = Number(gapDistance);
    if (!Number.isFinite(minGap) || minGap <= 0 || minGap > 20) {
      setError("Minimum gap distance must be between 0 and 20 metres.");
      return;
    }

    clearMapLayers();
    setError("");
    setResult(null);
    methodDataCacheRef.current = {};
    setCompareMode(false);
    setSplitPosition(50);
    setStatus("queued");
    setProgress(2);
    setStatusMessage("Uploading orthomosaic");

    const form = new FormData();
    form.append("orthomosaic", file);
    form.append("min_gap_m", String(minGap));

    try {
      const response = await fetch(`${API_BASE}/jobs`, {
        method: "POST",
        body: form,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Upload failed");
      }
      setJobId(payload.job_id);
      setStatus("queued");
      setProgress(5);
      setStatusMessage("Waiting for an analysis worker");
    } catch (uploadError) {
      setStatus("failed");
      setError(
        uploadError instanceof Error ? uploadError.message : "Upload failed.",
      );
    }
  }

  const isProcessing = status === "queued" || status === "processing";
  const canRun = Boolean(file) && !fileError && !isProcessing;
  const metricMethod = compareMode ? leftMethod : selectedMethod;
  const methodResult = result?.methods[metricMethod];
  const summary = methodResult?.summary;
  const selectedMethodLabel = METHOD_LABELS[selectedMethod];
  const metricMethodLabel = METHOD_LABELS[metricMethod];
  const headerMethodLabel = compareMode
    ? `${METHOD_LABELS[leftMethod]} ↔ ${METHOD_LABELS[rightMethod]}`
    : selectedMethodLabel;

  return (
    <main className="console-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Satellite size={21} />
            <span />
          </div>
          <div>
            <h1>Crop Row Gap Analyzer</h1>
            <p>Orthomosaic intelligence console</p>
          </div>
        </div>
        <div className="topbar-status">
          <span className={`status-chip status-${status}`}>
            <i />{" "}
            {status === "succeeded"
              ? "Completed"
              : isProcessing
                ? "Processing"
                : status === "failed"
                  ? "Attention"
                  : "System ready"}
          </span>
          <span className="method-chip">{headerMethodLabel}</span>
          <span className="signal">
            <LocateFixed size={16} /> GEO{" "}
            <b>
              <i />
              <i />
              <i />
              <i />
            </b>
          </span>
          <button className="icon-button" aria-label="Settings">
            <Settings size={20} />
          </button>
        </div>
      </header>

      <div className="workspace">
        <aside className="control-rail">
          <section className="control-section">
            <div className="section-kicker">
              <FileUp size={16} /> INPUT
            </div>
            <h2>Orthomosaic</h2>
            <input
              ref={fileInput}
              className="sr-only"
              type="file"
              accept=".tif,.tiff,image/tiff"
              onChange={(event) =>
                event.target.files?.[0] &&
                validateFile(event.target.files[0])
              }
            />
            <button
              type="button"
              className={`upload-zone ${dragActive ? "drag-active" : ""} ${file ? "has-file" : ""}`}
              onClick={() => fileInput.current?.click()}
              onDragEnter={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragOver={(event) => event.preventDefault()}
              onDragLeave={() => setDragActive(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDragActive(false);
                if (event.dataTransfer.files[0]) {
                  validateFile(event.dataTransfer.files[0]);
                }
              }}
            >
              {file ? <FileCheck2 size={30} /> : <FileUp size={30} />}
              <strong>{file ? file.name : "Drop GeoTIFF here"}</strong>
              <span>
                {file
                  ? `${formatBytes(file.size)} · ready`
                  : "or click to browse · max 500 MB"}
              </span>
            </button>
            {fileError && <p className="field-error">{fileError}</p>}
          </section>

          <section className="control-section form-stack">
            <label htmlFor="gap-distance">Minimum gap distance</label>
            <div className="unit-input">
              <input
                id="gap-distance"
                type="number"
                min="0.1"
                max="20"
                step="0.1"
                value={gapDistance}
                onChange={(event) => setGapDistance(event.target.value)}
              />
              <span>m</span>
            </div>

            <button
              className="run-button"
              disabled={!canRun}
              onClick={runAnalysis}
            >
              {isProcessing ? (
                <Activity className="spin" size={18} />
              ) : (
                <Play size={18} fill="currentColor" />
              )}
              {isProcessing ? "Analysis running" : "Run analysis"}
            </button>

            <button
              type="button"
              className={`compare-toggle ${compareMode ? "active" : ""}`}
              disabled={!result}
              aria-pressed={compareMode}
              onClick={() => void toggleComparison()}
            >
              <Columns2 size={17} />
              <span>Compare methods</span>
              <b>{compareMode ? "ON" : "OFF"}</b>
            </button>

            {compareMode ? (
              <div className="comparison-select-grid">
                <div>
                  <label htmlFor="left-method">Left method</label>
                  <div className="select-shell">
                    <select
                      id="left-method"
                      value={leftMethod}
                      onChange={(event) =>
                        void changeComparisonMethod(
                          "left",
                          event.target.value as MethodKey,
                        )
                      }
                    >
                      {Object.entries(METHOD_LABELS).map(([value, label]) => (
                        <option key={value} value={value}>
                          {label}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={17} />
                  </div>
                </div>
                <div>
                  <label htmlFor="right-method">Right method</label>
                  <div className="select-shell">
                    <select
                      id="right-method"
                      value={rightMethod}
                      onChange={(event) =>
                        void changeComparisonMethod(
                          "right",
                          event.target.value as MethodKey,
                        )
                      }
                    >
                      {Object.entries(METHOD_LABELS).map(([value, label]) => (
                        <option key={value} value={value}>
                          {label}
                        </option>
                      ))}
                    </select>
                    <ChevronDown size={17} />
                  </div>
                </div>
              </div>
            ) : (
              <>
                <label htmlFor="method">Result method</label>
                <div className="select-shell">
                  <select
                    id="method"
                    value={selectedMethod}
                    onChange={(event) =>
                      void changeMethod(event.target.value as MethodKey)
                    }
                  >
                    {Object.entries(METHOD_LABELS).map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={17} />
                </div>
              </>
            )}
            <p className="method-note">
              {compareMode
                ? "Drag the divider to compare. Statistics and CSV follow the left method."
                : "All three methods run together; switch results here."}
            </p>
          </section>

          <section className="analysis-panel">
            <div className="section-heading">
              <span>
                <Activity size={17} /> ANALYSIS
              </span>
              <small>{progress}%</small>
            </div>
            <div className="progress-track">
              <i style={{ width: `${progress}%` }} />
            </div>
            <p>{statusMessage}</p>
            {error && (
              <div className="error-card">
                <TriangleAlert size={16} /> {error}
              </div>
            )}

            <div className="metrics">
              <article>
                <Rows3 size={20} />
                <div>
                  <span>Rows detected</span>
                  <strong>{summary?.n_rows ?? "—"}</strong>
                </div>
              </article>
              <article className="metric-red">
                <Sprout size={20} />
                <div>
                  <span>Gaps found</span>
                  <strong>{summary?.n_gap_segments ?? "—"}</strong>
                </div>
              </article>
              <article>
                <Ruler size={20} />
                <div>
                  <span>Total gap length</span>
                  <strong>
                    {summary ? `${summary.total_gap_length_m} m` : "—"}
                  </strong>
                </div>
              </article>
            </div>

            <a
              className={`download-button ${methodResult ? "" : "disabled"}`}
              href={methodResult ? apiUrl(methodResult.csv_path) : undefined}
              aria-disabled={!methodResult}
            >
              <Download size={18} /> Download {metricMethodLabel} CSV
            </a>
          </section>
        </aside>

        <section className="map-workspace">
          <div
            ref={mapElement}
            className="map-canvas"
            aria-label="Orthomosaic analysis map"
          />
          {compareMode && result && (
            <>
              <div className="comparison-badge comparison-badge-left">
                <span>LEFT</span>
                <strong>{METHOD_LABELS[leftMethod]}</strong>
              </div>
              <div className="comparison-badge comparison-badge-right">
                <span>RIGHT</span>
                <strong>{METHOD_LABELS[rightMethod]}</strong>
              </div>
              <div
                className={`comparison-divider ${isSplitDragging ? "dragging" : ""}`}
                style={{ left: `${splitPosition}%` }}
                role="separator"
                aria-label="Method comparison divider"
                aria-orientation="vertical"
                aria-valuemin={5}
                aria-valuemax={95}
                aria-valuenow={Math.round(splitPosition)}
                tabIndex={0}
                onPointerDown={(event) => {
                  event.currentTarget.setPointerCapture(event.pointerId);
                  setIsSplitDragging(true);
                  updateSplitPosition(event.clientX);
                }}
                onPointerMove={(event) => {
                  if (
                    event.currentTarget.hasPointerCapture(event.pointerId)
                  ) {
                    updateSplitPosition(event.clientX);
                  }
                }}
                onPointerUp={(event) => {
                  if (
                    event.currentTarget.hasPointerCapture(event.pointerId)
                  ) {
                    event.currentTarget.releasePointerCapture(event.pointerId);
                  }
                  setIsSplitDragging(false);
                }}
                onPointerCancel={() => setIsSplitDragging(false)}
                onKeyDown={(event) => {
                  if (event.key === "ArrowLeft") {
                    event.preventDefault();
                    nudgeSplitPosition(-2);
                  }
                  if (event.key === "ArrowRight") {
                    event.preventDefault();
                    nudgeSplitPosition(2);
                  }
                }}
              >
                <span>
                  <Columns2 size={18} />
                </span>
              </div>
            </>
          )}
          {!result && (
            <div className="map-empty">
              <div>
                <Layers3 size={25} />
              </div>
              <strong>
                {isProcessing ? "Processing field data" : "Awaiting orthomosaic"}
              </strong>
              <span>
                {isProcessing
                  ? statusMessage
                  : "Upload a georeferenced GeoTIFF to begin"}
              </span>
            </div>
          )}
          {!compareMode && (
            <div className="map-method">
              <span>
                <i /> {result ? "ANALYSIS ACTIVE" : "METHOD READY"}
              </span>
              <strong>{selectedMethodLabel}</strong>
            </div>
          )}
          <div className="layer-panel">
            <div>
              <Layers3 size={16} /> OVERLAYS
            </div>
            <button
              className={showRows ? "active" : ""}
              onClick={() => setShowRows((value) => !value)}
            >
              <i className="row-swatch" /> Row centerlines <Check size={15} />
            </button>
            <button
              className={showGaps ? "active" : ""}
              onClick={() => setShowGaps((value) => !value)}
            >
              <i className="gap-swatch" /> Detected gaps <Check size={15} />
            </button>
          </div>
          <div className="coordinate-readout">
            <span>{coords}</span>
            <small>WGS 84 · EPSG:4326</small>
          </div>
          <div className="map-footer-stat">
            <span>GAP RATE</span>
            <strong>{summary ? `${summary.overall_gap_pct}%` : "—"}</strong>
          </div>
        </section>
      </div>
    </main>
  );
}
