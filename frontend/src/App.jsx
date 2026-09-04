import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL;
const MAX_FILE_SIZE_MB = 10;

// Defect catalogue — the six categories the trained model recognizes.
const DEFECT_CATALOGUE = [
  {
    key: "crazing",
    label: "Crazing",
    description:
      "A fine network of surface cracks caused by thermal or mechanical stress during processing.",
  },
  {
    key: "inclusion",
    label: "Inclusion",
    description:
      "Foreign material such as slag or oxide trapped within the steel surface.",
  },
  {
    key: "patches",
    label: "Patches",
    description:
      "Localized irregular regions where the surface finish differs from the surrounding area.",
  },
  {
    key: "pitted_surface",
    label: "Pitted Surface",
    description:
      "Small cavities or pits formed on the surface, typically from corrosion or rolling defects.",
  },
  {
    key: "rolled_in_scale",
    label: "Rolled-in Scale",
    description:
      "Oxide scale that has been pressed into the steel surface during hot rolling.",
  },
  {
    key: "scratches",
    label: "Scratches",
    description:
      "Linear marks caused by friction or mechanical abrasion during handling or processing.",
  },
];

const DESCRIPTIONS = Object.fromEntries(
  DEFECT_CATALOGUE.map((d) => [d.key, d.description])
);

const LABELS = Object.fromEntries(
  DEFECT_CATALOGUE.map((d) => [d.key, d.label])
);

function statusForConfidence(confidence) {
  if (confidence >= 90) return { label: "High confidence", tone: "ok" };
  if (confidence >= 70) return { label: "Moderate confidence", tone: "warn" };
  return { label: "Low confidence — review recommended", tone: "alert" };
}

function Gauge({ value }) {
  // Semi-circular instrument-style gauge, 0–100 mapped across 180°.
  const clamped = Math.max(0, Math.min(100, value || 0));
  const angle = (clamped / 100) * 180;
  const needleAngle = angle - 90; // -90..90 relative to vertical
  const status = statusForConfidence(clamped);

  return (
    <div className="gauge">
      <svg viewBox="0 0 200 110" className="gauge-svg">
        <path
          d="M 10 100 A 90 90 0 0 1 190 100"
          className="gauge-track"
          fill="none"
        />
        <path
          d="M 10 100 A 90 90 0 0 1 190 100"
          className={`gauge-fill gauge-fill-${status.tone}`}
          fill="none"
          strokeDasharray={`${(angle / 180) * 282.6} 282.6`}
        />
        <g
          className="gauge-needle"
          style={{ transform: `rotate(${needleAngle}deg)` }}
        >
          <line x1="100" y1="100" x2="100" y2="28" />
        </g>
        <circle cx="100" cy="100" r="6" className="gauge-hub" />
      </svg>
      <div className="gauge-readout">
        <span className="gauge-value">{clamped.toFixed(1)}%</span>
        <span className={`gauge-status gauge-status-${status.tone}`}>
          {status.label}
        </span>
      </div>
    </div>
  );
}

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [serviceStatus, setServiceStatus] = useState("checking");
  const fileInputRef = useRef(null);
  const previewRef = useRef(null);

  // Ping the backend once on load so the status indicator reflects reality
  // instead of always claiming to be online.
  useEffect(() => {
    let cancelled = false;
    axios
      .get(`${API_URL}/health`, { timeout: 5000 })
      .then(() => {
        if (!cancelled) setServiceStatus("online");
      })
      .catch(() => {
        if (!cancelled) setServiceStatus("offline");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Revoke the previous preview's object URL whenever it's replaced or the
  // component unmounts, so we don't leak blob URLs.
  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

  const loadFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPG, PNG, or similar).");
      return;
    }
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`Image is too large. Please upload a file under ${MAX_FILE_SIZE_MB}MB.`);
      return;
    }
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    const nextPreview = URL.createObjectURL(file);
    previewRef.current = nextPreview;
    setError("");
    setSelectedImage(file);
    setPreview(nextPreview);
    setPrediction("");
    setConfidence(null);
  };

  const handleImageChange = (e) => loadFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    loadFile(e.dataTransfer.files[0]);
  };

  const handlePredict = async () => {
    if (!selectedImage) {
      setError("Upload a steel surface image before running analysis.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      setLoading(true);
      setError("");
      const response = await axios.post(`${API_URL}/predict`, formData);
      setPrediction(response.data.prediction);
      setConfidence(response.data.confidence);
    } catch (err) {
      console.error(err);
      const backendMessage = err.response?.data?.detail;
      setError(
        typeof backendMessage === "string"
          ? backendMessage
          : "Analysis could not be completed. Check that the inspection service is running and try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = null;
    setSelectedImage(null);
    setPreview(null);
    setPrediction("");
    setConfidence(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="page">
      <header className="masthead">
        <div className="masthead-row">
          <div className="brand">
            <svg
              className="brand-mark"
              viewBox="0 0 44 44"
              aria-hidden="true"
            >
              <rect x="2" y="2" width="40" height="40" rx="6" />
              <path d="M10 16 H34 M22 16 V34" className="brand-mark-cut" />
            </svg>
            <div className="brand-text">
              <span className="brand-name">TATA STEEL</span>
              <span className="brand-division">Corporate Functions</span>
            </div>
          </div>
          <div className="masthead-status">
            <span className={`status-dot status-dot-${serviceStatus}`} />
            {serviceStatus === "online" && "Inspection service online"}
            {serviceStatus === "offline" && "Inspection service unreachable"}
            {serviceStatus === "checking" && "Checking inspection service…"}
          </div>
        </div>
        <div className="masthead-rule" />
        <div className="masthead-title">
          <h1>Defect Analysis System</h1>
          <p>
            Upload an image of a steel surface and the system will identify
            the defect category automatically.
          </p>
        </div>
      </header>

      <main className="content">
        <section className="panel upload-panel">
          <h2 className="panel-title">1. Upload surface image</h2>

          <label
            className={`dropzone ${isDragging ? "dropzone-active" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            {preview ? (
              <img src={preview} alt="Steel surface preview" className="preview-image" />
            ) : (
              <div className="dropzone-empty">
                <svg viewBox="0 0 24 24" className="dropzone-icon" aria-hidden="true">
                  <path d="M12 16V4M12 4L7 9M12 4l5 5" />
                  <path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
                </svg>
                <span className="dropzone-label">
                  Drag and drop an image, or click to browse
                </span>
                <span className="dropzone-hint">JPG or PNG · single surface image</span>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              hidden
            />
          </label>

          {error && <p className="error-text">{error}</p>}

          <div className="action-row">
            <button
              className="btn-primary"
              onClick={handlePredict}
              disabled={loading || !selectedImage}
            >
              {loading ? "Analyzing…" : "Analyze Image"}
            </button>
            {(selectedImage || prediction) && (
              <button className="btn-secondary" onClick={reset} disabled={loading}>
                Clear
              </button>
            )}
          </div>

          <div className="catalogue">
            <h3 className="catalogue-title">Defect catalogue</h3>
            <div className="catalogue-chips">
              {DEFECT_CATALOGUE.map((d) => (
                <span
                  key={d.key}
                  className={`chip ${prediction === d.key ? "chip-active" : ""}`}
                >
                  {d.label}
                </span>
              ))}
            </div>
          </div>
        </section>

        <section className="panel result-panel">
          <h2 className="panel-title">2. Inspection result</h2>

          {!prediction && !loading && (
            <div className="result-empty">
              <p>Results will appear here once an image has been analyzed.</p>
            </div>
          )}

          {loading && (
            <div className="result-loading">
              <div className="spinner" />
              <p>Running defect classification…</p>
            </div>
          )}

          {prediction && !loading && (
            <div className="result-body">
              <Gauge value={confidence} />
              <div className="result-defect">
                <span className="result-defect-label">Defect identified</span>
                <span className="result-defect-name">{LABELS[prediction] || prediction}</span>
              </div>
              <p className="result-description">{DESCRIPTIONS[prediction]}</p>
            </div>
          )}
        </section>
      </main>

      <footer className="site-footer">
        <span className="footer-brand">TATA STEEL</span>
        <span className="footer-tagline">We Also Make Tomorrow</span>
        <span className="footer-division">Corporate Functions · Defect Analysis System</span>
      </footer>
    </div>
  );
}

export default App;
