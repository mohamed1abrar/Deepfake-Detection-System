import { useState, useEffect } from "react";
import "./App.css";

export default function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setTimeout(() => setVisible(true), 300);
  }, []);

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setLoading(true);
    setResult(null);

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: fd
      });
      const data = await res.json();
      setResult(data);
    } catch {
      alert("Backend not reachable");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={`app ${visible ? "fade-in" : ""}`}>

      {/* BACKGROUND LIGHT */}
      <div className="ambient-glow"></div>

      {/* HERO */}
      <section className="hero">
        <h1 className="glow-text">Deepfake Detection System</h1>
        <p className="hero-desc">
          A premium AI-powered media forensics system designed to analyze images
          and videos for signs of manipulation using deep learning and computer
          vision.
        </p>
      </section>

      {/* INFO STRIP */}
      <section className="info-strip">
        <div className="info-box">
          <h3>Computer Vision</h3>
          <p>
            Facial artifact detection using spatial feature analysis and
            convolutional neural networks.
          </p>
        </div>

        <div className="info-box">
          <h3>Deep Learning</h3>
          <p>
            Ensemble-based inference improves robustness and reduces false
            positives.
          </p>
        </div>

        <div className="info-box">
          <h3>Forensic Analysis</h3>
          <p>
            Frame-level inspection enables temporal manipulation tracking in
            videos.
          </p>
        </div>
      </section>

      {/* UPLOAD */}
      <section className="upload-section">
        <label className="upload-card">
          <input type="file" onChange={handleUpload} />
          <span>Upload Image or Video</span>
        </label>
      </section>

      {/* PREVIEW */}
      {preview && (
        <section className="preview animated-card">
          {result?.is_image ? (
            <img src={preview} alt="preview" />
          ) : (
            <video src={preview} controls />
          )}
        </section>
      )}

      {loading && <div className="loader shimmer"></div>}

      {/* RESULTS */}
      {result && (
        <>
          <section className="results">
            <div className="result-main animated-card">
              <h2>{result.ensemble.label}</h2>
              <p>
                Estimated manipulation probability:{" "}
                <strong>{Math.round(result.ensemble.avg_fake * 100)}%</strong>
              </p>
              <ConfidenceBar value={result.ensemble.avg_fake} />
            </div>
          </section>

          <section className="model-section">
            <h2 className="section-title">Model Confidence Breakdown</h2>

            <div className="model-grid">
              {result.per_model.map((m, i) => (
                <div key={i} className="model-card animated-card">
                  <h4>{m.name}</h4>
                  <p>{Math.round(m.fake_prob * 100)}%</p>
                  <ConfidenceBar value={m.fake_prob} />
                </div>
              ))}
            </div>
          </section>

          {!result.is_image && result.per_frame && (
            <section className="frames-section">
              <h2 className="section-title">Frame-Level Analysis</h2>

              <div className="table-wrap animated-card">
                <table>
                  <thead>
                    <tr>
                      <th>Frame</th>
                      <th>Fake Probability</th>
                      <th>Label</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.per_frame.slice(0, 150).map((f, i) => (
                      <tr key={i}>
                        <td>{f.frame}</td>
                        <td>{(f.avg * 100).toFixed(2)}%</td>
                        <td className={f.label}>{f.label}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}

      {/* FOOTER */}
      <footer className="footer">
        <p>AI-Powered Media Forensics Platform</p>
        <p>Deep Learning • Computer Vision • Trust</p>
      </footer>
    </div>
  );
}

function ConfidenceBar({ value }) {
  return (
    <div className="bar">
      <div
        className="bar-fill shimmer"
        style={{ width: `${value * 100}%` }}
      />
    </div>
  );
}
