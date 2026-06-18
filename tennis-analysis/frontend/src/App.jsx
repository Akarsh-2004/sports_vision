import { useCallback, useEffect, useState } from "react";
import RadarChart from "./components/RadarChart";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);

  const pollJob = useCallback(async (id) => {
    const res = await fetch(`${API}/jobs/${id}`);
    const data = await res.json();
    setStatus(data);
    if (data.status === "completed" && data.match_id) {
      const r = await fetch(`${API}/results/${data.match_id}`);
      setResults(await r.json());
    }
    return data;
  }, []);

  useEffect(() => {
    if (!jobId || status?.status === "completed" || status?.status === "failed") return;
    const t = setInterval(async () => {
      const d = await pollJob(jobId);
      if (d.status === "completed" || d.status === "failed") clearInterval(t);
    }, 2000);
    return () => clearInterval(t);
  }, [jobId, status, pollJob]);

  const handleUpload = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("max_frames", "500");
    const res = await fetch(`${API}/analyze/upload`, { method: "POST", body: form });
    const data = await res.json();
    setJobId(data.job_id);
    setStatus(data);
    setResults(null);
  };

  const scores = results?.scores;

  return (
    <div className="app">
      <header>
        <h1>Tennis Match Analysis</h1>
        <p>Upload match footage for AI-powered performance analysis (17-stage pipeline)</p>
      </header>

      <div className="card">
        <div
          className="upload-zone"
          onClick={() => document.getElementById("file-input").click()}
        >
          {file ? file.name : "Click to select match video (.mp4)"}
        </div>
        <input
          id="file-input"
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button onClick={handleUpload} disabled={!file || status?.status === "running"}>
          Analyze Match
        </button>
        {status && (
          <>
            <p style={{ marginTop: "1rem", color: "#8b98a5" }}>
              Status: {status.status} — {status.stage}
              {status.error && ` (${status.error})`}
            </p>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${(status.progress || 0) * 100}%` }} />
            </div>
          </>
        )}
      </div>

      {scores && (
        <div className="card">
          <h2>Performance Scores</h2>
          <div className="score-grid">
            {[
              ["Overall", scores.overall],
              ["Serve", scores.serve],
              ["Return", scores.return],
              ["Movement", scores.movement],
              ["Consistency", scores.consistency],
              ["Aggression", scores.aggression],
              ["Stamina", scores.stamina],
              ["Coverage", scores.court_coverage],
            ].map(([lbl, val]) => (
              <div key={lbl} className="score-card">
                <div className="val">{Math.round(val)}</div>
                <div className="lbl">{lbl}</div>
              </div>
            ))}
          </div>
          <RadarChart scores={scores} />
        </div>
      )}

      {results?.movement && (
        <div className="card">
          <h2>Movement</h2>
          <p>Distance: {results.movement.total_distance_m.toFixed(0)} m</p>
          <p>Max speed: {results.movement.max_speed_kmh.toFixed(1)} km/h</p>
          <p>Sprints: {results.movement.sprint_count}</p>
        </div>
      )}

      {results?.summary && (
        <div className="card">
          <h2>Report</h2>
          <div className="report">{results.summary}</div>
        </div>
      )}
    </div>
  );
}
