import { useCallback, useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL || "/api";

function formatTime(frame, fps) {
  if (!fps) return "—";
  const s = frame / fps;
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, "0");
  return `${m}:${sec}`;
}

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
    const res = await fetch(`${API}/analyze/upload`, { method: "POST", body: form });
    const data = await res.json();
    setJobId(data.job_id);
    setStatus(data);
    setResults(null);
  };

  const stats = results?.stats;
  const scores = stats?.scores;
  const pointScore = results?.score;
  const fps = stats?.fps || 25;
  const matchId = stats?.match_id || status?.match_id;

  return (
    <div className="app">
      <header>
        <h1>Padel Analysis</h1>
        <p>Upload match footage for AI coaching insights and point tracking</p>
      </header>

      <div className="card">
        <div className="upload-zone" onClick={() => document.getElementById("file-input").click()}>
          {file ? file.name : "Select match video"}
        </div>
        <input
          id="file-input"
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        <button onClick={handleUpload} disabled={!file || status?.status === "running"}>
          Analyze
        </button>
        {status && (
          <div className="status">
            <span>{status.status}</span>
            <span className="muted">{status.stage}</span>
            {status.error && <span className="error">{status.error}</span>}
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${(status.progress || 0) * 100}%` }} />
            </div>
          </div>
        )}
      </div>

      {pointScore?.score && (
        <div className="card scoreboard">
          <h2>Match Score</h2>
          <div className="teams">
            <div className="team">
              <span className="label">Team A</span>
              <span className="pts">{pointScore.score.A ?? 0}</span>
            </div>
            <span className="divider">—</span>
            <div className="team">
              <span className="label">Team B</span>
              <span className="pts">{pointScore.score.B ?? 0}</span>
            </div>
          </div>
          <p className="muted meta">
            {pointScore.points_complete ?? 0} points · {pointScore.points_interrupted ?? 0} interrupted
          </p>
        </div>
      )}

      {scores && (
        <div className="card">
          <h2>Performance</h2>
          <div className="score-grid">
            {[
              ["Overall", scores.overall],
              ["Net play", scores.net_play],
              ["Wall defense", scores.wall_defense],
              ["Positioning", scores.positioning],
              ["Shot quality", scores.shot_quality],
              ["Stamina", scores.stamina],
            ].map(([lbl, val]) => (
              <div key={lbl} className="score-card">
                <div className="val">{Math.round(val ?? 0)}</div>
                <div className="lbl">{lbl}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {stats?.movement && (
        <div className="card">
          <h2>Movement</h2>
          <div className="stats-row">
            <span>{stats.movement.total_distance_m?.toFixed(0)} m covered</span>
            <span>{stats.movement.max_speed_kmh?.toFixed(1)} km/h max</span>
            <span>{stats.movement.sprint_count} sprints</span>
          </div>
        </div>
      )}

      {pointScore?.points?.length > 0 && (
        <div className="card">
          <h2>Points</h2>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Winner</th>
                <th>Shots</th>
                <th>Duration</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {pointScore.points.map((p, i) => (
                <tr key={p.rally_id ?? i}>
                  <td>{i + 1}</td>
                  <td>{p.winner_side || "—"}</td>
                  <td>{p.shot_count}</td>
                  <td>{p.duration_s}s</td>
                  <td className={p.end_reason === "point_complete" ? "ok" : "muted"}>
                    {p.end_reason?.replace("_", " ")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {results?.rallies_all?.length > 0 && !pointScore?.points?.length && (
        <div className="card">
          <h2>Rallies</h2>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Shots</th>
                <th>Wall hits</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {results.rallies_all.map((r, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>{r.rally_length_shots}</td>
                  <td>{r.wall_hits}</td>
                  <td>{formatTime(r.start_frame, fps)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {(results?.summary_md || stats?.summary) && (
        <div className="card">
          <h2>Coach Report</h2>
          <div className="report">{results?.summary_md || stats?.summary}</div>
        </div>
      )}

      {matchId && status?.status === "completed" && (
        <div className="card links">
          <a href={`${API}/results/${matchId}/dashboard`} target="_blank" rel="noreferrer">
            Open full dashboard
          </a>
        </div>
      )}
    </div>
  );
}
