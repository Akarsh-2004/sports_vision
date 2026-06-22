#!/usr/bin/env python3
"""Generate AI Coach v2 interactive HTML dashboard."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _rel(path: str, match_dir: Path) -> str:
    if not path:
        return ""
    pp = Path(path)
    if pp.is_absolute():
        try:
            return str(pp.relative_to(match_dir)).replace("\\", "/")
        except ValueError:
            return str(pp).replace("\\", "/")
    return path.replace("\\", "/")


def generate(match_dir: Path, video_path: Path | None = None) -> Path:
    full = match_dir / "full_output.json"
    if not full.exists():
        raise FileNotFoundError(full)
    data = json.loads(full.read_text(encoding="utf-8"))
    stats = data.get("stats", data)
    intel = data.get("intelligence", {})
    viz = data.get("visualizations", {})
    coach = data.get("coach_highlights", {})
    active = data.get("active_play", {})
    tactical = stats.get("tactical", {})
    movement = stats.get("movement", {})
    scores = stats.get("scores", {})
    self_eval = intel.get("self_evaluation", {}).get("module_confidence", {})
    rally_graphs = intel.get("rally_graphs", [])
    recs = intel.get("recommendations", [])
    patterns = intel.get("pattern_mining", {})
    shots = intel.get("shot_understanding", [])
    timeline = intel.get("timeline_events", [])

    manifest = coach.get("manifest", [])
    if not manifest and (match_dir / "highlights" / "manifest.json").exists():
        m = json.loads((match_dir / "highlights" / "manifest.json").read_text(encoding="utf-8"))
        manifest = m.get("events", [])

    category_labels = coach.get("by_category", {})
    if not category_labels and (match_dir / "highlights" / "manifest.json").exists():
        m = json.loads((match_dir / "highlights" / "manifest.json").read_text(encoding="utf-8"))
        category_labels = m.get("by_category", {})

    # Copy viz into match dir
    viz_paths: dict[str, str] = {}
    for key in ("heatmap", "radar", "shot_chart", "timeline"):
        src = viz.get(key, "")
        if src and Path(src).exists():
            dst = match_dir / "viz" / Path(src).name
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            viz_paths[key] = f"viz/{dst.name}"

    video_rel = ""
    if video_path and video_path.exists():
        vdst = match_dir / video_path.name
        if not vdst.exists():
            try:
                shutil.copy2(video_path, vdst)
            except OSError:
                video_rel = str(video_path).replace("\\", "/")
        if vdst.exists():
            video_rel = video_path.name

    match_id = stats.get("match_id", match_dir.name)
    overall = scores.get("overall", 0)
    fps = stats.get("fps", 25)
    net_pct = tactical.get("net_dominance_pct", movement.get("net_zone_pct", 0) * 100)
    stroke_dist = stats.get("stroke_distribution", {})

    points_detected = len(data.get("rallies_all", []))

    payload = json.dumps(
        {
            "match_id": match_id,
            "fps": fps,
            "overall": overall,
            "points_detected": points_detected,
            "manifest": manifest,
            "category_labels": category_labels,
            "timeline": timeline[:120],
            "shots": shots,
            "rally_graphs": rally_graphs,
            "recommendations": recs,
            "patterns": patterns,
            "self_eval": self_eval,
            "tactical": tactical,
            "movement": movement,
            "stroke_dist": stroke_dist,
            "active_ratio": active.get("active_ratio", 0),
            "viz": viz_paths,
            "video": video_rel,
        }
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Padel Coach — {match_id}</title>
<style>
:root {{
  --bg: #0b0f14; --panel: #12181f; --border: #1e2a38; --text: #e8ecf0;
  --muted: #8b9aab; --accent: #00c2a8; --accent2: #3d8bfd; --warn: #f5a623;
  --danger: #ff5c5c; --sidebar-w: 220px; --right-w: 320px;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; }}
.app {{ display: grid; grid-template-columns: var(--sidebar-w) 1fr var(--right-w); grid-template-rows: 1fr 120px; height: 100vh; }}
.sidebar {{ grid-row: 1 / 3; background: var(--panel); border-right: 1px solid var(--border); padding: 1rem 0; display: flex; flex-direction: column; }}
.logo {{ padding: 0 1.25rem 1rem; font-weight: 700; font-size: 1.1rem; color: var(--accent); }}
.nav-item {{ padding: 0.75rem 1.25rem; cursor: pointer; color: var(--muted); border-left: 3px solid transparent; }}
.nav-item:hover, .nav-item.active {{ background: #1a2430; color: var(--text); border-left-color: var(--accent); }}
.main {{ display: flex; flex-direction: column; overflow: hidden; }}
.panel {{ display: none; flex: 1; overflow-y: auto; padding: 1rem 1.25rem; }}
.panel.active {{ display: block; }}
.right {{ background: var(--panel); border-left: 1px solid var(--border); padding: 1rem; overflow-y: auto; }}
.timeline-bar {{ grid-column: 2; background: #0e141c; border-top: 1px solid var(--border); padding: 0.5rem 1rem; overflow-x: auto; white-space: nowrap; }}
video {{ width: 100%; max-height: 52vh; border-radius: 10px; background: #000; }}
.overlay-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-top: 0.75rem; }}
.stat-pill {{ background: #1a2430; border-radius: 8px; padding: 0.5rem 0.75rem; font-size: 0.85rem; }}
.stat-pill strong {{ display: block; font-size: 1.1rem; color: var(--accent); }}
.card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }}
.card h3 {{ margin: 0 0 0.75rem; font-size: 0.95rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
.score-ring {{ font-size: 2.5rem; font-weight: 700; color: var(--accent); }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.hl-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 0.75rem; }}
.hl-card {{ background: #1a2430; border-radius: 10px; padding: 0.75rem; cursor: pointer; border: 1px solid transparent; transition: border 0.15s; }}
.hl-card:hover {{ border-color: var(--accent); }}
.hl-card .meta {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; }}
.tag {{ display: inline-block; background: #243040; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin: 0.1rem; }}
.cat-tabs {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 1rem; }}
.cat-tab {{ padding: 0.4rem 0.75rem; border-radius: 20px; background: #1a2430; cursor: pointer; font-size: 0.85rem; border: 1px solid var(--border); }}
.cat-tab.active {{ background: var(--accent); color: #000; border-color: var(--accent); }}
.tl-icon {{ display: inline-block; width: 28px; height: 28px; border-radius: 50%; margin: 0 4px; cursor: pointer; text-align: center; line-height: 28px; font-size: 14px; vertical-align: middle; }}
.tl-icon:hover {{ transform: scale(1.15); }}
.evidence-row {{ display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid var(--border); cursor: pointer; }}
.evidence-row:hover {{ color: var(--accent); }}
.chat-box {{ display: flex; flex-direction: column; height: calc(100vh - 180px); }}
.chat-messages {{ flex: 1; overflow-y: auto; margin-bottom: 0.75rem; }}
.chat-msg {{ margin: 0.5rem 0; padding: 0.6rem 0.8rem; border-radius: 10px; max-width: 95%; font-size: 0.9rem; line-height: 1.4; }}
.chat-msg.ai {{ background: #1a2a3a; }}
.chat-msg.user {{ background: #1e3a2f; margin-left: auto; }}
.chat-input {{ display: flex; gap: 0.5rem; }}
.chat-input input {{ flex: 1; padding: 0.6rem; border-radius: 8px; border: 1px solid var(--border); background: #0e141c; color: var(--text); }}
button {{ background: var(--accent); color: #000; border: none; padding: 0.5rem 1rem; border-radius: 8px; cursor: pointer; font-weight: 600; }}
button.secondary {{ background: #243040; color: var(--text); }}
.conf-bar {{ height: 6px; background: #243040; border-radius: 3px; margin-top: 4px; }}
.conf-fill {{ height: 100%; background: var(--accent); border-radius: 3px; }}
img.chart {{ max-width: 100%; border-radius: 8px; }}
.commentary {{ font-style: italic; color: var(--muted); font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.5; }}
.layers label {{ display: block; margin: 0.35rem 0; font-size: 0.9rem; }}
.improve {{ color: #5dffa8; }} .needs {{ color: var(--warn); }}
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="logo">🎾 AI Padel Coach</div>
    <div class="nav-item active" data-panel="coach">🧠 AI Coach</div>
    <div class="nav-item" data-panel="replay">🎥 Match Replay</div>
    <div class="nav-item" data-panel="highlights">⭐ Highlights</div>
    <div class="nav-item" data-panel="analytics">📈 Analytics</div>
    <div class="nav-item" data-panel="training">🏋 Training</div>
    <div class="nav-item" data-panel="chat">💬 Ask AI</div>
    <div style="flex:1"></div>
    <div class="nav-item" style="font-size:0.8rem">{match_id}</div>
  </aside>

  <main class="main">
    <section class="panel active" id="panel-coach">
      <div class="grid2">
        <div class="card">
          <h3>Today's Match Score</h3>
          <div class="score-ring" id="coach-score">{overall:.0f}</div>
          <div>⭐⭐⭐⭐☆ · Active play {active.get('active_ratio', 0)*100:.0f}% · Points {len(data.get('rallies_all', []))}</div>
        </div>
        <div class="card">
          <h3>Module Confidence</h3>
          <div id="confidence-bars"></div>
        </div>
      </div>
      <div class="grid2">
        <div class="card">
          <h3 class="improve">✔ Strengths</h3>
          <ul id="strengths-list"></ul>
        </div>
        <div class="card">
          <h3 class="needs">✖ Needs Work</h3>
          <ul id="weakness-list"></ul>
        </div>
      </div>
      <div class="card">
        <h3>▶ Important Clips</h3>
        <div class="hl-grid" id="coach-clips"></div>
      </div>
      <div class="card">
        <h3>Recommended Drills</h3>
        <div id="drills-list"></div>
      </div>
    </section>

    <section class="panel" id="panel-replay">
      <video id="player" controls src=""></video>
      <div class="overlay-stats" id="live-overlay"></div>
      <div class="card" style="margin-top:1rem">
        <h3>Replay Layers</h3>
        <div class="layers grid2">
          <label><input type="checkbox" checked disabled> Player tracking</label>
          <label><input type="checkbox" id="layer-ball"> Ball path (data)</label>
          <label><input type="checkbox" id="layer-zones"> Court zones</label>
          <label><input type="checkbox" id="layer-pressure"> Pressure zones</label>
        </div>
      </div>
      <div class="card">
        <h3>Current Rally Chain</h3>
        <div id="rally-chains"></div>
      </div>
    </section>

    <section class="panel" id="panel-highlights">
      <div class="cat-tabs" id="category-tabs"></div>
      <div class="hl-grid" id="highlight-gallery"></div>
      <div class="card" id="highlight-detail" style="display:none">
        <h3 id="hl-title">Highlight</h3>
        <video id="hl-player" controls style="max-height:40vh"></video>
        <p class="commentary" id="hl-commentary"></p>
        <div id="hl-overlay"></div>
      </div>
    </section>

    <section class="panel" id="panel-analytics">
      <div class="grid2">
        <div class="card stat-click" data-stat="net">
          <h3>Net Control</h3>
          <strong style="font-size:2rem">{net_pct:.0f}%</strong>
          <p class="meta">Click for evidence clips</p>
        </div>
        <div class="card stat-click" data-stat="movement">
          <h3>Distance Covered</h3>
          <strong style="font-size:2rem">{movement.get('total_distance_m', 0):.0f}m</strong>
        </div>
      </div>
      <div class="card" id="evidence-panel" style="display:none">
        <h3>Evidence</h3>
        <div id="evidence-list"></div>
      </div>
      <div class="grid2">
        <div class="card"><h3>Movement Heatmap</h3><img class="chart" id="img-heatmap" alt="heatmap"/></div>
        <div class="card"><h3>Performance Radar</h3><img class="chart" id="img-radar" alt="radar"/></div>
        <div class="card"><h3>Shot Chart</h3><img class="chart" id="img-shot" alt="shots"/></div>
        <div class="card"><h3>Timeline</h3><img class="chart" id="img-timeline" alt="timeline"/></div>
      </div>
    </section>

    <section class="panel" id="panel-training">
      <div id="training-cards"></div>
    </section>

    <section class="panel" id="panel-chat">
      <div class="chat-box">
        <div class="chat-messages" id="chat-messages">
          <div class="chat-msg ai">Ask me anything about this match. Try: "show every vibora", "show poor recovery", "longest rally", "why was this mistake bad?"</div>
        </div>
        <div class="chat-input">
          <input id="chat-input" placeholder="Ask AI about your match..." />
          <button onclick="sendChat()">Send</button>
        </div>
      </div>
    </section>
  </main>

  <aside class="right">
    <h3 style="margin-top:0;color:var(--muted);font-size:0.85rem">LIVE AI CO-PILOT</h3>
    <div class="card">
      <div id="copilot-rally">Rally: —</div>
      <div id="copilot-pressure">Pressure: —</div>
      <div id="copilot-net">Net control: —</div>
      <div id="copilot-winner">Expected winner: —</div>
    </div>
    <div class="card">
      <h3>AI Commentary</h3>
      <p class="commentary" id="copilot-commentary">Select a highlight or timeline event to hear coach analysis.</p>
    </div>
    <div class="card">
      <h3>Patterns Found</h3>
      <div id="pattern-list"></div>
    </div>
  </aside>

  <div class="timeline-bar" id="timeline-bar"></div>
</div>

<script>
const DATA = {payload};

function $(id) {{ return document.getElementById(id); }}
function esc(s) {{ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }}

function clipUrl(h) {{
  if (!h.clip_path) return null;
  const p = String(h.clip_path).replace(/^highlights[/\\\\]/, '');
  return 'highlights/' + p;
}}

function playHighlight(h, useHlPlayer=false) {{
  const v = useHlPlayer ? $('hl-player') : $('player');
  const hp = clipUrl(h);
  if (hp && useHlPlayer) {{
    $('hl-player').src = hp;
    $('hl-player').play();
  }} else if (DATA.video) {{
    v.src = DATA.video;
    v.currentTime = Math.max(0, (h.start_frame || 0) / DATA.fps);
    v.play();
  }}
  $('copilot-commentary').textContent = h.commentary || 'Replay this moment for tactical review.';
  const ov = h.overlay || {{}};
  $('live-overlay').innerHTML = `
    <div class="stat-pill"><span>Ball Speed</span><strong>${{ov.ball_speed_kmh || '—'}} km/h</strong></div>
    <div class="stat-pill"><span>Pressure</span><strong>${{ov.pressure || '—'}}</strong></div>
    <div class="stat-pill"><span>Win Prob</span><strong>${{ov.win_probability ? Math.round(ov.win_probability*100)+'%' : '—'}}</strong></div>
    <div class="stat-pill"><span>Decision</span><strong>${{ov.decision || h.stroke || '—'}}</strong></div>`;
  $('copilot-rally').textContent = 'Rally: ' + (h.rally_length ? h.rally_length + ' shots' : h.level || 'moment');
  $('copilot-pressure').textContent = 'Pressure: ' + (ov.pressure || 'medium');
  $('copilot-winner').textContent = 'Excitement: ' + (h.excitement || '—') + '/100';
  document.querySelector('[data-panel="replay"]')?.click?.();
  switchPanel('replay');
}}

function switchPanel(name) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  $('panel-' + name)?.classList.add('active');
  document.querySelector(`[data-panel="${{name}}"]`)?.classList.add('active');
}}

document.querySelectorAll('.nav-item[data-panel]').forEach(el => {{
  el.addEventListener('click', () => switchPanel(el.dataset.panel));
}});

function renderCoach() {{
  const top = [...DATA.manifest].sort((a,b) => b.excitement - a.excitement).slice(0, 8);
  $('coach-clips').innerHTML = top.map(h => `
    <div class="hl-card" onclick='playHighlight(${{JSON.stringify(h)}})'>
      <strong>${{esc(h.primary_category?.replace(/_/g,' ') || 'Moment')}}</strong>
      <div class="meta">${{h.start_time}} · ${{h.excitement}}/100 · ${{h.level}}</div>
      <div>${{(h.tags||[]).map(t=>`<span class="tag">${{esc(t)}}</span>`).join('')}}</div>
    </div>`).join('');

  const conf = DATA.self_eval || {{}};
  $('confidence-bars').innerHTML = Object.entries(conf).map(([k,v]) => `
    <div style="margin:0.4rem 0;font-size:0.85rem">${{k}}: ${{(v*100).toFixed(0)}}%
      <div class="conf-bar"><div class="conf-fill" style="width:${{v*100}}%"></div></div>
    </div>`).join('');

  const strengths = [];
  if ((conf.overall||0) > 0.7) strengths.push('Reliable tracking foundation');
  if (DATA.tactical?.wall_usage_pct > 20) strengths.push('Active wall play');
  if ((DATA.movement?.net_zone_pct||0) > 0.15) strengths.push('Net presence');
  $('strengths-list').innerHTML = (strengths.length ? strengths : ['Shot variety detected']).map(s=>`<li>${{s}}</li>`).join('');

  const weak = [];
  if ((DATA.tactical?.net_dominance_pct||0) < 30) weak.push('Net control — approach more aggressively');
  if (DATA.recommendations?.length) weak.push(DATA.recommendations[0].advice);
  $('weakness-list').innerHTML = weak.map(s=>`<li>${{esc(s)}}</li>`).join('') || '<li>Review defensive recoveries</li>';

  $('drills-list').innerHTML = (DATA.recommendations||[]).slice(0,3).map(r => `
    <div style="margin:0.5rem 0;padding:0.5rem;background:#1a2430;border-radius:8px">
      <strong>${{esc(r.advice)}}</strong><br><span class="meta">${{esc(r.evidence)}}</span>
    </div>`).join('') || '<p>Recovery footwork · Wall defense · Transition movement</p>';
}}

const CAT_EMOJI = {{
  best_rallies:'🏆', best_smashes:'💥', best_defense:'🛡', best_viboras:'🎯',
  best_volleys:'🎾', longest_rally:'🔥', fastest_point:'⚡', smartest_point:'🧠',
  biggest_mistake:'😬', wall_play:'🧱', net_battle:'⚔', coaching_moments:'📋', top_moments:'⭐'
}};

function renderHighlights(cat='all') {{
  const tabs = ['all', ...Object.keys(CAT_EMOJI)];
  $('category-tabs').innerHTML = tabs.map(c => `
    <div class="cat-tab ${{c===cat?'active':''}}" onclick="renderHighlights('${{c}}')">${{c==='all'?'All':(CAT_EMOJI[c]||'')+' '+c.replace(/_/g,' ')}}</div>`).join('');

  let items = DATA.manifest;
  if (cat !== 'all') items = items.filter(h => (h.categories||[]).includes(cat));
  items = [...items].sort((a,b) => b.excitement - a.excitement);

  $('highlight-gallery').innerHTML = items.map(h => `
    <div class="hl-card" onclick="showHighlight(${{h.event_id}})">
      <strong>${{esc((h.stroke||h.primary_category||'moment').replace(/_/g,' '))}}</strong>
      <div class="meta">${{h.start_time}}–${{h.end_time}} · L${{h.level}} · ${{h.excitement}}/100</div>
      <div>${{(h.categories||[]).slice(0,3).map(c=>`<span class="tag">${{esc(c.replace(/_/g,' '))}}</span>`).join('')}}</div>
      <p class="commentary">${{esc((h.commentary||'').slice(0,120))}}...</p>
    </div>`).join('') || '<p>No highlights in this category yet.</p>';
}}

function showHighlight(eid) {{
  const h = DATA.manifest.find(x => x.event_id === eid);
  if (!h) return;
  $('highlight-detail').style.display = 'block';
  $('hl-title').textContent = (h.stroke || h.primary_category || 'Highlight').replace(/_/g,' ') + ' · ' + h.start_time;
  $('hl-commentary').textContent = h.commentary || '';
  $('hl-overlay').innerHTML = Object.entries(h.overlay||{{}}).map(([k,v]) => `<span class="tag">${{k}}: ${{v}}</span>`).join(' ');
  playHighlight(h, true);
}}

function renderTimeline() {{
  const icons = {{ serve:'🟢', winner:'🔴', smash:'🔵', error:'🟣', wall:'🧱', volley:'🎾', lob:'🟡', player_hit:'⚪' }};
  const bar = $('timeline-bar');
  const events = DATA.timeline.length ? DATA.timeline : DATA.manifest.map(m => ({{
    time_s: m.time_s, type: m.stroke || m.primary_category, frame: m.start_frame, stroke: m.stroke
  }}));
  const dur = Math.max(...events.map(e => e.time_s||0), 60);
  bar.innerHTML = '<span style="color:var(--muted);margin-right:8px">Timeline</span>' +
    events.slice(0,60).map(e => {{
      const icon = icons[e.type] || icons[e.stroke] || '●';
      const color = e.type?.includes('error') ? '#ff5c5c' : '#3d8bfd';
      return `<span class="tl-icon" style="background:${{color}}22;border:1px solid ${{color}}" title="${{e.type||''}} ${{e.stroke||''}} @ ${{e.time_s}}s" onclick="seekFrame(${{e.frame||0}}, '${{esc(e.type||'')}}')">${{icon}}</span>`;
    }}).join('');
}}

function seekFrame(frame, label) {{
  const v = $('player');
  if (DATA.video) {{ v.src = DATA.video; v.currentTime = Math.max(0, frame/DATA.fps - 2); v.play(); }}
  $('copilot-commentary').textContent = 'Timeline: ' + label + ' at frame ' + frame;
  switchPanel('replay');
}}

function renderAnalytics() {{
  ['heatmap','radar','shot_chart','timeline'].forEach(k => {{
    const img = $('img-' + k.replace('_chart','').replace('shot','shot'));
    const key = k === 'shot' ? 'shot_chart' : k;
    if (DATA.viz[key]) $( 'img-' + (key==='shot_chart'?'shot':key==='heatmap'?'heatmap':key) ).src = DATA.viz[key];
  }});
  if (DATA.viz.shot_chart) $('img-shot').src = DATA.viz.shot_chart;
  if (DATA.viz.heatmap) $('img-heatmap').src = DATA.viz.heatmap;
  if (DATA.viz.radar) $('img-radar').src = DATA.viz.radar;
  if (DATA.viz.timeline) $('img-timeline').src = DATA.viz.timeline;
}}

document.querySelectorAll('.stat-click').forEach(el => {{
  el.addEventListener('click', () => {{
    const stat = el.dataset.stat;
    $('evidence-panel').style.display = 'block';
    let clips = [];
    if (stat === 'net') clips = DATA.shots.filter(s => (s.region||'').includes('net') || (s.region||'').includes('attack'));
    else clips = DATA.shots.slice(0, 10);
    $('evidence-list').innerHTML = clips.map(s => `
      <div class="evidence-row" onclick="seekFrame(${{s.frame}}, '${{s.stroke}}')">
        <span>${{s.stroke}} @ ${{(s.frame/DATA.fps).toFixed(1)}}s</span>
        <span>${{s.intent}} · ${{s.pressure}}</span>
      </div>`).join('') || '<p>No evidence clips</p>';
  }});
}});

function renderTraining() {{
  const skills = [
    {{ name:'Movement', stars:4, drill:'Lateral split-step drill', clips: DATA.shots.filter(s=>s.intent==='defensive').slice(0,2) }},
    {{ name:'Net Control', stars:2, drill:'Approach + volley pattern', clips: DATA.shots.filter(s=>(s.region||'').includes('net')).slice(0,2) }},
    {{ name:'Wall Play', stars:3, drill:'Glass recovery repetition', clips: DATA.shots.filter(s=>(s.region||'').includes('glass')).slice(0,2) }},
  ];
  $('training-cards').innerHTML = skills.map(sk => `
    <div class="card">
      <h3>${{sk.name}} ${{'★'.repeat(sk.stars)}}${{'☆'.repeat(5-sk.stars)}}</h3>
      <p><strong>Drill:</strong> ${{sk.drill}}</p>
      <p><strong>Match examples:</strong> ${{sk.clips.map(c=>`<button class="secondary" onclick="seekFrame(${{c.frame}},'${{c.stroke}}')">${{c.stroke}}</button>`).join(' ') || '—'}}</p>
    </div>`).join('');
}}

function renderRallyChains() {{
  $('rally-chains').innerHTML = (DATA.rally_graphs||[]).map(r => `
    <div style="padding:0.5rem;margin:0.25rem 0;background:#1a2430;border-radius:8px;font-family:monospace;font-size:0.85rem">
      <strong>Rally ${{r.rally_id}}</strong>: ${{esc(r.chain||'empty')}}
    </div>`).join('') || '<p>No rally chains</p>';
}}

function renderPatterns() {{
  const insights = DATA.patterns?.insights || [];
  $('pattern-list').innerHTML = insights.slice(0,5).map(i => `<div class="tag" style="display:block;margin:0.3rem 0">${{esc(i)}}</div>`).join('')
    || '<span class="meta">Patterns will appear after more match data.</span>';
}}

function sendChat() {{
  const input = $('chat-input');
  const q = (input.value || '').trim().toLowerCase();
  if (!q) return;
  const msgs = $('chat-messages');
  msgs.innerHTML += `<div class="chat-msg user">${{esc(input.value)}}</div>`;
  input.value = '';
  const reply = answerQuery(q);
  msgs.innerHTML += `<div class="chat-msg ai">${{reply.html}}</div>`;
  msgs.scrollTop = msgs.scrollHeight;
  if (reply.frame) seekFrame(reply.frame, reply.label || '');
}}

function answerQuery(q) {{
  let hits = [];
  if (q.includes('vibora')) hits = DATA.manifest.filter(h => (h.stroke||'').includes('vibora') || (h.chain||'').includes('vibora'));
  else if (q.includes('smash')) hits = DATA.manifest.filter(h => (h.stroke||'').includes('smash') || (h.categories||[]).includes('best_smashes'));
  else if (q.includes('volley')) hits = DATA.manifest.filter(h => (h.stroke||'').includes('volley'));
  else if (q.includes('wall')) hits = DATA.manifest.filter(h => (h.categories||[]).includes('wall_play'));
  else if (q.includes('mistake') || q.includes('poor') || q.includes('error')) hits = DATA.manifest.filter(h => (h.categories||[]).includes('biggest_mistake'));
  else if (q.includes('longest') || q.includes('long rally')) hits = DATA.manifest.filter(h => (h.categories||[]).includes('longest_rally') || h.rally_length >= 8);
  else if (q.includes('defense')) hits = DATA.manifest.filter(h => (h.categories||[]).includes('best_defense'));
  else if (q.includes('why')) {{
    const mistake = DATA.manifest.find(h => (h.categories||[]).includes('biggest_mistake'));
    if (mistake) return {{ html: mistake.commentary + '<br><br><button onclick="playHighlight('+JSON.stringify(mistake).replace(/"/g,'&quot;')+')">▶ Watch clip</button>', frame: mistake.start_frame, label: 'mistake' }};
  }}
  else hits = DATA.manifest.slice(0, 5);

  if (!hits.length) return {{ html: 'No matching clips found. Try: vibora, smash, wall play, mistakes, longest rally.' }};
  const list = hits.slice(0,8).map(h => `<button class="secondary" style="margin:0.2rem" onclick='playHighlight(${{JSON.stringify(h)}})'>${{h.start_time}} ${{esc(h.stroke||h.primary_category||'')}}</button>`).join('');
  return {{ html: `Found <strong>${{hits.length}}</strong> clips:<br>${{list}}`, frame: hits[0].start_frame, label: hits[0].stroke }};
}}

$('chat-input')?.addEventListener('keydown', e => {{ if (e.key === 'Enter') sendChat(); }});

// Init
if (DATA.video) $('player').src = DATA.video;
renderCoach();
renderHighlights();
renderTimeline();
renderAnalytics();
renderTraining();
renderRallyChains();
renderPatterns();
</script>
</body>
</html>"""

    out = match_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--match-dir", required=True)
    p.add_argument("--video")
    args = p.parse_args()
    path = generate(Path(args.match_dir), Path(args.video) if args.video else None)
    print(f"Dashboard: {path}")
