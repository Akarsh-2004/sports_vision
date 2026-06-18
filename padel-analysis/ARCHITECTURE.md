# Padel Intelligence Engine — Architecture

> **Computer vision is the sensor layer. The product is the reasoning layer.**

This document describes the target architecture for `padel-analysis/`, reorganized from a CV module pipeline into a **Padel Intelligence Engine** comparable in intent to Playtomic Vision, SwingVision, or coach-first analytics platforms.

---

## Design Principle

| Wrong question | Right question |
|----------------|----------------|
| "Did we detect a forehand?" | "Should the player have hit forehand cross-court from that position?" |
| "How many pixels did the player move?" | "Was the player too deep? Who controlled the net?" |
| "What is the ball bounding box?" | "Player → glass → ground → opponent → lob → smash → winner" |

---

## Seven Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 7 — Report Engine     coach / player / training      │
├─────────────────────────────────────────────────────────────┤
│  Layer 6 — Knowledge Engine  structured facts → LLM         │
├─────────────────────────────────────────────────────────────┤
│  Layer 5 — Tactical Engine   positioning, decisions, shape  │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 — Interaction Engine event graph (not stroke list) │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 — Match State Engine FSM: RALLY, NET_ATTACK, ...   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 — Geometry Layer    court coordinates for ALL      │
├─────────────────────────────────────────────────────────────┤
│  Layer 1 — Vision Layer      YOLO, tracking, pose (sensors) │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   DIGITAL TWIN    │  ← source of truth
                    └───────────────────┘
```

---

## Layer 1 — Vision (Sensors)

**Path:** `backend/detection/`, `backend/tracking/`, `backend/analytics/pose/`

Answers: **"What happened in pixels?"**

| Module | Role |
|--------|------|
| Court calibration | Homography |
| Player detection/tracking | YOLO → BoT-SORT (+ future: appearance + pose embeddings) |
| Ball pipeline | YOLO → optical flow → Kalman → trajectory fit → physics bounce |
| Pose | MediaPipe / RTMPose |
| Stroke classifier | Temporal model (future: VideoMAE) |

Vision output is **never** the final product — it feeds Layer 2.

---

## Layer 2 — Geometry ⭐

**Path:** `backend/intelligence/geometry/`

Answers: **"Where is everything in court meters?"**

Every frame becomes structured entities:

```
Frame 385
  Player A: (3.21, 6.71)  speed 4.2 m/s  zone=transition
  Player B: (8.12, 5.84)  ...
  Ball:     (4.82, 9.32)  speed 24 km/h
```

| Entity | Fields |
|--------|--------|
| Player | ID, position, speed, direction, acceleration, zone, side role |
| Ball | position, velocity, acceleration, height, bounce type |
| Court | walls, net, service boxes, attack/defense zones |

**Nothing downstream depends on camera angle.**

---

## Layer 3 — Match State Engine ⭐

**Path:** `backend/intelligence/match_state/`

Answers: **"What phase of the match is this?"**

Finite-state machine per frame:

`SERVE` → `RETURN` → `RALLY` → `LOB_DEFENSE` → `NET_ATTACK` → `WALL_EXCHANGE` → `RESET` → `POINT_OVER` → `DEAD_TIME`

Downstream analytics filter by state instead of raw frames.

---

## Layer 4 — Interaction Engine ⭐

**Path:** `backend/intelligence/interaction/`

Answers: **"What is the sequence of play?"**

Not: `forehand, forehand, volley`

But: `Forehand → glass → ground → opponent → lob → smash → winner`

| Node types | `PLAYER_HIT`, `BALL_WALL_GLASS`, `BALL_GROUND`, `BALL_NET`, `POINT_END` |
| Shot intent | `AGGRESSIVE`, `DEFENSIVE`, `SETUP`, `FINISHING`, `RECOVERY` |

Stored as **RallyGraph** — queryable interaction chains.

---

## Layer 5 — Tactical Engine ⭐

**Path:** `backend/intelligence/tactical/`

Answers: **"What would a coach say?"**

| Coach metric | Example |
|--------------|---------|
| Positioning | Too deep? Wrong lane? |
| Court control | Net dominance duration |
| Pressure | Forced defensive shots |
| Team shape | Partner spacing, rotation, formation |
| Decision quality | "Smash from glass zone = poor choice" |

Rules encoded in `tactical/rules.py` from padel domain knowledge.

---

## Layer 6 — Knowledge Engine ⭐

**Path:** `backend/intelligence/knowledge/`

Answers: **"What facts should the LLM reason over?"**

```json
{
  "shot": "bandeja",
  "speed_kmh": 92,
  "player_zone": "net",
  "opponent_positions": [...],
  "intent": "finishing",
  "outcome": "winner"
}
```

Thousands of structured facts — not raw video — fed to LLM (`KnowledgeReasoner`).

---

## Layer 7 — Report Engine

**Path:** `backend/intelligence/report/`

| Report | Audience |
|--------|----------|
| `coach_report.md` | Tactical decisions, positioning |
| `player_report.md` | Actionable focus areas |
| `training_report.md` | Drill recommendations |
| `match_summary` | State distribution, interactions |

Plus existing viz: heatmaps, shot chart, timeline, highlights.

---

## Digital Twin ⭐

**Path:** `backend/intelligence/world/digital_twin.py`

**Source of truth** for every frame:

```
WorldFrame
├── geometry (players, ball, court)
├── match_state
├── interactions[]
├── tactical snapshot
└── stroke observation
```

All layers read/write through the twin — not through ad-hoc module outputs.

---

## Sports Analytics Database

**Path:** `backend/storage/database.py` → `data/reports/padel_intelligence.db`

| Table | Purpose |
|-------|---------|
| `matches` | Match metadata |
| `rallies` | Interaction chains |
| `shots` | Structured shot facts (queryable) |
| `events` | Point-level events |
| `tactical` | Coach metrics |

Example queries:
- "Every bandeja from net after defensive lob"
- "Compare net control vs last 5 matches"

---

## Active Play Gate (Broadcast)

**Path:** `backend/analytics/events/active_play.py`

Filters non-game footage before intelligence runs:
- Audience close-ups
- Between-point dead time
- No on-court players

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Layer 1 Vision | ✅ Working (heuristic ball; YOLO players) |
| Layer 2 Geometry | ✅ `GeometryProjector` + entities |
| Layer 3 Match State | ✅ FSM (basic transitions) |
| Layer 4 Interactions | ✅ Graph + rally chains |
| Layer 5 Tactical | ✅ Coach rules + positioning |
| Layer 6 Knowledge | ✅ Structured facts + LLM hook |
| Layer 7 Reports | ✅ Coach/player/training |
| Digital Twin | ✅ Integrated in orchestrator |
| SQLite DB | ✅ Shot/rally/tactical storage |
| Ball: optical flow + physics | 🔲 Planned |
| Player: appearance embeddings | 🔲 Planned |
| VideoMAE strokes | 🔲 Planned |
| Auto landmark calibration | 🔲 Planned |

---

## Directory Map

```
backend/
├── detection/          # Layer 1
├── tracking/           # Layer 1
├── court/              # Layer 1 + 2
├── analytics/          # Layer 1 helpers + active play
├── intelligence/
│   ├── geometry/       # Layer 2
│   ├── world/          # Digital Twin
│   ├── match_state/    # Layer 3
│   ├── interaction/    # Layer 4
│   ├── tactical/       # Layer 5
│   ├── knowledge/      # Layer 6
│   ├── report/         # Layer 7
│   └── pipeline.py     # Layers 2–7 orchestration
├── storage/            # Analytics DB
└── pipeline/
    └── orchestrator.py # Layer 1 + intelligence hub
```

---

## Roadmap to 9.5/10

1. **Domain depth** — padel coaching rules from match study (formations, wall play, transitions)
2. **Ball physics pipeline** — optical flow, trajectory repair, bounce classification
3. **Player re-ID** — court geometry + appearance recovery after occlusion
4. **Temporal stroke model** — VideoMAE/SlowFast replacing pose heuristics
5. **Calibration** — Roboflow landmark model for broadcast angles
6. **Dashboard** — query DB + digital twin replay
7. **Multi-match** — comparative analytics across sessions

---

## Running

```bash
python run.py run video.mp4   # intelligence.enabled: true in padel.yaml
```

Outputs include `intelligence` block in `full_output.json`, `player_report.md`, `training_report.md`, and `padel_intelligence.db`.
