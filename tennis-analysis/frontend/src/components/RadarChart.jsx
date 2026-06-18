import Plot from "react-plotly.js";

export default function RadarChart({ scores }) {
  const labels = ["Serve", "Return", "Movement", "Consistency", "Aggression", "Stamina", "Coverage"];
  const values = [
    scores.serve,
    scores.return,
    scores.movement,
    scores.consistency,
    scores.aggression,
    scores.stamina,
    scores.court_coverage,
  ];

  return (
    <Plot
      data={[
        {
          type: "scatterpolar",
          r: [...values, values[0]],
          theta: [...labels, labels[0]],
          fill: "toself",
          fillcolor: "rgba(29, 155, 240, 0.2)",
          line: { color: "#1d9bf0" },
        },
      ]}
      layout={{
        polar: {
          radialaxis: { visible: true, range: [0, 100], gridcolor: "#2f3336" },
          bgcolor: "transparent",
        },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#e7e9ea" },
        margin: { t: 30, b: 30, l: 40, r: 40 },
        height: 320,
      }}
      config={{ displayModeBar: false }}
      style={{ width: "100%" }}
    />
  );
}
