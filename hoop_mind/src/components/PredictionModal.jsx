import React from "react";
import Modal from "./Modal";
import teams from "../data/teams";

export default function PredictionModal({ visible, onClose, detail }) {
  if (!detail || detail.loading)
    return (
      <Modal visible={visible} onClose={onClose}>
        <p style={{ color: "white" }}>Loading prediction...</p>
      </Modal>
    );

  if (detail.message)
    return (
      <Modal visible={visible} onClose={onClose}>
        <p style={{ color: "white" }}>{detail.message}</p>
      </Modal>
    );

  const home = teams.find((t) => t.abbrev === detail.home_team) || {};
  const away = teams.find((t) => t.abbrev === detail.away_team) || {};

  const homeRate = detail.pred_home_score;
  const awayRate = detail.pred_away_score;

  return (
    <Modal visible={visible} onClose={onClose}>
      <div
        style={{
          padding: 20,
          color: "#050911ff",
          fontFamily: "system-ui",
          textAlign: "center",
        }}
      >
        {/* Title */}
        <h2 style={{ fontSize: 24, marginBottom: 16 }}>Prediction Result</h2>

        {/* ---------------- TEAM VS TEAM ---------------- */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-around",
            marginBottom: 20,
          }}
        >
          {/* Away Team */}
          <div style={{ textAlign: "center" }}>
            <img
              src={away.logo}
              alt={away.name}
              style={{ width: 70, filter: "drop-shadow(0 0 6px #1e40af)" }}
            />
            <p style={{ marginTop: 6, fontSize: 18 }}>{away.abbrev}</p>
          </div>

          <p style={{ fontSize: 22, margin: "0 20px" }}>VS</p>

          {/* Home Team */}
          <div style={{ textAlign: "center" }}>
            <img
              src={home.logo}
              alt={home.name}
              style={{ width: 70, filter: "drop-shadow(0 0 6px #047857)" }}
            />
            <p style={{ marginTop: 6, fontSize: 18 }}>{home.abbrev}</p>
          </div>
        </div>

        {/* ---------------- GAME TIME ---------------- */}
        <p style={{ color: "#9ca3af", marginBottom: 20 }}>
          <strong>{detail.game_date}</strong>
        </p>

        {/* ---------------- WIN RATE BAR ---------------- */}
        <div style={{ marginTop: 20 }}>
          <p style={{ marginBottom: 8, fontSize: 16 }}>Score Prediction</p>

          <div
            style={{
              display: "flex",
              width: "100%",
              height: 20,
              borderRadius: 10,
              overflow: "hidden",
              background: "#1f2937",
              border: "1px solid rgba(255,255,255,0.1)",
            }}
          >
            <div
              style={{
                width: `${awayRate*100}%`,
                background: "#2563eb",
              }}
            />

            <div
              style={{
                width: `${homeRate*100}%`,
                background: "#16a34a",
              }}
            />
          </div>

          {/* Labels under bar */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: 6,
              fontSize: 14,
              color: "#131923ff",
            }}
          >
            <span>
              {away.abbrev}: <strong>{awayRate}</strong>
            </span>
            <span>
              {home.abbrev}: <strong>{homeRate}</strong>
            </span>
          </div>
        </div>

        {/* ---------------- PREDICTED DIFF ---------------- */}
        <p style={{ marginTop: 24, fontSize: 16 }}>
          Predicted Point Difference:
          <strong> {detail.predicted_point_diff}</strong>
        </p>
      </div>
    </Modal>
  );
}
