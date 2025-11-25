// src/App.jsx
import { useEffect, useState } from "react";
import teams from "../data/teams";
import PredictionModal from "./PredictionModal";
import Modal from "./Modal";

const API_KEY = import.meta.env.VITE_SPORTS_DATA_IO_KEY;

// Format API date: YYYY-MMM-DD (ex: 2025-NOV-18)
function formatDateToSportsData(date) {
  const d = new Date(date);
  const year = d.getFullYear();
  const month = d.getMonth() + 1;
  const day = String(d.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function App() {
    const [recentFinals, setRecentFinals] = useState([]);
    const [upcomingScheduled, setUpcomingScheduled] = useState([]);
    const [loading, setLoading] = useState(true);

  // Fetch games for a single date
  async function fetchGamesByDate(date) {
    const formatted = formatDateToSportsData(date);
    const url = `https://api.sportsdata.io/v3/nba/scores/json/ScoresBasic/${formatted}?key=${API_KEY}`;
    
    const res = await fetch(url, {
      headers: { "Ocp-Apim-Subscription-Key": API_KEY }
    });

    if (!res.ok) return [];
    return await res.json();
  }

  // Requirement (1): Get 5 most recent Final games
  async function loadRecentFinals() {
    let finals = [];
    let dayOffset = 0;

    while (finals.length < 6) {
      const date = new Date();
      date.setDate(date.getDate() - dayOffset);

      const games = await fetchGamesByDate(date);

      finals.push(...games.filter(g => g.Status === "Final"));

      dayOffset++;
    }

    return finals.slice(0, 6);
  }

  // Requirement (2): Get 5 upcoming Scheduled games
  async function loadUpcomingScheduled() {
    let scheduled = [];
    let dayOffset = 0;

    while (scheduled.length < 6) {
      const date = new Date();
      date.setDate(date.getDate() + dayOffset);

      const games = await fetchGamesByDate(date);

      scheduled.push(...games.filter(g => g.Status === "Scheduled"));

      dayOffset++;
    }

    return scheduled.slice(0, 6);
  }

  // Load both lists on mount
  useEffect(() => {
    async function load() {
      setLoading(true);

      const finals = await loadRecentFinals();
      const scheduled = await loadUpcomingScheduled();

      setRecentFinals(finals);
      setUpcomingScheduled(scheduled);

      setLoading(false);
    }

    load();
  }, []);

  if (loading) {
    return <p style={{ color: "white", padding: 20 }}>Loading games...</p>;
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "#0b1120",
        color: "#e5e7eb",
        padding: "24px",
        fontFamily: "system-ui",
      }}
    >

      {/* Recent Finals */}
      <section style={{ marginBottom: 40 }}>
        <h1 style={{ fontSize: 30, marginBottom: 10 }}>Recent Final Games</h1>
        <GameGrid games={recentFinals} />
      </section>

      {/* Upcoming Scheduled */}
      <section>
        <h1 style={{ fontSize: 30, marginBottom: 10 }}>Upcoming Scheduled Games</h1>
        <GameGrid games={upcomingScheduled} />
      </section>
    </div>
  );
}

function GameGrid({ games }) {
    const [modalVisible, setModalVisible] = useState(false);
    const [selectedDetail, setSelectedDetail] = useState(null);

    function getTeamInfo(abbrev) {
        return teams.find((t) => t.abbrev === abbrev) || {};
    }

    async function openWindow(game) {
        const game_date = game.Day?.split("T")[0];
        const home_team = game.HomeTeam;
        const away_team = game.AwayTeam;

        // show loading inside modal
        setSelectedDetail({ loading: true });
        setModalVisible(true);

        try {
            const res = await fetch("http://3.101.69.200:8000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    game_date,
                    home_team,
                    away_team
                })
            });

            const data = await res.json();

            setSelectedDetail(data);

        } catch (err) {
            console.error(err);
            setSelectedDetail({ message: "Error fetching prediction." });
        }
    }

    return (
        <div
        style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: "16px",
        }}
        >
        {games.map((game) => {
            const away = getTeamInfo(game.AwayTeam);
            const home = getTeamInfo(game.HomeTeam);

            return (
            <div
                key={game.GameID}
                style={{
                borderRadius: "16px",
                padding: "16px",
                background:
                    "radial-gradient(circle at top left, rgba(59,130,246,0.25), transparent 60%), #020617",
                border: "1px solid rgba(148,163,184,0.25)",
                position: "relative",
                }}
            >
                {/* DATE (top-left) */}
                <div
                style={{
                    position: "absolute",
                    top: 10,
                    left: 12,
                    fontSize: 16,
                    color: "#9ca3af",
                }}
                >
                {game.Day?.split("T")[0]}
                </div>

                {/* STATUS (top-right) */}
                {/* <div
                style={{
                    position: "absolute",
                    top: 10,
                    right: 12,
                    fontSize: 12,
                    borderRadius: 20,
                    padding: "2px 8px",
                    backgroundColor:
                    game.Status === "Final" ? "#14532d" : "#1e3a8a",
                    color: "white",
                }}
                >
                    {game.Status === "Final" ? "Predict history" : "Predict"}
                </div> */}
                <button
                    onClick={() => openWindow(game)}
                    style={{
                        position: "absolute",
                        top: 10,
                        right: 12,
                        fontSize: 12,
                        borderRadius: 5,
                        padding: "7px",
                        backgroundColor:
                        game.Status === "Final" ? "#14532d" : "#1e3a8a",
                        color: "white",
                    }}
                >
                {game.Status === "Final" ? "Predict history" : "Predict"}
                </button>

                {/* TEAM ICONS + SCORE ROW */}
                <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginTop: 30,
                }}
                >
                {/* AWAY SIDE */}
                <div
                    style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    }}
                >
                    <img src={away.logo} alt={away.name} style={{ width: 55 }} />
                    <div style={{ marginTop: 6 }}>{away.abbrev}</div>
                </div>

                {/* SCORE */}
                <div style={{ textAlign: "center" }}>
                    {game.Status === "Final" ? (
                    <div style={{ fontSize: 26, fontWeight: "bold" }}>
                        {game.AwayTeamScore} <span style={{ fontSize: 16 }}>vs</span>{" "}
                        {game.HomeTeamScore}
                    </div>
                    ) : (
                    <div style={{ fontSize: 16 }}>
                        {game.DateTime?.split("T")[0]}
                    </div>
                    )}
                </div>

                {/* HOME SIDE */}
                <div
                    style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    }}
                >
                    <img src={home.logo} alt={home.name} style={{ width: 55 }} />
                    <div style={{ marginTop: 6 }}>{home.abbrev}</div>
                </div>
                </div>
                <PredictionModal
                  visible={modalVisible}
                  onClose={() => setModalVisible(false)}
                  detail={selectedDetail}
                />
            </div>
            );
        })}
        </div>
    );
}

export default App;