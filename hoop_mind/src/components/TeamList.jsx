// src/components/TeamList.js
import React, { useState, useEffect } from "react";
import teams from "../data/teams";

const API_KEY = "092b739b3032487eb53c025ce0b32f8e";

const TeamList = () => {
  const [team1, setTeam1] = useState(null);
  const [team2, setTeam2] = useState(null);
  const [seasonStats, setSeasonStats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);

  // Fetch season stats for all teams once
  useEffect(() => {
    const fetchSeasonStats = async () => {
      try {
        const url = `https://api.sportsdata.io/v3/nba/scores/json/TeamSeasonStats/2025?key=${API_KEY}`;
        const res = await fetch(url);
        const data = await res.json();
        if (!Array.isArray(data)) {
          console.error("Unexpected API response:", data);
          return;
        }
        setSeasonStats(data);
      } catch (err) {
        console.error("Error fetching season stats:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchSeasonStats();
  }, []);

  const handleClick = (team) => {
    if (!team1) {
      setTeam1(team);
    } else if (!team2 && team.abbrev !== team1.abbrev) {
      setTeam2(team);
    } else if (team.abbrev === team1.abbrev) {
      setTeam1(null);
    } else if (team.abbrev === team2?.abbrev) {
      setTeam2(null);
    }
  };

  const getRecord = (teamAbbrev) => {
    const stat = seasonStats.find((s) => s.Team === teamAbbrev);
    if (!stat) return null;
    return {
      games: stat.Games,
      wins: stat.Wins,
      losses: stat.Losses,
    };
  };

  const record1 = team1 ? getRecord(team1.abbrev) : null;
  const record2 = team2 ? getRecord(team2.abbrev) : null;

  // compute win rate
  const getWinRate = (record) => {
    if (!record || record.games === 0) return 0;
    return ((record.wins / record.games) * 100).toFixed(1);
  };

  const handlePredict = () => {
    if (!team1 || !team2) {
      alert("Please select two teams first!");
      return;
    }
    setShowModal(true);
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      {/* Title & Predict Button */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "20px",
        }}
      >
        <h1 style={{ textAlign: "center", flex: 1 }}>Choose Your Teams</h1>
        <button
          onClick={handlePredict}
          style={{
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            padding: "10px 16px",
            borderRadius: "8px",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          Predict
        </button>
      </div>

      {/* Matchup Display */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "30px",
          gap: "30px",
        }}
      >
        {/* Team 1 */}
        <div style={{ textAlign: "center" }}>
          {team1 ? (
            <>
              <img
                src={team1.logo}
                alt={team1.name}
                width="90"
                height="90"
                style={{ borderRadius: "50%" }}
              />
              <p style={{ fontWeight: "bold", marginTop: "8px" }}>
                {team1.abbrev}
              </p>
              {record1 && (
                <p style={{ fontSize: "0.9em", color: "#555" }}>
                  Last season: {record1.wins}-{record1.losses}
                </p>
              )}
            </>
          ) : (
            <p style={{ color: "#888" }}>Select Team 1</p>
          )}
        </div>

        {/* VS text */}
        <h2 style={{ margin: "0 10px" }}>VS</h2>

        {/* Team 2 */}
        <div style={{ textAlign: "center" }}>
          {team2 ? (
            <>
              <img
                src={team2.logo}
                alt={team2.name}
                width="90"
                height="90"
                style={{ borderRadius: "50%" }}
              />
              <p style={{ fontWeight: "bold", marginTop: "8px" }}>
                {team2.abbrev}
              </p>
              {record2 && (
                <p style={{ fontSize: "0.9em", color: "#555" }}>
                  Last season: {record2.wins}-{record2.losses}
                </p>
              )}
            </>
          ) : (
            <p style={{ color: "#888" }}>Select Team 2</p>
          )}
        </div>
      </div>

      {/* Team Grid */}
      {loading ? (
        <p>Loading season stats...</p>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(100px, 1fr))",
            gap: "20px",
          }}
        >
          {teams.map((team) => (
            <div
              key={team.abbrev}
              onClick={() => handleClick(team)}
              style={{
                cursor: "pointer",
                textAlign: "center",
                transform:
                  team1?.abbrev === team.abbrev || team2?.abbrev === team.abbrev
                    ? "scale(1.1)"
                    : "scale(1)",
                transition: "transform 0.2s ease",
              }}
            >
              <img
                src={team.logo}
                alt={team.name}
                width="80"
                height="80"
                style={{
                  border:
                    team1?.abbrev === team.abbrev ||
                    team2?.abbrev === team.abbrev
                      ? "3px solid #007bff"
                      : "1px solid #ccc",
                  borderRadius: "50%",
                  padding: "5px",
                }}
              />
              <p
                style={{
                  marginTop: "6px",
                  fontSize: "0.9em",
                  fontWeight:
                    team1?.abbrev === team.abbrev ||
                    team2?.abbrev === team.abbrev
                      ? "bold"
                      : "normal",
                }}
              >
                {team.abbrev}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Modal (Win Rate Prediction) */}
      {showModal && team1 && team2 && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100vw",
            height: "100vh",
            backgroundColor: "rgba(0,0,0,0.5)",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            zIndex: 1000,
          }}
        >
          <div
            style={{
              background: "#fff",
              borderRadius: "12px",
              padding: "30px 40px",
              width: "320px",
              textAlign: "center",
              boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
            }}
          >
            <h2>🏀 Win Rate Prediction</h2>
            <div style={{ marginTop: "15px" }}>
              <p>
                <strong>{team1.name}</strong>:{" "}
                {record1 ? `${getWinRate(record1)}%` : "N/A"}
              </p>
              <p>
                <strong>{team2.name}</strong>:{" "}
                {record2 ? `${getWinRate(record2)}%` : "N/A"}
              </p>
            </div>
            <button
              onClick={() => setShowModal(false)}
              style={{
                marginTop: "20px",
                backgroundColor: "#007bff",
                color: "white",
                border: "none",
                padding: "8px 14px",
                borderRadius: "6px",
                cursor: "pointer",
              }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default TeamList;
