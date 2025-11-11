import React from "react";
import TeamList from "../components/TeamList";
import "./Intro.css";

export default function Intro() {
  return (
    <div className="container">
      <h1 style={{ textAlign: "center", margin: "20px" }}>Choose Your Team</h1>
      <TeamList />
    </div>
  );
}
