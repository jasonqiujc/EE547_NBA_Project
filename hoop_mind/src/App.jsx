// src/App.js
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import TeamList from "./components/TeamList";
import TeamHistory from "./components/TeamHistory";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<TeamList />} />
        <Route path="/history" element={<TeamHistory />} />
      </Routes>
    </Router>
  );
}

export default App;
