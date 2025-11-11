// src/App.js
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import TeamList from "./components/TeamList";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<TeamList />} />
      </Routes>
    </Router>
  );
}

export default App;
