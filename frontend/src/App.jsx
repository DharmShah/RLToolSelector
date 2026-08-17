import { useState } from "react";
import LandingPage from "./LandingPage";
import Chatbot from "./Chatbot";
import "./App.css";

function App() {
  const [page, setPage] = useState("landing");

  return page === "landing" ? (
    <LandingPage
      onLaunch={() => setPage("chatbot")}
    />
  ) : (
    <Chatbot
      onBack={() => setPage("landing")}
    />
  );
}

export default App;