import { useState } from "react";
import {
  ArrowLeft,
  BrainCircuit,
  CircleDot,
  RotateCcw,
  Send,
  Sparkles,
  User,
  Zap,
} from "lucide-react";

const API_URL = "http://localhost:8000";

// Tools the sidebar knows how to highlight.
// Keys should match whatever the backend sends back in `data.tool`.
const TOOLS = [
  { key: "search", label: "Search" },
  { key: "calculator", label: "Calculator" },
  { key: "llm", label: "LLM" },
];

function Chatbot({ onBack }) {
  const [input, setInput] = useState("");

  const [messages, setMessages] = useState([
    {
      role: "agent",
      text: "I'm ready. Give me a task and I'll figure out the best way to handle it.",
    },
  ]);

  const [isThinking, setIsThinking] = useState(false);

  // Dynamic agent state, now driven by the backend response
  // instead of being hardcoded.
  const [reward, setReward] = useState(null); // null until we get a real value
  const [selectedTool, setSelectedTool] = useState(null); // e.g. "search" | "calculator" | "llm"

  /*
   * ========================================================
   * SEND MESSAGE
   * ========================================================
   */

  const sendMessage = async () => {
    if (!input.trim() || isThinking) {
      return;
    }

    const userMessage = input.trim();

    /*
     * Save the conversation BEFORE adding the
     * current user message.
     *
     * Backend will receive previous history.
     */

    const history = messages.map((message) => ({
      role: message.role === "agent"
        ? "assistant"
        : "user",

      content: message.text,
    }));

    // Add user message immediately
    setMessages((previousMessages) => [
      ...previousMessages,
      {
        role: "user",
        text: userMessage,
      },
    ]);

    setInput("");
    setIsThinking(true);

    try {
      const response = await fetch(
        `${API_URL}/api/chat`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            message: userMessage,
            history: history,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Server error: ${response.status}`
        );
      }

      const data = await response.json();

      /*
       * Add actual LLM response
       */

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "agent",
          text: data.response,
        },
      ]);

      /*
       * Update reward + selected tool from this turn.
       * Expecting the backend to send something like:
       *   { response: "...", tool: "search", reward: 0.87 }
       * Falls back gracefully if either field is missing.
       */

      if (typeof data.reward === "number") {
        setReward(data.reward);
      }

      if (typeof data.tool === "string") {
        setSelectedTool(data.tool.toLowerCase());
      }

    } catch (error) {

      console.error("Chat error:", error);

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          role: "agent",
          text:
            "Sorry, I couldn't connect to the AI backend. Make sure FastAPI is running on port 8000.",
        },
      ]);

    } finally {
      setIsThinking(false);
    }
  };


  /*
   * ========================================================
   * ENTER KEY
   * ========================================================
   */

  const handleKeyDown = (event) => {

    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      sendMessage();
    }
  };


  /*
   * ========================================================
   * RESET
   * ========================================================
   */

  const resetChat = () => {

    setMessages([
      {
        role: "agent",
        text:
          "I'm ready. Give me a task and I'll figure out the best way to handle it.",
      },
    ]);

    setInput("");
    setIsThinking(false);
    setReward(null);
    setSelectedTool(null);
  };


  /*
   * ========================================================
   * HELPERS
   * ========================================================
   */

  const formatReward = (value) => {
    if (value === null || value === undefined) {
      return "—";
    }
    const sign = value >= 0 ? "+" : "";
    return `${sign}${value.toFixed(2)}`;
  };


  /*
   * ========================================================
   * UI
   * ========================================================
   */

  return (
    <div className="agent-console">

      {/* ===================================================
          HEADER
      =================================================== */}

      <header className="console-header">

        <div className="console-brand">

          <button
            className="icon-button"
            onClick={onBack}
            aria-label="Back"
          >
            <ArrowLeft size={18} />
          </button>

          <div className="brand-icon">
            <BrainCircuit size={20} />
          </div>

          <div>
            <strong>NEXUS</strong>
            <span>RL AGENT</span>
          </div>

        </div>


        <div className="header-status">

          <CircleDot size={13} />

          SYSTEM ONLINE

        </div>


        <button
          className="icon-button"
          onClick={resetChat}
          aria-label="Reset chat"
        >
          <RotateCcw size={17} />
        </button>

      </header>


      {/* ===================================================
          MAIN LAYOUT
      =================================================== */}

      <div className="console-layout">


        {/* =================================================
            SIDEBAR
        ================================================= */}

        <aside className="agent-sidebar">

          <div className="sidebar-title">

            <span>
              AGENT STATE
            </span>

            <span className="online-dot"></span>

          </div>


          <div className="state-card">

            <span className="state-label">
              POLICY
            </span>

            <div className="policy-item">
              <span>Tool Selection</span>
              <strong>
                {selectedTool
                  ? TOOLS.find((t) => t.key === selectedTool)?.label ?? selectedTool
                  : "ACTIVE"}
              </strong>
            </div>

            <div className="policy-item">
              <span>RL Environment</span>
              <strong>READY</strong>
            </div>

            <div className="policy-item">
              <span>Agent Mode</span>
              <strong>AUTO</strong>
            </div>

          </div>


          <div className="state-card">

            <span className="state-label">
              REWARD
            </span>

            <div className="big-stat">
              {formatReward(reward)}
            </div>

            <small>
              Current expected reward
            </small>

          </div>


          <div className="state-card">

            <span className="state-label">
              ENVIRONMENT
            </span>

            {TOOLS.map((tool) => (
              <div
                key={tool.key}
                className={`environment-status${
                  selectedTool === tool.key ? " active" : ""
                }`}
              >
                <span>{tool.label}</span>
                <strong>
                  {selectedTool === tool.key ? "IN USE" : "READY"}
                </strong>
              </div>
            ))}

          </div>


          <div className="sidebar-footer">

            <Zap size={15} />

            Learning mode enabled

          </div>

        </aside>


        {/* =================================================
            CHAT
        ================================================= */}

        <main className="chat-area">

          <div className="chat-header">

            <div>

              <span className="section-label">
                AGENT CONSOLE
              </span>

              <h1>
                What should I do?
              </h1>

            </div>


            <div className="model-badge">

              <Sparkles size={14} />

              GPT-OSS-20B

            </div>

          </div>


          {/* =================================================
              MESSAGES
          ================================================= */}

          <div className="messages">

            {messages.map(
              (message, index) => (

                <div
                  key={index}
                  className={`message-row ${
                    message.role
                  }`}
                >

                  <div className="message-avatar">

                    {message.role === "user" ? (
                      <User size={16} />
                    ) : (
                      <BrainCircuit size={16} />
                    )}

                  </div>


                  <div className="message-content">

                    <span className="message-role">

                      {message.role === "user"
                        ? "YOU"
                        : "NEXUS"}

                    </span>

                    <p>
                      {message.text}
                    </p>

                  </div>

                </div>

              )
            )}


            {/* =================================================
                THINKING
            ================================================= */}

            {isThinking && (

              <div className="thinking">

                <BrainCircuit size={17} />

                <span>
                  NEXUS is thinking...
                </span>

                <div className="thinking-dots">

                  <i></i>
                  <i></i>
                  <i></i>

                </div>

              </div>

            )}

          </div>


          {/* =================================================
              INPUT
          ================================================= */}

          <div className="input-container">

            <div className="input-wrapper">

              <textarea
                value={input}
                onChange={(event) =>
                  setInput(event.target.value)
                }
                onKeyDown={handleKeyDown}
                placeholder="Give the agent a task..."
                rows="1"
                disabled={isThinking}
              />


              <button
                className="send-button"
                onClick={sendMessage}
                disabled={
                  !input.trim() ||
                  isThinking
                }
                aria-label="Send"
              >

                <Send size={18} />

              </button>

            </div>


            <div className="input-hint">

              <span>
                ENTER TO SEND
              </span>

              <span>
                GROQ • GPT-OSS-20B
              </span>

            </div>

          </div>

        </main>

      </div>

    </div>
  );
}


export default Chatbot;