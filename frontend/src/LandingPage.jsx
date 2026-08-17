import {
  ArrowRight,
  BrainCircuit,
  Calculator,
  Search,
  Sparkles,
  Zap,
} from "lucide-react";

function LandingPage({ onLaunch }) {
  return (
    <div className="landing-page">
      {/* Navbar */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-icon">
            <BrainCircuit size={22} />
          </div>
          <span>NEXUS</span>
        </div>

        <div className="nav-links">
          <a href="#how-it-works">How it works</a>
          <a href="#research">Research</a>
          <button className="nav-launch" onClick={onLaunch}>
            Launch Agent
            <ArrowRight size={16} />
          </button>
        </div>
      </nav>

      {/* Hero */}
      <main className="hero-section">
        <div className="hero-badge">
          <span className="pulse-dot"></span>
          REINFORCEMENT LEARNING × LLM AGENTS
        </div>

        <h1>
          An AI agent that
          <br />
          <span>learns what to do.</span>
        </h1>

        <p className="hero-description">
          NEXUS uses reinforcement learning to decide whether a task
          requires search, calculation, reasoning, or another tool —
          instead of blindly calling everything.
        </p>

        <div className="hero-actions">
          <button className="primary-button" onClick={onLaunch}>
            Enter Agent
            <ArrowRight size={18} />
          </button>

          <a href="#how-it-works" className="secondary-button">
            Explore the system
          </a>
        </div>

        {/* Tool Cards */}
        <div className="tool-preview">
          <div className="tool-card">
            <Search size={22} />
            <span>Search</span>
            <small>Real-world knowledge</small>
          </div>

          <div className="tool-card active">
            <Calculator size={22} />
            <span>Calculator</span>
            <small>Precise computation</small>
          </div>

          <div className="tool-card">
            <Sparkles size={22} />
            <span>LLM</span>
            <small>Reasoning & generation</small>
          </div>
        </div>
      </main>

      {/* How it works */}
      <section id="how-it-works" className="info-section">
        <div className="section-label">01 — SYSTEM</div>

        <h2>
          The agent doesn't just answer.
          <br />
          <span>It chooses how to answer.</span>
        </h2>

        <div className="pipeline">
          <div className="pipeline-item">
            <strong>01</strong>
            <h3>User Prompt</h3>
            <p>The agent receives the user's task.</p>
          </div>

          <div className="pipeline-line"></div>

          <div className="pipeline-item">
            <strong>02</strong>
            <h3>RL Policy</h3>
            <p>The learned policy selects the best action.</p>
          </div>

          <div className="pipeline-line"></div>

          <div className="pipeline-item">
            <strong>03</strong>
            <h3>Tool Execution</h3>
            <p>The selected tool performs the task.</p>
          </div>

          <div className="pipeline-line"></div>

          <div className="pipeline-item">
            <strong>04</strong>
            <h3>LLM Response</h3>
            <p>The result is transformed into a final answer.</p>
          </div>
        </div>
      </section>

      {/* Research */}
      <section id="research" className="research-section">
        <div className="research-content">
          <div>
            <div className="section-label">02 — RESEARCH</div>

            <h2>
              Learning the
              <br />
              <span>right action.</span>
            </h2>
          </div>

          <p>
            Instead of hardcoding rules such as "if the query contains
            numbers, use a calculator", NEXUS learns tool-selection
            behavior through rewards.
          </p>
        </div>

        <div className="reward-card">
          <div className="reward-header">
            <span>RL POLICY</span>
            <span className="live-status">● LIVE</span>
          </div>

          <div className="probability">
            <div>
              <span>Search</span>
              <strong>3%</strong>
            </div>
            <div className="bar">
              <div style={{ width: "3%" }}></div>
            </div>
          </div>

          <div className="probability selected">
            <div>
              <span>Calculator</span>
              <strong>94%</strong>
            </div>
            <div className="bar">
              <div style={{ width: "94%" }}></div>
            </div>
          </div>

          <div className="probability">
            <div>
              <span>LLM</span>
              <strong>3%</strong>
            </div>
            <div className="bar">
              <div style={{ width: "3%" }}></div>
            </div>
          </div>

          <div className="reward-value">
            <Zap size={18} />
            Expected Reward: <strong>+0.94</strong>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="final-cta">
        <Sparkles size={28} />

        <h2>Ready to test the agent?</h2>

        <p>
          Give it a task and watch the decision-making process happen.
        </p>

        <button className="primary-button" onClick={onLaunch}>
          Launch NEXUS
          <ArrowRight size={18} />
        </button>
      </section>

      <footer>
        <span>© 2026 NEXUS AI</span>
        <span>Reinforcement Learning × Generative AI</span>
      </footer>
    </div>
  );
}

export default LandingPage;