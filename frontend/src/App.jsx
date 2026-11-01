import { useState } from "react";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function readResponse(response) {
    const body = await response.text();
    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      try {
        return JSON.parse(body);
      } catch {
        return { detail: body || "The server returned invalid JSON." };
      }
    }

    return { detail: body.trim() || `Server returned HTTP ${response.status}.` };
  }

  async function submitPrompt(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setAnswer("");

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const result = await readResponse(response);

      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          throw new Error(
            "The request was not authorized. Reopen the current tunnel URL and complete its email verification, then try again.",
          );
        }
        throw new Error(
          result.detail || `The request failed with HTTP ${response.status}.`,
        );
      }

      if (typeof result.text !== "string") {
        throw new Error(
          "The inference server returned an unexpected response. Check that the React UI and FastAPI server use the same tunnel URL.",
        );
      }

      setAnswer(result.text);
    } catch (requestError) {
      setError(requestError.message || "Unable to reach the inference server.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="page">
      <header className="hero">
        <p className="eyebrow">INSTRUMENTATION ENGINEERING</p>
        <h1>Loop Assistant</h1>
        <p className="intro">
          Ask a question about instrumentation loops. Include project details
          when available; the assistant will call out assumptions and missing
          information.
        </p>
      </header>

      <form className="prompt-card" onSubmit={submitPrompt}>
        <label htmlFor="prompt">Your prompt</label>
        <textarea
          id="prompt"
          name="prompt"
          rows="7"
          placeholder="For example: Explain the engineering checks for a 4-20 mA HART pressure transmitter loop to a DCS."
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          required
        />
        <div className="form-footer">
          <span>Model configuration is managed by the server.</span>
          <button type="submit" disabled={busy || !prompt.trim()}>
            {busy ? "Generating…" : "Generate response"}
          </button>
        </div>
      </form>

      {error && (
        <section className="message error" role="alert">
          <h2>Request failed</h2>
          <p>{error}</p>
        </section>
      )}

      {busy && (
        <p className="loading" role="status">
          The model is generating a response. First-time model loading may take
          several minutes.
        </p>
      )}

      {answer && (
        <section className="message answer" aria-live="polite">
          <h2>Response</h2>
          <pre>{answer}</pre>
        </section>
      )}

      <footer className="disclaimer">
        AI output is illustrative, not an approved design or work instruction.
        Verify every technical detail against approved project documents and
        applicable standards with a qualified engineer.
      </footer>
    </main>
  );
}
