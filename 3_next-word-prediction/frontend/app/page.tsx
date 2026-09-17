"use client";

import { FormEvent, useState } from "react";

type Prediction = {
  word: string;
  probability: number;
};

type ApiResponse = {
  input_text: string;
  generated_text: string;
  predictions: Prediction[];
};

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export default function Home() {
  const [text, setText] = useState("machine learning");
  const [numWords, setNumWords] = useState(5);
  const [topK, setTopK] = useState(5);
  const [temperature, setTemperature] = useState(1);
  const [sample, setSample] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  async function submitPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          num_words: numWords,
          top_k: topK,
          temperature,
          sample,
        }),
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      setResult(await response.json());
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Prediction request failed",
      );
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="shell">
      <section className="workspace">
        <div className="panel controls">
          <div>
            <p className="eyebrow">Hybrid language inference</p>
            <h1>Next Word Predictor</h1>
          </div>

          <form onSubmit={submitPrediction}>
            <label>
              Seed text
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                rows={4}
              />
            </label>

            <div className="grid">
              <label>
                Words
                <input
                  min={1}
                  max={20}
                  type="number"
                  value={numWords}
                  onChange={(event) => setNumWords(Number(event.target.value))}
                />
              </label>
              <label>
                Top K
                <input
                  min={1}
                  max={20}
                  type="number"
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </label>
            </div>

            <label>
              Temperature
              <input
                min={0.1}
                max={3}
                step={0.1}
                type="range"
                value={temperature}
                onChange={(event) => setTemperature(Number(event.target.value))}
              />
              <span>{temperature.toFixed(1)}</span>
            </label>

            <label className="toggle">
              <input
                checked={sample}
                type="checkbox"
                onChange={(event) => setSample(event.target.checked)}
              />
              Sample from top-k
            </label>

            <button disabled={isLoading || !text.trim()} type="submit">
              {isLoading ? "Predicting..." : "Predict"}
            </button>
          </form>
        </div>

        <div className="panel results">
          {error ? <p className="error">{error}</p> : null}

          <div>
            <p className="eyebrow">Generated text</p>
            <p className="generated">
              {result?.generated_text ?? "Run a prediction to see generated text."}
            </p>
          </div>

          <div>
            <p className="eyebrow">Top candidates</p>
            <div className="predictions">
              {(result?.predictions ?? []).map((prediction) => (
                <div className="prediction" key={prediction.word}>
                  <span>{prediction.word}</span>
                  <strong>{(prediction.probability * 100).toFixed(2)}%</strong>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
