import { useEffect, useRef, useState } from "react";

const API_URL = "http://localhost:8000/api/simplify";

function LoadingSpinner() {
  return (
    <span className="flex items-center justify-center">
      <svg
        className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
        fill="none"
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
      Simplifying...
    </span>
  );
}

function App() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const abortRef = useRef(null);

  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();

    const text = input.trim();
    if (!text) {
      setError("Please enter some text to simplify.");
      setResult(null);
      return;
    }

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({ text }),
      });

      let data = null;
      const contentType = response.headers.get("content-type") || "";

      if (contentType.includes("application/json")) {
        data = await response.json();
      } else {
        const message = await response.text();
        throw new Error(message || "API request failed");
      }

      if (!response.ok) {
        const detail =
          (data && (data.detail || data.message)) || "API request failed";
        throw new Error(detail);
      }

      setResult(data);
    } catch (err) {
      if (err?.name === "AbortError") return;
      setResult(null);
      setError(err?.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const scores = result?.readability_scores || {};

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50">
      <div className="max-w-5xl mx-auto px-6 py-12">
        <div className="text-center mb-12">
          <div className="inline-block p-3 bg-green-100 rounded-full mb-4">
            Financial Literacy Assistant
          </div>

          <svg
            className="w-12 h-12 text-green-600 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
            />
          </svg>

          <h1 className="text-5xl font-bold bg-gradient-to-r from-green-600 to-emerald-600 bg-clip-text text-transparent mb-3">
            Financial Literacy Assistant
          </h1>

          <p className="text-gray-600 text-lg">
            Transform complex financial text into plain language
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-100">
          <form onSubmit={handleSubmit}>
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              Enter Financial Text
            </label>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="w-full p-4 border-2 border-gray-200 rounded-xl focus:border-green-500 focus:ring-2 focus:ring-green-200 transition-all resize-none"
              rows={8}
              placeholder="Paste complex financial terminology, investment concepts, or economic explanations here..."
            />

            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="mt-4 w-full py-4 bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              {loading ? <LoadingSpinner /> : "Simplify Text"}
            </button>
          </form>
        </div>

        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-8 rounded-r-xl">
            <p className="text-red-700 font-medium">{error}</p>
          </div>
        )}

        {result && (
          <div className="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100">
            <div className="bg-gradient-to-r from-green-600 to-emerald-600 p-6">
              <h2 className="text-2xl font-bold text-white flex items-center">
                <svg
                  className="w-6 h-6 mr-2"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Simplified Version
              </h2>
            </div>

            <div className="p-8">
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-xl mb-6 border border-green-200">
                <p className="text-gray-800 text-lg leading-relaxed">
                  {result.simplified_text}
                </p>
              </div>

              <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                <svg
                  className="w-5 h-5 mr-2 text-green-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
                Readability Score
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-5 rounded-xl border border-green-200">
                  <p className="text-sm text-gray-600 mb-1">
                    Flesch Reading Ease
                  </p>
                  <p className="text-3xl font-bold text-green-700">
                    {scores["Flesch Reading Ease"] ?? "—"}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-amber-50 to-orange-50 p-5 rounded-xl border border-amber-200">
                  <p className="text-sm text-gray-600 mb-1">Gunning Fog Index</p>
                  <p className="text-3xl font-bold text-amber-700">
                    {scores["Gunning Fog Index"] ?? "—"}
                  </p>
                </div>

                <div className="bg-gradient-to-br from-teal-50 to-cyan-50 p-5 rounded-xl border border-teal-200">
                  <p className="text-sm text-gray-600 mb-1">
                    Kincaid Grade Level
                  </p>
                  <p className="text-3xl font-bold text-teal-700">
                    {scores["Kincaid Grade Level"] ?? "—"}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;







