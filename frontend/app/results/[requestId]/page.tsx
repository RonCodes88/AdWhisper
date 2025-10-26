"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { getResults, checkStatus } from "@/lib/api";
import type { BiasReport, AnalysisStatus } from "@/types/api";
import { LoadingSpinner } from "@/app/components/LoadingSpinner";
import { StatusTracker } from "@/app/components/StatusTracker";
import { ScoreCard } from "@/app/components/ScoreCard";
import { BiasIssueCard } from "@/app/components/BiasIssueCard";
import { RecommendationCard } from "@/app/components/RecommendationCard";

export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const requestId = params.requestId as string;

  const [status, setStatus] = useState<AnalysisStatus | null>(null);
  const [results, setResults] = useState<BiasReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!requestId) {
      setError("Invalid request ID");
      setIsLoading(false);
      return;
    }

    let pollInterval: NodeJS.Timeout | null = null;

    const poll = async () => {
      try {
        const statusData = await checkStatus(requestId);
        setStatus(statusData);

        if (statusData.status === "complete") {
          // Fetch full results
          const resultsData = await getResults(requestId);
          setResults(resultsData);
          setIsLoading(false);

          // Stop polling
          if (pollInterval) {
            clearInterval(pollInterval);
          }
        } else if (statusData.status === "error") {
          setError(statusData.message || "Analysis failed");
          setIsLoading(false);

          // Stop polling
          if (pollInterval) {
            clearInterval(pollInterval);
          }
        }
      } catch (err: any) {
        setError(err.message || "Failed to fetch analysis status");
        setIsLoading(false);

        // Stop polling on error
        if (pollInterval) {
          clearInterval(pollInterval);
        }
      }
    };

    // Initial fetch
    poll();

    // Set up polling every 5 seconds
    pollInterval = setInterval(poll, 5000);

    // Cleanup
    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [requestId]);

  // Group bias issues by severity
  const groupedIssues = {
    critical:
      results?.bias_issues?.filter((i) => i.severity === "critical") || [],
    high: results?.bias_issues?.filter((i) => i.severity === "high") || [],
    medium: results?.bias_issues?.filter((i) => i.severity === "medium") || [],
    low: results?.bias_issues?.filter((i) => i.severity === "low") || [],
  };

  return (
    <div className="w-full min-h-screen relative bg-background pt-20 pb-16">
      <div className="w-full max-w-6xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-2 font-serif">
            Bias Analysis Results
          </h1>
          <p className="text-muted-foreground">
            Request ID:{" "}
            <code className="px-2 py-1 bg-muted rounded text-sm">
              {requestId}
            </code>
          </p>
        </div>

        {/* Error State */}
        {error && (
          <div className="mb-8 p-6 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start gap-3">
              <span className="text-red-600 text-2xl">⚠️</span>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-red-900 mb-1">
                  Analysis Failed
                </h3>
                <p className="text-sm text-red-700 mb-4">{error}</p>
                <button
                  onClick={() => router.push("/upload")}
                  className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors text-sm font-medium"
                >
                  Try Again
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Status Tracker (shown while processing) */}
        {isLoading && status && (
          <div className="mb-12">
            <StatusTracker
              currentStage={status.current_stage}
              agents={status.agents}
            />
            <div className="text-center mt-4">
              <LoadingSpinner size="md" text={status.message} />
            </div>
          </div>
        )}

        {/* Results Display */}
        {!isLoading && results && results.overall_score !== undefined && (
          <div className="space-y-8">
            {/* Overall Score */}
            <section>
              <ScoreCard score={results.overall_score} />
            </section>

            {/* Assessment Summary */}
            {results.assessment && (
              <section className="bg-white border border-gray-200 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-foreground mb-3">
                  Assessment Summary
                </h2>
                <p className="text-muted-foreground leading-relaxed">
                  {results.assessment}
                </p>
              </section>
            )}

            {/* Score Breakdown */}
            {results.score_breakdown && (
              <section className="bg-white border border-gray-200 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-foreground mb-4">
                  Score Breakdown
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-muted-foreground mb-1">
                      Text Analysis
                    </p>
                    <p className="text-2xl font-bold text-foreground">
                      {results.score_breakdown.text_score.toFixed(1)} / 10
                    </p>
                  </div>
                  <div className="p-4 bg-purple-50 rounded-lg">
                    <p className="text-sm text-muted-foreground mb-1">
                      Visual Analysis
                    </p>
                    <p className="text-2xl font-bold text-foreground">
                      {results.score_breakdown.visual_score.toFixed(1)} / 10
                    </p>
                  </div>
                  <div className="p-4 bg-orange-50 rounded-lg">
                    <p className="text-sm text-muted-foreground mb-1">
                      Intersectional Penalty
                    </p>
                    <p className="text-2xl font-bold text-foreground">
                      -
                      {results.score_breakdown.intersectional_penalty.toFixed(
                        2
                      )}
                    </p>
                  </div>
                </div>
              </section>
            )}

           

            {/* Top Concerns */}
            {results.top_concerns && results.top_concerns.length > 0 && (
              <section className="bg-orange-50 border border-orange-200 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-foreground mb-4 flex items-center gap-2">
                  <span>🎯</span>
                  Top Concerns
                </h2>
                <ul className="space-y-2">
                  {results.top_concerns.map((concern, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="text-orange-600 font-bold">
                        {index + 1}.
                      </span>
                      <span className="text-foreground">{concern}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {/* Bias Issues by Severity */}
            {results.bias_issues && results.bias_issues.length > 0 && (
              <section>
                <h2 className="text-2xl font-semibold text-foreground mb-6">
                  Detected Bias Issues
                </h2>

                {/* Critical Issues */}
                {groupedIssues.critical.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-red-600 mb-3 flex items-center gap-2">
                      <span>🚨</span>
                      Critical Issues ({groupedIssues.critical.length})
                    </h3>
                    <div className="space-y-4">
                      {groupedIssues.critical.map((issue, index) => (
                        <BiasIssueCard key={index} issue={issue} />
                      ))}
                    </div>
                  </div>
                )}

                {/* High Severity Issues */}
                {groupedIssues.high.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-orange-600 mb-3 flex items-center gap-2">
                      <span>⚠️</span>
                      High Severity Issues ({groupedIssues.high.length})
                    </h3>
                    <div className="space-y-4">
                      {groupedIssues.high.map((issue, index) => (
                        <BiasIssueCard key={index} issue={issue} />
                      ))}
                    </div>
                  </div>
                )}

                {/* Medium Severity Issues */}
                {groupedIssues.medium.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-yellow-600 mb-3 flex items-center gap-2">
                      <span>⚡</span>
                      Medium Severity Issues ({groupedIssues.medium.length})
                    </h3>
                    <div className="space-y-4">
                      {groupedIssues.medium.map((issue, index) => (
                        <BiasIssueCard key={index} issue={issue} />
                      ))}
                    </div>
                  </div>
                )}

                {/* Low Severity Issues */}
                {groupedIssues.low.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-blue-600 mb-3 flex items-center gap-2">
                      <span>ℹ️</span>
                      Low Severity Issues ({groupedIssues.low.length})
                    </h3>
                    <div className="space-y-4">
                      {groupedIssues.low.map((issue, index) => (
                        <BiasIssueCard key={index} issue={issue} />
                      ))}
                    </div>
                  </div>
                )}
              </section>
            )}

            {/* Recommendations */}
            {results.recommendations && results.recommendations.length > 0 && (
              <section>
                <h2 className="text-2xl font-semibold text-foreground mb-6">
                  Recommendations
                </h2>
                <div className="space-y-4">
                  {results.recommendations.map((recommendation, index) => (
                    <RecommendationCard
                      key={index}
                      recommendation={recommendation}
                      index={index}
                    />
                  ))}
                </div>
              </section>
            )}

            {/* Benchmark Comparison */}
            {results.benchmark_comparison && (
              <section className="bg-white border border-gray-200 rounded-lg p-6">
                <h2 className="text-xl font-semibold text-foreground mb-4">
                  Benchmark Comparison
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">
                      Average Score (Similar Ads)
                    </p>
                    <p className="text-3xl font-bold text-foreground">
                      {results.benchmark_comparison.average_score.toFixed(1)} /
                      10
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">
                      Your Percentile
                    </p>
                    <p className="text-3xl font-bold text-foreground">
                      {Math.round(results.benchmark_comparison.percentile)}th
                    </p>
                  </div>
                </div>
              </section>
            )}

            {/* Metadata */}
            <section className="bg-gray-50 border border-gray-200 rounded-lg p-6">
              <h2 className="text-xl font-semibold text-foreground mb-4">
                Analysis Metadata
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                {results.confidence !== undefined && (
                  <div>
                    <p className="text-muted-foreground mb-1">Confidence</p>
                    <p className="text-foreground font-medium">
                      {Math.round(results.confidence * 100)}%
                    </p>
                  </div>
                )}
                {results.processing_time_ms && (
                  <div>
                    <p className="text-muted-foreground mb-1">
                      Processing Time
                    </p>
                    <p className="text-foreground font-medium">
                      {(results.processing_time_ms / 1000).toFixed(2)}s
                    </p>
                  </div>
                )}
                <div>
                  <p className="text-muted-foreground mb-1">Analyzed At</p>
                  <p className="text-foreground font-medium">
                    {new Date(results.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            </section>

            {/* Action Buttons */}
            <section className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={() => router.push("/upload")}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
              >
                Analyze Another Ad
              </button>
              <button
                onClick={() => window.print()}
                className="px-6 py-3 bg-white border-2 border-gray-300 text-foreground rounded-lg font-medium hover:bg-gray-50 transition-colors"
              >
                Print Report
              </button>
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
