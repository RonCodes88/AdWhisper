"use client";

import type { BiasIssue } from "@/types/api";

interface BiasIssueCardProps {
  issue: BiasIssue;
}

export function BiasIssueCard({ issue }: BiasIssueCardProps) {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "bg-red-100 text-red-800 border-red-300";
      case "high":
        return "bg-orange-100 text-orange-800 border-orange-300";
      case "medium":
        return "bg-yellow-100 text-yellow-800 border-yellow-300";
      case "low":
        return "bg-blue-100 text-blue-800 border-blue-300";
      default:
        return "bg-gray-100 text-gray-800 border-gray-300";
    }
  };

  const formatCategory = (category: string) => {
    return category.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h4 className="text-lg font-semibold text-foreground mb-1">
            {formatCategory(issue.category)}
          </h4>
          <p className="text-sm text-muted-foreground">
            Source: {issue.source.replace("_", " ")}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <span
            className={`px-3 py-1 text-xs font-medium rounded-full border ${getSeverityColor(
              issue.severity
            )}`}
          >
            {issue.severity.toUpperCase()}
          </span>
          <span className="text-xs text-muted-foreground">
            {Math.round(issue.confidence * 100)}% confidence
          </span>
        </div>
      </div>

      {/* Description */}
      <p className="text-sm text-foreground/80 mb-4">{issue.description}</p>

      {/* Examples */}
      {issue.examples && issue.examples.length > 0 && (
        <div className="mt-4">
          <p className="text-xs font-medium text-muted-foreground mb-2">
            Examples:
          </p>
          <div className="flex flex-wrap gap-2">
            {issue.examples.map((example, index) => (
              <span
                key={index}
                className="px-3 py-1 bg-gray-100 text-gray-700 rounded-md text-sm font-mono"
              >
                &quot;{example}&quot;
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
