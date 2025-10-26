"use client";

import type { Recommendation } from "@/types/api";

interface RecommendationCardProps {
  recommendation: Recommendation;
  index: number;
}

export function RecommendationCard({
  recommendation,
  index,
}: RecommendationCardProps) {
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "critical":
        return "bg-red-500 text-white";
      case "high":
        return "bg-orange-500 text-white";
      case "medium":
        return "bg-yellow-500 text-white";
      case "low":
        return "bg-blue-500 text-white";
      default:
        return "bg-gray-500 text-white";
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case "critical":
        return "🚨";
      case "high":
        return "⚠️";
      case "medium":
        return "⚡";
      case "low":
        return "ℹ️";
      default:
        return "•";
    }
  };

  const formatCategory = (category: string) => {
    return category.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-all hover:border-primary/50">
      <div className="flex gap-4">
        {/* Priority Badge */}
        <div className="flex-shrink-0">
          <div
            className={`w-10 h-10 rounded-full ${getPriorityColor(
              recommendation.priority
            )} flex items-center justify-center font-bold text-sm`}
          >
            {index + 1}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1">
          {/* Header */}
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="text-xl">
                {getPriorityIcon(recommendation.priority)}
              </span>
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                {formatCategory(recommendation.category)}
              </span>
            </div>
            <span
              className={`px-2 py-1 text-xs font-medium rounded ${getPriorityColor(
                recommendation.priority
              )}`}
            >
              {recommendation.priority.toUpperCase()}
            </span>
          </div>

          {/* Action */}
          <h4 className="text-base font-semibold text-foreground mb-2">
            {recommendation.action}
          </h4>

          {/* Impact */}
          <p className="text-sm text-muted-foreground">
            {recommendation.impact}
          </p>
        </div>
      </div>
    </div>
  );
}
