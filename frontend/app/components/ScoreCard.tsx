"use client";

import { getScoreInterpretation } from "@/types/api";
import { useEffect, useState } from "react";

interface ScoreCardProps {
  score: number;
  showAnimation?: boolean;
}

export function ScoreCard({ score, showAnimation = true }: ScoreCardProps) {
  const [displayScore, setDisplayScore] = useState(showAnimation ? 0 : score);
  const interpretation = getScoreInterpretation(score);

  useEffect(() => {
    if (showAnimation) {
      const duration = 1500; // 1.5 seconds
      const steps = 60;
      const increment = score / steps;
      let current = 0;

      const timer = setInterval(() => {
        current += increment;
        if (current >= score) {
          setDisplayScore(score);
          clearInterval(timer);
        } else {
          setDisplayScore(current);
        }
      }, duration / steps);

      return () => clearInterval(timer);
    }
  }, [score, showAnimation]);

  const getColorClass = () => {
    switch (interpretation.level) {
      case "critical":
        return "text-red-600 border-red-600";
      case "warning":
        return "text-orange-600 border-orange-600";
      case "caution":
        return "text-yellow-600 border-yellow-600";
      case "good":
        return "text-green-600 border-green-600";
      case "excellent":
        return "text-blue-600 border-blue-600";
      default:
        return "text-gray-600 border-gray-600";
    }
  };

  const getBgColorClass = () => {
    switch (interpretation.level) {
      case "critical":
        return "bg-red-50";
      case "warning":
        return "bg-orange-50";
      case "caution":
        return "bg-yellow-50";
      case "good":
        return "bg-green-50";
      case "excellent":
        return "bg-blue-50";
      default:
        return "bg-gray-50";
    }
  };

  const circumference = 2 * Math.PI * 70; // radius = 70
  const strokeDashoffset = circumference - (displayScore / 10) * circumference;

  return (
    <div
      className={`${getBgColorClass()} rounded-lg p-8 border-2 ${getColorClass()} transition-all duration-300`}
    >
      <div className="flex flex-col md:flex-row items-center justify-between gap-6">
        {/* Circular Score Display */}
        <div className="relative w-40 h-40">
          <svg
            className="w-full h-full transform -rotate-90"
            viewBox="0 0 160 160"
          >
            {/* Background circle */}
            <circle
              cx="80"
              cy="80"
              r="70"
              stroke="currentColor"
              strokeWidth="12"
              fill="none"
              className="text-gray-200"
            />
            {/* Progress circle */}
            <circle
              cx="80"
              cy="80"
              r="70"
              stroke="currentColor"
              strokeWidth="12"
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              className={`${getColorClass()} transition-all duration-1000 ease-out`}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`text-4xl font-bold ${getColorClass()}`}>
              {displayScore.toFixed(1)}
            </span>
            <span className="text-sm text-muted-foreground">/ 10</span>
          </div>
        </div>

        {/* Score Interpretation */}
        <div className="flex-1 text-center md:text-left">
          <div className="flex items-center justify-center md:justify-start gap-2 mb-2">
            <span className="text-3xl">{interpretation.icon}</span>
            <h3 className={`text-2xl font-bold ${getColorClass()}`}>
              {interpretation.message}
            </h3>
          </div>
          <p className="text-lg text-foreground/80 mb-1">
            {interpretation.action}
          </p>
          <div className="mt-4 inline-block px-4 py-2 bg-white rounded-full text-sm font-medium shadow-sm">
            Score: {score.toFixed(1)} / 10
          </div>
        </div>
      </div>
    </div>
  );
}
