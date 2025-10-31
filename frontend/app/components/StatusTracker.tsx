"use client";

interface AgentStatus {
  name: string;
  status: "pending" | "processing" | "complete" | "error";
  message?: string;
  timestamp?: string;
}

interface StatusTrackerProps {
  currentStage:
    | "submitted"
    | "ingestion"
    | "analyzing_text"
    | "analyzing_visual"
    | "scoring"
    | "complete"
    | "error";
  agents?: {
    ingestion_agent?: AgentStatus;
    text_bias_agent?: AgentStatus;
    visual_bias_agent?: AgentStatus;
    scoring_agent?: AgentStatus;
  };
}

const agentStages = [
  { id: "submitted", label: "Submitted", icon: "📤", agentKey: null },
  {
    id: "ingestion",
    label: "Ingestion Agent",
    icon: "📥",
    agentKey: "ingestion_agent",
  },
  {
    id: "analyzing",
    label: "Bias Analysis",
    icon: "🔍",
    agentKey: "parallel",
    subAgents: [
      { key: "text_bias_agent", label: "Text Bias", icon: "📝" },
      { key: "visual_bias_agent", label: "Visual Bias", icon: "👁️" },
    ],
  },
  {
    id: "scoring",
    label: "Scoring Agent",
    icon: "⚖️",
    agentKey: "scoring_agent",
  },
  { id: "complete", label: "Complete", icon: "✅", agentKey: null },
];

export function StatusTracker({ currentStage, agents }: StatusTrackerProps) {
  const getStageIndex = (stage: string) => {
    switch (stage) {
      case "submitted":
        return 0;
      case "ingestion":
        return 1;
      case "analyzing_text":
      case "analyzing_visual":
        return 2;
      case "scoring":
        return 3;
      case "complete":
        return 4;
      case "error":
        return -1;
      default:
        return 0;
    }
  };

  const currentIndex = getStageIndex(currentStage);

  const getAgentStatus = (key: string): AgentStatus | undefined => {
    return agents?.[key as keyof typeof agents];
  };

  return (
    <div className="w-full max-w-4xl mx-auto py-8">
      {/* Error State */}
      {currentStage === "error" && (
        <div className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
          <div className="flex items-center gap-2 text-destructive">
            <span className="text-2xl">⚠️</span>
            <p className="font-semibold">Analysis Error</p>
          </div>
        </div>
      )}

      <div className="flex justify-between items-start relative">
        {/* Progress Line */}
        <div className="absolute top-5 left-0 right-0 h-0.5 bg-gray-200 z-0" />
        <div
          className="absolute top-5 left-0 h-0.5 bg-primary z-0 transition-all duration-500"
          style={{
            width: `${(currentIndex / (agentStages.length - 1)) * 100}%`,
          }}
        />

        {/* Stage Markers */}
        {agentStages.map((stage, index) => {
          const isActive = index === currentIndex;
          const isComplete = index < currentIndex;
          const isPending = index > currentIndex;

          // Get agent status for this stage
          const agentStatus =
            stage.agentKey && stage.agentKey !== "parallel"
              ? getAgentStatus(stage.agentKey)
              : null;

          return (
            <div
              key={stage.id}
              className="flex flex-col items-center z-10 flex-1"
            >
              {/* Main Stage */}
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center text-xl transition-all duration-300 ${
                  isActive
                    ? "bg-primary text-primary-foreground scale-110 shadow-lg ring-4 ring-primary/20"
                    : isComplete
                    ? "bg-primary/80 text-primary-foreground"
                    : "bg-gray-200 text-gray-400"
                }`}
              >
                {stage.icon}
              </div>

              <p
                className={`mt-2 text-xs font-semibold text-center ${
                  isActive
                    ? "text-foreground"
                    : isPending
                    ? "text-muted-foreground"
                    : "text-foreground/70"
                }`}
              >
                {stage.label}
              </p>

              {/* Agent Status Message */}
              {agentStatus && isActive && (
                <p className="mt-1 text-[10px] text-muted-foreground text-center max-w-[100px] truncate">
                  {agentStatus.message || agentStatus.status}
                </p>
              )}

              {/* Sub-agents for parallel processing */}
              {stage.subAgents && index === currentIndex && (
                <div className="mt-4 flex flex-col gap-2 min-w-[180px]">
                  {stage.subAgents.map((subAgent) => {
                    const subStatus = getAgentStatus(subAgent.key);
                    const subActive = subStatus?.status === "processing";
                    const subComplete = subStatus?.status === "complete";
                    const subError = subStatus?.status === "error";

                    return (
                      <div
                        key={subAgent.key}
                        className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
                          subActive
                            ? "bg-primary/10 border border-primary/30"
                            : subComplete
                            ? "bg-primary/5 border border-primary/20"
                            : subError
                            ? "bg-destructive/10 border border-destructive/30"
                            : "bg-gray-100 border border-gray-200"
                        }`}
                      >
                        <span className="text-sm">{subAgent.icon}</span>
                        <div className="flex-1 min-w-0">
                          <span
                            className={`text-xs font-medium block ${
                              subActive
                                ? "text-primary"
                                : subComplete
                                ? "text-primary/70"
                                : subError
                                ? "text-destructive"
                                : "text-muted-foreground"
                            }`}
                          >
                            {subAgent.label}
                          </span>
                          {subStatus?.message && (
                            <span className="text-[10px] text-muted-foreground block truncate">
                              {subStatus.message}
                            </span>
                          )}
                        </div>
                        {subActive && (
                          <div className="ml-auto flex gap-0.5">
                            <div className="w-1 h-1 bg-primary rounded-full animate-pulse" />
                            <div
                              className="w-1 h-1 bg-primary rounded-full animate-pulse"
                              style={{ animationDelay: "150ms" }}
                            />
                            <div
                              className="w-1 h-1 bg-primary rounded-full animate-pulse"
                              style={{ animationDelay: "300ms" }}
                            />
                          </div>
                        )}
                        {subComplete && (
                          <span className="ml-auto text-xs text-primary">
                            ✓
                          </span>
                        )}
                        {subError && (
                          <span className="ml-auto text-xs text-destructive">
                            ✗
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Active indicator for non-parallel stages */}
              {isActive && !stage.subAgents && (
                <div className="mt-2">
                  <div className="flex gap-1">
                    <div
                      className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce"
                      style={{ animationDelay: "0ms" }}
                    />
                    <div
                      className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce"
                      style={{ animationDelay: "150ms" }}
                    />
                    <div
                      className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce"
                      style={{ animationDelay: "300ms" }}
                    />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Architecture Info */}
      {currentIndex >= 1 && currentIndex <= 3 && (
        <div className="mt-8 p-4 bg-muted/50 rounded-lg border border-border">
          <p className="text-xs text-muted-foreground text-center">
            <span className="font-semibold">Multi-Agent System:</span> Each
            agent specializes in detecting specific bias types using
            RAG-enhanced analysis
          </p>
        </div>
      )}
    </div>
  );
}
