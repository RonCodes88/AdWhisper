"use client"

import { useState } from "react"

interface BiasIssue {
  category: string
  severity: string
  source: string
  description: string
  impact: string
  examples: string[]
  confidence: number
}

interface Recommendation {
  priority: string
  category: string
  action: string
  impact: string
}

interface AgentStatus {
  name: string
  status: 'pending' | 'processing' | 'completed' | 'error'
  score?: number
  issuesFound?: number
  message?: string
  logs?: string[]
}

interface BiasAnalysisReport {
  request_id: string
  content_url?: string
  content_type?: string
  overall_bias_score: number
  bias_level: string
  assessment: string
  score_breakdown: {
    text_score: number
    visual_score: number
    intersectional_penalty: number
    weighted_score: number
  }
  text_analysis_status: string
  visual_analysis_status: string
  total_issues: number
  critical_severity_count: number
  high_severity_count: number
  medium_severity_count: number
  low_severity_count: number
  bias_issues?: BiasIssue[]
  top_concerns?: string[]
  recommendations?: Recommendation[]
  confidence: number
  processing_time_ms: number
  timestamp: string
}

export default function UploadPage() {
  const [youtubeUrl, setYoutubeUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<BiasAnalysisReport | null>(null)
  const [error, setError] = useState("")
  const [requestId, setRequestId] = useState<string | null>(null)
  const [agentStatuses, setAgentStatuses] = useState<AgentStatus[]>([])
  const [showLogs, setShowLogs] = useState(true)

  // Helper to add log to agent
  const addAgentLog = (agentName: string, log: string) => {
    setAgentStatuses(prev => prev.map(agent => 
      agent.name === agentName
        ? { ...agent, logs: [...(agent.logs || []), log] }
        : agent
    ))
  }

  // No more polling or client-side aggregation needed!
  // Backend returns the complete report directly.

  const handleYoutubeSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setAnalysisResult(null)
    setRequestId(null)
    setAgentStatuses([])

    console.log("========================================")
    console.log("🎬 YouTube Analysis Submission Started")
    console.log("========================================")
    console.log("📝 YouTube URL:", youtubeUrl)

    // Validate YouTube URL
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/
    if (!youtubeRegex.test(youtubeUrl)) {
      console.log("❌ Validation failed: Invalid YouTube URL")
      setError("Please enter a valid YouTube URL")
      return
    }
    console.log("✅ URL validation passed")

    setIsLoading(true)
    console.log("⏳ Setting loading state to true")

    try {
      const apiUrl = "http://localhost:8000/api/analyze-youtube"
      const payload = { youtube_url: youtubeUrl }
      
      console.log("\n📤 Sending request to ingestion agent:")
      console.log("   URL:", apiUrl)
      console.log("   Method: POST")
      console.log("   Payload:", payload)
      console.log("   Timestamp:", new Date().toISOString())
      
      const startTime = Date.now()
      
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      })

      const responseTime = Date.now() - startTime
      console.log(`\n📥 Response received (${responseTime}ms)`)
      console.log("   Status:", response.status)
      console.log("   Status Text:", response.statusText)
      console.log("   OK:", response.ok)

      if (!response.ok) {
        console.log("❌ Response not OK")
        const errorText = await response.text()
        console.log("   Error body:", errorText)
        throw new Error(`Failed to submit video (${response.status})`)
      }

      console.log("📦 Parsing JSON response...")
      const result = await response.json()
      console.log("✅ Response parsed successfully:")
      console.log(result)
      
      // The result IS the final bias report! No need to poll!
      console.log("🎉 Complete bias report received from backend!")
      console.log(`📊 Overall Score: ${result.overall_bias_score || 'N/A'}`)
      console.log(`🏷️  Bias Level: ${result.bias_level || 'Unknown'}`)
      console.log(`📋 Total Issues: ${result.total_issues || 0}`)
      console.log(`💡 Recommendations: ${result.recommendations?.length || 0}`)
      
      const reqId = result.request_id
      setRequestId(reqId)
      
      // Safely extract scores with defaults
      const textScore = result.score_breakdown?.text_score ?? 5.0
      const visualScore = result.score_breakdown?.visual_score ?? 5.0
      const overallScore = result.overall_bias_score ?? 5.0
      const totalIssues = result.total_issues ?? 0
      
      // Set all agent statuses to completed
      setAgentStatuses([
        { name: 'Ingestion Agent', status: 'completed', message: 'Video processed ✓' },
        { name: 'Text Bias Agent', status: 'completed', score: textScore, message: `Text Score: ${textScore.toFixed(1)}/10` },
        { name: 'Visual Bias Agent', status: 'completed', score: visualScore, message: `Visual Score: ${visualScore.toFixed(1)}/10` },
        { name: 'Scoring Agent', status: 'completed', score: overallScore, issuesFound: totalIssues, message: `Final Score: ${overallScore.toFixed(1)}/10` }
      ])
      
      // Set the final report directly
      setAnalysisResult(result)
      console.log("✅ Final analysis result set in state")
      console.log("========================================\n")
      
    } catch (err) {
      console.log("\n❌ ERROR OCCURRED:")
      console.log("   Type:", err instanceof Error ? err.constructor.name : typeof err)
      console.log("   Message:", err instanceof Error ? err.message : String(err))
      console.log("   Full error:", err)
      console.log("========================================\n")
      
      const errorMessage = err instanceof Error ? err.message : "An error occurred while analyzing the video"
      setError(errorMessage)
      
      // Check if it's a network error
      if (err instanceof TypeError && err.message.includes("fetch")) {
        setError("Cannot connect to backend server. Make sure agents are running (ports 8000, 8101, 8102)")
      }
    } finally {
      console.log("🔄 Cleaning up - setting loading to false")
      setIsLoading(false)
    }
  }

  const getBiasScoreColor = (score: number) => {
    if (score >= 9) return "text-green-600"
    if (score >= 7) return "text-yellow-600"
    if (score >= 4) return "text-orange-600"
    return "text-red-600"
  }

  const getBiasScoreLabel = (score: number) => {
    if (score >= 9) return "Minimal Bias"
    if (score >= 7) return "Minor Bias"
    if (score >= 4) return "Moderate Bias"
    return "Significant Bias"
  }

  const getStatusColor = (status: string) => {
    if (status === 'completed') return 'bg-green-500'
    if (status === 'processing') return 'bg-yellow-500 animate-pulse'
    if (status === 'error') return 'bg-red-500'
    return 'bg-gray-400'
  }

  return (
    <div className="w-full min-h-screen relative bg-background overflow-x-hidden flex flex-col justify-start items-center">
      <div className="relative flex flex-col justify-start items-center w-full">
        <div className="w-full max-w-none px-4 sm:px-6 md:px-8 lg:px-0 lg:max-w-[1060px] lg:w-[1060px] relative flex flex-col justify-start items-start min-h-screen">
          {/* Border lines matching landing page */}
          <div className="h-full absolute left-4 sm:left-6 md:left-8 lg:left-0 top-0 bg-border shadow-[1px_0px_0px_white] z-0 w-px"></div>
          <div className="h-full absolute right-4 sm:right-6 md:right-8 lg:right-0 top-0 bg-border shadow-[1px_0px_0px_white] z-0 w-px"></div>

          <div className="self-stretch pt-[9px] overflow-hidden border-b border-border flex flex-col justify-center items-center gap-8 lg:gap-12 relative z-10">
            
            {/* Hero Section */}
            <div className="pt-16 sm:pt-20 md:pt-24 lg:pt-32 pb-8 sm:pb-12 md:pb-16 flex flex-col justify-start items-center px-2 sm:px-4 md:px-8 lg:px-0 w-full">
              <div className="w-full max-w-[800px] flex flex-col justify-center items-center gap-4 sm:gap-5 md:gap-6">
                <div className="text-center flex justify-center flex-col text-foreground text-[32px] sm:text-[42px] md:text-[52px] lg:text-[64px] font-bold leading-[1.1] font-serif">
                  Analyze your ad
                </div>
                <div className="w-full max-w-[600px] text-center flex justify-center flex-col text-muted-foreground text-base sm:text-lg md:text-xl leading-[1.5] font-sans font-medium">
                  Upload your YouTube ad link for instant bias detection and actionable recommendations
                </div>
              </div>

              {/* YouTube URL Input Card */}
              <div className="w-full max-w-[700px] mt-8 sm:mt-10 md:mt-12">
                <div className="bg-white shadow-[0px_0px_0px_0.75px_rgba(0,0,0,0.08)] overflow-hidden rounded-[6px] border border-border/50">
                  <form onSubmit={handleYoutubeSubmit} className="p-6 sm:p-8 space-y-5">
                    
                    <div className="flex items-center gap-3 pb-4 border-b border-border">
                      <div className="w-10 h-10 bg-red-50 rounded-full flex items-center justify-center">
                        <svg className="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
                        </svg>
                      </div>
                      <div className="flex flex-col">
                        <h2 className="text-foreground text-lg font-semibold leading-6 font-sans">YouTube Video</h2>
                        <p className="text-muted-foreground text-sm font-normal">Paste a link to analyze for bias</p>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label htmlFor="youtube-url" className="block text-sm font-medium text-foreground font-sans">
                        Video URL
                      </label>
                      <input
                        id="youtube-url"
                        type="text"
                        value={youtubeUrl}
                        onChange={(e) => setYoutubeUrl(e.target.value)}
                        placeholder="https://www.youtube.com/watch?v=..."
                        className="w-full px-4 py-3 border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent bg-background text-foreground font-sans text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        disabled={isLoading}
                      />
                    </div>

                    {error && (
                      <div className="bg-red-50 border border-red-200 rounded-md p-4">
                        <p className="text-red-800 text-sm font-medium">{error}</p>
                      </div>
                    )}

                    <button
                      type="submit"
                      disabled={isLoading || !youtubeUrl}
                      className="w-full h-12 px-8 py-2 relative bg-primary shadow-[0px_0px_0px_2.5px_rgba(255,255,255,0.08)_inset] overflow-hidden rounded-full flex justify-center items-center transition-all hover:shadow-[0px_0px_0px_2.5px_rgba(255,255,255,0.12)_inset] disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <div className="absolute inset-0 h-full w-full bg-gradient-to-b from-[rgba(255,255,255,0)] to-[rgba(0,0,0,0.10)] mix-blend-multiply pointer-events-none"></div>
                      <div className="flex items-center gap-2 text-primary-foreground text-[15px] font-medium leading-5 font-sans">
                        {isLoading ? (
                          <>
                            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                            Analyzing video...
                          </>
                        ) : (
                          <>
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                            </svg>
                            Analyze for bias
                          </>
                        )}
                      </div>
                    </button>
                  </form>
                </div>
              </div>


              {/* Analysis Results Section */}
              {analysisResult && (
                <div className="w-full max-w-[900px] mt-8 sm:mt-10">
                  <div className="bg-white shadow-[0px_0px_0px_0.75px_rgba(0,0,0,0.08)] overflow-hidden rounded-[6px] border border-border/50">
                    
                    {/* Overall Score Header */}
                    <div className="px-6 sm:px-8 py-6 border-b border-border bg-gradient-to-b from-background/50 to-background">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <h2 className="text-foreground text-2xl sm:text-3xl font-bold font-serif mb-2">Bias Analysis</h2>
                          <p className="text-muted-foreground text-sm font-sans break-all">{youtubeUrl}</p>
                        </div>
                        <div className="flex flex-col items-end flex-shrink-0">
                          <div className={`text-5xl sm:text-6xl font-bold font-serif ${getBiasScoreColor(analysisResult.overall_bias_score ?? 5.0)}`}>
                            {(analysisResult.overall_bias_score ?? 5.0).toFixed(1)}
                          </div>
                          <div className="text-sm text-muted-foreground font-sans">/ 10</div>
                          <div className={`text-sm font-semibold mt-1 ${getBiasScoreColor(analysisResult.overall_bias_score ?? 5.0)}`}>
                            {analysisResult.bias_level}
                          </div>
                        </div>
                      </div>
                      
                      {/* Assessment */}
                      <div className="mt-4 p-4 bg-muted/30 rounded-md">
                        <p className="text-sm text-foreground font-sans leading-relaxed">{analysisResult.assessment}</p>
                      </div>
                    </div>

                    {/* Detailed Analysis Grid */}
                    <div className="grid md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-border">
                      
                      {/* Text Bias Analysis */}
                      <div className="p-6 sm:p-8">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-8 h-8 bg-yellow-50 rounded-full flex items-center justify-center">
                            <svg className="w-4 h-4 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                          </div>
                          <h3 className="text-foreground text-lg font-semibold font-sans">Text Analysis</h3>
                        </div>
                        
                        <div className="mb-3">
                          <div className="flex items-baseline gap-2">
                            <span className="text-4xl font-bold text-foreground font-serif">{(analysisResult.score_breakdown?.text_score ?? 5.0).toFixed(1)}</span>
                            <span className="text-sm text-muted-foreground font-sans">/ 10</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 text-sm">
                              <div className="w-2 h-2 rounded-full bg-green-500"></div>
                          <p className="text-muted-foreground font-sans">Analysis complete</p>
                        </div>
                      </div>

                      {/* Visual Bias Analysis */}
                      <div className="p-6 sm:p-8">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-8 h-8 bg-yellow-50 rounded-full flex items-center justify-center">
                            <svg className="w-4 h-4 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                          </div>
                          <h3 className="text-foreground text-lg font-semibold font-sans">Visual Analysis</h3>
                        </div>
                        
                        <div className="mb-3">
                          <div className="flex items-baseline gap-2">
                            <span className="text-4xl font-bold text-foreground font-serif">{(analysisResult.score_breakdown?.visual_score ?? 5.0).toFixed(1)}</span>
                            <span className="text-sm text-muted-foreground font-sans">/ 10</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-2 text-sm">
                              <div className="w-2 h-2 rounded-full bg-green-500"></div>
                          <p className="text-muted-foreground font-sans">Analysis complete</p>
                        </div>
                      </div>
                    </div>

                    {/* Top Concerns Section */}
                    {analysisResult.top_concerns && analysisResult.top_concerns.length > 0 && (
                      <div className="px-6 sm:px-8 py-6 border-t border-border bg-red-50/50">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                            <svg className="w-4 h-4 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                            </svg>
                          </div>
                          <h3 className="text-foreground text-lg font-semibold font-sans">Top Concerns</h3>
                        </div>
                        
                        <div className="space-y-3">
                          {analysisResult.top_concerns.map((concern: string, idx: number) => (
                            <div key={idx} className="flex items-start gap-3 p-4 bg-white rounded-md border border-red-200">
                              <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                </svg>
                              </div>
                              <p className="text-sm text-foreground font-sans leading-relaxed flex-1">{concern}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Bias Issues Details Section */}
                    {analysisResult.bias_issues && analysisResult.bias_issues.length > 0 && (
                      <div className="px-6 sm:px-8 py-6 border-t border-border">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
                            <svg className="w-4 h-4 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                            </svg>
                          </div>
                          <h3 className="text-foreground text-lg font-semibold font-sans">Detailed Bias Findings</h3>
                          <span className="ml-auto text-sm text-muted-foreground font-medium">
                            {analysisResult.bias_issues.length} issue{analysisResult.bias_issues.length !== 1 ? 's' : ''} found
                          </span>
                        </div>
                        
                        <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
                          {analysisResult.bias_issues.map((issue: BiasIssue, idx: number) => (
                            <div key={idx} className="p-4 bg-muted/50 rounded-md border border-border">
                              <div className="flex items-start justify-between gap-3 mb-2">
                                <div className="flex items-center gap-2">
                                  <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                    issue.severity === 'critical' ? 'bg-red-100 text-red-700' :
                                    issue.severity === 'high' ? 'bg-orange-100 text-orange-700' :
                                    issue.severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                                    'bg-blue-100 text-blue-700'
                                  }`}>
                                    {issue.severity.toUpperCase()}
                                  </span>
                                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                                    {issue.category.replace('_', ' ')}
                                  </span>
                                </div>
                                <span className="text-xs text-muted-foreground">
                                  {issue.source === 'text_bias_agent' ? '📝 Text' : '👁️ Visual'}
                                </span>
                              </div>
                              <p className="text-sm text-foreground font-sans leading-relaxed mb-2">
                                {issue.description}
                              </p>
                              {issue.examples && issue.examples.length > 0 && (
                                <div className="mt-2 p-2 bg-white/50 rounded border border-border/50">
                                  <p className="text-xs font-medium text-muted-foreground mb-1">Examples:</p>
                                  {issue.examples.map((example, exIdx) => (
                                    <p key={exIdx} className="text-xs text-foreground font-mono italic pl-2 border-l-2 border-border mb-1">
                                      &ldquo;{example}&rdquo;
                                    </p>
                                  ))}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Recommendations Section */}
                    {analysisResult.recommendations && analysisResult.recommendations.length > 0 && (
                      <div className="px-6 sm:px-8 py-6 border-t border-border bg-green-50/30">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                            <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                          </div>
                          <h3 className="text-foreground text-lg font-semibold font-sans">Actionable Recommendations</h3>
                        </div>
                        
                        <div className="space-y-3">
                          {analysisResult.recommendations.map((rec: Recommendation, idx: number) => (
                            <div key={idx} className="p-4 bg-white rounded-md border border-green-200 shadow-sm">
                              <div className="flex items-start gap-3">
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                                  rec.priority === 'critical' || rec.priority === 'high' ? 'bg-red-500' :
                                  rec.priority === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                                }`}>
                                <span className="text-white text-sm font-bold">{idx + 1}</span>
                                </div>
                                <div className="flex-1">
                                  <div className="flex items-center gap-2 mb-1">
                                    <h4 className="text-sm font-semibold text-foreground">{rec.category}</h4>
                                    <span className={`text-xs px-2 py-0.5 rounded font-medium ${
                                      rec.priority === 'critical' || rec.priority === 'high' ? 'bg-red-100 text-red-700' :
                                      rec.priority === 'medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-green-100 text-green-700'
                                    }`}>
                                      {rec.priority}
                                    </span>
                                  </div>
                                  <p className="text-sm text-foreground font-sans leading-relaxed mb-2">
                                    <span className="font-medium">Action:</span> {rec.action}
                                  </p>
                                  <p className="text-xs text-muted-foreground italic">
                                    <span className="font-medium">Impact:</span> {rec.impact}
                                  </p>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
