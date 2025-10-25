"use client"

import { useState } from "react"

export default function UploadPage() {
  const [youtubeUrl, setYoutubeUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [error, setError] = useState("")

  const handleYoutubeSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setAnalysisResult(null)

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
      
      console.log("\n📤 Sending request to backend:")
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
        throw new Error(`Failed to analyze video (${response.status})`)
      }

      console.log("📦 Parsing JSON response...")
      const result = await response.json()
      console.log("✅ Response parsed successfully:")
      console.log(result)
      
      setAnalysisResult(result)
      console.log("✅ Analysis result set in state")
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
        setError("Cannot connect to backend server. Make sure it's running on http://localhost:8000")
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
                <div className="w-full max-w-[800px] mt-8 sm:mt-10">
                  <div className="bg-white shadow-[0px_0px_0px_0.75px_rgba(0,0,0,0.08)] overflow-hidden rounded-[6px] border border-border/50">
                    
                    {/* Overall Score Header */}
                    <div className="px-6 sm:px-8 py-6 border-b border-border bg-gradient-to-b from-background/50 to-background">
                      <div className="flex items-center justify-between">
                        <div>
                          <h2 className="text-foreground text-2xl sm:text-3xl font-bold font-serif mb-1">Bias Analysis</h2>
                          <p className="text-muted-foreground text-sm font-sans">{analysisResult.video_title || analysisResult.youtube_url}</p>
                        </div>
                        <div className="flex flex-col items-end">
                          <div className={`text-5xl sm:text-6xl font-bold font-serif ${getBiasScoreColor(analysisResult.bias_score)}`}>
                            {analysisResult.bias_score}
                          </div>
                          <div className="text-sm text-muted-foreground font-sans">/ 10</div>
                          <div className={`text-xs font-semibold mt-1 ${getBiasScoreColor(analysisResult.bias_score)}`}>
                            {getBiasScoreLabel(analysisResult.bias_score)}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Detailed Analysis Grid */}
                    <div className="grid md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-border">
                      
                      {/* Text Bias Analysis */}
                      {analysisResult.text_bias && (
                        <div className="p-6 sm:p-8">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="w-8 h-8 bg-accent/10 rounded-full flex items-center justify-center">
                              <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                              </svg>
                            </div>
                            <h3 className="text-foreground text-lg font-semibold font-sans">Text Analysis</h3>
                          </div>
                          
                          <div className="mb-4">
                            <div className="flex items-baseline gap-2">
                              <span className="text-3xl font-bold text-foreground font-serif">{analysisResult.text_bias.score}</span>
                              <span className="text-sm text-muted-foreground font-sans">/ 10</span>
                            </div>
                          </div>

                          {analysisResult.text_bias.issues && analysisResult.text_bias.issues.length > 0 && (
                            <div className="space-y-2">
                              {analysisResult.text_bias.issues.map((issue: string, idx: number) => (
                                <div key={idx} className="flex items-start gap-2">
                                  <div className="w-1.5 h-1.5 rounded-full bg-accent mt-2 flex-shrink-0"></div>
                                  <p className="text-sm text-foreground font-sans leading-relaxed">{issue}</p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Visual Bias Analysis */}
                      {analysisResult.visual_bias && (
                        <div className="p-6 sm:p-8">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="w-8 h-8 bg-accent/10 rounded-full flex items-center justify-center">
                              <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                              </svg>
                            </div>
                            <h3 className="text-foreground text-lg font-semibold font-sans">Visual Analysis</h3>
                          </div>
                          
                          <div className="mb-4">
                            <div className="flex items-baseline gap-2">
                              <span className="text-3xl font-bold text-foreground font-serif">{analysisResult.visual_bias.score}</span>
                              <span className="text-sm text-muted-foreground font-sans">/ 10</span>
                            </div>
                          </div>

                          {analysisResult.visual_bias.issues && analysisResult.visual_bias.issues.length > 0 && (
                            <div className="space-y-2">
                              {analysisResult.visual_bias.issues.map((issue: string, idx: number) => (
                                <div key={idx} className="flex items-start gap-2">
                                  <div className="w-1.5 h-1.5 rounded-full bg-accent mt-2 flex-shrink-0"></div>
                                  <p className="text-sm text-foreground font-sans leading-relaxed">{issue}</p>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Recommendations Section */}
                    {analysisResult.recommendations && analysisResult.recommendations.length > 0 && (
                      <div className="px-6 sm:px-8 py-6 border-t border-border bg-muted/30">
                        <div className="flex items-center gap-2 mb-4">
                          <div className="w-8 h-8 bg-accent/10 rounded-full flex items-center justify-center">
                            <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                            </svg>
                          </div>
                          <h3 className="text-foreground text-lg font-semibold font-sans">Recommendations</h3>
                        </div>
                        
                        <div className="space-y-3">
                          {analysisResult.recommendations.map((rec: string, idx: number) => (
                            <div key={idx} className="flex items-start gap-3 p-3 bg-white rounded-md border border-border/50">
                              <div className="w-5 h-5 bg-accent rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                <span className="text-white text-xs font-bold">{idx + 1}</span>
                              </div>
                              <p className="text-sm text-foreground font-sans leading-relaxed">{rec}</p>
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
