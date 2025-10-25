"use client"

import { useState } from "react"

export default function UploadPage() {
  const [dragActive, setDragActive] = useState(false)
  const [youtubeUrl, setYoutubeUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [error, setError] = useState("")

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      // Handle file upload here
      console.log("File dropped:", e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      // Handle file upload here
      console.log("File selected:", e.target.files[0])
    }
  }

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

  return (
    <div className="w-full min-h-screen relative bg-background flex flex-col justify-start items-center pt-20">
      <div className="w-full max-w-4xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4 font-serif">
            Analyze Your Ad Content
          </h1>
          <p className="text-muted-foreground text-lg">
            Upload ad content or paste a YouTube link for bias analysis
          </p>
        </div>

        {/* YouTube URL Input Section */}
        <div className="mb-8">
          <div className="bg-white border border-border rounded-lg p-6 shadow-sm">
            <h2 className="text-xl font-semibold text-foreground mb-4 flex items-center gap-2">
              <svg className="w-6 h-6 text-red-600" fill="currentColor" viewBox="0 0 24 24">
                <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
              </svg>
              YouTube Video Analysis
            </h2>
            
            <form onSubmit={handleYoutubeSubmit} className="space-y-4">
              <div>
                <label htmlFor="youtube-url" className="block text-sm font-medium text-foreground mb-2">
                  YouTube URL
                </label>
                <input
                  id="youtube-url"
                  type="text"
                  value={youtubeUrl}
                  onChange={(e) => setYoutubeUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="w-full px-4 py-3 border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/50 bg-background text-foreground"
                  disabled={isLoading}
                />
              </div>

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <p className="text-red-800 text-sm">{error}</p>
                </div>
              )}

              <button
                type="submit"
                disabled={isLoading || !youtubeUrl}
                className="w-full px-6 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isLoading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Analyzing Video...
                  </>
                ) : (
                  "Analyze YouTube Video"
                )}
              </button>
            </form>
          </div>
        </div>

        {/* Analysis Results Section */}
        {analysisResult && (
          <div className="bg-white border border-border rounded-lg p-6 shadow-sm mb-8">
            <h2 className="text-xl font-semibold text-foreground mb-4">Analysis Results</h2>
            
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-medium text-foreground mb-2">Overall Bias Score</h3>
                <div className="flex items-center gap-4">
                  <div className="text-4xl font-bold text-primary">{analysisResult.bias_score}</div>
                  <div className="text-sm text-muted-foreground">/ 10</div>
                </div>
              </div>

              {analysisResult.text_bias && (
                <div>
                  <h3 className="text-lg font-medium text-foreground mb-2">Text Bias Analysis</h3>
                  <p className="text-sm text-muted-foreground mb-2">Score: {analysisResult.text_bias.score}</p>
                  {analysisResult.text_bias.issues && analysisResult.text_bias.issues.length > 0 && (
                    <ul className="list-disc list-inside space-y-1">
                      {analysisResult.text_bias.issues.map((issue: string, idx: number) => (
                        <li key={idx} className="text-sm text-foreground">{issue}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {analysisResult.visual_bias && (
                <div>
                  <h3 className="text-lg font-medium text-foreground mb-2">Visual Bias Analysis</h3>
                  <p className="text-sm text-muted-foreground mb-2">Score: {analysisResult.visual_bias.score}</p>
                  {analysisResult.visual_bias.issues && analysisResult.visual_bias.issues.length > 0 && (
                    <ul className="list-disc list-inside space-y-1">
                      {analysisResult.visual_bias.issues.map((issue: string, idx: number) => (
                        <li key={idx} className="text-sm text-foreground">{issue}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {analysisResult.recommendations && analysisResult.recommendations.length > 0 && (
                <div>
                  <h3 className="text-lg font-medium text-foreground mb-2">Recommendations</h3>
                  <ul className="list-disc list-inside space-y-1">
                    {analysisResult.recommendations.map((rec: string, idx: number) => (
                      <li key={idx} className="text-sm text-foreground">{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Divider */}
        <div className="relative my-8">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-border"></div>
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-4 bg-background text-muted-foreground">or upload files directly</span>
          </div>
        </div>

        {/* File Upload Section */}
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50"
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            multiple
            accept="image/*,video/*,.txt,.doc,.docx"
            onChange={handleChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          
          <div className="space-y-4">
            <div className="w-16 h-16 mx-auto bg-muted rounded-full flex items-center justify-center">
              <svg
                className="w-8 h-8 text-muted-foreground"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
            </div>
            
            <div>
              <p className="text-lg font-medium text-foreground mb-2">
                {dragActive ? "Drop files here" : "Drag and drop files here"}
              </p>
              <p className="text-sm text-muted-foreground">
                or click to browse files
              </p>
            </div>
            
            <div className="text-xs text-muted-foreground">
              Supports: Images, Videos, Text files (TXT, DOC, DOCX)
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
