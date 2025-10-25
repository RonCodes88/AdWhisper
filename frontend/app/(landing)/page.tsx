"use client"

import type React from "react"

import { useState, useEffect, useRef } from "react"
import Link from "next/link"
import CTASection from "../components/cta-section"

function Badge({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="px-[14px] py-[6px] bg-white shadow-[0px_0px_0px_4px_rgba(55,50,47,0.05)] overflow-hidden rounded-[90px] flex justify-start items-center gap-[8px] border border-accent/20 shadow-xs">
      <div className="w-[14px] h-[14px] relative overflow-hidden flex items-center justify-center">{icon}</div>
      <div className="text-center flex justify-center flex-col text-foreground text-xs font-medium leading-3 font-sans">
        {text}
      </div>
    </div>
  )
}

function FeatureCard({
  title,
  description,
  isActive,
  progress,
  onClick,
}: {
  title: string
  description: string
  isActive: boolean
  progress: number
  onClick: () => void
}) {
  return (
    <div
      className={`w-full px-6 md:px-6 py-8 md:py-9 overflow-hidden flex flex-col justify-start items-start cursor-pointer relative rounded-lg border min-h-[180px] md:min_h-[200px] transition-shadow ${
        isActive
          ? "bg-white border-primary/30 shadow-[0_0_0_1px_rgba(0,0,0,0.08)_inset]"
          : "bg-white border-border hover:shadow-sm"
      }`}
      onClick={onClick}
    >
      <div className="self-stretch flex justify-center flex-col text-foreground text-sm md:text-sm font-semibold leading-6 md:leading-6 font-sans mb-2">
        {title}
      </div>
      <div className="self-stretch text-muted-foreground text-[13px] md:text-[13px] font-normal leading-[22px] md:leading-[22px] font-sans">
        {description}
      </div>

      {isActive && (
        <div className="absolute bottom-0 left-0 w-full h-0.5 bg-border">
          <div className="h-full bg-primary transition-all duration-100 ease-linear" style={{ width: `${progress}%` }} />
        </div>
      )}
    </div>
  )
}

export default function LandingPage() {
  const [activeCard, setActiveCard] = useState(0)
  const [progress, setProgress] = useState(0)
  const mountedRef = useRef(true)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    // Reset mounted ref when component mounts
    mountedRef.current = true
    
    const progressInterval = setInterval(() => {
      if (!mountedRef.current) return

      setProgress((prev) => {
        if (prev >= 100) {
          if (mountedRef.current) {
            setActiveCard((current) => (current + 1) % 3)
          }
          return 0
        }
        return prev + 2
      })
    }, 100)
    
    intervalRef.current = progressInterval

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      mountedRef.current = false
    }
  }, [])

  const handleCardClick = (index: number) => {
    if (!mountedRef.current) return
    setActiveCard(index)
    setProgress(0)
  }

  return (
    <div className="w-full min-h-screen relative bg-background overflow-x-hidden flex flex-col justify-start items-center">
      
      {/* Gradient Splotches - scroll with page */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden w-full">
        {/* Secondary (coral/orange) gradient splotch - top left */}
        <div 
          className="absolute top-32 left-[-100px] w-[600px] h-[600px] rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(252, 211, 77, 0.5) 0%, rgba(251, 146, 60, 0.25) 40%, transparent 70%)',
            filter: 'blur(100px)'
          }}
        ></div>
        
        {/* Secondary (yellow/orange) gradient splotch - top right */}
        <div 
          className="absolute top-20 right-[-150px] w-[550px] h-[550px] rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(252, 211, 77, 0.45) 0%, rgba(251, 146, 60, 0.22) 40%, transparent 70%)',
            filter: 'blur(90px)'
          }}
        ></div>
        
        {/* Primary (dark) gradient splotch - center left */}
        <div 
          className="absolute top-[40%] left-[-80px] w-[450px] h-[450px] rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(56, 56, 58, 0.12) 0%, rgba(56, 56, 58, 0.06) 40%, transparent 70%)',
            filter: 'blur(70px)'
          }}
        ></div>
        
        {/* Secondary (coral) gradient splotch - center */}
        <div 
          className="absolute top-[50%] left-[50%] transform -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(251, 146, 60, 0.35) 0%, rgba(252, 211, 77, 0.18) 40%, transparent 70%)',
            filter: 'blur(110px)'
          }}
        ></div>
        
        {/* Secondary (yellow) gradient splotch - bottom right */}
        <div 
          className="absolute top-[80%] right-[-100px] w-[500px] h-[500px] rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(252, 211, 77, 0.4) 0%, rgba(251, 146, 60, 0.2) 40%, transparent 70%)',
            filter: 'blur(85px)'
          }}
        ></div>
      </div>

      <div className="relative flex flex-col justify-start items-center w-full">
        <div className="w-full max-w-none px-4 sm:px-6 md:px-8 lg:px-0 lg:max-w-[1060px] lg:w-[1060px] relative flex flex-col justify-start items-start min-h-screen">
          <div className="h-full absolute left-4 sm:left-6 md:left-8 lg:left-0 top-0 bg-border shadow-[1px_0px_0px_white] z-0"></div>
          <div className="h-full absolute right-4 sm:right-6 md:right-8 lg:right-0 top-0 bg-border shadow-[1px_0px_0px_white] z-0"></div>

          <div className="self-stretch pt-[9px] overflow-hidden border-b border-border flex flex-col justify-center items-center gap-4 sm:gap-6 md:gap-8 lg:gap-[66px] relative z-10">

            {/* Hero Section */}
            <div className="pt-16 sm:pt-20 md:pt-24 lg:pt-[216px] pb-8 sm:pb-12 md:pb-16 flex flex-col justify-start items-center px-2 sm:px-4 md:px-8 lg:px-0 w-full sm:pl-0 sm:pr-0 pl-0 pr-0 relative">

              <div className="w-full max-w-[937px] lg:w-[937px] flex flex-col justify-center items-center gap-3 sm:gap-4 md:gap-5 lg:gap-6 relative z-10">
                <div className="self-stretch rounded-[3px] flex flex-col justify-center items-center gap-4 sm:gap-5 md:gap-6 lg:gap-8">
                  <div className="w-full max-w-[748.71px] lg:w-[748.71px] text-center flex justify-center flex-col text-foreground text-[24px] xs:text-[28px] sm:text-[36px] md:text-[52px] lg:text-[80px] font-bold leading-[1.1] sm:leading-[1.15] md:leading-[1.2] lg:leading-24 font-serif px-2 sm:px-4 md:px-0">
                    <span className="block animate-pop-in [animation-duration:1600ms] [animation-delay:200ms]">Detect bias in ads</span>
                    <span className="block animate-pop-in [animation-duration:1600ms] [animation-delay:650ms]">before they go live</span>
                  </div>
                  <div className="w-full max-w-[720px] lg:w-[720px] text-center flex justify-center flex-col text-muted-foreground sm:text-lg md:text-xl leading-[1.5] sm:leading-[1.55] md:leading-[1.6] font-sans px-4 sm:px-6 md:px-0 lg:text-lg font-medium text-sm text-pretty animate-fade-up [animation-delay:650ms]">
                    AI-powered multi-agent system that scans ad scripts and visuals for subconscious bias, suggests inclusive rewrites, and delivers audit reports.
                  </div>
                </div>
              </div>

              <div className="w-full max-w-[497px] lg:w-[497px] flex flex-col justify-center items-center gap-6 sm:gap-8 md:gap-10 lg:gap-12 relative z-10 mt-6 sm:mt-8 md:mt-10 lg:mt-12">
                <div className="backdrop-blur-[8.25px] flex justify-start items-center gap-4">
                  <Link href="/upload">
                    <button className="h-10 sm:h-11 md:h-12 px-6 sm:px-8 md:px-10 lg:px-12 py-2 sm:py-[6px] relative bg-primary shadow-[0px_0px_0px_2.5px_rgba(255,255,255,0.08)_inset] overflow-hidden rounded-full flex justify-center items-center cursor-pointer hover:opacity-90 transition-opacity">
                      <div className="absolute inset-0 h-full w-full bg-gradient-to-b from-[rgba(255,255,255,0)] to-[rgba(0,0,0,0.10)] mix-blend-multiply pointer-events-none"></div>
                      <div className="flex flex-col justify-center text-primary-foreground text-sm sm:text-base md:text-[15px] font-medium leading-5 font-sans">
                        Try for free
                      </div>
                    </button>
                  </Link>
                </div>
              </div>

              <div className="absolute top-[232px] sm:top-[248px] md:top-[264px] lg:top-[320px] left-1/2 transform -translate-x-1/2 z-0 pointer-events-none">
                <img
                  src="/mask-group-pattern.svg"
                  alt=""
                  className="w-[936px] sm:w-[1404px] md:w-[2106px] lg:w-[2808px] h-auto opacity-30 sm:opacity-40 md:opacity-50 mix-blend-multiply"
                  style={{
                    filter: "hue-rotate(200deg) saturate(0.5) brightness(1.3)",
                  }}
                />
              </div>

              <div className="w-full max-w-[960px] lg:w-[960px] pt-2 sm:pt-4 pb-6 sm:pb-8 md:pb-10 px-2 sm:px-4 md:px-6 lg:px-11 flex flex-col justify-center items-center gap-2 relative z-5 my-8 sm:my-12 md:my-16 lg:my-16 mb-0 lg:pb-0">
                <div className="w-full max-w-[900px] lg:w-[900px] h-[220px] sm:h-[300px] md:h-[480px] lg:h-[620px] bg-white shadow-[0px_0px_0px_0.9056603908538818px_rgba(0,0,0,0.08)] overflow-hidden rounded-[6px] sm:rounded-[8px] lg:rounded-[9.06px] flex flex-col justify-start items-start">
                  <div className="self-stretch flex-1 flex justify-start items-start">
                    <div className="w-full h-full flex items-center justify-center">
                      <div className="relative w-full h-full overflow-hidden">
                        <div
                          className={`absolute inset-0 transition-all duration-500 ease-in-out ${
                            activeCard === 0 ? "opacity-100 scale-100 blur-0" : "opacity-0 scale-95 blur-sm"
                          }`}
                        >
                          <img
                            src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/dsadsadsa.jpg-xTHS4hGwCWp2H5bTj8np6DXZUyrxX7.jpeg"
                            alt="Text Bias Analysis Dashboard"
                            className="w-full h-full object-cover"
                          />
                        </div>

                        <div
                          className={`absolute inset-0 transition-all duration-500 ease-in-out ${
                            activeCard === 1 ? "opacity-100 scale-100 blur-0" : "opacity-0 scale-95 blur-sm"
                          }`}
                        >
                          <img
                            src="/analytics-dashboard-with-charts-graphs-and-data-vi.jpg"
                            alt="Visual Bias Detection"
                            className="w-full h-full object-cover"
                          />
                        </div>

                        <div
                          className={`absolute inset-0 transition-all duration-500 ease-in-out ${
                            activeCard === 2 ? "opacity-100 scale-100 blur-0" : "opacity-0 scale-95 blur-sm"
                          }`}
                        >
                          <img
                            src="/data-visualization-dashboard-with-interactive-char.jpg"
                            alt="Bias Audit Report"
                            className="w-full h-full object-contain"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="self-stretch px-0 sm:px-2 md:px-0">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 sm:gap-4 md:gap-6">
                  <FeatureCard
                    title="Text Bias Analysis"
                    description="AI scans ad scripts for gendered, racial, or stereotypical language and flags problematic patterns."
                    isActive={activeCard === 0}
                    progress={activeCard === 0 ? progress : 0}
                    onClick={() => handleCardClick(0)}
                  />
                  <FeatureCard
                    title="Visual Bias Detection"
                    description="Analyzes ad visuals for representation diversity, framing, and patterns that could imply bias."
                    isActive={activeCard === 1}
                    progress={activeCard === 1 ? progress : 0}
                    onClick={() => handleCardClick(1)}
                  />
                  <FeatureCard
                    title="Audit & Scoring"
                    description="Generates comprehensive bias audit reports with risk scores and actionable recommendations."
                    isActive={activeCard === 2}
                    progress={activeCard === 2 ? progress : 0}
                    onClick={() => handleCardClick(2)}
                  />
                </div>
              </div>

              <CTASection />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
