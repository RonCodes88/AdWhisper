"use client"

import React, { useEffect, useRef } from "react";
import Image from "next/image";

export default function CTASection() {
  const logos = [
    { src: "/claude.png", alt: "Claude" },
    { src: "/fetchai.png", alt: "Fetch.ai" },
    { src: "/chroma.png", alt: "Chroma" },
  ]

  // Duplicate logos 2x for seamless infinite scroll
  const duplicatedLogos = [...logos, ...logos]

  // rAF-powered marquee for perfectly continuous scrolling without visible resets
  const trackRef = useRef<HTMLDivElement | null>(null)
  useEffect(() => {
    const el = trackRef.current
    if (!el) return

    let raf = 0
    let lastTimestamp = performance.now()
    let offsetPx = 0

    // Pixels per second; increase for faster scroll
    const speedPxPerSec = 260

    const step = (now: number) => {
      const deltaSec = (now - lastTimestamp) / 1000
      lastTimestamp = now

      offsetPx += speedPxPerSec * deltaSec
      const halfWidth = el.scrollWidth / 2 // because content is duplicated exactly twice
      if (halfWidth > 0) {
        if (offsetPx >= halfWidth) offsetPx -= halfWidth
        el.style.transform = `translate3d(${-offsetPx}px, 0, 0)`
      }

      raf = requestAnimationFrame(step)
    }

    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <div className="w-full relative overflow-hidden flex flex-col justify-center items-center gap-2">
      {/* Content */}
      <div className="self-stretch px-6 md:px-24 py-12 md:py-12 border-t border-b border-[rgba(55,50,47,0.12)] flex justify-center items-center gap-6 relative z-10">
        <div className="absolute inset-0 w-full h-full overflow-hidden">
          <div className="w-full h-full relative">
            {Array.from({ length: 300 }).map((_, i) => (
              <div
                key={i}
                className="absolute h-4 w-full rotate-[-45deg] origin-top-left outline outline-[0.5px] outline-[rgba(3,7,18,0.08)] outline-offset-[-0.25px]"
                style={{
                  top: `${i * 16 - 120}px`,
                  left: "-100%",
                  width: "300%",
                }}
              ></div>
            ))}
          </div>
        </div>

        <div className="w-full max-w-[586px] px-6 py-5 md:py-8 overflow-hidden rounded-lg flex flex-col justify-start items-center gap-6 relative z-20">
          <div className="self-stretch flex flex-col justify-start items-start gap-3">
            <div className="self-stretch text-center flex justify-center flex-col text-[#49423D] text-3xl md:text-5xl font-semibold leading-tight md:leading-[56px] font-sans tracking-tight">
              Reshape the way you advertise.
            </div>
            <div className="self-stretch text-center text-[#605A57] text-base leading-7 font-sans font-medium">
              Prevent costly bias issues before launch. Audit your ads with AI
              <br />
              and ensure every campaign promotes inclusivity and trust.
            </div>
          </div>

          <div className="w-full max-w-[497px] flex flex-col justify-center items-center gap-12">
            <div className="flex justify-start items-center gap-4">
              <div className="h-10 px-12 py-[6px] relative bg-[#37322F] shadow-[0px_0px_0px_2.5px_rgba(255,255,255,0.08)_inset] overflow-hidden rounded-full flex justify-center items-center cursor-pointer hover:bg-[#2A2520] transition-colors">
                <div className="absolute inset-0 h-full w-full bg-gradient-to-b from-[rgba(255,255,255,0)] to-[rgba(0,0,0,0.10)] mix-blend-multiply pointer-events-none"></div>
                <div className="flex flex-col justify-center text-white text-[13px] font-medium leading-5 font-sans">
                  Start for free
                </div>
              </div>
            </div>
          </div>

          {/* Powered by logos - Infinite scroll */}
          <div className="w-full overflow-hidden py-6">
            <div className="flex" ref={trackRef} style={{ willChange: 'transform' }}>
              {duplicatedLogos.map((logo, i) => (
                <div
                  key={i}
                  className="inline-flex items-center justify-center flex-shrink-0"
                  style={{ padding: '0 2rem', height: '5rem', width: '12rem' }}
                >
                  {logo.src.includes('?') ? (
                    <img
                      src={logo.src}
                      alt={logo.alt}
                      width={150}
                      height={80}
                      className="object-contain"
                      style={{ maxHeight: '4rem' }}
                      loading="eager"
                    />
                  ) : (
                    <Image
                      src={logo.src}
                      alt={logo.alt}
                      width={150}
                      height={80}
                      className="object-contain"
                      style={{ maxHeight: '4rem' }}
                      priority
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
