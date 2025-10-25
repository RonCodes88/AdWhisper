"use client"
import Link from "next/link"
import { useState, useEffect, useRef } from "react"

export function Navbar() {
  const [isScrolled, setIsScrolled] = useState(false)
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20)
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => document.removeEventListener("mousedown", handleClickOutside)
  }, [])

  return (
    <div className={`w-full fixed left-0 top-0 flex justify-center items-center z-50 px-6 sm:px-8 md:px-12 lg:px-0 bg-background/80 backdrop-blur-sm transition-all duration-300 ${
      isScrolled ? "h-10 sm:h-12 md:h-14" : "h-12 sm:h-14 md:h-16 lg:h-[84px]"
    }`}>
      <div className={`w-full h-0 absolute left-0 border-t border-border shadow-[0px_1px_0px_white] transition-all duration-300 ${
        isScrolled ? "top-5 sm:top-6 md:top-7" : "top-6 sm:top-7 md:top-8 lg:top-[42px]"
      }`}></div>

      <div className={`w-full max-w-[calc(100%-32px)] sm:max-w-[calc(100%-48px)] md:max-w-[calc(100%-64px)] lg:max-w-[700px] lg:w-[700px] py-1.5 sm:py-2 px-3 sm:px-4 md:px-4 pr-2 sm:pr-3 bg-background backdrop-blur-sm shadow-[0px_0px_0px_2px_white] rounded-[50px] flex justify-between items-center relative z-30 transition-all duration-300 ${
        isScrolled ? "h-8 sm:h-9 md:h-10" : "h-10 sm:h-11 md:h-12"
      }`}>
        <div className="flex justify-start">
          <Link href="/" className="flex justify-start items-center hover:opacity-80 transition-opacity">
            <div className={`flex flex-col justify-center text-foreground font-semibold leading-5 font-serif transition-all duration-300 ${
              isScrolled ? "text-xs sm:text-sm md:text-base" : "text-sm sm:text-base md:text-lg lg:text-xl"
            }`}>
              AdWhisper
            </div>
          </Link>
        </div>

        <div className="flex justify-end items-center gap-4 sm:gap-6">
          <Link href="/upload" className="flex justify-start items-center hover:opacity-80 transition-opacity">
            <div className={`flex flex-col justify-center text-muted-foreground hover:text-foreground font-medium leading-[14px] font-sans transition-all duration-300 ${
              isScrolled ? "text-[10px] sm:text-xs md:text-[11px]" : "text-xs md:text-[13px]"
            }`}>
              Upload
            </div>
          </Link>
          
          <div className="relative" ref={dropdownRef}>
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              type="button"
              className="flex justify-center items-center hover:opacity-80 transition-opacity focus:outline-none cursor-pointer"
            >
              <div className={`rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-primary-foreground font-semibold shadow-sm transition-all duration-300 ${
                isScrolled ? "w-6 h-6 sm:w-7 sm:h-7 text-[10px] sm:text-xs" : "w-7 h-7 sm:w-8 sm:h-8 text-xs sm:text-sm"
              }`}>
                AC
              </div>
            </button>

            <div
              className={`absolute right-0 mt-2 w-48 bg-white border border-border rounded-lg shadow-lg overflow-hidden z-[100] transition-all duration-200 ease-in-out origin-top-right ${
                isDropdownOpen
                  ? "opacity-100 scale-100 pointer-events-auto"
                  : "opacity-0 scale-95 pointer-events-none"
              }`}
            >
              <Link
                href="/portal"
                onClick={() => setIsDropdownOpen(false)}
                className="block px-4 py-3 text-sm text-foreground hover:bg-gray-50 transition-colors"
              >
                Profile
              </Link>
              <button
                type="button"
                disabled
                className="w-full text-left px-4 py-3 text-sm text-gray-400 cursor-not-allowed bg-gray-50"
              >
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
