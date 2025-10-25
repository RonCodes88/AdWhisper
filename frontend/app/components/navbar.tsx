"use client"
import Link from "next/link"

export function Navbar() {
  return (
    <div className="w-full h-12 sm:h-14 md:h-16 lg:h-[84px] fixed left-0 top-0 flex justify-center items-center z-50 px-6 sm:px-8 md:px-12 lg:px-0 bg-background/80 backdrop-blur-sm">
      <div className="w-full h-0 absolute left-0 top-6 sm:top-7 md:top-8 lg:top-[42px] border-t border-border shadow-[0px_1px_0px_white]"></div>

      <div className="w-full max-w-[calc(100%-32px)] sm:max-w-[calc(100%-48px)] md:max-w-[calc(100%-64px)] lg:max-w-[700px] lg:w-[700px] h-10 sm:h-11 md:h-12 py-1.5 sm:py-2 px-3 sm:px-4 md:px-4 pr-2 sm:pr-3 bg-background backdrop-blur-sm shadow-[0px_0px_0px_2px_white] overflow-hidden rounded-[50px] flex justify-between items-center relative z-30">
        <div className="flex justify-start">
          <Link href="/" className="flex justify-start items-center hover:opacity-80 transition-opacity">
            <div className="flex flex-col justify-center text-foreground text-sm sm:text-base md:text-lg lg:text-xl font-semibold leading-5 font-serif">
              AdWhisper
            </div>
          </Link>
        </div>

        <div className="flex justify-end">
          <Link href="/upload" className="flex justify-start items-center hover:opacity-80 transition-opacity">
            <div className="flex flex-col justify-center text-muted-foreground hover:text-foreground text-xs md:text-[13px] font-medium leading-[14px] font-sans transition-colors">
              Upload
            </div>
          </Link>
        </div>
      </div>
    </div>
  )
}
