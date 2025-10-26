"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { submitAd, validateFile, formatFileSize } from "@/lib/api";
import { LoadingSpinner } from "@/app/components/LoadingSpinner";

export default function UploadPage() {
  const router = useRouter();
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (file: File) => {
    setError(null);
    const validation = validateFile(file);

    if (!validation.valid) {
      setError(validation.error || "Invalid file");
      return;
    }

    setSelectedFile(file);
  };

  const handleSubmit = async () => {
    setError(null);

    // Validation
    if (!selectedFile) {
      setError("Please upload a file");
      return;
    }

    setIsUploading(true);

    try {
      const params: any = {};

      if (selectedFile) {
        const isImage = selectedFile.type.startsWith("image/");
        const isVideo = selectedFile.type.startsWith("video/");

        if (isImage) {
          params.imageFile = selectedFile;
        } else if (isVideo) {
          params.videoFile = selectedFile;
        }
      }

      const response = await submitAd(params);

      // Redirect to results page
      router.push(`/results/${response.request_id}`);
    } catch (err: any) {
      setError(err.message || "Failed to submit ad for analysis");
      setIsUploading(false);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setError(null);
  };

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

          {/* Main Content */}
          <main className="w-full flex-1 flex items-start justify-center px-0 pt-24 sm:pt-28 md:pt-32 lg:pt-40 pb-12 relative z-10">
            <div className="w-full max-w-2xl px-4 sm:px-6 md:px-8 lg:px-0">
              {/* Page Title */}
              <div className="text-center mb-12">
                <h2 className="text-foreground text-[32px] sm:text-[42px] md:text-[52px] lg:text-[64px] font-bold leading-[1.1] font-serif mb-4">
                  Analyze your ad
                </h2>
                <p className="text-muted-foreground text-base sm:text-lg md:text-xl leading-[1.5] font-sans font-medium">
                  Upload your ad for instant bias detection and actionable recommendations
                </p>
              </div>

          {/* Upload Card */}
          <div className="bg-white shadow-[0px_0px_0px_0.75px_rgba(0,0,0,0.08)] overflow-hidden rounded-[6px] border border-border/50 p-6 sm:p-8">
            {/* Upload Section */}
            <div className="mb-6">
              <div className="flex items-start gap-3 mb-4">
                <div className="w-10 h-10 flex items-center justify-center">
                  <svg className="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <h3 className="text-foreground text-lg font-semibold leading-6 font-sans mb-1">
                    Image or Video Upload
                  </h3>
                  <p className="text-muted-foreground text-sm font-normal font-sans">
                    Upload a file to analyze for bias
                  </p>
                </div>
              </div>

              {/* File Upload Area */}
              <div className="mt-4">
                <label className="block text-sm font-medium text-foreground mb-2 font-sans">
                  Upload File
                </label>
                <div
                  className={`relative border-2 border-dashed rounded-lg transition-all ${
                    dragActive
                      ? "border-primary bg-primary/5"
                      : "border-border bg-background hover:border-border/60"
                  } ${isUploading ? "opacity-50 pointer-events-none" : ""}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  {!selectedFile ? (
                    <>
                      <input
                        type="file"
                        accept="image/*,video/*"
                        onChange={handleChange}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                        disabled={isUploading}
                      />
                      <div className="px-6 py-8 text-center">
                        <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-background flex items-center justify-center">
                          <svg
                            className="w-6 h-6 text-muted-foreground"
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
                        <p className="text-sm text-foreground mb-1 font-sans">
                          <span className="font-medium">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-xs text-muted-foreground font-sans">
                          Images (JPG, PNG, GIF, WEBP) or Videos (MP4, MOV, AVI, WEBM)
                        </p>
                        <p className="text-xs text-muted-foreground/70 mt-1 font-sans">Max file size: 50MB</p>
                      </div>
                    </>
                  ) : (
                    <div className="px-6 py-4 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded bg-primary/10 flex items-center justify-center text-xl">
                          {selectedFile.type.startsWith("image/") ? "🖼️" : "🎥"}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-foreground font-sans">
                            {selectedFile.name}
                          </p>
                          <p className="text-xs text-muted-foreground font-sans">
                            {formatFileSize(selectedFile.size)}
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={clearFile}
                        className="text-sm text-destructive hover:text-destructive/80 font-medium transition-colors font-sans"
                        disabled={isUploading}
                      >
                        Remove
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleSubmit}
              disabled={isUploading || !selectedFile}
              className="w-full h-12 px-8 py-2 relative bg-primary shadow-[0px_0px_0px_2.5px_rgba(255,255,255,0.08)_inset] overflow-hidden rounded-full flex justify-center items-center transition-all hover:shadow-[0px_0px_0px_2.5px_rgba(255,255,255,0.12)_inset] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="absolute inset-0 h-full w-full bg-gradient-to-b from-[rgba(255,255,255,0)] to-[rgba(0,0,0,0.10)] mix-blend-multiply pointer-events-none"></div>
              <div className="flex items-center gap-2 text-primary-foreground text-[15px] font-medium leading-5 font-sans">
                {isUploading ? (
                  <>
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Analyzing...
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
          </div>

              {/* Error Message */}
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-md p-4">
                  <p className="text-red-800 text-sm font-medium">{error}</p>
                </div>
              )}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
