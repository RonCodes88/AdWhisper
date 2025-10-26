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
      <div className="relative flex flex-col justify-start items-center w-full">
        <div className="w-full max-w-none px-4 sm:px-6 md:px-8 lg:px-0 lg:max-w-[1060px] lg:w-[1060px] relative flex flex-col justify-start items-start min-h-screen">
          {/* Border lines matching landing page */}
          <div className="h-full absolute left-4 sm:left-6 md:left-8 lg:left-0 top-0 bg-border shadow-[1px_0px_0px_white] z-0 w-px"></div>
          <div className="h-full absolute right-4 sm:right-6 md:right-8 lg:right-0 top-0 bg-border shadow-[1px_0px_0px_white] z-0 w-px"></div>

          {/* Main Content */}
          <main className="w-full flex-1 flex items-start justify-center px-0 pt-24 sm:pt-28 md:pt-32 lg:pt-40 pb-12 relative z-10">
            <div className="w-full max-w-2xl px-4 sm:px-6 md:px-8 lg:px-0">
              {/* Page Title */}
              <div className="text-center mb-12">
            <h2 className="text-[40px] font-bold text-foreground mb-3 leading-tight font-serif">
              Analyze your ad
            </h2>
            <p className="text-muted-foreground text-base font-sans">
              Upload your ad for instant bias detection and<br />
              actionable recommendations
            </p>
          </div>

          {/* Upload Card */}
          <div className="bg-card border border-border rounded-xl shadow-sm p-8 mb-6">
            {/* Upload Section */}
            <div className="mb-6">
              <div className="flex items-start gap-3 mb-4">
                <div className="w-6 h-6 flex items-center justify-center">
                  <svg className="w-5 h-5 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div className="flex-1">
                  <h3 className="text-[15px] font-semibold text-foreground mb-1 font-sans">
                    Image or Video Upload
                  </h3>
                  <p className="text-sm text-muted-foreground font-sans">
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
                      : "border-border bg-muted/30 hover:border-border/60"
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
                        <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
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
              className="w-full py-3 px-4 bg-primary hover:bg-primary/90 disabled:bg-muted disabled:text-muted-foreground text-primary-foreground font-medium rounded-lg transition-colors flex items-center justify-center gap-2 disabled:cursor-not-allowed font-sans"
            >
              {isUploading ? (
                <>
                  <LoadingSpinner size="sm" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <span>Analyze for bias</span>
                </>
              )}
            </button>
          </div>

              {/* Error Message */}
              {error && (
                <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-start gap-3">
                  <span className="text-destructive text-lg">⚠️</span>
                  <div className="flex-1">
                    <p className="text-sm font-medium text-destructive font-sans">{error}</p>
                  </div>
                  <button
                    onClick={() => setError(null)}
                    className="text-destructive hover:text-destructive/80 text-sm font-medium font-sans"
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
