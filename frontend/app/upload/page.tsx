"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { submitAd, validateFile, formatFileSize } from "@/lib/api";
import { LoadingSpinner } from "@/app/components/LoadingSpinner";

export default function UploadPage() {
  const router = useRouter();
  const [dragActive, setDragActive] = useState(false);
  const [textContent, setTextContent] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

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
    setSuccess(null);

    // Validation
    if (!textContent.trim() && !selectedFile) {
      setError("Please provide text content or upload a file");
      return;
    }

    setIsUploading(true);

    try {
      const params: any = {};

      if (textContent.trim()) {
        params.textContent = textContent;
      }

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

      setSuccess("Analysis started successfully!");

      // Redirect to results page after a brief delay
      setTimeout(() => {
        router.push(`/results/${response.request_id}`);
      }, 1000);
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
    <div className="w-full min-h-screen relative bg-background flex flex-col justify-start items-center pt-20">
      <div className="w-full max-w-4xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-4 font-serif">
            Upload Your Ad Content
          </h1>
          <p className="text-muted-foreground text-lg">
            Upload your ad scripts, images, or videos for bias analysis
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
            <span className="text-red-600 text-xl">⚠️</span>
            <div className="flex-1">
              <p className="text-sm font-medium text-red-900">{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-600 hover:text-red-800"
            >
              ✕
            </button>
          </div>
        )}

        {/* Success Message */}
        {success && (
          <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg flex items-start gap-3">
            <span className="text-green-600 text-xl">✓</span>
            <p className="text-sm font-medium text-green-900">{success}</p>
          </div>
        )}

        {/* Text Content Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-2">
            Ad Text Content (Optional)
          </label>
          <textarea
            value={textContent}
            onChange={(e) => setTextContent(e.target.value)}
            placeholder="Paste your ad copy, script, or description here..."
            className="w-full h-32 px-4 py-3 border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary resize-none"
            disabled={isUploading}
          />
        </div>

        {/* File Upload Area */}
        <div
          className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragActive
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50"
          } ${isUploading ? "opacity-50 pointer-events-none" : ""}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {!selectedFile && (
            <>
              <input
                type="file"
                accept="image/*,video/*"
                onChange={handleChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                disabled={isUploading}
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
                    {dragActive ? "Drop file here" : "Drag and drop file here"}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or click to browse files
                  </p>
                </div>

                <div className="text-xs text-muted-foreground">
                  Supports: Images (JPG, PNG, GIF, WEBP) and Videos (MP4, MOV,
                  AVI, WEBM)
                  <br />
                  Max file size: 50MB
                </div>
              </div>
            </>
          )}

          {selectedFile && (
            <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-primary/10 rounded flex items-center justify-center">
                  {selectedFile.type.startsWith("image/") ? "🖼️" : "🎥"}
                </div>
                <div className="text-left">
                  <p className="text-sm font-medium text-foreground">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>
              <button
                onClick={clearFile}
                className="px-3 py-1 text-sm text-red-600 hover:text-red-800 hover:bg-red-50 rounded transition-colors"
                disabled={isUploading}
              >
                Remove
              </button>
            </div>
          )}
        </div>

        {/* Submit Button */}
        <div className="mt-8 text-center">
          <button
            onClick={handleSubmit}
            disabled={isUploading || (!textContent.trim() && !selectedFile)}
            className="px-8 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-w-[200px]"
          >
            {isUploading ? (
              <span className="flex items-center justify-center gap-2">
                <LoadingSpinner size="sm" />
                Uploading...
              </span>
            ) : (
              "Start Analysis"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
