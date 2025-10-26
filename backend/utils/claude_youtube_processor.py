"""
Claude-based YouTube Video Processor for AdWhisper
Uses Claude to intelligently extract bias-relevant frames and process transcripts
"""

import os
import sys
import base64
import cv2
import numpy as np
import requests
from typing import Dict, Any, List, Optional
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import json
from io import BytesIO

class ClaudeYouTubeProcessor:
    """Processes YouTube videos using Claude for intelligent frame and transcript extraction"""
    
    def __init__(self):
        # Create frames directory within the codebase
        self.temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "extracted_frames")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.claude_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    
    def extract_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL"""
        if "youtube.com/watch?v=" in youtube_url:
            return youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            return youtube_url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError(f"Invalid YouTube URL: {youtube_url}")
    
    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript from YouTube API"""
        try:
            # Get list of available transcripts using the correct API
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)
            
            # Try to find English transcript first
            transcript = None
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                # If English fails, try to get any available transcript
                try:
                    transcript = transcript_list.find_generated_transcript(['en'])
                except:
                    # Get the first available transcript
                    for t in transcript_list:
                        transcript = t
                        break
            
            if transcript:
                # Fetch the actual transcript data
                transcript_data = transcript.fetch()
                # Extract text from snippets
                transcript_text = " ".join([snippet.text for snippet in transcript_data.snippets])
                return transcript_text
            else:
                print(f"⚠️ No transcripts available for video {video_id}")
                return None
                
        except Exception as e:
            print(f"⚠️ Could not fetch transcript from API: {e}")
            return None
    
    def get_video_metadata(self, youtube_url: str) -> tuple[Dict[str, Any], str]:
        """
        Get video metadata and duration without downloading
        """
        print(f"📊 Getting video metadata from: {youtube_url}")
        
        try:
            yt = YouTube(youtube_url)
            video_id = self.extract_video_id(youtube_url)
            
            metadata = {
                "title": yt.title,
                "author": yt.author,
                "duration": yt.length,
                "views": yt.views,
                "description": yt.description,
                "video_id": video_id
            }
            
            print(f"📹 Video: {yt.title}")
            print(f"⏱️  Duration: {yt.length}s")
            
            return metadata, video_id
            
        except Exception as e:
            print(f"❌ Error getting video metadata: {e}")
            raise
    
    def extract_frames_from_video(self, video_path: str, duration: int) -> List[np.ndarray]:
        """
        Extract frames every 5 seconds from actual video file
        """
        print(f"🎬 Extracting frames every 5 seconds from video file...")
        
        frames = []
        frame_timestamps = []
        
        # Calculate frame intervals (every 5 seconds)
        frame_interval = 5  # seconds
        total_frames = max(1, duration // frame_interval)
        
        print(f"📊 Video duration: {duration}s, will extract {total_frames} frames")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"📊 Video info: {frame_count} frames, {fps} FPS")
            
            for i in range(total_frames):
                timestamp = i * frame_interval
                frame_timestamps.append(timestamp)
                
                # Calculate frame number for this timestamp
                frame_number = int(timestamp * fps)
                
                # Seek to the specific frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    frames.append(frame)
                    print(f"📸 Frame {i+1}: {timestamp}s (size: {frame.shape})")
                    
                    # Save frame as image for verification
                    frame_filename = f"frame_{i+1}_{timestamp}s.jpg"
                    frame_path = os.path.join(self.temp_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    print(f"💾 Saved frame to: {frame_path}")
                else:
                    print(f"⚠️ Could not extract frame at {timestamp}s")
            
            cap.release()
            
        except Exception as e:
            print(f"❌ Error extracting frames from video: {e}")
            raise
        
        if not frames:
            raise ValueError("No frames could be extracted from the video")
        
        print(f"✅ Extracted {len(frames)} frames from video")
        return frames
    
    def download_video_with_ytdlp(self, youtube_url: str) -> tuple[str, Dict[str, Any]]:
        """
        Download YouTube video using yt-dlp (more reliable than pytube)
        """
        print(f"📥 Downloading video with yt-dlp from: {youtube_url}")
        
        try:
            import yt_dlp
            
            video_id = self.extract_video_id(youtube_url)
            video_path = os.path.join(self.temp_dir, f"{video_id}.mp4")
            
            # Configure yt-dlp options
            ydl_opts = {
                'outtmpl': video_path,
                'format': 'best[height<=720]',  # Limit to 720p for faster processing
                'quiet': False,
                'no_warnings': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(youtube_url, download=False)
                
                # Check duration
                duration = info.get('duration', 0)
                if duration > 300:  # 5 minutes max
                    raise ValueError(f"Video too long: {duration}s (max 300s allowed)")
                
                print(f"📹 Video: {info.get('title', 'Unknown')}")
                print(f"⏱️  Duration: {duration}s")
                
                # Download the video
                ydl.download([youtube_url])
                
                if not os.path.exists(video_path):
                    raise ValueError("Video file was not created")
                
                print(f"✅ Video downloaded: {video_path}")
                
                metadata = {
                    "title": info.get('title', 'Unknown'),
                    "author": info.get('uploader', 'Unknown'),
                    "duration": duration,
                    "views": info.get('view_count', 0),
                    "description": info.get('description', ''),
                    "video_id": video_id
                }
                
                return video_path, metadata
                
        except Exception as e:
            print(f"❌ Error downloading video with yt-dlp: {e}")
            raise

    def extract_frames_with_claude(self, video_path: str, transcript: str) -> List[np.ndarray]:
        """
        Use Claude to intelligently select bias-relevant frames from the video
        """
        print(f"🧠 Using Claude to analyze video for bias-relevant frames...")
        
        # Extract all frames first
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_timestamps = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video info: {frame_count} frames, {fps} FPS")
        
        # Extract frames at regular intervals
        frame_interval = max(1, frame_count // 20)  # Get ~20 frames max
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                frames.append(frame.copy())
                timestamp = frame_idx / fps
                frame_timestamps.append(timestamp)
                print(f"📸 Frame {len(frames)}: {frame_idx} (time: {timestamp:.1f}s)")
            
            frame_idx += 1
        
        cap.release()
        
        if not frames:
            print("⚠️ No frames extracted")
            return []
        
        # Use Claude to analyze frames and select bias-relevant ones
        selected_frames = self._claude_select_bias_frames(frames, frame_timestamps, transcript)
        
        print(f"✅ Claude selected {len(selected_frames)} bias-relevant frames")
        return selected_frames
    
    def _claude_select_bias_frames(self, frames: List[np.ndarray], timestamps: List[float], transcript: str) -> List[np.ndarray]:
        """
        Use Claude to select the most bias-relevant frames
        """
        try:
            # Convert frames to base64 for Claude analysis
            frame_data = []
            for i, frame in enumerate(frames):
                # Resize frame for Claude (smaller for faster processing)
                resized_frame = cv2.resize(frame, (320, 240))
                _, buffer = cv2.imencode('.jpg', resized_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                frame_data.append({
                    "frame_index": i,
                    "timestamp": timestamps[i],
                    "image": frame_b64
                })
            
            # Create prompt for Claude
            prompt = f"""
You are analyzing a YouTube video for potential bias. I'll provide you with:
1. The video transcript: "{transcript}"
2. {len(frames)} frames from the video with timestamps

Your task: Select the 5 most bias-relevant frames that would help detect:
- Gender bias (representation, roles, stereotypes)
- Racial/ethnic bias (diversity, representation)
- Age bias (ageism, generational stereotypes)
- Socioeconomic bias (class representation)
- Disability bias (accessibility, representation)
- LGBTQ+ bias (representation, stereotypes)

For each frame, consider:
- Who is shown and how they're portrayed
- Visual composition and framing
- Context from the transcript at that timestamp
- Potential stereotypes or bias indicators

Return a JSON response with the frame indices (0-based) of the 5 most important frames:
{{
    "selected_frames": [frame_index1, frame_index2, frame_index3, frame_index4, frame_index5],
    "reasoning": "Brief explanation of why these frames are most relevant for bias detection"
}}
"""
            
            # Call Claude API
            response = self._call_claude_api(prompt, frame_data)
            
            if response and "selected_frames" in response:
                selected_indices = response["selected_frames"]
                selected_frames = [frames[i] for i in selected_indices if i < len(frames)]
                print(f"🧠 Claude reasoning: {response.get('reasoning', 'No reasoning provided')}")
                return selected_frames
            else:
                print("⚠️ Claude didn't return valid frame selection, using first 5 frames")
                return frames[:5]
                
        except Exception as e:
            print(f"⚠️ Error in Claude frame selection: {e}")
            print("📸 Falling back to first 5 frames")
            return frames[:5]
    
    def process_transcript_with_claude(self, transcript: str) -> Dict[str, Any]:
        """
        Use Claude to process and analyze the transcript for bias indicators
        """
        print(f"🧠 Using Claude to analyze transcript for bias...")
        
        try:
            prompt = f"""
You are analyzing a YouTube video transcript for potential bias. Here's the transcript:

"{transcript}"

Your task: Analyze this transcript and provide a comprehensive bias analysis including:

1. **Gender Bias**: Look for gendered language, stereotypes, role assignments
2. **Racial/Ethnic Bias**: Check for racial stereotypes, cultural assumptions
3. **Age Bias**: Look for ageist language, generational stereotypes
4. **Socioeconomic Bias**: Check for class-based assumptions, economic stereotypes
5. **Disability Bias**: Look for ableist language, accessibility issues
6. **LGBTQ+ Bias**: Check for heteronormative assumptions, LGBTQ+ representation

For each bias type, provide:
- Specific examples from the transcript
- Severity level (Low/Medium/High)
- Potential impact
- Suggested improvements

Return a JSON response:
{{
    "overall_bias_score": 0-100,
    "bias_analysis": {{
        "gender": {{
            "severity": "Low/Medium/High",
            "examples": ["example1", "example2"],
            "impact": "description",
            "suggestions": ["suggestion1", "suggestion2"]
        }},
        "racial_ethnic": {{...}},
        "age": {{...}},
        "socioeconomic": {{...}},
        "disability": {{...}},
        "lgbtq": {{...}}
    }},
    "summary": "Overall assessment of bias in the transcript",
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""
            
            response = self._call_claude_api(prompt, [])
            
            if response:
                print(f"✅ Claude transcript analysis complete")
                return response
            else:
                print("⚠️ Claude transcript analysis failed")
                return {"error": "Claude analysis failed"}
                
        except Exception as e:
            print(f"❌ Error in Claude transcript analysis: {e}")
            return {"error": str(e)}
    
    def _call_claude_api(self, prompt: str, frame_data: List[Dict] = None) -> Optional[Dict]:
        """
        Call Claude API directly through Anthropic
        """
        try:
            # Prepare the request payload for Anthropic API
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 4000,
                "temperature": 0.1
            }
            
            # Add images if provided
            if frame_data:
                content = [{"type": "text", "text": prompt}]
                for frame_info in frame_data:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame_info['image']
                        }
                    })
                payload["messages"] = [{"role": "user", "content": content}]
            else:
                payload["messages"] = [{"role": "user", "content": prompt}]
            
            # Make API call to Anthropic
            headers = {
                "x-api-key": self.claude_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["content"][0]["text"]
                
                # Try to parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    print(f"⚠️ Claude returned non-JSON response: {content[:200]}...")
                    return {"raw_response": content}
            else:
                print(f"❌ Claude API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling Claude API: {e}")
            return None
    
    def frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        """Convert frames to base64 strings"""
        base64_frames = []
        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(frame_b64)
        return base64_frames
    
    def process_youtube_video(self, youtube_url: str) -> Dict[str, Any]:
        """
        Complete processing of YouTube video using Claude
        """
        print(f"\n🎬 PROCESSING YOUTUBE VIDEO WITH CLAUDE")
        print(f"🔗 URL: {youtube_url}")
        
        try:
            # Extract video ID
            video_id = self.extract_video_id(youtube_url)
            print(f"📺 Video ID: {video_id}")
            
            # Get transcript
            print(f"📝 EXTRACTING TRANSCRIPT...")
            transcript = self.get_transcript(video_id)
            if not transcript:
                raise ValueError("No transcript available for this video")
            
            print(f"✅ TRANSCRIPT EXTRACTED:")
            print(f"   📏 Length: {len(transcript)} characters")
            print(f"   📄 Preview: {transcript[:200]}...")
            print(f"   📄 FULL TRANSCRIPT:")
            print(f"   {'-' * 40}")
            print(f"   {transcript}")
            print(f"   {'-' * 40}")
            
            # Download video using yt-dlp
            print(f"📥 DOWNLOADING VIDEO...")
            video_path, metadata = self.download_video_with_ytdlp(youtube_url)
            
            # Extract frames every 5 seconds from actual video
            print(f"🎬 EXTRACTING FRAMES EVERY 5 SECONDS...")
            frames = self.extract_frames_from_video(video_path, metadata['duration'])
            
            # Process transcript with Claude
            print(f"🧠 PROCESSING TRANSCRIPT WITH CLAUDE...")
            transcript_analysis = self.process_transcript_with_claude(transcript)
            
            # Convert frames to base64
            frames_base64 = self.frames_to_base64(frames)
            
            # Clean up video file
            try:
                os.remove(video_path)
                print(f"🗑️ Cleaned up video file: {video_path}")
            except:
                pass
            
            result = {
                "success": True,
                "video_id": video_id,
                "transcript": transcript,
                "transcript_analysis": transcript_analysis,
                "frames": frames,
                "frames_base64": frames_base64,
                "num_frames": len(frames),
                "metadata": metadata,
                "thumbnail_url": f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            }
            
            print(f"✅ PROCESSING COMPLETE!")
            print(f"   📝 Transcript: {len(transcript)} chars")
            print(f"   🎬 Frames: {len(frames)} extracted (every 5 seconds)")
            print(f"   📊 Metadata: {metadata['title']}")
            print(f"   💾 Frame images saved to: {self.temp_dir}")
            print("=" * 50)
            
            return result
            
        except Exception as e:
            print(f"\n❌ PROCESSING FAILED: {e}")
            print("=" * 50)
            return {
                "success": False,
                "error": str(e),
                "transcript": "",
                "frames": [],
                "frames_base64": [],
                "num_frames": 0,
                "metadata": {}
            }

# Global instance
_claude_youtube_processor = None

def get_claude_youtube_processor() -> ClaudeYouTubeProcessor:
    """Get global Claude YouTube processor instance"""
    global _claude_youtube_processor
    if _claude_youtube_processor is None:
        _claude_youtube_processor = ClaudeYouTubeProcessor()
    return _claude_youtube_processor

def extract_youtube_content_with_claude(youtube_url: str) -> Dict[str, Any]:
    """Extract content from YouTube video using Claude"""
    processor = get_claude_youtube_processor()
    return processor.process_youtube_video(youtube_url)
