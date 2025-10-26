"""
YouTube Content Processor

Extracts transcripts, metadata, and thumbnails from YouTube videos
for bias analysis in the AdWhisper system.
"""

import re
from typing import Optional, Dict, Any, List
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
try:
    from pytube import YouTube
except ImportError:
    YouTube = None


class YouTubeProcessor:
    """Process YouTube videos to extract content for bias analysis"""

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats.

        Supports:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        """
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    @staticmethod
    def get_transcript(video_id: str, languages: List[str] = ['en']) -> Dict[str, Any]:
        """
        Get transcript/captions from YouTube video.

        Args:
            video_id: YouTube video ID
            languages: List of language codes to try (default: ['en'])

        Returns:
            Dict with transcript text, segments, and metadata
        """
        try:
            # Create API instance (new API requires instantiation)
            api = YouTubeTranscriptApi()

            # Try to fetch transcript
            segments = None
            used_language = None
            is_auto_generated = False

            try:
                # Try to fetch with preferred languages
                result = api.fetch(video_id, languages=languages)
                # Convert snippets to dict format for compatibility
                segments = [{"text": s.text, "start": s.start, "duration": s.duration}
                           for s in result.snippets]
                used_language = result.language_code
                is_auto_generated = result.is_generated
            except Exception as fetch_error:
                # If direct fetch fails, try listing and selecting manually
                try:
                    transcript_list = api.list(video_id)

                    # Try manual transcripts first
                    for transcript_info in transcript_list:
                        if not transcript_info.is_generated and transcript_info.language_code in languages:
                            fetched = transcript_info.fetch()
                            segments = [{"text": s.text, "start": s.start, "duration": s.duration}
                                       for s in fetched.snippets]
                            used_language = fetched.language_code
                            is_auto_generated = False
                            break

                    # If no manual transcript, try auto-generated
                    if segments is None:
                        for transcript_info in transcript_list:
                            if transcript_info.is_generated and transcript_info.language_code in languages:
                                fetched = transcript_info.fetch()
                                segments = [{"text": s.text, "start": s.start, "duration": s.duration}
                                           for s in fetched.snippets]
                                used_language = fetched.language_code
                                is_auto_generated = True
                                break
                except:
                    pass

            # If still no segments, return error
            if segments is None:
                return {
                    "success": False,
                    "error": "No transcript available for this video",
                    "text": "",
                    "segments": [],
                    "language": None,
                    "is_generated": False
                }

            # Combine all text
            full_text = " ".join([seg['text'] for seg in segments])

            return {
                "success": True,
                "text": full_text,
                "segments": segments,
                "language": used_language,
                "is_generated": is_auto_generated,
                "error": None
            }

        except TranscriptsDisabled:
            return {
                "success": False,
                "error": "Transcripts are disabled for this video",
                "text": "",
                "segments": [],
                "language": None,
                "is_generated": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error fetching transcript: {str(e)}",
                "text": "",
                "segments": [],
                "language": None,
                "is_generated": False
            }

    @staticmethod
    def get_video_metadata(video_id: str) -> Dict[str, Any]:
        """
        Get video metadata using pytube (fallback to basic info if unavailable).

        Args:
            video_id: YouTube video ID

        Returns:
            Dict with video metadata (title, channel, duration, etc.)
        """
        try:
            if YouTube is None:
                return {
                    "success": False,
                    "error": "pytube not installed",
                    "title": None,
                    "channel": None,
                    "duration": None,
                    "views": None,
                    "description": None
                }

            url = f"https://www.youtube.com/watch?v={video_id}"
            yt = YouTube(url)

            return {
                "success": True,
                "title": yt.title,
                "channel": yt.author,
                "duration": yt.length,  # in seconds
                "views": yt.views,
                "description": yt.description,
                "publish_date": str(yt.publish_date) if yt.publish_date else None,
                "thumbnail_url": yt.thumbnail_url,
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error fetching metadata: {str(e)}",
                "title": None,
                "channel": None,
                "duration": None,
                "views": None,
                "description": None,
                "thumbnail_url": None
            }

    @staticmethod
    def get_thumbnail_url(video_id: str, quality: str = "maxresdefault") -> str:
        """
        Get thumbnail URL for YouTube video.

        Args:
            video_id: YouTube video ID
            quality: Thumbnail quality (maxresdefault, sddefault, hqdefault, mqdefault, default)

        Returns:
            Thumbnail URL
        """
        return f"https://img.youtube.com/vi/{video_id}/{quality}.jpg"

    @classmethod
    def process_youtube_video(cls, url: str) -> Dict[str, Any]:
        """
        Complete processing of YouTube video.

        Extracts video ID, transcript, metadata, and thumbnail URL.

        Args:
            url: YouTube video URL

        Returns:
            Dict with all extracted content and metadata
        """
        # Extract video ID
        video_id = cls.extract_video_id(url)
        if not video_id:
            return {
                "success": False,
                "error": "Invalid YouTube URL",
                "video_id": None,
                "transcript": None,
                "metadata": None,
                "thumbnail_url": None
            }

        # Get transcript
        transcript_data = cls.get_transcript(video_id)

        # Get metadata
        metadata = cls.get_video_metadata(video_id)

        # Get thumbnail URL
        thumbnail_url = cls.get_thumbnail_url(video_id)

        return {
            "success": True,
            "video_id": video_id,
            "transcript": transcript_data,
            "metadata": metadata,
            "thumbnail_url": thumbnail_url,
            "youtube_url": url,
            "error": None
        }


# Convenience functions for direct use
def extract_youtube_content(url: str) -> Dict[str, Any]:
    """
    Main entry point for extracting content from YouTube video.

    Usage:
        result = extract_youtube_content("https://www.youtube.com/watch?v=...")
        if result["success"]:
            transcript_text = result["transcript"]["text"]
            thumbnail_url = result["thumbnail_url"]
    """
    processor = YouTubeProcessor()
    return processor.process_youtube_video(url)


if __name__ == "__main__":
    # Test the processor
    print("YouTube Content Processor - Test Mode\n")

    # Test URL - Replace with actual video
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    print(f"Testing with URL: {test_url}\n")

    result = extract_youtube_content(test_url)

    if result["success"]:
        print("✅ Processing successful!\n")
        print(f"Video ID: {result['video_id']}")
        print(f"\nTranscript available: {result['transcript']['success']}")
        if result['transcript']['success']:
            print(f"Transcript language: {result['transcript']['language']}")
            print(f"Transcript length: {len(result['transcript']['text'])} characters")
            print(f"First 200 chars: {result['transcript']['text'][:200]}...")
        else:
            print(f"Transcript error: {result['transcript']['error']}")

        print(f"\nMetadata available: {result['metadata']['success']}")
        if result['metadata']['success']:
            print(f"Title: {result['metadata']['title']}")
            print(f"Channel: {result['metadata']['channel']}")
            print(f"Duration: {result['metadata']['duration']} seconds")

        print(f"\nThumbnail URL: {result['thumbnail_url']}")
    else:
        print(f"❌ Processing failed: {result['error']}")
