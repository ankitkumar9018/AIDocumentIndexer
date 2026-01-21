"""
MP3 Chapter Markers Service
===========================

Adds chapter markers to MP3 files for navigation.
Supports ID3v2 chapter frames (CHAP) and table of contents (CTOC).

Features:
1. Add individual chapter markers
2. Create table of contents
3. Chapter thumbnail images (optional)
4. URL links for chapters (optional)
5. Parse existing chapters

Compatible with:
- Apple Podcasts
- Overcast
- Pocket Casts
- VLC
- Most modern podcast players
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Chapter:
    """Represents a chapter marker in an audio file."""
    title: str
    start_ms: int  # Start time in milliseconds
    end_ms: Optional[int] = None  # End time (optional, defaults to next chapter start)
    url: Optional[str] = None  # Optional URL link
    image_path: Optional[Path] = None  # Optional chapter thumbnail

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "url": self.url,
            "image_path": str(self.image_path) if self.image_path else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chapter":
        """Create from dictionary."""
        return cls(
            title=data["title"],
            start_ms=data["start_ms"],
            end_ms=data.get("end_ms"),
            url=data.get("url"),
            image_path=Path(data["image_path"]) if data.get("image_path") else None,
        )


@dataclass
class ChapterResult:
    """Result of chapter marking operation."""
    success: bool
    chapters_added: int
    has_toc: bool
    output_path: Path
    error: Optional[str] = None


class ChapterMarkerService:
    """
    Add chapter markers to MP3 files.

    Uses mutagen for ID3 tag manipulation.
    """

    def __init__(self):
        """Initialize chapter marker service."""
        self._mutagen_available = self._check_mutagen()

    def _check_mutagen(self) -> bool:
        """Check if mutagen is available."""
        try:
            import mutagen
            return True
        except ImportError:
            logger.warning(
                "mutagen not installed. Chapter markers will be disabled. "
                "Install with: pip install mutagen"
            )
            return False

    async def add_chapters(
        self,
        mp3_path: Path,
        chapters: List[Chapter],
        toc_title: str = "Table of Contents",
        output_path: Optional[Path] = None,
    ) -> ChapterResult:
        """
        Add chapter markers to an MP3 file.

        Args:
            mp3_path: Path to the MP3 file
            chapters: List of chapters to add
            toc_title: Title for the table of contents
            output_path: Optional output path (modifies in place if not provided)

        Returns:
            ChapterResult with operation details
        """
        if not self._mutagen_available:
            return ChapterResult(
                success=False,
                chapters_added=0,
                has_toc=False,
                output_path=mp3_path,
                error="mutagen not installed",
            )

        if not chapters:
            return ChapterResult(
                success=True,
                chapters_added=0,
                has_toc=False,
                output_path=mp3_path,
            )

        try:
            from mutagen.mp3 import MP3
            from mutagen.id3 import ID3, CHAP, CTOC, TIT2, WXXX, APIC

            # Copy file if output path is different
            actual_path = mp3_path
            if output_path and output_path != mp3_path:
                import shutil
                shutil.copy(mp3_path, output_path)
                actual_path = output_path

            # Load MP3 file
            audio = MP3(str(actual_path))

            # Ensure ID3 tags exist
            if audio.tags is None:
                audio.add_tags()

            # Get audio duration for end time calculation
            duration_ms = int(audio.info.length * 1000)

            # Sort chapters by start time
            sorted_chapters = sorted(chapters, key=lambda c: c.start_ms)

            # Fill in missing end times
            for i, chapter in enumerate(sorted_chapters):
                if chapter.end_ms is None:
                    if i < len(sorted_chapters) - 1:
                        # End at next chapter start
                        chapter.end_ms = sorted_chapters[i + 1].start_ms
                    else:
                        # Last chapter ends at audio end
                        chapter.end_ms = duration_ms

            # Add chapter frames
            chapter_ids = []
            for i, chapter in enumerate(sorted_chapters):
                chapter_id = f"chp{i}"
                chapter_ids.append(chapter_id)

                # Create sub-frames for chapter
                sub_frames = [TIT2(encoding=3, text=chapter.title)]

                # Add URL if provided
                if chapter.url:
                    sub_frames.append(WXXX(encoding=3, desc="", url=chapter.url))

                # Add image if provided
                if chapter.image_path and chapter.image_path.exists():
                    try:
                        mime_type = "image/jpeg"
                        if chapter.image_path.suffix.lower() == ".png":
                            mime_type = "image/png"

                        with open(chapter.image_path, "rb") as img_file:
                            sub_frames.append(APIC(
                                encoding=3,
                                mime=mime_type,
                                type=3,  # Cover (front)
                                desc="Chapter Image",
                                data=img_file.read(),
                            ))
                    except Exception as e:
                        logger.warning(
                            "Failed to add chapter image",
                            chapter=chapter.title,
                            error=str(e),
                        )

                # Create chapter frame
                audio.tags.add(CHAP(
                    element_id=chapter_id,
                    start_time=chapter.start_ms,
                    end_time=chapter.end_ms,
                    sub_frames=sub_frames,
                ))

            # Add table of contents
            audio.tags.add(CTOC(
                element_id="toc",
                flags=3,  # Ordered + Top-level
                child_element_ids=chapter_ids,
                sub_frames=[TIT2(encoding=3, text=toc_title)],
            ))

            # Save changes
            audio.save()

            logger.info(
                "Added chapter markers to MP3",
                path=str(actual_path),
                chapters=len(sorted_chapters),
            )

            return ChapterResult(
                success=True,
                chapters_added=len(sorted_chapters),
                has_toc=True,
                output_path=actual_path,
            )

        except Exception as e:
            logger.error(
                "Failed to add chapter markers",
                error=str(e),
                path=str(mp3_path),
            )
            return ChapterResult(
                success=False,
                chapters_added=0,
                has_toc=False,
                output_path=mp3_path,
                error=str(e),
            )

    async def get_chapters(self, mp3_path: Path) -> List[Chapter]:
        """
        Extract existing chapters from an MP3 file.

        Args:
            mp3_path: Path to the MP3 file

        Returns:
            List of chapters found
        """
        if not self._mutagen_available:
            return []

        try:
            from mutagen.mp3 import MP3
            from mutagen.id3 import ID3, CHAP

            audio = MP3(str(mp3_path))

            if audio.tags is None:
                return []

            chapters = []

            # Find all CHAP frames
            for key in audio.tags.keys():
                if key.startswith("CHAP"):
                    frame = audio.tags[key]
                    if isinstance(frame, CHAP):
                        # Extract title from sub-frames
                        title = "Untitled"
                        url = None

                        for sub_frame in frame.sub_frames:
                            if hasattr(sub_frame, 'text'):
                                title = str(sub_frame.text[0]) if sub_frame.text else title
                            if hasattr(sub_frame, 'url'):
                                url = sub_frame.url

                        chapters.append(Chapter(
                            title=title,
                            start_ms=frame.start_time,
                            end_ms=frame.end_time,
                            url=url,
                        ))

            # Sort by start time
            chapters.sort(key=lambda c: c.start_ms)

            return chapters

        except Exception as e:
            logger.warning(
                "Failed to read chapter markers",
                error=str(e),
                path=str(mp3_path),
            )
            return []

    async def remove_chapters(self, mp3_path: Path) -> bool:
        """
        Remove all chapter markers from an MP3 file.

        Args:
            mp3_path: Path to the MP3 file

        Returns:
            True if successful
        """
        if not self._mutagen_available:
            return False

        try:
            from mutagen.mp3 import MP3

            audio = MP3(str(mp3_path))

            if audio.tags is None:
                return True

            # Find and remove all CHAP and CTOC frames
            keys_to_remove = [
                key for key in audio.tags.keys()
                if key.startswith("CHAP") or key.startswith("CTOC")
            ]

            for key in keys_to_remove:
                del audio.tags[key]

            audio.save()

            logger.info(
                "Removed chapter markers from MP3",
                path=str(mp3_path),
                removed=len(keys_to_remove),
            )

            return True

        except Exception as e:
            logger.warning(
                "Failed to remove chapter markers",
                error=str(e),
                path=str(mp3_path),
            )
            return False

    def create_chapters_from_sections(
        self,
        sections: List[Dict[str, Any]],
        start_offset_ms: int = 0,
    ) -> List[Chapter]:
        """
        Create chapter markers from document sections.

        Utility method to convert document outline to chapters.

        Args:
            sections: List of {"title": str, "duration_ms": int} dicts
            start_offset_ms: Offset for first chapter (e.g., after intro)

        Returns:
            List of chapters with calculated timings
        """
        chapters = []
        current_time = start_offset_ms

        for section in sections:
            title = section.get("title", "Section")
            duration = section.get("duration_ms", 60000)  # Default 1 minute

            chapters.append(Chapter(
                title=title,
                start_ms=current_time,
                end_ms=current_time + duration,
            ))

            current_time += duration

        return chapters

    def format_time(self, ms: int) -> str:
        """
        Format milliseconds as human-readable time.

        Args:
            ms: Time in milliseconds

        Returns:
            Formatted time string (e.g., "1:23:45" or "23:45")
        """
        seconds = ms // 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


# Singleton instance
_chapter_marker_service: Optional[ChapterMarkerService] = None


def get_chapter_marker_service() -> ChapterMarkerService:
    """Get or create the singleton chapter marker service."""
    global _chapter_marker_service
    if _chapter_marker_service is None:
        _chapter_marker_service = ChapterMarkerService()
    return _chapter_marker_service
