"""
Audio Mixer Service
===================

Mixes voice audio with background music and sound effects
for production-quality audio output.

Features:
1. Background music ducking (auto-lower when voice plays)
2. Intro/outro sound effects
3. Music looping to match voice length
4. Volume normalization
5. Crossfade transitions

Reference: Phase 2 improvements plan
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)


class MusicStyle(str, Enum):
    """Available background music styles."""
    CORPORATE = "corporate"
    AMBIENT = "ambient"
    UPBEAT = "upbeat"
    MINIMAL = "minimal"
    PODCAST = "podcast"
    NONE = "none"


class SoundEffect(str, Enum):
    """Available sound effects."""
    PODCAST_INTRO = "podcast_intro"
    PODCAST_OUTRO = "podcast_outro"
    TRANSITION = "transition"
    NOTIFICATION = "notification"
    NONE = "none"


@dataclass
class MixerConfig:
    """Configuration for audio mixing."""
    background_music: MusicStyle = MusicStyle.NONE
    intro_effect: SoundEffect = SoundEffect.NONE
    outro_effect: SoundEffect = SoundEffect.NONE

    # Volume levels (0.0 to 1.0)
    voice_volume: float = 1.0
    music_volume: float = 0.15  # Background music is quieter
    effect_volume: float = 0.8

    # Ducking settings (lower music when voice plays)
    duck_music: bool = True
    duck_amount: float = 0.6  # How much to lower music (0.6 = -6dB)
    duck_fade_ms: int = 500  # Fade duration for ducking

    # Transitions
    crossfade_ms: int = 1000  # Crossfade between sections
    intro_fade_ms: int = 2000  # Fade in duration
    outro_fade_ms: int = 3000  # Fade out duration


@dataclass
class AudioSegment:
    """Represents an audio segment with metadata."""
    path: Path
    duration_ms: int
    sample_rate: int = 44100
    channels: int = 2

    # Timing for mixing
    start_ms: int = 0  # When this segment starts in the final mix


@dataclass
class MixResult:
    """Result of audio mixing operation."""
    output_path: Path
    duration_ms: int
    segments_mixed: int
    has_background_music: bool
    has_intro: bool
    has_outro: bool
    peak_level: float  # Highest audio level (for normalization check)


class AudioMixer:
    """
    Mix voice audio with background music and effects.

    Uses pydub for audio processing. Falls back gracefully
    if pydub is not available.
    """

    def __init__(
        self,
        assets_path: Optional[Path] = None,
    ):
        """
        Initialize audio mixer.

        Args:
            assets_path: Path to audio assets (music, effects)
        """
        # Default assets path in backend/assets/audio
        if assets_path is None:
            backend_root = Path(__file__).resolve().parent.parent.parent
            assets_path = backend_root / "assets" / "audio"

        self.assets_path = assets_path
        self._pydub_available = self._check_pydub()

        # Create default assets structure
        self._ensure_assets_structure()

    def _check_pydub(self) -> bool:
        """Check if pydub is available."""
        try:
            from pydub import AudioSegment as PydubSegment
            return True
        except ImportError:
            logger.warning(
                "pydub not installed. Audio mixing will be disabled. "
                "Install with: pip install pydub"
            )
            return False

    def _ensure_assets_structure(self):
        """Create default assets directory structure."""
        dirs = [
            self.assets_path / "music",
            self.assets_path / "effects",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _get_music_path(self, style: MusicStyle) -> Optional[Path]:
        """Get path to background music file."""
        if style == MusicStyle.NONE:
            return None

        # Look for music file matching style
        music_dir = self.assets_path / "music"
        for ext in [".mp3", ".wav", ".ogg"]:
            path = music_dir / f"{style.value}{ext}"
            if path.exists():
                return path

        # Fall back to any music file
        for file in music_dir.iterdir():
            if file.suffix.lower() in [".mp3", ".wav", ".ogg"]:
                return file

        return None

    def _get_effect_path(self, effect: SoundEffect) -> Optional[Path]:
        """Get path to sound effect file."""
        if effect == SoundEffect.NONE:
            return None

        effects_dir = self.assets_path / "effects"
        for ext in [".mp3", ".wav", ".ogg"]:
            path = effects_dir / f"{effect.value}{ext}"
            if path.exists():
                return path

        return None

    async def mix_audio(
        self,
        voice_track: Path,
        output_path: Path,
        config: Optional[MixerConfig] = None,
    ) -> MixResult:
        """
        Mix voice audio with background music and effects.

        Args:
            voice_track: Path to the main voice audio file
            output_path: Where to save the mixed audio
            config: Mixing configuration

        Returns:
            MixResult with output details
        """
        if config is None:
            config = MixerConfig()

        if not self._pydub_available:
            # Just copy voice track if pydub not available
            logger.info("pydub not available, copying voice track unchanged")
            import shutil
            shutil.copy(voice_track, output_path)
            return MixResult(
                output_path=output_path,
                duration_ms=0,  # Unknown without pydub
                segments_mixed=1,
                has_background_music=False,
                has_intro=False,
                has_outro=False,
                peak_level=0.0,
            )

        # Import pydub here after checking availability
        from pydub import AudioSegment as PydubSegment

        logger.info(
            "Starting audio mix",
            voice_track=str(voice_track),
            music_style=config.background_music.value,
            duck_music=config.duck_music,
        )

        # Load voice track
        voice = PydubSegment.from_file(str(voice_track))
        voice_duration_ms = len(voice)

        # Adjust voice volume
        if config.voice_volume != 1.0:
            voice = voice + (20 * (config.voice_volume - 1))  # Convert to dB

        # Start with voice as base
        mixed = voice
        has_background = False
        has_intro = False
        has_outro = False

        # Add intro effect if specified
        intro_duration_ms = 0
        if config.intro_effect != SoundEffect.NONE:
            intro_path = self._get_effect_path(config.intro_effect)
            if intro_path:
                intro = PydubSegment.from_file(str(intro_path))
                intro_duration_ms = len(intro)

                # Adjust volume
                intro = intro + (20 * (config.effect_volume - 1))

                # Add fade out to intro
                intro = intro.fade_out(min(config.crossfade_ms, intro_duration_ms // 2))

                # Prepend intro with crossfade
                mixed = intro.append(mixed, crossfade=config.crossfade_ms)
                has_intro = True
                logger.debug("Added intro effect", duration_ms=intro_duration_ms)

        # Add outro effect if specified
        outro_duration_ms = 0
        if config.outro_effect != SoundEffect.NONE:
            outro_path = self._get_effect_path(config.outro_effect)
            if outro_path:
                outro = PydubSegment.from_file(str(outro_path))
                outro_duration_ms = len(outro)

                # Adjust volume
                outro = outro + (20 * (config.effect_volume - 1))

                # Add fade in to outro
                outro = outro.fade_in(min(config.crossfade_ms, outro_duration_ms // 2))

                # Append outro with crossfade
                mixed = mixed.append(outro, crossfade=config.crossfade_ms)
                has_outro = True
                logger.debug("Added outro effect", duration_ms=outro_duration_ms)

        # Add background music if specified
        if config.background_music != MusicStyle.NONE:
            music_path = self._get_music_path(config.background_music)
            if music_path:
                music = PydubSegment.from_file(str(music_path))

                # Adjust volume (background music should be quieter)
                music_db = 20 * (config.music_volume - 1)  # Convert to dB
                music = music + music_db

                # Loop music to match mixed duration
                mixed_duration = len(mixed)
                music = self._loop_to_length(music, mixed_duration)

                # Apply ducking if enabled
                if config.duck_music:
                    # Create ducking envelope based on voice presence
                    # For simplicity, duck during the voice section
                    voice_start = intro_duration_ms
                    voice_end = intro_duration_ms + voice_duration_ms

                    # Create ducked version
                    duck_db = 20 * (config.duck_amount - 1)

                    # Split music into sections
                    if voice_start > 0:
                        pre_voice = music[:voice_start]
                    else:
                        pre_voice = PydubSegment.silent(duration=0)

                    during_voice = music[voice_start:voice_end]
                    during_voice = during_voice + duck_db  # Duck the music

                    if voice_end < mixed_duration:
                        post_voice = music[voice_end:]
                    else:
                        post_voice = PydubSegment.silent(duration=0)

                    # Reassemble with fades
                    music = pre_voice
                    if len(during_voice) > 0:
                        # Add fade transitions
                        during_voice = during_voice.fade_in(config.duck_fade_ms)
                        during_voice = during_voice.fade_out(config.duck_fade_ms)
                        music = music + during_voice
                    if len(post_voice) > 0:
                        music = music + post_voice

                # Add fade in/out to music
                music = music.fade_in(config.intro_fade_ms)
                music = music.fade_out(config.outro_fade_ms)

                # Overlay music under mixed audio
                mixed = music.overlay(mixed)
                has_background = True
                logger.debug("Added background music", style=config.background_music.value)

        # Normalize to prevent clipping
        peak = mixed.max
        if peak > 32767:  # 16-bit max
            # Reduce volume to prevent clipping
            reduction = 32767 / peak
            mixed = mixed + (20 * (reduction - 1))
            logger.debug("Normalized audio to prevent clipping", reduction=reduction)

        # Export
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine output format from extension
        output_format = output_path.suffix.lower().lstrip(".")
        if output_format == "mp3":
            mixed.export(str(output_path), format="mp3", bitrate="192k")
        elif output_format == "wav":
            mixed.export(str(output_path), format="wav")
        else:
            mixed.export(str(output_path), format="mp3", bitrate="192k")

        logger.info(
            "Audio mixing complete",
            output_path=str(output_path),
            duration_ms=len(mixed),
            has_background=has_background,
            has_intro=has_intro,
            has_outro=has_outro,
        )

        return MixResult(
            output_path=output_path,
            duration_ms=len(mixed),
            segments_mixed=1 + (1 if has_background else 0) + (1 if has_intro else 0) + (1 if has_outro else 0),
            has_background_music=has_background,
            has_intro=has_intro,
            has_outro=has_outro,
            peak_level=mixed.max / 32767,
        )

    def _loop_to_length(self, audio, target_length_ms: int):
        """Loop audio segment to match target length."""
        from pydub import AudioSegment as PydubSegment

        if len(audio) >= target_length_ms:
            return audio[:target_length_ms]

        # Loop until we have enough
        result = audio
        while len(result) < target_length_ms:
            result = result + audio

        return result[:target_length_ms]

    async def create_intro_outro_pack(
        self,
        style: str = "podcast",
    ) -> Dict[str, Path]:
        """
        Generate simple intro/outro sound effects using synthesis.

        This creates basic effects when no assets are available.
        Requires pydub and scipy.

        Args:
            style: Style of effects to create

        Returns:
            Dict with paths to created effect files
        """
        if not self._pydub_available:
            logger.warning("Cannot create effects without pydub")
            return {}

        try:
            from pydub import AudioSegment as PydubSegment
            from pydub.generators import Sine
        except ImportError:
            logger.warning("pydub generators not available")
            return {}

        effects = {}
        effects_dir = self.assets_path / "effects"

        # Create a simple intro chime
        intro_path = effects_dir / "podcast_intro.mp3"
        if not intro_path.exists():
            # Create ascending chime
            chime = Sine(440).to_audio_segment(duration=200)
            chime = chime + Sine(554).to_audio_segment(duration=200)
            chime = chime + Sine(659).to_audio_segment(duration=400)
            chime = chime.fade_in(50).fade_out(200)
            chime = chime - 10  # Lower volume
            chime.export(str(intro_path), format="mp3")
            effects["intro"] = intro_path

        # Create a simple outro chime
        outro_path = effects_dir / "podcast_outro.mp3"
        if not outro_path.exists():
            # Create descending chime
            chime = Sine(659).to_audio_segment(duration=200)
            chime = chime + Sine(554).to_audio_segment(duration=200)
            chime = chime + Sine(440).to_audio_segment(duration=600)
            chime = chime.fade_in(50).fade_out(400)
            chime = chime - 10  # Lower volume
            chime.export(str(outro_path), format="mp3")
            effects["outro"] = outro_path

        return effects


# Singleton instance
_audio_mixer: Optional[AudioMixer] = None


def get_audio_mixer() -> AudioMixer:
    """Get or create the singleton audio mixer."""
    global _audio_mixer
    if _audio_mixer is None:
        _audio_mixer = AudioMixer()
    return _audio_mixer
