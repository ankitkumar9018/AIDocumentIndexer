"""
AIDocumentIndexer - Text-to-Speech Service
===========================================

Provides text-to-speech capabilities for audio overviews.

Supports multiple providers:
- OpenAI TTS (primary, high quality)
- ElevenLabs (voice cloning, premium)
- Local/Coqui TTS (self-hosted, free)
"""

import asyncio
import io
import os
import tempfile
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, BinaryIO, Union

import structlog
from pydantic import BaseModel

from backend.services.base import BaseService, ServiceException, ProviderException
from backend.core.config import settings

logger = structlog.get_logger(__name__)

# Phase 55: Import audit logging for fallback events
try:
    from backend.services.audit import audit_service_fallback, audit_service_error
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


class TTSProvider(str, Enum):
    """Available TTS providers."""
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    EDGE = "edge"  # Free Microsoft Edge TTS - no API key needed
    CARTESIA = "cartesia"  # Ultra-low latency streaming TTS (40ms TTFA)
    CHATTERBOX = "chatterbox"  # Resemble AI open-source TTS (ultra-realistic)
    COSYVOICE = "cosyvoice"  # Alibaba CosyVoice2 open-source TTS
    LOCAL = "local"
    COQUI = "coqui"  # Alias for local Coqui TTS


class VoiceConfig(BaseModel):
    """Configuration for a voice."""
    provider: TTSProvider
    voice_id: str
    name: str
    speed: float = 1.0
    pitch: float = 1.0
    style: Optional[str] = None  # Provider-specific style


class AudioSegment(BaseModel):
    """A segment of generated audio."""
    speaker: str
    text: str
    audio_data: Optional[bytes] = None
    duration_ms: Optional[int] = None
    file_path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# Default voice mappings
OPENAI_VOICES = {
    "alloy": {"name": "Alloy", "gender": "neutral", "style": "conversational"},
    "echo": {"name": "Echo", "gender": "male", "style": "warm"},
    "fable": {"name": "Fable", "gender": "female", "style": "expressive"},
    "onyx": {"name": "Onyx", "gender": "male", "style": "deep"},
    "nova": {"name": "Nova", "gender": "female", "style": "friendly"},
    "shimmer": {"name": "Shimmer", "gender": "female", "style": "soft"},
}


class BaseTTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Synthesize text to audio bytes."""
        pass

    @abstractmethod
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass


class OpenAITTSProvider(BaseTTSProvider):
    """OpenAI TTS provider using the TTS-1 and TTS-1-HD models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "tts-1"):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model  # tts-1 or tts-1-hd

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Generate speech using OpenAI TTS API."""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self.api_key)

            response = await client.audio.speech.create(
                model=self.model,
                voice=voice_id,
                input=text,
                speed=max(0.25, min(4.0, speed)),
                response_format="mp3",
            )

            return response.content

        except Exception as e:
            logger.error("OpenAI TTS failed", error=str(e))
            raise ProviderException("openai", f"TTS synthesis failed: {str(e)}")

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Return available OpenAI voices."""
        return [
            {"id": voice_id, **info}
            for voice_id, info in OPENAI_VOICES.items()
        ]

    def get_provider_name(self) -> str:
        return "openai"


class ElevenLabsTTSProvider(BaseTTSProvider):
    """
    ElevenLabs TTS provider for premium voice synthesis.

    Phase 32: Enhanced with official SDK support for Flash v2.5 (75ms TTFB)
    and 70+ languages.

    Models:
    - eleven_flash_v2_5: Ultra-low latency (75ms TTFB)
    - eleven_multilingual_v2: High quality, 29 languages
    - eleven_turbo_v2_5: Fast, 32 languages
    """

    # Available ElevenLabs models
    MODELS = {
        "flash": "eleven_flash_v2_5",        # Fastest - 75ms TTFB
        "multilingual": "eleven_multilingual_v2",  # High quality
        "turbo": "eleven_turbo_v2_5",        # Good balance
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "flash",  # Default to fastest model
    ):
        self.api_key = api_key or getattr(settings, "ELEVENLABS_API_KEY", None) or os.getenv("ELEVENLABS_API_KEY")
        self.model_id = self.MODELS.get(model, model)
        self._client = None

    def _get_client(self):
        """Get or create ElevenLabs client."""
        if self._client is None:
            try:
                from elevenlabs import ElevenLabs
                self._client = ElevenLabs(api_key=self.api_key)
            except ImportError:
                logger.warning("elevenlabs SDK not installed, using HTTP fallback")
                return None
        return self._client

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Generate speech using ElevenLabs API."""
        if not self.api_key:
            raise ProviderException("elevenlabs", "ElevenLabs API key not configured")

        # Try official SDK first
        client = self._get_client()
        if client:
            return await self._synthesize_with_sdk(client, text, voice_id, speed, **kwargs)

        # Fallback to HTTP
        return await self._synthesize_with_http(text, voice_id, speed, **kwargs)

    async def _synthesize_with_sdk(
        self,
        client,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Synthesize using official ElevenLabs SDK."""
        try:
            from elevenlabs import VoiceSettings
            import asyncio

            # Map speed to ElevenLabs stability (inverse relationship)
            # Higher speed = lower stability for faster output
            stability = max(0.3, min(1.0, 0.7 / speed if speed > 0 else 0.5))

            voice_settings = VoiceSettings(
                stability=stability,
                similarity_boost=kwargs.get("similarity_boost", 0.75),
                style=kwargs.get("style", 0.0),
                use_speaker_boost=True,
            )

            # Run in thread pool since SDK is sync
            loop = asyncio.get_running_loop()

            def generate():
                audio = client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id=self.model_id,
                    voice_settings=voice_settings,
                    output_format="mp3_44100_128",
                )
                # Convert generator to bytes
                return b"".join(audio)

            audio_data = await loop.run_in_executor(None, generate)
            return audio_data

        except Exception as e:
            logger.error("ElevenLabs SDK synthesis failed", error=str(e))
            # Fallback to HTTP
            return await self._synthesize_with_http(text, voice_id, speed, **kwargs)

    async def _synthesize_with_http(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Fallback HTTP-based synthesis."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": text,
                        "model_id": self.model_id,
                        "voice_settings": {
                            "stability": 0.5,
                            "similarity_boost": 0.75,
                            "style": kwargs.get("style", 0.0),
                            "use_speaker_boost": True,
                        },
                    },
                    timeout=60.0,
                )

                if response.status_code != 200:
                    raise ProviderException(
                        "elevenlabs",
                        f"API error: {response.status_code} - {response.text}",
                    )

                return response.content

        except httpx.RequestError as e:
            raise ProviderException("elevenlabs", f"Request failed: {str(e)}")

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Fetch available voices from ElevenLabs."""
        if not self.api_key:
            return []

        # Try SDK first
        client = self._get_client()
        if client:
            try:
                import asyncio
                loop = asyncio.get_running_loop()

                def fetch_voices():
                    response = client.voices.get_all()
                    return [
                        {
                            "id": voice.voice_id,
                            "name": voice.name,
                            "category": getattr(voice, "category", "custom"),
                            "preview_url": getattr(voice, "preview_url", None),
                            "labels": getattr(voice, "labels", {}),
                        }
                        for voice in response.voices
                    ]

                return await loop.run_in_executor(None, fetch_voices)

            except Exception as e:
                logger.warning("Failed to fetch voices via SDK", error=str(e))

        # Fallback to HTTP
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.elevenlabs.io/v1/voices",
                    headers={"xi-api-key": self.api_key},
                )

                if response.status_code == 200:
                    data = response.json()
                    return [
                        {
                            "id": voice["voice_id"],
                            "name": voice["name"],
                            "category": voice.get("category", "custom"),
                            "preview_url": voice.get("preview_url"),
                        }
                        for voice in data.get("voices", [])
                    ]

        except Exception as e:
            logger.warning("Failed to fetch ElevenLabs voices", error=str(e))

        return []

    def get_provider_name(self) -> str:
        return "elevenlabs"


# Edge TTS voice mappings (free Microsoft voices)
# Updated list based on currently available voices from Microsoft
EDGE_VOICES = {
    # Premium multilingual US voices (recommended - best quality)
    "en-US-AvaMultilingualNeural": {"name": "Ava (Multilingual)", "gender": "female", "style": "natural, warm", "quality": "premium"},
    "en-US-AndrewMultilingualNeural": {"name": "Andrew (Multilingual)", "gender": "male", "style": "natural, friendly", "quality": "premium"},
    "en-US-EmmaMultilingualNeural": {"name": "Emma (Multilingual)", "gender": "female", "style": "natural, clear", "quality": "premium"},
    "en-US-BrianMultilingualNeural": {"name": "Brian (Multilingual)", "gender": "male", "style": "natural, professional", "quality": "premium"},

    # High-quality US voices
    "en-US-AvaNeural": {"name": "Ava", "gender": "female", "style": "warm, conversational", "quality": "high"},
    "en-US-AndrewNeural": {"name": "Andrew", "gender": "male", "style": "friendly, casual", "quality": "high"},
    "en-US-EmmaNeural": {"name": "Emma", "gender": "female", "style": "clear, professional", "quality": "high"},
    "en-US-BrianNeural": {"name": "Brian", "gender": "male", "style": "professional", "quality": "high"},
    "en-US-JennyNeural": {"name": "Jenny", "gender": "female", "style": "friendly, conversational", "quality": "high"},
    "en-US-GuyNeural": {"name": "Guy", "gender": "male", "style": "conversational", "quality": "high"},
    "en-US-AriaNeural": {"name": "Aria", "gender": "female", "style": "professional", "quality": "high"},
    "en-US-ChristopherNeural": {"name": "Christopher", "gender": "male", "style": "warm, authoritative", "quality": "high"},
    "en-US-EricNeural": {"name": "Eric", "gender": "male", "style": "calm, reassuring", "quality": "high"},
    "en-US-MichelleNeural": {"name": "Michelle", "gender": "female", "style": "warm, engaging", "quality": "high"},
    "en-US-RogerNeural": {"name": "Roger", "gender": "male", "style": "energetic", "quality": "high"},
    "en-US-SteffanNeural": {"name": "Steffan", "gender": "male", "style": "casual, friendly", "quality": "high"},
    "en-US-AnaNeural": {"name": "Ana", "gender": "female", "style": "cheerful, young", "quality": "high"},

    # British voices
    "en-GB-SoniaNeural": {"name": "Sonia (UK)", "gender": "female", "style": "professional, british", "quality": "high"},
    "en-GB-RyanNeural": {"name": "Ryan (UK)", "gender": "male", "style": "friendly, british", "quality": "high"},

    # Australian voices
    "en-AU-NatashaNeural": {"name": "Natasha (AU)", "gender": "female", "style": "friendly, australian", "quality": "high"},
    "en-AU-WilliamNeural": {"name": "William (AU)", "gender": "male", "style": "warm, australian", "quality": "high"},
}


# =============================================================================
# Multi-Language Voice Mappings (Phase 2 Enhancement)
# =============================================================================

# Maps language codes to recommended voices per provider
MULTILINGUAL_VOICE_MAP: Dict[str, Dict[str, Dict[str, str]]] = {
    # English
    "en": {
        "openai": {"female": "nova", "male": "onyx", "neutral": "alloy"},
        "edge": {"female": "en-US-JennyNeural", "male": "en-US-GuyNeural"},
        "elevenlabs": {"female": "Rachel", "male": "Adam"},
    },
    # Spanish
    "es": {
        "openai": {"female": "nova", "male": "onyx", "neutral": "alloy"},
        "edge": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
        "elevenlabs": {"female": "Valentina", "male": "Antoni"},
    },
    # French
    "fr": {
        "openai": {"female": "shimmer", "male": "echo", "neutral": "alloy"},
        "edge": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
        "elevenlabs": {"female": "Charlotte", "male": "Thomas"},
    },
    # German
    "de": {
        "openai": {"female": "nova", "male": "echo", "neutral": "alloy"},
        "edge": {"female": "de-DE-KatjaNeural", "male": "de-DE-ConradNeural"},
        "elevenlabs": {"female": "Gisela", "male": "Daniel"},
    },
    # Italian
    "it": {
        "openai": {"female": "shimmer", "male": "onyx", "neutral": "alloy"},
        "edge": {"female": "it-IT-ElsaNeural", "male": "it-IT-DiegoNeural"},
        "elevenlabs": {"female": "Lucia", "male": "Marco"},
    },
    # Portuguese
    "pt": {
        "openai": {"female": "nova", "male": "echo", "neutral": "alloy"},
        "edge": {"female": "pt-BR-FranciscaNeural", "male": "pt-BR-AntonioNeural"},
        "elevenlabs": {"female": "Gabriela", "male": "Pedro"},
    },
    # Dutch
    "nl": {
        "openai": {"female": "shimmer", "male": "onyx", "neutral": "alloy"},
        "edge": {"female": "nl-NL-ColetteNeural", "male": "nl-NL-MaartenNeural"},
        "elevenlabs": {"female": "Rachel", "male": "Adam"},  # Fallback
    },
    # Russian
    "ru": {
        "openai": {"female": "nova", "male": "echo", "neutral": "alloy"},
        "edge": {"female": "ru-RU-SvetlanaNeural", "male": "ru-RU-DmitryNeural"},
        "elevenlabs": {"female": "Rachel", "male": "Adam"},  # Fallback
    },
    # Chinese (Mandarin)
    "zh": {
        "openai": {"female": "nova", "male": "onyx", "neutral": "alloy"},
        "edge": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
        "elevenlabs": {"female": "Lily", "male": "Harry"},
    },
    # Japanese
    "ja": {
        "openai": {"female": "shimmer", "male": "echo", "neutral": "alloy"},
        "edge": {"female": "ja-JP-NanamiNeural", "male": "ja-JP-KeitaNeural"},
        "elevenlabs": {"female": "Hana", "male": "Kenji"},
    },
    # Korean
    "ko": {
        "openai": {"female": "nova", "male": "onyx", "neutral": "alloy"},
        "edge": {"female": "ko-KR-SunHiNeural", "male": "ko-KR-InJoonNeural"},
        "elevenlabs": {"female": "Rachel", "male": "Adam"},  # Fallback
    },
    # Arabic
    "ar": {
        "openai": {"female": "nova", "male": "echo", "neutral": "alloy"},
        "edge": {"female": "ar-SA-ZariyahNeural", "male": "ar-SA-HamedNeural"},
        "elevenlabs": {"female": "Rachel", "male": "Adam"},  # Fallback
    },
    # Hindi
    "hi": {
        "openai": {"female": "shimmer", "male": "onyx", "neutral": "alloy"},
        "edge": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
        "elevenlabs": {"female": "Rachel", "male": "Adam"},  # Fallback
    },
    # Polish
    "pl": {
        "openai": {"female": "nova", "male": "echo", "neutral": "alloy"},
        "edge": {"female": "pl-PL-AgnieszkaNeural", "male": "pl-PL-MarekNeural"},
        "elevenlabs": {"female": "Rachel", "male": "Adam"},  # Fallback
    },
    # Turkish
    "tr": {
        "openai": {"female": "shimmer", "male": "onyx", "neutral": "alloy"},
        "edge": {"female": "tr-TR-EmelNeural", "male": "tr-TR-AhmetNeural"},
        "elevenlabs": {"female": "Rachel", "male": "Adam"},  # Fallback
    },
}


def get_voice_for_language(
    language: str,
    provider: TTSProvider,
    gender: str = "neutral",
) -> str:
    """
    Get the best voice for a language and provider.

    Args:
        language: ISO 639-1 language code (e.g., "en", "es", "fr")
        provider: TTS provider to use
        gender: Preferred gender ("female", "male", "neutral")

    Returns:
        Voice ID for the provider
    """
    # Normalize language code
    lang = language.lower()[:2]

    # Get provider name
    provider_name = provider.value if isinstance(provider, TTSProvider) else str(provider).lower()

    # Look up voice
    lang_voices = MULTILINGUAL_VOICE_MAP.get(lang, MULTILINGUAL_VOICE_MAP["en"])
    provider_voices = lang_voices.get(provider_name, lang_voices.get("edge", {}))

    # Get voice by gender preference
    voice = provider_voices.get(gender)
    if not voice:
        # Fall back to any available voice
        voice = next(iter(provider_voices.values()), None)

    # Final fallback
    if not voice:
        if provider_name == "openai":
            voice = "alloy"
        elif provider_name == "edge":
            voice = "en-US-JennyNeural"
        else:
            voice = "Rachel"

    return voice


# Language names for user display
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese (Mandarin)",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "pl": "Polish",
    "tr": "Turkish",
}


def get_supported_languages() -> List[Dict[str, str]]:
    """Get list of supported languages with display names."""
    return [
        {"code": code, "name": name}
        for code, name in LANGUAGE_NAMES.items()
    ]


class EdgeTTSProvider(BaseTTSProvider):
    """
    Free Microsoft Edge TTS provider.

    Uses edge-tts library which accesses Microsoft's free TTS service.
    No API key required. High quality neural voices.
    """

    def __init__(self):
        self._edge_tts = None

    async def _check_edge_tts(self):
        """Check if edge-tts is installed."""
        try:
            import edge_tts
            return edge_tts
        except ImportError:
            raise ProviderException(
                "edge",
                "edge-tts not installed. Install with: pip install edge-tts",
            )

    def _preprocess_text_for_naturalness(self, text: str) -> str:
        """Convert script markers to more natural speech patterns."""
        import re

        # Convert laughter/chuckle markers to natural sounds
        text = re.sub(r'\[laughs?\]', 'Ha ha ha.', text, flags=re.IGNORECASE)
        text = re.sub(r'\[chuckles?\]', 'Heh heh.', text, flags=re.IGNORECASE)
        text = re.sub(r'\[giggles?\]', 'Hee hee.', text, flags=re.IGNORECASE)

        # Convert other action markers to natural sounds
        text = re.sub(r'\[sighs?\]', 'Ahhh...', text, flags=re.IGNORECASE)
        text = re.sub(r'\[clears throat\]', 'Ahem.', text, flags=re.IGNORECASE)
        text = re.sub(r'\[coughs?\]', 'Ahem ahem.', text, flags=re.IGNORECASE)
        text = re.sub(r'\[pauses?\]', '...', text, flags=re.IGNORECASE)
        text = re.sub(r'\[thinks?\]', 'Hmm...', text, flags=re.IGNORECASE)
        text = re.sub(r'\[surprised\]', 'Oh!', text, flags=re.IGNORECASE)
        text = re.sub(r'\[excited\]', '', text, flags=re.IGNORECASE)

        # Remove any remaining bracketed actions
        text = re.sub(r'\[[^\]]+\]', '', text)

        # Add natural pauses after certain conversational phrases (if not already paused)
        pause_phrases = [
            (r'(Well,)(?!\s*\.\.\.)', r'\1 ...'),
            (r'(You know,)(?!\s*\.\.\.)', r'\1 ...'),
            (r'(I mean,)(?!\s*\.\.\.)', r'\1 ...'),
            (r'(So,)(?!\s*\.\.\.)', r'\1 ...'),
            (r'(Actually,)(?!\s*\.\.\.)', r'\1 ...'),
            (r'(Honestly,)(?!\s*\.\.\.)', r'\1 ...'),
            (r'(Like,)(?!\s*\.\.\.)', r'\1 ...'),
            (r'(Okay so,)(?!\s*\.\.\.)', r'\1 ...'),
            (r"(Here's the thing,)(?!\s*\.\.\.)", r'\1 ...'),
            (r'(But then,)(?!\s*\.\.\.)', r'\1 ...'),
        ]
        for pattern, replacement in pause_phrases:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Add micro-pauses before important words for emphasis (using commas)
        # "it's really important" -> "it's, really important"
        text = re.sub(r'\b(really|very|actually|literally|absolutely|definitely)\b', r', \1', text, count=1)

        # Ensure em-dashes create a pause effect
        text = re.sub(r'—', '... ', text)
        text = re.sub(r'–', '... ', text)

        # Convert multiple periods to proper pause
        text = re.sub(r'\.{2,}', '...', text)

        # Add slight pause before rhetorical questions
        text = re.sub(r'(\?\s*)(Right\?|You know\?|Isn\'t that|Aren\'t they)', r'\1 ... \2', text)

        # Clean up extra spaces but preserve pause markers
        text = re.sub(r'\s+', ' ', text).strip()
        # Ensure pauses have proper spacing
        text = re.sub(r'\s*\.\.\.\s*', ' ... ', text)
        text = re.sub(r'^\s*\.\.\.\s*', '... ', text)

        return text.strip()

    def _get_speech_params_from_emotion(self, emotion: str | None, text: str) -> tuple[str, str]:
        """Get rate and pitch adjustments based on emotion and text content."""
        # Default values
        rate_adjust = 0
        pitch_adjust = 0

        # Adjust based on emotion - expanded for more natural variation
        if emotion:
            emotion_lower = emotion.lower()
            if emotion_lower in ("excited", "enthusiastic", "energetic"):
                rate_adjust = 10  # Faster, more energy
                pitch_adjust = 6  # Higher pitch
            elif emotion_lower in ("curious", "interested", "intrigued"):
                rate_adjust = -3  # Slightly slower
                pitch_adjust = 4  # Slightly higher (questioning tone)
            elif emotion_lower in ("thoughtful", "contemplative", "reflective"):
                rate_adjust = -10  # Slower, more measured
                pitch_adjust = -3
            elif emotion_lower in ("surprised", "amazed", "shocked"):
                rate_adjust = 5
                pitch_adjust = 10  # Much higher for surprise
            elif emotion_lower in ("funny", "amused", "playful"):
                rate_adjust = 5
                pitch_adjust = 5
            elif emotion_lower in ("serious", "concerned", "worried"):
                rate_adjust = -6
                pitch_adjust = -4
            elif emotion_lower in ("skeptical", "doubtful", "uncertain"):
                rate_adjust = -4
                pitch_adjust = 2  # Slight rise for questioning
            elif emotion_lower in ("impressed", "admiring"):
                rate_adjust = -2
                pitch_adjust = 3
            elif emotion_lower in ("confident", "assertive"):
                rate_adjust = 2
                pitch_adjust = -2  # Lower, more authoritative
            elif emotion_lower in ("confused", "puzzled"):
                rate_adjust = -5
                pitch_adjust = 4  # Rising intonation
            elif emotion_lower in ("agreeing", "affirming"):
                rate_adjust = 3
                pitch_adjust = 2
            elif emotion_lower in ("dismissive", "casual"):
                rate_adjust = 5
                pitch_adjust = -1

        # Adjust based on text content
        if text:
            # Questions should have rising intonation (handled by TTS, but we can hint)
            if text.strip().endswith("?"):
                pitch_adjust += 2

            # Exclamations get more energy
            if "!" in text:
                rate_adjust += 2
                pitch_adjust += 2

            # Short reactions are faster
            if len(text.split()) <= 5:
                rate_adjust += 3

            # Long explanations can be slightly slower
            if len(text.split()) > 30:
                rate_adjust -= 3

        # Clamp values to reasonable ranges
        rate_adjust = max(-15, min(15, rate_adjust))
        pitch_adjust = max(-10, min(10, pitch_adjust))

        rate_str = f"+{rate_adjust}%" if rate_adjust >= 0 else f"{rate_adjust}%"
        pitch_str = f"+{pitch_adjust}Hz" if pitch_adjust >= 0 else f"{pitch_adjust}Hz"

        return rate_str, pitch_str

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Generate speech using Edge TTS (free)."""
        edge_tts = await self._check_edge_tts()

        # Get emotion from kwargs for prosody adjustment
        emotion = kwargs.get("emotion") or kwargs.get("style")

        # Preprocess text for more natural speech
        text = self._preprocess_text_for_naturalness(text)

        # Map deprecated/unavailable voices to working alternatives
        VOICE_FALLBACKS = {
            # Deprecated voices -> working alternatives
            "en-US-TonyNeural": "en-US-GuyNeural",
            "en-US-SaraNeural": "en-US-JennyNeural",
            "en-US-DavisNeural": "en-US-EricNeural",
            "en-US-JaneNeural": "en-US-MichelleNeural",
            "en-US-JasonNeural": "en-US-RogerNeural",
            "en-US-NancyNeural": "en-US-AriaNeural",
            "en-GB-LibbyNeural": "en-GB-SoniaNeural",
            # OpenAI voice names -> Edge equivalents
            "alloy": "en-US-AndrewMultilingualNeural",
            "echo": "en-US-GuyNeural",
            "fable": "en-US-AriaNeural",
            "onyx": "en-US-ChristopherNeural",
            "nova": "en-US-AvaMultilingualNeural",
            "shimmer": "en-US-JennyNeural",
            # Generic names
            "host1": "en-US-AndrewMultilingualNeural",
            "host2": "en-US-AvaMultilingualNeural",
        }

        # Apply fallback if voice is deprecated or not available
        if voice_id in VOICE_FALLBACKS:
            voice_id = VOICE_FALLBACKS[voice_id]
        elif voice_id not in EDGE_VOICES:
            # Default to a reliable voice if unknown
            voice_id = "en-US-JennyNeural"

        # Get emotion-based speech parameters
        rate_str, pitch_str = self._get_speech_params_from_emotion(emotion, text)

        # Apply base speed adjustment on top of emotion
        base_rate_percent = int((speed - 1.0) * 100)
        if base_rate_percent != 0:
            # Parse current rate and add base speed
            current_rate = int(rate_str.replace("%", "").replace("+", ""))
            combined_rate = current_rate + base_rate_percent
            rate_str = f"+{combined_rate}%" if combined_rate >= 0 else f"{combined_rate}%"

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name

        try:
            communicate = edge_tts.Communicate(text, voice_id, rate=rate_str, pitch=pitch_str)
            await communicate.save(temp_path)

            with open(temp_path, "rb") as f:
                audio_data = f.read()

            return audio_data

        except Exception as e:
            logger.error("Edge TTS failed", error=str(e))
            raise ProviderException("edge", f"TTS synthesis failed: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Return available Edge TTS voices."""
        return [
            {"id": voice_id, **info}
            for voice_id, info in EDGE_VOICES.items()
        ]

    def get_provider_name(self) -> str:
        return "edge"


class LocalTTSProvider(BaseTTSProvider):
    """Local TTS provider using system TTS or Coqui TTS."""

    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"):
        self.model_name = model_name
        self._tts = None

    async def _get_tts(self):
        """Lazy-load TTS model."""
        if self._tts is None:
            try:
                from TTS.api import TTS
                self._tts = TTS(model_name=self.model_name, progress_bar=False)
            except ImportError:
                raise ProviderException(
                    "local",
                    "Coqui TTS not installed. Install with: pip install TTS",
                )
            except Exception as e:
                raise ProviderException("local", f"Failed to load TTS model: {str(e)}")
        return self._tts

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Generate speech using local TTS."""
        tts = await self._get_tts()

        # Generate to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        try:
            # Run in thread pool to not block
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: tts.tts_to_file(text=text, file_path=temp_path, speed=speed),
            )

            # Read the audio data
            with open(temp_path, "rb") as f:
                audio_data = f.read()

            return audio_data

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Return available local voices/models."""
        return [
            {"id": "default", "name": "Default Voice", "provider": "local"},
        ]

    def get_provider_name(self) -> str:
        return "local"


class TTSService(BaseService):
    """
    Text-to-Speech service for generating audio from text.

    Supports multiple providers with automatic fallback.
    """

    def __init__(
        self,
        session=None,
        organization_id=None,
        user_id=None,
        default_provider: TTSProvider = TTSProvider.OPENAI,
    ):
        super().__init__(session, organization_id, user_id)
        self.default_provider = default_provider
        self._providers: Dict[TTSProvider, BaseTTSProvider] = {}

    def _get_provider(self, provider: TTSProvider) -> BaseTTSProvider:
        """Get or create a TTS provider instance."""
        if provider not in self._providers:
            if provider == TTSProvider.OPENAI:
                self._providers[provider] = OpenAITTSProvider()
            elif provider == TTSProvider.ELEVENLABS:
                self._providers[provider] = ElevenLabsTTSProvider()
            elif provider == TTSProvider.EDGE:
                self._providers[provider] = EdgeTTSProvider()
            elif provider == TTSProvider.CARTESIA:
                # Ultra-low latency Cartesia TTS (40ms TTFA)
                from backend.services.audio.cartesia_tts import CartesiaTTSProvider
                self._providers[provider] = CartesiaTTSProvider()
            elif provider == TTSProvider.CHATTERBOX:
                # Resemble AI Chatterbox - ultra-realistic open-source TTS
                from backend.services.audio.ultra_fast_tts import ChatterboxTTS, UltraFastTTSConfig
                config = UltraFastTTSConfig()
                self._providers[provider] = ChatterboxTTS(config)
            elif provider == TTSProvider.COSYVOICE:
                # Alibaba CosyVoice2 - open-source streaming TTS
                from backend.services.audio.ultra_fast_tts import CosyVoiceTTS, UltraFastTTSConfig
                config = UltraFastTTSConfig()
                self._providers[provider] = CosyVoiceTTS(config)
            elif provider in (TTSProvider.LOCAL, TTSProvider.COQUI):
                # Both LOCAL and COQUI use the local Coqui TTS provider
                self._providers[provider] = LocalTTSProvider()
            else:
                raise ServiceException(f"Unknown TTS provider: {provider}")

        return self._providers[provider]

    async def synthesize_text(
        self,
        text: str,
        voice_id: str,
        provider: Optional[TTSProvider] = None,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            provider: TTS provider to use
            speed: Speech speed multiplier
            **kwargs: Provider-specific options

        Returns:
            Audio data as bytes (MP3 for OpenAI/ElevenLabs, WAV for local)
        """
        provider = provider or self.default_provider

        self.log_debug(
            "Synthesizing text",
            provider=provider.value,
            voice_id=voice_id,
            text_length=len(text),
        )

        tts_provider = self._get_provider(provider)

        try:
            audio_data = await tts_provider.synthesize(
                text=text,
                voice_id=voice_id,
                speed=speed,
                **kwargs,
            )

            self.log_debug(
                "Synthesis complete",
                audio_size=len(audio_data),
            )

            return audio_data

        except ProviderException:
            raise
        except Exception as e:
            self.log_error("TTS synthesis failed", error=e, provider=provider.value)
            raise ServiceException(
                f"TTS synthesis failed: {str(e)}",
                code="TTS_ERROR",
            )

    async def synthesize_parallel(
        self,
        segments: List[Dict[str, Any]],
        speaker_voices: Dict[str, VoiceConfig],
        max_concurrent: int = 5,
    ) -> List[AudioSegment]:
        """
        Synthesize multiple audio segments in parallel for 3-5x speedup.

        Args:
            segments: List of dicts with 'speaker', 'text', and optional 'emotion' keys
            speaker_voices: Mapping of speaker IDs to voice configs
            max_concurrent: Maximum concurrent TTS requests (default 5)

        Returns:
            List of AudioSegment objects with audio_data populated
        """
        self.log_info(
            "Synthesizing segments in parallel",
            segment_count=len(segments),
            max_concurrent=max_concurrent,
        )

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def synthesize_segment(idx: int, segment: Dict[str, Any]) -> AudioSegment:
            """Synthesize a single segment with concurrency limiting."""
            async with semaphore:
                speaker = segment.get("speaker", "default")
                text = segment.get("text", "")
                emotion = segment.get("emotion")

                if not text.strip():
                    return AudioSegment(speaker=speaker, text=text)

                # Get voice config
                voice_config = speaker_voices.get(speaker)
                if not voice_config:
                    voice_config = VoiceConfig(
                        provider=self.default_provider,
                        voice_id="alloy" if self.default_provider == TTSProvider.OPENAI else "default",
                        name=speaker,
                    )

                try:
                    provider = self._get_provider(voice_config.provider)
                    audio_data = await provider.synthesize(
                        text=text,
                        voice_id=voice_config.voice_id,
                        speed=voice_config.speed,
                        style=voice_config.style,
                        emotion=emotion,
                    )

                    return AudioSegment(
                        speaker=speaker,
                        text=text,
                        audio_data=audio_data,
                    )

                except Exception as e:
                    self.log_warning(
                        f"TTS failed for segment {idx}, attempting fallback",
                        error=str(e),
                        speaker=speaker,
                    )
                    # Try fallback provider (Edge TTS is free and reliable)
                    return await self._synthesize_with_fallback(segment, voice_config, str(e))

        # Create all synthesis tasks
        tasks = [
            synthesize_segment(idx, segment)
            for idx, segment in enumerate(segments)
        ]

        # Execute in parallel with gather
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results - handle any exceptions
        audio_segments = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                self.log_error(
                    f"Segment {idx} synthesis failed completely",
                    error=str(result),
                )
                # Create empty segment placeholder to maintain order
                audio_segments.append(AudioSegment(
                    speaker=segments[idx].get("speaker", "default"),
                    text=segments[idx].get("text", ""),
                    audio_data=None,
                ))
            else:
                audio_segments.append(result)

        self.log_info(
            "Parallel synthesis complete",
            total_segments=len(audio_segments),
            successful=sum(1 for s in audio_segments if s.audio_data),
        )

        return audio_segments

    async def _synthesize_with_fallback(
        self,
        segment: Dict[str, Any],
        original_config: VoiceConfig,
        original_error: str = "",
    ) -> AudioSegment:
        """
        Attempt synthesis with fallback provider (Edge TTS).

        Args:
            segment: Segment to synthesize
            original_config: Original voice config that failed
            original_error: Error message from the original provider

        Returns:
            AudioSegment with audio from fallback provider
        """
        speaker = segment.get("speaker", "default")
        text = segment.get("text", "")
        emotion = segment.get("emotion")

        try:
            # Try Edge TTS as fallback (free, reliable)
            edge_provider = self._get_provider(TTSProvider.EDGE)
            audio_data = await edge_provider.synthesize(
                text=text,
                voice_id="en-US-JennyNeural",  # Reliable default
                speed=original_config.speed,
                emotion=emotion,
            )

            self.log_info(
                "Fallback synthesis successful",
                speaker=speaker,
                provider="edge",
            )

            # Phase 55: Log TTS fallback to audit system
            if AUDIT_AVAILABLE:
                try:
                    import asyncio
                    asyncio.create_task(audit_service_fallback(
                        service_type="tts",
                        primary_provider=original_config.provider.value,
                        fallback_provider="edge",
                        error_message=original_error,
                        context={"speaker": speaker, "text_length": len(text)},
                        user_id=self.user_id,
                    ))
                except Exception:
                    pass  # Don't let audit logging break TTS

            return AudioSegment(
                speaker=speaker,
                text=text,
                audio_data=audio_data,
            )

        except Exception as e:
            self.log_error(
                "Fallback synthesis also failed",
                error=str(e),
            )

            # Phase 55: Log TTS error when all fallbacks exhausted
            if AUDIT_AVAILABLE:
                try:
                    import asyncio
                    asyncio.create_task(audit_service_error(
                        service_type="tts",
                        provider=f"{original_config.provider.value}, edge",
                        error_message=f"Primary: {original_error}, Fallback: {str(e)}",
                        context={"speaker": speaker, "text_length": len(text)},
                        user_id=self.user_id,
                    ))
                except Exception:
                    pass  # Don't let audit logging break TTS

            return AudioSegment(speaker=speaker, text=text, audio_data=None)

    async def synthesize_dialogue(
        self,
        turns: List[Dict[str, Any]],
        speaker_voices: Dict[str, VoiceConfig],
        output_path: Optional[str] = None,
        add_pauses: bool = True,
        pause_between_speakers_ms: int = 500,
        parallel: bool = True,  # Use parallel synthesis by default (3-5x faster)
        max_concurrent: int = 5,
    ) -> Union[bytes, str]:
        """
        Synthesize a multi-speaker dialogue.

        Args:
            turns: List of dialogue turns with 'speaker' and 'text' keys
            speaker_voices: Mapping of speaker IDs to voice configs
            output_path: If provided, save to file and return path
            add_pauses: Whether to add pauses between speakers
            pause_between_speakers_ms: Pause duration between different speakers
            parallel: Whether to synthesize segments in parallel (default True, 3-5x faster)
            max_concurrent: Maximum concurrent TTS requests when parallel=True

        Returns:
            Combined audio bytes or file path
        """
        self.log_info(
            "Synthesizing dialogue",
            turn_count=len(turns),
            speakers=list(speaker_voices.keys()),
            parallel=parallel,
        )

        # Filter out empty turns
        valid_turns = [t for t in turns if t.get("text", "").strip()]

        if parallel and len(valid_turns) > 1:
            # Use parallel synthesis for 3-5x speedup
            segments = await self.synthesize_parallel(
                segments=valid_turns,
                speaker_voices=speaker_voices,
                max_concurrent=max_concurrent,
            )
        else:
            # Sequential synthesis (fallback or single segment)
            segments = []
            for turn in valid_turns:
                speaker = turn.get("speaker", "default")
                text = turn.get("text", "")
                emotion = turn.get("emotion")  # Get emotion for prosody variation

                # Get voice config
                voice_config = speaker_voices.get(speaker)
                if not voice_config:
                    self.log_warning(f"No voice config for speaker: {speaker}, using default")
                    voice_config = VoiceConfig(
                        provider=self.default_provider,
                        voice_id="alloy" if self.default_provider == TTSProvider.OPENAI else "default",
                        name=speaker,
                    )

                # Generate audio for this turn with emotion for prosody variation
                provider = self._get_provider(voice_config.provider)
                audio_data = await provider.synthesize(
                    text=text,
                    voice_id=voice_config.voice_id,
                    speed=voice_config.speed,
                    style=voice_config.style,
                    emotion=emotion,  # Pass emotion for pitch/rate variation
                )

                segments.append(AudioSegment(
                    speaker=speaker,
                    text=text,
                    audio_data=audio_data,
                ))

        # Combine segments
        combined_audio = await self._combine_audio_segments(
            segments,
            add_pauses=add_pauses,
            pause_ms=pause_between_speakers_ms,
        )

        if output_path:
            # Save to file
            with open(output_path, "wb") as f:
                f.write(combined_audio)
            return output_path

        return combined_audio

    async def _combine_audio_segments(
        self,
        segments: List[AudioSegment],
        add_pauses: bool = True,
        pause_ms: int = 500,
    ) -> bytes:
        """Combine multiple audio segments into a single file with natural pauses."""
        try:
            from pydub import AudioSegment as PydubSegment
            import random

            combined = PydubSegment.empty()

            # Different pause durations for natural feel
            speaker_change_pause_ms = pause_ms + 200  # Longer pause when speaker changes
            same_speaker_pause_ms = 150  # Short breathing pause between same speaker turns
            paragraph_pause_ms = pause_ms + 400  # Extra pause after longer segments

            last_speaker = None
            turn_count = 0

            for i, segment in enumerate(segments):
                if segment.audio_data:
                    # Detect format from audio data
                    audio = PydubSegment.from_mp3(io.BytesIO(segment.audio_data))

                    if add_pauses and i > 0:
                        if last_speaker and last_speaker != segment.speaker:
                            # Speaker change - add longer pause with slight variation
                            pause_duration = speaker_change_pause_ms + random.randint(-50, 100)
                            combined += PydubSegment.silent(duration=pause_duration)
                        else:
                            # Same speaker continuing - add short breathing pause
                            pause_duration = same_speaker_pause_ms + random.randint(-30, 50)
                            combined += PydubSegment.silent(duration=pause_duration)

                    combined += audio
                    last_speaker = segment.speaker
                    turn_count += 1

                    # Add extra pause every few turns to simulate natural conversation rhythm
                    if add_pauses and turn_count % 4 == 0 and i < len(segments) - 1:
                        combined += PydubSegment.silent(duration=random.randint(100, 250))

            # Export as MP3
            buffer = io.BytesIO()
            combined.export(buffer, format="mp3", bitrate="192k")
            return buffer.getvalue()

        except ImportError:
            self.log_warning("pydub not available, returning first segment only")
            # Fallback: just concatenate bytes (won't work properly but avoids crash)
            return b"".join(s.audio_data for s in segments if s.audio_data)

    async def get_voices(
        self,
        provider: Optional[TTSProvider] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get available voices from providers.

        Args:
            provider: Specific provider, or None for all

        Returns:
            Dict mapping provider names to voice lists
        """
        result = {}

        providers_to_check = [provider] if provider else list(TTSProvider)

        for p in providers_to_check:
            try:
                tts_provider = self._get_provider(p)
                voices = await tts_provider.get_available_voices()
                result[p.value] = voices
            except Exception as e:
                self.log_warning(f"Failed to get voices for {p.value}", error=str(e))
                result[p.value] = []

        return result

    async def estimate_cost(
        self,
        text: str,
        provider: TTSProvider = TTSProvider.OPENAI,
    ) -> Dict[str, float]:
        """
        Estimate the cost of synthesizing text.

        Returns dict with provider-specific cost estimates.
        """
        char_count = len(text)

        # Rough cost estimates (as of 2024-2025)
        costs = {
            "openai_tts1": char_count * 0.000015,  # $15 per 1M chars
            "openai_tts1_hd": char_count * 0.00003,  # $30 per 1M chars
            "elevenlabs": char_count * 0.00018,  # ~$0.18 per 1K chars
            "cartesia": char_count * 0.000025,  # ~$25 per 1M chars (Sonic 2.0)
            "chatterbox": 0.0,  # Free (open-source, local compute only)
            "cosyvoice": 0.0,  # Free (open-source, local compute only)
            "edge": 0.0,  # Free (Microsoft Edge TTS)
            "local": 0.0,  # Free (compute cost only)
            "coqui": 0.0,  # Free (local Coqui TTS)
        }

        # Map provider to cost key
        cost_key = provider.value
        if provider == TTSProvider.OPENAI:
            cost_key = "openai_tts1"

        return {
            "character_count": char_count,
            "estimated_cost_usd": costs.get(cost_key, 0.0),
            "costs_by_provider": costs,
        }

    # -------------------------------------------------------------------------
    # Multi-Language Support (Phase 2 Enhancement)
    # -------------------------------------------------------------------------

    async def synthesize_multilingual(
        self,
        text: str,
        target_language: str,
        provider: Optional[TTSProvider] = None,
        gender: str = "neutral",
        speed: float = 1.0,
        translate_first: bool = False,
    ) -> bytes:
        """
        Synthesize audio in a target language.

        Automatically selects the best voice for the language
        and optionally translates the text first.

        Args:
            text: Text to synthesize
            target_language: ISO 639-1 language code (e.g., "en", "es", "fr")
            provider: TTS provider (None uses default)
            gender: Voice gender preference ("female", "male", "neutral")
            speed: Speech speed (0.5-2.0)
            translate_first: Whether to translate text to target language first

        Returns:
            Audio bytes (MP3 format)
        """
        actual_provider = provider or self.default_provider

        # Get appropriate voice for language
        voice_id = get_voice_for_language(
            language=target_language,
            provider=actual_provider,
            gender=gender,
        )

        self.log_info(
            "Synthesizing multilingual audio",
            language=target_language,
            provider=actual_provider.value,
            voice=voice_id,
            translate=translate_first,
        )

        # Optionally translate text
        if translate_first and target_language.lower()[:2] != "en":
            text = await self._translate_text(text, target_language)

        # Synthesize with selected voice
        return await self.synthesize_text(
            text=text,
            voice_id=voice_id,
            provider=actual_provider,
            speed=speed,
        )

    async def _translate_text(
        self,
        text: str,
        target_language: str,
    ) -> str:
        """
        Translate text to target language using LLM.

        Args:
            text: Text to translate
            target_language: Target language code

        Returns:
            Translated text
        """
        try:
            from backend.services.llm import EnhancedLLMFactory

            llm, _ = await EnhancedLLMFactory.get_chat_model_for_operation(
                operation="translation",
                user_id=None,
            )

            lang_name = LANGUAGE_NAMES.get(target_language.lower()[:2], target_language)

            prompt = f"""Translate the following text to {lang_name}.
Maintain the tone and style of the original text.
Only output the translation, nothing else.

Text to translate:
{text}"""

            response = await llm.ainvoke(prompt)
            return response.content.strip()

        except Exception as e:
            self.log_warning(f"Translation failed: {e}, using original text")
            return text

    async def synthesize_dialogue_multilingual(
        self,
        dialogue_turns: List[Dict[str, Any]],
        target_language: str,
        speaker_genders: Optional[Dict[str, str]] = None,
        provider: Optional[TTSProvider] = None,
        translate_first: bool = False,
        output_path: Optional[str] = None,
    ) -> Union[bytes, str]:
        """
        Synthesize a multi-speaker dialogue in a target language.

        Args:
            dialogue_turns: List of {"speaker": str, "text": str} dicts
            target_language: ISO 639-1 language code
            speaker_genders: Optional mapping of speaker names to genders
            provider: TTS provider (None uses default)
            translate_first: Whether to translate texts first
            output_path: Optional path to save audio file

        Returns:
            Audio bytes or file path
        """
        actual_provider = provider or self.default_provider
        speaker_genders = speaker_genders or {}

        # Build voice configs for each speaker
        speaker_voices: Dict[str, VoiceConfig] = {}
        speakers = list(set(turn.get("speaker", "default") for turn in dialogue_turns))

        # Alternate genders if not specified
        default_genders = ["female", "male"]

        for i, speaker in enumerate(speakers):
            gender = speaker_genders.get(speaker, default_genders[i % 2])
            voice_id = get_voice_for_language(
                language=target_language,
                provider=actual_provider,
                gender=gender,
            )
            speaker_voices[speaker] = VoiceConfig(
                provider=actual_provider,
                voice_id=voice_id,
                name=speaker,
            )

        # Optionally translate all texts
        if translate_first and target_language.lower()[:2] != "en":
            dialogue_turns = [
                {
                    **turn,
                    "text": await self._translate_text(turn.get("text", ""), target_language),
                }
                for turn in dialogue_turns
            ]

        # Use existing dialogue synthesis
        return await self.synthesize_dialogue(
            dialogue_turns=dialogue_turns,
            speaker_voices=speaker_voices,
            output_path=output_path,
        )

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of languages supported for TTS."""
        return get_supported_languages()
