"""
AIDocumentIndexer - Cartesia TTS Service
==========================================

Ultra-low latency streaming Text-to-Speech using Cartesia Sonic 2.0.

Key Features:
- 40ms Time-to-First-Audio (TTFA) - industry leading
- Real-time streaming via WebSocket
- Preview generation (first 30s immediately)
- Redis caching with 7-day TTL
- Emotion and prosody control

Based on Cartesia Sonic 2.0 (2024-2025):
- Lowest latency in the market
- High-quality neural voices
- Multilingual support
- Voice cloning capabilities

Usage:
    from backend.services.audio.cartesia_tts import CartesiaTTSProvider, CartesiaStreamingTTS

    # Basic synthesis
    provider = CartesiaTTSProvider()
    audio = await provider.synthesize("Hello world", voice_id="sonic-english")

    # Streaming for real-time playback
    streaming = CartesiaStreamingTTS()
    async for chunk in streaming.stream("Long text here...", voice_id="sonic-english"):
        await websocket.send_bytes(chunk)
"""

import asyncio
import hashlib
import io
import json
import struct
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import structlog

from backend.core.config import settings
from backend.services.audio.tts_service import BaseTTSProvider
from backend.services.base import ProviderException

logger = structlog.get_logger(__name__)


# =============================================================================
# Cartesia Voice Mappings
# =============================================================================

class CartesiaVoiceStyle(str, Enum):
    """Voice styles available in Cartesia."""
    NEUTRAL = "neutral"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ENTHUSIASTIC = "enthusiastic"
    CALM = "calm"
    SERIOUS = "serious"


# Cartesia's pre-built voices (Sonic 2.0)
CARTESIA_VOICES = {
    # English - US
    "sonic-english-male-1": {
        "name": "Alex (US)",
        "gender": "male",
        "language": "en",
        "locale": "en-US",
        "style": "conversational",
        "description": "Warm, friendly male voice for podcasts",
    },
    "sonic-english-female-1": {
        "name": "Sarah (US)",
        "gender": "female",
        "language": "en",
        "locale": "en-US",
        "style": "professional",
        "description": "Clear, professional female voice",
    },
    "sonic-english-male-2": {
        "name": "Marcus (US)",
        "gender": "male",
        "language": "en",
        "locale": "en-US",
        "style": "neutral",
        "description": "Neutral, authoritative male voice",
    },
    "sonic-english-female-2": {
        "name": "Emily (US)",
        "gender": "female",
        "language": "en",
        "locale": "en-US",
        "style": "friendly",
        "description": "Warm, engaging female voice",
    },
    # English - UK
    "sonic-english-uk-male": {
        "name": "James (UK)",
        "gender": "male",
        "language": "en",
        "locale": "en-GB",
        "style": "professional",
        "description": "British male voice",
    },
    "sonic-english-uk-female": {
        "name": "Charlotte (UK)",
        "gender": "female",
        "language": "en",
        "locale": "en-GB",
        "style": "conversational",
        "description": "British female voice",
    },
    # Multilingual voices
    "sonic-multilingual-male": {
        "name": "Global (Male)",
        "gender": "male",
        "language": "multilingual",
        "locale": "multi",
        "style": "neutral",
        "description": "Multilingual male voice",
    },
    "sonic-multilingual-female": {
        "name": "Global (Female)",
        "gender": "female",
        "language": "multilingual",
        "locale": "multi",
        "style": "neutral",
        "description": "Multilingual female voice",
    },
}

# Voice fallback mapping from other providers
VOICE_FALLBACK_MAP = {
    # OpenAI voices
    "alloy": "sonic-english-male-1",
    "echo": "sonic-english-male-2",
    "fable": "sonic-english-female-1",
    "onyx": "sonic-english-male-2",
    "nova": "sonic-english-female-2",
    "shimmer": "sonic-english-female-1",
    # Generic mappings
    "male": "sonic-english-male-1",
    "female": "sonic-english-female-1",
    "default": "sonic-english-male-1",
}


@dataclass
class CartesiaConfig:
    """Configuration for Cartesia TTS."""
    api_key: str = ""
    api_base_url: str = "https://api.cartesia.ai"
    ws_url: str = "wss://api.cartesia.ai/tts/websocket"
    model_id: str = "sonic-2"  # Sonic 2.0
    output_format: str = "mp3"  # mp3, wav, raw
    sample_rate: int = 24000
    bit_depth: int = 16
    container: str = "mp3"

    # Streaming settings
    chunk_size: int = 4096
    buffer_ms: int = 50  # Buffer before streaming starts

    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 604800  # 7 days

    def __post_init__(self):
        if not self.api_key:
            self.api_key = settings.CARTESIA_API_KEY


@dataclass
class StreamingMetrics:
    """Metrics for streaming TTS."""
    start_time: float = 0.0
    first_byte_time: float = 0.0
    total_bytes: int = 0
    chunk_count: int = 0
    duration_ms: float = 0.0

    @property
    def ttfa_ms(self) -> float:
        """Time to first audio in milliseconds."""
        if self.first_byte_time and self.start_time:
            return (self.first_byte_time - self.start_time) * 1000
        return 0.0


# =============================================================================
# Redis Cache for TTS
# =============================================================================

class TTSCache:
    """
    Redis-based cache for TTS audio with 7-day TTL.

    Caches full audio and preview segments separately.
    """

    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: int = 604800):
        self.redis_url = redis_url or settings.REDIS_URL
        self.ttl_seconds = ttl_seconds
        self._redis = None

    async def _get_redis(self):
        """Lazy-load Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False,  # Binary data
                )
            except Exception as e:
                logger.warning(f"Redis connection failed for TTS cache: {e}")
                return None
        return self._redis

    def _make_key(self, text: str, voice_id: str, params: Dict[str, Any]) -> str:
        """Create cache key from text and parameters."""
        param_str = json.dumps(params, sort_keys=True)
        content = f"{text}|{voice_id}|{param_str}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"tts:cartesia:{hash_val}"

    async def get(
        self,
        text: str,
        voice_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[bytes]:
        """Get cached audio if available."""
        redis = await self._get_redis()
        if not redis:
            return None

        key = self._make_key(text, voice_id, params or {})
        try:
            data = await redis.get(key)
            if data:
                logger.debug("TTS cache hit", key=key[:20])
            return data
        except Exception as e:
            logger.warning(f"TTS cache get failed: {e}")
            return None

    async def set(
        self,
        text: str,
        voice_id: str,
        audio_data: bytes,
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Cache audio data."""
        redis = await self._get_redis()
        if not redis:
            return False

        key = self._make_key(text, voice_id, params or {})
        try:
            await redis.setex(key, self.ttl_seconds, audio_data)
            logger.debug("TTS cached", key=key[:20], size=len(audio_data))
            return True
        except Exception as e:
            logger.warning(f"TTS cache set failed: {e}")
            return False

    async def get_preview(
        self,
        text: str,
        voice_id: str,
        preview_seconds: int = 30,
    ) -> Optional[bytes]:
        """Get cached preview audio."""
        key = f"tts:preview:{hashlib.sha256(f'{text}|{voice_id}'.encode()).hexdigest()[:16]}"
        redis = await self._get_redis()
        if not redis:
            return None
        try:
            return await redis.get(key)
        except Exception:
            return None

    async def set_preview(
        self,
        text: str,
        voice_id: str,
        preview_data: bytes,
        preview_seconds: int = 30,
    ) -> bool:
        """Cache preview audio."""
        key = f"tts:preview:{hashlib.sha256(f'{text}|{voice_id}'.encode()).hexdigest()[:16]}"
        redis = await self._get_redis()
        if not redis:
            return False
        try:
            await redis.setex(key, self.ttl_seconds, preview_data)
            return True
        except Exception:
            return False


# =============================================================================
# Cartesia TTS Provider (HTTP API)
# =============================================================================

class CartesiaTTSProvider(BaseTTSProvider):
    """
    Cartesia TTS provider using HTTP API.

    Suitable for:
    - Batch audio generation
    - When streaming is not required
    - Generating complete audio files
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "sonic-2",
        enable_cache: bool = True,
    ):
        self.config = CartesiaConfig(api_key=api_key or "", model_id=model_id)
        self.cache = TTSCache() if enable_cache else None

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        **kwargs,
    ) -> bytes:
        """
        Synthesize text to audio using Cartesia HTTP API.

        Args:
            text: Text to synthesize
            voice_id: Cartesia voice ID or fallback voice name
            speed: Speech speed (0.5-2.0)
            **kwargs: Additional params (emotion, pitch, etc.)

        Returns:
            Audio data as bytes (MP3 format)
        """
        if not self.config.api_key:
            raise ProviderException("cartesia", "Cartesia API key not configured")

        # Map voice ID if needed
        actual_voice_id = VOICE_FALLBACK_MAP.get(voice_id, voice_id)
        if actual_voice_id not in CARTESIA_VOICES:
            actual_voice_id = "sonic-english-male-1"  # Default fallback

        # Build params for caching
        params = {
            "speed": speed,
            "emotion": kwargs.get("emotion"),
            "pitch": kwargs.get("pitch"),
        }

        # Check cache first
        if self.cache:
            cached = await self.cache.get(text, actual_voice_id, params)
            if cached:
                return cached

        try:
            import httpx

            # Build request payload
            payload = {
                "model_id": self.config.model_id,
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": actual_voice_id,
                },
                "output_format": {
                    "container": self.config.container,
                    "sample_rate": self.config.sample_rate,
                    "encoding": "mp3" if self.config.container == "mp3" else "pcm_s16le",
                },
            }

            # Add optional parameters
            if speed != 1.0:
                payload["voice"]["speed"] = max(0.5, min(2.0, speed))

            emotion = kwargs.get("emotion")
            if emotion:
                payload["voice"]["emotion"] = self._map_emotion(emotion)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config.api_base_url}/tts/bytes",
                    headers={
                        "X-API-Key": self.config.api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )

                if response.status_code != 200:
                    error_detail = response.text[:200]
                    raise ProviderException(
                        "cartesia",
                        f"API error {response.status_code}: {error_detail}",
                    )

                audio_data = response.content

            # Cache the result
            if self.cache:
                await self.cache.set(text, actual_voice_id, audio_data, params)

            logger.info(
                "Cartesia synthesis complete",
                text_length=len(text),
                audio_size=len(audio_data),
                voice=actual_voice_id,
            )

            return audio_data

        except httpx.RequestError as e:
            raise ProviderException("cartesia", f"Request failed: {str(e)}")
        except Exception as e:
            if isinstance(e, ProviderException):
                raise
            raise ProviderException("cartesia", f"Synthesis failed: {str(e)}")

    def _map_emotion(self, emotion: str) -> List[Dict[str, Any]]:
        """Map emotion string to Cartesia emotion controls."""
        emotion_map = {
            "excited": [{"name": "positivity", "level": "high"}, {"name": "surprise", "level": "medium"}],
            "curious": [{"name": "curiosity", "level": "high"}],
            "thoughtful": [{"name": "sadness", "level": "low"}],
            "surprised": [{"name": "surprise", "level": "high"}],
            "funny": [{"name": "positivity", "level": "high"}],
            "serious": [{"name": "anger", "level": "low"}],
            "happy": [{"name": "positivity", "level": "high"}],
            "sad": [{"name": "sadness", "level": "medium"}],
            "neutral": [],
        }
        return emotion_map.get(emotion.lower(), [])

    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Return available Cartesia voices."""
        return [
            {"id": voice_id, **info}
            for voice_id, info in CARTESIA_VOICES.items()
        ]

    def get_provider_name(self) -> str:
        return "cartesia"


# =============================================================================
# Cartesia Streaming TTS (WebSocket)
# =============================================================================

class CartesiaStreamingTTS:
    """
    Cartesia streaming TTS using WebSocket for ultra-low latency.

    Key Features:
    - 40ms Time-to-First-Audio (TTFA)
    - Real-time streaming for live playback
    - Preview generation (first 30s immediately)
    - Sentence-level chunking for natural pauses

    Usage:
        streaming = CartesiaStreamingTTS()

        # Full streaming
        async for chunk in streaming.stream("Long text...", "sonic-english-male-1"):
            await websocket.send_bytes(chunk)

        # Preview only (first 30 seconds)
        preview = await streaming.generate_preview("Long text...", "sonic-english-male-1")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
    ):
        self.config = CartesiaConfig(api_key=api_key or "")
        self.cache = TTSCache() if enable_cache else None
        self.metrics = StreamingMetrics()

    async def stream(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        emotion: Optional[str] = None,
        on_first_chunk: Optional[callable] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks in real-time via WebSocket.

        Args:
            text: Text to synthesize
            voice_id: Voice ID
            speed: Speech speed
            emotion: Optional emotion
            on_first_chunk: Callback when first chunk arrives (for metrics)

        Yields:
            Audio chunks as bytes
        """
        if not self.config.api_key:
            raise ProviderException("cartesia", "Cartesia API key not configured")

        # Map voice ID
        actual_voice_id = VOICE_FALLBACK_MAP.get(voice_id, voice_id)
        if actual_voice_id not in CARTESIA_VOICES:
            actual_voice_id = "sonic-english-male-1"

        # Reset metrics
        self.metrics = StreamingMetrics(start_time=time.time())

        try:
            import websockets

            # Build WebSocket URL with API key
            ws_url = f"{self.config.ws_url}?api_key={self.config.api_key}&cartesia_version=2024-06-10"

            async with websockets.connect(ws_url) as ws:
                # Send synthesis request
                request = {
                    "model_id": self.config.model_id,
                    "transcript": text,
                    "voice": {
                        "mode": "id",
                        "id": actual_voice_id,
                    },
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": self.config.sample_rate,
                    },
                    "context_id": f"stream-{int(time.time() * 1000)}",
                }

                if speed != 1.0:
                    request["voice"]["speed"] = max(0.5, min(2.0, speed))

                if emotion:
                    provider = CartesiaTTSProvider(api_key=self.config.api_key)
                    request["voice"]["emotion"] = provider._map_emotion(emotion)

                await ws.send(json.dumps(request))

                # Receive and yield audio chunks
                first_chunk = True

                async for message in ws:
                    if isinstance(message, bytes):
                        # Raw audio data
                        if first_chunk:
                            self.metrics.first_byte_time = time.time()
                            first_chunk = False
                            if on_first_chunk:
                                on_first_chunk(self.metrics.ttfa_ms)

                        self.metrics.chunk_count += 1
                        self.metrics.total_bytes += len(message)
                        yield message

                    elif isinstance(message, str):
                        # JSON control message
                        data = json.loads(message)
                        if data.get("type") == "done":
                            break
                        elif data.get("type") == "error":
                            raise ProviderException("cartesia", data.get("message", "Unknown error"))

                # Calculate final duration
                self.metrics.duration_ms = (time.time() - self.metrics.start_time) * 1000

                logger.info(
                    "Cartesia streaming complete",
                    ttfa_ms=round(self.metrics.ttfa_ms, 2),
                    total_bytes=self.metrics.total_bytes,
                    chunks=self.metrics.chunk_count,
                    duration_ms=round(self.metrics.duration_ms, 2),
                )

        except Exception as e:
            if "websockets" in str(type(e).__module__):
                raise ProviderException("cartesia", f"WebSocket error: {str(e)}")
            raise

    async def stream_to_buffer(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        emotion: Optional[str] = None,
    ) -> Tuple[bytes, StreamingMetrics]:
        """
        Stream audio and collect into buffer.

        Returns:
            Tuple of (audio_bytes, metrics)
        """
        chunks = []
        async for chunk in self.stream(text, voice_id, speed, emotion):
            chunks.append(chunk)

        # Convert PCM to MP3 for storage
        pcm_data = b"".join(chunks)
        mp3_data = await self._pcm_to_mp3(pcm_data)

        return mp3_data, self.metrics

    async def generate_preview(
        self,
        text: str,
        voice_id: str,
        preview_seconds: int = 30,
        speed: float = 1.0,
    ) -> bytes:
        """
        Generate a preview of the first N seconds of audio.

        Uses sentence-level truncation for natural cutoff.

        Args:
            text: Full text to synthesize
            voice_id: Voice ID
            preview_seconds: Duration of preview in seconds
            speed: Speech speed

        Returns:
            MP3 audio bytes for preview
        """
        # Check cache first
        if self.cache:
            cached = await self.cache.get_preview(text, voice_id, preview_seconds)
            if cached:
                logger.debug("Preview cache hit")
                return cached

        # Estimate characters for preview (approx 15 chars/second at normal speed)
        chars_per_second = 15 / speed
        target_chars = int(preview_seconds * chars_per_second)

        # Truncate at sentence boundary
        preview_text = self._truncate_at_sentence(text, target_chars)

        logger.info(
            "Generating preview",
            full_length=len(text),
            preview_length=len(preview_text),
            target_seconds=preview_seconds,
        )

        # Generate preview audio
        audio_data, metrics = await self.stream_to_buffer(preview_text, voice_id, speed)

        # Cache preview
        if self.cache:
            await self.cache.set_preview(text, voice_id, audio_data, preview_seconds)

        return audio_data

    def _truncate_at_sentence(self, text: str, target_chars: int) -> str:
        """Truncate text at a sentence boundary near target length."""
        if len(text) <= target_chars:
            return text

        # Find sentence boundaries
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)

        result = ""
        for sentence in sentences:
            if len(result) + len(sentence) > target_chars:
                break
            result += sentence + " "

        return result.strip() or text[:target_chars]

    async def _pcm_to_mp3(self, pcm_data: bytes) -> bytes:
        """Convert raw PCM audio to MP3."""
        try:
            from pydub import AudioSegment

            # Create AudioSegment from raw PCM
            audio = AudioSegment(
                data=pcm_data,
                sample_width=2,  # 16-bit
                frame_rate=self.config.sample_rate,
                channels=1,  # Mono
            )

            # Export as MP3
            buffer = io.BytesIO()
            audio.export(buffer, format="mp3", bitrate="192k")
            return buffer.getvalue()

        except ImportError:
            logger.warning("pydub not available, returning raw PCM")
            return pcm_data


# =============================================================================
# Streaming Audio Endpoint Helper
# =============================================================================

class StreamingAudioResponse:
    """
    Helper for streaming audio responses in FastAPI.

    Usage in API endpoint:
        @router.get("/audio/stream/{text}")
        async def stream_audio(text: str):
            streaming = CartesiaStreamingTTS()
            return StreamingAudioResponse(
                streaming.stream(text, "sonic-english-male-1")
            )
    """

    def __init__(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        media_type: str = "audio/mpeg",
    ):
        self.generator = audio_generator
        self.media_type = media_type

    async def __aiter__(self):
        async for chunk in self.generator:
            yield chunk


# =============================================================================
# Factory Functions
# =============================================================================

def create_cartesia_provider(
    api_key: Optional[str] = None,
    streaming: bool = False,
    enable_cache: bool = True,
) -> BaseTTSProvider | CartesiaStreamingTTS:
    """
    Create a Cartesia TTS provider.

    Args:
        api_key: API key (defaults to settings.CARTESIA_API_KEY)
        streaming: If True, return streaming provider
        enable_cache: Enable Redis caching

    Returns:
        CartesiaTTSProvider or CartesiaStreamingTTS
    """
    if streaming:
        return CartesiaStreamingTTS(api_key=api_key, enable_cache=enable_cache)
    return CartesiaTTSProvider(api_key=api_key, enable_cache=enable_cache)


async def quick_synthesize(
    text: str,
    voice: str = "sonic-english-male-1",
    speed: float = 1.0,
) -> bytes:
    """
    Quick synthesis helper for simple use cases.

    Args:
        text: Text to synthesize
        voice: Voice ID
        speed: Speech speed

    Returns:
        MP3 audio bytes
    """
    provider = CartesiaTTSProvider()
    return await provider.synthesize(text, voice, speed)


async def quick_preview(
    text: str,
    voice: str = "sonic-english-male-1",
    seconds: int = 30,
) -> bytes:
    """
    Quick preview generation helper.

    Args:
        text: Full text
        voice: Voice ID
        seconds: Preview duration

    Returns:
        MP3 audio bytes for preview
    """
    streaming = CartesiaStreamingTTS()
    return await streaming.generate_preview(text, voice, seconds)
