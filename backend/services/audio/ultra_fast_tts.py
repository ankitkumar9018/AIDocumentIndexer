"""
AIDocumentIndexer - Ultra-Fast TTS Providers
=============================================

Phase 45: Ultra-low latency TTS providers for real-time voice agents.

Providers implemented:
- Murf Falcon: 55ms model latency, ~130ms TTFA, $0.01/min
- Smallest.ai Lightning: RTF 0.01 - fastest TTS available
- CosyVoice2: 150ms streaming latency, open-source
- Fish Speech 1.5: ELO 1339, DualAR architecture

Research sources:
- https://smallest.ai/blog/fastest-text-to-speech-apis
- https://www.f22labs.com/blogs/13-text-to-speech-tts-solutions-in-2025/
"""

import asyncio
import io
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Any, Union

import structlog

from backend.core.config import settings
from backend.services.base import ProviderException

logger = structlog.get_logger(__name__)


class UltraFastTTSProvider(str, Enum):
    """Ultra-fast TTS providers with sub-100ms latency."""
    MURF = "murf"               # 55ms model latency
    SMALLEST = "smallest"       # RTF 0.01 - fastest
    COSYVOICE = "cosyvoice"     # 150ms streaming, open-source
    FISH_SPEECH = "fish_speech" # ELO 1339, DualAR
    CHATTERBOX = "chatterbox"   # Resemble AI open-source, ultra-realistic


@dataclass
class UltraFastTTSConfig:
    """Configuration for ultra-fast TTS."""
    provider: UltraFastTTSProvider = UltraFastTTSProvider.SMALLEST

    # Murf settings
    murf_api_key: Optional[str] = None
    murf_voice_id: str = "en-US-natalie"

    # Smallest.ai settings
    smallest_api_key: Optional[str] = None
    smallest_model: str = "lightning"  # lightning or turbo
    smallest_voice_id: str = "emily"

    # CosyVoice settings
    cosyvoice_model_path: Optional[str] = None
    cosyvoice_voice: str = "english_female"

    # Fish Speech settings
    fish_api_key: Optional[str] = None
    fish_voice_id: str = "default"

    # Chatterbox settings (Resemble AI open-source)
    chatterbox_model_path: Optional[str] = None
    chatterbox_voice: str = "default"
    chatterbox_exaggeration: float = 0.5  # Emotional expressiveness (0.0-1.0)
    chatterbox_cfg_weight: float = 0.5  # Classifier-free guidance weight

    # Common settings
    sample_rate: int = 24000
    enable_streaming: bool = True
    chunk_size: int = 4096


@dataclass
class TTSChunk:
    """A chunk of streamed audio data."""
    audio_data: bytes
    is_final: bool = False
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseUltraFastTTS(ABC):
    """Abstract base class for ultra-fast TTS providers."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """Synthesize text to audio bytes."""
        pass

    @abstractmethod
    async def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[TTSChunk]:
        """Stream audio chunks as they are generated."""
        pass

    @abstractmethod
    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get available voices."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass

    @abstractmethod
    def get_latency_ms(self) -> int:
        """Get expected latency in milliseconds."""
        pass


class MurfFalconTTS(BaseUltraFastTTS):
    """
    Murf Falcon TTS - 55ms model latency, ~130ms TTFA.

    Features:
    - Ultra-low latency (55ms model, 130ms TTFA)
    - 120+ voices across 20+ languages
    - $0.01/minute pricing
    - Voice cloning support

    API docs: https://murf.ai/api
    """

    VOICES = {
        # English (US)
        "en-US-natalie": {"name": "Natalie", "gender": "female", "style": "conversational"},
        "en-US-clint": {"name": "Clint", "gender": "male", "style": "professional"},
        "en-US-julia": {"name": "Julia", "gender": "female", "style": "friendly"},
        "en-US-ken": {"name": "Ken", "gender": "male", "style": "authoritative"},
        # English (UK)
        "en-GB-harry": {"name": "Harry", "gender": "male", "style": "british"},
        "en-GB-evelyn": {"name": "Evelyn", "gender": "female", "style": "elegant"},
        # More languages available via API
    }

    def __init__(self, config: UltraFastTTSConfig):
        self.config = config
        self.api_key = config.murf_api_key or os.getenv("MURF_API_KEY")
        self.base_url = "https://api.murf.ai/v1"
        self._client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """Synthesize text to audio using Murf Falcon."""
        if not self.api_key:
            raise ProviderException("murf", "MURF_API_KEY not configured")

        voice = voice_id or self.config.murf_voice_id

        try:
            client = await self._get_client()

            response = await client.post(
                "/speech/generate",
                json={
                    "text": text,
                    "voiceId": voice,
                    "style": kwargs.get("style", "conversational"),
                    "rate": kwargs.get("rate", 1.0),
                    "pitch": kwargs.get("pitch", 1.0),
                    "sampleRate": self.config.sample_rate,
                    "format": "mp3",
                    "model": "falcon",  # Use Falcon for lowest latency
                },
            )

            if response.status_code != 200:
                raise ProviderException(
                    "murf",
                    f"API error: {response.status_code} - {response.text}",
                )

            return response.content

        except Exception as e:
            logger.error("Murf synthesis failed", error=str(e))
            raise ProviderException("murf", f"Synthesis failed: {str(e)}")

    async def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[TTSChunk]:
        """Stream audio from Murf (if streaming API available)."""
        import time

        if not self.api_key:
            raise ProviderException("murf", "MURF_API_KEY not configured")

        voice = voice_id or self.config.murf_voice_id
        start_time = time.time()
        first_chunk = True

        try:
            client = await self._get_client()

            async with client.stream(
                "POST",
                "/speech/stream",
                json={
                    "text": text,
                    "voiceId": voice,
                    "model": "falcon",
                    "format": "mp3",
                },
            ) as response:
                if response.status_code != 200:
                    # Fallback to non-streaming
                    audio_data = await self.synthesize(text, voice_id, **kwargs)
                    yield TTSChunk(
                        audio_data=audio_data,
                        is_final=True,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                    return

                async for chunk in response.aiter_bytes(self.config.chunk_size):
                    latency = None
                    if first_chunk:
                        latency = (time.time() - start_time) * 1000
                        first_chunk = False

                    yield TTSChunk(
                        audio_data=chunk,
                        is_final=False,
                        latency_ms=latency,
                    )

                yield TTSChunk(audio_data=b"", is_final=True)

        except Exception as e:
            logger.error("Murf streaming failed", error=str(e))
            # Fallback to non-streaming
            audio_data = await self.synthesize(text, voice_id, **kwargs)
            yield TTSChunk(
                audio_data=audio_data,
                is_final=True,
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get available Murf voices."""
        return [
            {"id": voice_id, **info}
            for voice_id, info in self.VOICES.items()
        ]

    def get_provider_name(self) -> str:
        return "murf"

    def get_latency_ms(self) -> int:
        return 55  # 55ms model latency


class SmallestAITTS(BaseUltraFastTTS):
    """
    Smallest.ai Lightning TTS - RTF 0.01, fastest available.

    Features:
    - Real Time Factor (RTF) of 0.01 - fastest TTS
    - Sub-100ms synthesis for short texts
    - Multiple voice options
    - Streaming support

    API docs: https://smallest.ai/docs
    """

    VOICES = {
        "emily": {"name": "Emily", "gender": "female", "style": "natural"},
        "james": {"name": "James", "gender": "male", "style": "professional"},
        "sarah": {"name": "Sarah", "gender": "female", "style": "friendly"},
        "michael": {"name": "Michael", "gender": "male", "style": "confident"},
        "olivia": {"name": "Olivia", "gender": "female", "style": "warm"},
        "david": {"name": "David", "gender": "male", "style": "casual"},
    }

    def __init__(self, config: UltraFastTTSConfig):
        self.config = config
        self.api_key = config.smallest_api_key or os.getenv("SMALLEST_API_KEY")
        self.base_url = "https://api.smallest.ai/v1"
        self._client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """Synthesize text using Smallest.ai Lightning."""
        if not self.api_key:
            raise ProviderException("smallest", "SMALLEST_API_KEY not configured")

        voice = voice_id or self.config.smallest_voice_id

        try:
            client = await self._get_client()

            response = await client.post(
                "/tts/generate",
                json={
                    "text": text,
                    "voice_id": voice,
                    "model": self.config.smallest_model,  # lightning or turbo
                    "sample_rate": self.config.sample_rate,
                    "speed": kwargs.get("speed", 1.0),
                    "format": "mp3",
                },
            )

            if response.status_code != 200:
                raise ProviderException(
                    "smallest",
                    f"API error: {response.status_code} - {response.text}",
                )

            return response.content

        except Exception as e:
            logger.error("Smallest.ai synthesis failed", error=str(e))
            raise ProviderException("smallest", f"Synthesis failed: {str(e)}")

    async def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[TTSChunk]:
        """Stream audio from Smallest.ai."""
        import time

        if not self.api_key:
            raise ProviderException("smallest", "SMALLEST_API_KEY not configured")

        voice = voice_id or self.config.smallest_voice_id
        start_time = time.time()
        first_chunk = True

        try:
            client = await self._get_client()

            async with client.stream(
                "POST",
                "/tts/stream",
                json={
                    "text": text,
                    "voice_id": voice,
                    "model": "lightning",
                    "format": "mp3",
                },
            ) as response:
                if response.status_code != 200:
                    # Fallback to non-streaming
                    audio_data = await self.synthesize(text, voice_id, **kwargs)
                    yield TTSChunk(
                        audio_data=audio_data,
                        is_final=True,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                    return

                async for chunk in response.aiter_bytes(self.config.chunk_size):
                    latency = None
                    if first_chunk:
                        latency = (time.time() - start_time) * 1000
                        first_chunk = False

                    yield TTSChunk(
                        audio_data=chunk,
                        is_final=False,
                        latency_ms=latency,
                    )

                yield TTSChunk(audio_data=b"", is_final=True)

        except Exception as e:
            logger.error("Smallest.ai streaming failed", error=str(e))
            # Fallback to non-streaming
            audio_data = await self.synthesize(text, voice_id, **kwargs)
            yield TTSChunk(
                audio_data=audio_data,
                is_final=True,
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get available Smallest.ai voices."""
        return [
            {"id": voice_id, **info}
            for voice_id, info in self.VOICES.items()
        ]

    def get_provider_name(self) -> str:
        return "smallest"

    def get_latency_ms(self) -> int:
        return 30  # Sub-100ms with RTF 0.01


class CosyVoiceTTS(BaseUltraFastTTS):
    """
    CosyVoice2 - 150ms streaming latency, open-source.

    Features:
    - Open-source model from Alibaba
    - 150ms streaming latency
    - Voice cloning with few-shot learning
    - Cross-lingual voice synthesis

    GitHub: https://github.com/FunAudioLLM/CosyVoice
    """

    VOICES = {
        "english_female": {"name": "English Female", "gender": "female", "lang": "en"},
        "english_male": {"name": "English Male", "gender": "male", "lang": "en"},
        "chinese_female": {"name": "Chinese Female", "gender": "female", "lang": "zh"},
        "chinese_male": {"name": "Chinese Male", "gender": "male", "lang": "zh"},
    }

    def __init__(self, config: UltraFastTTSConfig):
        self.config = config
        self.model_path = config.cosyvoice_model_path
        self._model = None

    async def _load_model(self):
        """Lazy-load CosyVoice model."""
        if self._model is not None:
            return self._model

        try:
            # CosyVoice requires local model files
            # This is a placeholder - actual implementation depends on model setup
            logger.info("Loading CosyVoice model", path=self.model_path)

            # Attempt to import CosyVoice
            try:
                from cosyvoice import CosyVoice
                self._model = CosyVoice(self.model_path or "CosyVoice-300M")
            except ImportError:
                logger.warning("CosyVoice not installed, using HTTP API fallback")
                self._model = "http_fallback"

            return self._model

        except Exception as e:
            logger.error("Failed to load CosyVoice model", error=str(e))
            raise ProviderException("cosyvoice", f"Model loading failed: {str(e)}")

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """Synthesize text using CosyVoice."""
        model = await self._load_model()
        voice = voice_id or self.config.cosyvoice_voice

        if model == "http_fallback":
            # Use HTTP API if available
            return await self._synthesize_http(text, voice, **kwargs)

        try:
            loop = asyncio.get_running_loop()

            def generate():
                # Generate audio using local model
                audio = model.inference_sft(text, voice)

                # Convert to bytes (assuming numpy array output)
                import numpy as np
                import soundfile as sf

                buffer = io.BytesIO()
                sf.write(buffer, audio, self.config.sample_rate, format="WAV")
                return buffer.getvalue()

            audio_data = await loop.run_in_executor(None, generate)
            return audio_data

        except Exception as e:
            logger.error("CosyVoice synthesis failed", error=str(e))
            raise ProviderException("cosyvoice", f"Synthesis failed: {str(e)}")

    async def _synthesize_http(
        self,
        text: str,
        voice: str,
        **kwargs,
    ) -> bytes:
        """HTTP API fallback for CosyVoice (if hosted service available)."""
        import httpx

        # This would use a hosted CosyVoice service if available
        # For now, raise an exception indicating local model is needed
        raise ProviderException(
            "cosyvoice",
            "CosyVoice requires local model. Install with: pip install cosyvoice",
        )

    async def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[TTSChunk]:
        """Stream audio from CosyVoice."""
        import time

        start_time = time.time()
        model = await self._load_model()
        voice = voice_id or self.config.cosyvoice_voice

        if model == "http_fallback":
            # Non-streaming fallback
            audio_data = await self.synthesize(text, voice_id, **kwargs)
            yield TTSChunk(
                audio_data=audio_data,
                is_final=True,
                latency_ms=(time.time() - start_time) * 1000,
            )
            return

        try:
            loop = asyncio.get_running_loop()
            first_chunk = True

            # CosyVoice2 supports streaming
            def stream_generator():
                for chunk in model.inference_stream(text, voice):
                    yield chunk

            for audio_chunk in await loop.run_in_executor(None, list, stream_generator()):
                latency = None
                if first_chunk:
                    latency = (time.time() - start_time) * 1000
                    first_chunk = False

                # Convert numpy to bytes
                import numpy as np
                chunk_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()

                yield TTSChunk(
                    audio_data=chunk_bytes,
                    is_final=False,
                    latency_ms=latency,
                )

            yield TTSChunk(audio_data=b"", is_final=True)

        except Exception as e:
            logger.error("CosyVoice streaming failed", error=str(e))
            # Fallback to non-streaming
            audio_data = await self.synthesize(text, voice_id, **kwargs)
            yield TTSChunk(
                audio_data=audio_data,
                is_final=True,
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get available CosyVoice voices."""
        return [
            {"id": voice_id, **info}
            for voice_id, info in self.VOICES.items()
        ]

    def get_provider_name(self) -> str:
        return "cosyvoice"

    def get_latency_ms(self) -> int:
        return 150  # 150ms streaming latency


class FishSpeechTTS(BaseUltraFastTTS):
    """
    Fish Speech 1.5 - ELO 1339, DualAR architecture.

    Features:
    - High ELO rating (1339) for naturalness
    - DualAR (Dual Autoregressive) architecture
    - Voice cloning with reference audio
    - Multi-language support

    GitHub: https://github.com/fishaudio/fish-speech
    """

    VOICES = {
        "default": {"name": "Default", "gender": "neutral", "style": "natural"},
        "narrator": {"name": "Narrator", "gender": "male", "style": "storytelling"},
        "assistant": {"name": "Assistant", "gender": "female", "style": "helpful"},
    }

    def __init__(self, config: UltraFastTTSConfig):
        self.config = config
        self.api_key = config.fish_api_key or os.getenv("FISH_SPEECH_API_KEY")
        self.base_url = os.getenv("FISH_SPEECH_API_URL", "http://localhost:8080")
        self._client = None
        self._model = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def _load_local_model(self):
        """Load local Fish Speech model if available."""
        if self._model is not None:
            return self._model

        try:
            from fish_speech import FishSpeech
            self._model = FishSpeech()
            return self._model
        except ImportError:
            return None

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """Synthesize text using Fish Speech."""
        voice = voice_id or self.config.fish_voice_id

        # Try local model first
        local_model = await self._load_local_model()
        if local_model:
            return await self._synthesize_local(text, voice, **kwargs)

        # Fall back to API
        return await self._synthesize_api(text, voice, **kwargs)

    async def _synthesize_local(
        self,
        text: str,
        voice: str,
        **kwargs,
    ) -> bytes:
        """Synthesize using local Fish Speech model."""
        try:
            loop = asyncio.get_running_loop()

            def generate():
                audio = self._model.synthesize(
                    text=text,
                    voice=voice,
                    **kwargs,
                )

                # Convert to bytes
                import numpy as np
                import soundfile as sf

                buffer = io.BytesIO()
                sf.write(buffer, audio, self.config.sample_rate, format="WAV")
                return buffer.getvalue()

            return await loop.run_in_executor(None, generate)

        except Exception as e:
            logger.error("Fish Speech local synthesis failed", error=str(e))
            raise ProviderException("fish_speech", f"Synthesis failed: {str(e)}")

    async def _synthesize_api(
        self,
        text: str,
        voice: str,
        **kwargs,
    ) -> bytes:
        """Synthesize using Fish Speech API."""
        try:
            client = await self._get_client()

            response = await client.post(
                "/v1/tts",
                json={
                    "text": text,
                    "voice_id": voice,
                    "format": "wav",
                    "sample_rate": self.config.sample_rate,
                },
            )

            if response.status_code != 200:
                raise ProviderException(
                    "fish_speech",
                    f"API error: {response.status_code} - {response.text}",
                )

            return response.content

        except Exception as e:
            logger.error("Fish Speech API synthesis failed", error=str(e))
            raise ProviderException("fish_speech", f"Synthesis failed: {str(e)}")

    async def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[TTSChunk]:
        """Stream audio from Fish Speech."""
        import time

        start_time = time.time()
        voice = voice_id or self.config.fish_voice_id

        # Fish Speech supports streaming via API
        try:
            client = await self._get_client()
            first_chunk = True

            async with client.stream(
                "POST",
                "/v1/tts/stream",
                json={
                    "text": text,
                    "voice_id": voice,
                    "format": "wav",
                },
            ) as response:
                if response.status_code != 200:
                    # Fallback to non-streaming
                    audio_data = await self.synthesize(text, voice_id, **kwargs)
                    yield TTSChunk(
                        audio_data=audio_data,
                        is_final=True,
                        latency_ms=(time.time() - start_time) * 1000,
                    )
                    return

                async for chunk in response.aiter_bytes(self.config.chunk_size):
                    latency = None
                    if first_chunk:
                        latency = (time.time() - start_time) * 1000
                        first_chunk = False

                    yield TTSChunk(
                        audio_data=chunk,
                        is_final=False,
                        latency_ms=latency,
                    )

                yield TTSChunk(audio_data=b"", is_final=True)

        except Exception as e:
            logger.error("Fish Speech streaming failed", error=str(e))
            # Fallback to non-streaming
            audio_data = await self.synthesize(text, voice_id, **kwargs)
            yield TTSChunk(
                audio_data=audio_data,
                is_final=True,
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get available Fish Speech voices."""
        return [
            {"id": voice_id, **info}
            for voice_id, info in self.VOICES.items()
        ]

    def get_provider_name(self) -> str:
        return "fish_speech"

    def get_latency_ms(self) -> int:
        return 80  # Estimated based on DualAR architecture


class ChatterboxTTS(BaseUltraFastTTS):
    """
    Chatterbox TTS - Ultra-realistic open-source TTS from Resemble AI.

    Features:
    - State-of-the-art voice quality (outperforms ElevenLabs on many benchmarks)
    - Emotional expressiveness via exaggeration parameter
    - Fast inference (~100ms for short texts)
    - Voice cloning with reference audio
    - Apache 2.0 license (fully open-source)

    GitHub: https://github.com/resemble-ai/chatterbox
    """

    VOICES = {
        "default": {"name": "Default", "gender": "neutral", "style": "natural"},
        "narrator": {"name": "Narrator", "gender": "male", "style": "storytelling"},
        "conversational": {"name": "Conversational", "gender": "female", "style": "friendly"},
        "professional": {"name": "Professional", "gender": "male", "style": "business"},
    }

    def __init__(self, config: UltraFastTTSConfig):
        self.config = config
        self.model_path = config.chatterbox_model_path
        self.exaggeration = config.chatterbox_exaggeration
        self.cfg_weight = config.chatterbox_cfg_weight
        self._model = None

    async def _load_model(self):
        """Lazy-load Chatterbox model."""
        if self._model is not None:
            return self._model

        try:
            logger.info("Loading Chatterbox model", path=self.model_path)

            # Try to import chatterbox
            try:
                from chatterbox.tts import ChatterboxTTS as ChatterboxModel
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = ChatterboxModel.from_pretrained(device=device)
                logger.info("Chatterbox model loaded", device=device)

            except ImportError:
                logger.warning("Chatterbox not installed, using HTTP API fallback")
                self._model = "http_fallback"

            return self._model

        except Exception as e:
            logger.error("Failed to load Chatterbox model", error=str(e))
            raise ProviderException("chatterbox", f"Model loading failed: {str(e)}")

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> bytes:
        """Synthesize text using Chatterbox."""
        model = await self._load_model()
        voice = voice_id or self.config.chatterbox_voice

        if model == "http_fallback":
            return await self._synthesize_http(text, voice, **kwargs)

        try:
            loop = asyncio.get_running_loop()

            def generate():
                import torch
                import torchaudio

                # Get exaggeration from kwargs or config
                exaggeration = kwargs.get("exaggeration", self.exaggeration)
                cfg_weight = kwargs.get("cfg_weight", self.cfg_weight)

                # Generate audio
                # Chatterbox uses reference audio for voice cloning
                # For default voices, we use pre-cached reference audios
                wav = model.generate(
                    text=text,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )

                # Convert to bytes (wav format)
                buffer = io.BytesIO()
                torchaudio.save(buffer, wav.unsqueeze(0), model.sr, format="wav")
                return buffer.getvalue()

            audio_data = await loop.run_in_executor(None, generate)
            return audio_data

        except Exception as e:
            logger.error("Chatterbox synthesis failed", error=str(e))
            raise ProviderException("chatterbox", f"Synthesis failed: {str(e)}")

    async def _synthesize_http(
        self,
        text: str,
        voice: str,
        **kwargs,
    ) -> bytes:
        """HTTP API fallback for Chatterbox (if hosted service available)."""
        # Check if there's a hosted Chatterbox service
        api_url = os.getenv("CHATTERBOX_API_URL")
        if api_url:
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{api_url}/synthesize",
                        json={
                            "text": text,
                            "voice": voice,
                            "exaggeration": kwargs.get("exaggeration", self.exaggeration),
                            "cfg_weight": kwargs.get("cfg_weight", self.cfg_weight),
                        },
                        timeout=30.0,
                    )

                    if response.status_code == 200:
                        return response.content

                    raise ProviderException(
                        "chatterbox",
                        f"API error: {response.status_code}",
                    )
            except httpx.RequestError as e:
                raise ProviderException("chatterbox", f"Request failed: {str(e)}")

        raise ProviderException(
            "chatterbox",
            "Chatterbox requires local model. Install with: pip install chatterbox-tts",
        )

    async def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[TTSChunk]:
        """Stream audio from Chatterbox."""
        import time

        start_time = time.time()
        model = await self._load_model()
        voice = voice_id or self.config.chatterbox_voice

        if model == "http_fallback":
            # Non-streaming fallback
            audio_data = await self.synthesize(text, voice_id, **kwargs)
            yield TTSChunk(
                audio_data=audio_data,
                is_final=True,
                latency_ms=(time.time() - start_time) * 1000,
            )
            return

        try:
            # Chatterbox supports streaming generation
            loop = asyncio.get_running_loop()
            first_chunk = True

            def stream_generator():
                import torch

                exaggeration = kwargs.get("exaggeration", self.exaggeration)
                cfg_weight = kwargs.get("cfg_weight", self.cfg_weight)

                # Split text into sentences for streaming
                import re
                sentences = re.split(r'(?<=[.!?])\s+', text)

                for sentence in sentences:
                    if sentence.strip():
                        wav = model.generate(
                            text=sentence,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                        )
                        yield wav

            for audio_chunk in await loop.run_in_executor(None, list, stream_generator()):
                latency = None
                if first_chunk:
                    latency = (time.time() - start_time) * 1000
                    first_chunk = False

                # Convert tensor to bytes
                import numpy as np
                chunk_bytes = (audio_chunk.numpy() * 32767).astype(np.int16).tobytes()

                yield TTSChunk(
                    audio_data=chunk_bytes,
                    is_final=False,
                    latency_ms=latency,
                )

            yield TTSChunk(audio_data=b"", is_final=True)

        except Exception as e:
            logger.error("Chatterbox streaming failed", error=str(e))
            # Fallback to non-streaming
            audio_data = await self.synthesize(text, voice_id, **kwargs)
            yield TTSChunk(
                audio_data=audio_data,
                is_final=True,
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def get_voices(self) -> List[Dict[str, Any]]:
        """Get available Chatterbox voices."""
        return [
            {"id": voice_id, **info}
            for voice_id, info in self.VOICES.items()
        ]

    def get_provider_name(self) -> str:
        return "chatterbox"

    def get_latency_ms(self) -> int:
        return 100  # ~100ms for short texts


class UltraFastTTSService:
    """
    Unified service for ultra-fast TTS providers.

    Provides automatic fallback chain:
    Smallest.ai -> Murf Falcon -> Fish Speech -> CosyVoice

    Usage:
        service = UltraFastTTSService()
        audio = await service.synthesize("Hello world")

        # Or with streaming
        async for chunk in service.stream("Hello world"):
            play_audio(chunk.audio_data)
    """

    def __init__(self, config: Optional[UltraFastTTSConfig] = None):
        self.config = config or UltraFastTTSConfig()
        self._providers: Dict[UltraFastTTSProvider, BaseUltraFastTTS] = {}

        # Fallback chain (fastest to slowest)
        self._fallback_chain = [
            UltraFastTTSProvider.SMALLEST,
            UltraFastTTSProvider.MURF,
            UltraFastTTSProvider.FISH_SPEECH,
            UltraFastTTSProvider.CHATTERBOX,
            UltraFastTTSProvider.COSYVOICE,
        ]

    def _get_provider(self, provider: UltraFastTTSProvider) -> BaseUltraFastTTS:
        """Get or create a provider instance."""
        if provider not in self._providers:
            if provider == UltraFastTTSProvider.MURF:
                self._providers[provider] = MurfFalconTTS(self.config)
            elif provider == UltraFastTTSProvider.SMALLEST:
                self._providers[provider] = SmallestAITTS(self.config)
            elif provider == UltraFastTTSProvider.COSYVOICE:
                self._providers[provider] = CosyVoiceTTS(self.config)
            elif provider == UltraFastTTSProvider.FISH_SPEECH:
                self._providers[provider] = FishSpeechTTS(self.config)
            elif provider == UltraFastTTSProvider.CHATTERBOX:
                self._providers[provider] = ChatterboxTTS(self.config)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        return self._providers[provider]

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        provider: Optional[UltraFastTTSProvider] = None,
        **kwargs,
    ) -> bytes:
        """
        Synthesize text to audio with automatic fallback.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier (provider-specific)
            provider: Specific provider to use (None for auto-fallback)
            **kwargs: Provider-specific options

        Returns:
            Audio data as bytes
        """
        if provider:
            # Use specific provider
            tts = self._get_provider(provider)
            return await tts.synthesize(text, voice_id, **kwargs)

        # Try fallback chain
        last_error = None
        for fallback_provider in self._fallback_chain:
            try:
                tts = self._get_provider(fallback_provider)
                return await tts.synthesize(text, voice_id, **kwargs)
            except Exception as e:
                logger.warning(
                    f"Provider {fallback_provider.value} failed, trying next",
                    error=str(e),
                )
                last_error = e
                continue

        raise ProviderException(
            "ultra_fast_tts",
            f"All providers failed. Last error: {last_error}",
        )

    async def stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        provider: Optional[UltraFastTTSProvider] = None,
        **kwargs,
    ) -> AsyncIterator[TTSChunk]:
        """
        Stream audio with automatic fallback.

        Args:
            text: Text to synthesize
            voice_id: Voice identifier
            provider: Specific provider (None for auto-select)
            **kwargs: Provider-specific options

        Yields:
            TTSChunk objects with audio data
        """
        selected_provider = provider or self.config.provider

        try:
            tts = self._get_provider(selected_provider)
            async for chunk in tts.stream(text, voice_id, **kwargs):
                yield chunk
        except Exception as e:
            logger.warning(
                f"Streaming from {selected_provider.value} failed",
                error=str(e),
            )
            # Fallback to non-streaming synthesis
            audio_data = await self.synthesize(text, voice_id, **kwargs)
            yield TTSChunk(audio_data=audio_data, is_final=True)

    async def get_all_voices(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get voices from all available providers."""
        result = {}

        for provider in UltraFastTTSProvider:
            try:
                tts = self._get_provider(provider)
                voices = await tts.get_voices()
                result[provider.value] = voices
            except Exception as e:
                logger.warning(f"Failed to get voices from {provider.value}", error=str(e))
                result[provider.value] = []

        return result

    def get_latencies(self) -> Dict[str, int]:
        """Get expected latencies for all providers."""
        return {
            UltraFastTTSProvider.SMALLEST.value: 30,
            UltraFastTTSProvider.MURF.value: 55,
            UltraFastTTSProvider.FISH_SPEECH.value: 80,
            UltraFastTTSProvider.CHATTERBOX.value: 100,
            UltraFastTTSProvider.COSYVOICE.value: 150,
        }


# Factory function
_service_instance: Optional[UltraFastTTSService] = None


def get_tts_config_from_settings() -> UltraFastTTSConfig:
    """Create UltraFastTTSConfig from environment settings."""
    from backend.core.config import settings

    return UltraFastTTSConfig(
        provider=UltraFastTTSProvider(
            getattr(settings, 'ULTRA_FAST_TTS_PROVIDER', 'smallest')
        ),
        # Chatterbox settings from environment
        chatterbox_exaggeration=getattr(settings, 'CHATTERBOX_EXAGGERATION', 0.5),
        chatterbox_cfg_weight=getattr(settings, 'CHATTERBOX_CFG_WEIGHT', 0.5),
        # CosyVoice settings from environment
        cosyvoice_model_path=getattr(settings, 'COSYVOICE_MODEL_PATH', None) or None,
        # Fish Speech settings from environment
        fish_api_key=getattr(settings, 'FISH_SPEECH_API_KEY', None) or None,
        # Smallest API key from environment
        smallest_api_key=getattr(settings, 'SMALLEST_API_KEY', None) or None,
    )


async def get_ultra_fast_tts(
    config: Optional[UltraFastTTSConfig] = None,
) -> UltraFastTTSService:
    """Get or create ultra-fast TTS service instance."""
    global _service_instance

    if _service_instance is None or config is not None:
        # Use provided config or create from settings
        actual_config = config or get_tts_config_from_settings()
        _service_instance = UltraFastTTSService(actual_config)

    return _service_instance


__all__ = [
    "UltraFastTTSProvider",
    "UltraFastTTSConfig",
    "TTSChunk",
    "UltraFastTTSService",
    "MurfFalconTTS",
    "SmallestAITTS",
    "CosyVoiceTTS",
    "FishSpeechTTS",
    "ChatterboxTTS",
    "get_ultra_fast_tts",
]
