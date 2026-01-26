"""
AIDocumentIndexer - Audio Overview Services
============================================

Generate engaging audio discussions from documents (NotebookLM-inspired feature).

This module provides:
- Script generation for different podcast formats
- Text-to-speech integration (OpenAI TTS, ElevenLabs, Cartesia, Coqui)
- Ultra-low latency streaming TTS with Cartesia Sonic 2.0
- Audio overview orchestration
"""

from backend.services.audio.script_generator import ScriptGenerator
from backend.services.audio.tts_service import TTSService, TTSProvider
from backend.services.audio.audio_overview import AudioOverviewService
from backend.services.audio.cartesia_tts import (
    CartesiaTTSProvider,
    CartesiaStreamingTTS,
    create_cartesia_provider,
    quick_synthesize,
    quick_preview,
)

# Phase 45: Ultra-Fast TTS
from backend.services.audio.ultra_fast_tts import (
    UltraFastTTSService,
    UltraFastTTSProvider,
    UltraFastTTSConfig,
    TTSChunk,
    MurfFalconTTS,
    SmallestAITTS,
    CosyVoiceTTS,
    FishSpeechTTS,
    get_ultra_fast_tts,
)

__all__ = [
    "ScriptGenerator",
    "TTSService",
    "TTSProvider",
    "AudioOverviewService",
    # Cartesia streaming TTS
    "CartesiaTTSProvider",
    "CartesiaStreamingTTS",
    "create_cartesia_provider",
    "quick_synthesize",
    "quick_preview",
    # Phase 45: Ultra-Fast TTS
    "UltraFastTTSService",
    "UltraFastTTSProvider",
    "UltraFastTTSConfig",
    "TTSChunk",
    "MurfFalconTTS",
    "SmallestAITTS",
    "CosyVoiceTTS",
    "FishSpeechTTS",
    "get_ultra_fast_tts",
]
