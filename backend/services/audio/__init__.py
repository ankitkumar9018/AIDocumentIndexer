"""
AIDocumentIndexer - Audio Overview Services
============================================

Generate engaging audio discussions from documents (NotebookLM-inspired feature).

This module provides:
- Script generation for different podcast formats
- Text-to-speech integration (OpenAI TTS, ElevenLabs, Coqui)
- Audio overview orchestration
"""

from backend.services.audio.script_generator import ScriptGenerator
from backend.services.audio.tts_service import TTSService
from backend.services.audio.audio_overview import AudioOverviewService

__all__ = [
    "ScriptGenerator",
    "TTSService",
    "AudioOverviewService",
]
