# Audio Overviews

Generate audio summaries of your documents with AI-powered text-to-speech.

## Overview

Audio overviews let you listen to document summaries, perfect for:
- Commuting or exercising
- Accessibility needs
- Quick document review

## Generating Audio Overviews

### From Document View

1. Open a document
2. Click "Generate Audio Overview"
3. Select voice and style options
4. Wait for generation (typically 30-60 seconds)
5. Click play or download

### From Chat

Ask the AI to summarize and generate audio:
> "Create an audio summary of the Q3 report"

## Voice Options

### Cartesia Voices (Default)

High-quality, natural-sounding voices with multiple options:
- Professional (business documents)
- Conversational (general content)
- Narrative (stories, long-form)

### Edge TTS

Free Microsoft Edge voices, good quality for basic needs.

## Configuration

Set your preferred TTS provider in settings:

```env
TTS_PROVIDER=cartesia
CARTESIA_API_KEY=your_key
```

## Audio Formats

- MP3 (default)
- WAV (higher quality)
- OGG (smaller files)

## Next Steps

- [Knowledge Graph](05-knowledge-graph.md) - Visualize document relationships
- [Voice Agents](08-voice-agents.md) - Create voice-enabled chatbots
