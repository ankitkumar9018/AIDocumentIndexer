# Voice Agents Tutorial

Create TTS-enabled voice assistants that can speak responses using multiple providers.

## Overview

Voice Agents combine the power of RAG with text-to-speech (TTS) to create AI assistants that can speak their responses. They're perfect for:

- Audio content generation
- Accessibility features
- Voice-based interfaces
- Automated narration

## Supported TTS Providers

| Provider | Latency | Quality | Cost |
|----------|---------|---------|------|
| **OpenAI TTS** | ~300ms | Excellent | $0.015/1K chars |
| **ElevenLabs** | ~200ms | Premium | $0.30/1K chars |
| **Cartesia Sonic** | <55ms | Excellent | $0.01/1K chars |
| **Edge TTS** | ~100ms | Good | Free |

## Creating a Voice Agent

### Method 1: Agent Builder UI

1. Navigate to **Agents** → **Create Agent**
2. Select **Voice** as the agent mode
3. Configure settings:
   - **Name**: Give your agent a descriptive name
   - **System Prompt**: Define personality and behavior
   - **TTS Provider**: Choose from OpenAI, ElevenLabs, Cartesia, or Edge
   - **Voice**: Select a voice for your provider
   - **Speed**: Adjust speaking rate (0.5x - 2.0x)

4. Click **Save Agent**

### Method 2: Workflow Designer

1. Navigate to **Workflows** → **Create Workflow**
2. Drag **Voice Agent** from the AI tab
3. Configure in the right panel:
   - Agent ID (optional)
   - Prompt/Task
   - TTS Settings (provider, voice, speed)
   - RAG settings

## Voice Selection by Provider

### OpenAI Voices
- `alloy` - Neutral, balanced
- `echo` - Male voice
- `fable` - British accent
- `onyx` - Deep male voice
- `nova` - Female voice
- `shimmer` - Soft female voice

### ElevenLabs Voices
- `rachel` - Female, warm
- `drew` - Male, professional
- `clyde` - Male, deep
- `paul` - Male, authoritative
- `domi` - Female, energetic
- `dave` - British male

### Cartesia Sonic Voices
- `sonic-default` - Balanced
- `sonic-male` - Male voice
- `sonic-female` - Female voice

### Edge TTS Voices (Free)
- `en-US-AriaNeural` - US Female
- `en-US-GuyNeural` - US Male
- `en-GB-SoniaNeural` - UK Female
- `en-AU-NatashaNeural` - Australian Female

## API Usage

### Create Voice Agent

```bash
POST /api/v1/agent/agents
Content-Type: application/json

{
  "name": "Audio Assistant",
  "description": "Speaks document summaries",
  "agent_mode": "voice",
  "system_prompt": "You are a professional narrator...",
  "tts_config": {
    "provider": "openai",
    "voice_id": "nova",
    "speed": 1.0
  }
}
```

### Execute Voice Agent

```bash
POST /api/v1/agent/agents/{agent_id}/execute
Content-Type: application/json

{
  "query": "Summarize the Q4 financial report",
  "return_audio": true
}
```

Response includes:
```json
{
  "text": "The Q4 financial report shows...",
  "audio_url": "/api/v1/audio/stream/abc123",
  "audio_format": "mp3",
  "duration_seconds": 45.2
}
```

## Using in Workflows

Voice Agents in workflows can:

1. **Generate audio output** from any text input
2. **Use RAG** to answer questions with voice
3. **Chain with other nodes** for complex pipelines

Example workflow:
```
START → Document Retrieval → Voice Agent → END
         ↓                      ↓
    Get relevant docs     Speak summary
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tts_provider` | string | "openai" | TTS provider to use |
| `voice_id` | string | "alloy" | Voice ID for the provider |
| `speed` | number | 1.0 | Speaking speed (0.5-2.0) |
| `use_rag` | boolean | true | Enable document retrieval |
| `doc_filter` | string | "" | Filter documents by folder/tag |
| `audio_format` | string | "mp3" | Output format (mp3, wav, ogg) |
| `include_transcript` | boolean | true | Include text in response |

## Best Practices

1. **Choose the right provider**:
   - Use Cartesia for lowest latency (<55ms)
   - Use ElevenLabs for highest quality
   - Use Edge TTS for free/unlimited usage

2. **Optimize prompts for speech**:
   - Avoid complex formatting (tables, code blocks)
   - Use natural, conversational language
   - Keep responses concise for audio

3. **Handle long content**:
   - Break into chunks for streaming
   - Use context compression for long documents
   - Set appropriate timeouts

## Troubleshooting

### No audio output
- Check API key for your TTS provider
- Verify the voice ID is valid for the provider
- Check network connectivity

### Poor audio quality
- Try a different voice
- Adjust the speed setting
- Switch to a higher-quality provider

### Slow response times
- Switch to Cartesia Sonic for faster latency
- Enable streaming if available
- Reduce context length