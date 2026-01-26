# Custom Agents

Build AI chatbots and assistants tailored to your use case.

## What are Custom Agents?

Custom agents are AI assistants that:
- Have specialized knowledge from your documents
- Follow custom instructions and personality
- Can be voice-enabled
- Support external publishing/embedding

## Creating an Agent

### From the Dashboard

1. Navigate to Agents page
2. Click "Create Agent"
3. Configure agent settings:
   - **Name**: Display name for the agent
   - **Description**: What the agent does
   - **Knowledge Base**: Select document collections
   - **System Prompt**: Define personality and behavior
   - **Voice**: Enable TTS and select voice

### Agent Types

- **Chatbot**: Text-only conversation
- **Voice Assistant**: TTS-enabled responses
- **RAG Agent**: Document-focused Q&A
- **Support Agent**: Customer service focused

## System Prompt Best Practices

```
You are a helpful customer support agent for [Company].
You have access to our product documentation and FAQ.

Guidelines:
- Be friendly and professional
- Always cite sources when possible
- If unsure, say so and offer to escalate
- Never make up information
```

## Knowledge Base Selection

Choose which documents the agent can access:
- Select specific collections
- Filter by access tier
- Include/exclude specific documents

## Testing Your Agent

1. Use the built-in chat preview
2. Test edge cases and common questions
3. Refine the system prompt based on results

## Publishing Agents

Share agents externally:
- Generate embed code for websites
- Create shareable links
- Set access controls

See [Agent Publishing](13-agent-publishing.md) for details.

## Next Steps

- [Voice Agents](08-voice-agents.md) - Add voice capabilities
- [Agent Publishing](13-agent-publishing.md) - Embed agents externally
