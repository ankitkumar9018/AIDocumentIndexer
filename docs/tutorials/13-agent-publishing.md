# Agent Publishing Tutorial

Embed AI agents in external websites with secure tokens.

## Overview

Agent Publishing allows you to deploy your AI agents to external websites, apps, or services. Published agents get a unique embed token that enables secure access without exposing your API keys.

## Publishing an Agent

### Via Agent Builder UI

1. Navigate to **Agents** → Select your agent
2. Click the **Publish** button
3. Configure publish settings:
   - **Allowed Domains**: Restrict which domains can embed
   - **Rate Limit**: Max requests per minute
   - **Branding**: Custom appearance settings
4. Click **Publish Agent**
5. Copy the embed token

### Via API

```bash
POST /api/v1/agent/agents/{agent_id}/publish
Content-Type: application/json

{
  "allowed_domains": ["example.com", "*.mycompany.com"],
  "rate_limit": 60,
  "branding": {
    "name": "Support Assistant",
    "logo_url": "https://example.com/logo.png",
    "primary_color": "#007bff"
  }
}
```

Response:
```json
{
  "embed_token": "emb_a1b2c3d4e5f6...",
  "embed_url": "https://yourapp.com/embed/emb_a1b2c3d4e5f6",
  "is_published": true
}
```

## Embedding Options

### Option 1: iframe Embed

Simple embedding with an iframe:

```html
<iframe
  src="https://yourapp.com/embed/emb_a1b2c3d4e5f6"
  width="400"
  height="600"
  frameborder="0"
></iframe>
```

### Option 2: JavaScript Widget

More control with JavaScript:

```html
<div id="ai-agent-container"></div>
<script src="https://yourapp.com/embed/widget.js"></script>
<script>
  AIAgent.init({
    container: '#ai-agent-container',
    token: 'emb_a1b2c3d4e5f6',
    theme: 'light',
    position: 'bottom-right'
  });
</script>
```

### Option 3: Direct API Integration

Use the embed endpoints directly:

```javascript
// Get agent configuration
const config = await fetch(
  'https://yourapp.com/api/v1/embed/emb_a1b2c3d4e5f6/config'
).then(r => r.json());

// Chat with the agent
const response = await fetch(
  'https://yourapp.com/api/v1/embed/emb_a1b2c3d4e5f6/chat',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: 'Hello, I need help',
      session_id: 'user-session-123'
    })
  }
).then(r => r.json());

// Voice interaction (for voice agents)
const audio = await fetch(
  'https://yourapp.com/api/v1/embed/emb_a1b2c3d4e5f6/voice',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message: 'Read me the summary',
      session_id: 'user-session-123'
    })
  }
).then(r => r.blob());
```

## Embed API Endpoints

### GET /embed/{token}/config

Returns agent configuration for the embed:

```json
{
  "name": "Support Assistant",
  "description": "AI-powered support agent",
  "agent_mode": "chat",
  "branding": {
    "name": "Support Assistant",
    "logo_url": "...",
    "primary_color": "#007bff"
  },
  "capabilities": ["chat", "rag"]
}
```

### POST /embed/{token}/chat

Send a chat message:

**Request:**
```json
{
  "message": "How do I reset my password?",
  "session_id": "user-123",
  "context": {
    "page_url": "https://example.com/help",
    "user_tier": "premium"
  }
}
```

**Response:**
```json
{
  "response": "To reset your password, follow these steps...",
  "sources": [
    {"title": "Password Reset Guide", "url": "..."}
  ],
  "session_id": "user-123"
}
```

### POST /embed/{token}/voice

Get voice response (for voice agents):

**Request:**
```json
{
  "message": "Summarize my account status",
  "session_id": "user-123"
}
```

**Response:**
```json
{
  "text": "Your account is in good standing...",
  "audio_url": "/api/v1/audio/stream/xyz789",
  "duration_seconds": 12.5
}
```

## Security Configuration

### Domain Restrictions

Limit which domains can embed your agent:

```json
{
  "allowed_domains": [
    "example.com",
    "*.example.com",
    "app.mycompany.com"
  ]
}
```

### Rate Limiting

Prevent abuse with rate limits:

```json
{
  "rate_limit": 60,  // requests per minute
  "rate_limit_by": "ip"  // or "session"
}
```

### CORS Configuration

The embed endpoints automatically handle CORS for allowed domains.

## Monitoring Published Agents

### Usage Analytics

Track embed usage in the Admin dashboard:
- Total requests
- Unique sessions
- Popular queries
- Error rates

### Audit Logs

All embed interactions are logged:
- Timestamp
- Session ID
- Query
- Response status
- Domain origin

## Managing Published Agents

### Update Settings

```bash
PUT /api/v1/agent/agents/{agent_id}/publish
Content-Type: application/json

{
  "allowed_domains": ["newdomain.com"],
  "rate_limit": 120
}
```

### Unpublish Agent

```bash
DELETE /api/v1/agent/agents/{agent_id}/publish
```

This revokes the embed token immediately.

### Rotate Token

Generate a new embed token (invalidates old one):

```bash
POST /api/v1/agent/agents/{agent_id}/publish/rotate
```

## Best Practices

1. **Restrict domains** - Only allow known domains
2. **Set rate limits** - Prevent abuse and control costs
3. **Monitor usage** - Watch for unusual patterns
4. **Use sessions** - Track conversations properly
5. **Test thoroughly** - Verify in target environment

## Troubleshooting

### "Origin not allowed" error
- Add the domain to allowed_domains
- Check for typos in domain configuration
- Verify wildcard patterns

### Rate limit exceeded
- Increase rate_limit setting
- Implement client-side throttling
- Contact admin for higher limits

### Agent not responding
- Check agent is still published
- Verify embed token is correct
- Review agent error logs
