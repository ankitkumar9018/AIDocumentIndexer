"""
AIDocumentIndexer - Messaging Bot Webhook Routes
=================================================

Webhook endpoints for Telegram and WhatsApp bots.

Endpoints:
- POST /messaging/telegram/webhook - Telegram bot webhook
- POST /messaging/whatsapp/webhook - WhatsApp (Twilio) webhook
- GET  /messaging/telegram/setup   - Set up Telegram webhook
- GET  /messaging/status           - Get bot status
"""

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Form, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import structlog

from backend.services.messaging_bots import (
    MessagingPlatform,
    TelegramBot,
    WhatsAppBot,
    BotConfig,
    IncomingMessage,
    OutgoingMessage,
    MessagingBotHandler,
    get_bot_handler,
    create_bot_handler,
)
from backend.api.middleware.auth import require_admin
from backend.services.permissions import UserContext

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/messaging", tags=["Messaging Bots"])


# =============================================================================
# Response Models
# =============================================================================

class BotStatusResponse(BaseModel):
    """Bot status response."""
    telegram: Dict[str, Any]
    whatsapp: Dict[str, Any]


class WebhookSetupResponse(BaseModel):
    """Webhook setup response."""
    success: bool
    webhook_url: str
    message: str


# =============================================================================
# RAG Query Function
# =============================================================================

async def _rag_query(query: str, collection: Optional[str] = None) -> str:
    """
    Query the RAG system.

    This is the function passed to the bot handler to generate responses.
    """
    try:
        from backend.services.rag import RAGService

        rag = RAGService()
        result = await rag.query(
            question=query,
            collection_filter=collection,
            top_k=5,
        )

        return result.content or "I couldn't find relevant information."

    except Exception as e:
        logger.error("RAG query failed in messaging bot", error=str(e))
        return "I'm having trouble accessing the knowledge base. Please try again later."


# =============================================================================
# Telegram Webhook
# =============================================================================

@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """
    Telegram Bot webhook endpoint.

    Receives updates from Telegram and processes messages.
    """
    handler = get_bot_handler()
    if not handler:
        # Initialize handler on first request
        handler = await create_bot_handler(_rag_query)

    bot = handler.get_bot(MessagingPlatform.TELEGRAM)
    if not bot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram bot not configured",
        )

    # Verify webhook if secret is set
    signature = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    body = await request.body()

    if bot.config.webhook_secret and not bot.verify_webhook(body, signature):
        logger.warning("Invalid Telegram webhook signature")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature",
        )

    # Parse update
    update = await request.json()
    message = bot.parse_update(update)

    if message:
        # Process in background to respond quickly
        import asyncio
        asyncio.create_task(handler.process_and_reply(message))

    # Always return 200 to Telegram
    return {"ok": True}


@router.get("/telegram/setup", response_model=WebhookSetupResponse)
async def setup_telegram_webhook(
    user: UserContext = Depends(require_admin),
):
    """
    Set up Telegram webhook.

    Configures Telegram to send updates to this server.
    Requires admin access.
    """
    handler = get_bot_handler()
    if not handler:
        handler = await create_bot_handler(_rag_query)

    bot = handler.get_bot(MessagingPlatform.TELEGRAM)
    if not bot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Telegram bot not configured. Set TELEGRAM_BOT_TOKEN environment variable.",
        )

    # Get base URL
    base_url = os.getenv("BASE_URL", "https://your-domain.com")
    webhook_url = f"{base_url}/api/v1/messaging/telegram/webhook"

    # Set webhook
    success = await bot.set_webhook(webhook_url)

    if success:
        logger.info("Telegram webhook set up", webhook_url=webhook_url)
        return WebhookSetupResponse(
            success=True,
            webhook_url=webhook_url,
            message="Webhook configured successfully",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set up webhook",
        )


# =============================================================================
# WhatsApp Webhook
# =============================================================================

@router.post("/whatsapp/webhook")
async def whatsapp_webhook(
    request: Request,
    # Twilio sends form-urlencoded data
    From: str = Form(None),
    To: str = Form(None),
    Body: str = Form(None),
    MessageSid: str = Form(None),
    NumMedia: str = Form("0"),
    MediaUrl0: str = Form(None),
    MediaContentType0: str = Form(None),
    ProfileName: str = Form(None),
):
    """
    WhatsApp (Twilio) webhook endpoint.

    Receives incoming WhatsApp messages via Twilio.
    Returns TwiML response.
    """
    handler = get_bot_handler()
    if not handler:
        handler = await create_bot_handler(_rag_query)

    bot = handler.get_bot(MessagingPlatform.WHATSAPP)
    if not bot:
        # Return empty TwiML if not configured
        return PlainTextResponse(
            content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            media_type="application/xml",
        )

    # Parse message
    form_data = {
        "From": From,
        "To": To,
        "Body": Body,
        "MessageSid": MessageSid,
        "NumMedia": NumMedia,
        "MediaUrl0": MediaUrl0,
        "MediaContentType0": MediaContentType0,
        "ProfileName": ProfileName,
    }

    message = bot.parse_webhook(form_data)

    if message:
        # Generate response
        response_text = await handler.handle_message(message)

        if response_text:
            # Return TwiML with response
            twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{_escape_xml(response_text[:1600])}</Message>
</Response>'''
            return PlainTextResponse(content=twiml, media_type="application/xml")

    # Empty response
    return PlainTextResponse(
        content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="application/xml",
    )


def _escape_xml(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# =============================================================================
# Status & Management
# =============================================================================

@router.get("/status", response_model=BotStatusResponse)
async def get_bot_status(
    user: UserContext = Depends(require_admin),
):
    """
    Get status of messaging bots.

    Shows which bots are configured and active.
    Requires admin access.
    """
    handler = get_bot_handler()

    telegram_status = {
        "configured": False,
        "active": False,
        "bot_username": None,
    }

    whatsapp_status = {
        "configured": False,
        "active": False,
        "phone_number": None,
    }

    if handler:
        telegram_bot = handler.get_bot(MessagingPlatform.TELEGRAM)
        if telegram_bot:
            telegram_status["configured"] = True
            telegram_status["active"] = telegram_bot.config.enabled

        whatsapp_bot = handler.get_bot(MessagingPlatform.WHATSAPP)
        if whatsapp_bot:
            whatsapp_status["configured"] = True
            whatsapp_status["active"] = whatsapp_bot.config.enabled
            whatsapp_status["phone_number"] = os.getenv("TWILIO_WHATSAPP_NUMBER", "")

    # Check environment variables
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        telegram_status["configured"] = True

    if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_WHATSAPP_NUMBER"):
        whatsapp_status["configured"] = True

    return BotStatusResponse(
        telegram=telegram_status,
        whatsapp=whatsapp_status,
    )


@router.post("/test/{platform}")
async def test_bot(
    platform: str,
    message: str = "Hello, this is a test message!",
    chat_id: str = None,
    user: UserContext = Depends(require_admin),
):
    """
    Test sending a message via bot.

    Requires admin access and a chat_id to send to.
    """
    handler = get_bot_handler()
    if not handler:
        handler = await create_bot_handler(_rag_query)

    try:
        platform_enum = MessagingPlatform(platform.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid platform: {platform}",
        )

    bot = handler.get_bot(platform_enum)
    if not bot:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{platform} bot not configured",
        )

    if not chat_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chat_id is required for testing",
        )

    outgoing = OutgoingMessage(
        chat_id=chat_id,
        text=message,
    )

    result = await bot.send_message(outgoing)

    return {
        "success": True,
        "result": result,
    }
