"""
AIDocumentIndexer - Teams Bot Core
===================================

Core bot class for Microsoft Teams integration.
"""

import uuid
from typing import Optional, Dict, Any

import structlog

logger = structlog.get_logger(__name__)


class TeamsBot:
    """
    Microsoft Teams Bot implementation.

    Handles bot registration, message processing, and
    integration with AIDocumentIndexer services.
    """

    def __init__(
        self,
        app_id: str,
        app_password: str,
        organization_id: Optional[str] = None,
    ):
        """
        Initialize Teams Bot.

        Args:
            app_id: Microsoft App ID (from Azure Bot registration)
            app_password: Microsoft App Password
            organization_id: AIDocumentIndexer organization context
        """
        self.app_id = app_id
        self.app_password = app_password
        self.organization_id = organization_id
        self._handler = None

        logger.info("Teams bot initialized", app_id=app_id[:8] + "...")

    async def process_activity(
        self,
        activity: Dict[str, Any],
        auth_header: str,
    ) -> Dict[str, Any]:
        """
        Process an incoming activity from Teams.

        Args:
            activity: The activity JSON from Teams
            auth_header: Authorization header for validation

        Returns:
            Response to send back
        """
        try:
            activity_type = activity.get("type", "")
            activity_id = activity.get("id", str(uuid.uuid4()))

            logger.info(
                "Processing Teams activity",
                activity_type=activity_type,
                activity_id=activity_id,
            )

            # Route based on activity type
            if activity_type == "message":
                return await self._handle_message(activity)
            elif activity_type == "conversationUpdate":
                return await self._handle_conversation_update(activity)
            elif activity_type == "invoke":
                return await self._handle_invoke(activity)
            elif activity_type == "messageReaction":
                return await self._handle_reaction(activity)
            else:
                logger.debug(f"Unhandled activity type: {activity_type}")
                return {"status": "ok"}

        except Exception as e:
            logger.error("Error processing Teams activity", error=e)
            return {"status": "error", "message": str(e)}

    async def _handle_message(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message."""
        from backend.integrations.teams_bot.handlers import TeamsActivityHandler

        handler = TeamsActivityHandler(
            organization_id=self.organization_id,
        )

        return await handler.on_message(activity)

    async def _handle_conversation_update(
        self,
        activity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle conversation update (member added/removed)."""
        members_added = activity.get("membersAdded", [])
        members_removed = activity.get("membersRemoved", [])

        # Check if bot was added
        bot_id = activity.get("recipient", {}).get("id", "")

        for member in members_added:
            if member.get("id") == bot_id:
                # Bot was added to conversation
                return await self._send_welcome_message(activity)

        return {"status": "ok"}

    async def _send_welcome_message(
        self,
        activity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send welcome message when bot is added."""
        from backend.integrations.teams_bot.cards import create_welcome_card

        card = create_welcome_card()

        return {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card,
                }
            ],
        }

    async def _handle_invoke(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Handle invoke activities (card actions, etc.)."""
        invoke_name = activity.get("name", "")
        value = activity.get("value", {})

        if invoke_name == "adaptiveCard/action":
            action = value.get("action", {})
            action_type = action.get("type", "")
            action_data = action.get("data", {})

            return await self._handle_card_action(action_type, action_data, activity)

        return {"status": 200, "body": {"status": "ok"}}

    async def _handle_card_action(
        self,
        action_type: str,
        action_data: Dict[str, Any],
        activity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle adaptive card action."""
        from backend.integrations.teams_bot.handlers import TeamsActivityHandler

        handler = TeamsActivityHandler(organization_id=self.organization_id)

        if action_type == "search":
            query = action_data.get("query", "")
            return await handler.handle_search(query, activity)
        elif action_type == "ask":
            question = action_data.get("question", "")
            return await handler.handle_question(question, activity)

        return {"status": 200, "body": {"status": "ok"}}

    async def _handle_reaction(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Handle message reactions."""
        # Log reactions for analytics
        reactions_added = activity.get("reactionsAdded", [])
        reactions_removed = activity.get("reactionsRemoved", [])

        for reaction in reactions_added:
            logger.debug(
                "Reaction added",
                reaction_type=reaction.get("type"),
                message_id=activity.get("replyToId"),
            )

        return {"status": "ok"}

    async def send_proactive_message(
        self,
        conversation_reference: Dict[str, Any],
        message: str,
    ) -> bool:
        """
        Send a proactive message to a conversation.

        Args:
            conversation_reference: Stored conversation reference
            message: Message to send

        Returns:
            True if sent successfully
        """
        try:
            # This would use the Bot Framework SDK to send proactive messages
            # For now, return True as placeholder
            logger.info(
                "Sending proactive message",
                conversation_id=conversation_reference.get("conversation", {}).get("id"),
            )
            return True
        except Exception as e:
            logger.error("Failed to send proactive message", error=e)
            return False


def create_teams_bot(
    app_id: str,
    app_password: str,
    organization_id: Optional[str] = None,
) -> TeamsBot:
    """
    Create a Teams bot instance.

    Args:
        app_id: Microsoft App ID
        app_password: Microsoft App Password
        organization_id: Organization context

    Returns:
        TeamsBot instance
    """
    return TeamsBot(
        app_id=app_id,
        app_password=app_password,
        organization_id=organization_id,
    )
