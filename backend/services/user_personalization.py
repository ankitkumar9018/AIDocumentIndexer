"""
AIDocumentIndexer - User Personalization Service
=================================================

Tracks and learns user preferences over time to personalize:
1. Response style (length, format, vocabulary)
2. Topic preferences and interests
3. Query patterns for optimization
4. Feedback-driven improvement

Adapts the RAG experience based on observed user behavior.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import Counter

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ResponsePreferences:
    """User preferences for response format and style."""
    preferred_length: str = "medium"  # "short", "medium", "detailed"
    preferred_format: str = "prose"   # "prose", "bullets", "structured"
    expertise_level: str = "general"  # "beginner", "general", "expert"
    wants_citations: bool = True
    wants_follow_ups: bool = True


@dataclass
class FeedbackEntry:
    """Record of user feedback on a response."""
    query: str
    response_id: str
    rating: int  # 1-5
    feedback_text: Optional[str]
    response_length: int
    response_format: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response_id": self.response_id,
            "rating": self.rating,
            "feedback_text": self.feedback_text,
            "response_length": self.response_length,
            "response_format": self.response_format,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackEntry":
        return cls(
            query=data["query"],
            response_id=data["response_id"],
            rating=data["rating"],
            feedback_text=data.get("feedback_text"),
            response_length=data.get("response_length", 0),
            response_format=data.get("response_format", "prose"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else datetime.utcnow(),
        )


@dataclass
class UserProfile:
    """Complete user profile with preferences and history."""
    user_id: str
    preferences: ResponsePreferences = field(default_factory=ResponsePreferences)
    topics_of_interest: List[str] = field(default_factory=list)
    query_patterns: Dict[str, int] = field(default_factory=dict)  # pattern -> count
    feedback_history: List[FeedbackEntry] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    total_queries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": {
                "preferred_length": self.preferences.preferred_length,
                "preferred_format": self.preferences.preferred_format,
                "expertise_level": self.preferences.expertise_level,
                "wants_citations": self.preferences.wants_citations,
                "wants_follow_ups": self.preferences.wants_follow_ups,
            },
            "topics_of_interest": self.topics_of_interest,
            "query_patterns": self.query_patterns,
            "feedback_history": [f.to_dict() for f in self.feedback_history[-50:]],  # Keep last 50
            "last_active": self.last_active.isoformat(),
            "created_at": self.created_at.isoformat(),
            "total_queries": self.total_queries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        prefs = data.get("preferences", {})
        return cls(
            user_id=data["user_id"],
            preferences=ResponsePreferences(
                preferred_length=prefs.get("preferred_length", "medium"),
                preferred_format=prefs.get("preferred_format", "prose"),
                expertise_level=prefs.get("expertise_level", "general"),
                wants_citations=prefs.get("wants_citations", True),
                wants_follow_ups=prefs.get("wants_follow_ups", True),
            ),
            topics_of_interest=data.get("topics_of_interest", []),
            query_patterns=data.get("query_patterns", {}),
            feedback_history=[FeedbackEntry.from_dict(f) for f in data.get("feedback_history", [])],
            last_active=datetime.fromisoformat(data["last_active"]) if isinstance(data.get("last_active"), str) else datetime.utcnow(),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.utcnow(),
            total_queries=data.get("total_queries", 0),
        )


class UserPersonalizationService:
    """
    Learn and apply user preferences for personalized RAG responses.

    Tracks:
    - Response preferences (length, format, vocabulary)
    - Topic interests
    - Query patterns
    - Feedback for continuous improvement
    """

    def __init__(
        self,
        cache=None,
        min_feedback_for_learning: int = 5,
        profile_ttl_days: int = 90,
    ):
        """
        Initialize personalization service.

        Args:
            cache: RedisCache or similar for persistence
            min_feedback_for_learning: Minimum feedback entries before adapting
            profile_ttl_days: Days to keep profile data
        """
        self.cache = cache
        self.min_feedback_for_learning = min_feedback_for_learning
        self.profile_ttl = profile_ttl_days * 86400

        # In-memory cache of profiles
        self._profiles: Dict[str, UserProfile] = {}

    async def get_profile(self, user_id: str) -> UserProfile:
        """
        Get or create user profile.

        Args:
            user_id: User identifier

        Returns:
            UserProfile
        """
        # Check memory cache
        if user_id in self._profiles:
            return self._profiles[user_id]

        # Check persistent cache
        if self.cache:
            try:
                data = await self.cache.get(f"user_profile:{user_id}")
                if data:
                    parsed = json.loads(data) if isinstance(data, str) else data
                    profile = UserProfile.from_dict(parsed)
                    self._profiles[user_id] = profile
                    return profile
            except Exception as e:
                logger.warning("Failed to load profile", error=str(e))

        # Create new profile
        profile = UserProfile(user_id=user_id)
        self._profiles[user_id] = profile
        return profile

    async def save_profile(self, profile: UserProfile) -> None:
        """Save profile to persistent storage."""
        self._profiles[profile.user_id] = profile

        if self.cache:
            try:
                await self.cache.set(
                    f"user_profile:{profile.user_id}",
                    json.dumps(profile.to_dict()),
                    ttl=self.profile_ttl,
                )
            except Exception as e:
                logger.warning("Failed to save profile", error=str(e))

    async def record_query(
        self,
        user_id: str,
        query: str,
        topics: Optional[List[str]] = None,
    ) -> None:
        """
        Record a user query for pattern analysis.

        Args:
            user_id: User identifier
            query: The query text
            topics: Optional detected topics
        """
        profile = await self.get_profile(user_id)
        profile.total_queries += 1
        profile.last_active = datetime.utcnow()

        # Track query pattern (query type)
        pattern = self._classify_query_pattern(query)
        profile.query_patterns[pattern] = profile.query_patterns.get(pattern, 0) + 1

        # Update topics of interest
        if topics:
            for topic in topics:
                if topic not in profile.topics_of_interest:
                    profile.topics_of_interest.append(topic)
            # Keep top 20 topics
            profile.topics_of_interest = profile.topics_of_interest[-20:]

        await self.save_profile(profile)

    async def record_feedback(
        self,
        user_id: str,
        query: str,
        response_id: str,
        rating: int,
        feedback_text: Optional[str] = None,
        response_length: int = 0,
        response_format: str = "prose",
    ) -> None:
        """
        Record user feedback and update preferences.

        Args:
            user_id: User identifier
            query: The original query
            response_id: ID of the response
            rating: 1-5 rating
            feedback_text: Optional text feedback
            response_length: Length of response in characters
            response_format: Format of response
        """
        profile = await self.get_profile(user_id)

        feedback = FeedbackEntry(
            query=query,
            response_id=response_id,
            rating=rating,
            feedback_text=feedback_text,
            response_length=response_length,
            response_format=response_format,
        )

        profile.feedback_history.append(feedback)

        # Keep last 100 feedback entries
        if len(profile.feedback_history) > 100:
            profile.feedback_history = profile.feedback_history[-100:]

        # Learn from feedback
        await self._update_preferences_from_feedback(profile)

        await self.save_profile(profile)

        logger.info(
            "Feedback recorded",
            user_id=user_id[:8],
            rating=rating,
            feedback_count=len(profile.feedback_history),
        )

    async def _update_preferences_from_feedback(self, profile: UserProfile) -> None:
        """Learn preferences from feedback patterns."""
        recent = profile.feedback_history[-self.min_feedback_for_learning:]

        if len(recent) < self.min_feedback_for_learning:
            return

        # Analyze high-rated vs low-rated responses
        high_rated = [f for f in recent if f.rating >= 4]
        low_rated = [f for f in recent if f.rating <= 2]

        if not high_rated:
            return

        # Learn preferred response length
        avg_high_length = sum(f.response_length for f in high_rated) / len(high_rated)
        if avg_high_length < 500:
            profile.preferences.preferred_length = "short"
        elif avg_high_length > 1500:
            profile.preferences.preferred_length = "detailed"
        else:
            profile.preferences.preferred_length = "medium"

        # Learn preferred format
        format_counts = Counter(f.response_format for f in high_rated)
        if format_counts:
            profile.preferences.preferred_format = format_counts.most_common(1)[0][0]

        # Detect expertise level from query complexity
        if profile.total_queries > 10:
            technical_patterns = ["how does", "explain the mechanism", "technical details"]
            simple_patterns = ["what is", "define", "simple explanation"]

            query_text = " ".join(f.query.lower() for f in recent)
            technical_count = sum(1 for p in technical_patterns if p in query_text)
            simple_count = sum(1 for p in simple_patterns if p in query_text)

            if technical_count > simple_count:
                profile.preferences.expertise_level = "expert"
            elif simple_count > technical_count:
                profile.preferences.expertise_level = "beginner"

        logger.debug(
            "Preferences updated",
            user_id=profile.user_id[:8],
            length=profile.preferences.preferred_length,
            format=profile.preferences.preferred_format,
            expertise=profile.preferences.expertise_level,
        )

    def _classify_query_pattern(self, query: str) -> str:
        """Classify query into a pattern category."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["what is", "define", "explain"]):
            return "definitional"
        elif any(w in query_lower for w in ["how to", "how do", "steps"]):
            return "procedural"
        elif any(w in query_lower for w in ["why", "because", "reason"]):
            return "causal"
        elif any(w in query_lower for w in ["compare", "difference", "versus"]):
            return "comparative"
        elif any(w in query_lower for w in ["find", "search", "look for"]):
            return "search"
        elif any(w in query_lower for w in ["summary", "summarize", "overview"]):
            return "summary"
        else:
            return "general"

    def get_prompt_adjustments(self, profile: UserProfile) -> str:
        """
        Generate prompt adjustments based on user profile.

        Args:
            profile: User profile

        Returns:
            String of adjustments to add to prompts
        """
        adjustments = []

        # Length preference
        if profile.preferences.preferred_length == "short":
            adjustments.append("Keep your response brief and concise (2-3 paragraphs max).")
        elif profile.preferences.preferred_length == "detailed":
            adjustments.append("Provide a detailed, comprehensive response with examples.")

        # Format preference
        if profile.preferences.preferred_format == "bullets":
            adjustments.append("Use bullet points for better readability.")
        elif profile.preferences.preferred_format == "structured":
            adjustments.append("Use clear sections with headings.")

        # Expertise level
        if profile.preferences.expertise_level == "expert":
            adjustments.append("Use technical terminology appropriate for an expert.")
        elif profile.preferences.expertise_level == "beginner":
            adjustments.append("Explain concepts simply, avoiding jargon.")

        # Citations preference
        if not profile.preferences.wants_citations:
            adjustments.append("Integrate source references naturally without explicit citations.")

        return "\n".join(adjustments) if adjustments else ""

    async def get_personalized_prompt_additions(self, user_id: str) -> str:
        """
        Get personalized prompt additions for a user.

        Args:
            user_id: User identifier

        Returns:
            Prompt additions string
        """
        profile = await self.get_profile(user_id)
        return self.get_prompt_adjustments(profile)

    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Get insights about a user's behavior and preferences.

        Args:
            user_id: User identifier

        Returns:
            Dictionary of insights
        """
        profile = await self.get_profile(user_id)

        # Calculate average rating
        recent_feedback = profile.feedback_history[-20:]
        avg_rating = sum(f.rating for f in recent_feedback) / len(recent_feedback) if recent_feedback else None

        # Get top query patterns
        top_patterns = sorted(
            profile.query_patterns.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "user_id": user_id,
            "total_queries": profile.total_queries,
            "member_since": profile.created_at.isoformat(),
            "last_active": profile.last_active.isoformat(),
            "preferences": {
                "length": profile.preferences.preferred_length,
                "format": profile.preferences.preferred_format,
                "expertise": profile.preferences.expertise_level,
            },
            "top_topics": profile.topics_of_interest[:5],
            "query_patterns": dict(top_patterns),
            "average_rating": avg_rating,
            "feedback_count": len(profile.feedback_history),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_service_instance: Optional[UserPersonalizationService] = None


def get_personalization_service(cache=None) -> UserPersonalizationService:
    """
    Get or create the personalization service singleton.

    Args:
        cache: Optional cache for persistence

    Returns:
        UserPersonalizationService instance
    """
    global _service_instance

    if _service_instance is None:
        _service_instance = UserPersonalizationService(cache=cache)

    return _service_instance


async def personalize_prompt(user_id: str, base_prompt: str) -> str:
    """
    Add personalization to a prompt.

    Args:
        user_id: User identifier
        base_prompt: Original prompt

    Returns:
        Personalized prompt
    """
    service = get_personalization_service()
    additions = await service.get_personalized_prompt_additions(user_id)

    if additions:
        return f"{base_prompt}\n\nUser preferences:\n{additions}"

    return base_prompt
