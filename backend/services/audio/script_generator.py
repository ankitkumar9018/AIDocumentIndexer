"""
AIDocumentIndexer - Script Generator
=====================================

Generates dialogue scripts for audio overviews using LLM.

Supports multiple formats:
- Deep Dive: Comprehensive exploration of document content
- Brief: 5-minute summary
- Critique: Constructive feedback and analysis
- Debate: Two contrasting viewpoints
- Lecture: Educational single-speaker format
- Interview: Q&A style with expert
"""

import json
import re
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator
from enum import Enum

import structlog
from pydantic import BaseModel

from backend.services.base import BaseService, ServiceException
from backend.db.models import AudioOverviewFormat

logger = structlog.get_logger(__name__)


class DialogueTurn(BaseModel):
    """A single turn in the dialogue."""
    speaker: str  # "host1", "host2", "expert", etc.
    text: str
    emotion: Optional[str] = None  # "curious", "excited", "thoughtful"
    pause_after: Optional[float] = None  # seconds


class DialogueScript(BaseModel):
    """Complete dialogue script for audio generation."""
    title: str
    format: str
    estimated_duration_seconds: int
    speakers: List[Dict[str, str]]  # [{"id": "host1", "name": "Alex", "voice": "alloy"}]
    turns: List[DialogueTurn]
    metadata: Dict[str, Any] = {}


# Format-specific prompts
FORMAT_PROMPTS = {
    AudioOverviewFormat.DEEP_DIVE: """
You are creating a script for an engaging podcast-style deep dive discussion between two hosts.

The hosts are having a natural, enthusiastic conversation exploring the documents in depth.
They should:
- Build on each other's points naturally
- Ask clarifying questions
- Share genuine curiosity and insights
- Use conversational language (not too formal)
- Include moments of "aha!" discovery
- Reference specific details from the documents
- Make complex topics accessible

Target duration: 15-20 minutes
""",

    AudioOverviewFormat.BRIEF: """
You are creating a script for a quick, focused summary between two hosts.

The hosts should:
- Get to the key points quickly
- Use clear, concise language
- Highlight the most important takeaways
- Keep energy high and engaging
- Avoid going too deep into any single topic

Target duration: 5 minutes
""",

    AudioOverviewFormat.CRITIQUE: """
You are creating a script for a thoughtful critique and analysis discussion.

The hosts should:
- Provide balanced analysis (strengths and weaknesses)
- Support opinions with evidence from documents
- Consider different perspectives
- Be constructive in criticism
- Discuss implications and applications

Target duration: 10-15 minutes
""",

    AudioOverviewFormat.DEBATE: """
You are creating a script for a debate between two hosts with contrasting viewpoints.

The hosts should:
- Take opposing positions on key topics
- Make strong arguments for their sides
- Challenge each other respectfully
- Use evidence from documents to support claims
- Eventually find some common ground

Target duration: 12-15 minutes
""",

    AudioOverviewFormat.LECTURE: """
You are creating a script for an educational lecture-style presentation.

The single presenter should:
- Explain concepts clearly and systematically
- Use examples and analogies
- Build from basics to more complex ideas
- Engage the audience with rhetorical questions
- Summarize key points periodically

Target duration: 10-15 minutes
""",

    AudioOverviewFormat.INTERVIEW: """
You are creating a script for an expert interview.

The interviewer asks insightful questions while the expert provides deep knowledge.
They should:
- Start with context-setting questions
- Progressively dive deeper
- Allow for detailed explanations
- Follow up on interesting points
- Summarize key insights

Target duration: 12-15 minutes
""",
}

# Duration preference multipliers
# These modify the base duration for each format
DURATION_MULTIPLIERS = {
    "short": 0.5,      # Half the standard duration
    "standard": 1.0,   # Default duration
    "extended": 1.5,   # 50% longer
}

# Base durations per format in minutes
BASE_DURATIONS = {
    AudioOverviewFormat.DEEP_DIVE: 17,    # 15-20 min → 17 avg
    AudioOverviewFormat.BRIEF: 5,          # 5 min
    AudioOverviewFormat.CRITIQUE: 12,      # 10-15 min → 12 avg
    AudioOverviewFormat.DEBATE: 13,        # 12-15 min → 13 avg
    AudioOverviewFormat.LECTURE: 12,       # 10-15 min → 12 avg
    AudioOverviewFormat.INTERVIEW: 13,     # 12-15 min → 13 avg
}

# Voice configurations per format
DEFAULT_SPEAKERS = {
    AudioOverviewFormat.DEEP_DIVE: [
        {"id": "host1", "name": "Alex", "voice": "alloy", "style": "conversational"},
        {"id": "host2", "name": "Jordan", "voice": "echo", "style": "conversational"},
    ],
    AudioOverviewFormat.BRIEF: [
        {"id": "host1", "name": "Alex", "voice": "alloy", "style": "energetic"},
        {"id": "host2", "name": "Jordan", "voice": "echo", "style": "energetic"},
    ],
    AudioOverviewFormat.CRITIQUE: [
        {"id": "host1", "name": "Alex", "voice": "alloy", "style": "thoughtful"},
        {"id": "host2", "name": "Jordan", "voice": "echo", "style": "analytical"},
    ],
    AudioOverviewFormat.DEBATE: [
        {"id": "host1", "name": "Alex", "voice": "alloy", "style": "passionate"},
        {"id": "host2", "name": "Jordan", "voice": "echo", "style": "passionate"},
    ],
    AudioOverviewFormat.LECTURE: [
        {"id": "lecturer", "name": "Professor Chen", "voice": "onyx", "style": "educational"},
    ],
    AudioOverviewFormat.INTERVIEW: [
        {"id": "interviewer", "name": "Alex", "voice": "alloy", "style": "curious"},
        {"id": "expert", "name": "Dr. Morgan", "voice": "fable", "style": "knowledgeable"},
    ],
}


class ScriptGenerator(BaseService):
    """
    Generates dialogue scripts for audio overviews.

    Uses LLM to create natural, engaging conversations based on
    document content and selected format.
    """

    def __init__(
        self,
        session=None,
        organization_id=None,
        user_id=None,
        llm_provider: str = "openai",
    ):
        super().__init__(session, organization_id, user_id)
        self.llm_provider = llm_provider

    def _calculate_target_duration(
        self,
        format: AudioOverviewFormat,
        duration_preference: str = "standard",
    ) -> int:
        """Calculate target duration in minutes based on format and preference."""
        base_duration = BASE_DURATIONS.get(format, 12)
        multiplier = DURATION_MULTIPLIERS.get(duration_preference, 1.0)
        return max(2, int(base_duration * multiplier))  # Minimum 2 minutes

    async def generate_script(
        self,
        document_contents: List[Dict[str, Any]],
        format: AudioOverviewFormat,
        title: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        target_duration_minutes: Optional[int] = None,
        duration_preference: str = "standard",
        host1_name: Optional[str] = None,
        host2_name: Optional[str] = None,
    ) -> DialogueScript:
        """
        Generate a dialogue script from document contents.

        Args:
            document_contents: List of dicts with 'name', 'content', 'summary' keys
            format: The audio format to generate
            title: Optional custom title
            custom_instructions: Additional instructions for the script
            target_duration_minutes: Override default duration (takes precedence over preference)
            duration_preference: 'short', 'standard', or 'extended'

        Returns:
            DialogueScript with all dialogue turns
        """
        # Normalize format to enum (handle both string and enum)
        if isinstance(format, str):
            format = AudioOverviewFormat(format)

        # Calculate target duration
        if target_duration_minutes is None:
            target_duration_minutes = self._calculate_target_duration(format, duration_preference)

        self.log_info(
            "Generating script",
            format=format.value,
            document_count=len(document_contents),
            target_duration_minutes=target_duration_minutes,
            duration_preference=duration_preference,
        )

        # Get format-specific configuration
        format_prompt = FORMAT_PROMPTS.get(format, FORMAT_PROMPTS[AudioOverviewFormat.DEEP_DIVE])
        speakers = DEFAULT_SPEAKERS.get(format, DEFAULT_SPEAKERS[AudioOverviewFormat.DEEP_DIVE])

        # Apply custom speaker names if provided
        if host1_name or host2_name:
            speakers = [dict(s) for s in speakers]  # Make copies to avoid modifying defaults
            for speaker in speakers:
                if speaker["id"] == "host1" and host1_name:
                    speaker["name"] = host1_name
                elif speaker["id"] == "host2" and host2_name:
                    speaker["name"] = host2_name
                elif speaker["id"] == "interviewer" and host1_name:
                    speaker["name"] = host1_name
                elif speaker["id"] == "expert" and host2_name:
                    speaker["name"] = host2_name
                elif speaker["id"] == "lecturer" and host1_name:
                    speaker["name"] = host1_name

        # Build document context
        doc_context = self._build_document_context(document_contents)

        # Generate the script using LLM
        script_data = await self._call_llm_for_script(
            doc_context=doc_context,
            format_prompt=format_prompt,
            speakers=speakers,
            title=title,
            custom_instructions=custom_instructions,
            target_duration_minutes=target_duration_minutes,
        )

        # Parse and validate
        script = self._parse_script_response(script_data, format, speakers, title)

        self.log_info(
            "Script generated",
            turn_count=len(script.turns),
            estimated_duration=script.estimated_duration_seconds,
        )

        return script

    async def generate_script_streaming(
        self,
        document_contents: List[Dict[str, Any]],
        format: AudioOverviewFormat,
        title: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        duration_preference: str = "standard",
        host1_name: Optional[str] = None,
        host2_name: Optional[str] = None,
    ) -> AsyncGenerator[DialogueTurn, None]:
        """
        Generate script turns as a stream for real-time processing.

        Yields DialogueTurn objects as they are generated.
        """
        # Normalize format to enum (handle both string and enum)
        if isinstance(format, str):
            format = AudioOverviewFormat(format)

        # Calculate target duration
        target_duration_minutes = self._calculate_target_duration(format, duration_preference)

        self.log_info(
            "Starting streaming script generation",
            format=format.value,
            target_duration_minutes=target_duration_minutes,
            duration_preference=duration_preference,
        )

        format_prompt = FORMAT_PROMPTS.get(format, FORMAT_PROMPTS[AudioOverviewFormat.DEEP_DIVE])
        speakers = DEFAULT_SPEAKERS.get(format, DEFAULT_SPEAKERS[AudioOverviewFormat.DEEP_DIVE])

        # Apply custom speaker names if provided
        if host1_name or host2_name:
            speakers = [dict(s) for s in speakers]  # Make copies to avoid modifying defaults
            for speaker in speakers:
                if speaker["id"] == "host1" and host1_name:
                    speaker["name"] = host1_name
                elif speaker["id"] == "host2" and host2_name:
                    speaker["name"] = host2_name
                elif speaker["id"] == "interviewer" and host1_name:
                    speaker["name"] = host1_name
                elif speaker["id"] == "expert" and host2_name:
                    speaker["name"] = host2_name
                elif speaker["id"] == "lecturer" and host1_name:
                    speaker["name"] = host1_name

        doc_context = self._build_document_context(document_contents)

        async for turn in self._stream_llm_script(
            doc_context=doc_context,
            format_prompt=format_prompt,
            speakers=speakers,
            custom_instructions=custom_instructions,
            target_duration_minutes=target_duration_minutes,
        ):
            yield turn

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON issues from LLM output.

        Handles:
        - Trailing commas before } or ]
        - Smart quotes (curly quotes) → straight quotes
        - Unescaped newlines in strings
        - Single quotes used instead of double quotes (carefully)
        - Control characters in strings
        """
        repaired = json_str

        # Replace smart quotes with straight quotes
        repaired = repaired.replace('"', '"').replace('"', '"')
        repaired = repaired.replace(''', "'").replace(''', "'")

        # Remove trailing commas before } or ]
        # This handles: {"a": 1,} and ["a",]
        repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)

        # Handle unescaped newlines within strings
        # First, let's track if we're inside a string and escape any raw newlines
        result = []
        in_string = False
        escape_next = False
        i = 0
        while i < len(repaired):
            char = repaired[i]

            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue

            if char == '\\':
                escape_next = True
                result.append(char)
                i += 1
                continue

            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue

            if in_string:
                # If we find a raw newline inside a string, escape it
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                # Remove other control characters
                elif ord(char) < 32:
                    pass  # Skip control characters
                else:
                    result.append(char)
            else:
                result.append(char)
            i += 1

        repaired = ''.join(result)

        self.log_info("JSON repair attempted", original_len=len(json_str), repaired_len=len(repaired))
        return repaired

    def _build_document_context(self, document_contents: List[Dict[str, Any]]) -> str:
        """Build context string from document contents."""
        context_parts = []

        for doc in document_contents:
            name = doc.get("name", "Untitled")
            content = doc.get("content", "")
            summary = doc.get("summary", "")

            # Truncate content if too long (keep first 10k chars)
            if len(content) > 10000:
                content = content[:10000] + "\n[... content truncated ...]"

            part = f"""
### Document: {name}

Summary: {summary}

Content:
{content}
"""
            context_parts.append(part)

        return "\n\n---\n\n".join(context_parts)

    async def _call_llm_for_script(
        self,
        doc_context: str,
        format_prompt: str,
        speakers: List[Dict[str, str]],
        title: Optional[str],
        custom_instructions: Optional[str],
        target_duration_minutes: Optional[int],
    ) -> Dict[str, Any]:
        """Call LLM to generate the script."""
        from backend.services.llm import EnhancedLLMFactory
        from langchain_core.messages import SystemMessage, HumanMessage

        # Build speaker info with names for the prompt
        speaker_info = []
        for s in speakers:
            speaker_info.append(f'{s["id"]} = {s["name"]}')
        speaker_mapping = ", ".join(speaker_info)

        system_prompt = f"""You are an expert podcast script writer creating NATURAL, HUMAN-SOUNDING dialogue.

{format_prompt}

SPEAKERS: {speaker_mapping}
Use the speaker IDs (host1, host2) in the JSON but write dialogue AS IF you are {speakers[0]["name"] if speakers else "the first host"} and {speakers[1]["name"] if len(speakers) > 1 else "the second host"}.

OUTPUT FORMAT:
Return a JSON object with this structure:
{{
    "title": "Episode title",
    "turns": [
        {{"speaker": "host1", "text": "Hey everyone! Welcome back to the show. So, um, today we're diving into something really interesting.", "emotion": "excited"}},
        {{"speaker": "host2", "text": "Yeah, I've been looking forward to this one! So... where do we even start?", "emotion": "curious"}},
        {{"speaker": "host1", "text": "Right, right. Well, let me think... I'd say the most surprising thing is...", "emotion": "thoughtful"}},
        {{"speaker": "host2", "text": "Oh wow, really? That's fascinating. Ha! I didn't expect that.", "emotion": "surprised"}},
        ...
    ]
}}

CRITICAL - MAKE IT SOUND HUMAN, NOT ROBOTIC:
The dialogue MUST sound like real people talking, not a scripted presentation. Include:

1. FILLER WORDS (use in ~40% of turns): "um", "uh", "like", "you know", "actually", "basically", "I mean", "kind of", "sort of"
   Example: "So, um, the interesting thing here is..." NOT "The interesting thing here is..."

2. REACTIONS AND INTERJECTIONS (use in ~50% of turns):
   - Agreement: "Yeah", "Right, right", "Mm-hmm", "Exactly", "Totally", "Oh for sure", "Absolutely"
   - Surprise: "Oh!", "Wow", "Wait, really?", "No way!", "Huh", "Whoa", "Oh my gosh"
   - Thinking: "Hmm...", "Let me think...", "So...", "Well...", "I wonder..."
   - Amusement: "Ha!", "Ha ha", "That's funny", "Heh", "Oh man"
   - Skepticism: "I don't know about that...", "Really though?", "Hmm, interesting..."

3. NATURAL PAUSES using "..." (use VERY frequently - almost every turn):
   Example: "So... I was reading this and... honestly, it blew my mind."
   Use pauses mid-thought: "The thing is... when you really think about it..."

4. INCOMPLETE THOUGHTS AND SELF-CORRECTIONS:
   Example: "And the thing is— well, actually, let me back up a second."
   Example: "It's like... no wait, that's not quite right. What I mean is..."

5. CASUAL CONVERSATIONAL STARTERS:
   - "Here's the thing..."
   - "You know what's crazy?"
   - "I gotta say..."
   - "Okay, so..."
   - "The way I see it..."
   - "Can I just say..."
   - "This is where it gets interesting..."
   - "So get this..."

6. VERBAL ACKNOWLEDGMENTS when listening:
   - "Mm-hmm", "Right", "Yeah yeah", "Okay", "Got it", "I see"
   - Use short responses before longer ones: "Yeah, yeah, and that's exactly why..."

7. NATURAL SPEECH PATTERNS:
   - Vary sentence length dramatically (mix very short with longer)
   - Use contractions always: "don't", "can't", "it's", "that's", "I'm"
   - Repeat words for emphasis: "It's really, really important"
   - Trail off sometimes: "And then the whole thing just..."
   - Ask rhetorical questions: "Right?", "You know?", "Isn't that wild?"

8. EMOTIONAL VARIATION - vary emotions throughout:
   - excited, curious, thoughtful, surprised, amused, skeptical, impressed, confused, confident

BAD EXAMPLE (too robotic):
"The document discusses three main points. First, we have the introduction. Second, there is the methodology."

GOOD EXAMPLE (natural):
"Okay so... the document, um, it covers like three big things. And honestly? The first one really caught my attention. So basically... wait, let me back up. You know what's funny is..."

ANOTHER GOOD EXAMPLE:
"Yeah yeah, and— oh this is the good part— so they found that... get this... the numbers were completely off. Like, not even close!"

NEVER USE:
- Bracketed actions like [laughs], [chuckles], [sighs] - TTS reads these literally as words!
- Formal academic language
- Perfect grammar in every sentence
- Robotic transitions like "Moving on to the next point" or "Let's discuss"
- Overly polished, rehearsed-sounding phrases
- Starting every response the same way
"""

        if custom_instructions:
            system_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{custom_instructions}"

        if target_duration_minutes:
            words_target = target_duration_minutes * 150  # ~150 words per minute
            system_prompt += f"\n\nTARGET: Approximately {words_target} total words ({target_duration_minutes} minutes)"

        user_prompt = f"""Create an engaging podcast script based on these documents:

{doc_context}

{f'Title suggestion: {title}' if title else 'Create an appropriate title.'}

Generate the complete script in JSON format."""

        # Get LLM using the default chat configuration (same as chat mode)
        # This uses whatever LLM is configured as default in Admin > Settings > LLM Configuration
        llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="chat",  # Use default chat LLM
            user_id=str(self._user_id) if self._user_id else None,
        )

        # Log the LLM config being used
        self.log_info(
            "Using LLM for script generation",
            provider=config.provider_type if config else "unknown",
            model=config.model if config else "unknown",
            has_api_key=bool(config.api_key) if config else False,
        )

        # Check if we have a valid API key
        if config and not config.api_key and config.provider_type not in ["ollama"]:
            raise ServiceException(
                f"No API key configured for LLM provider '{config.provider_type}'. "
                f"Please configure the API key in Admin > Settings > LLM Configuration or set the environment variable.",
                code="LLM_CONFIG_ERROR",
            )

        # Call LLM with messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        self.log_info("Calling LLM for script generation", prompt_length=len(user_prompt))
        try:
            response = await llm.ainvoke(messages)
            content = response.content
        except Exception as e:
            self.log_error("LLM call failed", error=e)
            raise ServiceException(
                f"LLM call failed: {str(e)}. Check your API key and network connection.",
                code="LLM_CALL_ERROR",
            )
        self.log_info("LLM response received", response_length=len(content) if content else 0)

        # Parse JSON response - handle markdown code blocks
        try:
            if not content or not content.strip():
                raise ServiceException(
                    "LLM returned empty response",
                    code="SCRIPT_GENERATION_ERROR",
                    details={"response": "empty"},
                )

            # Extract JSON from response - handle various formats:
            # 1. Pure JSON
            # 2. Markdown code blocks (```json ... ``` or ``` ... ```)
            # 3. Text before/after JSON block
            cleaned_content = content.strip()

            # Try to find JSON in code blocks first
            # Match ```json ... ``` or ``` ... ```
            code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', cleaned_content)
            if code_block_match:
                cleaned_content = code_block_match.group(1).strip()
            else:
                # No code block, try to find JSON object directly
                # Look for content starting with { and ending with }
                json_match = re.search(r'(\{[\s\S]*\})', cleaned_content)
                if json_match:
                    cleaned_content = json_match.group(1).strip()

            self.log_info("Parsing JSON response", cleaned_length=len(cleaned_content))

            # Try to parse, and if it fails, attempt to repair common LLM JSON issues
            try:
                return json.loads(cleaned_content)
            except json.JSONDecodeError:
                # Attempt to repair common JSON issues from LLMs
                repaired = self._repair_json(cleaned_content)
                return json.loads(repaired)

        except json.JSONDecodeError as e:
            self.log_error("Failed to parse LLM response as JSON", error=e, raw_content=content[:1000] if content else "empty")
            raise ServiceException(
                f"Failed to generate valid script: {str(e)}",
                code="SCRIPT_GENERATION_ERROR",
                details={"raw_response": content[:500] if content else "empty"},
            )

    async def _stream_llm_script(
        self,
        doc_context: str,
        format_prompt: str,
        speakers: List[Dict[str, str]],
        custom_instructions: Optional[str],
        target_duration_minutes: int = 12,
    ) -> AsyncGenerator[DialogueTurn, None]:
        """Stream dialogue turns from LLM."""
        # For streaming, we use a different approach - request one turn at a time
        # This is a simplified implementation; production would use proper streaming
        from backend.services.llm import EnhancedLLMFactory
        from langchain_core.messages import SystemMessage, HumanMessage

        # Get LLM using the default chat configuration (same as chat mode)
        # This uses whatever LLM is configured as default in Admin > Settings > LLM Configuration
        llm, config = await EnhancedLLMFactory.get_chat_model_for_operation(
            operation="chat",  # Use default chat LLM
            user_id=str(self._user_id) if self._user_id else None,
        )

        # Build speaker info with names
        speaker_info = []
        for s in speakers:
            speaker_info.append(f'{s["id"]} = {s["name"]}')
        speaker_mapping = ", ".join(speaker_info)

        conversation_history = []
        turn_count = 0
        # Calculate max turns based on target duration (approx 3-4 turns per minute)
        max_turns = max(10, int(target_duration_minutes * 3.5))

        # Build duration guidance
        duration_guidance = f"TARGET DURATION: {target_duration_minutes} minutes (approximately {max_turns} turns)"

        system_prompt = f"""You are generating a NATURAL, HUMAN-SOUNDING podcast script one turn at a time.

{format_prompt}

{duration_guidance}

SPEAKERS: {speaker_mapping}

For each turn, respond with JSON:
{{"speaker": "host1", "text": "So, um, here's the thing...", "emotion": "curious", "continue": true}}

CRITICAL - SOUND HUMAN, NOT ROBOTIC:
1. Use filler words (~40% of turns): "um", "uh", "like", "you know", "actually", "kind of", "basically"
2. Start with reactions (~50% of turns): "Yeah", "Oh!", "Hmm...", "Right, right", "Ha!", "Whoa", "Okay so"
3. Use "..." for natural pauses FREQUENTLY: "So... I was thinking...", "The thing is... when you..."
4. Sound casual and conversational: "Here's the thing...", "Okay so...", "I gotta say...", "Can I just say..."
5. Use verbal acknowledgments: "Mm-hmm", "Yeah yeah", "Right", "Got it"
6. Vary emotions: excited, curious, thoughtful, surprised, amused, skeptical, impressed
7. Mix short and long responses. Use rhetorical questions: "Right?", "You know?"
8. Self-correct sometimes: "It's like... no wait, what I mean is..."

EXAMPLE of natural turn:
{{"speaker": "host1", "text": "Yeah, so... the interesting thing is, um, when you look at this closely... it's actually pretty wild. Like, not what you'd expect at all.", "emotion": "curious", "continue": true}}

ANOTHER EXAMPLE:
{{"speaker": "host2", "text": "Oh wow, really? That's— wait, so you're saying the whole thing was just... off?", "emotion": "surprised", "continue": true}}

NEVER USE: [laughs], [sighs], or any bracketed actions - TTS reads them as words!
NEVER start every turn the same way. Vary your openings!

Set "continue" to false when approaching the end.
{custom_instructions or ""}
"""

        while turn_count < max_turns:
            history_text = "\n".join([
                f"{t['speaker']}: {t['text']}" for t in conversation_history[-10:]
            ])

            user_prompt = f"""Documents:
{doc_context[:5000]}

Previous turns:
{history_text if history_text else "(Start of conversation)"}

Generate the next turn:"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = await llm.ainvoke(messages)
            content = response.content

            try:
                # Extract JSON from response - handle various formats
                cleaned_content = content.strip()

                # Try to find JSON in code blocks first
                code_block_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', cleaned_content)
                if code_block_match:
                    cleaned_content = code_block_match.group(1).strip()
                else:
                    # No code block, try to find JSON object directly
                    json_match = re.search(r'(\{[\s\S]*\})', cleaned_content)
                    if json_match:
                        cleaned_content = json_match.group(1).strip()

                # Try to parse, and if it fails, attempt to repair
                try:
                    turn_data = json.loads(cleaned_content)
                except json.JSONDecodeError:
                    cleaned_content = self._repair_json(cleaned_content)
                    turn_data = json.loads(cleaned_content)
                turn = DialogueTurn(
                    speaker=turn_data["speaker"],
                    text=turn_data["text"],
                    emotion=turn_data.get("emotion"),
                )
                conversation_history.append(turn_data)
                yield turn

                if not turn_data.get("continue", True):
                    break

                turn_count += 1

            except (json.JSONDecodeError, KeyError) as e:
                self.log_warning("Failed to parse streaming turn", error=str(e), raw_content=content[:200] if content else "empty")
                break

    def _parse_script_response(
        self,
        data: Dict[str, Any],
        format: AudioOverviewFormat,
        speakers: List[Dict[str, str]],
        title: Optional[str],
    ) -> DialogueScript:
        """Parse and validate the LLM response into a DialogueScript."""
        # Normalize format to enum (handle both string and enum)
        if isinstance(format, str):
            format = AudioOverviewFormat(format)

        turns = []
        total_words = 0

        for turn_data in data.get("turns", []):
            turn = DialogueTurn(
                speaker=turn_data.get("speaker", "host1"),
                text=turn_data.get("text", ""),
                emotion=turn_data.get("emotion"),
                pause_after=turn_data.get("pause_after"),
            )
            turns.append(turn)
            total_words += len(turn.text.split())

        # Estimate duration (150 words per minute average for speech)
        estimated_seconds = int((total_words / 150) * 60)

        return DialogueScript(
            title=data.get("title", title or "Audio Overview"),
            format=format.value,
            estimated_duration_seconds=estimated_seconds,
            speakers=speakers,
            turns=turns,
            metadata={
                "word_count": total_words,
                "turn_count": len(turns),
            },
        )

    async def estimate_duration(
        self,
        document_contents: List[Dict[str, Any]],
        format: AudioOverviewFormat,
    ) -> Dict[str, int]:
        """
        Estimate the duration of an audio overview without generating it.

        Returns dict with min, max, and target duration in seconds.
        """
        # Normalize format to enum (handle both string and enum)
        if isinstance(format, str):
            format = AudioOverviewFormat(format)

        # Calculate based on document length and format
        total_chars = sum(
            len(doc.get("content", "")) + len(doc.get("summary", ""))
            for doc in document_contents
        )

        # Base durations per format (in seconds)
        format_durations = {
            AudioOverviewFormat.BRIEF: {"min": 180, "max": 360, "target": 300},
            AudioOverviewFormat.DEEP_DIVE: {"min": 600, "max": 1200, "target": 900},
            AudioOverviewFormat.CRITIQUE: {"min": 480, "max": 900, "target": 720},
            AudioOverviewFormat.DEBATE: {"min": 600, "max": 1080, "target": 840},
            AudioOverviewFormat.LECTURE: {"min": 480, "max": 900, "target": 720},
            AudioOverviewFormat.INTERVIEW: {"min": 600, "max": 1080, "target": 840},
        }

        base = format_durations.get(format, format_durations[AudioOverviewFormat.DEEP_DIVE])

        # Adjust based on content length
        content_factor = min(2.0, max(0.5, total_chars / 10000))

        return {
            "min_seconds": int(base["min"] * content_factor),
            "max_seconds": int(base["max"] * content_factor),
            "target_seconds": int(base["target"] * content_factor),
        }
