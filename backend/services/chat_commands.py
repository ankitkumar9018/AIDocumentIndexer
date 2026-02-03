"""
AIDocumentIndexer - Chat Commands Service
==========================================

Enables users to call features, agents, and skills directly from chat.
Parses special commands and routes them to appropriate handlers.

Commands:
- /organize - Organize documents into smart collections
- /insights - Show insight feed
- /check-duplicates - Check for duplicates
- /highlights - Generate smart highlights
- /check-conflicts - Analyze for conflicts
- /think - Extended thinking
- /ensemble - Multi-model query
- /verify - Verify answer
- /agent - Call an agent
- /skill - Execute a skill
- /help - Show help
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio

import structlog

logger = structlog.get_logger(__name__)


class CommandCategory(str, Enum):
    """Categories of chat commands."""
    INTELLIGENCE = "intelligence"  # AI/reasoning features
    ORGANIZATION = "organization"  # Collections, tagging
    ANALYSIS = "analysis"  # DNA, conflicts, insights
    AGENTS = "agents"  # Agent execution
    SKILLS = "skills"  # Skill execution
    TOOLS = "tools"  # Calculator, converter, etc.
    SYSTEM = "system"  # Help, settings


@dataclass
class CommandDefinition:
    """Definition of a chat command."""
    name: str
    aliases: List[str]
    description: str
    usage: str
    examples: List[str]
    category: CommandCategory
    requires_args: bool = False
    min_args: int = 0
    max_args: int = -1  # -1 means unlimited
    handler: Optional[str] = None  # Handler function name


@dataclass
class CommandResult:
    """Result of executing a command."""
    success: bool
    command: str
    message: str
    data: Optional[Dict[str, Any]] = None
    follow_up_actions: List[str] = field(default_factory=list)
    show_in_chat: bool = True
    execute_rag: bool = False  # Whether to also run RAG query
    modified_query: Optional[str] = None  # Modified query for RAG


@dataclass
class ParsedCommand:
    """Parsed command from user input."""
    command: str
    args: List[str]
    raw_args: str
    full_text: str
    has_command: bool = True


class ChatCommandsService:
    """
    Service for parsing and executing chat commands.

    Allows users to invoke features directly from the chat interface
    using slash commands like /organize, /think, /agent, etc.
    """

    def __init__(self):
        self._commands: Dict[str, CommandDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        self._register_default_commands()

    def _register_default_commands(self):
        """Register all default commands."""

        # Intelligence Commands
        self.register_command(CommandDefinition(
            name="think",
            aliases=["thinking", "reason", "analyze"],
            description="Use extended thinking for complex queries",
            usage="/think <query>",
            examples=[
                "/think What are the implications of our Q3 results?",
                "/think Compare approaches A and B for this problem",
            ],
            category=CommandCategory.INTELLIGENCE,
            requires_args=True,
            min_args=1,
            handler="handle_think",
        ))

        self.register_command(CommandDefinition(
            name="ensemble",
            aliases=["multi", "vote", "consensus"],
            description="Query multiple AI models and get consensus",
            usage="/ensemble <query>",
            examples=[
                "/ensemble What's the best approach for caching?",
            ],
            category=CommandCategory.INTELLIGENCE,
            requires_args=True,
            min_args=1,
            handler="handle_ensemble",
        ))

        self.register_command(CommandDefinition(
            name="verify",
            aliases=["check", "validate"],
            description="Verify an answer against sources",
            usage="/verify <statement>",
            examples=[
                "/verify The project deadline is March 15th",
            ],
            category=CommandCategory.INTELLIGENCE,
            requires_args=True,
            min_args=1,
            handler="handle_verify",
        ))

        self.register_command(CommandDefinition(
            name="cot",
            aliases=["chain", "step-by-step"],
            description="Use chain-of-thought reasoning",
            usage="/cot <question>",
            examples=[
                "/cot How should we approach this migration?",
            ],
            category=CommandCategory.INTELLIGENCE,
            requires_args=True,
            min_args=1,
            handler="handle_cot",
        ))

        # Organization Commands
        self.register_command(CommandDefinition(
            name="organize",
            aliases=["cluster", "group", "categorize"],
            description="Organize documents into smart collections",
            usage="/organize [strategy]",
            examples=[
                "/organize",
                "/organize by-topic",
                "/organize by-entity",
            ],
            category=CommandCategory.ORGANIZATION,
            requires_args=False,
            handler="handle_organize",
        ))

        # Analysis Commands
        self.register_command(CommandDefinition(
            name="insights",
            aliases=["feed", "recommendations"],
            description="Show your personalized insight feed",
            usage="/insights [limit]",
            examples=[
                "/insights",
                "/insights 10",
            ],
            category=CommandCategory.ANALYSIS,
            requires_args=False,
            handler="handle_insights",
        ))

        self.register_command(CommandDefinition(
            name="check-duplicates",
            aliases=["duplicates", "dups", "similar"],
            description="Check for duplicate content",
            usage="/check-duplicates <content or document>",
            examples=[
                "/check-duplicates This is some content to check",
            ],
            category=CommandCategory.ANALYSIS,
            requires_args=False,
            handler="handle_duplicates",
        ))

        self.register_command(CommandDefinition(
            name="highlights",
            aliases=["highlight", "key-points", "summarize"],
            description="Generate smart highlights for content",
            usage="/highlights <document_name or content>",
            examples=[
                "/highlights Q3 Report",
                "/highlights the last document",
            ],
            category=CommandCategory.ANALYSIS,
            requires_args=False,
            handler="handle_highlights",
        ))

        self.register_command(CommandDefinition(
            name="check-conflicts",
            aliases=["conflicts", "contradictions", "inconsistencies"],
            description="Check for conflicting information",
            usage="/check-conflicts [topic]",
            examples=[
                "/check-conflicts",
                "/check-conflicts pricing",
            ],
            category=CommandCategory.ANALYSIS,
            requires_args=False,
            handler="handle_conflicts",
        ))

        self.register_command(CommandDefinition(
            name="digest",
            aliases=["summary", "daily", "weekly"],
            description="Get a digest of recent activity",
            usage="/digest [daily|weekly]",
            examples=[
                "/digest",
                "/digest weekly",
            ],
            category=CommandCategory.ANALYSIS,
            requires_args=False,
            handler="handle_digest",
        ))

        # Agent Commands
        self.register_command(CommandDefinition(
            name="agent",
            aliases=["run-agent", "execute-agent"],
            description="Run a specific agent",
            usage="/agent <agent_name> [task]",
            examples=[
                "/agent researcher Find information about competitors",
                "/agent writer Draft a summary of the meeting notes",
            ],
            category=CommandCategory.AGENTS,
            requires_args=True,
            min_args=1,
            handler="handle_agent",
        ))

        self.register_command(CommandDefinition(
            name="agents",
            aliases=["list-agents", "available-agents"],
            description="List available agents",
            usage="/agents",
            examples=["/agents"],
            category=CommandCategory.AGENTS,
            requires_args=False,
            handler="handle_list_agents",
        ))

        # Skill Commands
        self.register_command(CommandDefinition(
            name="skill",
            aliases=["run-skill", "execute-skill"],
            description="Execute a skill",
            usage="/skill <skill_name> [args]",
            examples=[
                "/skill summarize last 5 documents",
                "/skill translate to Spanish",
            ],
            category=CommandCategory.SKILLS,
            requires_args=True,
            min_args=1,
            handler="handle_skill",
        ))

        self.register_command(CommandDefinition(
            name="skills",
            aliases=["list-skills", "available-skills"],
            description="List available skills",
            usage="/skills",
            examples=["/skills"],
            category=CommandCategory.SKILLS,
            requires_args=False,
            handler="handle_list_skills",
        ))

        # Tool Commands
        self.register_command(CommandDefinition(
            name="calc",
            aliases=["calculate", "math"],
            description="Perform a calculation",
            usage="/calc <expression>",
            examples=[
                "/calc 15% of 250",
                "/calc sqrt(144) + 5^2",
            ],
            category=CommandCategory.TOOLS,
            requires_args=True,
            min_args=1,
            handler="handle_calc",
        ))

        self.register_command(CommandDefinition(
            name="convert",
            aliases=["conv", "unit"],
            description="Convert units",
            usage="/convert <value> <from_unit> to <to_unit>",
            examples=[
                "/convert 100 km to miles",
                "/convert 72 fahrenheit to celsius",
            ],
            category=CommandCategory.TOOLS,
            requires_args=True,
            min_args=3,
            handler="handle_convert",
        ))

        self.register_command(CommandDefinition(
            name="date",
            aliases=["time", "when"],
            description="Date calculations",
            usage="/date <operation>",
            examples=[
                "/date today",
                "/date 30 days from now",
                "/date days between 2024-01-01 and 2024-12-31",
            ],
            category=CommandCategory.TOOLS,
            requires_args=False,
            handler="handle_date",
        ))

        # System Commands
        self.register_command(CommandDefinition(
            name="help",
            aliases=["?", "commands", "h"],
            description="Show help for commands",
            usage="/help [command]",
            examples=[
                "/help",
                "/help think",
                "/help agents",
            ],
            category=CommandCategory.SYSTEM,
            requires_args=False,
            handler="handle_help",
        ))

        self.register_command(CommandDefinition(
            name="settings",
            aliases=["config", "preferences"],
            description="View or change settings",
            usage="/settings [setting_name] [value]",
            examples=[
                "/settings",
                "/settings thinking_level high",
            ],
            category=CommandCategory.SYSTEM,
            requires_args=False,
            handler="handle_settings",
        ))

    def register_command(self, command: CommandDefinition):
        """Register a command."""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._commands[alias] = command

    def parse(self, text: str) -> ParsedCommand:
        """
        Parse user input to extract command if present.

        Args:
            text: User input text

        Returns:
            ParsedCommand with parsed information
        """
        text = text.strip()

        # Check if starts with /
        if not text.startswith("/"):
            return ParsedCommand(
                command="",
                args=[],
                raw_args=text,
                full_text=text,
                has_command=False,
            )

        # Extract command and args
        parts = text[1:].split(maxsplit=1)
        command = parts[0].lower() if parts else ""
        raw_args = parts[1] if len(parts) > 1 else ""
        args = raw_args.split() if raw_args else []

        return ParsedCommand(
            command=command,
            args=args,
            raw_args=raw_args,
            full_text=text,
            has_command=True,
        )

    async def execute(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> CommandResult:
        """
        Execute a parsed command.

        Args:
            parsed: Parsed command
            user_id: User executing the command
            context: Additional context (session, documents, etc.)

        Returns:
            CommandResult with execution result
        """
        if not parsed.has_command:
            return CommandResult(
                success=False,
                command="",
                message="No command found",
                execute_rag=True,
                modified_query=parsed.full_text,
            )

        # Look up command
        cmd_def = self._commands.get(parsed.command)
        if not cmd_def:
            # Unknown command - suggest similar
            suggestions = self._find_similar_commands(parsed.command)
            return CommandResult(
                success=False,
                command=parsed.command,
                message=f"Unknown command: /{parsed.command}",
                data={"suggestions": suggestions},
                follow_up_actions=["Try /help to see available commands"],
            )

        # Validate args
        if cmd_def.requires_args and len(parsed.args) < cmd_def.min_args:
            return CommandResult(
                success=False,
                command=parsed.command,
                message=f"Missing required arguments. Usage: {cmd_def.usage}",
                data={"examples": cmd_def.examples},
            )

        # Execute handler
        handler_name = cmd_def.handler or f"handle_{cmd_def.name.replace('-', '_')}"
        handler = getattr(self, handler_name, None)

        if not handler:
            logger.warning(f"No handler found for command: {cmd_def.name}")
            return CommandResult(
                success=False,
                command=parsed.command,
                message=f"Command handler not implemented: {cmd_def.name}",
            )

        try:
            result = await handler(parsed, user_id, context or {})
            return result
        except Exception as e:
            logger.error(f"Command execution failed: {e}", exc_info=True)
            return CommandResult(
                success=False,
                command=parsed.command,
                message=f"Error executing command: {str(e)}",
            )

    def _find_similar_commands(self, command: str, limit: int = 3) -> List[str]:
        """Find similar command names for suggestions."""
        all_commands = set(c.name for c in self._commands.values())
        similar = []

        for cmd in all_commands:
            # Simple similarity check
            if command in cmd or cmd in command:
                similar.append(cmd)
            elif len(set(command) & set(cmd)) > len(command) // 2:
                similar.append(cmd)

        return similar[:limit]

    def get_all_commands(self) -> List[CommandDefinition]:
        """
        Get all unique registered commands.

        Returns list of CommandDefinition objects (excludes aliases).
        """
        # Get unique commands (aliases map to same definition)
        seen_names = set()
        unique_commands = []

        for cmd_def in self._commands.values():
            if cmd_def.name not in seen_names:
                seen_names.add(cmd_def.name)
                unique_commands.append(cmd_def)

        # Sort by category and name
        unique_commands.sort(key=lambda c: (c.category.value, c.name))
        return unique_commands

    # =============================================================================
    # Command Handlers
    # =============================================================================

    async def handle_help(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /help command."""
        if parsed.args:
            # Help for specific command
            cmd_name = parsed.args[0].lower().lstrip("/")
            cmd_def = self._commands.get(cmd_name)

            if cmd_def:
                return CommandResult(
                    success=True,
                    command="help",
                    message=f"**/{cmd_def.name}** - {cmd_def.description}\n\n"
                            f"**Usage:** `{cmd_def.usage}`\n\n"
                            f"**Examples:**\n" + "\n".join(f"- `{e}`" for e in cmd_def.examples),
                    data={"command": cmd_def.name},
                )
            else:
                return CommandResult(
                    success=False,
                    command="help",
                    message=f"Unknown command: {cmd_name}",
                )

        # General help - group by category
        help_text = "# Available Commands\n\n"

        by_category: Dict[CommandCategory, List[CommandDefinition]] = {}
        seen = set()

        for cmd in self._commands.values():
            if cmd.name in seen:
                continue
            seen.add(cmd.name)

            if cmd.category not in by_category:
                by_category[cmd.category] = []
            by_category[cmd.category].append(cmd)

        category_names = {
            CommandCategory.INTELLIGENCE: "🧠 Intelligence",
            CommandCategory.ORGANIZATION: "📁 Organization",
            CommandCategory.ANALYSIS: "📊 Analysis",
            CommandCategory.AGENTS: "🤖 Agents",
            CommandCategory.SKILLS: "⚡ Skills",
            CommandCategory.TOOLS: "🔧 Tools",
            CommandCategory.SYSTEM: "⚙️ System",
        }

        for category in CommandCategory:
            if category not in by_category:
                continue

            help_text += f"## {category_names.get(category, category.value)}\n"
            for cmd in sorted(by_category[category], key=lambda c: c.name):
                help_text += f"- `/{cmd.name}` - {cmd.description}\n"
            help_text += "\n"

        help_text += "\nUse `/help <command>` for detailed help on a specific command."

        return CommandResult(
            success=True,
            command="help",
            message=help_text,
        )

    async def handle_think(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /think command - extended thinking."""
        query = parsed.raw_args

        return CommandResult(
            success=True,
            command="think",
            message=f"🧠 Using extended thinking for: {query[:100]}...",
            data={
                "mode": "extended_thinking",
                "query": query,
                "thinking_level": "high",
            },
            execute_rag=True,
            modified_query=query,
        )

    async def handle_ensemble(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /ensemble command - multi-model query."""
        query = parsed.raw_args

        return CommandResult(
            success=True,
            command="ensemble",
            message=f"🔄 Querying multiple models for: {query[:100]}...",
            data={
                "mode": "ensemble",
                "query": query,
                "strategy": "confidence",
            },
            execute_rag=True,
            modified_query=query,
        )

    async def handle_verify(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /verify command - answer verification."""
        statement = parsed.raw_args

        return CommandResult(
            success=True,
            command="verify",
            message=f"✅ Verifying statement: {statement[:100]}...",
            data={
                "mode": "verification",
                "statement": statement,
            },
            execute_rag=True,
            modified_query=f"Verify this statement against sources: {statement}",
        )

    async def handle_cot(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /cot command - chain of thought."""
        question = parsed.raw_args

        return CommandResult(
            success=True,
            command="cot",
            message=f"📝 Using step-by-step reasoning for: {question[:100]}...",
            data={
                "mode": "chain_of_thought",
                "question": question,
            },
            execute_rag=True,
            modified_query=question,
        )

    async def handle_organize(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /organize command - smart collections."""
        strategy = parsed.args[0] if parsed.args else "hybrid"

        return CommandResult(
            success=True,
            command="organize",
            message=f"📁 Organizing documents using {strategy} strategy...",
            data={
                "action": "organize_documents",
                "strategy": strategy,
            },
            follow_up_actions=[
                "View results in Collections panel",
                "Adjust strategy if needed",
            ],
        )

    async def handle_insights(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /insights command - insight feed."""
        limit = int(parsed.args[0]) if parsed.args and parsed.args[0].isdigit() else 10

        return CommandResult(
            success=True,
            command="insights",
            message=f"💡 Fetching your top {limit} insights...",
            data={
                "action": "get_insights",
                "limit": limit,
                "user_id": user_id,
            },
        )

    async def handle_duplicates(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /check-duplicates command."""
        content = parsed.raw_args or context.get("last_content", "")

        return CommandResult(
            success=True,
            command="check-duplicates",
            message="🔍 Checking for duplicate content...",
            data={
                "action": "check_duplicates",
                "content_preview": content[:100] if content else None,
            },
        )

    async def handle_highlights(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /highlights command."""
        target = parsed.raw_args or "current document"

        return CommandResult(
            success=True,
            command="highlights",
            message=f"✨ Generating smart highlights for: {target[:50]}...",
            data={
                "action": "generate_highlights",
                "target": target,
            },
        )

    async def handle_conflicts(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /check-conflicts command."""
        topic = parsed.raw_args if parsed.args else None

        return CommandResult(
            success=True,
            command="check-conflicts",
            message=f"⚠️ Analyzing for conflicts{' on topic: ' + topic if topic else ''}...",
            data={
                "action": "check_conflicts",
                "topic": topic,
            },
        )

    async def handle_digest(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /digest command."""
        period = "weekly" if parsed.args and "week" in parsed.args[0].lower() else "daily"

        return CommandResult(
            success=True,
            command="digest",
            message=f"📰 Generating your {period} digest...",
            data={
                "action": "generate_digest",
                "period": period,
                "user_id": user_id,
            },
        )

    async def handle_agent(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /agent command."""
        agent_name = parsed.args[0] if parsed.args else None
        task = " ".join(parsed.args[1:]) if len(parsed.args) > 1 else None

        if not agent_name:
            return CommandResult(
                success=False,
                command="agent",
                message="Please specify an agent name. Use /agents to see available agents.",
            )

        return CommandResult(
            success=True,
            command="agent",
            message=f"🤖 Running agent '{agent_name}'...",
            data={
                "action": "run_agent",
                "agent_name": agent_name,
                "task": task,
            },
        )

    async def handle_list_agents(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /agents command."""
        # TODO: Fetch from agent service
        agents = [
            {"name": "researcher", "description": "Research and gather information"},
            {"name": "writer", "description": "Write and edit content"},
            {"name": "analyst", "description": "Analyze data and documents"},
            {"name": "summarizer", "description": "Summarize long content"},
        ]

        message = "# Available Agents\n\n"
        for agent in agents:
            message += f"- **{agent['name']}** - {agent['description']}\n"
        message += "\nUse `/agent <name> <task>` to run an agent."

        return CommandResult(
            success=True,
            command="agents",
            message=message,
            data={"agents": agents},
        )

    async def handle_skill(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /skill command."""
        skill_name = parsed.args[0] if parsed.args else None
        args = " ".join(parsed.args[1:]) if len(parsed.args) > 1 else None

        if not skill_name:
            return CommandResult(
                success=False,
                command="skill",
                message="Please specify a skill name. Use /skills to see available skills.",
            )

        return CommandResult(
            success=True,
            command="skill",
            message=f"⚡ Executing skill '{skill_name}'...",
            data={
                "action": "run_skill",
                "skill_name": skill_name,
                "args": args,
            },
        )

    async def handle_list_skills(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /skills command."""
        # TODO: Fetch from skills service
        skills = [
            {"name": "summarize", "description": "Summarize documents"},
            {"name": "translate", "description": "Translate content"},
            {"name": "extract", "description": "Extract specific information"},
            {"name": "compare", "description": "Compare documents"},
        ]

        message = "# Available Skills\n\n"
        for skill in skills:
            message += f"- **{skill['name']}** - {skill['description']}\n"
        message += "\nUse `/skill <name> <args>` to execute a skill."

        return CommandResult(
            success=True,
            command="skills",
            message=message,
            data={"skills": skills},
        )

    async def handle_calc(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /calc command."""
        expression = parsed.raw_args

        return CommandResult(
            success=True,
            command="calc",
            message=f"🔢 Calculating: {expression}",
            data={
                "action": "calculate",
                "expression": expression,
            },
        )

    async def handle_convert(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /convert command."""
        return CommandResult(
            success=True,
            command="convert",
            message=f"📐 Converting: {parsed.raw_args}",
            data={
                "action": "convert_units",
                "input": parsed.raw_args,
            },
        )

    async def handle_date(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /date command."""
        if not parsed.args:
            from datetime import datetime
            now = datetime.now()
            return CommandResult(
                success=True,
                command="date",
                message=f"📅 Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            )

        return CommandResult(
            success=True,
            command="date",
            message=f"📅 Date calculation: {parsed.raw_args}",
            data={
                "action": "date_calculate",
                "input": parsed.raw_args,
            },
        )

    async def handle_settings(
        self,
        parsed: ParsedCommand,
        user_id: str,
        context: Dict[str, Any],
    ) -> CommandResult:
        """Handle /settings command."""
        if not parsed.args:
            return CommandResult(
                success=True,
                command="settings",
                message="⚙️ Opening settings panel...",
                data={"action": "open_settings"},
            )

        setting_name = parsed.args[0]
        setting_value = parsed.args[1] if len(parsed.args) > 1 else None

        return CommandResult(
            success=True,
            command="settings",
            message=f"⚙️ {'Setting' if setting_value else 'Getting'} {setting_name}",
            data={
                "action": "update_setting" if setting_value else "get_setting",
                "name": setting_name,
                "value": setting_value,
            },
        )


# Singleton instance
_chat_commands_service: Optional[ChatCommandsService] = None


def get_chat_commands_service() -> ChatCommandsService:
    """Get or create the chat commands service singleton."""
    global _chat_commands_service
    if _chat_commands_service is None:
        _chat_commands_service = ChatCommandsService()
    return _chat_commands_service
