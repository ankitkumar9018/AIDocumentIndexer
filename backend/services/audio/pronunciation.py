"""
Pronunciation Dictionary Service
================================

Custom pronunciations for TTS to handle:
- Technical terms
- Acronyms
- Brand names
- Foreign words
- User-defined pronunciations

Improves TTS quality by replacing terms with
pronunciation-friendly versions before synthesis.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern
from pathlib import Path
import json
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PronunciationRule:
    """A pronunciation replacement rule."""
    term: str  # Original term to match
    pronunciation: str  # How it should be pronounced
    case_sensitive: bool = False
    word_boundary: bool = True  # Only match whole words
    enabled: bool = True
    category: str = "general"  # For organization
    notes: str = ""  # Usage notes

    def to_dict(self) -> Dict:
        return {
            "term": self.term,
            "pronunciation": self.pronunciation,
            "case_sensitive": self.case_sensitive,
            "word_boundary": self.word_boundary,
            "enabled": self.enabled,
            "category": self.category,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PronunciationRule":
        return cls(**data)


class PronunciationDictionary:
    """
    Custom pronunciations for TTS.

    Maintains a dictionary of terms and their pronunciations,
    applying them to text before TTS synthesis.
    """

    # Built-in pronunciations for common technical terms
    DEFAULT_PRONUNCIATIONS: Dict[str, str] = {
        # Acronyms - spell out
        "API": "A P I",
        "APIs": "A P Is",
        "CLI": "C L I",
        "GUI": "G U I",
        "URL": "U R L",
        "URLs": "U R Ls",
        "HTML": "H T M L",
        "CSS": "C S S",
        "SDK": "S D K",
        "IDE": "I D E",
        "DNS": "D N S",
        "SSL": "S S L",
        "TLS": "T L S",
        "SSH": "S S H",
        "FTP": "F T P",
        "HTTP": "H T T P",
        "HTTPS": "H T T P S",
        "REST": "rest",
        "CRUD": "crud",
        "ORM": "O R M",
        "AI": "A I",
        "ML": "M L",
        "NLP": "N L P",
        "LLM": "L L M",
        "LLMs": "L L Ms",
        "GPT": "G P T",
        "GPU": "G P U",
        "CPU": "C P U",
        "RAM": "ram",
        "ROM": "rom",
        "SSD": "S S D",
        "HDD": "H D D",
        "AWS": "A W S",
        "GCP": "G C P",
        "MVP": "M V P",
        "POC": "P O C",
        "QA": "Q A",
        "UAT": "U A T",
        "UAV": "U A V",
        "VPN": "V P N",
        "VR": "V R",
        "AR": "A R",
        "IoT": "I O T",
        "SaaS": "sass",
        "PaaS": "pass",
        "IaaS": "I ass",
        "PDF": "P D F",
        "XML": "X M L",
        "YAML": "yaml",
        "JSON": "jason",
        "CSV": "C S V",
        "SQL": "sequel",
        "NoSQL": "no sequel",
        "SQLite": "sequel lite",
        "MySQL": "my sequel",
        "PostgreSQL": "post gres sequel",

        # Technical terms - phonetic spellings
        "nginx": "engine x",
        "OAuth": "oh auth",
        "OAuth2": "oh auth 2",
        "kubectl": "kube control",
        "async": "a sink",
        "asyncio": "a sink I O",
        "NumPy": "num pie",
        "PyPI": "pie P I",
        "PyTorch": "pie torch",
        "TensorFlow": "tensor flow",
        "Kubernetes": "koo ber net ees",
        "k8s": "kates",
        "GitHub": "git hub",
        "GitLab": "git lab",
        "dev": "dev",
        "devs": "devs",
        "DevOps": "dev ops",
        "frontend": "front end",
        "backend": "back end",
        "fullstack": "full stack",
        "localhost": "local host",
        "sudo": "sue do",
        "chmod": "change mod",
        "chown": "change own",
        "grep": "grep",
        "sed": "said",
        "awk": "awk",
        "regex": "reg ex",
        "RegEx": "reg ex",
        "regexp": "reg exp",
        "UUID": "you you I D",
        "GUID": "goo id",
        "i18n": "internationalization",
        "l10n": "localization",
        "CORS": "cors",
        "CSRF": "C S R F",
        "XSS": "X S S",
        "RBAC": "R back",
        "JWT": "J W T",
        "JWTs": "J W Ts",
        "TOML": "tom L",
        "env": "env",
        ".env": "dot env",
        "dotenv": "dot env",
        "README": "read me",
        "webpack": "web pack",
        "npm": "N P M",
        "npx": "N P X",
        "pnpm": "P N P M",
        "yarn": "yarn",
        "ESLint": "E S lint",
        "TypeScript": "type script",
        "JavaScript": "java script",
        "Node.js": "node J S",
        "Next.js": "next J S",
        "Vue.js": "view J S",
        "React": "react",
        "Angular": "angular",
        "Svelte": "svelt",
        "FastAPI": "fast A P I",
        "GraphQL": "graph Q L",
        "gRPC": "G R P C",
        "WebSocket": "web socket",
        "WebSockets": "web sockets",

        # Common abbreviations
        "vs": "versus",
        "ie": "that is",
        "eg": "for example",
        "etc": "et cetera",
        "w/": "with",
        "w/o": "without",

        # Symbols in text
        "->": "arrow",
        "=>": "fat arrow",
        "!=": "not equal",
        "==": "equals",
        "===": "strict equals",
        ">=": "greater than or equal",
        "<=": "less than or equal",
        "&&": "and",
        "||": "or",
    }

    def __init__(
        self,
        custom_dict: Optional[Dict[str, str]] = None,
        rules_path: Optional[Path] = None,
    ):
        """
        Initialize pronunciation dictionary.

        Args:
            custom_dict: Additional custom pronunciations
            rules_path: Path to JSON file with custom rules
        """
        self.rules: List[PronunciationRule] = []

        # Load default pronunciations
        for term, pronunciation in self.DEFAULT_PRONUNCIATIONS.items():
            self.rules.append(PronunciationRule(
                term=term,
                pronunciation=pronunciation,
                category="technical",
            ))

        # Load custom dict
        if custom_dict:
            for term, pronunciation in custom_dict.items():
                self.rules.append(PronunciationRule(
                    term=term,
                    pronunciation=pronunciation,
                    category="custom",
                ))

        # Load from file
        if rules_path and rules_path.exists():
            self._load_rules_from_file(rules_path)

        # Compile patterns for efficiency
        self._compile_patterns()

        logger.info(
            "Pronunciation dictionary initialized",
            total_rules=len(self.rules),
        )

    def _compile_patterns(self):
        """Pre-compile regex patterns for all rules."""
        self._compiled_rules: List[tuple] = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Build pattern
            pattern_str = re.escape(rule.term)

            if rule.word_boundary:
                pattern_str = r"\b" + pattern_str + r"\b"

            flags = 0 if rule.case_sensitive else re.IGNORECASE
            pattern = re.compile(pattern_str, flags)

            self._compiled_rules.append((pattern, rule.pronunciation, rule))

        # Sort by term length (longest first) to avoid partial replacements
        self._compiled_rules.sort(key=lambda x: len(x[2].term), reverse=True)

    def _load_rules_from_file(self, path: Path):
        """Load pronunciation rules from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    self.rules.append(PronunciationRule.from_dict(item))
            elif isinstance(data, dict):
                # Simple term -> pronunciation mapping
                for term, pronunciation in data.items():
                    self.rules.append(PronunciationRule(
                        term=term,
                        pronunciation=pronunciation,
                        category="file",
                    ))

            logger.info("Loaded pronunciation rules from file", path=str(path))

        except Exception as e:
            logger.warning(
                "Failed to load pronunciation rules",
                path=str(path),
                error=str(e),
            )

    def apply_pronunciations(
        self,
        text: str,
        categories: Optional[List[str]] = None,
    ) -> str:
        """
        Replace terms with pronunciation-friendly versions.

        Args:
            text: Text to process
            categories: Optional list of categories to apply (None = all)

        Returns:
            Text with pronunciations applied
        """
        if not text:
            return text

        result = text

        for pattern, pronunciation, rule in self._compiled_rules:
            if categories and rule.category not in categories:
                continue

            result = pattern.sub(pronunciation, result)

        return result

    def add_rule(
        self,
        term: str,
        pronunciation: str,
        category: str = "custom",
        case_sensitive: bool = False,
        word_boundary: bool = True,
    ) -> PronunciationRule:
        """
        Add a new pronunciation rule.

        Args:
            term: Term to match
            pronunciation: How to pronounce it
            category: Category for organization
            case_sensitive: Whether matching is case-sensitive
            word_boundary: Whether to only match whole words

        Returns:
            The created rule
        """
        rule = PronunciationRule(
            term=term,
            pronunciation=pronunciation,
            category=category,
            case_sensitive=case_sensitive,
            word_boundary=word_boundary,
        )

        self.rules.append(rule)
        self._compile_patterns()

        logger.debug("Added pronunciation rule", term=term, pronunciation=pronunciation)

        return rule

    def remove_rule(self, term: str) -> bool:
        """
        Remove a pronunciation rule.

        Args:
            term: Term to remove

        Returns:
            True if rule was removed, False if not found
        """
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.term != term]

        if len(self.rules) < initial_count:
            self._compile_patterns()
            return True

        return False

    def get_rules(self, category: Optional[str] = None) -> List[PronunciationRule]:
        """
        Get all rules, optionally filtered by category.

        Args:
            category: Optional category to filter by

        Returns:
            List of matching rules
        """
        if category:
            return [r for r in self.rules if r.category == category]
        return self.rules.copy()

    def save_custom_rules(self, path: Path, category: str = "custom"):
        """
        Save custom rules to JSON file.

        Args:
            path: Path to save rules
            category: Category to save (default: "custom")
        """
        rules_to_save = [
            r.to_dict() for r in self.rules
            if r.category == category
        ]

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(rules_to_save, f, indent=2)

        logger.info("Saved pronunciation rules", path=str(path), count=len(rules_to_save))

    def preview_changes(self, text: str) -> List[Dict[str, str]]:
        """
        Preview what pronunciations would be applied to text.

        Args:
            text: Text to analyze

        Returns:
            List of changes that would be made
        """
        changes = []

        for pattern, pronunciation, rule in self._compiled_rules:
            matches = pattern.findall(text)
            for match in matches:
                changes.append({
                    "original": match,
                    "pronunciation": pronunciation,
                    "category": rule.category,
                })

        return changes


# Singleton instance
_pronunciation_dict: Optional[PronunciationDictionary] = None


def get_pronunciation_dictionary() -> PronunciationDictionary:
    """Get or create the singleton pronunciation dictionary."""
    global _pronunciation_dict
    if _pronunciation_dict is None:
        _pronunciation_dict = PronunciationDictionary()
    return _pronunciation_dict


def apply_pronunciations(text: str, custom_dict: Optional[Dict[str, str]] = None) -> str:
    """
    Convenience function to apply pronunciations to text.

    Args:
        text: Text to process
        custom_dict: Optional additional custom pronunciations

    Returns:
        Text with pronunciations applied
    """
    dictionary = get_pronunciation_dictionary()

    # Add any custom pronunciations temporarily
    if custom_dict:
        for term, pronunciation in custom_dict.items():
            dictionary.add_rule(term, pronunciation, category="temp")

    result = dictionary.apply_pronunciations(text)

    # Remove temporary rules
    if custom_dict:
        for term in custom_dict.keys():
            dictionary.remove_rule(term)

    return result
