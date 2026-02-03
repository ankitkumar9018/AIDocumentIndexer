"""
AIDocumentIndexer - Menu Configuration Service
===============================================

Role-based, configurable menu system with:
- Simple/Complete mode toggle
- Admin-controlled section visibility
- Preset role levels with granular access
- Organization-level overrides
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import structlog

logger = structlog.get_logger(__name__)


class MenuMode(str, Enum):
    """Menu display modes."""
    SIMPLE = "simple"      # Essential features only
    COMPLETE = "complete"  # All features


class RoleLevel(int, Enum):
    """Preset role levels for access control."""
    INTERN = 1       # Chat, Upload only
    VIEWER = 2       # + Documents, Search
    EDITOR = 3       # + Create, Workflows
    ANALYST = 4      # + Reports, Analytics, KG
    MANAGER = 5      # + Connectors, Settings
    ADMIN = 6        # Full access + Admin panel
    SUPER_ADMIN = 7  # All + System settings


# Role level descriptions for UI
ROLE_DESCRIPTIONS = {
    RoleLevel.INTERN: "Basic access: Chat and Upload only",
    RoleLevel.VIEWER: "Read access: View documents and search",
    RoleLevel.EDITOR: "Edit access: Create content and use workflows",
    RoleLevel.ANALYST: "Analysis access: Reports, analytics, knowledge graph",
    RoleLevel.MANAGER: "Management access: Connectors and settings",
    RoleLevel.ADMIN: "Admin access: Full control including user management",
    RoleLevel.SUPER_ADMIN: "Super Admin: Complete system access",
}


@dataclass
class MenuSection:
    """Definition of a menu section."""
    key: str
    label: str
    icon: str
    path: str
    min_role_level: RoleLevel = RoleLevel.VIEWER
    is_simple_mode: bool = False  # Show in simple mode
    is_enabled: bool = True
    sort_order: int = 0
    parent_key: Optional[str] = None
    children: List["MenuSection"] = field(default_factory=list)
    badge: Optional[str] = None  # e.g., "New", "Beta"
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "icon": self.icon,
            "path": self.path,
            "min_role_level": self.min_role_level.value,
            "is_simple_mode": self.is_simple_mode,
            "is_enabled": self.is_enabled,
            "sort_order": self.sort_order,
            "parent_key": self.parent_key,
            "children": [c.to_dict() for c in self.children],
            "badge": self.badge,
            "description": self.description,
        }


@dataclass
class UserMenuPreferences:
    """User's menu preferences."""
    user_id: str
    menu_mode: MenuMode = MenuMode.SIMPLE
    collapsed_sections: List[str] = field(default_factory=list)
    pinned_sections: List[str] = field(default_factory=list)
    favorites: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "menu_mode": self.menu_mode.value,
            "collapsed_sections": self.collapsed_sections,
            "pinned_sections": self.pinned_sections,
            "favorites": self.favorites,
        }


@dataclass
class OrganizationMenuSettings:
    """Organization-level menu overrides."""
    org_id: str
    section_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # section_key -> {"is_enabled": bool, "min_role_level": int}
    custom_roles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # role_name -> {"level": int, "sections": [...], "description": str}
    default_mode: MenuMode = MenuMode.SIMPLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "org_id": self.org_id,
            "section_overrides": self.section_overrides,
            "custom_roles": self.custom_roles,
            "default_mode": self.default_mode.value,
        }


# =============================================================================
# Default Menu Configuration
# =============================================================================

DEFAULT_MENU_SECTIONS = [
    # ===== SIMPLE MODE SECTIONS (Essential) =====
    MenuSection(
        key="chat",
        label="Chat",
        icon="MessageSquare",
        path="/dashboard/chat",
        min_role_level=RoleLevel.INTERN,
        is_simple_mode=True,
        sort_order=1,
        description="AI-powered chat with your documents",
    ),
    MenuSection(
        key="upload",
        label="Upload",
        icon="Upload",
        path="/dashboard/upload",
        min_role_level=RoleLevel.INTERN,
        is_simple_mode=True,
        sort_order=2,
        description="Upload and process documents",
    ),
    MenuSection(
        key="documents",
        label="Documents",
        icon="FileText",
        path="/dashboard/documents",
        min_role_level=RoleLevel.VIEWER,
        is_simple_mode=True,
        sort_order=3,
        description="Browse and manage documents",
    ),
    MenuSection(
        key="create",
        label="Create",
        icon="PenTool",
        path="/dashboard/create",
        min_role_level=RoleLevel.EDITOR,
        is_simple_mode=True,
        sort_order=4,
        description="Generate new content",
    ),

    # ===== COMPLETE MODE SECTIONS =====

    # Intelligence Group
    MenuSection(
        key="intelligence",
        label="Intelligence",
        icon="Brain",
        path="/dashboard/intelligence",
        min_role_level=RoleLevel.ANALYST,
        is_simple_mode=False,
        sort_order=10,
        description="AI-powered insights and analysis",
        children=[
            MenuSection(
                key="smart-collections",
                label="Smart Collections",
                icon="FolderKanban",
                path="/dashboard/intelligence/collections",
                min_role_level=RoleLevel.ANALYST,
                parent_key="intelligence",
            ),
            MenuSection(
                key="insights",
                label="Insight Feed",
                icon="Lightbulb",
                path="/dashboard/intelligence/insights",
                min_role_level=RoleLevel.ANALYST,
                parent_key="intelligence",
            ),
            MenuSection(
                key="conflicts",
                label="Conflict Detector",
                icon="AlertTriangle",
                path="/dashboard/intelligence/conflicts",
                min_role_level=RoleLevel.ANALYST,
                parent_key="intelligence",
            ),
            MenuSection(
                key="document-dna",
                label="Document DNA",
                icon="Dna",
                path="/dashboard/intelligence/dna",
                min_role_level=RoleLevel.ANALYST,
                parent_key="intelligence",
            ),
        ],
    ),

    # Knowledge Graph
    MenuSection(
        key="knowledge-graph",
        label="Knowledge Graph",
        icon="Network",
        path="/dashboard/knowledge-graph",
        min_role_level=RoleLevel.ANALYST,
        is_simple_mode=False,
        sort_order=11,
        description="Explore entity relationships",
    ),

    # Workflows
    MenuSection(
        key="workflows",
        label="Workflows",
        icon="GitBranch",
        path="/dashboard/workflows",
        min_role_level=RoleLevel.EDITOR,
        is_simple_mode=False,
        sort_order=12,
        description="Automate document processing",
    ),

    # Reports
    MenuSection(
        key="reports",
        label="Reports",
        icon="BarChart3",
        path="/dashboard/reports",
        min_role_level=RoleLevel.ANALYST,
        is_simple_mode=False,
        sort_order=13,
        description="Generate analytics reports",
    ),

    # Research
    MenuSection(
        key="research",
        label="Research",
        icon="Search",
        path="/dashboard/research",
        min_role_level=RoleLevel.ANALYST,
        is_simple_mode=False,
        sort_order=14,
        description="Deep research mode",
    ),

    # Connectors
    MenuSection(
        key="connectors",
        label="Connectors",
        icon="Plug",
        path="/dashboard/connectors",
        min_role_level=RoleLevel.MANAGER,
        is_simple_mode=False,
        sort_order=20,
        description="Connect external data sources",
    ),

    # Tools Group
    MenuSection(
        key="tools",
        label="Tools",
        icon="Wrench",
        path="/dashboard/tools",
        min_role_level=RoleLevel.EDITOR,
        is_simple_mode=False,
        sort_order=21,
        description="Utility tools",
        children=[
            MenuSection(
                key="pdf-tools",
                label="PDF Tools",
                icon="FileType",
                path="/dashboard/tools/pdf",
                min_role_level=RoleLevel.EDITOR,
                parent_key="tools",
            ),
            MenuSection(
                key="scraper",
                label="Web Scraper",
                icon="Globe",
                path="/dashboard/scraper",
                min_role_level=RoleLevel.ANALYST,
                parent_key="tools",
            ),
            MenuSection(
                key="database",
                label="Database Query",
                icon="Database",
                path="/dashboard/database",
                min_role_level=RoleLevel.ANALYST,
                parent_key="tools",
            ),
        ],
    ),

    # Audio/Video
    MenuSection(
        key="audio",
        label="Audio/Video",
        icon="Mic",
        path="/dashboard/audio",
        min_role_level=RoleLevel.EDITOR,
        is_simple_mode=False,
        sort_order=22,
        description="Process audio and video content",
    ),

    # Skills
    MenuSection(
        key="skills",
        label="Skills",
        icon="Zap",
        path="/dashboard/skills",
        min_role_level=RoleLevel.EDITOR,
        is_simple_mode=False,
        sort_order=23,
        badge="Beta",
        description="Custom AI skills",
    ),

    # Prompts
    MenuSection(
        key="prompts",
        label="Prompts",
        icon="MessageCircle",
        path="/dashboard/prompts",
        min_role_level=RoleLevel.EDITOR,
        is_simple_mode=False,
        sort_order=24,
        description="Manage prompt templates",
    ),

    # ===== MANAGEMENT SECTIONS =====

    # Sync
    MenuSection(
        key="sync",
        label="Sync Status",
        icon="RefreshCw",
        path="/dashboard/sync",
        min_role_level=RoleLevel.MANAGER,
        is_simple_mode=False,
        sort_order=30,
        description="Monitor sync operations",
    ),

    # Watcher
    MenuSection(
        key="watcher",
        label="File Watcher",
        icon="Eye",
        path="/dashboard/watcher",
        min_role_level=RoleLevel.MANAGER,
        is_simple_mode=False,
        sort_order=31,
        description="Monitor file changes",
    ),

    # Costs
    MenuSection(
        key="costs",
        label="Cost Tracking",
        icon="DollarSign",
        path="/dashboard/costs",
        min_role_level=RoleLevel.MANAGER,
        is_simple_mode=False,
        sort_order=32,
        description="Track API and compute costs",
    ),

    # ===== ADMIN SECTIONS =====

    MenuSection(
        key="admin",
        label="Admin",
        icon="Shield",
        path="/dashboard/admin",
        min_role_level=RoleLevel.ADMIN,
        is_simple_mode=False,
        sort_order=50,
        description="Administration panel",
        children=[
            MenuSection(
                key="admin-settings",
                label="Settings",
                icon="Settings",
                path="/dashboard/admin/settings",
                min_role_level=RoleLevel.ADMIN,
                parent_key="admin",
            ),
            MenuSection(
                key="admin-users",
                label="Users",
                icon="Users",
                path="/dashboard/admin/users",
                min_role_level=RoleLevel.ADMIN,
                parent_key="admin",
            ),
            MenuSection(
                key="admin-organizations",
                label="Organizations",
                icon="Building",
                path="/dashboard/admin/organizations",
                min_role_level=RoleLevel.ADMIN,
                parent_key="admin",
            ),
            MenuSection(
                key="admin-agents",
                label="Agents",
                icon="Bot",
                path="/dashboard/admin/agents",
                min_role_level=RoleLevel.ADMIN,
                parent_key="admin",
            ),
            MenuSection(
                key="admin-audit",
                label="Audit Logs",
                icon="ScrollText",
                path="/dashboard/admin/audit-logs",
                min_role_level=RoleLevel.ADMIN,
                parent_key="admin",
            ),
            MenuSection(
                key="admin-menu",
                label="Menu Configuration",
                icon="Menu",
                path="/dashboard/admin/menu",
                min_role_level=RoleLevel.ADMIN,
                parent_key="admin",
            ),
        ],
    ),

    # API Access
    MenuSection(
        key="api-access",
        label="API Access",
        icon="Key",
        path="/dashboard/api-access",
        min_role_level=RoleLevel.MANAGER,
        is_simple_mode=False,
        sort_order=51,
        description="Manage API keys",
    ),

    # Gateway (Super Admin)
    MenuSection(
        key="gateway",
        label="API Gateway",
        icon="Server",
        path="/dashboard/gateway",
        min_role_level=RoleLevel.SUPER_ADMIN,
        is_simple_mode=False,
        sort_order=60,
        description="API gateway configuration",
    ),
]


class MenuConfigService:
    """
    Service for managing menu configuration and access control.
    """

    def __init__(self, db_service: Any = None):
        self.db_service = db_service
        self._default_sections = {s.key: s for s in self._flatten_sections(DEFAULT_MENU_SECTIONS)}
        self._user_preferences: Dict[str, UserMenuPreferences] = {}
        self._org_settings: Dict[str, OrganizationMenuSettings] = {}

    def _flatten_sections(self, sections: List[MenuSection]) -> List[MenuSection]:
        """Flatten nested sections."""
        result = []
        for section in sections:
            result.append(section)
            if section.children:
                result.extend(self._flatten_sections(section.children))
        return result

    def get_menu_for_user(
        self,
        user_id: str,
        role_level: int,
        org_id: Optional[str] = None,
        mode: Optional[MenuMode] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get the menu configuration for a specific user.

        Args:
            user_id: User's unique identifier
            role_level: User's role level (1-7)
            org_id: Optional organization ID for overrides
            mode: Override menu mode (simple/complete)

        Returns:
            List of visible menu sections
        """
        # Get user preferences
        prefs = self._user_preferences.get(user_id, UserMenuPreferences(user_id=user_id))
        effective_mode = mode or prefs.menu_mode

        # Get org settings if applicable
        org_settings = self._org_settings.get(org_id) if org_id else None

        # Filter and build menu
        visible_sections = []

        for section in DEFAULT_MENU_SECTIONS:
            visible_section = self._filter_section(
                section=section,
                role_level=role_level,
                mode=effective_mode,
                org_settings=org_settings,
                pinned=section.key in prefs.pinned_sections,
            )
            if visible_section:
                visible_sections.append(visible_section)

        # Sort by pinned status then sort_order
        visible_sections.sort(key=lambda s: (not s.get("pinned", False), s.get("sort_order", 0)))

        return visible_sections

    def _filter_section(
        self,
        section: MenuSection,
        role_level: int,
        mode: MenuMode,
        org_settings: Optional[OrganizationMenuSettings],
        pinned: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Filter a section based on role, mode, and org settings."""

        # Check org-level override first
        if org_settings and section.key in org_settings.section_overrides:
            override = org_settings.section_overrides[section.key]
            if not override.get("is_enabled", True):
                return None
            min_level = override.get("min_role_level", section.min_role_level.value)
        else:
            if not section.is_enabled:
                return None
            min_level = section.min_role_level.value

        # Check role level
        if role_level < min_level:
            return None

        # Check mode (simple mode shows only is_simple_mode=True sections)
        if mode == MenuMode.SIMPLE and not section.is_simple_mode:
            return None

        # Build section dict
        result = section.to_dict()
        result["pinned"] = pinned

        # Filter children recursively
        if section.children:
            visible_children = []
            for child in section.children:
                visible_child = self._filter_section(
                    section=child,
                    role_level=role_level,
                    mode=mode,
                    org_settings=org_settings,
                )
                if visible_child:
                    visible_children.append(visible_child)
            result["children"] = visible_children

            # If no visible children, hide parent group
            if not visible_children and section.parent_key is None:
                return None

        return result

    def get_all_sections(self) -> List[Dict[str, Any]]:
        """Get all menu sections (for admin configuration)."""
        return [s.to_dict() for s in DEFAULT_MENU_SECTIONS]

    def get_user_preferences(self, user_id: str) -> UserMenuPreferences:
        """Get user's menu preferences."""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = UserMenuPreferences(user_id=user_id)
        return self._user_preferences[user_id]

    def update_user_preferences(
        self,
        user_id: str,
        menu_mode: Optional[MenuMode] = None,
        collapsed_sections: Optional[List[str]] = None,
        pinned_sections: Optional[List[str]] = None,
        favorites: Optional[List[str]] = None,
    ) -> UserMenuPreferences:
        """Update user's menu preferences."""
        prefs = self.get_user_preferences(user_id)

        if menu_mode is not None:
            prefs.menu_mode = menu_mode
        if collapsed_sections is not None:
            prefs.collapsed_sections = collapsed_sections
        if pinned_sections is not None:
            prefs.pinned_sections = pinned_sections
        if favorites is not None:
            prefs.favorites = favorites

        self._user_preferences[user_id] = prefs
        logger.info(f"Updated menu preferences for user {user_id}")
        return prefs

    def get_org_settings(self, org_id: str) -> OrganizationMenuSettings:
        """Get organization's menu settings."""
        if org_id not in self._org_settings:
            self._org_settings[org_id] = OrganizationMenuSettings(org_id=org_id)
        return self._org_settings[org_id]

    def update_org_settings(
        self,
        org_id: str,
        section_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        custom_roles: Optional[Dict[str, Dict[str, Any]]] = None,
        default_mode: Optional[MenuMode] = None,
    ) -> OrganizationMenuSettings:
        """Update organization's menu settings (admin only)."""
        settings = self.get_org_settings(org_id)

        if section_overrides is not None:
            settings.section_overrides = section_overrides
        if custom_roles is not None:
            settings.custom_roles = custom_roles
        if default_mode is not None:
            settings.default_mode = default_mode

        self._org_settings[org_id] = settings
        logger.info(f"Updated menu settings for org {org_id}")
        return settings

    def toggle_section(
        self,
        org_id: str,
        section_key: str,
        is_enabled: bool,
    ) -> bool:
        """Toggle a section's visibility for an organization."""
        settings = self.get_org_settings(org_id)

        if section_key not in settings.section_overrides:
            settings.section_overrides[section_key] = {}

        settings.section_overrides[section_key]["is_enabled"] = is_enabled
        self._org_settings[org_id] = settings

        logger.info(f"Toggled section {section_key} to {is_enabled} for org {org_id}")
        return True

    def set_section_role_level(
        self,
        org_id: str,
        section_key: str,
        min_role_level: int,
    ) -> bool:
        """Set minimum role level for a section."""
        if min_role_level < 1 or min_role_level > 7:
            raise ValueError("Role level must be between 1 and 7")

        settings = self.get_org_settings(org_id)

        if section_key not in settings.section_overrides:
            settings.section_overrides[section_key] = {}

        settings.section_overrides[section_key]["min_role_level"] = min_role_level
        self._org_settings[org_id] = settings

        logger.info(f"Set section {section_key} min level to {min_role_level} for org {org_id}")
        return True

    def create_custom_role(
        self,
        org_id: str,
        role_name: str,
        level: int,
        sections: List[str],
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a custom role for an organization."""
        settings = self.get_org_settings(org_id)

        custom_role = {
            "level": level,
            "sections": sections,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
        }

        settings.custom_roles[role_name] = custom_role
        self._org_settings[org_id] = settings

        logger.info(f"Created custom role {role_name} for org {org_id}")
        return custom_role

    def get_role_descriptions(self) -> Dict[int, str]:
        """Get descriptions for all preset role levels."""
        return {level.value: desc for level, desc in ROLE_DESCRIPTIONS.items()}

    def search_sections(self, query: str) -> List[Dict[str, Any]]:
        """Search menu sections by label or key."""
        query = query.lower()
        results = []

        for section in self._flatten_sections(DEFAULT_MENU_SECTIONS):
            if query in section.label.lower() or query in section.key.lower():
                results.append(section.to_dict())

        return results


# Singleton instance
_menu_service: Optional[MenuConfigService] = None


def get_menu_service() -> MenuConfigService:
    """Get or create the menu service singleton."""
    global _menu_service
    if _menu_service is None:
        _menu_service = MenuConfigService()
    return _menu_service
