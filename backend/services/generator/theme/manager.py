"""
Theme Manager

Provides CRUD operations for managing document themes.
Handles loading, saving, and organizing theme configurations.
"""

import os
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

import structlog

from .models import ThemeProfile, PPTXTheme, DOCXTheme, XLSXTheme, PDFTheme

logger = structlog.get_logger(__name__)

# Default themes directory
DEFAULT_THEMES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "..", "..", "..",  # Navigate to project root
    "data", "themes"
)


class ThemeManager:
    """Manages theme storage, retrieval, and organization.

    Provides CRUD operations for themes and handles serialization
    to/from JSON files.
    """

    def __init__(self, themes_dir: Optional[str] = None):
        """Initialize the theme manager.

        Args:
            themes_dir: Directory for theme storage (default: data/themes)
        """
        self.themes_dir = themes_dir or DEFAULT_THEMES_DIR
        self._ensure_themes_dir()
        self._cache: Dict[str, ThemeProfile] = {}

    def _ensure_themes_dir(self):
        """Ensure the themes directory exists."""
        os.makedirs(self.themes_dir, exist_ok=True)
        # Create subdirectories for different file types
        for subdir in ["pptx", "docx", "xlsx", "pdf"]:
            os.makedirs(os.path.join(self.themes_dir, subdir), exist_ok=True)

    def _get_theme_path(self, theme_id: str, file_type: str = "pptx") -> str:
        """Get the file path for a theme.

        Args:
            theme_id: Unique theme identifier
            file_type: Document type (pptx, docx, xlsx, pdf)

        Returns:
            Full path to theme JSON file
        """
        return os.path.join(self.themes_dir, file_type, f"{theme_id}.json")

    def list_themes(self, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available themes.

        Args:
            file_type: Optional filter by document type

        Returns:
            List of theme metadata dictionaries
        """
        themes = []

        if file_type:
            dirs_to_scan = [os.path.join(self.themes_dir, file_type)]
        else:
            dirs_to_scan = [
                os.path.join(self.themes_dir, ft)
                for ft in ["pptx", "docx", "xlsx", "pdf"]
            ]

        for dir_path in dirs_to_scan:
            if not os.path.exists(dir_path):
                continue

            for filename in os.listdir(dir_path):
                if not filename.endswith(".json"):
                    continue

                theme_id = filename[:-5]  # Remove .json
                theme_path = os.path.join(dir_path, filename)

                try:
                    with open(theme_path, 'r') as f:
                        data = json.load(f)

                    themes.append({
                        "id": theme_id,
                        "name": data.get("name", theme_id),
                        "description": data.get("description", ""),
                        "file_type": os.path.basename(dir_path),
                        "path": theme_path,
                    })
                except Exception as e:
                    logger.warning(f"Could not read theme {theme_path}: {e}")

        return themes

    def get_theme(self, theme_id: str, file_type: str = "pptx") -> Optional[ThemeProfile]:
        """Get a theme by ID.

        Args:
            theme_id: Unique theme identifier
            file_type: Document type

        Returns:
            ThemeProfile object or None if not found
        """
        # Check cache first
        cache_key = f"{file_type}:{theme_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        theme_path = self._get_theme_path(theme_id, file_type)

        if not os.path.exists(theme_path):
            return None

        try:
            with open(theme_path, 'r') as f:
                data = json.load(f)

            # Create ThemeProfile from data
            theme = ThemeProfile(
                name=data.get("name", theme_id),
                description=data.get("description", ""),
                primary=data.get("primary", "#333333"),
                secondary=data.get("secondary", "#666666"),
                accent=data.get("accent", "#0066CC"),
                background=data.get("background", "#FFFFFF"),
                text=data.get("text", "#333333"),
                font_heading=data.get("font_heading", "Arial"),
                font_body=data.get("font_body", "Arial"),
            )

            # Cache the theme
            self._cache[cache_key] = theme

            return theme

        except Exception as e:
            logger.error(f"Could not load theme {theme_id}: {e}")
            return None

    def save_theme(self, theme_id: str, theme: ThemeProfile,
                   file_type: str = "pptx") -> bool:
        """Save a theme to storage.

        Args:
            theme_id: Unique identifier for the theme
            theme: ThemeProfile to save
            file_type: Document type

        Returns:
            True if saved successfully
        """
        theme_path = self._get_theme_path(theme_id, file_type)

        try:
            data = {
                "name": theme.name,
                "description": theme.description,
                "primary": theme.primary,
                "secondary": theme.secondary,
                "accent": theme.accent,
                "background": theme.background,
                "text": theme.text,
                "font_heading": theme.font_heading,
                "font_body": theme.font_body,
            }

            with open(theme_path, 'w') as f:
                json.dump(data, f, indent=2)

            # Update cache
            cache_key = f"{file_type}:{theme_id}"
            self._cache[cache_key] = theme

            logger.info(f"Saved theme {theme_id} to {theme_path}")
            return True

        except Exception as e:
            logger.error(f"Could not save theme {theme_id}: {e}")
            return False

    def delete_theme(self, theme_id: str, file_type: str = "pptx") -> bool:
        """Delete a theme.

        Args:
            theme_id: Unique identifier of theme to delete
            file_type: Document type

        Returns:
            True if deleted successfully
        """
        theme_path = self._get_theme_path(theme_id, file_type)

        if not os.path.exists(theme_path):
            return False

        try:
            os.remove(theme_path)

            # Remove from cache
            cache_key = f"{file_type}:{theme_id}"
            self._cache.pop(cache_key, None)

            logger.info(f"Deleted theme {theme_id}")
            return True

        except Exception as e:
            logger.error(f"Could not delete theme {theme_id}: {e}")
            return False

    def import_from_template(self, template_path: str,
                             theme_id: Optional[str] = None) -> Optional[str]:
        """Import a theme from a document template.

        Args:
            template_path: Path to template file
            theme_id: Optional ID for the theme (auto-generated if not provided)

        Returns:
            Theme ID if imported successfully, None otherwise
        """
        from .extractor import ThemeExtractorFactory

        # Detect file type
        ext = os.path.splitext(template_path)[1].lower()
        file_type_map = {
            ".pptx": "pptx",
            ".docx": "docx",
            ".xlsx": "xlsx",
            ".pdf": "pdf",
            ".html": "pdf",
        }
        file_type = file_type_map.get(ext)

        if not file_type:
            logger.error(f"Unsupported template type: {ext}")
            return None

        # Extract theme
        extractor = ThemeExtractorFactory.get_extractor(template_path)
        if not extractor:
            logger.error(f"No extractor available for {template_path}")
            return None

        try:
            theme_data = extractor.extract(template_path)

            # Convert to ThemeProfile
            theme = ThemeProfile(
                name=theme_id or os.path.basename(template_path),
                description=f"Imported from {os.path.basename(template_path)}",
                primary=theme_data.get("primary", "#333333"),
                secondary=theme_data.get("secondary", "#666666"),
                accent=theme_data.get("accent1", "#0066CC"),
                background=theme_data.get("background", "#FFFFFF"),
                text=theme_data.get("text", "#333333"),
                font_heading=theme_data.get("font_heading", "Arial"),
                font_body=theme_data.get("font_body", "Arial"),
            )

            # Generate ID if not provided
            if not theme_id:
                base_name = os.path.splitext(os.path.basename(template_path))[0]
                theme_id = base_name.lower().replace(" ", "_")

            # Save the theme
            if self.save_theme(theme_id, theme, file_type):
                return theme_id
            return None

        except Exception as e:
            logger.error(f"Could not import theme from {template_path}: {e}")
            return None

    def get_default_theme(self, file_type: str = "pptx") -> ThemeProfile:
        """Get the default theme for a file type.

        Args:
            file_type: Document type

        Returns:
            Default ThemeProfile
        """
        # Try to get the "default" theme
        default_theme = self.get_theme("default", file_type)
        if default_theme:
            return default_theme

        # Return a built-in default
        return ThemeProfile(
            name="Default",
            description="Default professional theme",
            primary="#2563EB",
            secondary="#64748B",
            accent="#10B981",
            background="#FFFFFF",
            text="#1E293B",
            font_heading="Arial",
            font_body="Arial",
        )

    def clear_cache(self):
        """Clear the theme cache."""
        self._cache.clear()


# Singleton instance
_manager_instance = None


def get_theme_manager() -> ThemeManager:
    """Get the singleton ThemeManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ThemeManager()
    return _manager_instance
