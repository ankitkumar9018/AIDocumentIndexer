"""
AIDocumentIndexer - Menu Configuration API Routes
==================================================

Endpoints for menu configuration and access control:
- Get user's visible menu
- Update user preferences
- Admin: Configure section visibility
- Admin: Set role levels
- Admin: Create custom roles
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.menu_config import (
    get_menu_service,
    MenuMode,
    RoleLevel,
    MenuConfigService,
)

router = APIRouter(prefix="/menu", tags=["Menu Configuration"])


# =============================================================================
# Request/Response Models
# =============================================================================

class UserPreferencesRequest(BaseModel):
    """Request to update user preferences."""
    menu_mode: Optional[str] = None  # "simple" or "complete"
    collapsed_sections: Optional[List[str]] = None
    pinned_sections: Optional[List[str]] = None
    favorites: Optional[List[str]] = None


class SectionOverrideRequest(BaseModel):
    """Request to override section settings."""
    section_key: str
    is_enabled: Optional[bool] = None
    min_role_level: Optional[int] = Field(None, ge=1, le=7)


class CustomRoleRequest(BaseModel):
    """Request to create a custom role."""
    role_name: str
    level: int = Field(..., ge=1, le=7)
    sections: List[str]
    description: str = ""


class OrganizationSettingsRequest(BaseModel):
    """Request to update organization settings."""
    section_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    default_mode: Optional[str] = None


class MenuResponse(BaseModel):
    """Response containing menu configuration."""
    sections: List[Dict[str, Any]]
    mode: str
    role_level: int


# =============================================================================
# User Endpoints
# =============================================================================

@router.get("/")
async def get_user_menu(
    user_id: str = Query(..., description="User ID"),
    role_level: int = Query(1, ge=1, le=7, description="User's role level"),
    org_id: Optional[str] = Query(None, description="Organization ID"),
    mode: Optional[str] = Query(None, description="Override menu mode"),
    service: MenuConfigService = Depends(get_menu_service),
) -> MenuResponse:
    """
    Get the menu configuration for the current user.

    Returns only sections the user has access to based on:
    - Their role level
    - Menu mode (simple/complete)
    - Organization settings
    """
    menu_mode = None
    if mode:
        try:
            menu_mode = MenuMode(mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

    sections = service.get_menu_for_user(
        user_id=user_id,
        role_level=role_level,
        org_id=org_id,
        mode=menu_mode,
    )

    # Get user's actual mode
    prefs = service.get_user_preferences(user_id)
    effective_mode = mode or prefs.menu_mode.value

    return MenuResponse(
        sections=sections,
        mode=effective_mode,
        role_level=role_level,
    )


@router.get("/preferences")
async def get_user_preferences(
    user_id: str = Query(..., description="User ID"),
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Get user's menu preferences."""
    prefs = service.get_user_preferences(user_id)
    return prefs.to_dict()


@router.post("/preferences")
async def update_user_preferences(
    user_id: str = Query(..., description="User ID"),
    request: UserPreferencesRequest = ...,
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Update user's menu preferences."""
    menu_mode = None
    if request.menu_mode:
        try:
            menu_mode = MenuMode(request.menu_mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.menu_mode}")

    prefs = service.update_user_preferences(
        user_id=user_id,
        menu_mode=menu_mode,
        collapsed_sections=request.collapsed_sections,
        pinned_sections=request.pinned_sections,
        favorites=request.favorites,
    )

    return prefs.to_dict()


@router.post("/toggle-mode")
async def toggle_menu_mode(
    user_id: str = Query(..., description="User ID"),
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Toggle between simple and complete menu mode."""
    prefs = service.get_user_preferences(user_id)
    new_mode = MenuMode.COMPLETE if prefs.menu_mode == MenuMode.SIMPLE else MenuMode.SIMPLE

    updated = service.update_user_preferences(user_id=user_id, menu_mode=new_mode)
    return {"mode": updated.menu_mode.value, "message": f"Switched to {updated.menu_mode.value} mode"}


@router.get("/search")
async def search_menu_sections(
    query: str = Query(..., min_length=1, description="Search query"),
    service: MenuConfigService = Depends(get_menu_service),
) -> List[Dict[str, Any]]:
    """Search menu sections by label or key."""
    return service.search_sections(query)


# =============================================================================
# Admin Endpoints
# =============================================================================

@router.get("/admin/all-sections")
async def get_all_sections(
    service: MenuConfigService = Depends(get_menu_service),
) -> List[Dict[str, Any]]:
    """Get all menu sections (admin only)."""
    return service.get_all_sections()


@router.get("/admin/role-levels")
async def get_role_levels(
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Get all role levels with descriptions."""
    return {
        "levels": [
            {"level": level.value, "name": level.name, "description": desc}
            for level, desc in zip(RoleLevel, service.get_role_descriptions().values())
        ]
    }


@router.get("/admin/org-settings")
async def get_org_settings(
    org_id: str = Query(..., description="Organization ID"),
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Get organization's menu settings."""
    settings = service.get_org_settings(org_id)
    return settings.to_dict()


@router.post("/admin/org-settings")
async def update_org_settings(
    org_id: str = Query(..., description="Organization ID"),
    request: OrganizationSettingsRequest = ...,
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Update organization's menu settings."""
    default_mode = None
    if request.default_mode:
        try:
            default_mode = MenuMode(request.default_mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {request.default_mode}")

    settings = service.update_org_settings(
        org_id=org_id,
        section_overrides=request.section_overrides,
        default_mode=default_mode,
    )

    return settings.to_dict()


@router.post("/admin/toggle-section")
async def toggle_section(
    org_id: str = Query(..., description="Organization ID"),
    section_key: str = Query(..., description="Section key to toggle"),
    is_enabled: bool = Query(..., description="Enable or disable"),
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Toggle a section's visibility for an organization."""
    success = service.toggle_section(org_id, section_key, is_enabled)
    return {
        "success": success,
        "section_key": section_key,
        "is_enabled": is_enabled,
    }


@router.post("/admin/section-role-level")
async def set_section_role_level(
    org_id: str = Query(..., description="Organization ID"),
    section_key: str = Query(..., description="Section key"),
    min_role_level: int = Query(..., ge=1, le=7, description="Minimum role level"),
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Set minimum role level for a section."""
    success = service.set_section_role_level(org_id, section_key, min_role_level)
    return {
        "success": success,
        "section_key": section_key,
        "min_role_level": min_role_level,
    }


@router.post("/admin/custom-role")
async def create_custom_role(
    org_id: str = Query(..., description="Organization ID"),
    request: CustomRoleRequest = ...,
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Create a custom role for an organization."""
    role = service.create_custom_role(
        org_id=org_id,
        role_name=request.role_name,
        level=request.level,
        sections=request.sections,
        description=request.description,
    )
    return {"role_name": request.role_name, "role": role}


@router.get("/admin/custom-roles")
async def get_custom_roles(
    org_id: str = Query(..., description="Organization ID"),
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Get all custom roles for an organization."""
    settings = service.get_org_settings(org_id)
    return {"roles": settings.custom_roles}


@router.delete("/admin/custom-role")
async def delete_custom_role(
    org_id: str = Query(..., description="Organization ID"),
    role_name: str = Query(..., description="Role name to delete"),
    service: MenuConfigService = Depends(get_menu_service),
) -> Dict[str, Any]:
    """Delete a custom role."""
    settings = service.get_org_settings(org_id)

    if role_name not in settings.custom_roles:
        raise HTTPException(status_code=404, detail=f"Role not found: {role_name}")

    del settings.custom_roles[role_name]
    service._org_settings[org_id] = settings

    return {"success": True, "deleted": role_name}
