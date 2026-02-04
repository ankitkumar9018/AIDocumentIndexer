"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import {
  MenuSection,
  UserMenuPreferences,
  MenuMode,
  fetchUserMenu,
  fetchUserPreferences,
  updateUserPreferences,
  toggleMenuMode,
  searchMenuSections,
  SIMPLE_MODE_SECTIONS,
} from "@/lib/menu-config";

interface UseMenuOptions {
  userId: string;
  roleLevel: number;
  orgId?: string;
}

interface UseMenuReturn {
  // Menu data
  sections: MenuSection[];
  mode: MenuMode;
  preferences: UserMenuPreferences | null;

  // Loading states
  isLoading: boolean;
  error: Error | null;

  // Actions
  toggleMode: () => Promise<void>;
  pinSection: (key: string) => Promise<void>;
  unpinSection: (key: string) => Promise<void>;
  collapseSection: (key: string) => Promise<void>;
  expandSection: (key: string) => Promise<void>;
  addFavorite: (key: string) => Promise<void>;
  removeFavorite: (key: string) => Promise<void>;
  searchSections: (query: string) => Promise<MenuSection[]>;
  refresh: () => Promise<void>;

  // Computed
  pinnedSections: MenuSection[];
  favoriteSections: MenuSection[];
  collapsedKeys: Set<string>;
  isSimpleMode: boolean;
}

export function useMenu({ userId, roleLevel, orgId }: UseMenuOptions): UseMenuReturn {
  const [sections, setSections] = useState<MenuSection[]>([]);
  const [mode, setMode] = useState<MenuMode>("simple");
  const [preferences, setPreferences] = useState<UserMenuPreferences | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Fetch menu and preferences
  const fetchMenu = useCallback(async () => {
    if (!userId) return;

    setIsLoading(true);
    setError(null);

    try {
      const [menuData, prefsData] = await Promise.all([
        fetchUserMenu(userId, roleLevel, orgId),
        fetchUserPreferences(userId),
      ]);

      setSections(menuData.sections);
      setMode(menuData.mode as MenuMode);
      setPreferences(prefsData);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Failed to fetch menu"));
    } finally {
      setIsLoading(false);
    }
  }, [userId, roleLevel, orgId]);

  useEffect(() => {
    fetchMenu();
  }, [fetchMenu]);

  // Toggle between simple and complete mode
  const handleToggleMode = useCallback(async () => {
    try {
      const result = await toggleMenuMode(userId);
      setMode(result.mode);

      // Refresh menu with new mode
      const menuData = await fetchUserMenu(userId, roleLevel, orgId, result.mode);
      setSections(menuData.sections);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Failed to toggle mode"));
    }
  }, [userId, roleLevel, orgId]);

  // Pin/Unpin sections
  const pinSection = useCallback(
    async (key: string) => {
      if (!preferences) return;

      const newPinned = [...preferences.pinnedSections, key];
      const updated = await updateUserPreferences(userId, { pinnedSections: newPinned });
      setPreferences(updated);

      // Refresh menu to reflect pinned status
      const menuData = await fetchUserMenu(userId, roleLevel, orgId, mode);
      setSections(menuData.sections);
    },
    [userId, roleLevel, orgId, mode, preferences]
  );

  const unpinSection = useCallback(
    async (key: string) => {
      if (!preferences) return;

      const newPinned = preferences.pinnedSections.filter((k) => k !== key);
      const updated = await updateUserPreferences(userId, { pinnedSections: newPinned });
      setPreferences(updated);

      const menuData = await fetchUserMenu(userId, roleLevel, orgId, mode);
      setSections(menuData.sections);
    },
    [userId, roleLevel, orgId, mode, preferences]
  );

  // Collapse/Expand sections
  const collapseSection = useCallback(
    async (key: string) => {
      if (!preferences) return;

      const newCollapsed = [...preferences.collapsedSections, key];
      const updated = await updateUserPreferences(userId, { collapsedSections: newCollapsed });
      setPreferences(updated);
    },
    [userId, preferences]
  );

  const expandSection = useCallback(
    async (key: string) => {
      if (!preferences) return;

      const newCollapsed = preferences.collapsedSections.filter((k) => k !== key);
      const updated = await updateUserPreferences(userId, { collapsedSections: newCollapsed });
      setPreferences(updated);
    },
    [userId, preferences]
  );

  // Favorites
  const addFavorite = useCallback(
    async (key: string) => {
      if (!preferences) return;

      const newFavorites = [...preferences.favorites, key];
      const updated = await updateUserPreferences(userId, { favorites: newFavorites });
      setPreferences(updated);
    },
    [userId, preferences]
  );

  const removeFavorite = useCallback(
    async (key: string) => {
      if (!preferences) return;

      const newFavorites = preferences.favorites.filter((k) => k !== key);
      const updated = await updateUserPreferences(userId, { favorites: newFavorites });
      setPreferences(updated);
    },
    [userId, preferences]
  );

  // Search
  const handleSearchSections = useCallback(async (query: string) => {
    return searchMenuSections(query);
  }, []);

  // Refresh
  const refresh = useCallback(async () => {
    await fetchMenu();
  }, [fetchMenu]);

  // Computed values
  const pinnedSections = useMemo(() => {
    if (!preferences) return [];
    return sections.filter((s) => preferences.pinnedSections.includes(s.key));
  }, [sections, preferences]);

  const favoriteSections = useMemo(() => {
    if (!preferences) return [];
    return sections.filter((s) => preferences.favorites.includes(s.key));
  }, [sections, preferences]);

  const collapsedKeys = useMemo(() => {
    return new Set(preferences?.collapsedSections || []);
  }, [preferences]);

  const isSimpleMode = mode === "simple";

  return {
    sections,
    mode,
    preferences,
    isLoading,
    error,
    toggleMode: handleToggleMode,
    pinSection,
    unpinSection,
    collapseSection,
    expandSection,
    addFavorite,
    removeFavorite,
    searchSections: handleSearchSections,
    refresh,
    pinnedSections,
    favoriteSections,
    collapsedKeys,
    isSimpleMode,
  };
}

// Context for sharing menu state across components
import { createContext, useContext, ReactNode } from "react";

interface MenuContextValue extends UseMenuReturn {}

const MenuContext = createContext<MenuContextValue | null>(null);

interface MenuProviderProps {
  children: ReactNode;
  userId: string;
  roleLevel: number;
  orgId?: string;
}

export function MenuProvider({ children, userId, roleLevel, orgId }: MenuProviderProps) {
  const menuState = useMenu({ userId, roleLevel, orgId });

  return <MenuContext.Provider value={menuState}>{children}</MenuContext.Provider>;
}

export function useMenuContext(): MenuContextValue {
  const context = useContext(MenuContext);
  if (!context) {
    throw new Error("useMenuContext must be used within a MenuProvider");
  }
  return context;
}
