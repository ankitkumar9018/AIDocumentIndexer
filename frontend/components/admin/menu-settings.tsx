"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogTrigger } from "@/components/ui/dialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Menu,
  Settings,
  Shield,
  Users,
  Plus,
  Trash2,
  Save,
  RefreshCw,
  Info,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import {
  MenuSection,
  OrganizationMenuSettings,
  CustomRole,
  ROLE_LEVELS,
  getRoleName,
  getRoleDescription,
  getIcon,
  fetchAllSections,
  fetchOrgSettings,
  toggleSection,
  setSectionRoleLevel,
  createCustomRole,
  deleteCustomRole,
  updateOrgSettings,
} from "@/lib/menu-config";

interface MenuSettingsProps {
  orgId: string;
}

export function MenuSettings({ orgId }: MenuSettingsProps) {
  const [sections, setSections] = useState<MenuSection[]>([]);
  const [settings, setSettings] = useState<OrganizationMenuSettings | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Custom role form
  const [showRoleDialog, setShowRoleDialog] = useState(false);
  const [newRole, setNewRole] = useState({
    roleName: "",
    level: 3,
    sections: [] as string[],
    description: "",
  });

  // Load data
  const loadData = async () => {
    setIsLoading(true);
    try {
      const [sectionsData, settingsData] = await Promise.all([
        fetchAllSections(),
        fetchOrgSettings(orgId),
      ]);
      setSections(sectionsData);
      setSettings(settingsData);
    } catch (error) {
      console.error("Failed to load menu settings:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [orgId]);

  // Handle section toggle
  const handleToggleSection = async (sectionKey: string, isEnabled: boolean) => {
    try {
      await toggleSection(orgId, sectionKey, isEnabled);

      // Update local state
      setSettings((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          sectionOverrides: {
            ...prev.sectionOverrides,
            [sectionKey]: {
              ...prev.sectionOverrides[sectionKey],
              isEnabled,
            },
          },
        };
      });

      showSaveMessage("success", `Section "${sectionKey}" ${isEnabled ? "enabled" : "disabled"}`);
    } catch (error) {
      showSaveMessage("error", "Failed to update section");
    }
  };

  // Handle role level change
  const handleRoleLevelChange = async (sectionKey: string, level: number) => {
    try {
      await setSectionRoleLevel(orgId, sectionKey, level);

      setSettings((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          sectionOverrides: {
            ...prev.sectionOverrides,
            [sectionKey]: {
              ...prev.sectionOverrides[sectionKey],
              minRoleLevel: level,
            },
          },
        };
      });

      showSaveMessage("success", `Role level updated for "${sectionKey}"`);
    } catch (error) {
      showSaveMessage("error", "Failed to update role level");
    }
  };

  // Handle create custom role
  const handleCreateRole = async () => {
    if (!newRole.roleName || newRole.sections.length === 0) return;

    setIsSaving(true);
    try {
      await createCustomRole(orgId, newRole);
      await loadData();
      setShowRoleDialog(false);
      setNewRole({ roleName: "", level: 3, sections: [], description: "" });
      showSaveMessage("success", `Role "${newRole.roleName}" created`);
    } catch (error) {
      showSaveMessage("error", "Failed to create role");
    } finally {
      setIsSaving(false);
    }
  };

  // Handle delete custom role
  const handleDeleteRole = async (roleName: string) => {
    try {
      await deleteCustomRole(orgId, roleName);
      await loadData();
      showSaveMessage("success", `Role "${roleName}" deleted`);
    } catch (error) {
      showSaveMessage("error", "Failed to delete role");
    }
  };

  // Handle default mode change
  const handleDefaultModeChange = async (mode: "simple" | "complete") => {
    try {
      await updateOrgSettings(orgId, { defaultMode: mode });
      setSettings((prev) => (prev ? { ...prev, defaultMode: mode } : prev));
      showSaveMessage("success", `Default mode set to "${mode}"`);
    } catch (error) {
      showSaveMessage("error", "Failed to update default mode");
    }
  };

  const showSaveMessage = (type: "success" | "error", text: string) => {
    setSaveMessage({ type, text });
    setTimeout(() => setSaveMessage(null), 3000);
  };

  // Get effective setting for a section
  const getSectionSetting = (sectionKey: string) => {
    const section = sections.find((s) => s.key === sectionKey);
    const override = settings?.sectionOverrides[sectionKey];

    return {
      isEnabled: override?.isEnabled ?? section?.isEnabled ?? true,
      minRoleLevel: override?.minRoleLevel ?? section?.minRoleLevel ?? 1,
    };
  };

  // Flatten sections for table
  const flattenSections = (items: MenuSection[], depth = 0): (MenuSection & { depth: number })[] => {
    return items.flatMap((item) => [
      { ...item, depth },
      ...(item.children ? flattenSections(item.children, depth + 1) : []),
    ]);
  };

  const flatSections = flattenSections(sections);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <RefreshCw className="h-8 w-8 mx-auto animate-spin text-muted-foreground" />
          <p className="mt-4 text-muted-foreground">Loading menu settings...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
            <Menu className="h-6 w-6" />
            Menu Configuration
          </h2>
          <p className="text-muted-foreground">
            Configure menu sections, role access, and create custom roles
          </p>
        </div>

        <Button onClick={loadData} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Save Message */}
      {saveMessage && (
        <Alert variant={saveMessage.type === "success" ? "default" : "destructive"}>
          {saveMessage.type === "success" ? (
            <CheckCircle className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          <AlertDescription>{saveMessage.text}</AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="sections">
        <TabsList>
          <TabsTrigger value="sections">
            <Menu className="h-4 w-4 mr-2" />
            Sections
          </TabsTrigger>
          <TabsTrigger value="roles">
            <Shield className="h-4 w-4 mr-2" />
            Role Levels
          </TabsTrigger>
          <TabsTrigger value="custom">
            <Users className="h-4 w-4 mr-2" />
            Custom Roles
          </TabsTrigger>
          <TabsTrigger value="defaults">
            <Settings className="h-4 w-4 mr-2" />
            Defaults
          </TabsTrigger>
        </TabsList>

        {/* Sections Tab */}
        <TabsContent value="sections">
          <Card>
            <CardHeader>
              <CardTitle>Menu Sections</CardTitle>
              <CardDescription>
                Enable or disable menu sections and set minimum role levels
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[500px]">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Section</TableHead>
                      <TableHead>Simple Mode</TableHead>
                      <TableHead>Enabled</TableHead>
                      <TableHead>Min Role</TableHead>
                      <TableHead>Path</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {flatSections.map((section) => {
                      const Icon = getIcon(section.icon);
                      const setting = getSectionSetting(section.key);

                      return (
                        <TableRow key={section.key}>
                          <TableCell>
                            <div
                              className="flex items-center gap-2"
                              style={{ paddingLeft: section.depth * 20 }}
                            >
                              <Icon className="h-4 w-4 text-muted-foreground" />
                              <span className="font-medium">{section.label}</span>
                              {section.badge && (
                                <Badge variant="secondary" className="text-xs">
                                  {section.badge}
                                </Badge>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>
                            {section.isSimpleMode ? (
                              <Badge variant="default">Yes</Badge>
                            ) : (
                              <Badge variant="outline">No</Badge>
                            )}
                          </TableCell>
                          <TableCell>
                            <Switch
                              checked={setting.isEnabled}
                              onCheckedChange={(checked) =>
                                handleToggleSection(section.key, checked)
                              }
                            />
                          </TableCell>
                          <TableCell>
                            <Select
                              value={setting.minRoleLevel.toString()}
                              onValueChange={(v) =>
                                handleRoleLevelChange(section.key, parseInt(v))
                              }
                            >
                              <SelectTrigger className="w-[140px]">
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                {ROLE_LEVELS.map((role) => (
                                  <SelectItem key={role.level} value={role.level.toString()}>
                                    {role.level}. {role.name}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </TableCell>
                          <TableCell className="text-muted-foreground text-sm">
                            {section.path}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Role Levels Tab */}
        <TabsContent value="roles">
          <Card>
            <CardHeader>
              <CardTitle>Preset Role Levels</CardTitle>
              <CardDescription>
                Overview of default role levels and their access permissions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {ROLE_LEVELS.map((role) => (
                  <div
                    key={role.level}
                    className="flex items-center justify-between p-4 border rounded-lg"
                  >
                    <div className="flex items-center gap-4">
                      <Badge variant="outline" className="text-lg px-4 py-2">
                        {role.level}
                      </Badge>
                      <div>
                        <h4 className="font-medium">{role.name}</h4>
                        <p className="text-sm text-muted-foreground">{role.description}</p>
                      </div>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {flatSections.filter((s) => s.minRoleLevel <= role.level).length} sections
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Custom Roles Tab */}
        <TabsContent value="custom">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Custom Roles</CardTitle>
                  <CardDescription>
                    Create custom roles with specific section access
                  </CardDescription>
                </div>
                <Dialog open={showRoleDialog} onOpenChange={setShowRoleDialog}>
                  <DialogTrigger asChild>
                    <Button>
                      <Plus className="h-4 w-4 mr-2" />
                      Create Role
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-2xl">
                    <DialogHeader>
                      <DialogTitle>Create Custom Role</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>Role Name</Label>
                          <Input
                            placeholder="e.g., Content Creator"
                            value={newRole.roleName}
                            onChange={(e) =>
                              setNewRole({ ...newRole, roleName: e.target.value })
                            }
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Base Level</Label>
                          <Select
                            value={newRole.level.toString()}
                            onValueChange={(v) =>
                              setNewRole({ ...newRole, level: parseInt(v) })
                            }
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {ROLE_LEVELS.map((role) => (
                                <SelectItem key={role.level} value={role.level.toString()}>
                                  {role.level}. {role.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <Label>Description</Label>
                        <Textarea
                          placeholder="Describe this role's purpose..."
                          value={newRole.description}
                          onChange={(e) =>
                            setNewRole({ ...newRole, description: e.target.value })
                          }
                        />
                      </div>

                      <div className="space-y-2">
                        <Label>Allowed Sections</Label>
                        <ScrollArea className="h-[200px] border rounded-lg p-2">
                          <div className="space-y-2">
                            {flatSections
                              .filter((s) => !s.parentKey)
                              .map((section) => {
                                const Icon = getIcon(section.icon);
                                return (
                                  <div
                                    key={section.key}
                                    className="flex items-center gap-2 p-2 hover:bg-muted rounded"
                                  >
                                    <input
                                      type="checkbox"
                                      id={`section-${section.key}`}
                                      checked={newRole.sections.includes(section.key)}
                                      onChange={(e) => {
                                        if (e.target.checked) {
                                          setNewRole({
                                            ...newRole,
                                            sections: [...newRole.sections, section.key],
                                          });
                                        } else {
                                          setNewRole({
                                            ...newRole,
                                            sections: newRole.sections.filter(
                                              (s) => s !== section.key
                                            ),
                                          });
                                        }
                                      }}
                                      className="rounded"
                                    />
                                    <Icon className="h-4 w-4" />
                                    <label htmlFor={`section-${section.key}`} className="flex-1">
                                      {section.label}
                                    </label>
                                  </div>
                                );
                              })}
                          </div>
                        </ScrollArea>
                      </div>
                    </div>
                    <DialogFooter>
                      <Button variant="outline" onClick={() => setShowRoleDialog(false)}>
                        Cancel
                      </Button>
                      <Button onClick={handleCreateRole} disabled={isSaving}>
                        {isSaving ? (
                          <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        ) : (
                          <Save className="h-4 w-4 mr-2" />
                        )}
                        Create Role
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>
            </CardHeader>
            <CardContent>
              {settings?.customRoles && Object.keys(settings.customRoles).length > 0 ? (
                <div className="space-y-4">
                  {Object.entries(settings.customRoles).map(([name, role]) => (
                    <div
                      key={name}
                      className="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="font-medium">{name}</h4>
                          <Badge variant="outline">Level {role.level}</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{role.description}</p>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {role.sections.slice(0, 5).map((s) => (
                            <Badge key={s} variant="secondary" className="text-xs">
                              {s}
                            </Badge>
                          ))}
                          {role.sections.length > 5 && (
                            <Badge variant="outline" className="text-xs">
                              +{role.sections.length - 5} more
                            </Badge>
                          )}
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleDeleteRole(name)}
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <Users className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No custom roles created yet</p>
                  <p className="text-sm">Click "Create Role" to add a custom role</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Defaults Tab */}
        <TabsContent value="defaults">
          <Card>
            <CardHeader>
              <CardTitle>Default Settings</CardTitle>
              <CardDescription>
                Configure default menu behavior for your organization
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div>
                  <Label className="text-base">Default Menu Mode</Label>
                  <p className="text-sm text-muted-foreground mb-4">
                    New users will start with this menu mode
                  </p>
                  <div className="flex gap-4">
                    <Button
                      variant={settings?.defaultMode === "simple" ? "default" : "outline"}
                      onClick={() => handleDefaultModeChange("simple")}
                    >
                      Simple Mode
                    </Button>
                    <Button
                      variant={settings?.defaultMode === "complete" ? "default" : "outline"}
                      onClick={() => handleDefaultModeChange("complete")}
                    >
                      Complete Mode
                    </Button>
                  </div>
                </div>

                <Separator />

                <div>
                  <Label className="text-base">Mode Descriptions</Label>
                  <div className="grid grid-cols-2 gap-4 mt-4">
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Simple Mode</h4>
                      <p className="text-sm text-muted-foreground mb-4">
                        Shows only essential features: Chat, Upload, Documents, Create
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {["chat", "upload", "documents", "create"].map((key) => {
                          const section = flatSections.find((s) => s.key === key);
                          if (!section) return null;
                          const Icon = getIcon(section.icon);
                          return (
                            <Badge key={key} variant="secondary">
                              <Icon className="h-3 w-3 mr-1" />
                              {section.label}
                            </Badge>
                          );
                        })}
                      </div>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Complete Mode</h4>
                      <p className="text-sm text-muted-foreground mb-4">
                        Shows all features based on user's role level
                      </p>
                      <p className="text-sm">
                        {flatSections.filter((s) => !s.parentKey).length} total sections available
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
