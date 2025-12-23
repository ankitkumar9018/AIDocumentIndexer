"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import {
  User,
  Mail,
  Shield,
  Key,
  Bell,
  Palette,
  Save,
  RefreshCw,
  Check,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface UserProfile {
  id: string;
  email: string;
  name: string;
  role: string;
  access_tier: number;
  tier_name: string;
  created_at: string;
}

interface UserPreferences {
  theme: "light" | "dark" | "system";
  notifications_enabled: boolean;
  email_notifications: boolean;
  default_collection: string;
  default_chat_model: string;
}

export default function ProfilePage() {
  const { data: session } = useSession();
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [preferences, setPreferences] = useState<UserPreferences>({
    theme: "system",
    notifications_enabled: true,
    email_notifications: false,
    default_collection: "",
    default_chat_model: "gpt-4o",
  });
  const [passwordForm, setPasswordForm] = useState({
    current_password: "",
    new_password: "",
    confirm_password: "",
  });
  const [passwordError, setPasswordError] = useState("");

  // Load user profile
  useEffect(() => {
    const loadProfile = async () => {
      setIsLoading(true);
      try {
        const token = localStorage.getItem("auth_token");
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/users/me`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );
        if (response.ok) {
          const data = await response.json();
          setProfile(data);
        }
      } catch (error) {
        console.error("Failed to load profile:", error);
        toast.error("Failed to load profile");
      } finally {
        setIsLoading(false);
      }
    };
    loadProfile();
  }, []);

  // Load preferences from localStorage
  useEffect(() => {
    const savedPrefs = localStorage.getItem("user_preferences");
    if (savedPrefs) {
      try {
        setPreferences(JSON.parse(savedPrefs));
      } catch {
        // Use defaults
      }
    }
  }, []);

  const handleSavePreferences = () => {
    setIsSaving(true);
    try {
      localStorage.setItem("user_preferences", JSON.stringify(preferences));
      toast.success("Preferences saved");
    } catch (error) {
      toast.error("Failed to save preferences");
    } finally {
      setIsSaving(false);
    }
  };

  const handleChangePassword = async () => {
    setPasswordError("");

    if (passwordForm.new_password !== passwordForm.confirm_password) {
      setPasswordError("Passwords do not match");
      return;
    }

    if (passwordForm.new_password.length < 8) {
      setPasswordError("Password must be at least 8 characters");
      return;
    }

    setIsSaving(true);
    try {
      const token = localStorage.getItem("auth_token");
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/users/me/password`,
        {
          method: "PUT",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            current_password: passwordForm.current_password,
            new_password: passwordForm.new_password,
          }),
        }
      );

      if (response.ok) {
        toast.success("Password changed successfully");
        setPasswordForm({
          current_password: "",
          new_password: "",
          confirm_password: "",
        });
      } else {
        const error = await response.json();
        setPasswordError(error.detail || "Failed to change password");
      }
    } catch (error) {
      setPasswordError("Failed to change password");
    } finally {
      setIsSaving(false);
    }
  };

  const getTierBadgeColor = (tier: number) => {
    if (tier >= 90) return "bg-purple-500";
    if (tier >= 50) return "bg-blue-500";
    if (tier >= 30) return "bg-green-500";
    return "bg-gray-500";
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="container mx-auto py-6 max-w-4xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Profile Settings</h1>
        <p className="text-muted-foreground">
          Manage your account settings and preferences
        </p>
      </div>

      <Tabs defaultValue="profile" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="profile" className="flex items-center gap-2">
            <User className="h-4 w-4" />
            Profile
          </TabsTrigger>
          <TabsTrigger value="preferences" className="flex items-center gap-2">
            <Palette className="h-4 w-4" />
            Preferences
          </TabsTrigger>
          <TabsTrigger value="security" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Security
          </TabsTrigger>
        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="profile">
          <Card>
            <CardHeader>
              <CardTitle>Account Information</CardTitle>
              <CardDescription>
                Your account details and access level
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center gap-4">
                <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center">
                  <User className="h-10 w-10 text-primary" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold">
                    {profile?.name || session?.user?.name || "User"}
                  </h3>
                  <p className="text-muted-foreground">
                    {profile?.email || session?.user?.email}
                  </p>
                  <div className="flex items-center gap-2 mt-2">
                    <Badge variant="outline">{profile?.role || "User"}</Badge>
                    <Badge className={getTierBadgeColor(profile?.access_tier || 10)}>
                      {profile?.tier_name || "Basic"}
                    </Badge>
                  </div>
                </div>
              </div>

              <div className="grid gap-4 pt-4 border-t">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-muted-foreground">User ID</Label>
                    <p className="font-mono text-sm">{profile?.id || "—"}</p>
                  </div>
                  <div>
                    <Label className="text-muted-foreground">Member Since</Label>
                    <p className="text-sm">
                      {profile?.created_at
                        ? new Date(profile.created_at).toLocaleDateString()
                        : "—"}
                    </p>
                  </div>
                </div>

                <div>
                  <Label className="text-muted-foreground">Access Tier</Label>
                  <div className="flex items-center gap-2 mt-1">
                    <div className="flex-1 bg-secondary rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${getTierBadgeColor(profile?.access_tier || 10)}`}
                        style={{ width: `${profile?.access_tier || 10}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium">
                      {profile?.access_tier || 10}/100
                    </span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Preferences Tab */}
        <TabsContent value="preferences">
          <Card>
            <CardHeader>
              <CardTitle>Preferences</CardTitle>
              <CardDescription>
                Customize your experience
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Theme</Label>
                    <p className="text-sm text-muted-foreground">
                      Choose your preferred color scheme
                    </p>
                  </div>
                  <Select
                    value={preferences.theme}
                    onValueChange={(value: "light" | "dark" | "system") =>
                      setPreferences({ ...preferences, theme: value })
                    }
                  >
                    <SelectTrigger className="w-32">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="system">System</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Enable in-app notifications
                    </p>
                  </div>
                  <Switch
                    checked={preferences.notifications_enabled}
                    onCheckedChange={(checked) =>
                      setPreferences({
                        ...preferences,
                        notifications_enabled: checked,
                      })
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Email Notifications</Label>
                    <p className="text-sm text-muted-foreground">
                      Receive important updates via email
                    </p>
                  </div>
                  <Switch
                    checked={preferences.email_notifications}
                    onCheckedChange={(checked) =>
                      setPreferences({
                        ...preferences,
                        email_notifications: checked,
                      })
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label>Default Chat Model</Label>
                  <Select
                    value={preferences.default_chat_model}
                    onValueChange={(value) =>
                      setPreferences({ ...preferences, default_chat_model: value })
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="gpt-4o">GPT-4o</SelectItem>
                      <SelectItem value="gpt-4o-mini">GPT-4o Mini</SelectItem>
                      <SelectItem value="claude-3-5-sonnet">Claude 3.5 Sonnet</SelectItem>
                      <SelectItem value="llama3.2">Llama 3.2 (Local)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex justify-end">
                <Button onClick={handleSavePreferences} disabled={isSaving}>
                  {isSaving ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Save className="h-4 w-4 mr-2" />
                  )}
                  Save Preferences
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security">
          <Card>
            <CardHeader>
              <CardTitle>Security Settings</CardTitle>
              <CardDescription>
                Manage your password and security options
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <h3 className="text-lg font-medium flex items-center gap-2">
                  <Key className="h-5 w-5" />
                  Change Password
                </h3>

                {passwordError && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{passwordError}</AlertDescription>
                  </Alert>
                )}

                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="current_password">Current Password</Label>
                    <Input
                      id="current_password"
                      type="password"
                      value={passwordForm.current_password}
                      onChange={(e) =>
                        setPasswordForm({
                          ...passwordForm,
                          current_password: e.target.value,
                        })
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="new_password">New Password</Label>
                    <Input
                      id="new_password"
                      type="password"
                      value={passwordForm.new_password}
                      onChange={(e) =>
                        setPasswordForm({
                          ...passwordForm,
                          new_password: e.target.value,
                        })
                      }
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="confirm_password">Confirm New Password</Label>
                    <Input
                      id="confirm_password"
                      type="password"
                      value={passwordForm.confirm_password}
                      onChange={(e) =>
                        setPasswordForm({
                          ...passwordForm,
                          confirm_password: e.target.value,
                        })
                      }
                    />
                  </div>
                </div>

                <Button onClick={handleChangePassword} disabled={isSaving}>
                  {isSaving ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Check className="h-4 w-4 mr-2" />
                  )}
                  Update Password
                </Button>
              </div>

              <div className="pt-6 border-t">
                <h3 className="text-lg font-medium flex items-center gap-2 mb-4">
                  <Bell className="h-5 w-5" />
                  Session Information
                </h3>
                <div className="text-sm text-muted-foreground space-y-2">
                  <p>
                    <span className="font-medium">Logged in as:</span>{" "}
                    {session?.user?.email}
                  </p>
                  <p>
                    <span className="font-medium">Session expires:</span>{" "}
                    {session?.expires
                      ? new Date(session.expires).toLocaleString()
                      : "—"}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
