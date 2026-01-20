"use client";

import { Shield } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { TabsContent } from "@/components/ui/tabs";

interface SecurityTabProps {
  localSettings: Record<string, unknown>;
  handleSettingChange: (key: string, value: unknown) => void;
}

export function SecurityTab({ localSettings, handleSettingChange }: SecurityTabProps) {
  return (
    <TabsContent value="security" className="space-y-6">
      {/* Security Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Shield className="h-5 w-5" />
            Security Settings
          </CardTitle>
          <CardDescription>Authentication and access control</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Require Email Verification</p>
              <p className="text-sm text-muted-foreground">
                New users must verify their email
              </p>
            </div>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={localSettings["security.require_email_verification"] as boolean ?? false}
              onChange={(e) => handleSettingChange("security.require_email_verification", e.target.checked)}
            />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Enable Two-Factor Authentication</p>
              <p className="text-sm text-muted-foreground">
                Require 2FA for admin accounts
              </p>
            </div>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={localSettings["security.enable_2fa"] as boolean ?? false}
              onChange={(e) => handleSettingChange("security.enable_2fa", e.target.checked)}
            />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg border">
            <div>
              <p className="font-medium">Enable Audit Logging</p>
              <p className="text-sm text-muted-foreground">
                Log all user actions for compliance
              </p>
            </div>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={localSettings["security.enable_audit_logging"] as boolean ?? true}
              onChange={(e) => handleSettingChange("security.enable_audit_logging", e.target.checked)}
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Session Timeout (minutes)</label>
            <Input
              type="number"
              value={localSettings["security.session_timeout_minutes"] as number ?? 60}
              onChange={(e) => handleSettingChange("security.session_timeout_minutes", parseInt(e.target.value) || 60)}
            />
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  );
}
