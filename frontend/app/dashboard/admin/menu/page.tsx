"use client";

import { MenuSettings } from "@/components/admin/menu-settings";

export default function AdminMenuPage() {
  // In a real app, get orgId from auth context
  const orgId = "default-org";

  return (
    <div className="container py-6">
      <MenuSettings orgId={orgId} />
    </div>
  );
}
