/**
 * SecurityTab Component Tests
 * ===========================
 * Tests for the Security settings tab component.
 */

import React from "react";
import { screen, fireEvent } from "@testing-library/react";
import { SecurityTab } from "@/app/dashboard/admin/settings/components/security-tab";
import { mockLocalSettings, renderTabComponent } from "./utils";

describe("SecurityTab", () => {
  const defaultProps = {
    localSettings: mockLocalSettings,
    handleSettingChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders the security tab content", () => {
      renderTabComponent(<SecurityTab {...defaultProps} />, "security");

      expect(screen.getByText("Security Settings")).toBeInTheDocument();
    });

    it("renders the email verification checkbox", () => {
      renderTabComponent(<SecurityTab {...defaultProps} />, "security");

      expect(screen.getByText("Require Email Verification")).toBeInTheDocument();
    });

    it("renders two-factor authentication option", () => {
      renderTabComponent(<SecurityTab {...defaultProps} />, "security");

      expect(screen.getByText("Enable Two-Factor Authentication")).toBeInTheDocument();
    });

    it("renders audit logging option", () => {
      renderTabComponent(<SecurityTab {...defaultProps} />, "security");

      expect(screen.getByText("Enable Audit Logging")).toBeInTheDocument();
    });

    it("renders session timeout input", () => {
      renderTabComponent(<SecurityTab {...defaultProps} />, "security");

      expect(screen.getByText("Session Timeout (minutes)")).toBeInTheDocument();
    });
  });

  describe("checkbox state", () => {
    it("shows checkboxes with correct states", () => {
      const props = {
        ...defaultProps,
        localSettings: {
          ...mockLocalSettings,
          "security.require_email_verification": false,
          "security.enable_2fa": true,
          "security.enable_audit_logging": true,
        },
      };

      renderTabComponent(<SecurityTab {...props} />, "security");

      const checkboxes = screen.getAllByRole("checkbox");
      expect(checkboxes).toHaveLength(3);
      // First checkbox (email verification) should be unchecked
      expect(checkboxes[0]).not.toBeChecked();
      // Second checkbox (2FA) should be checked
      expect(checkboxes[1]).toBeChecked();
      // Third checkbox (audit logging) should be checked
      expect(checkboxes[2]).toBeChecked();
    });

    it("shows email verification checked when enabled", () => {
      const props = {
        ...defaultProps,
        localSettings: {
          ...mockLocalSettings,
          "security.require_email_verification": true,
        },
      };

      renderTabComponent(<SecurityTab {...props} />, "security");

      const checkboxes = screen.getAllByRole("checkbox");
      expect(checkboxes[0]).toBeChecked();
    });
  });

  describe("interactions", () => {
    it("calls handleSettingChange when email verification checkbox is toggled", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
        localSettings: {
          ...mockLocalSettings,
          "security.require_email_verification": false,
        },
      };

      renderTabComponent(<SecurityTab {...props} />, "security");

      const checkboxes = screen.getAllByRole("checkbox");
      fireEvent.click(checkboxes[0]);

      expect(handleSettingChange).toHaveBeenCalledWith(
        "security.require_email_verification",
        true
      );
    });

    it("calls handleSettingChange when 2FA checkbox is toggled", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
        localSettings: {
          ...mockLocalSettings,
          "security.enable_2fa": false,
        },
      };

      renderTabComponent(<SecurityTab {...props} />, "security");

      const checkboxes = screen.getAllByRole("checkbox");
      fireEvent.click(checkboxes[1]);

      expect(handleSettingChange).toHaveBeenCalledWith(
        "security.enable_2fa",
        true
      );
    });

    it("calls handleSettingChange when session timeout is changed", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
      };

      renderTabComponent(<SecurityTab {...props} />, "security");

      const input = screen.getByRole("spinbutton");
      fireEvent.change(input, { target: { value: "120" } });

      expect(handleSettingChange).toHaveBeenCalledWith(
        "security.session_timeout_minutes",
        120
      );
    });
  });

  describe("default values", () => {
    it("uses default values when settings are undefined", () => {
      const props = {
        ...defaultProps,
        localSettings: {},
      };

      renderTabComponent(<SecurityTab {...props} />, "security");

      const checkboxes = screen.getAllByRole("checkbox");
      // Email verification defaults to false
      expect(checkboxes[0]).not.toBeChecked();
      // 2FA defaults to false
      expect(checkboxes[1]).not.toBeChecked();
      // Audit logging defaults to true
      expect(checkboxes[2]).toBeChecked();

      // Session timeout defaults to 60
      const input = screen.getByRole("spinbutton");
      expect(input).toHaveValue(60);
    });
  });
});
