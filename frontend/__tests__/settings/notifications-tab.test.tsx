/**
 * NotificationsTab Component Tests
 * =================================
 * Tests for the Notifications settings tab component.
 */

import React from "react";
import { screen, fireEvent } from "@testing-library/react";
import { NotificationsTab } from "@/app/dashboard/admin/settings/components/notifications-tab";
import { mockLocalSettings, renderTabComponent } from "./utils";

describe("NotificationsTab", () => {
  const defaultProps = {
    localSettings: mockLocalSettings,
    handleSettingChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders the notifications tab content", () => {
      renderTabComponent(<NotificationsTab {...defaultProps} />, "notifications");

      expect(screen.getByText("Notifications")).toBeInTheDocument();
    });

    it("renders processing completed option", () => {
      renderTabComponent(<NotificationsTab {...defaultProps} />, "notifications");

      expect(screen.getByText("Processing Completed")).toBeInTheDocument();
    });

    it("renders processing failed option", () => {
      renderTabComponent(<NotificationsTab {...defaultProps} />, "notifications");

      expect(screen.getByText("Processing Failed")).toBeInTheDocument();
    });

    it("renders cost alerts option", () => {
      renderTabComponent(<NotificationsTab {...defaultProps} />, "notifications");

      expect(screen.getByText("Cost Alerts")).toBeInTheDocument();
    });

    it("renders description text", () => {
      renderTabComponent(<NotificationsTab {...defaultProps} />, "notifications");

      expect(
        screen.getByText(/Configure notification preferences/)
      ).toBeInTheDocument();
    });
  });

  describe("checkbox states", () => {
    it("shows all checkboxes checked by default", () => {
      const props = {
        ...defaultProps,
        localSettings: {},
      };

      renderTabComponent(<NotificationsTab {...props} />, "notifications");

      const checkboxes = screen.getAllByRole("checkbox");
      expect(checkboxes).toHaveLength(3);
      // All default to true
      expect(checkboxes[0]).toBeChecked();
      expect(checkboxes[1]).toBeChecked();
      expect(checkboxes[2]).toBeChecked();
    });

    it("shows processing completed as unchecked when disabled", () => {
      const props = {
        ...defaultProps,
        localSettings: {
          ...mockLocalSettings,
          "notifications.processing_completed": false,
        },
      };

      renderTabComponent(<NotificationsTab {...props} />, "notifications");

      const checkboxes = screen.getAllByRole("checkbox");
      expect(checkboxes[0]).not.toBeChecked();
    });

    it("shows cost alerts as unchecked when disabled", () => {
      const props = {
        ...defaultProps,
        localSettings: {
          ...mockLocalSettings,
          "notifications.cost_alerts": false,
        },
      };

      renderTabComponent(<NotificationsTab {...props} />, "notifications");

      const checkboxes = screen.getAllByRole("checkbox");
      expect(checkboxes[2]).not.toBeChecked();
    });
  });

  describe("interactions", () => {
    it("calls handleSettingChange when processing completed is toggled", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
        localSettings: {
          ...mockLocalSettings,
          "notifications.processing_completed": true,
        },
      };

      renderTabComponent(<NotificationsTab {...props} />, "notifications");

      const checkboxes = screen.getAllByRole("checkbox");
      fireEvent.click(checkboxes[0]);

      expect(handleSettingChange).toHaveBeenCalledWith(
        "notifications.processing_completed",
        false
      );
    });

    it("calls handleSettingChange when processing failed is toggled", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
        localSettings: {
          ...mockLocalSettings,
          "notifications.processing_failed": true,
        },
      };

      renderTabComponent(<NotificationsTab {...props} />, "notifications");

      const checkboxes = screen.getAllByRole("checkbox");
      fireEvent.click(checkboxes[1]);

      expect(handleSettingChange).toHaveBeenCalledWith(
        "notifications.processing_failed",
        false
      );
    });

    it("calls handleSettingChange when cost alerts is toggled", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
        localSettings: {
          ...mockLocalSettings,
          "notifications.cost_alerts": true,
        },
      };

      renderTabComponent(<NotificationsTab {...props} />, "notifications");

      const checkboxes = screen.getAllByRole("checkbox");
      fireEvent.click(checkboxes[2]);

      expect(handleSettingChange).toHaveBeenCalledWith(
        "notifications.cost_alerts",
        false
      );
    });
  });
});
