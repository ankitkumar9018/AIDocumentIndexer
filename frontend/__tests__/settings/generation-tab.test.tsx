/**
 * GenerationTab Component Tests
 * =============================
 * Tests for the Document Generation settings tab component.
 */

import React from "react";
import { screen, fireEvent } from "@testing-library/react";
import { GenerationTab } from "@/app/dashboard/admin/settings/components/generation-tab";
import { mockLocalSettings, mockProvidersData, renderTabComponent } from "./utils";

describe("GenerationTab", () => {
  const defaultProps = {
    localSettings: mockLocalSettings,
    handleSettingChange: jest.fn(),
    providersData: mockProvidersData,
    ollamaLocalModels: undefined,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("rendering", () => {
    it("renders the generation tab content", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Document Generation Settings")).toBeInTheDocument();
    });

    it("renders output settings section", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Output Settings")).toBeInTheDocument();
    });

    it("renders include sources checkbox", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Include Sources")).toBeInTheDocument();
    });

    it("renders include images checkbox", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Include Images")).toBeInTheDocument();
    });

    it("renders image backend select", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Image Backend")).toBeInTheDocument();
    });

    it("renders quality review section", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Quality Review")).toBeInTheDocument();
      expect(screen.getByText("Enable Quality Review")).toBeInTheDocument();
    });

    it("renders vision analysis section", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Vision Analysis (PPTX)")).toBeInTheDocument();
    });

    it("renders content fitting section", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Content Fitting")).toBeInTheDocument();
    });

    it("renders chart generation section", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Chart Generation")).toBeInTheDocument();
    });

    it("renders style defaults section", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      expect(screen.getByText("Style Defaults")).toBeInTheDocument();
    });
  });

  describe("checkbox states", () => {
    it("shows include sources as checked when enabled", () => {
      const props = {
        ...defaultProps,
        localSettings: {
          ...mockLocalSettings,
          "generation.include_sources": true,
        },
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      const checkboxes = screen.getAllByRole("checkbox");
      expect(checkboxes[0]).toBeChecked();
    });

    it("shows include images as checked when enabled", () => {
      const props = {
        ...defaultProps,
        localSettings: {
          ...mockLocalSettings,
          "generation.include_images": true,
        },
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      const checkboxes = screen.getAllByRole("checkbox");
      expect(checkboxes[1]).toBeChecked();
    });
  });

  describe("select interactions", () => {
    it("renders image backend options", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      const select = screen.getByDisplayValue("Picsum (Free - No API Key)");
      expect(select).toBeInTheDocument();
    });

    it("calls handleSettingChange when image backend is changed", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      const select = screen.getByDisplayValue("Picsum (Free - No API Key)");
      fireEvent.change(select, { target: { value: "unsplash" } });

      expect(handleSettingChange).toHaveBeenCalledWith(
        "generation.image_backend",
        "unsplash"
      );
    });

    it("renders default tone options", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      const selects = screen.getAllByRole("combobox");
      const toneSelect = selects.find(
        (s) => s.getAttribute("value") === "professional" ||
               (s as HTMLSelectElement).value === "professional"
      );
      expect(toneSelect).toBeInTheDocument();
    });
  });

  describe("number input interactions", () => {
    it("renders minimum quality score input", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      const input = screen.getByDisplayValue("0.7");
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute("type", "number");
    });

    it("calls handleSettingChange when quality score is changed", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      const input = screen.getByDisplayValue("0.7");
      fireEvent.change(input, { target: { value: "0.8" } });

      expect(handleSettingChange).toHaveBeenCalledWith(
        "generation.min_quality_score",
        0.8
      );
    });

    it("renders chart DPI input", () => {
      renderTabComponent(<GenerationTab {...defaultProps} />, "generation");

      const input = screen.getByDisplayValue("150");
      expect(input).toBeInTheDocument();
    });
  });

  describe("checkbox interactions", () => {
    it("calls handleSettingChange when include sources is toggled", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
        localSettings: {
          ...mockLocalSettings,
          "generation.include_sources": true,
        },
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      const checkboxes = screen.getAllByRole("checkbox");
      fireEvent.click(checkboxes[0]);

      expect(handleSettingChange).toHaveBeenCalledWith(
        "generation.include_sources",
        false
      );
    });

    it("calls handleSettingChange when quality review is toggled", () => {
      const handleSettingChange = jest.fn();
      const props = {
        ...defaultProps,
        handleSettingChange,
        localSettings: {
          ...mockLocalSettings,
          "generation.enable_quality_review": true,
        },
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      const qualityCheckbox = screen.getAllByRole("checkbox").find(
        (cb) => cb.closest("div")?.textContent?.includes("Enable Quality Review")
      );

      if (qualityCheckbox) {
        fireEvent.click(qualityCheckbox);
        expect(handleSettingChange).toHaveBeenCalledWith(
          "generation.enable_quality_review",
          false
        );
      }
    });
  });

  describe("default values", () => {
    it("uses default values when settings are undefined", () => {
      const props = {
        ...defaultProps,
        localSettings: {},
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      // Check that defaults are applied
      expect(screen.getByDisplayValue("Picsum (Free - No API Key)")).toBeInTheDocument();
      expect(screen.getByDisplayValue("0.7")).toBeInTheDocument();
      expect(screen.getByDisplayValue("150")).toBeInTheDocument();
    });
  });

  describe("vision model options", () => {
    it("shows OpenAI options when OpenAI provider is active", () => {
      const props = {
        ...defaultProps,
        providersData: {
          providers: [
            { id: "1", provider_type: "openai", is_active: true },
          ],
        },
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      expect(screen.getAllByText("GPT-4o").length).toBeGreaterThan(0);
    });

    it("shows Anthropic options when Anthropic provider is active", () => {
      const props = {
        ...defaultProps,
        providersData: {
          providers: [
            { id: "1", provider_type: "anthropic", is_active: true },
          ],
        },
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      expect(screen.getAllByText("Claude 3.5 Sonnet").length).toBeGreaterThan(0);
    });

    it("shows no provider message when no providers are configured", () => {
      const props = {
        ...defaultProps,
        providersData: { providers: [] },
      };

      renderTabComponent(<GenerationTab {...props} />, "generation");

      expect(screen.getAllByText("No OpenAI provider configured").length).toBeGreaterThan(0);
    });
  });
});
