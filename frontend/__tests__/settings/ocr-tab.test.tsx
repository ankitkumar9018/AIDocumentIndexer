/**
 * OcrTab Component Tests
 * ======================
 * Tests for the OCR Configuration settings tab component.
 */

import React from "react";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import { OcrTab } from "@/app/dashboard/admin/settings/components/ocr-tab";
import { mockOcrData, renderTabComponent } from "./utils";

describe("OcrTab", () => {
  const defaultProps = {
    ocrLoading: false,
    ocrData: mockOcrData,
    refetchOCR: jest.fn(),
    updateOCRSettings: {
      mutate: jest.fn(),
    },
    downloadModels: {
      mutateAsync: jest.fn().mockResolvedValue({
        status: "success",
        downloaded: ["en", "de"],
      }),
    },
    downloadingModels: false,
    setDownloadingModels: jest.fn(),
    downloadResult: null,
    setDownloadResult: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("loading state", () => {
    it("shows loading spinner when ocrLoading is true", () => {
      const props = {
        ...defaultProps,
        ocrLoading: true,
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      // Should show a loading spinner
      expect(document.querySelector(".animate-spin")).toBeInTheDocument();
    });

    it("does not show loading spinner when ocrLoading is false", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      // Should not show loading in content area
      const content = screen.getByText("OCR Provider Configuration");
      expect(content).toBeInTheDocument();
    });
  });

  describe("rendering", () => {
    it("renders the OCR provider configuration card", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("OCR Provider Configuration")).toBeInTheDocument();
    });

    it("renders the downloaded models card", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("Downloaded Models")).toBeInTheDocument();
    });

    it("renders the OCR provider select", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("OCR Provider")).toBeInTheDocument();
    });

    it("renders provider options", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      // Default provider is PaddleOCR, visible as selected option
      expect(screen.getByText("PaddleOCR")).toBeInTheDocument();
      // Tesseract is in dropdown, may not be visible until opened
    });

    it("renders model variant select when not tesseract", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("Model Variant")).toBeInTheDocument();
    });

    it("renders language buttons", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("English")).toBeInTheDocument();
      expect(screen.getByText("German")).toBeInTheDocument();
      expect(screen.getByText("French")).toBeInTheDocument();
    });
  });

  describe("provider selection", () => {
    it("shows PaddleOCR settings when paddleocr is selected", () => {
      const props = {
        ...defaultProps,
        ocrData: {
          ...mockOcrData,
          settings: {
            ...mockOcrData.settings,
            "ocr.provider": "paddleocr",
          },
        },
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      expect(screen.getByText("Model Variant")).toBeInTheDocument();
      expect(screen.getByText("Languages")).toBeInTheDocument();
      expect(screen.getByText("Auto-Download Models")).toBeInTheDocument();
    });

    it("hides PaddleOCR settings when tesseract is selected", () => {
      const props = {
        ...defaultProps,
        ocrData: {
          ...mockOcrData,
          settings: {
            ...mockOcrData.settings,
            "ocr.provider": "tesseract",
          },
        },
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      expect(screen.queryByText("Model Variant")).not.toBeInTheDocument();
      expect(screen.queryByText("Languages")).not.toBeInTheDocument();
    });
  });

  describe("language selection", () => {
    it("shows selected languages as active", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      const englishButton = screen.getByText("English");
      const germanButton = screen.getByText("German");
      const frenchButton = screen.getByText("French");

      // English and German should be default selected
      expect(englishButton.closest("button")).toHaveClass("bg-primary");
      expect(germanButton.closest("button")).toHaveClass("bg-primary");
      // French should not be selected
      expect(frenchButton.closest("button")).not.toHaveClass("bg-primary");
    });

    it("calls updateOCRSettings when language is toggled", () => {
      const updateOCRSettings = { mutate: jest.fn() };
      const props = {
        ...defaultProps,
        updateOCRSettings,
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      const frenchButton = screen.getByText("French");
      fireEvent.click(frenchButton);

      expect(updateOCRSettings.mutate).toHaveBeenCalledWith({
        "ocr.paddle.languages": ["en", "de", "fr"],
      });
    });

    it("removes language when already selected", () => {
      const updateOCRSettings = { mutate: jest.fn() };
      const props = {
        ...defaultProps,
        updateOCRSettings,
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      const germanButton = screen.getByText("German");
      fireEvent.click(germanButton);

      expect(updateOCRSettings.mutate).toHaveBeenCalledWith({
        "ocr.paddle.languages": ["en"],
      });
    });
  });

  describe("model info display", () => {
    it("displays model directory", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("Model Directory")).toBeInTheDocument();
      expect(screen.getByText("./data/paddle_models")).toBeInTheDocument();
    });

    it("displays total size", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("Total Size")).toBeInTheDocument();
      expect(screen.getByText("250 MB")).toBeInTheDocument();
    });

    it("displays model status", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("installed")).toBeInTheDocument();
    });

    it("displays downloaded models count", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("2 models")).toBeInTheDocument();
    });

    it("displays individual model files", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("en_PP-OCRv4_det")).toBeInTheDocument();
      expect(screen.getByText("en_PP-OCRv4_rec")).toBeInTheDocument();
    });
  });

  describe("download models button", () => {
    it("renders download button", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("Download Selected Models")).toBeInTheDocument();
    });

    it("shows loading state when downloading", () => {
      const props = {
        ...defaultProps,
        downloadingModels: true,
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      expect(screen.getByText("Downloading Models...")).toBeInTheDocument();
    });

    it("disables button when downloading", () => {
      const props = {
        ...defaultProps,
        downloadingModels: true,
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      const button = screen.getByRole("button", { name: /downloading/i });
      expect(button).toBeDisabled();
    });

    it("calls download function when clicked", async () => {
      const setDownloadingModels = jest.fn();
      const setDownloadResult = jest.fn();
      const downloadModels = {
        mutateAsync: jest.fn().mockResolvedValue({
          status: "success",
          downloaded: ["en", "de"],
        }),
      };

      const props = {
        ...defaultProps,
        setDownloadingModels,
        setDownloadResult,
        downloadModels,
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      const button = screen.getByText("Download Selected Models");
      fireEvent.click(button);

      expect(setDownloadingModels).toHaveBeenCalledWith(true);

      await waitFor(() => {
        expect(downloadModels.mutateAsync).toHaveBeenCalledWith({
          languages: ["en", "de"],
          variant: "server",
        });
      });
    });
  });

  describe("download result display", () => {
    it("shows success message when download succeeds", () => {
      const props = {
        ...defaultProps,
        downloadResult: {
          success: true,
          message: "Downloaded 2 languages successfully",
        },
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      expect(screen.getByText("Downloaded 2 languages successfully")).toBeInTheDocument();
    });

    it("shows error message when download fails", () => {
      const props = {
        ...defaultProps,
        downloadResult: {
          success: false,
          message: "Download failed: network error",
        },
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      expect(screen.getByText("Download failed: network error")).toBeInTheDocument();
    });
  });

  describe("switches", () => {
    it("renders auto-download switch", () => {
      renderTabComponent(<OcrTab {...defaultProps} />, "ocr");

      expect(screen.getByText("Auto-Download Models")).toBeInTheDocument();
    });

    it("renders tesseract fallback switch when paddleocr is selected", () => {
      const props = {
        ...defaultProps,
        ocrData: {
          ...mockOcrData,
          settings: {
            ...mockOcrData.settings,
            "ocr.provider": "paddleocr",
          },
        },
      };

      renderTabComponent(<OcrTab {...props} />, "ocr");

      expect(screen.getByText("Tesseract Fallback")).toBeInTheDocument();
    });
  });
});
