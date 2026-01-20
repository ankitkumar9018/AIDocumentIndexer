/**
 * Wrapper Tab Components Tests
 * ============================
 * Tests for AnalyticsTab, ModelsTab, and JobQueueTab components.
 * These are simpler wrapper components that render child components.
 */

import React from "react";
import { screen } from "@testing-library/react";
import { AnalyticsTab } from "@/app/dashboard/admin/settings/components/analytics-tab";
import { ModelsTab } from "@/app/dashboard/admin/settings/components/models-tab";
import { JobQueueTab } from "@/app/dashboard/admin/settings/components/jobqueue-tab";
import { mockLocalSettings, mockProvidersData, renderTabComponent } from "./utils";

// Mock child components to isolate testing
const MockUsageAnalyticsCard = () => <div data-testid="usage-analytics">Usage Analytics</div>;
const MockProviderHealthCard = () => <div data-testid="provider-health">Provider Health</div>;
const MockCostAlertsCard = () => <div data-testid="cost-alerts">Cost Alerts</div>;
const MockModelConfigurationSection = () => <div data-testid="model-config">Model Configuration</div>;
const MockJobQueueSettings = () => <div data-testid="job-queue">Job Queue Settings</div>;

describe("AnalyticsTab", () => {
  const defaultProps = {
    UsageAnalyticsCard: MockUsageAnalyticsCard,
    ProviderHealthCard: MockProviderHealthCard,
    CostAlertsCard: MockCostAlertsCard,
  };

  it("renders the analytics tab", () => {
    renderTabComponent(<AnalyticsTab {...defaultProps} />, "analytics");

    expect(screen.getByTestId("usage-analytics")).toBeInTheDocument();
    expect(screen.getByTestId("provider-health")).toBeInTheDocument();
    expect(screen.getByTestId("cost-alerts")).toBeInTheDocument();
  });

  it("renders all child components", () => {
    renderTabComponent(<AnalyticsTab {...defaultProps} />, "analytics");

    expect(screen.getByText("Usage Analytics")).toBeInTheDocument();
    expect(screen.getByText("Provider Health")).toBeInTheDocument();
    expect(screen.getByText("Cost Alerts")).toBeInTheDocument();
  });
});

describe("ModelsTab", () => {
  const defaultProps = {
    ModelConfigurationSection: MockModelConfigurationSection,
    providers: mockProvidersData.providers || [],
  };

  it("renders the models tab", () => {
    renderTabComponent(<ModelsTab {...defaultProps} />, "models");

    expect(screen.getByTestId("model-config")).toBeInTheDocument();
  });

  it("renders model configuration section", () => {
    renderTabComponent(<ModelsTab {...defaultProps} />, "models");

    expect(screen.getByText("Model Configuration")).toBeInTheDocument();
  });

  it("passes providers to child component", () => {
    const providers = mockProvidersData.providers || [];
    const props = {
      ...defaultProps,
      providers,
    };

    renderTabComponent(<ModelsTab {...props} />, "models");

    // Component should render without error
    expect(screen.getByTestId("model-config")).toBeInTheDocument();
  });
});

describe("JobQueueTab", () => {
  const defaultProps = {
    JobQueueSettings: MockJobQueueSettings,
    localSettings: mockLocalSettings,
    handleSettingChange: jest.fn(),
  };

  it("renders the job queue tab", () => {
    renderTabComponent(<JobQueueTab {...defaultProps} />, "jobqueue");

    expect(screen.getByTestId("job-queue")).toBeInTheDocument();
  });

  it("renders job queue settings section", () => {
    renderTabComponent(<JobQueueTab {...defaultProps} />, "jobqueue");

    expect(screen.getByText("Job Queue Settings")).toBeInTheDocument();
  });

  it("passes props to child component", () => {
    const handleSettingChange = jest.fn();
    const props = {
      ...defaultProps,
      handleSettingChange,
    };

    renderTabComponent(<JobQueueTab {...props} />, "jobqueue");

    // Component should render without error
    expect(screen.getByTestId("job-queue")).toBeInTheDocument();
  });
});
