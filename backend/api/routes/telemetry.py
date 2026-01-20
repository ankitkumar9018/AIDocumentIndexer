"""
AIDocumentIndexer - Telemetry API Routes
=========================================

API endpoints for accessing Phase 15 telemetry data to measure
quality improvements from model-specific optimizations.
"""

from fastapi import APIRouter, Query
from typing import Optional, Dict, Any

from backend.services.rag_module.advanced_optimizations import get_telemetry

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.get("/summary")
async def get_telemetry_summary(
    model_name: Optional[str] = Query(None, description="Filter by specific model")
) -> Dict[str, Any]:
    """
    Get comprehensive telemetry summary.

    Returns metrics on:
    - Total queries processed
    - Citation rates (indicator of grounded responses)
    - Hallucination rate estimates
    - Phase 15 optimization usage
    - Per-model breakdowns

    **Use Cases:**
    - Measure Phase 15 impact on hallucination rates
    - Compare model performance
    - Track optimization feature usage

    **Example Response:**
    ```json
    {
      "total_queries": 1234,
      "overall_citation_rate": 0.87,
      "overall_hallucination_rate": 0.13,
      "uncertainty_rate": 0.08,
      "phase15_usage": {
        "json_mode": 45,
        "multi_sampling": 89,
        "model_specific_prompts": 1234
      },
      "by_model_summary": {
        "llama3.2:1b": {
          "queries": 450,
          "citation_rate": 0.82,
          "hallucination_rate": 0.18
        },
        "qwen2.5-7b": {
          "queries": 320,
          "citation_rate": 0.94,
          "hallucination_rate": 0.06
        }
      }
    }
    ```

    Args:
        model_name: Optional model name to filter by

    Returns:
        Telemetry summary dictionary
    """
    telemetry = get_telemetry()
    summary = telemetry.get_summary()

    if model_name:
        # Filter to specific model
        if model_name in summary.get("by_model_summary", {}):
            return {
                "model": model_name,
                "statistics": summary["by_model_summary"][model_name],
                "global_comparison": {
                    "model_citation_rate": summary["by_model_summary"][model_name]["citation_rate"],
                    "overall_citation_rate": summary["overall_citation_rate"],
                    "model_hallucination_rate": summary["by_model_summary"][model_name]["hallucination_rate"],
                    "overall_hallucination_rate": summary["overall_hallucination_rate"],
                }
            }
        else:
            return {
                "model": model_name,
                "error": "No data for this model",
                "available_models": list(summary.get("by_model_summary", {}).keys())
            }

    return summary


@router.get("/metrics/hallucination-rate")
async def get_hallucination_rate(
    model_name: Optional[str] = Query(None, description="Filter by specific model")
) -> Dict[str, float]:
    """
    Get estimated hallucination rate.

    **Methodology:**
    Hallucination rate is estimated as the percentage of responses
    without proper source citations. Responses with citations are
    grounded in retrieved documents and less likely to hallucinate.

    **Expected Impact of Phase 15:**
    - Before: ~40-60% hallucination rate for tiny models
    - After: ~10-20% hallucination rate with Phase 15 optimizations

    Args:
        model_name: Optional model name to filter by

    Returns:
        Dictionary with hallucination rate (0.0 to 1.0)
    """
    telemetry = get_telemetry()
    rate = telemetry.get_hallucination_rate(model_name)

    result = {
        "hallucination_rate": rate,
        "citation_rate": telemetry.get_citation_rate(model_name),
    }

    if model_name:
        result["model"] = model_name

    return result


@router.get("/metrics/phase15-usage")
async def get_phase15_usage() -> Dict[str, Any]:
    """
    Get Phase 15 optimization feature usage statistics.

    Shows how often each optional enhancement is being used:
    - JSON mode for structured output (Qwen models)
    - Multi-sampling for quality (tiny models)
    - Model-specific prompts (all Phase 15 models)

    Returns:
        Dictionary with usage counts
    """
    telemetry = get_telemetry()
    summary = telemetry.get_summary()

    phase15_usage = summary.get("phase15_usage", {})
    total_queries = summary.get("total_queries", 0)

    return {
        "total_queries": total_queries,
        "json_mode": {
            "count": phase15_usage.get("json_mode", 0),
            "percentage": (phase15_usage.get("json_mode", 0) / total_queries * 100) if total_queries > 0 else 0
        },
        "multi_sampling": {
            "count": phase15_usage.get("multi_sampling", 0),
            "percentage": (phase15_usage.get("multi_sampling", 0) / total_queries * 100) if total_queries > 0 else 0
        },
        "model_specific_prompts": {
            "count": phase15_usage.get("model_specific_prompts", 0),
            "percentage": (phase15_usage.get("model_specific_prompts", 0) / total_queries * 100) if total_queries > 0 else 0
        }
    }


@router.get("/comparison")
async def compare_models() -> Dict[str, Any]:
    """
    Compare performance across different models.

    Useful for:
    - Identifying best-performing models
    - Validating Phase 15 optimizations
    - Understanding model-specific characteristics

    Returns:
        Dictionary with per-model statistics sorted by citation rate
    """
    telemetry = get_telemetry()
    summary = telemetry.get_summary()

    models = summary.get("by_model_summary", {})

    # Sort by citation rate (higher is better)
    sorted_models = sorted(
        models.items(),
        key=lambda x: x[1].get("citation_rate", 0),
        reverse=True
    )

    return {
        "total_models": len(sorted_models),
        "rankings": [
            {
                "rank": i + 1,
                "model": model_name,
                "queries": stats["queries"],
                "citation_rate": stats["citation_rate"],
                "hallucination_rate": stats["hallucination_rate"],
                "avg_response_length": stats["avg_length"],
            }
            for i, (model_name, stats) in enumerate(sorted_models)
        ]
    }


@router.get("/health")
async def telemetry_health() -> Dict[str, Any]:
    """
    Check if telemetry system is operational.

    Returns:
        Status and basic stats
    """
    telemetry = get_telemetry()
    summary = telemetry.get_summary()

    return {
        "status": "operational",
        "total_queries_recorded": summary.get("total_queries", 0),
        "models_tracked": len(summary.get("by_model", {})),
        "recording_since": summary.get("first_recorded"),
        "last_update": summary.get("last_recorded"),
    }
