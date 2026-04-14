"""
Plain-English reporting tab based on trained MMM insights.
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.inference import compute_roas, get_channel_contributions
from utils import get_logger, safe_divide

log = get_logger(__name__)


def render_report_tab() -> None:
    st.header("📝 Insight Report")

    mmm = st.session_state.get("mmm")
    model_df = st.session_state.get("model_df")
    channel_cols = st.session_state.get("channel_cols", [])
    eval_report = st.session_state.get("eval_report", {})

    if mmm is None or model_df is None or not channel_cols:
        st.warning("⚠️ Train a model first to generate a plain-English report.")
        return

    with st.spinner("Generating report..."):
        try:
            contrib_df = get_channel_contributions(mmm, model_df, channel_cols)
            roas_df = compute_roas(mmm, model_df, channel_cols)
            report = _build_report(eval_report, contrib_df, roas_df, model_df, channel_cols)
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")
            log.exception(exc)
            return

    st.subheader("Executive Summary")
    st.write(report["summary"])

    st.subheader("Model Readout")
    for line in report["model_readout"]:
        st.write(f"- {line}")

    st.subheader("Channel Findings")
    for line in report["channel_findings"]:
        st.write(f"- {line}")

    st.subheader("Recommended Interpretation")
    for line in report["recommendations"]:
        st.write(f"- {line}")


def _build_report(eval_report, contrib_df, roas_df, model_df, channel_cols):
    in_sample = eval_report.get("in_sample", {})
    r2 = in_sample.get("r2")
    mape = in_sample.get("mape")
    rmse = in_sample.get("rmse")

    total_contrib = contrib_df[channel_cols].sum().sort_values(ascending=False)
    top_channel = total_contrib.index[0] if not total_contrib.empty else None
    top_share = safe_divide(float(total_contrib.iloc[0]), float(total_contrib.sum())) if len(total_contrib) else 0.0

    weakest_channel = total_contrib.index[-1] if not total_contrib.empty else None
    weakest_share = safe_divide(float(total_contrib.iloc[-1]), float(total_contrib.sum())) if len(total_contrib) else 0.0

    best_roas_row = roas_df.iloc[0] if not roas_df.empty else None
    worst_roas_row = roas_df.iloc[-1] if not roas_df.empty else None

    spend_totals = model_df[channel_cols].sum().sort_values(ascending=False)
    top_spend_channel = spend_totals.index[0] if not spend_totals.empty else None

    summary = _build_summary(r2, mape, top_channel, top_share, best_roas_row)

    model_readout = [
        _model_quality_sentence(r2, mape, rmse),
        _fit_balance_sentence(top_channel, top_share, weakest_channel, weakest_share),
        _spend_vs_return_sentence(top_spend_channel, best_roas_row, worst_roas_row),
    ]

    channel_findings = _channel_findings(total_contrib, roas_df)
    recommendations = _recommendations(r2, best_roas_row, worst_roas_row, top_spend_channel, top_channel)

    return {
        "summary": summary,
        "model_readout": [line for line in model_readout if line],
        "channel_findings": channel_findings,
        "recommendations": recommendations,
    }


def _build_summary(r2, mape, top_channel, top_share, best_roas_row) -> str:
    quality = "usable" if _is_finite(r2) else "incomplete"
    if _is_finite(r2):
        if r2 >= 0.7:
            quality = "strong"
        elif r2 >= 0.4:
            quality = "moderate"
        else:
            quality = "weak"

    top_channel_text = (
        f"{top_channel} drives the largest share of modeled contribution at about {top_share:.1%}."
        if top_channel is not None else
        "No dominant contributing channel could be identified."
    )

    roas_text = (
        f"The most efficient channel in this run is {best_roas_row['channel']} with ROAS of {best_roas_row['roas']:.2f}."
        if best_roas_row is not None and _is_finite(best_roas_row['roas']) else
        "ROAS rankings were not available."
    )

    mape_text = f" MAPE is {mape:.1%}." if _is_finite(mape) else ""
    return f"The model fit looks {quality}.{mape_text} {top_channel_text} {roas_text}"


def _model_quality_sentence(r2, mape, rmse) -> str:
    if not _is_finite(r2):
        return "The training fit metrics are incomplete, which usually means the model produced partial non-finite predictions during evaluation."

    return (
        f"Training fit shows R² of {r2:.3f}, MAPE of {mape:.1%}, and RMSE of {rmse:,.0f}. "
        f"This suggests the model is {'capturing the signal reasonably well' if r2 >= 0.4 else 'still noisy and should be interpreted carefully'}."
    )


def _fit_balance_sentence(top_channel, top_share, weakest_channel, weakest_share) -> str:
    if top_channel is None or weakest_channel is None:
        return ""

    return (
        f"{top_channel} is the largest modeled contributor at roughly {top_share:.1%} of total contribution, "
        f"while {weakest_channel} contributes about {weakest_share:.1%}. This helps separate scale leaders from marginal channels."
    )


def _spend_vs_return_sentence(top_spend_channel, best_roas_row, worst_roas_row) -> str:
    parts = []
    if top_spend_channel is not None:
        parts.append(f"The highest-spend channel is {top_spend_channel}.")
    if best_roas_row is not None and _is_finite(best_roas_row["roas"]):
        parts.append(f"The best efficiency is {best_roas_row['channel']} at ROAS {best_roas_row['roas']:.2f}.")
    if worst_roas_row is not None and _is_finite(worst_roas_row["roas"]):
        parts.append(f"The weakest efficiency is {worst_roas_row['channel']} at ROAS {worst_roas_row['roas']:.2f}.")
    return " ".join(parts)


def _channel_findings(total_contrib: pd.Series, roas_df: pd.DataFrame) -> list[str]:
    findings: list[str] = []

    for channel, value in total_contrib.head(3).items():
        findings.append(f"{channel} is among the top contribution drivers with modeled contribution of {value:,.0f}.")

    if not roas_df.empty:
        for _, row in roas_df.head(3).iterrows():
            findings.append(
                f"{row['channel']} ranks highly on efficiency with total spend of {row['total_spend']:,.0f} "
                f"and ROAS of {row['roas']:.2f}."
            )

    return findings


def _recommendations(r2, best_roas_row, worst_roas_row, top_spend_channel, top_channel) -> list[str]:
    recs: list[str] = []

    if _is_finite(r2) and r2 < 0.4:
        recs.append("Treat the channel rankings as directional rather than precise, because the current fit quality is still weak.")
    else:
        recs.append("Use the report as a decision-support summary, but confirm budget moves with the Optimise and Scenarios tabs.")

    if best_roas_row is not None and top_spend_channel is not None and best_roas_row["channel"] != top_spend_channel:
        recs.append(
            f"Compare {top_spend_channel} against {best_roas_row['channel']}, because the biggest spend bucket is not the most efficient one."
        )

    if worst_roas_row is not None and _is_finite(worst_roas_row["roas"]):
        recs.append(f"Review {worst_roas_row['channel']} for overspend, weak creative, or targeting inefficiency before scaling it further.")

    if top_channel is not None:
        recs.append(f"Protect measurement quality for {top_channel}, because it currently drives the largest modeled business impact.")

    return recs


def _is_finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False
