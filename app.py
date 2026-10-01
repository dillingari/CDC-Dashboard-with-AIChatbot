"""Provisional U.S. Natality Dashboard (2025) - Streamlit Application.

An interactive analytical dashboard designed for undergraduate business
analytics students to explore geographic, monthly, and infant-sex patterns
in provisional 2025 CDC natality counts.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# BUILT-IN HELPERS (used automatically if the optional `src/` folder is missing)
# -----------------------------------------------------------------------------
_FB_MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_FB_STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

_FB_DATA_FILE_STEM = "Provisional_Natality_2025_CDC"


def _fb_find_data_file() -> Path:
    """Look for the natality data file (.xlsx or .csv) near app.py."""
    base = Path(__file__).resolve().parent
    folders = [base, base / "data", base / "Data", base.parent, Path.cwd(), Path.cwd() / "data"]
    for folder in folders:
        for ext in (".xlsx", ".csv"):
            candidate = folder / f"{_FB_DATA_FILE_STEM}{ext}"
            if candidate.is_file():
                return candidate
    for ext in (".xlsx", ".csv"):  # last resort: search sub-folders
        for candidate in base.rglob(f"{_FB_DATA_FILE_STEM}{ext}"):
            return candidate
    raise FileNotFoundError(
        f"Could not find {_FB_DATA_FILE_STEM}.xlsx (or .csv). Upload the data file to the "
        "same GitHub folder as app.py."
    )


@st.cache_data(show_spinner=False)
def _fb_load_natality_data() -> pd.DataFrame:
    """Load and clean the natality data into the column names the dashboard uses."""
    path = _fb_find_data_file()
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)

    rename_map = {
        "state_of_residence": "State of Residence",
        "month": "Month",
        "month_code": "Month Code",
        "year_code": "Year Code",
        "sex_of_infant": "Sex of Infant",
        "births": "Births",
    }
    df = df.rename(columns={c: rename_map.get(str(c).strip().lower(), str(c).strip()) for c in df.columns})

    required = ["State of Residence", "Month", "Month Code", "Year Code", "Sex of Infant", "Births"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Data file is missing expected columns: {missing}")

    df = df[required].copy()
    df["State of Residence"] = df["State of Residence"].astype(str).str.strip()
    df["Month"] = df["Month"].astype(str).str.strip()
    df["Sex of Infant"] = df["Sex of Infant"].astype(str).str.strip()
    df["Births"] = pd.to_numeric(df["Births"], errors="coerce")
    df["Month Code"] = pd.to_numeric(df["Month Code"], errors="coerce")
    df["Year Code"] = pd.to_numeric(df["Year Code"], errors="coerce")
    df = df.dropna(subset=["Births", "Month Code"]).copy()
    df["Births"] = df["Births"].astype(int)
    df["Month Code"] = df["Month Code"].astype(int)
    df["Year Code"] = df["Year Code"].fillna(2025).astype(int)
    df["State Code"] = df["State of Residence"].map(_FB_STATE_CODES)
    df["Month"] = pd.Categorical(df["Month"], categories=_FB_MONTH_ORDER, ordered=True)
    return df.reset_index(drop=True)


def _fb_validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Basic integrity checks; returns (all_passed, list_of_messages)."""
    checks = [
        (len(df) == 1224, f"Row count is {len(df):,} (expected 1,224 = 51 geographies × 12 months × 2 sexes)."),
        (df["State of Residence"].nunique() == 51, f"{df['State of Residence'].nunique()} geographies found (expected 50 states + DC)."),
        (df["Month"].nunique() == 12, f"{df['Month'].nunique()} months found (expected 12)."),
        (df["Sex of Infant"].nunique() == 2, f"{df['Sex of Infant'].nunique()} infant-sex categories found (expected 2)."),
        (int(df.isna().sum().sum()) == 0, "No missing values in any column."),
        (bool((df["Births"] >= 0).all()), "No negative birth counts."),
        (bool(df["State Code"].notna().all()), "Every geography maps to a 2-letter postal code."),
    ]
    messages = [("✅ " if ok else "❌ ") + text for ok, text in checks]
    return all(ok for ok, _ in checks), messages


def _fb_calculate_kpis(df: pd.DataFrame, total_geos_available: int) -> dict:
    """Headline numbers for the KPI cards."""
    total = int(df["Births"].sum())
    n_geos = int(df["State of Residence"].nunique())
    n_months = int(df["Month"].nunique())
    by_geo = df.groupby("State of Residence")["Births"].sum().sort_values(ascending=False)
    by_month = df.groupby("Month", observed=True)["Births"].sum()
    top_month = by_month.idxmax()
    return {
        "total_births": total,
        "total_births_formatted": f"{total:,}",
        "geographies_label": f"{n_geos} of {total_geos_available}",
        "avg_monthly_births_formatted": f"{total / n_months:,.0f}" if n_months else "0",
        "top_geography_name": str(by_geo.index[0]),
        "top_geography_births": int(by_geo.iloc[0]),
        "top_month_name": str(top_month),
        "top_month_births": int(by_month.loc[top_month]),
    }


def _fb_calculate_sex_breakdown(df: pd.DataFrame) -> dict:
    """Male/female totals, shares and the male-per-female ratio."""
    by_sex = df.groupby("Sex of Infant")["Births"].sum()
    male = int(by_sex.get("Male", 0))
    female = int(by_sex.get("Female", 0))
    total = male + female
    return {
        "male_births": male,
        "female_births": female,
        "male_pct": round(100 * male / total, 1) if total else 0.0,
        "female_pct": round(100 * female / total, 1) if total else 0.0,
        "sex_ratio": (male / female) if female else 0.0,
    }


def _fb_get_filter_summary(
    selected_states, total_states, selected_months, total_months,
    selected_sex, matching_rows, total_rows,
) -> str:
    """Short text shown in the sidebar describing the active filters."""
    return (
        f"**Active filters**\n\n"
        f"- Geographies: {len(selected_states)} of {total_states}\n"
        f"- Months: {len(selected_months)} of {total_months}\n"
        f"- Infant sex: {selected_sex}\n\n"
        f"Showing {matching_rows:,} of {total_rows:,} rows."
    )


def _fb_style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        template="plotly_white",
        legend_title_text="",
    )
    return fig


def _fb_create_choropleth_map(df: pd.DataFrame) -> go.Figure:
    data = df.groupby(["State of Residence", "State Code"], as_index=False)["Births"].sum()
    fig = px.choropleth(
        data,
        locations="State Code",
        locationmode="USA-states",
        color="Births",
        hover_name="State of Residence",
        hover_data={"State Code": False, "Births": ":,"},
        scope="usa",
        color_continuous_scale="Blues",
        title="Live Births by State of Residence",
    )
    return _fb_style(fig, height=480)


def _fb_create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    data = df.groupby("Month", observed=True, as_index=False)["Births"].sum()
    data["Month"] = data["Month"].astype(str)
    fig = px.line(
        data, x="Month", y="Births", markers=True,
        title="Monthly Live Births",
        category_orders={"Month": [m for m in _FB_MONTH_ORDER if m in set(data["Month"])]},
    )
    fig.update_traces(hovertemplate="%{x}: %{y:,}<extra></extra>")
    fig.update_yaxes(tickformat=",")
    return _fb_style(fig)


def _fb_create_sex_comparison_chart(df: pd.DataFrame) -> go.Figure:
    data = df.groupby(["Month", "Sex of Infant"], observed=True, as_index=False)["Births"].sum()
    data["Month"] = data["Month"].astype(str)
    fig = px.bar(
        data, x="Month", y="Births", color="Sex of Infant", barmode="group",
        title="Monthly Births by Infant Sex",
        category_orders={"Month": [m for m in _FB_MONTH_ORDER if m in set(data["Month"])]},
    )
    fig.update_yaxes(tickformat=",")
    return _fb_style(fig)


def _fb_create_state_month_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(
        index="State of Residence", columns="Month", values="Births",
        aggfunc="sum", observed=True, fill_value=0,
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=list(pivot.index),
            colorscale="Blues",
            hovertemplate="%{y} · %{x}: %{z:,}<extra></extra>",
            colorbar=dict(title="Births"),
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Live Births by State and Month")
    return _fb_style(fig, height=max(420, 18 * len(pivot) + 120))


def _fb_create_state_ranking_chart(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    data = (
        df.groupby("State of Residence", as_index=False)["Births"].sum()
        .sort_values("Births", ascending=False)
        .head(int(top_n))
    )
    fig = px.bar(
        data, x="Births", y="State of Residence", orientation="h",
        title=f"Top {len(data)} Geographies by Live Births",
    )
    fig.update_yaxes(autorange="reversed", title_text="")
    fig.update_xaxes(tickformat=",")
    return _fb_style(fig, height=max(380, 24 * len(data) + 120))


def _fb_create_top_bottom_comparison_chart(df: pd.DataFrame, n: int = 5) -> go.Figure:
    ranked = (
        df.groupby("State of Residence", as_index=False)["Births"].sum()
        .sort_values("Births", ascending=False)
        .reset_index(drop=True)
    )
    n = max(1, int(n))
    if len(ranked) <= 2 * n:
        ranked["Group"] = "Selected"
        shown = ranked
        title = "Births by Geography"
    else:
        top = ranked.head(n).assign(Group=f"Top {n}")
        bottom = ranked.tail(n).assign(Group=f"Bottom {n}")
        shown = pd.concat([top, bottom])
        title = f"Top {n} vs. Bottom {n} Geographies"
    fig = px.bar(
        shown, x="Births", y="State of Residence", color="Group", orientation="h", title=title,
    )
    fig.update_yaxes(autorange="reversed", title_text="")
    fig.update_xaxes(tickformat=",")
    return _fb_style(fig, height=max(320, 30 * len(shown) + 120))


# Use the project's own `src/` modules when they exist; otherwise fall back to the
# built-in versions above so the app never fails with ModuleNotFoundError.
try:
    from src.charts import (
        create_choropleth_map,
        create_monthly_trend_chart,
        create_sex_comparison_chart,
        create_state_month_heatmap,
        create_state_ranking_chart,
        create_top_bottom_comparison_chart,
    )
except ImportError:
    create_choropleth_map = _fb_create_choropleth_map
    create_monthly_trend_chart = _fb_create_monthly_trend_chart
    create_sex_comparison_chart = _fb_create_sex_comparison_chart
    create_state_month_heatmap = _fb_create_state_month_heatmap
    create_state_ranking_chart = _fb_create_state_ranking_chart
    create_top_bottom_comparison_chart = _fb_create_top_bottom_comparison_chart

try:
    from src.data_loader import MONTH_ORDER, load_natality_data, validate_dataset
except ImportError:
    MONTH_ORDER = _FB_MONTH_ORDER
    load_natality_data = _fb_load_natality_data
    validate_dataset = _fb_validate_dataset

try:
    from src.metrics import calculate_kpis, calculate_sex_breakdown, get_filter_summary
except ImportError:
    calculate_kpis = _fb_calculate_kpis
    calculate_sex_breakdown = _fb_calculate_sex_breakdown
    get_filter_summary = _fb_get_filter_summary

# The openai library is only needed for the AI tab. If it is missing, the rest
# of the dashboard still works and the AI tab explains what to fix.
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
# Provider detection by API-key prefix (both use OpenAI-compatible endpoints).
LLM_PROVIDERS = {
    "gsk_": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-120b",
    },
    "xai-": {
        "name": "xAI (Grok)",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-3-mini",
    },
}

# Secret names accepted for the API key (checked in this order).
API_KEY_SECRET_NAMES = ("GROQ_API_KEY", "GROK_API_KEY", "XAI_API_KEY", "LLM_API_KEY")

# Groq models tried, in order, if the primary model is retired or unavailable.
GROQ_FALLBACK_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
]

# Substrings (lower-case) that mean "this model cannot be used" -> try the next one.
MODEL_UNAVAILABLE_MARKERS = (
    "not found",
    "decommissioned",
    "deprecated",
    "does not exist",
    "not available",
    "model_not_found",
    "model_decommissioned",
)

SUGGESTED_QUESTIONS = [
    "Which 3 states had the most births?",
    "Which month had the fewest births, and why might that be?",
    "What is the male-to-female ratio in the current selection?",
]

CHAT_HISTORY_LENGTH = 6      # number of most recent messages sent to the model
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 2000        # gpt-oss is a reasoning model and "thinks" using tokens

SYSTEM_PROMPT = """You are a friendly data assistant for the CDC/NCHS provisional 2025 U.S. natality (live births) data.

Rules:
- Answer ONLY from the data summary provided below. It reflects the user's current sidebar filters.
- All values are raw birth COUNTS, not rates. When comparing states, remind the user that population size drives the counts.
- The data is provisional and may be revised.
- If a question cannot be answered from the data (for example race, mother's age, other years), say so and suggest what data would be needed.
- Use thousands separators, double-check your arithmetic, and be concise."""

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ACCESSIBLE STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="2025 CDC Provisional Natality Dashboard",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean, readable business analytics presentation
st.markdown(
    """
    <style>
    /* Metric card background and contrast */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #1F2937;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #4B5563;
    }
    .disclaimer-card {
        padding: 0.9rem 1.2rem;
        background-color: #F8FAFC;
        border-left: 4px solid #0284C7;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #1E293B;
    }
    .warning-card {
        padding: 0.9rem 1.2rem;
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        border-radius: 4px;
        margin-bottom: 1.2rem;
        font-size: 0.92rem;
        color: #92400E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 2. DATA INGESTION & SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
try:
    raw_df = load_natality_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

ALL_GEOGRAPHIES: List[str] = sorted(raw_df["State of Residence"].unique().tolist())
ALL_MONTHS: List[str] = [m for m in MONTH_ORDER if m in raw_df["Month"].unique()]

# Initialize filter states in Streamlit session_state if not already set
if "filter_states" not in st.session_state:
    st.session_state["filter_states"] = ALL_GEOGRAPHIES.copy()
if "filter_months" not in st.session_state:
    st.session_state["filter_months"] = ALL_MONTHS.copy()
if "filter_sex" not in st.session_state:
    st.session_state["filter_sex"] = "All"


# Callback helpers for sidebar buttons
def select_all_states():
    st.session_state["filter_states"] = ALL_GEOGRAPHIES.copy()


def clear_all_states():
    st.session_state["filter_states"] = []


def select_all_months():
    st.session_state["filter_months"] = ALL_MONTHS.copy()


def clear_all_months():
    st.session_state["filter_months"] = []


def reset_all_filters():
    st.session_state["filter_states"] = ALL_GEOGRAPHIES.copy()
    st.session_state["filter_months"] = ALL_MONTHS.copy()
    st.session_state["filter_sex"] = "All"


# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & FILTERING
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔍 Filter Controls")
    st.caption("Customize your analytical view across geography, time, and infant sex.")

    # Reset button at top of sidebar
    st.button("↺ Reset All Filters", on_click=reset_all_filters, width="stretch", help="Reset to full 2025 dataset")

    st.markdown("---")

    # Geography selection
    st.subheader("1. State / Geography")
    col_state_btn1, col_state_btn2 = st.columns(2)
    with col_state_btn1:
        st.button("Select All", key="btn_sel_states", on_click=select_all_states, width="stretch")
    with col_state_btn2:
        st.button("Clear", key="btn_clr_states", on_click=clear_all_states, width="stretch")

    selected_states = st.multiselect(
        "Select Geographies:",
        options=ALL_GEOGRAPHIES,
        key="filter_states",
        help="Select one or more of the 50 states or District of Columbia.",
    )

    st.markdown("---")

    # Month selection
    st.subheader("2. Month (2025)")
    col_mo_btn1, col_mo_btn2 = st.columns(2)
    with col_mo_btn1:
        st.button("Select All", key="btn_sel_months", on_click=select_all_months, width="stretch")
    with col_mo_btn2:
        st.button("Clear", key="btn_clr_months", on_click=clear_all_months, width="stretch")

    selected_months = st.multiselect(
        "Select Months:",
        options=ALL_MONTHS,
        key="filter_months",
        help="Chronologically ordered calendar months for 2025.",
    )

    st.markdown("---")

    # Infant Sex selection
    st.subheader("3. Infant Sex")
    selected_sex = st.radio(
        "Filter by Infant Sex:",
        options=["All", "Female", "Male"],
        key="filter_sex",
        horizontal=True,
    )

    st.markdown("---")

    # Filter summary card
    filtered_df = raw_df[
        (raw_df["State of Residence"].isin(selected_states))
        & (raw_df["Month"].isin(selected_months))
    ]
    if selected_sex != "All":
        filtered_df = filtered_df[filtered_df["Sex of Infant"] == selected_sex]

    summary_text = get_filter_summary(
        selected_states=selected_states,
        total_states=len(ALL_GEOGRAPHIES),
        selected_months=selected_months,
        total_months=len(ALL_MONTHS),
        selected_sex=selected_sex,
        matching_rows=len(filtered_df),
        total_rows=len(raw_df),
    )
    st.info(summary_text)


# -----------------------------------------------------------------------------
# 4. MAIN HEADER & PEDAGOGICAL NOTICES
# -----------------------------------------------------------------------------
st.title("👶 Provisional U.S. Natality Dashboard (2025)")
st.markdown(
    "**An interactive business analytics platform for investigating geographic, "
    "temporal, and demographic live birth counts across the United States.**"
)

# Required disclaimers and source attribution
col_notice1, col_notice2 = st.columns([1.1, 1.0])
with col_notice1:
    st.markdown(
        """
        <div class="disclaimer-card">
            <b>🏛️ CDC Source Attribution & Provisional Status:</b><br>
            Data source: <i>Centers for Disease Control and Prevention (CDC), National Center for Health Statistics (NCHS),
            Provisional Natality on CDC WONDER Online Database (2025).</i><br>
            <b>Notice:</b> Data for 2025 are provisional and subject to standard ongoing vital registration updates.
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_notice2:
    st.markdown(
        """
        <div class="warning-card">
            <b>⚠️ Analytical Rule: Birth Counts vs. Birth Rates:</b><br>
            Figures presented are <b>raw counts of live births</b>, not birth rates or fertility rates.
            Comparing state totals directly reflects state population magnitude rather than fertility propensity.
            Calculating true birth rates requires female census denominators not present in this event registry.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 5. EMPTY FILTER STATE SAFEGUARD
# -----------------------------------------------------------------------------
if filtered_df.empty:
    st.warning(
        "⚠️ **No observations match the current filter selection.** "
        "Please select at least one State and one Month in the sidebar."
    )
    st.button("Reset Filters Now", on_click=reset_all_filters)
    st.stop()


# -----------------------------------------------------------------------------
# 6. AI DATA ASSISTANT (CHATBOT)
# -----------------------------------------------------------------------------
class AllModelsUnavailableError(Exception):
    """Raised when every candidate model is retired or unavailable."""


def get_secret(name: str) -> Optional[str]:
    """Safely read one Streamlit secret; returns None if missing or unreadable."""
    try:
        value = st.secrets[name]
    except Exception:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def get_api_key() -> Optional[str]:
    """Return the first API key found among the accepted secret names."""
    for name in API_KEY_SECRET_NAMES:
        key = get_secret(name)
        if key:
            return key
    return None


def detect_provider(api_key: str) -> Optional[dict]:
    """Pick the provider settings from the key prefix (gsk_ or xai-)."""
    for prefix, settings in LLM_PROVIDERS.items():
        if api_key.startswith(prefix):
            return settings
    return None


def get_model_candidates(provider: dict) -> List[str]:
    """Models to try in order: LLM_MODEL override or default, then Groq fallbacks."""
    primary = get_secret("LLM_MODEL") or provider["default_model"]
    candidates = [primary]
    if provider["name"] == "Groq":
        candidates += [m for m in GROQ_FALLBACK_MODELS if m not in candidates]
    # Start with the model that worked last time, if it is still a candidate.
    active = st.session_state.get("active_model")
    if active in candidates:
        candidates.remove(active)
        candidates.insert(0, active)
    return candidates


def is_model_unavailable_error(err: Exception) -> bool:
    """True only if the error says the model is retired, unknown or unavailable."""
    message = str(err).lower()
    return any(marker in message for marker in MODEL_UNAVAILABLE_MARKERS)


def build_data_context(df: pd.DataFrame) -> str:
    """Compact text summary of the FILTERED data (never the raw rows)."""
    sexes = sorted(df["Sex of Infant"].unique().tolist())
    sex_label = "All (Female and Male)" if len(sexes) > 1 else sexes[0]
    n_geos = df["State of Residence"].nunique()
    n_months = df["Month"].nunique()

    lines = [
        "ACTIVE FILTERS",
        f"- Infant sex: {sex_label}",
        f"- Geographies: {n_geos:,} of {len(ALL_GEOGRAPHIES):,} (states + DC)",
        f"- Months: {n_months:,} of {len(ALL_MONTHS):,}",
        "",
        f"TOTAL BIRTHS: {int(df['Births'].sum()):,}",
        "",
        "BIRTHS BY SEX",
    ]
    by_sex = df.groupby("Sex of Infant")["Births"].sum()
    for sex, births in by_sex.items():
        lines.append(f"- {sex}: {int(births):,}")

    lines += ["", "BIRTHS BY MONTH"]
    by_month = df.groupby("Month", observed=True)["Births"].sum()
    for month, births in by_month.items():
        lines.append(f"- {month}: {int(births):,}")

    pivot = df.pivot_table(
        index="State of Residence",
        columns="Sex of Infant",
        values="Births",
        aggfunc="sum",
        fill_value=0,
    )
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False)
    sex_cols = [c for c in ("Female", "Male") if c in pivot.columns]
    header_cols = ["Total"] + (sex_cols if len(sex_cols) > 1 else [])
    lines += ["", f"BIRTHS BY STATE (ranked high to low; {' | '.join(header_cols)})"]
    for rank, (state, row) in enumerate(pivot.iterrows(), start=1):
        values = " | ".join(f"{int(row[c]):,}" for c in header_cols)
        lines.append(f"{rank}. {state} | {values}")

    return "\n".join(lines)


def create_stream_with_fallback(client, provider: dict, messages: list):
    """Open a streaming chat completion, trying fallback models if one is retired.

    Only "model unavailable" errors trigger a fallback; any other error is raised.
    """
    candidates = get_model_candidates(provider)
    for model in candidates:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "stream": True,
        }
        if "gpt-oss" in model:
            # Sent in the request body so it works with any openai library version.
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        try:
            stream = client.chat.completions.create(**kwargs)
        except Exception as err:
            if is_model_unavailable_error(err):
                continue
            raise
        st.session_state["active_model"] = model
        return stream
    raise AllModelsUnavailableError()


def iter_stream_text(stream):
    """Yield only the text pieces of a streamed reply, skipping empty chunks."""
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        text = chunk.choices[0].delta.content
        if text:
            yield text


def friendly_error_message(err: Exception) -> str:
    """Translate an exception into a short, user-friendly message."""
    if isinstance(err, AllModelsUnavailableError):
        return (
            "None of the configured AI models are available. Add a LLM_MODEL secret "
            "in Streamlit Cloud with the name of a model your provider currently offers."
        )
    status = getattr(err, "status_code", None)
    text = str(err).lower()
    if status == 401 or "invalid api key" in text or "invalid_api_key" in text:
        return "The API key was rejected. Please check the key saved in your Streamlit Secrets."
    if status == 429 or "rate limit" in text or "rate_limit" in text:
        return "The free-tier rate limit was reached. Please wait a minute and try again."
    return f"Sorry, something went wrong contacting the AI service: {err}"


def get_ai_reply(client, provider: dict, df: pd.DataFrame, history: list) -> str:
    """Stream the assistant's answer into the page and return the final text."""
    try:
        system_message = {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nDATA SUMMARY (current filters):\n{build_data_context(df)}",
        }
        messages = [system_message] + history[-CHAT_HISTORY_LENGTH:]
        stream = create_stream_with_fallback(client, provider, messages)
        reply = st.write_stream(iter_stream_text(stream))
        if isinstance(reply, list):
            reply = "".join(str(part) for part in reply)
        if not reply or not str(reply).strip():
            reply = "The model returned an empty answer. Please try rephrasing your question."
            st.markdown(reply)
        return str(reply)
    except Exception as err:  # never crash the dashboard
        message = friendly_error_message(err)
        st.markdown(message)
        return message


def render_chatbot(filtered_df: pd.DataFrame) -> None:
    """Render the 'Ask the Data' tab for the currently filtered dataframe."""
    st.subheader("Ask the Data Assistant")

    api_key = get_api_key()
    if not api_key:
        st.warning(
            "No AI API key was found. To enable this tab, add your key as a secret named "
            "`GROQ_API_KEY` in Streamlit Cloud: open your app, then **⋮ → Settings → Secrets**, "
            'and add a line like `GROQ_API_KEY = "gsk_..."`. Save, and the app will restart.'
        )
        return

    provider = detect_provider(api_key)
    if provider is None:
        st.warning(
            "The API key format was not recognized. Groq keys start with `gsk_` and "
            "xAI keys start with `xai-`. Please check the key saved in your Streamlit Secrets."
        )
        return

    if OpenAI is None:
        st.error(
            "The `openai` package is not installed. Add `openai>=1.40.0` to requirements.txt "
            "and reboot the app."
        )
        return

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    active_model = st.session_state.get("active_model") or get_model_candidates(provider)[0]
    st.caption(
        f"Powered by {provider['name']} · model {active_model}. Answers are based on the data "
        "matching your current sidebar filters. AI can make mistakes — verify key numbers with the charts."
    )

    # Suggested questions + clear button
    button_cols = st.columns(len(SUGGESTED_QUESTIONS) + 1)
    suggested = None
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        if button_cols[i].button(question, key=f"suggested_q_{i}", width="stretch"):
            suggested = question
    if button_cols[-1].button("🗑️ Clear chat", key="clear_chat", width="stretch"):
        st.session_state["messages"] = []

    # Existing conversation
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed = st.chat_input("Ask a question about the births data…")
    user_question = typed or suggested
    if not user_question:
        return

    st.session_state["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    client = OpenAI(api_key=api_key, base_url=provider["base_url"], timeout=60.0)
    with st.chat_message("assistant"):
        reply = get_ai_reply(client, provider, filtered_df, st.session_state["messages"])
    st.session_state["messages"].append({"role": "assistant", "content": reply})


# -----------------------------------------------------------------------------
# 7. KPI METRICS CARDS
# -----------------------------------------------------------------------------
kpi_data = calculate_kpis(filtered_df, total_geos_available=len(ALL_GEOGRAPHIES))

kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
with kpi_col1:
    st.metric(
        label="Total Live Births",
        value=kpi_data["total_births_formatted"],
        help="Sum of live births recorded in the active filter selection.",
    )
with kpi_col2:
    st.metric(
        label="Selected Geographies",
        value=kpi_data["geographies_label"],
        help="Number of U.S. states and DC currently included.",
    )
with kpi_col3:
    st.metric(
        label="Avg. Births / Month",
        value=kpi_data["avg_monthly_births_formatted"],
        help="Total selected births divided by the number of active months.",
    )
with kpi_col4:
    st.metric(
        label="Peak Geography",
        value=kpi_data["top_geography_name"],
        delta=f"{kpi_data['top_geography_births']:,} births",
        delta_color="off",
        help="State with highest recorded volume in this selection.",
    )
with kpi_col5:
    st.metric(
        label="Peak Month",
        value=kpi_data["top_month_name"],
        delta=f"{kpi_data['top_month_births']:,} births",
        delta_color="off",
        help="Month with highest recorded volume in this selection.",
    )

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab_overview, tab_geo, tab_time_sex, tab_ai, tab_table, tab_about = st.tabs([
    "📊 Overview",
    "🗺️ Geographic Analysis",
    "📈 Monthly & Sex Analysis",
    "🤖 Ask the Data (AI)",
    "📋 Data Table & Download",
    "ℹ️ About the Data",
])

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
with tab_overview:
    st.subheader("Executive Analytical Summary")
    st.markdown(
        "A holistic overview of 2025 provisional natality data. Use this tab to identify broad seasonal "
        "trends, demographic distributions, and geographic volume concentration."
    )

    ov_col1, ov_col2 = st.columns([1.5, 1.0])
    with ov_col1:
        st.plotly_chart(create_monthly_trend_chart(filtered_df), width="stretch", key="chart_overview_trend")

    with ov_col2:
        # Biological sex distribution snapshot
        sex_metrics = calculate_sex_breakdown(filtered_df)
        st.markdown("#### 🔬 Infant Sex Distribution")
        st.markdown(
            f"- **Male Births:** `{sex_metrics['male_births']:,}` ({sex_metrics['male_pct']}%)  \n"
            f"- **Female Births:** `{sex_metrics['female_births']:,}` ({sex_metrics['female_pct']}%)  \n"
            f"- **Sex Ratio at Birth:** `{sex_metrics['sex_ratio']:.3f}` *(males per female)*"
        )
        st.caption(
            "💡 *Human Biology Benchmark:* Across global populations, the natural human secondary sex ratio "
            "typically centers between 1.04 and 1.06 males per female."
        )

        st.markdown("---")
        # Top 5 vs Bottom 5 quick glance
        st.plotly_chart(create_top_bottom_comparison_chart(filtered_df, n=5), width="stretch", key="chart_overview_top_bottom")


# -----------------------------------------------------------------------------
# TAB 2: GEOGRAPHIC ANALYSIS
# -----------------------------------------------------------------------------
with tab_geo:
    st.subheader("Geographic Natality Patterns")
    st.markdown(
        "Examine birth volume variations across U.S. states. Notice how raw birth counts closely mirror "
        "large state populations (e.g., California, Texas, Florida)."
    )

    # US Choropleth Map
    st.plotly_chart(create_choropleth_map(filtered_df), width="stretch", key="chart_geo_choropleth")

    st.markdown("---")

    col_rank1, col_rank2 = st.columns([1.6, 1.0])
    with col_rank1:
        # Ranking horizontal bar chart
        n_available = int(filtered_df["State of Residence"].nunique())
        if n_available > 5:
            top_n_slider = st.slider(
                "Number of States to Display in Ranking:",
                min_value=5,
                max_value=n_available,
                value=min(20, n_available),
                help="Control the number of top-ranking geographies shown.",
            )
        else:
            top_n_slider = n_available
            st.caption(f"Showing all {n_available} selected geographies (the ranking slider appears when more than 5 are selected).")
        st.plotly_chart(create_state_ranking_chart(filtered_df, top_n=top_n_slider), width="stretch", key="chart_geo_ranking")

    with col_rank2:
        st.markdown("#### 🔍 Disparity Highlights")
        geo_agg = (
            filtered_df.groupby("State of Residence")["Births"]
            .sum()
            .reset_index()
            .sort_values("Births", ascending=False)
        )
        if not geo_agg.empty:
            top_state = geo_agg.iloc[0]
            bottom_state = geo_agg.iloc[-1]
            multiplier = (top_state["Births"] / bottom_state["Births"]) if bottom_state["Births"] > 0 else 0

            st.info(
                f"**Volume Ratio:**  \n"
                f"**{top_state['State of Residence']}** recorded `{top_state['Births']:,}` births, which is "
                f"**{multiplier:.1f}×** the volume recorded in **{bottom_state['State of Residence']}** "
                f"(`{bottom_state['Births']:,}` births).  \n\n"
                f"*Pedagogical Reminder:* This enormous difference is primarily a function of resident population size, "
                f"not reproductive fertility rates."
            )

        # Full top vs bottom comparison
        st.plotly_chart(create_top_bottom_comparison_chart(filtered_df, n=5), width="stretch", key="chart_geo_top_bottom")


# -----------------------------------------------------------------------------
# TAB 3: MONTHLY & SEX ANALYSIS
# -----------------------------------------------------------------------------
with tab_time_sex:
    st.subheader("Seasonality and Demographic Distribution")
    st.markdown(
        "Analyze monthly birth fluctuations and explore whether infant-sex proportions remain stable "
        "across time and geography."
    )

    col_sex1, col_sex2 = st.columns([1.2, 1.0])
    with col_sex1:
        st.plotly_chart(create_sex_comparison_chart(filtered_df), width="stretch", key="chart_sex_comparison")

    with col_sex2:
        st.markdown("#### 📅 Monthly Breakdown Table")
        mo_summary = (
            filtered_df.groupby(["Month Code", "Month"], observed=True)
            .agg(
                Total_Births=("Births", "sum"),
                Female_Births=("Births", lambda s: s[filtered_df.loc[s.index, "Sex of Infant"] == "Female"].sum()),
                Male_Births=("Births", lambda s: s[filtered_df.loc[s.index, "Sex of Infant"] == "Male"].sum()),
            )
            .reset_index()
            .sort_values("Month Code")
        )
        mo_summary["Male/Female Ratio"] = (
            (mo_summary["Male_Births"] / mo_summary["Female_Births"]).round(3)
        )
        st.dataframe(
            mo_summary[["Month", "Total_Births", "Female_Births", "Male_Births", "Male/Female Ratio"]],
            column_config={
                "Total_Births": st.column_config.NumberColumn("Total Births", format="%d"),
                "Female_Births": st.column_config.NumberColumn("Female Births", format="%d"),
                "Male_Births": st.column_config.NumberColumn("Male Births", format="%d"),
                "Male/Female Ratio": st.column_config.NumberColumn("Sex Ratio (M/F)", format="%.3f"),
            },
            hide_index=True,
            width="stretch",
        )

    st.markdown("---")
    st.markdown("#### 🗺️ State-by-Month Seasonality Matrix")
    st.caption("Visualizing live birth counts across all selected states (Y-axis) and calendar months (X-axis).")
    st.plotly_chart(create_state_month_heatmap(filtered_df), width="stretch", key="chart_state_month_heatmap")


# -----------------------------------------------------------------------------
# TAB 4: ASK THE DATA (AI)
# -----------------------------------------------------------------------------
with tab_ai:
    render_chatbot(filtered_df)


# -----------------------------------------------------------------------------
# TAB 5: DATA TABLE AND DOWNLOAD
# -----------------------------------------------------------------------------
with tab_table:
    st.subheader("Data Access & Export")
    st.markdown(
        "Inspect the granular observations supporting this dashboard. "
        "You can search, sort columns, or download the active subset as a CSV file."
    )

    # Search filter within table
    search_query = st.text_input("🔎 Search by state or month name:", "")
    display_df = filtered_df.copy()
    if search_query:
        mask = (
            display_df["State of Residence"].str.contains(search_query, case=False, na=False)
            | display_df["Month"].astype(str).str.contains(search_query, case=False, na=False)
        )
        display_df = display_df[mask]

    # Display columns nicely
    cols_to_show = [
        "State of Residence",
        "State Code",
        "Month",
        "Month Code",
        "Year Code",
        "Sex of Infant",
        "Births",
    ]

    st.dataframe(
        display_df[cols_to_show],
        column_config={
            "Births": st.column_config.NumberColumn("Live Births", format="%d"),
            "Month Code": st.column_config.NumberColumn("Month Code", format="%d"),
            "Year Code": st.column_config.NumberColumn("Year Code", format="%d"),
        },
        width="stretch",
        hide_index=True,
    )

    st.markdown(f"**Showing {len(display_df):,} matching rows.**")

    # CSV Download Button
    csv_bytes = display_df[cols_to_show].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="provisional_natality_2025_filtered.csv",
        mime="text/csv",
        help="Export active selection for spreadsheet or external analysis.",
    )

    st.markdown("---")
    st.markdown("#### Summary Descriptive Statistics")
    stats_df = display_df["Births"].describe().to_frame().T
    st.dataframe(
        stats_df,
        column_config={
            "count": st.column_config.NumberColumn("Count", format="%d"),
            "mean": st.column_config.NumberColumn("Mean", format="%.1f"),
            "std": st.column_config.NumberColumn("Std Dev", format="%.1f"),
            "min": st.column_config.NumberColumn("Min", format="%d"),
            "25%": st.column_config.NumberColumn("25th Percentile", format="%d"),
            "50%": st.column_config.NumberColumn("Median (50%)", format="%d"),
            "75%": st.column_config.NumberColumn("75th Percentile", format="%d"),
            "max": st.column_config.NumberColumn("Max", format="%d"),
        },
        hide_index=True,
        width="stretch",
    )


# -----------------------------------------------------------------------------
# TAB 6: ABOUT THE DATA & METHODOLOGY
# -----------------------------------------------------------------------------
with tab_about:
    st.subheader("Data Architecture & Business Analytics Guide")
    st.markdown(
        """
        This tab provides foundational knowledge and critical concepts for business analytics
        students working with public health, demographic, and administrative event datasets.
        """
    )

    with st.expander("📌 1. The Denominator Problem: Counts vs. Fertility Rates", expanded=True):
        st.markdown(
            """
            In business analytics and epidemiology, a common error is confounding **event frequencies (counts)** 
            with **incidence or fertility rates**:
            - **Count:** Total number of live birth events that occurred in a specified place and time ($3,604,640$ total in 2025).
            - **General Fertility Rate (GFR):** Number of live births per 1,000 women of childbearing age (typically ages 15–44) residing in that state.
            
            Because this CDC dataset does **not** contain census population denominators, we cannot calculate fertility rates. 
            A high birth count in California or Texas simply reflects that these states have tens of millions of residents, 
            not necessarily that their birth rates are elevated.
            """
        )

    with st.expander("🔬 2. Human Biology & The Secondary Sex Ratio", expanded=False):
        st.markdown(
            """
            Across human populations, male live births consistently outnumber female live births.
            - **Expected Secondary Sex Ratio:** Approximately **1.04 to 1.06** males per female born.
            - In the 2025 provisional dataset, total male births ($1,844,988$) divided by female births ($1,759,652$) 
              yields a ratio of **1.048**, aligning with established biological expectations.
            """
        )

    with st.expander("🏛️ 3. What Does 'Provisional' Mean in CDC Surveillance?", expanded=False):
        st.markdown(
            """
            Provisional natality data are based on live birth records received and processed by the National Center 
            for Health Statistics (NCHS) as of a cutoff date.
            - State vital registries continuously transmit birth certificates to the CDC.
            - While provisional data provide near real-time surveillance, figures may undergo minor upward revisions 
              when finalized birth registries are published.
            """
        )

    with st.expander("📖 4. Data Dictionary", expanded=False):
        dict_data = [
            {"Column": "State of Residence", "Type": "Text / String", "Description": "U.S. state or District of Columbia where the mother resides."},
            {"Column": "State Code", "Type": "Text / String", "Description": "Standard 2-letter USPS postal code (e.g., CA, TX)."},
            {"Column": "Month", "Type": "Ordered Categorical", "Description": "Calendar month of infant birth (January through December)."},
            {"Column": "Month Code", "Type": "Integer (1-12)", "Description": "Numeric chronological code corresponding to the month."},
            {"Column": "Year Code", "Type": "Integer", "Description": "Calendar year of occurrence (2025)."},
            {"Column": "Sex of Infant", "Type": "Text / Categorical", "Description": "Biological sex recorded at birth ('Female' or 'Male')."},
            {"Column": "Births", "Type": "Integer", "Description": "Count of registered provisional live births."},
        ]
        st.table(pd.DataFrame(dict_data))

    with st.expander("✅ 5. Automated Data Quality & Verification Audit", expanded=False):
        is_valid, audit_messages = validate_dataset(raw_df)
        if is_valid:
            st.success("All automated data-integrity checks passed successfully!")
        else:
            st.error("Data integrity discrepancies detected.")
        for msg in audit_messages:
            st.write(f"- {msg}")
