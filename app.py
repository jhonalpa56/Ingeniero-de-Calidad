import streamlit as st
import pandas as pd

import rebalance

st.set_page_config(page_title="ETF Multi-horizonte", layout="wide")
st.title("Panel de Rebalanceo Multi-horizonte")

st.markdown(
    """
Ingrese sus posiciones y liquidez disponible. El analizador generará recomendaciones de compra/venta
para distintos plazos y mostrará los gráficos interactivos correspondientes.
"""
)

# --- Configuración de entradas ---
default_positions = pd.DataFrame(
    [
        {"ticker": t, "units": data.get("units", 0.0), "avg_cost": data.get("avg_cost", 0.0), "currency": data.get("currency", "USD")}
        for t, data in rebalance.USER_POSITIONS.items()
    ]
)

st.sidebar.header("Configuración")
selected_horizons = st.sidebar.multiselect(
    "Horizontes (días hábiles)",
    options=sorted(set(rebalance.DEFAULT_HORIZONS + [rebalance.FORECAST_HORIZON_DEFAULT])),
    default=rebalance.DEFAULT_HORIZONS,
)
available_cash = st.sidebar.number_input(
    "Liquidez disponible (USD)",
    value=float(rebalance.AVAILABLE_CASH_USD),
    step=100.0,
)

st.subheader("Posiciones actuales")
edited_df = st.data_editor(
    default_positions,
    num_rows="dynamic",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker"),
        "units": st.column_config.NumberColumn("Unidades", step=0.1),
        "avg_cost": st.column_config.NumberColumn("Costo medio (divisa local)", step=0.1),
        "currency": st.column_config.TextColumn("Divisa"),
    },
)

st.sidebar.markdown("---")
if st.sidebar.button("Generar recomendaciones"):
    positions_override = {}
    unknown_tickers: list[str] = []
    for row in edited_df.itertuples():
        ticker = str(row.ticker).strip().upper() if getattr(row, "ticker", "") else ""
        if not ticker:
            continue
        if ticker not in rebalance.tickers:
            unknown_tickers.append(ticker)
            continue
        try:
            units = float(row.units)
        except Exception:
            units = 0.0
        try:
            avg_cost = float(row.avg_cost)
        except Exception:
            avg_cost = 0.0
        currency = str(row.currency).upper() if getattr(row, "currency", "") else "USD"
        positions_override[ticker] = {"units": units, "avg_cost": avg_cost, "currency": currency}

    if unknown_tickers:
        st.warning(
            "Se omitieron tickers desconocidos: " + ", ".join(sorted(set(unknown_tickers)))
        )

    horizons = selected_horizons if selected_horizons else [rebalance.FORECAST_HORIZON_DEFAULT]

    with st.spinner("Calculando proyecciones..."):
        result = rebalance.run_analysis(
            horizons=horizons,
            cash=available_cash,
            positions_override=positions_override,
            return_figures=True,
        )

    metrics_df = result["metrics"].copy()
    metrics_df["price_usd"] = metrics_df["price_usd"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "n/a")
    for col in ["r1d", "r1w", "r1m", "dd30"]:
        metrics_df[col] = metrics_df[col].map(rebalance.pct_str)

    st.subheader("Momentum & DD30")
    st.dataframe(metrics_df[["price_usd", "currency", "r1d", "r1w", "r1m", "dd30", "bucket"]])

    missing = result["missing_tickers"]
    if missing:
        st.info(f"Tickers sin datos recientes: {missing}")

    analysis_results = result["analysis"]
    cash_available = result["cash_available"]

    for horizon in result["horizons"]:
        analysis_df = analysis_results.get(horizon)
        st.markdown(f"### Análisis a {horizon} días hábiles")
        if analysis_df is None or analysis_df.empty:
            st.write("Sin datos suficientes para este horizonte.")
            continue

        tmp = analysis_df.copy()
        tmp["price_usd"] = tmp["price_usd"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "n/a")
        for col in ["expected_return", "momentum_proj", "momentum_1w", "momentum_1m"]:
            tmp[col] = tmp[col].map(rebalance.pct_str)
        tmp["note"] = tmp["note"].replace("", "-")
        st.dataframe(tmp[["name", "bucket", "price_usd", "expected_return", "momentum_proj", "signal", "note"]])

        positions_view = analysis_df[analysis_df.get("units", 0) > 0].copy()
        if not positions_view.empty:
            positions_view["expected_return_pct"] = positions_view["expected_return"].map(rebalance.pct_str)
            positions_view["momentum_proj_pct"] = positions_view["momentum_proj"].map(rebalance.pct_str)
            st.markdown("#### Posiciones actuales")
            st.dataframe(
                positions_view[[
                    "name",
                    "units",
                    "position_currency",
                    "avg_cost_usd",
                    "price_usd",
                    "expected_return_pct",
                    "momentum_proj_pct",
                    "signal",
                    "rationale",
                    "note",
                ]]
            )
        opportunities = analysis_df[analysis_df.get("units", 0) == 0].copy()
        opportunities = opportunities.dropna(subset=["expected_return"])
        opportunities = opportunities.sort_values("expected_return", ascending=False)
        if not opportunities.empty and cash_available > 0:
            opportunities["expected_return_pct"] = opportunities["expected_return"].map(rebalance.pct_str)
            st.markdown("#### Oportunidades con liquidez")
            st.dataframe(
                opportunities[[
                    "name",
                    "bucket",
                    "price_usd",
                    "expected_return_pct",
                    "signal",
                    "rationale",
                    "note",
                ]].head(10)
            )

    summary_tables = result["summary"]
    if summary_tables:
        st.subheader("Resumen por horizonte")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Señales**")
            st.dataframe(summary_tables["signals"])
        with col2:
            st.markdown("**Retorno esperado (%)**")
            st.dataframe(summary_tables["expected"])
        with col3:
            st.markdown("**Momentum proyectado (%)**")
            st.dataframe(summary_tables["momentum"])

    chart_dict = result["chart_outputs"] if isinstance(result["chart_outputs"], dict) else {}
    if chart_dict:
        st.subheader("Gráficos interactivos")
        for ticker, fig in chart_dict.items():
            st.markdown(f"#### {ticker} - {rebalance.tickers.get(ticker, {}).get('name', '')}")
            st.plotly_chart(fig, use_container_width=True)
    elif rebalance.go is None:
        st.info("Instala plotly para visualizar gráficos interactivos.")
else:
    st.info("Configura tus posiciones y pulsa 'Generar recomendaciones' para comenzar.")
