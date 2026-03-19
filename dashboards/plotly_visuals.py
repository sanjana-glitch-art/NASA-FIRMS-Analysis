import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

df = pd.read_csv("NASA_FIRMS_2022-24.csv")
df["acq_date"] = pd.to_datetime(df["acq_date"])
df["month"] = df["acq_date"].dt.month

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

TYPE_MAP = {
    0: "Type 0 — Vegetation fire",
    1: "Type 1 — Volcano",
    2: "Type 2 — Static land source",
    3: "Type 3 — Offshore",
}

C22   = "#378ADD"
C23   = "#639922"
C24   = "#D85A30"
CAVG  = "#7F77DD"
CTYPE = ["#639922", "#BA7517", "#378ADD", "#D4537E"]

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
body { background:#f4f6f9!important; font-family:'Inter',sans-serif!important; }
.dash-header {
    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    border-radius:16px; padding:28px 32px; margin-bottom:24px;
    position:relative; overflow:hidden;
}
.dash-header::before {
    content:''; position:absolute; top:-50%; right:-10%;
    width:300px; height:300px;
    background:radial-gradient(circle,rgba(255,120,50,.15) 0%,transparent 70%);
    border-radius:50%;
}
.dash-header h2 { color:#fff; font-size:22px; font-weight:600; margin:0 0 6px; letter-spacing:-.3px; }
.dash-header p  { color:rgba(255,255,255,.55); font-size:13px; margin:0; }
.filter-panel {
    background:#fff; border-radius:14px; padding:20px 24px;
    margin-bottom:22px;
    box-shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
    border:1px solid #eaecf0;
}
.filter-label {
    font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:.07em; color:#6b7280; margin-bottom:10px; display:block;
}
.chart-card {
    background:#fff; border-radius:14px; padding:20px 24px;
    margin-bottom:20px;
    box-shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
    border:1px solid #eaecf0;
}
.chart-card-title { font-size:13px; font-weight:600; color:#111827; margin-bottom:2px; }
.chart-card-sub   { font-size:11px; color:#9ca3af; margin-bottom:14px; }
.divider { height:1px; background:#f3f4f6; margin:0 -24px 16px -24px; }
.kpi-card {
    background:#fff; border-radius:12px; padding:16px 18px;
    border:1px solid #eaecf0;
    box-shadow:0 1px 2px rgba(0,0,0,.04); height:100%;
}
.kpi-label {
    font-size:11px; color:#9ca3af; font-weight:500;
    text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px;
}
.kpi-value { font-size:24px; font-weight:600; line-height:1.2; margin-bottom:2px; }
.kpi-sub   { font-size:11px; color:#9ca3af; }
.footer-text { text-align:center; font-size:11px; color:#d1d5db; padding:12px 0 4px; }
"""

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="NASA FIRMS Dashboard",
)

app.index_string = (
    "<!DOCTYPE html><html><head>"
    "{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    "<style>" + CUSTOM_CSS + "</style>"
    "</head><body>{%app_entry%}"
    "<footer>{%config%}{%scripts%}{%renderer%}</footer>"
    "</body></html>"
)

def kpi_card(label, value, sub, color):
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value", style={"color": color}),
        html.Div(sub,   className="kpi-sub"),
    ], className="kpi-card")

app.layout = html.Div(
    style={"maxWidth": "1160px", "margin": "0 auto", "padding": "24px 16px"},
    children=[

        # Header
        html.Div([
            html.H2("🔥  NASA FIRMS — Wildfire EDA Dashboard"),
            html.P("MODIS satellite fire detections · Continental USA · 2022 – 2024"),
        ], className="dash-header"),

        # Filters
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Span("Year", className="filter-label"),
                    dcc.Checklist(
                        id="filter-year",
                        options=[{"label": str(y), "value": y} for y in [2022, 2023, 2024]],
                        value=[2022, 2023, 2024],
                        inline=True,
                        inputStyle={
                            "marginRight": "5px", "width": "15px",
                            "height": "15px", "cursor": "pointer",
                        },
                        labelStyle={
                            "marginRight": "22px", "fontSize": "13px",
                            "fontWeight": "500", "color": "#374151", "cursor": "pointer",
                        },
                    ),
                ], md=4, style={"borderRight": "1px solid #f3f4f6", "paddingRight": "28px"}),

                dbc.Col([
                    html.Span("Fire type", className="filter-label"),
                    dcc.Dropdown(
                        id="filter-type",
                        options=[{"label": TYPE_MAP[t], "value": t} for t in range(4)],
                        value=[0, 1, 2, 3],
                        multi=True,
                        placeholder="Select fire types...",
                        clearable=False,
                        style={"fontSize": "13px"},
                    ),
                ], md=8, style={"paddingLeft": "28px"}),
            ], align="center"),
        ], className="filter-panel"),

        # KPI row
        dbc.Row([
            dbc.Col(html.Div(id="kpi-total"), md=3),
            dbc.Col(html.Div(id="kpi-frp"),   md=3),
            dbc.Col(html.Div(id="kpi-conf"),  md=3),
            dbc.Col(html.Div(id="kpi-night"), md=3),
        ], className="g-3 mb-4"),

        # Chart 1
        html.Div([
            html.Div("Chart 1 — Annual fire detections", className="chart-card-title"),
            html.Div("Total MODIS detections per year, filtered by selection", className="chart-card-sub"),
            html.Div(className="divider"),
            dcc.Graph(id="chart1", config={"displayModeBar": False}, style={"height": "300px"}),
        ], className="chart-card"),

        # Chart 2
        html.Div([
            html.Div("Chart 2 — Monthly seasonality", className="chart-card-title"),
            html.Div("Detection counts by month across selected years + 3-year average", className="chart-card-sub"),
            html.Div(className="divider"),
            dcc.Graph(id="chart2", config={"displayModeBar": False}, style={"height": "320px"}),
        ], className="chart-card"),

        # Chart 3
        html.Div([
            html.Div("Chart 3 — Fire type distribution", className="chart-card-title"),
            html.Div("Proportion of each fire type within current filter selection", className="chart-card-sub"),
            html.Div(className="divider"),
            dcc.Graph(id="chart3", config={"displayModeBar": False}, style={"height": "360px"}),
        ], className="chart-card"),

        # Footer
        html.Div(
            "Data source: NASA FIRMS · MODIS · Terra & Aqua satellites · Collection 6 / 6.1",
            className="footer-text",
        ),
    ]
)


# ── Layout helper
def base_layout(**overrides):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#374151"),
        margin=dict(t=10, b=50, l=70, r=20),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=12)),
        yaxis=dict(gridcolor="#f3f4f6", zeroline=False, tickfont=dict(size=12)),
    )
    layout.update(overrides)
    return layout


# ── Callback
@app.callback(
    Output("kpi-total", "children"),
    Output("kpi-frp",   "children"),
    Output("kpi-conf",  "children"),
    Output("kpi-night", "children"),
    Output("chart1", "figure"),
    Output("chart2", "figure"),
    Output("chart3", "figure"),
    Input("filter-year", "value"),
    Input("filter-type", "value"),
)
def update_all(years, types):
    empty    = go.Figure()
    empty.update_layout(base_layout())
    blank    = kpi_card("—", "—", "", "#9ca3af")

    if not years or not types:
        return blank, blank, blank, blank, empty, empty, empty

    dff = df[df["year"].isin(years) & df["type"].isin(types)]
    if dff.empty:
        return blank, blank, blank, blank, empty, empty, empty

    # KPIs
    total     = len(dff)
    avg_frp   = dff["frp"].mean()
    avg_conf  = dff["confidence"].mean()
    pct_night = (dff["daynight"] == "N").mean() * 100

    k_total = kpi_card("Total detections", f"{total:,}",       "filtered records",     C22)
    k_frp   = kpi_card("Avg FRP",          f"{avg_frp:.1f} MW","fire radiative power", C24)
    k_conf  = kpi_card("Avg confidence",   f"{avg_conf:.1f}%", "detection quality",    CAVG)
    k_night = kpi_card("Night detections", f"{pct_night:.1f}%","of total",             C23)

    # Chart 1
    annual    = dff.groupby("year").size().reset_index(name="count")
    color_map = {2022: C22, 2023: C23, 2024: C24}

    fig1 = go.Figure(go.Bar(
        x=annual["year"].astype(str),
        y=annual["count"],
        text=annual["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        textfont=dict(size=12, color="#374151"),
        marker_color=[color_map.get(int(y), "#aaaaaa") for y in annual["year"]],
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Detections: %{y:,}<extra></extra>",
        cliponaxis=False,
    ))
    fig1.update_layout(base_layout(
        yaxis=dict(
            gridcolor="#f3f4f6", zeroline=False,
            title=dict(text="Detections", font=dict(size=12)),
            tickformat=",d", tickfont=dict(size=11),
        ),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=13, color="#374151")),
        bargap=0.5, showlegend=False,
        margin=dict(t=20, b=30, l=75, r=20),
    ))

    # Chart 2
    monthly = (
        dff.groupby(["year", "month"]).size()
        .reset_index(name="count")
        .pivot(index="month", columns="year", values="count")
        .reindex(range(1, 13)).fillna(0).astype(int)
    )
    monthly.columns = [int(c) for c in monthly.columns]

    fig2 = go.Figure()
    for yr, col in [(2022, C22), (2023, C23), (2024, C24)]:
        if yr in monthly.columns and yr in years:
            fig2.add_trace(go.Scatter(
                x=MONTH_LABELS, y=monthly[yr].tolist(),
                name=str(yr), mode="lines+markers",
                line=dict(color=col, width=2.5),
                marker=dict(color=col, size=7, line=dict(color="white", width=1.5)),
                hovertemplate=f"{yr} %{{x}}: <b>%{{y:,}}</b><extra></extra>",
            ))
    if len(monthly.columns) >= 2:
        avg = monthly.mean(axis=1).round(0).astype(int)
        fig2.add_trace(go.Scatter(
            x=MONTH_LABELS, y=avg.tolist(),
            name="3-yr avg", mode="lines+markers",
            line=dict(color=CAVG, width=2, dash="dot"),
            marker=dict(color=CAVG, size=5),
            hovertemplate="Avg %{x}: <b>%{y:,}</b><extra></extra>",
        ))
    fig2.update_layout(base_layout(
        yaxis=dict(
            gridcolor="#f3f4f6", zeroline=False,
            title=dict(text="Detections", font=dict(size=12)),
            tickformat=",d", tickfont=dict(size=11),
        ),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=12)),
        hovermode="x unified",
        legend=dict(
            orientation="h", y=-0.2, x=0.5, xanchor="center",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=10, b=65, l=75, r=20),
    ))

    # Chart 3
    type_counts = dff["type"].value_counts().sort_index()
    labels = [TYPE_MAP.get(int(i), f"Type {i}") for i in type_counts.index]
    colors = [CTYPE[int(i)] for i in type_counts.index]

    fig3 = go.Figure(go.Pie(
        labels=labels,
        values=type_counts.values.tolist(),
        hole=0.60,
        marker=dict(colors=colors, line=dict(color="white", width=2.5)),
        textinfo="percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
        pull=[0.025] * len(type_counts),
        sort=False,
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#374151"),
        margin=dict(t=10, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(
            orientation="v", x=0.72, y=0.5,
            xanchor="left", yanchor="middle",
            font=dict(size=12), bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        annotations=[dict(
            text=f"<b>{total:,}</b><br>total",
            x=0.36, y=0.5,
            font=dict(size=15, color="#111827"),
            showarrow=False, align="center",
        )],
    )

    return k_total, k_frp, k_conf, k_night, fig1, fig2, fig3


# ── Run
if __name__ == "__main__":
    app.run(debug=True, port=8050)
