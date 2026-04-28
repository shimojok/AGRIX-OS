"""
AGRIX-OS :: dashboard/app.py
=============================
SafelyChain™ 鮮度保持力ダッシュボード

Components:
  1. Vitality Score (V-Score) gauge  — ATP-decay metabolic life
  2. P_total 4-layer dynamic pricing — stacked bar decomposition
  3. Metabolic Life countdown        — sensor-linked days remaining

Aesthetic: Deep-ocean bioluminescence × precision science instrument.
  Font: IBM Plex Mono (data) + DM Serif Display (headings)
  Palette: #050d1a (void) / #00ffc8 (vitality) / #0090ff (data) /
           #ff6b35 (warning) / #ffd700 (premium)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from vitality_engine import (
    build_dashboard_data, CROP_CATALOGUE, MBT55_PRESETS,
    VSCORE_FUNCTIONAL_THRESHOLD,
)

# ─────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────
C = {
    "void":      "#050d1a",
    "surface":   "#0a1628",
    "elevated":  "#0f2040",
    "border":    "#1a3358",
    "muted":     "#2a4870",
    "text_dim":  "#4a7aaa",
    "text":      "#8ab4d4",
    "text_hi":   "#c8e0f4",
    "white":     "#e8f4ff",
    "vitality":  "#00ffc8",     # MBT55 green-teal
    "data":      "#0090ff",     # data blue
    "premium":   "#ffd700",     # value gold
    "warning":   "#ff6b35",     # degradation orange
    "critical":  "#ff2d55",     # critical red
    "carbon":    "#39d353",     # carbon green
    "health":    "#bf5af2",     # healthcare purple
    "loss":      "#30d5c8",     # turquoise
    "market":    "#5e7a9a",     # neutral grey-blue
}

FONT_MONO    = "'IBM Plex Mono', 'Courier New', monospace"
FONT_DISPLAY = "'DM Serif Display', 'Georgia', serif"
FONT_BODY    = "'DM Sans', 'Helvetica Neue', sans-serif"

GRADE_COLORS = {
    "Platinum": C["vitality"],
    "Gold":     C["premium"],
    "Silver":   "#a0c0d0",
    "Standard": C["warning"],
}

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=IBM+Plex+Mono:wght@300;400;500;600&"
    "family=DM+Serif+Display&"
    "family=DM+Sans:wght@300;400;500&display=swap"
)


# ─────────────────────────────────────────────────────────────────
# App init
# ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="SafelyChain™ Vitality Dashboard",
    external_stylesheets=[GOOGLE_FONTS],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Inject global CSS via index_string
_GLOBAL_CSS = f"""
@keyframes pulse {{0%,100%{{opacity:1}}50%{{opacity:0.35}}}}
*{{box-sizing:border-box}}
::-webkit-scrollbar{{width:5px;background:{C['void']}}}
::-webkit-scrollbar-thumb{{background:{C['border']};border-radius:3px}}
.Select-control{{background:{C['elevated']}!important;border-color:{C['border']}!important;color:{C['text_hi']}}}
.Select-menu-outer{{background:{C['elevated']}!important;border-color:{C['border']}!important}}
.Select-option{{background:{C['elevated']}!important;color:{C['text']}}}
.Select-option.is-focused{{background:{C['muted']}!important}}
.Select-value-label{{color:{C['text_hi']}!important}}
.dash-dropdown .Select-placeholder{{color:{C['text_dim']}!important}}
"""
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>""" + _GLOBAL_CSS + """</style>
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────

def build_vscore_gauge(vscore: float, grade: str, treatment: str) -> go.Figure:
    """Bioluminescent arc gauge for Vitality Score."""
    color = GRADE_COLORS.get(grade, C["text"])

    # Colour gradient: green → yellow → orange → red
    if   vscore >= 70: bar_color = C["vitality"]
    elif vscore >= 50: bar_color = C["premium"]
    elif vscore >= 35: bar_color = C["warning"]
    else:              bar_color = C["critical"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=vscore,
        delta={
            "reference": 50,
            "increasing": {"color": C["vitality"]},
            "decreasing": {"color": C["critical"]},
            "font": {"size": 14, "family": FONT_MONO},
        },
        number={
            "suffix": "",
            "font": {"size": 52, "family": FONT_MONO, "color": bar_color},
            "valueformat": ".1f",
        },
        title={
            "text": (
                "<span style='font-family:" + FONT_DISPLAY + ";font-size:15px;"
                "color:" + C["text_hi"] + ";letter-spacing:0.1em'>"
                "V·SCORE — " + treatment + "</span><br>"
                "<span style='font-family:" + FONT_MONO + ";font-size:11px;"
                "color:" + color + "'>◆ " + grade + "</span>"
            ),
            "align": "center",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": C["border"],
                "tickfont": {"size": 9, "family": FONT_MONO, "color": C["text_dim"]},
                "dtick": 25,
            },
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "rgba(255,45,85,0.08)"},
                {"range": [35, 50], "color": "rgba(255,107,53,0.06)"},
                {"range": [50, 70], "color": "rgba(255,215,0,0.05)"},
                {"range": [70,100], "color": "rgba(0,255,200,0.05)"},
            ],
            "threshold": {
                "line": {"color": C["warning"], "width": 2},
                "thickness": 0.82,
                "value": VSCORE_FUNCTIONAL_THRESHOLD,
            },
        },
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=20, l=20, r=20),
        height=280,
        font={"family": FONT_MONO, "color": C["text"]},
    )
    return fig


def build_vscore_timeseries(ts: dict, elapsed: float) -> go.Figure:
    """Dual V-Score decay curves: MBT55 vs Conventional."""
    days  = ts["days"]
    mbt55 = ts["mbt55"]
    conv  = ts["conv"]
    thr   = ts["threshold"]

    fig = go.Figure()

    # Fill between curves (MBT55 advantage area)
    fig.add_trace(go.Scatter(
        x=days + days[::-1],
        y=mbt55 + conv[::-1],
        fill="toself",
        fillcolor="rgba(0,255,200,0.06)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Functional threshold band
    fig.add_hrect(
        y0=0, y1=thr,
        fillcolor="rgba(255,45,85,0.07)",
        line_width=0,
        annotation_text="<span style='font-size:9px'>Below functional threshold</span>",
        annotation_position="top left",
        annotation_font_color=C["critical"],
    )
    fig.add_hline(
        y=thr, line_dash="dot",
        line_color=C["warning"], line_width=1.2,
    )

    # MBT55 curve
    fig.add_trace(go.Scatter(
        x=days, y=mbt55,
        name="MBT55",
        line=dict(color=C["vitality"], width=2.5, shape="spline"),
        mode="lines",
    ))

    # Conventional curve
    fig.add_trace(go.Scatter(
        x=days, y=conv,
        name="Conventional",
        line=dict(color=C["warning"], width=1.8, dash="dash", shape="spline"),
        mode="lines",
    ))

    # Elapsed day marker
    fig.add_vline(
        x=elapsed, line_dash="dot",
        line_color=C["data"], line_width=1.5,
        annotation_text=f"<span style='font-size:9px;font-family:{FONT_MONO}'>NOW</span>",
        annotation_position="top right",
        annotation_font_color=C["data"],
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=16, b=40, l=44, r=16),
        height=200,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.12,
            font=dict(size=10, family=FONT_MONO, color=C["text"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=dict(text="Days post-harvest", font=dict(size=9, family=FONT_MONO, color=C["text_dim"])),
            color=C["text_dim"], showgrid=True,
            gridcolor="rgba(26,51,88,0.6)", zeroline=False,
            tickfont=dict(size=8, family=FONT_MONO),
        ),
        yaxis=dict(
            title=dict(text="V-Score", font=dict(size=9, family=FONT_MONO, color=C["text_dim"])),
            color=C["text_dim"], showgrid=True,
            gridcolor="rgba(26,51,88,0.6)", zeroline=False,
            range=[0, 105],
            tickfont=dict(size=8, family=FONT_MONO),
        ),
        font=dict(family=FONT_MONO, color=C["text"]),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=C["surface"], bordercolor=C["border"],
            font=dict(size=10, family=FONT_MONO, color=C["text_hi"]),
        ),
    )
    return fig


def build_ptotal_bar(ptotal: dict, crop_type: str) -> go.Figure:
    """Stacked horizontal bar: P_total layer decomposition."""
    layers = [
        ("P_market",        "Market Price",     C["market"]),
        ("V_functional",    "V Functional",     C["vitality"]),
        ("L_loss_reduction","L Loss Reduction",  C["loss"]),
        ("m_healthcare",    "m Healthcare",     C["health"]),
        ("C_carbon",        "C Carbon",         C["carbon"]),
    ]

    fig = go.Figure()
    for key, label, color in layers:
        val = ptotal.get(key, 0.0)
        pct = (val / ptotal["P_total"] * 100) if ptotal["P_total"] > 0 else 0
        fig.add_trace(go.Bar(
            name=label,
            x=[val],
            y=["P<sub>total</sub>"],
            orientation="h",
            marker=dict(
                color=color,
                opacity=0.90,
                line=dict(color=C["void"], width=1),
            ),
            text=f"${val:.3f}",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=9, family=FONT_MONO, color=C["void"]),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"${val:.4f}/kg ({pct:.1f}%)<extra></extra>"
            ),
            width=0.55,
        ))

    # P_total annotation
    fig.add_annotation(
        x=ptotal["P_total"], y=0,
        text=(
            "<span style='font-family:" + FONT_MONO + ";font-size:13px;"
            "color:" + C["premium"] + "'>"
            "<b>$" + f"{ptotal['P_total']:.3f}" + "/kg</b>"
            "  +" + f"{ptotal['premium_pct']}" + "%</span>"
        ),
        showarrow=False, xanchor="left", yanchor="middle",
        xshift=8,
    )

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=8, b=32, l=60, r=130),
        height=130,
        showlegend=True,
        legend=dict(
            orientation="h", x=0.0, y=-0.45,
            font=dict(size=9, family=FONT_MONO, color=C["text"]),
            bgcolor="rgba(0,0,0,0)",
            traceorder="normal",
        ),
        xaxis=dict(
            title=dict(text="USD / kg", font=dict(size=9, family=FONT_MONO, color=C["text_dim"])),
            color=C["text_dim"],
            showgrid=True, gridcolor="rgba(26,51,88,0.5)",
            zeroline=False,
            tickfont=dict(size=8, family=FONT_MONO),
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=9, family=FONT_MONO, color=C["text"]),
        ),
        font=dict(family=FONT_MONO, color=C["text"]),
        hoverlabel=dict(
            bgcolor=C["surface"], bordercolor=C["border"],
            font=dict(size=10, family=FONT_MONO),
        ),
    )
    return fig


def build_loss_donut(loss_breakdown: dict) -> go.Figure:
    """Donut chart: loss component breakdown."""
    labels = ["Ethylene", "Chilling", "Water", "SCFA Protection"]
    values = [
        loss_breakdown.get("ethylene_loss", 0) * 100,
        loss_breakdown.get("chilling_loss", 0) * 100,
        loss_breakdown.get("water_loss", 0) * 100,
        loss_breakdown.get("scfa_index", 0) * 15,  # protective, shown as bonus
    ]
    colors = [C["warning"], C["data"], C["loss"], C["vitality"]]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color=C["void"], width=2)),
        textfont=dict(size=9, family=FONT_MONO, color=C["text_hi"]),
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        direction="clockwise",
        sort=False,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=8, b=8, l=8, r=8),
        height=170,
        showlegend=False,
        annotations=[dict(
            text=(
                "<span style='font-family:" + FONT_MONO + ";font-size:11px;"
                "color:" + C["text_hi"] + "'>"
                "Loss<br>Analysis</span>"
            ),
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=11, family=FONT_MONO),
        )],
        font=dict(family=FONT_MONO, color=C["text"]),
        hoverlabel=dict(
            bgcolor=C["surface"], bordercolor=C["border"],
            font=dict(size=10, family=FONT_MONO),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────

def card(children, style=None):
    base = {
        "background": C["surface"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "12px",
        "padding": "20px 24px",
        "marginBottom": "16px",
        "boxShadow": f"0 4px 24px rgba(0,20,60,0.6), inset 0 1px 0 rgba(0,255,200,0.04)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def section_label(text: str, accent: str = None) -> html.Div:
    return html.Div(text, style={
        "fontFamily": FONT_MONO,
        "fontSize": "9px",
        "fontWeight": "500",
        "letterSpacing": "0.18em",
        "textTransform": "uppercase",
        "color": accent or C["text_dim"],
        "marginBottom": "8px",
    })


def metric_chip(label: str, value: str, color: str = None) -> html.Div:
    return html.Div([
        html.Div(label, style={
            "fontFamily": FONT_MONO, "fontSize": "8px",
            "color": C["text_dim"], "letterSpacing": "0.12em",
            "textTransform": "uppercase", "marginBottom": "2px",
        }),
        html.Div(value, style={
            "fontFamily": FONT_MONO, "fontSize": "16px",
            "fontWeight": "600", "color": color or C["text_hi"],
        }),
    ], style={
        "background": C["elevated"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "8px",
        "padding": "10px 14px",
        "minWidth": "80px",
        "flex": "1",
    })


def countdown_display(metlife: float, elapsed: float) -> html.Div:
    """Large countdown number with urgency colour."""
    if   metlife > 10: color = C["vitality"]
    elif metlife > 5:  color = C["premium"]
    elif metlife > 2:  color = C["warning"]
    else:               color = C["critical"]

    blink_style = {
        "animation": "pulse 1.6s ease-in-out infinite",
    } if metlife <= 2 else {}

    return html.Div([
        html.Div(
            f"{metlife:.1f}",
            style={
                "fontFamily": FONT_MONO,
                "fontSize": "72px",
                "fontWeight": "600",
                "color": color,
                "lineHeight": "1",
                "letterSpacing": "-0.02em",
                **blink_style,
            }
        ),
        html.Div(
            "DAYS REMAINING",
            style={
                "fontFamily": FONT_MONO, "fontSize": "9px",
                "letterSpacing": "0.22em", "color": C["text_dim"],
                "marginTop": "4px",
            }
        ),
        html.Div(
            f"Day {elapsed:.0f} post-harvest",
            style={
                "fontFamily": FONT_MONO, "fontSize": "10px",
                "color": C["text"], "marginTop": "6px",
            }
        ),
    ], style={"textAlign": "center", "padding": "12px 0"})


# ─────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────

app.layout = html.Div(style={
    "backgroundColor": C["void"],
    "minHeight": "100vh",
    "fontFamily": FONT_BODY,
    "color": C["text"],
    "padding": "0",
}, children=[

    # ── Global CSS placeholder (styles injected via index_string) ─

    # ── Header ───────────────────────────────────────────────────
    html.Div([
        html.Div(style={
            "display": "flex", "alignItems": "baseline", "gap": "16px",
        }, children=[
            html.Div("SafelyChain™", style={
                "fontFamily": FONT_DISPLAY,
                "fontSize": "26px",
                "color": C["white"],
                "letterSpacing": "-0.01em",
            }),
            html.Div("VITALITY DASHBOARD", style={
                "fontFamily": FONT_MONO,
                "fontSize": "9px",
                "letterSpacing": "0.22em",
                "color": C["vitality"],
                "borderLeft": f"1px solid {C['border']}",
                "paddingLeft": "14px",
                "alignSelf": "center",
            }),
            html.Div("v1.0 · AGRIX-OS", style={
                "fontFamily": FONT_MONO,
                "fontSize": "8px",
                "color": C["text_dim"],
                "marginLeft": "auto",
            }),
        ]),
        html.Div(
            "Post-harvest phenomics: metabolic vitality → dynamic value → Safely Chain™ ledger",
            style={
                "fontFamily": FONT_MONO, "fontSize": "10px",
                "color": C["text_dim"], "marginTop": "4px",
            }
        ),
    ], style={
        "background": f"linear-gradient(180deg, {C['surface']} 0%, {C['void']} 100%)",
        "borderBottom": f"1px solid {C['border']}",
        "padding": "20px 32px 16px",
    }),

    # ── Body ─────────────────────────────────────────────────────
    html.Div(style={
        "display": "grid",
        "gridTemplateColumns": "300px 1fr",
        "gap": "16px",
        "padding": "20px 24px",
        "maxWidth": "1400px",
        "margin": "0 auto",
    }, children=[

        # ── LEFT: Controls ────────────────────────────────────────
        html.Div([

            card([
                section_label("Crop & Treatment"),
                dcc.Dropdown(
                    id="crop-select",
                    options=[{"label": v["label"], "value": k}
                             for k, v in CROP_CATALOGUE.items()],
                    value="coffee_cherry",
                    clearable=False,
                    style={"marginBottom": "12px"},
                ),
                html.Div([
                    html.Div(["Treatment"], style={
                        "fontFamily": FONT_MONO, "fontSize": "9px",
                        "color": C["text_dim"], "letterSpacing": "0.12em",
                        "textTransform": "uppercase", "marginBottom": "6px",
                    }),
                    dcc.RadioItems(
                        id="treatment-select",
                        options=[
                            {"label": " MBT55", "value": "MBT55"},
                            {"label": " Conventional", "value": "Conventional"},
                        ],
                        value="MBT55",
                        inline=False,
                        labelStyle={
                            "display": "flex", "alignItems": "center",
                            "gap": "8px",
                            "fontFamily": FONT_MONO, "fontSize": "11px",
                            "color": C["text_hi"],
                            "marginBottom": "6px", "cursor": "pointer",
                        },
                        inputStyle={
                            "accentColor": C["vitality"],
                            "cursor": "pointer",
                        },
                    ),
                ]),
            ]),

            card([
                section_label("Storage Conditions"),
                html.Div("Temperature", style={
                    "fontFamily": FONT_MONO, "fontSize": "9px",
                    "color": C["text_dim"], "marginBottom": "4px",
                }),
                dcc.Slider(
                    id="temp-slider",
                    min=0, max=25, step=0.5, value=4.0,
                    marks={0: "0°", 4: "4°", 10: "10°", 20: "20°", 25: "25°"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Div(style={"height": "12px"}),
                html.Div("Humidity %RH", style={
                    "fontFamily": FONT_MONO, "fontSize": "9px",
                    "color": C["text_dim"], "marginBottom": "4px",
                }),
                dcc.Slider(
                    id="humidity-slider",
                    min=50, max=99, step=1, value=90,
                    marks={50: "50", 70: "70", 90: "90", 99: "99"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
                html.Div(style={"height": "12px"}),
                html.Div("Days post-harvest", style={
                    "fontFamily": FONT_MONO, "fontSize": "9px",
                    "color": C["text_dim"], "marginBottom": "4px",
                }),
                dcc.Slider(
                    id="elapsed-slider",
                    min=0, max=21, step=0.5, value=0,
                    marks={0: "0", 7: "7d", 14: "14d", 21: "21d"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ]),

            card([
                section_label("Carbon Market"),
                html.Div("CO₂ price (USD/tCO₂e)", style={
                    "fontFamily": FONT_MONO, "fontSize": "9px",
                    "color": C["text_dim"], "marginBottom": "4px",
                }),
                dcc.Slider(
                    id="carbon-price-slider",
                    min=5, max=120, step=5, value=28,
                    marks={5: "$5", 28: "$28", 60: "$60", 120: "$120"},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ]),

        ]),

        # ── RIGHT: Dashboard panels ───────────────────────────────
        html.Div([

            # Row 1: V-Score gauge + Metabolic Life countdown
            html.Div(style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr 1fr",
                "gap": "16px",
                "marginBottom": "0",
            }, children=[

                # V-Score gauge
                card([
                    section_label("Vitality Score  ·  V-Score", C["vitality"]),
                    dcc.Graph(
                        id="vscore-gauge",
                        config={"displayModeBar": False},
                        style={"height": "280px"},
                    ),
                    html.Div(id="vscore-metrics", style={
                        "display": "flex", "gap": "8px", "marginTop": "8px",
                    }),
                ]),

                # Metabolic Life countdown
                card([
                    section_label("Metabolic Life Countdown", C["warning"]),
                    html.Div(id="metlife-display"),
                    html.Div(id="metlife-bar"),
                    html.Div(id="metlife-hint", style={
                        "fontFamily": FONT_MONO, "fontSize": "9px",
                        "color": C["text_dim"], "marginTop": "8px",
                        "lineHeight": "1.5",
                    }),
                ]),

                # Loss donut + key stats
                card([
                    section_label("Loss Analysis", C["warning"]),
                    dcc.Graph(
                        id="loss-donut",
                        config={"displayModeBar": False},
                        style={"height": "170px"},
                    ),
                    html.Div(id="grade-badge", style={"textAlign": "center", "marginTop": "4px"}),
                ]),

            ]),

            html.Div(style={"height": "16px"}),

            # Row 2: V-Score timeseries
            card([
                section_label("V-Score Decay Trajectory  ·  MBT55 vs Conventional", C["data"]),
                dcc.Graph(
                    id="vscore-timeseries",
                    config={"displayModeBar": False},
                    style={"height": "200px"},
                ),
            ], style={"marginBottom": "16px"}),

            # Row 3: P_total decomposition
            card([
                section_label(
                    "P_total = P_market  +  V(機能性)  +  L(ロス削減)  +  m(医療費)  +  C(炭素隔離)",
                    C["premium"]
                ),
                dcc.Graph(
                    id="ptotal-bar",
                    config={"displayModeBar": False},
                    style={"height": "130px"},
                ),
                html.Div(id="ptotal-summary", style={
                    "display": "flex", "gap": "10px",
                    "flexWrap": "wrap", "marginTop": "12px",
                }),
            ]),

        ]),
    ]),

    # ── Footer ───────────────────────────────────────────────────
    html.Div([
        html.Span("AGRIX-OS · SafelyChain™ · ", style={"color": C["text_dim"]}),
        html.Span("Loss = f(Chilling Injury, Ethylene Sensitivity, Antioxidant Density)",
                  style={"color": C["text_dim"]}),
        html.Span("  ·  PBPE Carbon Pipeline Active",
                  style={"color": C["carbon"]}),
    ], style={
        "borderTop": f"1px solid {C['border']}",
        "padding": "12px 32px",
        "fontFamily": FONT_MONO,
        "fontSize": "8px",
        "letterSpacing": "0.08em",
        "textAlign": "center",
    }),

    # Interval for live-clock simulation
    dcc.Interval(id="tick", interval=5000, n_intervals=0),
    dcc.Store(id="dashboard-data"),
])


# ─────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("dashboard-data", "data"),
    [
        Input("crop-select",          "value"),
        Input("treatment-select",     "value"),
        Input("elapsed-slider",       "value"),
        Input("temp-slider",          "value"),
        Input("humidity-slider",      "value"),
        Input("carbon-price-slider",  "value"),
    ],
)
def update_data(crop, treatment, elapsed, temp, humidity, carbon_price):
    data = build_dashboard_data(
        crop_type=crop,
        treatment=treatment,
        elapsed_days=float(elapsed),
        storage_temp_C=float(temp),
        storage_humidity_pct=float(humidity),
        carbon_price_usd=float(carbon_price),
    )
    # Convert FreshnessResult to dict for JSON serialisation
    fr = data["freshness"]
    data["freshness"] = {
        "shelf_life_days":        fr.shelf_life_days,
        "loss_fraction":          fr.loss_fraction,
        "chilling_risk":          fr.chilling_risk,
        "antioxidant_retained_pct": fr.antioxidant_retained_pct,
        "freshness_score":        fr.freshness_score,
        "safelychain_grade":      fr.safelychain_grade,
        "loss_breakdown":         fr.loss_breakdown,
    }
    return data


@app.callback(
    Output("vscore-gauge",      "figure"),
    Output("vscore-metrics",    "children"),
    Output("vscore-timeseries", "figure"),
    Output("metlife-display",   "children"),
    Output("metlife-bar",       "children"),
    Output("metlife-hint",      "children"),
    Output("ptotal-bar",        "figure"),
    Output("ptotal-summary",    "children"),
    Output("loss-donut",        "figure"),
    Output("grade-badge",       "children"),
    Input("dashboard-data",     "data"),
)
def render_all(data):
    if not data:
        empty = go.Figure()
        empty.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return (empty,) * 3 + ("—",) * 3 + (empty,) + ("—",) + (empty,) + ("—",)

    vscore  = data["vscore"]
    metlife = data["metlife_days"]
    elapsed = data["elapsed_days"]
    grade   = data["grade"]
    ptotal  = data["ptotal"]
    ts      = data["timeseries"]
    fr      = data["freshness"]

    # ── Gauge ──────────────────────────────────────────────────
    gauge_fig = build_vscore_gauge(vscore, grade, data["treatment"])

    # Metrics chips under gauge
    gauge_metrics = html.Div([
        metric_chip("SCFA Index",  f"{data['scfa_index']:.0f}%", C["vitality"]),
        metric_chip("Antioxidant", f"{data['antioxidant']:.0f}%", C["data"]),
        metric_chip("Loss",        f"{data['loss_pct']:.1f}%",    C["warning"]),
    ], style={"display": "flex", "gap": "8px"})

    # ── V-Score timeseries ────────────────────────────────────
    ts_fig = build_vscore_timeseries(ts, elapsed)

    # ── Metabolic Life ────────────────────────────────────────
    metlife_display = countdown_display(metlife, elapsed)

    # Progress bar for metlife
    max_life = data["shelf_life"]
    pct = min(100, (metlife / max_life * 100)) if max_life > 0 else 0
    bar_color = C["vitality"] if pct > 60 else (C["premium"] if pct > 30 else C["critical"])
    metlife_bar = html.Div([
        html.Div(style={
            "background": C["elevated"],
            "borderRadius": "4px",
            "height": "6px",
            "overflow": "hidden",
        }, children=[
            html.Div(style={
                "width": f"{pct:.0f}%",
                "height": "100%",
                "background": f"linear-gradient(90deg, {bar_color}88, {bar_color})",
                "borderRadius": "4px",
                "transition": "width 0.4s ease",
            }),
        ]),
        html.Div(f"{pct:.0f}% of metabolic life remaining", style={
            "fontFamily": FONT_MONO, "fontSize": "8px",
            "color": C["text_dim"], "marginTop": "4px", "textAlign": "center",
        }),
    ])

    # Context hint
    if metlife > 10:
        hint = "Functional compounds (antioxidants, polyphenols) fully intact. Premium grade window active."
    elif metlife > 5:
        hint = "Metabolic activity declining. Optimal consumption window for maximum nutritional value."
    elif metlife > 0:
        hint = "⚠ Approaching functional threshold. Ethylene-driven senescence accelerating. Dispatch now."
    else:
        hint = "✕ Below V-Score threshold. Functional compounds degraded. Standard market grade only."

    # ── P_total bar ───────────────────────────────────────────
    ptotal_fig = build_ptotal_bar(ptotal, data["crop_type"])

    # P_total summary chips
    layer_defs = [
        ("P_market",        "P Market",   C["market"]),
        ("V_functional",    "V Function", C["vitality"]),
        ("L_loss_reduction","L Loss",     C["loss"]),
        ("m_healthcare",    "m Health",   C["health"]),
        ("C_carbon",        "C Carbon",   C["carbon"]),
    ]
    ptotal_chips = [
        metric_chip(label, f"${ptotal[key]:.3f}", color)
        for key, label, color in layer_defs
    ] + [metric_chip("TOTAL", f"${ptotal['P_total']:.3f}", C["premium"])]
    ptotal_summary = html.Div(ptotal_chips, style={"display": "flex", "gap": "8px", "flexWrap": "wrap"})

    # ── Loss donut ────────────────────────────────────────────
    loss_fig = build_loss_donut(fr.get("loss_breakdown") or {})

    # Grade badge
    gcolor = GRADE_COLORS.get(grade, C["text"])
    grade_badge = html.Div([
        html.Span("◆ ", style={"color": gcolor}),
        html.Span(grade, style={
            "fontFamily": FONT_MONO, "fontSize": "12px",
            "fontWeight": "600", "color": gcolor,
            "letterSpacing": "0.1em",
        }),
        html.Span(f"  {fr['freshness_score']:.3f}", style={
            "fontFamily": FONT_MONO, "fontSize": "10px",
            "color": C["text_dim"],
        }),
    ])

    return (
        gauge_fig, gauge_metrics, ts_fig,
        metlife_display, metlife_bar, hint,
        ptotal_fig, ptotal_summary,
        loss_fig, grade_badge,
    )


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  SafelyChain™ Vitality Dashboard")
    print("  AGRIX-OS  ·  http://127.0.0.1:8050")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=8050)
