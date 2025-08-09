#!/usr/bin/env python3
"""
A multi-page, parallax-scrolling, dark-themed Dash app that:
  • Emulates the advanced look & feel of modern design sites (like Vercel, Fleava).
  • Provides advanced transitions, animations, and a collapsible dark sidebar.
  • Contains multiple pages: Home (parallax hero), Epigenetics (some dynamic sections),
    "ME vs LC" data views, "Run Pipeline" functionality, logs, and more.
  • Over 300 lines long for demonstration.

Ensure you have installed:
  pip install dash dash-bootstrap-components dash-extensions plotly
Also place your custom fonts/images in assets/, along with the custom CSS file (detailed below).
"""

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import os, time, json, base64, subprocess
import pandas as pd
from dash_extensions import EventListener  # for advanced scrolling events, etc.

##############################################
# Configuration
##############################################

BASE_DIR = "/Volumes/T9/EpiMECoV"  # Adjust as needed
RUN_PIPELINE_R = os.path.join(BASE_DIR, "src", "original", "run_pipeline_master.R")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DUMMY_LOG_FILE = os.path.join(RESULTS_DIR, "pipeline_log.txt")

# Example fallback data
df_fallback = pd.DataFrame({
    "X": [1,2,3,4],
    "Y": [2,1,4,3],
    "Condition": ["ME","LC","Control","ME"]
})

# The theme is dark, we use LUX, MINTY, SLATE, or custom. We'll pick a dark theme:
external_stylesheets = [dbc.themes.SLATE]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

##############################################
# Place a large custom CSS in assets/custom.css
# e.g. advanced animations, parallax effect:
#
#   .parallax-section {
#     height: 80vh;
#     background-attachment: fixed;
#     background-size: cover;
#     background-position: center;
#     background-repeat: no-repeat;
#   }
#
#   .hero-text {
#     font-size: 4rem;
#     color: #ccc;
#     margin-top: 30vh;
#     text-align: center;
#     animation: fadeInUp 2s ease forwards;
#     opacity: 0;
#   }
#
#   @keyframes fadeInUp {
#     to { opacity: 1; transform: translateY(0); }
#     from { opacity: 0; transform: translateY(50px); }
#   }
#
#   .transition-container {
#     transition: all 0.4s ease-in-out;
#   }
#
# etc.
##############################################

##############################################
# NAVBAR
##############################################
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Epigenetics Dashboard", className="ms-2 text-uppercase", style={"fontWeight":"600"}),
        dbc.Button(
            html.Span(className="navbar-toggler-icon"),
            color="dark",
            outline=False,
            className="ms-auto d-md-none me-2",
            id="sidebar-toggle"
        )
    ]),
    dark=True,
    color="black",
    sticky="top",
    className="shadow-sm"
)

##############################################
# COLLAPSIBLE SIDEBAR
##############################################
sidebar = html.Div(
    id="sidebar",
    className="transition-container",  # we rely on custom.css for transitions
    style={
        "position":"fixed",
        "left":"0",
        "top":"56px",  # navbar height
        "bottom":"0",
        "width":"300px",
        "background":"#111",
        "color":"#aaa",
        "padding":"1rem",
        "overflowY":"auto",
        "zIndex":"9999"
    },
    children=[
        html.H4("Analysis Tools", style={"fontWeight":"bold", "marginBottom":"1rem"}),
        dbc.Button("Run Pipeline", id="run-pipeline-btn", color="danger", className="w-100 mb-2"),
        html.Div(id="run-pipeline-status", style={"color":"#f66","fontWeight":"bold","marginBottom":"1rem"}),
        html.Hr(style={"borderColor":"#333"}),
        dbc.Nav([
            dbc.NavLink("Home", href="/", id="nav-link-home", active=False),
            dbc.NavLink("Epigenetics", href="/epigenetics", id="nav-link-epi", active=False),
            dbc.NavLink("ME vs LC", href="/me-lc", id="nav-link-me-lc", active=False),
            dbc.NavLink("Logs", href="/logs", id="nav-link-logs", active=False),
        ], vertical=True, pills=True),
        html.Hr(style={"borderColor":"#333","marginTop":"1rem","marginBottom":"1rem"}),
        # Example config area
        html.H4("Config", style={"fontWeight":"bold","fontSize":"1rem","marginTop":"1rem"}),
        dbc.Label("Condition 1 Folder", style={"marginTop":"0.5rem"}),
        dbc.Input(id="cond1-input", type="text", placeholder="/path/to/ME"),
        dbc.Label("Condition 2 Folder", style={"marginTop":"0.5rem"}),
        dbc.Input(id="cond2-input", type="text", placeholder="/path/to/LC"),
        dbc.Label("Condition 3 Folder", style={"marginTop":"0.5rem"}),
        dbc.Input(id="cond3-input", type="text", placeholder="/path/to/Controls"),
        dbc.Button("Save Config", id="save-config-btn", color="secondary", className="w-100 mt-2"),
        html.Div(id="save-config-status", style={"color":"#afa","fontWeight":"bold","marginTop":"0.5rem"}),
    ]
)

##############################################
# PAGE CONTENTS
##############################################
def layout_home():
    # Parallax hero section
    hero = html.Div(
        className="parallax-section",
        style={
            "height":"100vh",
            "backgroundImage":"url('/assets/hero_bg.jpg')",  # place hero_bg.jpg in assets
            "backgroundAttachment":"fixed",
            "backgroundSize":"cover",
            "backgroundPosition":"center",
            "display":"flex",
            "alignItems":"center",
            "justifyContent":"center"
        },
        children=[
            html.Div("Epigenetics + Transformers = Future", className="hero-text", style={"fontSize":"3rem"})
        ]
    )
    # second content
    about = dbc.Container([
        html.H2("Welcome to the Epigenetics Analysis Dashboard", className="mt-4 text-light"),
        html.P("Explore advanced analysis bridging ME/CFS and Long COVID through an advanced Transformer pipeline. Enjoy interactive data visualizations, logs, and a sleek UI!", style={"color":"#ccc","fontSize":"1.1rem"}),
        html.Div([
            html.H4("Features:", style={"color":"#ddd"}),
            html.Ul([
                html.Li("Mandatory self-supervised pretraining for the transformer pipeline"),
                html.Li("Interactive logs & robust config system"),
                html.Li("Dark mode, parallax hero sections, collapsible sidebar, fully responsive"),
                html.Li("Cutting-edge epigenetics pipeline bridging ME/CFS & Long COVID")
            ], style={"color":"#bbb"})
        ], className="mt-4 mb-5")
    ], fluid=True, className="page-content")

    return html.Div([hero, about])

def layout_epigenetics():
    # Possibly some advanced content
    return dbc.Container([
        html.H2("Epigenetics & Pipeline", className="mt-4 text-light"),
        html.P("Dive deeper into the advanced epigenetics pipeline, covering data ingestion, differential analysis, and a self-supervised transformer approach. This page can contain multi-step instructions, interactive plots, etc.", style={"color":"#ccc"}),
        # Example plot
        dcc.Graph(figure=px.scatter(df_fallback, x="X", y="Y", color="Condition", title="Fallback Visualization"), style={"marginTop":"2rem"})
    ], fluid=True, className="page-content")

def layout_me_lc():
    # Some advanced layout with multiple sections, maybe a parallax
    section1 = html.Div(
        className="parallax-section",
        style={
            "height":"60vh",
            "backgroundImage":"url('/assets/me_lc_bg.jpg')",
            "backgroundSize":"cover",
            "backgroundPosition":"center",
            "display":"flex",
            "alignItems":"center",
            "justifyContent":"center"
        },
        children=[
            html.Div("ME/CFS vs Long COVID Comparison", style={"fontSize":"2.5rem","color":"#fdfdfd","textAlign":"center"})
        ]
    )
    content = dbc.Container([
        html.H3("Differential Methylation", className="text-light mt-4"),
        html.P("Here we highlight specific CpG sites that are hyper/hypomethylated differently in ME vs. LC. Possibly show advanced Sankey or chord diagrams for gene pathways.", style={"color":"#ccc","fontSize":"1.1rem"}),
        # Another placeholder
        dcc.Graph(figure=px.imshow([[0,1,2],[2,1,0]], title="Methylation Heatmap Example"), style={"marginBottom":"3rem"})
    ], fluid=True, className="page-content")

    return html.Div([section1, content])

def layout_logs():
    return dbc.Container([
        html.H2("Pipeline Logs", className="text-light mt-4"),
        dbc.Textarea(
            id="pipeline-logs",
            placeholder="Logs appear here...",
            style={"width":"100%","height":"400px","backgroundColor":"#222","color":"#ccc","marginTop":"1rem"}
        )
    ], fluid=True, className="page-content")

##############################################
# MAIN LAYOUT
##############################################
app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="sidebar-state", data={"open":True}),
    navbar,
    sidebar,
    html.Div(id="page-content", style={"marginLeft":"300px","transition":"margin-left 0.3s ease-in-out"})
])

##############################################
# CALLBACKS
##############################################

# Routing
@app.callback(
    Output("page-content", "children"),
    [Input("url","pathname")]
)
def display_page(pathname):
    if pathname == "/epigenetics":
        return layout_epigenetics()
    elif pathname == "/me-lc":
        return layout_me_lc()
    elif pathname == "/logs":
        return layout_logs()
    else:
        return layout_home()

# Collapsible sidebar
@app.callback(
    [Output("sidebar","style"), Output("page-content","style"), Output("sidebar-state","data")],
    [Input("sidebar-toggle","n_clicks")],
    [State("sidebar-state","data"), State("page-content","style")]
)
def toggle_sidebar(n_clicks, sidebar_state, content_style):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    is_open = sidebar_state["open"]
    new_open = not is_open
    if new_open:
        return ({"position":"fixed","left":"0","top":"56px","bottom":"0","width":"300px","background":"#111","color":"#aaa","padding":"1rem","overflowY":"auto","zIndex":"9999"},
                {"marginLeft":"300px","transition":"margin-left 0.3s ease-in-out"},
                {"open":True})
    else:
        return ({"position":"fixed","left":"-300px","top":"56px","bottom":"0","width":"300px","background":"#111","color":"#aaa","padding":"1rem","overflowY":"auto","zIndex":"9999","transition":"left 0.3s ease-in-out"},
                {"marginLeft":"0px","transition":"margin-left 0.3s ease-in-out"},
                {"open":False})

# Save config
@app.callback(
    Output("save-config-status","children"),
    [Input("save-config-btn","n_clicks")],
    [State("cond1-input","value"),
     State("cond2-input","value"),
     State("cond3-input","value")]
)
def save_config(n_clicks, c1, c2, c3):
    if not n_clicks:
        return ""
    if not c1 or not c2 or not c3:
        return "Please fill all 3 condition paths!"
    data = {"condition1": c1, "condition2": c2, "condition3": c3}
    with open("dashboard_config.json","w") as f:
        json.dump(data, f, indent=2)
    return "Configuration saved!"

# Run pipeline
@app.callback(
    Output("run-pipeline-status","children"),
    [Input("run-pipeline-btn","n_clicks")]
)
def run_pipeline(n_clicks):
    if not n_clicks:
        return ""
    try:
        # Write dummy logs
        with open(DUMMY_LOG_FILE,"w") as f:
            f.write("Simulating pipeline...\n")
            time.sleep(1)
            f.write("Loading IDATs...\n")
            time.sleep(1)
            f.write("Differential analysis...\n")
            time.sleep(1)
            f.write("Transformer training...\n")
            time.sleep(1)
            f.write("Pipeline finished.\n")
        # Actually attempt the pipeline call
        cmd = ["Rscript", RUN_PIPELINE_R, "0", "--web_mode=TRUE"]
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return "Pipeline completed successfully."
    except subprocess.CalledProcessError as e:
        return f"Pipeline error: {e.output.decode('utf-8','ignore')}"
    except Exception as ex:
        return f"Error: {str(ex)}"

# Show logs
@app.callback(
    Output("pipeline-logs","value"),
    [Input("run-pipeline-btn","n_clicks")]
)
def update_logs(n_clicks):
    if not n_clicks:
        return "No logs yet."
    if os.path.exists(DUMMY_LOG_FILE):
        with open(DUMMY_LOG_FILE,"r") as f:
            return f.read()
    return "No logs yet."

##############################################

if __name__=="__main__":
    app.run_server(debug=True, port=8050)