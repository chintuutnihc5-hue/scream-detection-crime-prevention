import os
import random
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

from src.scream_detection import detect_scream, generate_scream_wav


app = dash.Dash(__name__)


def build_demo_incidents():
    now = datetime.now()
    incidents = []
    for i in range(12):
        time = now - timedelta(minutes=5 * (11 - i))
        score = round(random.uniform(0.65, 0.98), 3)
        incidents.append(
            {
                "timestamp": time.strftime("%H:%M"),
                "location": ["North Plaza", "Station East", "School Gate", "Market Lane", "Park Entrance"][i % 5],
                "risk_score": score,
                "alert_level": "High" if score > 0.8 else "Medium",
                "x": 10 + (i % 4) * 15,
                "y": 20 + (i % 3) * 12,
            }
        )
    return pd.DataFrame(incidents)


def get_detection_summary():
    audio_file = "data/scream_sample.wav"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(audio_file):
        generate_scream_wav(audio_file)

    result = detect_scream(audio_file)
    return {
        "status": "Scream detected" if result["is_scream"] else "Low confidence",
        "score": result["score"],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }


incident_df = build_demo_incidents()
hourly_trend = pd.DataFrame(
    {
        "hour": ["08:00", "09:00", "10:00", "11:00", "12:00", "13:00"],
        "incidents": [2, 4, 6, 8, 5, 3],
    }
)

app.layout = html.Div(
    style={"padding": "20px", "backgroundColor": "#f5f7fb", "fontFamily": "Arial"},
    children=[
        html.H1("Scream Detection & Crime Prevention Dashboard", style={"marginBottom": "20px"}),
        html.Div(
            children=[
                html.Div(
                    [
                        html.H3("Live detection"),
                        html.H2(id="status-text", style={"margin": "0"}),
                        html.P(id="status-time", style={"color": "#5f6c7b"}),
                    ],
                    style={"backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "width": "32%"},
                ),
                html.Div(
                    [
                        html.H3("Confidence score"),
                        html.H2(id="score-text", style={"margin": "0"}),
                        html.P("Model confidence for synthesized scream sample", style={"color": "#5f6c7b"}),
                    ],
                    style={"backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "width": "32%"},
                ),
                html.Div(
                    [
                        html.H3("Active alerts"),
                        html.H2("08", style={"margin": "0"}),
                        html.P("High-priority detection events", style={"color": "#5f6c7b"}),
                    ],
                    style={"backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "width": "32%"},
                ),
            ],
            style={"display": "flex", "justifyContent": "space-between", "gap": "20px", "marginBottom": "30px"},
        ),
        html.Div(
            children=[
                dcc.Graph(
                    figure=px.line(
                        hourly_trend,
                        x="hour",
                        y="incidents",
                        title="Incident trend over time",
                        template="plotly_white",
                    ),
                    style={"height": "350px"},
                ),
                dcc.Graph(
                    figure=px.scatter(
                        incident_df,
                        x="x",
                        y="y",
                        color="alert_level",
                        size="risk_score",
                        hover_name="location",
                        title="Incident hotspots",
                        template="plotly_white",
                        range_x=[0, 100],
                        range_y=[0, 100],
                    ),
                    style={"height": "350px"},
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"},
        ),
        html.Div(
            style={"marginTop": "25px", "backgroundColor": "#ffffff", "padding": "20px", "borderRadius": "12px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"},
            children=[
                html.H3("Recent incidents"),
                dcc.Graph(
                    figure=px.bar(
                        incident_df,
                        x="timestamp",
                        y="risk_score",
                        color="alert_level",
                        title="Incident risk levels",
                        template="plotly_white",
                    ),
                    style={"height": "300px"},
                ),
            ],
        ),
        dcc.Interval(id="live-update", interval=5000, n_intervals=0),
    ],
)


@app.callback(
    [Output("status-text", "children"), Output("status-time", "children"), Output("score-text", "children")],
    [Input("live-update", "n_intervals")],
)
def update_status(_):
    summary = get_detection_summary()
    return (
        summary["status"],
        f"Updated at {summary['timestamp']}",
        f"{summary['score']:.2f}",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
