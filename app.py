"""
app.py — Interactive Epidemic Dashboard
========================================
Built with Plotly Dash for CovidSense India.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import json

from model import (load_india_data, calculate_daily_new_cases,
                   train_model, predict_next_7_days,
                   get_trend, get_risk_level)


# ==============================================================
# DATA LOADING
# ==============================================================

def load_dashboard_data():
    data = {}

    total = load_india_data()
    daily = calculate_daily_new_cases(total)
    data['daily_cases'] = daily

    model = train_model(daily)
    predictions = predict_next_7_days(model, daily)
    data['predictions'] = predictions

    try:
        data['hotspot_df'] = pd.read_csv(
            "data/hotspot_results.csv"
        )
    except FileNotFoundError:
        data['hotspot_df'] = create_dummy_state_data()

    try:
        vax_df = pd.read_csv("data/vaccine_data.csv")
        india_vax = vax_df[
            vax_df['location'] == 'India'
        ][['date', 'daily_vaccinations']].dropna()
        data['vaccine_df'] = india_vax
    except FileNotFoundError:
        data['vaccine_df'] = None

    return data


def create_dummy_state_data():
    states = [
        "Maharashtra", "Kerala", "Karnataka",
        "Tamil Nadu", "Delhi", "Uttar Pradesh",
        "West Bengal", "Rajasthan", "Gujarat",
        "Andhra Pradesh", "Telangana",
        "Madhya Pradesh", "Bihar",
        "Punjab", "Haryana"
    ]
    np.random.seed(42)
    df = pd.DataFrame({
        "State": states,
        "Severity": np.random.uniform(
            20, 90, len(states)
        ).round(1),
        "Risk": np.random.choice(
            ["High", "Medium", "Low"], len(states)
        ),
        "Change_Pct": np.random.uniform(
            -20, 60, len(states)
        ).round(1),
        "Recent_Avg": np.random.uniform(
            100, 5000, len(states)
        ).round(1),
        "Future_Avg": np.random.uniform(
            100, 6000, len(states)
        ).round(1),
        "Color": [
            "#FF4444" if s >= 70
            else "#FFA500" if s >= 40
            else "#44BB44"
            for s in np.random.uniform(20, 90, len(states))
        ]
    })
    return df.sort_values("Severity", ascending=False)


# ==============================================================
# GRAPH FUNCTIONS
# ==============================================================

def create_india_map(hotspot_df):
    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        geojson="https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
        locations=hotspot_df['State'],
        z=hotspot_df['Severity'],
        featureidkey="properties.ST_NM",
        colorscale=[
            [0, "#44BB44"],
            [0.4, "#FFA500"],
            [0.7, "#FF4444"],
            [1.0, "#8B0000"]
        ],
        zmin=0, zmax=100,
        marker_line_color='white',
        marker_line_width=0.5,
        colorbar_title="Severity<br>Index",
        hovertemplate=(
            "<b>%{location}</b><br>"
            "Severity: %{z:.1f}<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title={
            'text': '🇮🇳 India COVID Risk Map',
            'x': 0.5,
            'font': {'size': 18, 'color': 'white'}
        },
        geo=dict(
            scope='asia',
            showframe=False,
            showcoastlines=True,
            projection_type='mercator',
            center=dict(lat=22, lon=82),
            projection_scale=4,
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font_color='white',
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


def create_forecast_chart(daily_cases,
                           predictions,
                           state_name="India"):
    last_30 = daily_cases[-30:]
    actual_x = list(range(-29, 1))
    pred_x = list(range(1, 8))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_x,
        y=[int(x) for x in last_30],
        name='Actual Cases',
        line=dict(color='#4488FF', width=2),
        mode='lines+markers',
        marker=dict(size=4),
        hovertemplate='Day %{x}: %{y:,} cases<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=pred_x,
        y=predictions,
        name='Predicted Cases',
        line=dict(color='#FF4444', width=2,
                  dash='dash'),
        mode='lines+markers',
        marker=dict(size=6, symbol='diamond'),
        hovertemplate='Day +%{x}: %{y:,} cases<extra></extra>'
    ))

    fig.add_vline(
        x=0,
        line_dash="dot",
        line_color="yellow",
        annotation_text="Today",
        annotation_font_color="yellow"
    )

    fig.update_layout(
        title={
            'text': f'📈 7-Day Forecast — {state_name}',
            'x': 0.5,
            'font': {'size': 16, 'color': 'white'}
        },
        xaxis_title='Days (0 = Today)',
        yaxis_title='Daily New Cases',
        paper_bgcolor='#16213e',
        plot_bgcolor='#0f3460',
        font_color='white',
        height=350,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            font_color='white'
        ),
        hovermode='x unified'
    )

    return fig


def create_severity_gauge(severity_score,
                           state_name="India"):
    if severity_score >= 70:
        color = "#FF4444"
        label = "CRITICAL"
    elif severity_score >= 40:
        color = "#FFA500"
        label = "WATCH"
    else:
        color = "#44BB44"
        label = "STABLE"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=severity_score,
        title={
            'text': f"Severity Index<br>{state_name}",
            'font': {'color': 'white'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': 'white'
            },
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40],
                 'color': '#1a4a1a'},
                {'range': [40, 70],
                 'color': '#4a3a1a'},
                {'range': [70, 100],
                 'color': '#4a1a1a'}
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 3},
                'thickness': 0.75,
                'value': severity_score
            }
        },
        number={'font': {'color': 'white', 'size': 40}},
    ))

    fig.add_annotation(
        x=0.5, y=0.2,
        text=f"<b>{label}</b>",
        font=dict(size=20, color=color),
        showarrow=False
    )

    fig.update_layout(
        paper_bgcolor='#1a1a2e',
        font_color='white',
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_wave_timeline(daily_cases):
    fig = go.Figure()

    x = list(range(len(daily_cases)))
    fig.add_trace(go.Scatter(
        x=x,
        y=[int(c) for c in daily_cases],
        fill='tozeroy',
        name='Daily Cases',
        line=dict(color='#4488FF', width=1),
        fillcolor='rgba(68, 136, 255, 0.3)',
        hovertemplate='Day %{x}: %{y:,} cases<extra></extra>'
    ))

    waves = [
        dict(x=150, label="Wave 1",
             color="#FFD700"),
        dict(x=430, label="Wave 2\n(Delta)",
             color="#FF4444"),
        dict(x=660, label="Wave 3\n(Omicron)",
             color="#FF8C00")
    ]

    for wave in waves:
        if wave['x'] < len(daily_cases):
            fig.add_vline(
                x=wave['x'],
                line_dash="dash",
                line_color=wave['color'],
                opacity=0.7
            )
            fig.add_annotation(
                x=wave['x'],
                y=max(daily_cases) * 0.9,
                text=wave['label'],
                font=dict(
                    color=wave['color'],
                    size=11
                ),
                showarrow=False,
                bgcolor='rgba(0,0,0,0.5)'
            )

    fig.update_layout(
        title={
            'text': '📊 India COVID Wave History',
            'x': 0.5,
            'font': {'size': 16, 'color': 'white'}
        },
        xaxis_title='Days since Jan 22, 2020',
        yaxis_title='Daily New Cases',
        paper_bgcolor='#16213e',
        plot_bgcolor='#0f3460',
        font_color='white',
        height=350,
        showlegend=False
    )

    return fig


def create_vaccination_chart(daily_cases,
                              vaccine_df):
    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(daily_cases[-400:]))),
            y=[int(c) for c in daily_cases[-400:]],
            name='Daily Cases',
            line=dict(color='#FF4444', width=2)
        ),
        secondary_y=False
    )

    if vaccine_df is not None and \
       not vaccine_df.empty:
        fig.add_trace(
            go.Scatter(
                x=list(range(
                    len(vaccine_df[-400:])
                )),
                y=vaccine_df[
                    'daily_vaccinations'
                ].tail(400).tolist(),
                name='Daily Vaccinations',
                line=dict(
                    color='#44BB44',
                    width=2,
                    dash='dot'
                )
            ),
            secondary_y=True
        )
        title = '💉 Vaccination vs Cases'
    else:
        title = '💉 Daily Cases Timeline'

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'font': {'size': 15, 'color': 'white'}
        },
        paper_bgcolor='#16213e',
        plot_bgcolor='#0f3460',
        font_color='white',
        height=350,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)'
        )
    )

    fig.update_yaxes(
        title_text="Daily Cases",
        secondary_y=False,
        color='#FF4444'
    )
    fig.update_yaxes(
        title_text="Daily Vaccinations",
        secondary_y=True,
        color='#44BB44'
    )

    return fig


def create_top_states_bar(hotspot_df):
    top5 = hotspot_df.head(5)

    fig = go.Figure(go.Bar(
        x=top5['Severity'],
        y=top5['State'],
        orientation='h',
        marker_color=top5['Color'],
        text=[f"{s:.0f}" for s in
              top5['Severity']],
        textposition='outside',
        hovertemplate=(
            '%{y}: Severity %{x:.1f}'
            '<extra></extra>'
        )
    ))

    fig.update_layout(
        title={
            'text': '🔴 Top 5 High Risk States',
            'x': 0.5,
            'font': {'size': 15, 'color': 'white'}
        },
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font_color='white',
        height=300,
        xaxis=dict(
            range=[0, 110],
            title='Severity Index'
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=120, r=60, t=50, b=40)
    )

    return fig


# ==============================================================
# DASH APP LAYOUT
# ==============================================================

app = dash.Dash(
    __name__,
    title="CovidSense India",
    meta_tags=[{
        "name": "viewport",
        "content": "width=device-width, initial-scale=1"
    }]
)

print("Loading dashboard data...")
DASHBOARD_DATA = load_dashboard_data()
print("Data loaded successfully!")

daily = DASHBOARD_DATA['daily_cases']
preds = DASHBOARD_DATA['predictions']
recent_avg = sum(daily[-7:]) / 7
future_avg = sum(preds) / 7
change = ((future_avg - recent_avg) /
           max(recent_avg, 1)) * 100
national_severity = min(100, max(0, 50 + change))

app.layout = html.Div(
    style={
        'backgroundColor': '#0d0d1a',
        'minHeight': '100vh',
        'fontFamily': '"Segoe UI", sans-serif',
        'color': 'white',
        'padding': '20px'
    },
    children=[

        # HEADER
        html.Div(
            style={
                'textAlign': 'center',
                'padding': '20px',
                'borderBottom': '2px solid #4488FF',
                'marginBottom': '20px'
            },
            children=[
                html.H1(
                    "🦠 CovidSense India",
                    style={
                        'color': '#4488FF',
                        'fontSize': '2.5rem',
                        'margin': '0'
                    }
                ),
                html.P(
                    "Epidemic Spread Intelligence — Track C",
                    style={
                        'color': '#aaaaaa',
                        'margin': '5px 0 0 0'
                    }
                ),
                html.Div(
                    f"📅 {len(daily)} days analyzed | "
                    f"36 states monitored",
                    style={
                        'color': '#666',
                        'fontSize': '0.85rem',
                        'marginTop': '5px'
                    }
                )
            ]
        ),

        # ROW 1: Map + Gauge + Top States
        html.Div(
            style={
                'display': 'flex',
                'gap': '20px',
                'marginBottom': '20px'
            },
            children=[
                html.Div(
                    style={
                        'flex': '2',
                        'backgroundColor': '#1a1a2e',
                        'borderRadius': '12px',
                        'padding': '10px'
                    },
                    children=[
                        dcc.Graph(
                            id='india-map',
                            figure=create_india_map(
                                DASHBOARD_DATA['hotspot_df']
                            ),
                            config={
                                'displayModeBar': False
                            }
                        )
                    ]
                ),
                html.Div(
                    style={
                        'flex': '1',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'gap': '20px'
                    },
                    children=[
                        html.Div(
                            style={
                                'backgroundColor': '#1a1a2e',
                                'borderRadius': '12px',
                                'padding': '10px'
                            },
                            children=[
                                dcc.Graph(
                                    id='severity-gauge',
                                    figure=create_severity_gauge(
                                        round(
                                            national_severity,
                                            1
                                        ),
                                        "India"
                                    ),
                                    config={
                                        'displayModeBar': False
                                    }
                                )
                            ]
                        ),
                        html.Div(
                            style={
                                'backgroundColor': '#1a1a2e',
                                'borderRadius': '12px',
                                'padding': '10px'
                            },
                            children=[
                                dcc.Graph(
                                    id='top-states-bar',
                                    figure=create_top_states_bar(
                                        DASHBOARD_DATA['hotspot_df']
                                    ),
                                    config={
                                        'displayModeBar': False
                                    }
                                )
                            ]
                        )
                    ]
                )
            ]
        ),

        # STATE SELECTOR
        html.Div(
            style={'marginBottom': '15px'},
            children=[
                html.Label(
                    "🔍 Select State for Forecast:",
                    style={
                        'color': '#aaaaaa',
                        'marginBottom': '8px',
                        'display': 'block'
                    }
                ),
                dcc.Dropdown(
                    id='state-dropdown',
                    options=[
                        {
                            'label': 'India (National)',
                            'value': 'India'
                        }
                    ] + [
                        {
                            'label': row['State'],
                            'value': row['State']
                        }
                        for _, row in
                        DASHBOARD_DATA[
                            'hotspot_df'
                        ].iterrows()
                    ],
                    value='India',
                    style={
                        'backgroundColor': '#1a1a2e',
                        'color': 'white',
                        'border': '1px solid #4488FF',
                        'borderRadius': '8px'
                    }
                )
            ]
        ),

        # ROW 2: Forecast Chart
        html.Div(
            style={
                'backgroundColor': '#16213e',
                'borderRadius': '12px',
                'padding': '10px',
                'marginBottom': '20px'
            },
            children=[
                dcc.Graph(
                    id='forecast-chart',
                    figure=create_forecast_chart(
                        DASHBOARD_DATA['daily_cases'],
                        DASHBOARD_DATA['predictions'],
                        "India"
                    ),
                    config={'displayModeBar': True}
                )
            ]
        ),

        # ROW 3: Wave Timeline
        html.Div(
            style={
                'backgroundColor': '#16213e',
                'borderRadius': '12px',
                'padding': '10px',
                'marginBottom': '20px'
            },
            children=[
                dcc.Graph(
                    id='wave-timeline',
                    figure=create_wave_timeline(
                        DASHBOARD_DATA['daily_cases']
                    ),
                    config={'displayModeBar': True}
                )
            ]
        ),

        # ROW 4: Vaccination Chart
        html.Div(
            style={
                'backgroundColor': '#16213e',
                'borderRadius': '12px',
                'padding': '10px',
                'marginBottom': '20px'
            },
            children=[
                dcc.Graph(
                    id='vaccination-chart',
                    figure=create_vaccination_chart(
                        DASHBOARD_DATA['daily_cases'],
                        DASHBOARD_DATA['vaccine_df']
                    ),
                    config={'displayModeBar': True}
                )
            ]
        ),

        # FOOTER
        html.Div(
            style={
                'textAlign': 'center',
                'color': '#555',
                'borderTop': '1px solid #333',
                'paddingTop': '15px'
            },
            children=[
                html.P(
                    "CovidSense India | "
                    "CodeCure Biohackathon | Track C"
                ),
                html.P(
                    "Data: JHU CSSE + OWID | "
                    "Model: Linear Regression + Log Transform"
                )
            ]
        )
    ]
)


# ==============================================================
# CALLBACKS
# ==============================================================

@app.callback(
    [Output('forecast-chart', 'figure'),
     Output('severity-gauge', 'figure')],
    [Input('state-dropdown', 'value')]
)
def update_state_view(selected_state):
    hotspot_df = DASHBOARD_DATA['hotspot_df']

    if selected_state == 'India' or \
       selected_state is None:
        forecast = create_forecast_chart(
            DASHBOARD_DATA['daily_cases'],
            DASHBOARD_DATA['predictions'],
            "India"
        )
        gauge = create_severity_gauge(
            round(national_severity, 1),
            "India"
        )
    else:
        state_row = hotspot_df[
            hotspot_df['State'] == selected_state
        ]

        if state_row.empty:
            forecast = create_forecast_chart(
                DASHBOARD_DATA['daily_cases'],
                DASHBOARD_DATA['predictions'],
                selected_state
            )
            gauge = create_severity_gauge(
                50.0, selected_state
            )
        else:
            row = state_row.iloc[0]

            try:
                preds_str = row['Predictions']
                if isinstance(preds_str, str):
                    preds = [
                        int(x) for x in
                        preds_str.strip('[]').split(',')
                    ]
                else:
                    preds = DASHBOARD_DATA['predictions']
            except Exception:
                preds = DASHBOARD_DATA['predictions']

            forecast = create_forecast_chart(
                DASHBOARD_DATA['daily_cases'],
                preds,
                selected_state
            )
            gauge = create_severity_gauge(
                float(row['Severity']),
                selected_state
            )

    return forecast, gauge


# ==============================================================
# RUN
# ==============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CovidSense India Dashboard")
    print("  Open: http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8050)