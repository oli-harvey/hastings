import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def explore_count_col(df, count_cols, target_col='capped_incurred'):
    df = df.copy()
    sub_df = df[count_cols + [target_col]]

    # --- A. Value counts per column ---
    fig_counts = make_subplots(
        rows=1, cols=len(count_cols),
        subplot_titles=count_cols,
        horizontal_spacing=0.03
    )

    for i, col in enumerate(count_cols, start=1):
        vc = sub_df[col].value_counts().sort_index()
        fig_counts.add_trace(go.Bar(
            x=vc.values,
            y=vc.index.astype(str),
            orientation='h',
            showlegend=False
        ), row=1, col=i)

    fig_counts.update_layout(
        height=300,
        width=max(350, 160 * len(count_cols)),
        title_text="Value Counts per Column",
        template='simple_white',
        margin=dict(t=60, b=40, l=30, r=30)
    )
    fig_counts.show()

    # --- B. Mean target per column value ---
    fig_means = make_subplots(
        rows=1, cols=len(count_cols),
        subplot_titles=count_cols,
        horizontal_spacing=0.03
    )

    for i, col in enumerate(count_cols, start=1):
        means = sub_df.groupby(col, observed=True)[target_col].mean().sort_index()
        fig_means.add_trace(go.Bar(
            x=means.values,
            y=means.index.astype(str),
            orientation='h',
            marker_color='indianred',
            showlegend=False
        ), row=1, col=i)

    fig_means.update_layout(
        height=300,
        width=max(350, 160 * len(count_cols)),
        title_text=f"Mean {target_col} per Column Value",
        template='simple_white',
        margin=dict(t=60, b=40, l=30, r=30)
    )
    fig_means.show()

    # --- C. Distribution of # nonzero columns per row ---
    count_nonzero = (sub_df[count_cols] > 0).sum(axis=1)
    count_dist = count_nonzero.value_counts().sort_index()
    mean_count = count_nonzero.mean()

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=count_dist.index,
        y=count_dist.values,
        marker_color='steelblue'
    ))
    fig_dist.add_trace(go.Scatter(
        x=[mean_count],
        y=[count_dist.get(round(mean_count), 0)],
        mode='markers+text',
        text=[f"Mean: {mean_count:.2f}"],
        textposition='top center',
        marker=dict(color='orange', size=10)
    ))

    fig_dist.update_layout(
        title="# Rows by Count of Group Columns > 0",
        height=350,
        width=600,
        template='simple_white',
        margin=dict(t=60, b=40, l=60, r=30),
        xaxis_title="# Columns > 0",
        yaxis_title="Row Count",
        showlegend=False
    )
    fig_dist.show()