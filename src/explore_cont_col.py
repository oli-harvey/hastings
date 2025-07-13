import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def explore_cont_col(df, cont_col, target_col='capped_incurred', date_col='date_of_loss', bins=30):
    df = df.copy()
    missing = df[cont_col].isna().sum()
    missing_pct = round(100 * missing / len(df), 1)
    plot_df = df[df[cont_col].notna()].copy()

    # --- Histogram
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=plot_df[cont_col],
        nbinsx=bins,
        marker_color='teal',
        name=cont_col
    ))
    fig_hist.update_layout(
        title=f"Distribution of {cont_col}",
        xaxis_title=cont_col,
        yaxis_title='Count',
        template='simple_white',
        height=350,
        width=700,
        margin=dict(t=50, b=40, l=70, r=30)
    )
    fig_hist.show()

    # --- Scatter Plot vs Target
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=plot_df[cont_col],
        y=plot_df[target_col],
        mode='markers',
        marker=dict(size=5, color='darkorange', opacity=0.6),
        showlegend=False
    ))
    fig_scatter.update_layout(
        title=f"{target_col} vs {cont_col}",
        xaxis_title=cont_col,
        yaxis_title=target_col,
        template='simple_white',
        height=350,
        width=700,
        margin=dict(t=50, b=40, l=70, r=30)
    )
    fig_scatter.show()

    # --- Time Series of Bucketed Values (optional)
    if date_col in plot_df.columns:
        plot_df[date_col] = pd.to_datetime(plot_df[date_col])
        plot_df['bucket'] = pd.cut(plot_df[cont_col], bins=bins)

        time_group = (
            plot_df.groupby([pd.Grouper(key=date_col, freq='ME'), 'bucket'], observed=True)
            .size()
            .reset_index(name='count')
        )

    print(f"ℹ️  Missing '{cont_col}': {missing} rows ({missing_pct}%)")