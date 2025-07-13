import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def explore_cat_col(df, cat_col, target_col='capped_incurred', date_col='date_of_loss'):
    df = df.copy()
    missing = df[cat_col].isna().sum()
    missing_pct = round(100 * missing / len(df), 1)
    plot_df = df[df[cat_col].notna()].copy()
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    
    counts = plot_df[cat_col].value_counts().sort_values()
    order = counts.index.tolist()
    order_str = list(map(str, order))

    # --- % missing_target per category
    if 'missing_target' in df.columns:
        pct_missing_target = (
            plot_df.groupby(cat_col)['missing_target']
            .mean()
            .reindex(order) * 100
        )
    else:
        pct_missing_target = pd.Series(0, index=order)

    # --- Side-by-side bar chart: Counts + % missing_target
    fig_bar = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.12,
                            subplot_titles=[f"{cat_col} Counts", f"% {target_col} Missing"])

    fig_bar.add_trace(go.Bar(
        x=counts.values,
        y=order_str,
        orientation='h',
        marker_color='royalblue',
        showlegend=False
    ), row=1, col=1)

    fig_bar.add_trace(go.Bar(
        x=pct_missing_target.values,
        y=order_str,
        orientation='h',
        marker_color='crimson',
        showlegend=False
    ), row=1, col=2)

    fig_bar.update_layout(
        height=400,
        width=800,
        template='simple_white',
        margin=dict(t=60, b=40, l=70, r=30),
        xaxis_title='Count',
        xaxis2_title='% Missing Target',
        yaxis_title=cat_col,
        title_text=f"{cat_col} Overview",
        title_x=0.5
    )
    fig_bar.show()

    # --- Violin Plot ---
    go.Figure(go.Violin(
        y=plot_df[cat_col].astype(str),
        x=plot_df[target_col],
        orientation='h',
        line_color='blue',
        fillcolor='rgba(0,0,150,0.05)',
        meanline_visible=True,
        showlegend=False
    )).update_layout(
        title=f"{target_col} by {cat_col}",
        height=350,
        width=650,
        template='simple_white',
        margin=dict(t=50, b=40, l=70, r=30),
        xaxis_title=target_col,
        yaxis_title=cat_col,
        yaxis=dict(categoryorder='array', categoryarray=order_str),
        showlegend=False
    ).show()

    # --- Time Series Plot ---
    time_group = plot_df.groupby(
        [pd.Grouper(key=date_col, freq='ME'), cat_col], observed=True
    ).size().reset_index(name='count')

    fig3 = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, val in enumerate(order):
        sub = time_group[time_group[cat_col] == val]
        fig3.add_trace(go.Scatter(
            x=sub[date_col],
            y=sub['count'],
            mode='lines',
            name=str(val),
            line=dict(color=palette[i % len(palette)]),
            showlegend=True
        ))

    fig3.update_layout(
        title=f"{cat_col} Frequency Over Time",
        height=400,
        width=700,
        template='simple_white',
        margin=dict(t=50, b=40, l=70, r=30),
        xaxis_title=date_col,
        yaxis_title='Monthly Count',
        showlegend=True
    )
    fig3.show()

    print(f"ℹ️  Missing '{cat_col}': {missing} rows ({missing_pct}%)")