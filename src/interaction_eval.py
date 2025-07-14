import pandas as pd
import plotly.express as px
import numpy as np

def interaction_eval(df, target, col_a, col_b, show_chart=False):
    # Form interaction group
    df = df.copy()
    df['interaction'] = df[col_a].astype(str) + ' | ' + df[col_b].astype(str)

    # Compute group-level target stats
    group_stats = (
        df.groupby('interaction')[target]
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values('mean', ascending=False)
    )

    # Plot
    fig = px.bar(
        group_stats,
        x='interaction',
        y='mean',
        color='count',
        title=f'{col_a} × {col_b} vs {target}',
        labels={'mean': f'Mean {target}', 'interaction': 'Interaction'},
        height=400
    )
    if show_chart:
        fig.show()

    # Heuristic: high variance and enough groups with count > threshold
    mean_var = group_stats['mean'].var()
    valid_groups = (group_stats['count'] >= 10).sum()
    unique_vals = df['interaction'].nunique()

    return mean_var > 0.01 and valid_groups > min(5, unique_vals // 2)