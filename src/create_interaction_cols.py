import pandas as pd
import itertools

def create_interaction_cols(df, count_cols=None, cont_cols=None, cat_cols=None):
    count_cols = count_cols or []
    cont_cols = cont_cols or []
    cat_cols = cat_cols or []

    df = df.copy()

    num_cols = count_cols + cont_cols
    interaction_df = pd.DataFrame(index=df.index)

    # Numeric x Numeric
    for col1, col2 in itertools.combinations(num_cols, 2):
        interaction_df[f'{col1}_x_{col2}'] = df[col1].astype(float) * df[col2].astype(float)

    # Categorical x Numeric
    for cat in cat_cols:
        cat_vals = df[cat].astype(str)
        for val in cat_vals.unique():
            mask = (cat_vals == val).astype(int)
            for num in num_cols:
                interaction_df[f'{cat}_{val}_x_{num}'] = mask * df[num].astype(float)

    # Categorical x Categorical
    for cat1, cat2 in itertools.combinations(cat_cols, 2):
        c1_vals = df[cat1].astype(str)
        c2_vals = df[cat2].astype(str)
        for v1 in c1_vals.unique():
            for v2 in c2_vals.unique():
                interaction_df[f'{cat1}_{v1}_x_{cat2}_{v2}'] = ((c1_vals == v1) & (c2_vals == v2)).astype(int)

    return interaction_df