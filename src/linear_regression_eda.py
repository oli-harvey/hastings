import pandas as pd
import statsmodels.api as sm

def linear_regression_eda(
    df
  , cat_col=None
  , count_cols=None
  , cont_cols=None
  , target_col='capped_incurred'
  , recent=False
  , show_chart=True
):
    all_cols = [target_col, 'date_of_loss']
    if cat_col is not None:
        all_cols.append(cat_col)
    if count_cols is not None:
        all_cols.extend(count_cols)
    if cont_cols is not None:
        all_cols.extend(cont_cols)

    df = df[all_cols].copy()
    df = df.dropna()

    if recent:
        df = df[pd.to_datetime(df['date_of_loss']) >= pd.Timestamp('2012-01-01')]

    X_parts = []

    # Handle categorical column
    if cat_col is not None:
        df[cat_col] = df[cat_col].astype(str)
        ref = df[cat_col].mode().iloc[0]
        dummies = pd.get_dummies(df[cat_col])
        dummies = dummies.drop(columns=[ref])
        X_parts.append(dummies)
    else:
        ref = None

    # Handle numeric columns
    if count_cols is not None:
        X_parts.append(df[count_cols].astype(float))

    if cont_cols is not None:
        X_parts.append(df[cont_cols].astype(float))

    if not X_parts:
        raise ValueError("At least one of cat_col, count_cols, or cont_cols must be provided.")

    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X)

    y = pd.to_numeric(df[target_col], errors='coerce')
    valid_idx = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid_idx].astype(float)
    y = y.loc[valid_idx].astype(float)

    model = sm.OLS(y, X).fit()

    if show_chart:
        print(f"\nLinear Regression vs '{target_col}'" + (" (since 2012)" if recent else "") + ":")
        if cat_col is not None:
            print(f"  Reference category (intercept group): '{ref}'\n")
        print(model.summary())

    return model.rsquared
