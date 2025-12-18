# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pangolin as pg
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from pangolin import interface as pi
    from pangolin.blackjax import sample, E, var, std
    from matplotlib import pyplot as plt
    return np, pd, pi, sample


@app.cell
def _(pd):
    sleepstudy = pd.read_csv(
        "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv"
    )

    sleepstudy.head()
    return (sleepstudy,)


@app.cell
def _(sleepstudy):
    sleepstudy.drop(columns=sleepstudy.columns[0], inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    classic random-intercept + random-slope model:

    $$
    \begin{aligned}
    y_i &\sim \mathcal{N}(\mu_i, \sigma_y) \\
    \mu_i &= (\beta_0 + a_{s_i}) + (\beta_1 + b_{s_i}) \, x_i \\
    \beta_0 &\sim \mathcal{N}(0, 1000) \\
    \beta_1 &\sim \mathcal{N}(0, 1000) \\
    a_s &\sim \mathcal{N}(0, \sigma_a) \\
    b_s &\sim \mathcal{N}(0, \sigma_b) \\
    \sigma_a &= \exp(\mathcal{N}(0, 2)) \\
    \sigma_b &= \exp(\mathcal{N}(0, 2)) \\
    \sigma_y &= \exp(\mathcal{N}(0, 2)) \\
    \end{aligned}
    $$
    """)
    return


@app.cell
def _(sleepstudy):
    # J subjects
    J = len(sleepstudy["Subject"].unique())

    # N observations
    N = sleepstudy.shape[0]

    print(f"N = {N}, J = {J}")
    print(sleepstudy.head())
    return J, N


@app.cell
def _(J, N, np, pi, sample, sleepstudy):
    y_obs = sleepstudy["Reaction"].values.astype(float)
    x = sleepstudy["Days"].values.astype(float)
    subjects = sleepstudy["Subject"].values

    _, s = np.unique(subjects, return_inverse=True)

    sigma_y = pi.exp(pi.normal(0.0, 2.0))
    sigma_a = pi.exp(pi.normal(0.0, 2.0))
    sigma_b = pi.exp(pi.normal(0.0, 2.0))

    a = []
    b = []

    for j in range(J):
        a.append(pi.normal(0.0, sigma_a))
        b.append(pi.normal(0.0, sigma_b))

    beta0 = pi.normal(0.0, 1000.0)
    beta1 = pi.normal(0.0, 1000.0)

    y = []
    y_pred = []

    for i in range(N):
        mu = (beta0 + a[s[i]]) + (beta1 + b[s[i]]) * x[i]
        y.append(pi.normal(mu, sigma_y))
        y_pred.append(pi.normal(mu, sigma_y))

    samples = sample(
        y_pred,
        y,
        y_obs.tolist(),
        niter=1000
    )
    return (samples,)


@app.cell
def _(pd, samples):
    df = pd.DataFrame({f"param_{i}": samples[i] for i in range(len(samples))})

    df.head()
    return


if __name__ == "__main__":
    app.run()
