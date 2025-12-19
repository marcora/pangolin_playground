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
    import arviz as az

    from matplotlib import pyplot as plt
    from pangolin import interface as pi
    from pangolin.blackjax import sample, sample_arviz, E
    return np, pd, pg, pi, plt, sns


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
def _(np, pg, pi, sleepstudy):
    # data -------------------------------------------------------
    y_obs = sleepstudy["Reaction"].values.astype(float)
    x     = sleepstudy["Days"].values.astype(float)
    subj  = sleepstudy["Subject"].values.astype(str)

    _, s  = np.unique(subj, return_inverse=True)
    N     = len(y_obs)
    J     = len(np.unique(s))

    # priors -----------------------------------------------------
    beta0 = pi.normal(0., 1000.)
    beta1 = pi.normal(0., 1000.)

    sigma_y = pi.exp(pi.normal(0., 2.))
    sigma_a = pi.exp(pi.normal(0., 2.))
    sigma_b = pi.exp(pi.normal(0., 2.))

    a = [pi.normal(0., sigma_a) for _ in range(J)]
    b = [pi.normal(0., sigma_b) for _ in range(J)]

    # mu linpred ---------------------------------------------
    mu = [
        beta0 + a[s[i]] + (beta1 + b[s[i]]) * x[i]
        for i in range(N)
    ]

    # likelihood --------------------------------------------------
    y      = [pi.normal(mu[i], sigma_y) for i in range(N)]

    # posterior predictive ----------------------------------------
    y_pred = [pi.normal(mu[i], sigma_y) for i in range(N)]

    # inference ---------------------------------------------------
    params  = y_pred
    samples = pg.blackjax.sample(params, y, y_obs.tolist())
    return (samples,)


@app.cell
def _(plt, samples, sns):
    [sns.kdeplot(sample, legend=False, alpha=0.1) for sample in samples]
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
