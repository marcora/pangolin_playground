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

    from pangolin import interface as pi
    from pangolin.blackjax import sample, sample_arviz, E
    return az, np, pd, pg, pi


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    """)
    return


@app.cell
def _(pd):
    sleepstudy = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv")
    return (sleepstudy,)


@app.cell
def _(sleepstudy):
    sleepstudy.drop(columns=sleepstudy.columns[0], inplace=True)

    sleepstudy
    return


@app.cell
def _(sleepstudy):
    y_obs = sleepstudy["Reaction"].values.astype(float)
    x = sleepstudy["Days"].values.astype(float)
    subjects = sleepstudy["Subject"].values.astype(str)
    return subjects, x, y_obs


@app.cell
def _(np, subjects, y_obs):
    _, s = np.unique(subjects, return_inverse=True)
    N = len(y_obs)
    J = len(np.unique(s))
    return J, N, s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Classic random-intercept/random-slope model:

    $$
    \begin{aligned}
    y_i &\sim \mathcal{N}(\mu_i, \sigma^y) \\
    \mu_i &= (\alpha + a_{s[i]}) + (\beta + b_{s[i]}) \, x_i \\
    \sigma^y &= \exp(\mathcal{N}(0, 2)) \\
    \alpha &\sim \mathcal{N}(0, 1000) \\
    \beta &\sim \mathcal{N}(0, 1000) \\
    a_s &\sim \mathcal{N}(0, \sigma^a) \\
    b_s &\sim \mathcal{N}(0, \sigma^b) \\
    \sigma^a &= \exp(\mathcal{N}(0, 2)) \\
    \sigma^b &= \exp(\mathcal{N}(0, 2)) \\
    \end{aligned}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Priors and likelihood
    """)
    return


@app.cell
def _(J, N, pi, s, x):
    sigma_a = pi.exp(pi.normal(0, 2))
    sigma_b = pi.exp(pi.normal(0, 2))

    a = [pi.normal(0, sigma_a) for _ in range(J)]
    b = [pi.normal(0, sigma_b) for _ in range(J)]

    alpha = pi.normal(0, 1000)
    beta = pi.normal(0, 1000)

    sigma_y = pi.exp(pi.normal(0, 2))

    mu = [alpha + a[s[i]] + (beta + b[s[i]]) * x[i] for i in range(N)]

    y = [pi.normal(mu[i], sigma_y) for i in range(N)]
    return alpha, beta, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inference
    """)
    return


@app.cell
def _(alpha, beta, pg, y, y_obs):
    samples = pg.blackjax.sample((alpha, beta), y, y_obs.tolist())
    return


@app.cell
def _(alpha, beta, pg, y, y_obs):
    idata = pg.blackjax.sample_arviz({'alpha':alpha, 'beta':beta}, y, y_obs.tolist())
    return (idata,)


@app.cell
def _(az, idata):
    az.summary(idata)
    return


@app.cell
def _(az, idata):
    az.plot_trace(idata)
    return


if __name__ == "__main__":
    app.run()
