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
    S = len(np.unique(s))
    return N, S, s


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
    y[n] &\sim \mathcal{N}\big(\mu[n],\, \sigma_y\big), && \text{for } n = 1,\dots,N\\[6pt]
    \mu[n] &= \big(\alpha + a[s[n]]\big) \;+\; \big(\beta + b[s[n]]\big)\, x[n], && \text{for } n = 1,\dots,N\\[8pt]
    \log\sigma_y &\sim \mathcal{N}(0,2) \\[6pt]
    \alpha &\sim \mathcal{N}(0,1000) \\[4pt]
    \beta &\sim \mathcal{N}(0,1000) \\[8pt]
    a[s] &\sim \mathcal{N}(0, \sigma_a), && \text{for } s = 1,\dots,S\\[6pt]
    b[s] &\sim \mathcal{N}(0, \sigma_b), && \text{for } s = 1,\dots,S\\[8pt]
    \sigma_a &\sim \exp \mathcal{N}(0,2) \\[4pt]
    \sigma_b &\sim \exp \mathcal{N}(0,2)
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
def _(N, S, pi, s, x):
    sigma_a = pi.exp(pi.normal(0, 2))
    sigma_b = pi.exp(pi.normal(0, 2))

    a = [pi.normal(0, sigma_a) for s in range(S)]
    b = [pi.normal(0, sigma_b) for s in range(S)]

    alpha = pi.normal(0, 1000)
    beta = pi.normal(0, 1000)

    sigma_y = pi.exp(pi.normal(0, 2))

    mu = [alpha + a[s[n]] + (beta + b[s[n]]) * x[n] for n in range(N)]

    y = [pi.normal(mu[n], sigma_y) for n in range(N)]
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
