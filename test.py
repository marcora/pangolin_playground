import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Background Information

    In this notebook, we are exploring basic concepts of statistical analysis using the `pangolin` library. The variable `eps` is created as a normally distributed random variable, which is commonly used to introduce randomness and model uncertainty in statistical models.

    The relationship between the variables `x` and `y` is established with a linear equation where `y` is a function of `x` and `eps`. We also calculate the expected value (E) and variance (var) of `y`, which are fundamental statistical measures that help us understand the distribution and spread of data.

    This analysis can be useful in various applications, including regression analysis, forecasting, and understanding the nature of random processes.
    """)
    return


@app.cell
def _():
    import pangolin.interface as pi
    import pangolin as pg
    from pangolin.blackjax import E, var, sample
    import matplotlib.pyplot as plt
    import arviz as az
    return E, az, pi, sample, var


@app.cell
def _(pi):
    eps = pi.normal(0, 1)
    x = 4
    return eps, x


@app.cell
def _(eps, x):
    y = 2 + 3 * x + eps
    return (y,)


@app.cell
def _(E, y):
    E(y)
    return


@app.cell
def _(var, y):
    var(y)
    return


@app.cell
def _(az, sample, y):
    samples = sample(y)

    idata = az.from_dict(posterior={'y': samples})

    az.plot_trace(idata, var_names=['y'])
    return


if __name__ == "__main__":
    app.run()
