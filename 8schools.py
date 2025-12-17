import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pangolin import interface as pi
    from pangolin.blackjax import sample, E, var

    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt
    return np, pi, plt, sample, sns


@app.cell
def _():
    # data
    num_schools = 8
    observed = [28, 8, -3, 7, -1, 1, 18, 12]
    stddevs = [15, 10, 16, 11, 9, 11, 10, 18]
    return num_schools, observed, stddevs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The eight schools model was proposed by [Rubin in 1981](https://doi.org/10.2307/1164617).

    That there are eight schools, each of which ran some program to prepare students for the
    SAT. Each school estimated a treatment effect (given as `observed` above) and had
    some measurement error (given as `stddevs` above). (In Rubin's original paper these
    were real numbers, but at some point people turned them into integers, and I'll keep using
    integers for consistency.)

    It is assumed that the programs have different effectiveness in different schools. But probably
    not *that* different.

    The model is typically written like this:

    $$
    \begin{align}
    \mu \sim & \mathrm{Normal(0,10)} & \\
    \tau \sim & \mathrm{Lognormal(5,1)} & \\
    \theta_i \sim & \mathrm{Normal}(\mu,\tau),& \quad 1 \leq i\leq 8 \\
    y_i \sim & \mathrm{Normal}(\theta_i, \sigma_i),& \quad 1 \leq i\leq 8
    \end{align}
    $$

    Here $\mu$ represents the "typical" effectiveness of the program, while $\theta_i$ represents the
     actual effectivenes for school $i$ and $y_i$ is the measured effectivness with noise. $\tau$
     represents the variability of effectiveness from school to school.

    In Tensorflow Probability, the model is written in a [comically complicated way](https://www.tensorflow.org/probability/examples/Eight_Schools).
    """)
    return


@app.cell
def _(num_schools, pi, stddevs):
    # define model
    mu = pi.normal(0,10)
    tau = pi.exp(pi.normal(5,1))
    theta = [pi.normal(mu,tau) for i in range(num_schools)]
    y = [pi.normal(theta[i],stddevs[i]) for i in range(num_schools)]
    return theta, y


@app.cell
def _(pi, y):
    # mu, tau, theta[i], and y[i] are each scalar random variables.
    # if you call print_upstream() on any group of RVs, it will 
    # do a graph search and find everything above.

    pi.print_upstream(y)
    return


@app.cell
def _(observed, sample, theta, y):
    # do inference using pangolin's interface to blackjax
    theta_samps = sample(theta, y, observed, niter=10000)
    return (theta_samps,)


@app.cell
def _(theta_samps):
    # Since theta is a list of 8 scalar RVs, you get back a list of 8 length 10000 Numpy arrays,
    # each containing samples or 1 RV / school.
    [s.shape for s in theta_samps] == [(10000,)] * 8
    return


@app.cell
def _(plt, sns, theta_samps_1):
    # plot
    sns.swarmplot(theta_samps_1[::50, :], s=2)
    plt.xlabel('school')
    plt.ylabel('treatment effect')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice:
    * Schools with high observed effects get pulled down towards the overall mean (like school 0 with an observed effect of 28)
    * Schools with low observed effects get pulled up towards the overall mean (like school 2 with an observed effect of -3)
    * Schools with lower standard deviations have tighter posteriors. (School 4 has a stddev of 9, while school 7 has a stddev of 18)

    Bayes works!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Instead of unrolling everything, you can also define the model using vmap. This will be *much* more efficient in large models.
    """)
    return


@app.cell
def _(pi, stddevs):
    mu_1 = pi.normal(0, 10)
    tau_1 = pi.exp(pi.normal(5, 1))
    theta_1 = pi.vmap(pi.normal, None, axis_size=8)(mu_1, tau_1)
    y_1 = pi.vmap(pi.normal)(theta_1, pi.constant(stddevs))
    return mu_1, tau_1, theta_1, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, you can see that instead of everything being scalar, the internal representation involves three vector-valued RVs of length 8. To make things nicer, let's pass variable names when printing all the nodes.
    """)
    return


@app.cell
def _(mu_1, pi, tau_1, theta_1, y_1):
    pi.print_upstream(mu=mu_1, tau=tau_1, theta=theta_1, y=y_1)
    return


@app.cell
def _(np, observed, sample, theta_1, y_1):
    # inference
    theta_samps_1 = sample(theta_1, y_1, np.array(observed), niter=10000)
    return (theta_samps_1,)


@app.cell
def _(theta_samps_1):
    # now, the samples are one big array
    theta_samps_1.shape
    return


@app.cell
def _(plt, sns, theta_samps_1):
    # plot
    sns.swarmplot(theta_samps_1[::50, :], s=2)
    plt.xlabel('school')
    plt.ylabel('treatment effect')
    return


if __name__ == "__main__":
    app.run()
