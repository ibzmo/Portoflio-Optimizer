This a simpler version of Robert Martin's PyPortfolioOpt[1], partly for use by Imperial's Student investment Fund and also as part of my self-study of quantitative finance.

Like the original, I utilised ‘cvxpy’ (a Python embedded modelling language) for convex optimization problems as well as integrating it with ‘pandas’ data structures.

This is also coupled with Jupiter notebook tutorials which teach users how to utilise the library whilst also providing a foundational understanding of classical portfolio optimization methods such as Black-Litterman and Markovitz.



[1] - Martin, R. A., (2021). PyPortfolioOpt: portfolio optimization in Python. Journal of Open Source Software, 6(61), 3066, https://doi.org/10.21105/joss.03066


################
Expected Returns
################
The Features of the optimzer are as follows:

################
Expected Returns
################

Mean-variance optimization requires knowledge of the expected returns. In practice,
these are rather difficult to know with any certainty. Thus the best we can do is to
come up with estimates, for example by extrapolating historical data, This is the
main flaw in mean-variance optimization – the optimization procedure is sound, and provides
strong mathematical guarantees, *given the correct inputs*. This is one of the reasons
why I have emphasised modularity: users should be able to come up with their own
superior models and feed them into the optimizer.

.. caution::

    Supplying expected returns can do more harm than good. If
    predicting stock returns were as easy as calculating the mean historical return,
    we'd all be rich! For most use-cases, I would suggest that you focus your efforts
    on choosing an appropriate risk model (see :ref:`risk-models`). 

    As of v0.5.0, you can use :ref:`black-litterman` to significantly improve the quality of
    your estimate of the expected returns.

.. automodule:: pypfopt.expected_returns

    .. note::

        For any of these methods, if you would prefer to pass returns (the default is prices),
        set the boolean flag ``returns_data=True``

    .. autofunction:: mean_historical_return

        This is probably the default textbook approach. It is intuitive and easily interpretable,
        however the estimates are subject to large uncertainty. This is a problem especially in the
        context of a mean-variance optimizer, which will maximise the erroneous inputs.


    .. autofunction:: ema_historical_return

        The exponential moving average is a simple improvement over the mean historical
        return; it gives more credence to recent returns and thus aims to increase the relevance
        of the estimates. This is parameterised by the ``span`` parameter, which gives users
        the ability to decide exactly how much more weight is given to recent data.
        Generally, I would err on the side of a higher span – in the limit, this tends towards
        the mean historical return. However, if you plan on rebalancing much more frequently,
        there is a case to be made for lowering the span in order to capture recent trends.

    .. autofunction:: capm_return

    .. autofunction:: returns_from_prices

    .. autofunction:: prices_from_returns


.. References
.. ==========

###########
Risk Models
###########

In addition to the expected returns, mean-variance optimization requires a
**risk model**, some way of quantifying asset risk. The most commonly-used risk model
is the covariance matrix, which describes asset volatilities and their co-dependence. This is
important because one of the principles of diversification is that risk can be
reduced by making many uncorrelated bets (correlation is just normalised
covariance).

.. image:: ../media/corrplot.png
   :align: center
   :width: 60%
   :alt: plot of the covariance matrix


In many ways, the subject of risk models is far more important than that of
expected returns because historical variance is generally a much more persistent
statistic than mean historical returns. In fact, research by Kritzman et
al. (2010) [1]_ suggests that minimum variance portfolios, formed by optimising
without providing expected returns, actually perform much better out of sample.

The problem, however, is that in practice we do not have access to the covariance
matrix (in the same way that we don't have access to expected returns) – the only
thing we can do is to make estimates based on past data. The most straightforward
approach is to just calculate the **sample covariance matrix** based on historical
returns, but relatively recent (post-2000) research indicates that there are much
more robust statistical estimators of the covariance matrix. In addition to
providing a wrapper around the estimators in ``sklearn``, PyPortfolioOpt
provides some experimental alternatives such as semicovariance and exponentially weighted
covariance.

.. attention::

    Estimation of the covariance matrix is a very deep and actively-researched
    topic that involves statistics, econometrics, and numerical/computational
    approaches. PyPortfolioOpt implements several options, but there is a lot of room
    for more sophistication.


.. automodule:: pypfopt.risk_models

    .. note::

        For any of these methods, if you would prefer to pass returns (the default is prices),
        set the boolean flag ``returns_data=True``

    .. autofunction:: risk_matrix

    .. autofunction:: fix_nonpositive_semidefinite

        Not all the calculated covariance matrices will be positive semidefinite (PSD). This method
        checks if a matrix is PSD and fixes it if not.

    .. autofunction:: sample_cov

        This is the textbook default approach. The
        entries in the sample covariance matrix (which we denote as *S*) are the sample
        covariances between the *i* th and *j* th asset (the diagonals consist of
        variances). Although the sample covariance matrix is an unbiased estimator of the
        covariance matrix, i.e :math:`E(S) = \Sigma`, in practice it suffers from
        misspecification error and a lack of robustness. This is particularly problematic
        in mean-variance optimization, because the optimizer may give extra credence to
        the erroneous values.

        .. note::

            This should *not* be your default choice! Please use a shrinkage estimator
            instead.

    .. autofunction:: semicovariance

        The semivariance is the variance of all returns which are below some benchmark *B*
        (typically the risk-free rate) – it is a common measure of downside risk. There are multiple
        possible ways of defining a semicovariance matrix, the main differences lying in
        the 'pairwise' nature, i.e whether we should sum over :math:`\min(r_i,B)\min(r_j,B)`
        or :math:`\min(r_ir_j, B)`. In this implementation, we have followed the advice of
        Estrada (2007) [2]_, preferring:

        .. math::
            \frac{1}{n}\sum_{i = 1}^n {\sum_{j = 1}^n {\min \left( {{r_i},B} \right)} }
            \min \left( {{r_j},B} \right)

    .. autofunction:: exp_cov

        The exponential covariance matrix is a novel way of giving more weight to
        recent data when calculating covariance, in the same way that the exponential
        moving average price is often preferred to the simple average price. For a full
        explanation of how this estimator works, please refer to the
        `blog post <https://reasonabledeviations.com/2018/08/15/exponential-covariance/>`_
        on my academic website.

    .. autofunction:: cov_to_corr

    .. autofunction:: corr_to_cov



Shrinkage estimators
====================

A great starting point for those interested in understanding shrinkage estimators is
*Honey, I Shrunk the Sample Covariance Matrix* [3]_ by Ledoit and Wolf, which does a
good job at capturing the intuition behind them – we will adopt the
notation used therein. I have written a summary of this article, which is available
on my `website <https://reasonabledeviations.com/notes/papers/ledoit_wolf_covariance/>`_.
A more rigorous reference can be found in Ledoit and Wolf (2001) [4]_.

The essential idea is that the unbiased but often poorly estimated sample covariance can be
combined with a structured estimator :math:`F`, using the below formula (where
:math:`\delta` is the shrinkage constant):

.. math::
    \hat{\Sigma} = \delta F + (1-\delta) S

It is called shrinkage because it can be thought of as "shrinking" the sample
covariance matrix towards the other estimator, which is accordingly called the
**shrinkage target**. The shrinkage target may be significantly biased but has little
estimation error. There are many possible options for the target, and each one will
result in a different optimal shrinkage constant :math:`\delta`. PyPortfolioOpt offers
the following shrinkage methods:

- Ledoit-Wolf shrinkage:

    - ``constant_variance`` shrinkage, i.e the target is the diagonal matrix with the mean of
      asset variances on the diagonals and zeroes elsewhere. This is the shrinkage offered
      by ``sklearn.LedoitWolf``. 
    - ``single_factor`` shrinkage. Based on Sharpe's single-index model which effectively uses
      a stock's beta to the market as a risk model. See Ledoit and Wolf 2001 [4]_. 
    - ``constant_correlation`` shrinkage, in which all pairwise correlations are set to
      the average correlation (sample variances are unchanged). See Ledoit and Wolf 2003 [3]_

- Oracle approximating shrinkage (OAS), invented by Chen et al. (2010) [5]_, which
  has a lower mean-squared error than Ledoit-Wolf shrinkage when samples are
  Gaussian or near-Gaussian.

.. tip::

    For most use cases, I would just go with Ledoit Wolf shrinkage, as recommended by
    `Quantopian <https://www.quantopian.com/>`_ in their lecture series on quantitative
    finance.


My implementations have been translated from the Matlab code on
`Michael Wolf's webpage <https://www.econ.uzh.ch/en/people/faculty/wolf/publications.html>`_, with
the help of `xtuanta <https://github.com/robertmartin8/PyPortfolioOpt/issues/20>`_. 


.. autoclass:: CovarianceShrinkage
    :members:

    .. automethod:: __init__


References
==========

.. [1] Kritzman, Page & Turkington (2010) `In defense of optimization: The fallacy of 1/N <https://www.cfapubs.org/doi/abs/10.2469/faj.v66.n2.6>`_. Financial Analysts Journal, 66(2), 31-39.
.. [2] Estrada (2006), `Mean-Semivariance Optimization: A Heuristic Approach <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1028206>`_
.. [3] Ledoit, O., & Wolf, M. (2003). `Honey, I Shrunk the Sample Covariance Matrix <http://www.ledoit.net/honey.pdf>`_ The Journal of Portfolio Management, 30(4), 110–119. https://doi.org/10.3905/jpm.2004.110
.. [4] Ledoit, O., & Wolf, M. (2001). `Improved estimation of the covariance matrix of stock returns with an application to portfolio selection <http://www.ledoit.net/ole2.pdf>`_, 10, 603–621.
.. [5] Chen et al. (2010),  `Shrinkage Algorithms for MMSE Covariance Estimation <https://arxiv.org/pdf/0907.4698.pdf>`_, IEEE Transactions on Signals Processing, 58(10), 5016-5029.

##########################
Black-Litterman Allocation
##########################

The Black-Litterman (BL) model [1]_ takes a Bayesian approach to asset allocation.
Specifically, it combines a **prior** estimate of returns (for example, the market-implied
returns) with **views** on certain assets, to produce a **posterior** estimate of expected
returns. The advantages of this are:

- You can provide views on only a subset of assets and BL will meaningfully propagate it, 
  taking into account the covariance with other assets.
- You can provide *confidence* in your views.
- Using Black-Litterman posterior returns results in much more stable portfolios than
  using mean-historical return. 

Essentially, Black-Litterman treats the vector of expected returns itself as a quantity to
be estimated. The Black-Litterman formula is given below:

.. math:: 

    E(R) = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1}[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q]

- :math:`E(R)` is a Nx1 vector of expected returns, where *N* is the number of assets.
- :math:`Q` is a Kx1 vector of views.
- :math:`P` is the KxN **picking matrix** which maps views to the universe of assets.
  Essentially, it tells the model which view corresponds to which asset(s).
- :math:`\Omega` is the KxK **uncertainty matrix** of views. 
- :math:`\Pi` is the Nx1 vector of prior expected returns. 
- :math:`\Sigma` is the NxN covariance matrix of asset returns (as always)
- :math:`\tau` is a scalar tuning constant. 

Though the formula appears to be quite unwieldy, it turns out that the formula simply represents
a weighted average between the prior estimate of returns and the views, where the weighting
is determined by the confidence in the views and the parameter :math:`\tau`. 

Similarly, we can calculate a posterior estimate of the covariance matrix:

.. math::

    \hat{\Sigma} = \Sigma + [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1}


Though the algorithm is relatively simple, BL proved to be a challenge from a software
engineering perspective because it's not quite clear how best to fit it into PyPortfolioOpt's
API. The full discussion can be found on a `Github issue thread <https://github.com/robertmartin8/PyPortfolioOpt/issues/48>`_,
but I ultimately decided that though BL is not technically an optimizer, it didn't make sense to
split up its methods into `expected_returns` or `risk_models`. I have thus made it an independent
module and owing to the comparatively extensive theory, have given it a dedicated documentation page.
I'd like to thank  `Felipe Schneider <https://github.com/schneiderfelipe>`_ for his multiple
contributions to the Black-Litterman implementation. A full example of its usage, including the acquistion
of market cap data for free, please refer to the `cookbook recipe <https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/4-Black-Litterman-Allocation.ipynb>`_.

.. tip:: 

    Thomas Kirschenmann has built a neat interactive `Black-Litterman tool <https://github.com/thk3421-models/cardiel>`_
    on top of PyPortfolioOpt, which allows you to visualise BL outputs and compare optimization objectives.

Priors
======

You can think of the prior as the "default" estimate, in the absence of any information. 
Black and Litterman (1991) [2]_ provide the insight that a natural choice for this prior
is the market's estimate of the return, which is embedded into the market capitalisation
of the asset. 

Every asset in the market portfolio contributes a certain amount of risk to the portfolio.
Standard theory suggests that investors must be compensated for the risk that they take, so
we can attribute to each asset an expected compensation (i.e prior estimate of returns). This
is quantified by the market-implied risk premium, which is the market's excess return divided
by its variance: 

.. math::

    \delta = \frac{R-R_f}{\sigma^2}

To calculate the market-implied returns, we then use the following formula:

.. math::

    \Pi = \delta \Sigma w_{mkt}

Here, :math:`w_{mkt}` denotes the market-cap weights. This formula is calculating the total
amount of risk contributed by an asset and multiplying it with the market price of risk,
resulting in the market-implied returns vector :math:`\Pi`. We can use PyPortfolioOpt to calculate
this as follows::


    from pypfopt import black_litterman, risk_models

    """
    cov_matrix is a NxN sample covariance matrix
    mcaps is a dict of market caps
    market_prices is a series of S&P500 prices
    """
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, cov_matrix)


There is nothing stopping you from using any prior you see fit (but it must have the same dimensionality as the universe).
If you think that the mean historical returns are a good prior, 
you could go with that. But a significant body of research shows that mean historical returns are a completely uninformative
prior. 

.. note::

    You don't technically have to provide a prior estimate to the Black-Litterman model. This is particularly useful
    if your views (and confidences) were generated by some proprietary model, in which case BL is essentially a clever way
    of mixing your views.


Views
=====

In the Black-Litterman model, users can either provide **absolute** or **relative** views. Absolute views are statements like:
"AAPL will return 10%" or "XOM will drop 40%". Relative views, on the other hand, are statements like "GOOG will outperform FB by 3%".

These views must be specified in the vector :math:`Q` and mapped to the asset universe via the picking matrix :math:`P`. A brief
example of this is shown below, though a comprehensive guide is given by `Idzorek <https://faculty.fuqua.duke.edu/~charvey/Teaching/BA453_2006/Idzorek_onBL.pdf>`_.
Let's say that our universe is defined by the ordered list: SBUX, GOOG, FB, AAPL, BAC, JPM, T, GE, MSFT, XOM. We want to represent
four views on these 10 assets, two absolute and two relative:

1. SBUX will drop 20% (absolute)
2. MSFT will rise by 5% (absolute)
3. GOOG outperforms FB by 10%
4. BAC and JPM will outperform T and GE by 15%

The corresponding views vector is formed by taking the numbers above and putting them into a column::

    Q = np.array([-0.20, 0.05, 0.10, 0.15]).reshape(-1, 1)

The picking matrix is more interesting. Remember that its role is to link the views (which mention 8 assets) to the universe of 10
assets. Arguably, this is the most important part of the model because it is what allows us to propagate our expectations (and
confidences in expectations) into the model::

    P = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5, 0.5, -0.5, -0.5, 0, 0],
        ]
    )

A brief explanation of the above:

- Each view has a corresponding row in the picking matrix (the order matters)
- Absolute views have a single 1 in the column corresponding to the ticker's order in the universe. 
- Relative views have a positive number in the nominally outperforming asset columns and a negative number
  in the nominally underperforming asset columns. The numbers in each row should sum up to 0.


PyPortfolioOpt provides a helper method for inputting absolute views as either a ``dict`` or ``pd.Series`` – 
if you have relative views, you must build your picking matrix manually:: 

    from pypfopt.black_litterman import BlackLittermanModel

    viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.15}
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)


Confidence matrix and tau
=========================

The confidence matrix is a diagonal covariance matrix containing the variances of each view. One heuristic for calculating
:math:`\Omega` is to say that is proportional to the variance of the priors. This is reasonable - quantities that move
around a lot are harder to forecast! Hence PyPortfolioOpt does not require you to input a confidence matrix, and defaults to:

.. math::

    \Omega = \tau * P \Sigma P^T

Alternatively, we provide an implementation of Idzorek's method [1]_. This allows you to specify your view uncertainties as
percentage confidences. To use this, choose ``omega="idzorek"`` and pass a list of confidences (from 0 to 1) into the ``view_confidences``
parameter.

You are of course welcome to provide your own estimate. This is particularly applicable if your views are the output
of some statistical model, which may also provide the view uncertainty.

Another parameter that controls the relative weighting of the priors views is :math:`\tau`. There is a lot to be said about tuning
this parameter, with many contradictory rules of thumb. Indeed, there has been an entire paper written on it [3]_. We choose
the sensible default :math:`\tau = 0.05`.

.. note::

    If you use the default estimate of :math:`\Omega`, or ``omega="idzorek"``, it turns out that the value of :math:`\tau` does not matter. This
    is a consequence of the mathematics: the :math:`\tau` cancels in the matrix multiplications.


Output of the BL model
======================

The BL model outputs posterior estimates of the returns and covariance matrix. The default suggestion in the literature is to
then input these into an optimizer (see :ref:`efficient-frontier`). A quick alternative, which is quite useful for debugging, is
to calculate the weights implied by the returns vector [4]_. It is actually the reverse of the procedure we used to calculate the
returns implied by the market weights. 

.. math::

    w = (\delta \Sigma)^{-1} E(R)

In PyPortfolioOpt, this is available under ``BlackLittermanModel.bl_weights()``. Because the ``BlackLittermanModel`` class
inherits from ``BaseOptimizer``, this follows the same API as the ``EfficientFrontier`` objects::

    from pypfopt import black_litterman
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.efficient_frontier import EfficientFrontier

    viewdict = {"AAPL": 0.20, "BBY": -0.30, "BAC": 0, "SBUX": -0.2, "T": 0.15}
    bl = BlackLittermanModel(cov_matrix, absolute_views=viewdict)

    rets = bl.bl_returns()
    ef = EfficientFrontier(rets, cov_matrix)

    # OR use return-implied weights
    delta = black_litterman.market_implied_risk_aversion(market_prices)
    bl.bl_weights(delta)
    weights = bl.clean_weights()


Documentation reference
=======================

.. automodule:: pypfopt.black_litterman
    :members:
    :exclude-members: BlackLittermanModel

    .. autoclass:: BlackLittermanModel
        :members:

        .. automethod:: __init__

        .. caution::

            You **must** specify the covariance matrix and either absolute views or *both* Q and P, except in the special case
            where you provide exactly one view per asset, in which case P is inferred. 

References
==========

.. [1] Idzorek T. A step-by-step guide to the Black-Litterman model: Incorporating user-specified confidence levels. In: Forecasting Expected Returns in the Financial Markets. Elsevier Ltd; 2007. p. 17–38. 
.. [2] Black, F; Litterman, R. Combining investor views with market equilibrium. The Journal of Fixed Income, 1991.
.. [3] Walters, Jay, The Factor Tau in the Black-Litterman Model (October 9, 2013). Available at SSRN: https://ssrn.com/abstract=1701467 or http://dx.doi.org/10.2139/ssrn.1701467
.. [4] Walters J. The Black-Litterman Model in Detail (2014). SSRN Electron J.;(February 2007):1–65. 


=========

The rest of the features can be dissected at ****