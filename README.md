# NEST: Not Every Spread Is a Trade


A volatility-aware machine learning framework for statistical arbitrage and pairs trading.

> A spread is not a trade. A spread is only a candidate state.

Classical pairs trading often enters positions whenever a normalized spread becomes large. NEST instead asks whether the spread is actually tradeable after accounting for expected PnL, volatility, transaction costs, and regime effects.

## Core idea

For a pair of assets \((A,B)\), we construct a spread

$$
S_t^{A,B} = X_t^A - \beta_t^{A,B} X_t^B,
$$

where \(X_t^A\) and \(X_t^B\) are log-prices and \(\beta_t^{A,B}\) is a hedge ratio.

The project studies whether machine learning can improve classical spread trading by learning the map

$$
\Phi_t^{A,B}
\mapsto
\left(
\widehat p_t^{A,B},
\widehat m_t^{A,B},
\widehat \sigma_{t,H}^{A,B}
\right),
$$

where:

- $\(\widehat p_t^{A,B}\)$ is the predicted probability that the trade is profitable,
- $\(\widehat m_t^{A,B}\)$ is the predicted future net PnL,
- $\(\widehat \sigma_{t,H}^{A,B}\)$ is the predicted future spread risk.

## Project components

- Pair selection
- Hedge-ratio estimation
- Spread construction
- Volatility modeling
- Feature engineering
- Classification target construction
- Regression target construction
- Transaction-cost-aware backtesting
- Risk-based position sizing
- Baseline comparison

## First milestone

The first milestone is to implement a clean classical pairs-trading baseline:

1. Load price data.
2. Compute log-prices.
3. Select one candidate pair.
4. Estimate the hedge ratio.
5. Construct the spread.
6. Compute the normalized spread.
7. Backtest a naive z-score strategy.
8. Add transaction costs.

Only after this baseline is correct will the machine learning layer be added.

