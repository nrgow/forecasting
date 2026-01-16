# Requirements: CVXPY-Based Prediction Market Portfolio Optimizer

## 1. Purpose

The system SHALL provide a portfolio optimizer for binary prediction markets that allocates bankroll across multiple markets using log-utility (Kelly-style) optimization.

The optimizer SHALL support:
- Long positions in YES contracts
- Long positions in NO contracts
- Daily (or periodic) rebalancing
- User-supplied probability estimates
- A signed exposure interface that combines YES/NO into one target per market

The primary goal is to maximize expected log wealth subject to risk, exposure, and turnover constraints.

---

## 2. Scope and Assumptions

- Markets are binary outcome markets (YES/NO).
- Contracts settle at $1 for a correct outcome and $0 otherwise.
- Prices are expressed in the interval (0, 1).
- The optimizer operates on fractions of total bankroll.
- Leverage is NOT allowed.
- Short selling is NOT allowed (NO positions are modeled as buying the NO contract).

---

## 3. Inputs

### 3.1 Market Data (per rebalance)

The optimizer SHALL accept the following inputs:

| Name | Type | Description |
|-----|------|-------------|
| p_yes | float[n] | Market prices for YES contracts |
| q | float[n] | User-estimated probability of YES |
| current_alloc | float[n] | Current signed bankroll exposure per market |

Notes:
- p_no is derived as (1 - p_yes)
- current_alloc > 0 means YES exposure; current_alloc < 0 means NO exposure

---

## 4. Decision Variables

The optimizer SHALL solve for:

| Variable | Type | Meaning |
|----------|------|---------|
| f_yes[i] | float ≥ 0 | Fraction of bankroll allocated to YES in market i |
| f_no[i] | float ≥ 0 | Fraction of bankroll allocated to NO in market i |
| f_signed[i] | float | Signed exposure (f_yes[i] - f_no[i]) |

---

## 5. Constraints

### 5.1 Position Bounds

For each market i:
- 0 ≤ f_yes[i] ≤ max_fraction_per_market
- 0 ≤ f_no[i] ≤ max_fraction_per_market

### 5.2 Mutual Exclusivity

For each market i:
- f_yes[i] + f_no[i] ≤ max_fraction_per_market

### 5.3 Total Exposure

Across all markets:
- Σ (f_yes[i] + f_no[i]) ≤ max_total_fraction

### 5.4 Cash Safety

The optimizer SHALL ensure:
- All wealth multipliers remain strictly positive in all outcomes.

---

## 6. Objective Function

### 6.1 Wealth Multipliers

For market i:

If YES resolves:
- W_yes = 1 + f_yes[i] * (1 - p_yes[i]) / p_yes[i] - f_no[i]

If NO resolves:
- W_no = 1 + f_no[i] * (1 - p_no[i]) / p_no[i] - f_yes[i]

---

### 6.2 Expected Log Utility

For market i, expected log contribution:

q[i] * log(W_yes)
+ (1 - q[i]) * log(W_no)

Total objective:

maximize kelly_fraction * Σ expected_log_wealth

Where:
- kelly_fraction ∈ (0, 1] scales aggressiveness (fractional Kelly).

---

## 7. Rebalancing & Turnover Control

The optimizer SHALL optionally penalize turnover:

turnover = Σ |f_signed[i] - current_alloc[i]|

Objective adjustment:

maximize expected_log_wealth - turnover_penalty * turnover

---

## 8. Solver & Implementation

- The optimizer SHALL be implemented using CVXPY.
- The optimization problem MUST be convex.
- Supported solvers SHOULD include ECOS and SCS.
- Numerical safeguards SHALL be used to avoid log(0).
- Small allocations MAY be rounded to zero for presentation.
- The implementation SHALL expose marginal log-utility gains for YES/NO at the optimized allocation.

---

## 9. Outputs

The optimizer SHALL return:

| Output | Type | Description |
|-------|------|-------------|
| target_alloc | float[n] | Target signed exposures (positive=YES, negative=NO) |
| expected_log_growth | float | Optimized expected log growth rate |
| marginal_gain_yes | float[n] | Marginal log-utility gain for YES at target allocation |
| marginal_gain_no | float[n] | Marginal log-utility gain for NO at target allocation |
| marginal_gain_directional | float[n] | Marginal gain in the chosen direction |

---

## 10. Non-Goals

The system SHALL NOT:
- Perform probability estimation (this will come from the simulation pipeline)
- Execute trades
- Model order book dynamics
- Guarantee profitability

---

## 11. Extensibility (Future)

The design SHOULD allow future support for:
- Scenario-based correlated outcomes
- Market group constraints (exactly-one / at-most-one)
- Liquidity-aware sizing
- Transaction fees and slippage
- Multi-currency bankrolls

---

## 12. Entry Point

The system SHALL provide a CLI entrypoint that:
- Loads the latest per-market probability estimates for open markets
- Loads the latest market YES prices for those markets
- Runs the optimizer assuming an empty portfolio (zero current allocations)
