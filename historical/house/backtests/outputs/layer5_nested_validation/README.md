# Layer 5: District-Specific Residual Uncertainty

## Status

Rejected for production after nested leave-one-cycle-out validation.

## Proposed method

District-level historical residual RMSE was estimated using empirical
Bayes shrinkage and converted to bounded district uncertainty multipliers.

The production candidate used:

- District component SD: 4.0
- Shrinkage strength: 4.0
- Multiplier bounds: 0.80 to 1.20

## Initial non-nested result

The initial sensitivity analysis showed:

- Essentially unchanged pooled Brier score
- Improved pooled log loss
- Slightly worse calibration
- Slightly worse expected-seat error

The selected configuration appeared provisionally promising.

## Nested validation result

Each outer election cycle was excluded from both:

1. District multiplier estimation
2. Hyperparameter selection

Nested pooled changes:

- Brier: +0.000169, worse
- Log loss: -0.000719, better
- ECE: +0.000648, worse
- Absolute expected-seat error: +1.396588, worse

Cycle consistency:

- Brier improved in 1 of 4 cycles
- Log loss improved in 1 of 4 cycles
- Expected-seat error improved in 0 of 4 cycles
- Four different configurations were selected across four outer folds

Interval coverage also worsened at the 50%, 80%, and 95% levels.

## Decision

Do not use district-specific residual uncertainty in the production model.

The apparent initial benefit was not robust to nested hyperparameter
validation and was concentrated primarily in the 2020 election cycle.

The existing constant district uncertainty structure remains the
production specification until a future alternative demonstrates
stable out-of-sample improvement.
