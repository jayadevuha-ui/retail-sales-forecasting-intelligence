# Project Plan

## Business question

Can historical transaction patterns be transformed into a reliable short-horizon daily revenue forecast while also surfacing useful commercial insights?

## Scope

This is deliberately broader than a pure ML exercise. It combines:

1. transaction cleaning
2. descriptive analytics
3. time-series aggregation
4. temporal feature engineering
5. forecasting baselines
6. machine-learning forecasting
7. chronological validation
8. dashboard presentation

## Evaluation principle

No random train/test split is used for forecasting. The last 42 calendar days are held out so the model is tested on genuinely future observations relative to its training period.
