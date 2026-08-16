"""Component 4: as-of feature engineering.

Builds the information a scheduler actually had *before* each prediction
opportunity: for every eligible Component 3 target row, historical evidence
drawn strictly from inspections dated before that row's reference date.

The single rule the whole package is designed around:

    a feature for the row at inspection_date = d may use only records dated
    strictly before d

See ``docs/analysis/as_of_feature_engineering_findings.md`` for the measurements
behind the boundary and the missing-value rules, and
``docs/data_contracts/as_of_features.md`` for the output contract.
"""

from __future__ import annotations
