"""Component 9 -- probability calibration.

Components 6, 7 and 8 built a ranking. This component asks whether the number attached to
that ranking can be believed: when Sentinel says 0.30, does it happen 30% of the time?

Calibration is **not** expected to improve the ranking, and a monotone map cannot. Unchanged
NDE, PR-AUC and precision@k beside an improved ECE and Brier is the success case here, not a
null result.

See ADR 0024 (where the artifacts live), ADR 0025 (the selection protocol, pre-registered),
ADR 0026 (why the base models are re-executed) and ADR 0027 (what the calibrator is fed).
"""
