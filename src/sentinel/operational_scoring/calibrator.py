"""Loading a frozen calibrator from Component 9's persisted parameters. No fitting here.

This is the one place in Component 18 that is not a re-execution: Component 9's
calibrators are genuinely persisted, as extracted parameters rather than as an
estimator object (``calibration.models.FittedCalibrator``'s own docstring: "for Platt
that is two floats; for isotonic it is the breakpoint arrays"). Loading them is reading
a file, not refitting anything, and ``calibration.predict.apply()`` -- reused
unmodified -- reconstructs the mapping from exactly those extracted values.

``estimator`` is set to ``None`` on the returned :class:`FittedCalibrator`: nothing this
component calls ever reads it -- ``calibration.predict.apply`` operates only on the
extracted fields -- and setting it to a real scikit-learn object would require refitting
one for no reason, undermining the point of loading a persisted artifact.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from sentinel.calibration.definitions import Method
from sentinel.calibration.models import FittedCalibrator


class CalibratorLoadError(ValueError):
    """Raised when a frozen calibrator's persisted parameters cannot be found or read."""


def load_frozen_calibrator(
    *,
    base_model_name: str,
    method: str,
    fold_set: str,
    fold_id: str,
    parameters_path: Path,
    breakpoints_path: Path,
) -> FittedCalibrator:
    """Reconstruct one calibrator exactly as Component 9 froze it."""
    params = pl.read_parquet(parameters_path).filter(
        (pl.col("model_name") == base_model_name)
        & (pl.col("fold_set") == fold_set)
        & (pl.col("fold_id") == fold_id)
        & (pl.col("method") == method)
    )
    if params.is_empty():
        raise CalibratorLoadError(
            f"{parameters_path.name}: no {method!r} calibrator parameters for "
            f"{base_model_name}/{fold_set}/{fold_id}"
        )
    meta = params.row(0, named=True)
    method_enum = Method(method)

    if method_enum is Method.PLATT:
        terms = dict(zip(params["term"].to_list(), params["value"].to_list(), strict=True))
        for required in ("coef", "intercept"):
            if required not in terms:
                raise CalibratorLoadError(
                    f"{parameters_path.name}: Platt calibrator for {base_model_name}/"
                    f"{fold_id} is missing term {required!r}"
                )
        return FittedCalibrator(
            model_name=base_model_name,
            fold_set=fold_set,
            fold_id=fold_id,
            method=Method.PLATT,
            estimator=None,
            input_transform=str(meta["input_transform"]),
            fit_rows=int(meta["fit_rows"]),
            fit_positive_rate=meta["fit_positive_rate"],
            fit_start=meta["fit_start"],
            fit_end=meta["fit_end"],
            coefficient=float(terms["coef"]),
            intercept=float(terms["intercept"]),
        )

    breakpoints = (
        pl.read_parquet(breakpoints_path)
        .filter(
            (pl.col("model_name") == base_model_name)
            & (pl.col("fold_set") == fold_set)
            & (pl.col("fold_id") == fold_id)
        )
        .sort("breakpoint_index")
    )
    if breakpoints.is_empty():
        raise CalibratorLoadError(
            f"{breakpoints_path.name}: no isotonic breakpoints for "
            f"{base_model_name}/{fold_set}/{fold_id}"
        )
    return FittedCalibrator(
        model_name=base_model_name,
        fold_set=fold_set,
        fold_id=fold_id,
        method=Method.ISOTONIC,
        estimator=None,
        input_transform=str(meta["input_transform"]),
        fit_rows=int(meta["fit_rows"]),
        fit_positive_rate=meta["fit_positive_rate"],
        fit_start=meta["fit_start"],
        fit_end=meta["fit_end"],
        x_thresholds=tuple(breakpoints["x_threshold"].to_list()),
        y_thresholds=tuple(breakpoints["y_threshold"].to_list()),
        x_min=float(breakpoints["x_min"][0]),
        x_max=float(breakpoints["x_max"][0]),
    )


__all__ = ["CalibratorLoadError", "load_frozen_calibrator"]
