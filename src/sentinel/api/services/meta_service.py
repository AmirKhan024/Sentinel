"""Provenance and run-discovery endpoints. Reads manifests and filenames; computes nothing."""

from __future__ import annotations

import json

from sentinel.api.errors import ApiError, ArtifactNotFound
from sentinel.api.services.artifacts import resolve_latest
from sentinel.config import Settings
from sentinel.manifest import manifest_path_for

#: (settings dir attribute, anchor dataset slug), one per layer this API exposes. Deliberately
#: not every processed layer in the repository -- only the ones a product consumer of this API
#: can otherwise reach through it, per the project's "build only what a component needs" rule.
_COMPONENTS: dict[str, tuple[str, str]] = {
    "policy": ("policy_processed_dir", "inspection_recommendations"),
    "scheduling": ("scheduling_processed_dir", "inspection_schedule"),
    "explanations": ("explanations_processed_dir", "explanation_values"),
    "review": ("review_processed_dir", "human_review_queue"),
    "operational_selection": ("operational_selection_processed_dir", "operational_selection"),
}


class UnknownComponent(ApiError):
    status_code = 404
    error_code = "unknown_component"


def get_manifest(settings: Settings, component: str) -> dict[str, object]:
    if component not in _COMPONENTS:
        raise UnknownComponent(
            f"Unknown component {component!r}. Known: {', '.join(sorted(_COMPONENTS))}"
        )
    dir_attr, prefix = _COMPONENTS[component]
    path = resolve_latest(getattr(settings, dir_attr), prefix=prefix)
    manifest_path = manifest_path_for(path)
    if not manifest_path.exists():
        raise ArtifactNotFound(f"No manifest sidecar for the latest {component} run.")
    result: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return result


def list_runs(settings: Settings, component: str | None = None) -> list[dict[str, object]]:
    components = [component] if component is not None else list(_COMPONENTS)
    out: list[dict[str, object]] = []
    for one_component in components:
        if one_component not in _COMPONENTS:
            raise UnknownComponent(
                f"Unknown component {one_component!r}. Known: {', '.join(sorted(_COMPONENTS))}"
            )
        dir_attr, prefix = _COMPONENTS[one_component]
        directory = getattr(settings, dir_attr)
        for path in sorted(directory.glob(f"{prefix}_*.parquet")):
            out.append({"component": one_component, "path": str(path), "name": path.stem})
    return out


__all__ = ["UnknownComponent", "get_manifest", "list_runs"]
