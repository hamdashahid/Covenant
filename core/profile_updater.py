from __future__ import annotations

from typing import Any


class ProfileUpdater:
    def merge(
        self,
        profile: dict[str, Any],
        validated_fields: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        merged = dict(profile)
        conflicts: list[str] = []
        for key, value in validated_fields.items():
            if key in merged and merged[key] != value:
                conflicts.append(
                    f"Conflict for '{key}': existing={merged[key]} new={value}"
                )
            merged[key] = value
        return merged, conflicts
