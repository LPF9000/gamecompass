"""Ephemeral mixed-signal profiles: explicit judgments always override weak evidence.

The returned weights are inference features, never persisted as enjoyment labels.
A coefficient must come from a validation-selected configuration. No default
positive influence is silently assigned to imported ownership or playtime.
"""
from __future__ import annotations

import math
from typing import Collection, Mapping


class MixedPersonalizer:
    @staticmethod
    def combine(explicit: Mapping[str, float], implicit_hours: Mapping[str, float],
                coefficient: float, weighting: str,
                *, not_interested: Collection[str] = ()) -> dict[str, float]:
        """Combine signed judgments and weak observations without label conversion.

        Explicit neutral (zero) blocks an implicit positive for the same item.
        Not-interested items contribute no positive evidence even if an imported
        record or conflicting explicit value exists. This is separate from the
        serving layer's recommendation exclusions.

        ``normalized_log1p_hours`` divides log(1+hours) by the maximum over
        eligible unjudged observations, bounding each weak feature to [0,1].
        ``uniform`` records exposure/ownership only and does not imply liking.
        """
        if not math.isfinite(coefficient) or not 0 <= coefficient <= 1:
            raise ValueError('coefficient must be finite and in [0,1]')
        if weighting not in {'uniform', 'normalized_log1p_hours'}:
            raise ValueError('Unknown mixed implicit weighting')
        blocked = set(not_interested)
        result: dict[str, float] = {}
        for game, value in explicit.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError('Explicit weights must be finite')
            if game not in blocked:
                result[game] = value
        weak = {}
        for game, hours in implicit_hours.items():
            if game in explicit or game in blocked:
                continue
            hours = float(hours)
            if not math.isfinite(hours) or hours < 0:
                raise ValueError('Implicit hours must be finite and nonnegative')
            weak[game] = 1.0 if weighting == 'uniform' else math.log1p(hours)
        maximum = max(weak.values(), default=0.0)
        if coefficient and maximum:
            for game, value in weak.items():
                weight = coefficient * value / maximum
                if weight:
                    result[game] = weight
        return result
