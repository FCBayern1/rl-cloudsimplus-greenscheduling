"""Derive which datacenters can produce green power, from the experiment config.

The oracle scripts used to hardcode ``GREEN_DCS = [0, 1, 2]``, which silently
went wrong the moment the topology changed: the 8-DC sweep scenario puts a
fourth wind site at index 5 (``DC_Nordic2``) and two extra brown sites at 6/7,
so a hardcoded ``[0, 1, 2]`` both misses a green DC and mislabels the brown ones.

The predicate here matches what ``oracle_fdefer_gate.py`` already derives inline
(a DC is green-capable iff it owns turbines), tightened with the explicit
``green_energy_enabled`` flag so a DC that lists turbines but has green energy
switched off is not counted.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


def green_capable_dcs(
    datacenters: Optional[Sequence[Mapping[str, Any]]],
    num_dc: Optional[int] = None,
) -> List[int]:
    """Indices of the datacenters that can produce green power.

    A DC counts as green-capable when it has a non-empty ``turbine_ids`` list and
    ``green_energy_enabled`` is not explicitly false.

    Args:
        datacenters: the ``datacenters`` list from the experiment config. May be
            ``None``/empty for single-DC or legacy configs.
        num_dc: number of DCs the env actually instantiated. When given, indices
            at or beyond it are dropped (a config may list more DCs than the run
            uses) and it defines the fallback range.

    Returns:
        Sorted DC indices. Never empty: when no DC is green-capable (an all-brown
        topology) every DC is returned instead, because the callers use this list
        as their routing candidate set and an empty one would divide by zero.
        "No DC is greener than another" and "all DCs are candidates" coincide.
    """
    dcs = list(datacenters or [])
    green = [
        idx
        for idx, dc in enumerate(dcs)
        if isinstance(dc, Mapping)
        and dc.get("turbine_ids")
        and dc.get("green_energy_enabled", True)
    ]

    if num_dc is not None:
        green = [i for i in green if i < num_dc]

    if not green:
        fallback = list(range(num_dc if num_dc is not None else len(dcs)))
        logger.warning(
            "No green-capable DC found in config (%d datacenters); falling back to "
            "all %d DCs as routing candidates.", len(dcs), len(fallback)
        )
        return fallback

    return green


def describe_green_dcs(
    green_dcs: Sequence[int],
    datacenters: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """Human-readable ``dc0(DC_Nordic) dc5(DC_Nordic2)`` for run banners."""
    dcs = list(datacenters or [])
    parts = []
    for i in green_dcs:
        name = ""
        if i < len(dcs) and isinstance(dcs[i], Mapping):
            name = str(dcs[i].get("name", ""))
        parts.append(f"dc{i}({name})" if name else f"dc{i}")
    return " ".join(parts) if parts else "(none)"


def green_dcs_from_env(env: Any) -> List[int]:
    """Same derivation, taken off a constructed ``HierarchicalMultiDCEnv``."""
    return green_capable_dcs(
        getattr(env, "dc_configs", None), getattr(env, "num_datacenters", None)
    )
