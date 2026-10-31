"""Method ID -> callable (L2).  `15` §3; `06` fixes the IDs.

The registry is also the pre-registration record (`10` standing prohibition 7): adding an entry
after results have been seen requires an entry in `13_RESEARCH_LOG.md`.

Scope after extensions E1 and E3 (`15` §11.3): B1-B9 (B10 excepted), plus the two
cross-fitting scheme variants `06` B5 requires the implementation to offer and report
(`B5-kfold`, `B5-avglam`) and the two extra thresholds `10` H2's guard requires
(`B6-t1.5`, `B6-t3`; `B6` itself is the `|t| > 2` of `06` B6).  B7 and B8 arrived with E3; the finite-N variants `B9-vfull`, `B9-R1`, `B9-R2`,
`B9-R3` and `B9-ppiplus` with E4 (`04` §4.8.3-4.8.4, `07` Experiment E). B10 is
still deliberately absent rather than stubbed, and R4 has no entry because it is a
theorem about `B9`, not a separate expression.
"""
from __future__ import annotations

from typing import Callable

from .bootstrap import (method_b7, method_b7_b9999, method_b8, method_b8_b9999)
from .lc_ppi import (method_b9, method_b9_nolev, method_b9_ppiplus, method_b9_r1,
                     method_b9_hc, method_b9_r2, method_b9_r3, method_b9_r5,
    method_b9_tmin,
                     method_b9_vfull)
from .ppi import method_b1, method_b2
from .ppi_plus import (method_b3, method_b4, method_b5, method_b5_avglam,
                       method_b5_kfold, method_b6, method_b6_t15, method_b6_t3)
from .types import IntervalResult

MethodFn = Callable[..., IntervalResult]

METHODS: dict[str, MethodFn] = {
    "B1": method_b1,
    "B2": method_b2,
    "B3": method_b3,
    "B4": method_b4,
    "B5": method_b5,
    "B5-kfold": method_b5_kfold,
    "B5-avglam": method_b5_avglam,
    "B6": method_b6,
    "B6-t1.5": method_b6_t15,
    "B6-t3": method_b6_t3,
    "B7": method_b7,
    "B7-B9999": method_b7_b9999,
    "B8": method_b8,
    "B8-B9999": method_b8_b9999,
    "B9-nolev": method_b9_nolev,
    "B9": method_b9,
    "B9-vfull": method_b9_vfull,
    "B9-R1": method_b9_r1,
    "B9-R2": method_b9_r2,
    "B9-R3": method_b9_r3,
    "B9-R5": method_b9_r5,
    "B9-tmin": method_b9_tmin,
    "B9-HC": method_b9_hc,
    "B9-ppiplus": method_b9_ppiplus,
}

#: Methods that legitimately consume randomness (bootstrap).  The runner hands each of
#: these a generator derived from the cell seed (`15` §7.4); every other method gets None.
STOCHASTIC_METHODS: frozenset[str] = frozenset({"B7", "B7-B9999", "B8", "B8-B9999"})


def get(method_id: str) -> MethodFn:
    if method_id not in METHODS:
        raise KeyError(f"unknown method {method_id!r}; registered: {sorted(METHODS)}")
    return METHODS[method_id]
