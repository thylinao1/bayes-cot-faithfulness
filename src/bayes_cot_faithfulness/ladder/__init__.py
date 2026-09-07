"""Element 11's ladder: the organism, the twin, and what they are allowed to claim.

    "a planted-and-caught case demonstrates detection of an inserted manipulation; it
    does not establish that rho* maps to a real unmeasured confounder, nor that the
    method catches unplanted cases in the wild."
    -- PREREGISTRATION_jury_and_scale.md section 12.3

Modules
-------
``spec``            the partition, the budget, the cell ids, and every LANE CHOICE named
``trigger_data``    the four training sets, their manifest, and the overlap refusal
``lora_train``      one checkpoint, its config, its hashes, --dry-run and --tiny
``serve_manifest``  the checkpoint as a roster row and as a bcf/waves TSV row
``statistic``       element 11(f): organism minus twin, its interval, the A3.2 MDE
``comparison_set``  element 11(e): the four columns the sweep already measures
``heldout_family``  element 11(d): held out only when a freeze commit is given
"""

from __future__ import annotations

__all__ = [
    "comparison_set",
    "heldout_family",
    "lora_train",
    "serve_manifest",
    "spec",
    "statistic",
    "trigger_data",
]
