"""Optional extension points and ablation controls for TRIM.

The default settings preserve the standard TRIM behavior. Observation hooks
receive immutable records. The fixed-row builder and candidate-state scorer
also receive internal numerical state; the builder receives a copy and may
return only row identifiers and declarative metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping


RETENTION_CLASS_ACTION = "retention_class"
VERTICAL_REFINEMENT_ACTION = "vertical_refinement"
ACTION_TYPES = frozenset({
    RETENTION_CLASS_ACTION,
    VERTICAL_REFINEMENT_ACTION,
})

S0_INITIALIZATION = "s0"
ALL_ROWS_AT_MAX_GENERALIZATION = "all_rows_at_max_generalization"
INITIALIZATION_MODES = frozenset({
    S0_INITIALIZATION,
    ALL_ROWS_AT_MAX_GENERALIZATION,
})

FULL_SWAP_POSTPROCESSING = "full_swap"
NO_VERTICAL_POSTPROCESSING = "none"
EVICT_LOW_K_POSTPROCESSING = "evict_low_k"
VERTICAL_POSTPROCESSING_MODES = frozenset({
    FULL_SWAP_POSTPROCESSING,
    NO_VERTICAL_POSTPROCESSING,
    EVICT_LOW_K_POSTPROCESSING,
})

@dataclass(frozen=True)
class AblationActions:
    """Action ablations whose defaults preserve the TRIM method."""

    enabled_action_types: tuple[str, ...] = (
        RETENTION_CLASS_ACTION,
        VERTICAL_REFINEMENT_ACTION,
    )
    initialization: str = S0_INITIALIZATION
    vertical_postprocessing: str = FULL_SWAP_POSTPROCESSING
    move_out_threshold: int | None = None

    def __post_init__(self):
        enabled_action_types = tuple(self.enabled_action_types)
        object.__setattr__(self, "enabled_action_types", enabled_action_types)

        unknown_actions = set(enabled_action_types) - ACTION_TYPES
        if unknown_actions:
            raise ValueError(
                "Unknown action types: "
                + ", ".join(sorted(unknown_actions))
            )
        if not enabled_action_types:
            raise ValueError("enabled_action_types must contain at least one action type.")
        if len(set(enabled_action_types)) != len(enabled_action_types):
            raise ValueError("enabled_action_types cannot contain duplicates.")
        if self.initialization not in INITIALIZATION_MODES:
            raise ValueError(
                f"Unknown initialization mode: {self.initialization!r}."
            )
        if self.vertical_postprocessing not in VERTICAL_POSTPROCESSING_MODES:
            raise ValueError(
                "Unknown vertical postprocessing mode: "
                f"{self.vertical_postprocessing!r}."
            )
        if self.vertical_postprocessing == EVICT_LOW_K_POSTPROCESSING:
            if (
                isinstance(self.move_out_threshold, bool)
                or not isinstance(self.move_out_threshold, int)
                or self.move_out_threshold < 1
            ):
                raise ValueError(
                    "move_out_threshold must be a positive integer when "
                    "vertical_postprocessing='evict_low_k'."
                )
        elif self.move_out_threshold is not None:
            raise ValueError(
                "move_out_threshold is only used when "
                "vertical_postprocessing='evict_low_k'."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled_action_types": list(self.enabled_action_types),
            "initialization": self.initialization,
            "vertical_postprocessing": self.vertical_postprocessing,
            "move_out_threshold": self.move_out_threshold,
        }


@dataclass(frozen=True)
class FixedRowCandidateBuilderContext:
    """Input for a static row-candidate builder after the train split."""

    train_level0_encode: Any
    selected_attributes: tuple[str, ...]
    random_state: int
    max_generalization_level: Mapping[str, int]


@dataclass(frozen=True)
class FixedRowCandidatePlan:
    """A fixed partition used instead of dynamic retention-class enumeration."""

    candidate_groups: tuple[tuple[Any, tuple[Any, ...]], ...]
    initial_row_ids: tuple[Any, ...]
    generalization_level: Mapping[str, int]
    privacy_exempt_row_ids: tuple[Any, ...] = ()
    whole_group_admission: bool = True
    close_remaining_groups: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        groups = tuple(
            (key, tuple(row_ids))
            for key, row_ids in self.candidate_groups
        )
        if not groups:
            raise ValueError("candidate_groups must contain at least one group.")
        if any(not row_ids for _key, row_ids in groups):
            raise ValueError("Fixed row candidate groups cannot be empty.")
        object.__setattr__(self, "candidate_groups", groups)
        object.__setattr__(self, "initial_row_ids", tuple(self.initial_row_ids))
        object.__setattr__(
            self,
            "privacy_exempt_row_ids",
            tuple(self.privacy_exempt_row_ids),
        )
        object.__setattr__(
            self,
            "generalization_level",
            MappingProxyType({
                str(attribute): int(level)
                for attribute, level in dict(self.generalization_level).items()
            }),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )
        if not self.initial_row_ids:
            raise ValueError("initial_row_ids must contain at least one row.")
        if not isinstance(self.whole_group_admission, bool):
            raise TypeError("whole_group_admission must be bool.")
        if not isinstance(self.close_remaining_groups, bool):
            raise TypeError("close_remaining_groups must be bool.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_group_count": len(self.candidate_groups),
            "initial_row_count": len(self.initial_row_ids),
            "privacy_exempt_row_count": len(self.privacy_exempt_row_ids),
            "generalization_level": dict(self.generalization_level),
            "whole_group_admission": self.whole_group_admission,
            "close_remaining_groups": self.close_remaining_groups,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SelectionScoreContext:
    """Values available to a custom final candidate-scoring function."""

    iteration: int
    candidate_type: str
    candidate_id: Any
    utility_gain: float
    current_k: int
    next_k: int
    privacy_cost: float
    candidate_published_min_k: int
    release_target_k: int
    release_validation_loss: float
    release_row_count: int
    release_suppressed_row_count: int
    release_coverage_fraction: float
    candidate_state_score: float | None = None


@dataclass(frozen=True)
class RankedCandidate:
    """One enumerated action offered to an optional shortlist callback."""

    position: int
    candidate_type: str
    candidate_id: Any
    rank_score: float


@dataclass(frozen=True)
class CandidateShortlistContext:
    """Complete candidate set before the core's per-type LGA top-k filter."""

    iteration: int
    rank_top_k: int
    candidates: tuple[RankedCandidate, ...]


@dataclass(frozen=True)
class CandidateScorerRunContext:
    """Numerical inputs for constructing an optional candidate scorer."""

    train_original_encode: Any
    train_y: Any


@dataclass(frozen=True)
class CandidateStateScoreContext:
    """Current and candidate states used by an optional candidate scorer."""

    iteration: int
    candidate_type: str
    candidate_id: Any
    current_encode: Any
    current_y: Any
    candidate_encode: Any
    candidate_y: Any
    current_validation_encode: Any
    candidate_validation_encode: Any
    validation_y: Any


@dataclass(frozen=True)
class CandidateObservation:
    """Read-only record emitted after one call has scored all candidates."""

    iteration: int
    candidate_type: str
    candidate_id: Any
    rank_score: float
    selection_score: float
    utility_gain: float
    current_k: int
    next_k: int
    privacy_cost: float
    candidate_published_min_k: int
    release_target_k: int
    release_validation_loss: float
    release_row_count: int
    release_suppressed_row_count: int
    release_coverage_fraction: float
    selected: bool
    candidate_state_score: float | None = None


@dataclass(frozen=True)
class TRIMIterationObservation:
    """Read-only public outcome emitted after an iteration is committed."""

    iteration: int
    action: str
    attribute: str | None
    selected_row_count: int
    ordinary_min_k: int
    ordinary_published_row_count: int
    generalization_level: Mapping[str, int]
    release_target_k: int
    release_validation_loss: float
    release_row_count: int
    release_suppressed_row_count: int
    release_coverage_fraction: float
    release_utility_constraint_met: bool


SelectionScore = Callable[[SelectionScoreContext], float]
CandidateShortlist = Callable[[CandidateShortlistContext], tuple[int, ...]]
CandidateStateScorer = Callable[[CandidateStateScoreContext], float]
CandidateStateScorerFactory = Callable[
    [CandidateScorerRunContext],
    CandidateStateScorer,
]
FixedRowCandidateBuilder = Callable[
    [FixedRowCandidateBuilderContext],
    FixedRowCandidatePlan,
]
CandidateObserver = Callable[[CandidateObservation], None]
TRIMIterationObserver = Callable[[TRIMIterationObservation], None]


def default_selection_score(context: SelectionScoreContext) -> float:
    """Return the core score: validation utility gain / privacy cost."""

    return context.utility_gain / context.privacy_cost


__all__ = [
    "ACTION_TYPES",
    "ALL_ROWS_AT_MAX_GENERALIZATION",
    "AblationActions",
    "CandidateObservation",
    "CandidateObserver",
    "CandidateScorerRunContext",
    "CandidateShortlist",
    "CandidateShortlistContext",
    "CandidateStateScoreContext",
    "CandidateStateScorer",
    "CandidateStateScorerFactory",
    "EVICT_LOW_K_POSTPROCESSING",
    "FULL_SWAP_POSTPROCESSING",
    "TRIMIterationObservation",
    "TRIMIterationObserver",
    "FixedRowCandidateBuilder",
    "FixedRowCandidateBuilderContext",
    "FixedRowCandidatePlan",
    "INITIALIZATION_MODES",
    "NO_VERTICAL_POSTPROCESSING",
    "RankedCandidate",
    "RETENTION_CLASS_ACTION",
    "S0_INITIALIZATION",
    "SelectionScore",
    "SelectionScoreContext",
    "VERTICAL_POSTPROCESSING_MODES",
    "VERTICAL_REFINEMENT_ACTION",
    "default_selection_score",
]
