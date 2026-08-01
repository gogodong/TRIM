"""Declared Exp-6 candidate builders and selection rules."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from prototype import (
    RETENTION_CLASS_ACTION,
    VERTICAL_REFINEMENT_ACTION,
    CandidateScorerRunContext,
    CandidateShortlistContext,
    CandidateStateScoreContext,
    SelectionScoreContext,
)

from ..ranking_models import FixedOriginalInfluence
from .enumeration import KMeansClusterRingBuilder


@dataclass
class SeededUniformShortlist:
    """Select one candidate uniformly from the complete H+V action pool."""

    seed: int
    _random: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        self.seed = int(self.seed)
        self._random = random.Random(self.seed)

    def __call__(self, context: CandidateShortlistContext) -> tuple[int, ...]:
        if not context.candidates:
            return ()
        selected = self._random.randrange(len(context.candidates))
        return (context.candidates[selected].position,)


def privacy_only_score(context: SelectionScoreContext) -> float:
    """Ignore utility and prefer the smallest privacy cost."""

    return 1.0 / context.privacy_cost


def utility_only_score(context: SelectionScoreContext) -> float:
    """Ignore privacy cost and use the latest TRIM validation utility gain."""

    return context.utility_gain


def yang_if_utility_privacy_score(context: SelectionScoreContext) -> float:
    """Use Yang-style validation Utility Delta as TRIM's utility term."""

    if context.candidate_state_score is None:
        raise RuntimeError(
            "Yang-style IF selection requires candidate_state_scorer_factory."
        )
    return context.candidate_state_score / context.privacy_cost


@dataclass
class YangIFSelectionScorer:
    """Exp-33-style fixed-original IF projection for ordinary candidate states."""

    horizontal: FixedOriginalInfluence
    vertical: FixedOriginalInfluence
    _current_states: dict = field(default_factory=dict, init=False, repr=False)
    _current_losses: dict = field(default_factory=dict, init=False, repr=False)

    def __call__(self, context: CandidateStateScoreContext) -> float:
        if context.candidate_type == RETENTION_CLASS_ACTION:
            reference = self.horizontal
            scope = "last_layer"
        elif context.candidate_type == VERTICAL_REFINEMENT_ACTION:
            reference = self.vertical
            scope = "all"
        else:
            raise ValueError(
                f"Unsupported Yang-style candidate type: {context.candidate_type!r}."
            )
        cache_key = (int(context.iteration), scope)
        if cache_key not in self._current_states:
            current_state = reference.state(
                context.current_encode,
                context.current_y,
            )
            self._current_states[cache_key] = current_state
            self._current_losses[cache_key] = reference.estimated_validation_loss(
                current_state,
                context.current_validation_encode,
                context.validation_y,
            )
        candidate_state = reference.state(
            context.candidate_encode,
            context.candidate_y,
        )
        next_loss = reference.estimated_validation_loss(
            candidate_state,
            context.candidate_validation_encode,
            context.validation_y,
        )
        utility_delta = float(self._current_losses[cache_key] - next_loss)
        if not math.isfinite(utility_delta):
            raise ValueError("Yang-style IF produced a non-finite Utility Delta.")
        return utility_delta


@dataclass(frozen=True)
class YangIFSelectionScorerFactory:
    """Fit the explicit original-data MLP and freeze Exp-32/Exp-33 IF references."""

    model_factory: object
    horizontal_last: dict
    vertical_all: dict

    def __call__(
        self,
        context: CandidateScorerRunContext,
    ) -> YangIFSelectionScorer:
        model = self.model_factory()
        model.fit(context.train_original_encode, context.train_y)
        if getattr(model, "network_", None) is None:
            raise ValueError(
                "Exp-6b2 Yang-style IF requires an explicit fitted MLP "
                "selection estimator; incompatible models are not accepted."
            )
        required = {"damping", "cg_max_iter", "cg_tolerance"}
        for name, values in (
            ("horizontal_last", self.horizontal_last),
            ("vertical_all", self.vertical_all),
        ):
            missing = sorted(required - set(values))
            if missing:
                raise ValueError(f"{name} is missing solver values: {missing}.")
        horizontal = FixedOriginalInfluence(
            model,
            context.train_original_encode,
            context.train_y,
            parameter_scope="last_layer",
            damping=float(self.horizontal_last["damping"]),
            cg_max_iter=int(self.horizontal_last["cg_max_iter"]),
            cg_tolerance=float(self.horizontal_last["cg_tolerance"]),
        )
        vertical = FixedOriginalInfluence(
            model,
            context.train_original_encode,
            context.train_y,
            parameter_scope="all",
            damping=float(self.vertical_all["damping"]),
            cg_max_iter=int(self.vertical_all["cg_max_iter"]),
            cg_tolerance=float(self.vertical_all["cg_tolerance"]),
        )
        return YangIFSelectionScorer(horizontal=horizontal, vertical=vertical)


__all__ = [
    "KMeansClusterRingBuilder",
    "SeededUniformShortlist",
    "YangIFSelectionScorerFactory",
    "privacy_only_score",
    "utility_only_score",
    "yang_if_utility_privacy_score",
]
