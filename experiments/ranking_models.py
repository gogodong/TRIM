"""MLP-LGA and fixed-original influence utilities for Exp-32/Exp-33."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
import torch.nn.functional as F

from prototype.gpu_math import to_device_tensor


def _flat(values, parameters):
    parts = [
        (torch.zeros_like(parameter) if value is None else value).reshape(-1)
        for value, parameter in zip(values, parameters)
    ]
    return torch.cat(parts) if parts else torch.empty(0)


def _loss(model, network, X_tensor, y_tensor):
    outputs = network(X_tensor)
    target = model._classification_target_indices(
        y_tensor,
        device=X_tensor.device,
    )
    if model._is_multiclass():
        return F.cross_entropy(outputs, target)
    return F.binary_cross_entropy_with_logits(
        outputs.reshape(-1),
        target.to(dtype=X_tensor.dtype),
    )


def _gradient(loss, parameters, *, create_graph=False):
    return _flat(
        torch.autograd.grad(
            loss,
            parameters,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        ),
        parameters,
    )


@dataclass
class MLPLGAScorer:
    """First-order validation gain at one fitted MLP estimator."""

    model: object
    network: nn.Module
    parameters: list[torch.Tensor]
    current_gradient: torch.Tensor
    current_row_count: int
    device: torch.device
    dtype: torch.dtype
    allow_cpu_fallback: bool
    validation_gradients: dict[object, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def build(cls, model, current_encode, current_y):
        if getattr(model, "network_", None) is None:
            raise ValueError("The configured estimator must be a fitted MLP.")
        network = model.network_.to(device=model.device, dtype=model.dtype)
        network.eval()
        parameters = [parameter for parameter in network.parameters() if parameter.requires_grad]
        X_current = to_device_tensor(
            current_encode,
            device=model.device,
            dtype=model.dtype,
            allow_cpu_fallback=bool(model.allow_cpu_fallback),
        )
        y_current = to_device_tensor(
            current_y,
            device=X_current.device,
            dtype=X_current.dtype,
            allow_cpu_fallback=bool(model.allow_cpu_fallback),
        ).reshape(-1)
        current_gradient = _gradient(
            _loss(model, network, X_current, y_current),
            parameters,
        ).detach()
        return cls(
            model=model,
            network=network,
            parameters=parameters,
            current_gradient=current_gradient,
            current_row_count=int(X_current.shape[0]),
            device=X_current.device,
            dtype=X_current.dtype,
            allow_cpu_fallback=bool(model.allow_cpu_fallback),
        )

    def data_gradient(self, encode, labels):
        X_tensor = to_device_tensor(
            encode,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        y_tensor = to_device_tensor(
            labels,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        ).reshape(-1)
        if X_tensor.ndim != 2 or X_tensor.shape[0] != y_tensor.shape[0]:
            raise ValueError("MLP estimator features and labels must align.")
        return _gradient(
            _loss(self.model, self.network, X_tensor, y_tensor),
            self.parameters,
        ).detach()

    def validation_gradient(self, encode, labels, key):
        if key not in self.validation_gradients:
            self.validation_gradients[key] = self.data_gradient(encode, labels)
        return self.validation_gradients[key]

    def score_admission(self, admitted_encode, admitted_y, validation_encode, validation_y):
        started = time.perf_counter()
        admitted_count = int(len(admitted_encode))
        admitted_gradient = self.data_gradient(admitted_encode, admitted_y)
        next_gradient = (
            self.current_row_count * self.current_gradient
            + admitted_count * admitted_gradient
        ) / (self.current_row_count + admitted_count)
        validation_was_cached = "current_validation" in self.validation_gradients
        validation_started = time.perf_counter()
        validation_gradient = self.validation_gradient(
            validation_encode,
            validation_y,
            "current_validation",
        )
        shared_setup_seconds = (
            0.0
            if validation_was_cached
            else float(time.perf_counter() - validation_started)
        )
        score = float(
            (validation_gradient @ (next_gradient - self.current_gradient))
            .detach()
            .cpu()
        )
        elapsed = float(time.perf_counter() - started)
        return {
            "score": score,
            "time_seconds": max(0.0, elapsed - shared_setup_seconds),
            "shared_setup_seconds": shared_setup_seconds,
        }

    def score_transition(
        self,
        next_encode,
        next_y,
        validation_encode,
        validation_y,
        *,
        candidate_id,
    ):
        started = time.perf_counter()
        next_gradient = self.data_gradient(next_encode, next_y)
        validation_gradient = self.validation_gradient(
            validation_encode,
            validation_y,
            ("vertical", str(candidate_id)),
        )
        return {
            "score": float(
                (validation_gradient @ (next_gradient - self.current_gradient))
                .detach()
                .cpu()
            ),
            "time_seconds": float(time.perf_counter() - started),
            "shared_setup_seconds": 0.0,
        }


@dataclass(frozen=True)
class InfluenceState:
    vector: torch.Tensor
    distance: float
    row_count: int
    cg_iterations: int
    cg_residual_norm: float
    time_seconds: float


class FixedOriginalInfluence:
    """Yang-style fixed-original implicit-Hessian influence representation."""

    def __init__(
        self,
        model,
        original_encode,
        original_y,
        *,
        parameter_scope,
        damping,
        cg_max_iter,
        cg_tolerance,
    ):
        if getattr(model, "network_", None) is None:
            raise ValueError("Yang-style IF requires a fitted original MLP.")
        if parameter_scope not in {"all", "last_layer"}:
            raise ValueError("parameter_scope must be all or last_layer.")
        self.model = model
        self.network = model.network_.to(device=model.device, dtype=model.dtype)
        self.network.eval()
        if parameter_scope == "all":
            self.parameters = [parameter for parameter in self.network.parameters() if parameter.requires_grad]
        else:
            layers = [module for module in self.network.modules() if isinstance(module, nn.Linear)]
            if not layers:
                raise ValueError("The original MLP has no Linear layer.")
            last_layer = layers[-1]
            self.parameters = [last_layer.weight]
            if last_layer.bias is not None:
                self.parameters.append(last_layer.bias)
        self.parameter_scope = parameter_scope
        self.device = model.device
        self.dtype = model.dtype
        self.allow_cpu_fallback = bool(model.allow_cpu_fallback)
        self.damping = float(damping)
        self.cg_max_iter = int(cg_max_iter)
        self.cg_tolerance = float(cg_tolerance)
        if self.damping < 0 or self.cg_max_iter <= 0 or self.cg_tolerance < 0:
            raise ValueError("Invalid Yang-style IF solver parameters.")
        self.original_X = self.features(original_encode)
        self.original_y = self.labels(original_y)
        self.original_gradient = self.data_gradient(
            self.original_X,
            self.original_y,
        )

    def features(self, encode):
        return to_device_tensor(
            encode,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )

    def labels(self, labels):
        return to_device_tensor(
            labels,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        ).reshape(-1)

    def data_gradient(self, X_tensor, y_tensor, *, create_graph=False):
        gradient = _gradient(
            _loss(self.model, self.network, X_tensor, y_tensor),
            self.parameters,
            create_graph=create_graph,
        )
        return gradient if create_graph else gradient.detach()

    def hessian_vector(self, vector):
        gradient = self.data_gradient(
            self.original_X,
            self.original_y,
            create_graph=True,
        )
        values = torch.autograd.grad(
            gradient @ vector,
            self.parameters,
            allow_unused=True,
        )
        regularization = float(getattr(self.model, "weight_decay", 0.0))
        return _flat(values, self.parameters).detach() + (
            regularization + self.damping
        ) * vector

    def solve(self, rhs):
        solution = torch.zeros_like(rhs)
        residual = rhs.detach().clone()
        direction = residual.clone()
        residual_sq = residual @ residual
        epsilon = torch.finfo(rhs.dtype).eps
        iterations = 0
        for iterations in range(1, self.cg_max_iter + 1):
            if float(torch.sqrt(residual_sq).detach().cpu()) <= self.cg_tolerance:
                iterations -= 1
                break
            product = self.hessian_vector(direction)
            denominator = direction @ product
            if float(torch.abs(denominator).detach().cpu()) <= epsilon:
                break
            alpha = residual_sq / denominator
            solution = solution + alpha * direction
            residual = residual - alpha * product
            next_residual_sq = residual @ residual
            if float(torch.sqrt(next_residual_sq).detach().cpu()) <= self.cg_tolerance:
                residual_sq = next_residual_sq
                break
            direction = residual + (
                next_residual_sq / residual_sq.clamp_min(epsilon)
            ) * direction
            residual_sq = next_residual_sq
        return (
            solution.detach(),
            int(iterations),
            float(torch.linalg.norm(residual).detach().cpu()),
        )

    def state(self, encode, labels):
        started = time.perf_counter()
        X_tensor = self.features(encode)
        y_tensor = self.labels(labels)
        gradient = self.data_gradient(X_tensor, y_tensor)
        vector, iterations, residual_norm = self.solve(
            gradient - self.original_gradient
        )
        return InfluenceState(
            vector=vector,
            distance=float(torch.linalg.norm(vector).detach().cpu()),
            row_count=int(X_tensor.shape[0]),
            cg_iterations=iterations,
            cg_residual_norm=residual_norm,
            time_seconds=float(time.perf_counter() - started),
        )

    def estimated_validation_loss(self, state, validation_encode, validation_y):
        X_validation = self.features(validation_encode)
        y_validation = self.labels(validation_y)
        loss = _loss(self.model, self.network, X_validation, y_validation)
        gradient = _gradient(loss, self.parameters).detach()
        return float((loss - gradient @ state.vector).detach().cpu())


class CandidateExperiment:
    """Collect comparable candidate measurements for Exp-32 or Exp-33.

    Candidate ranking is performed on the ordinary candidate state. The caller
    receives both the measurement pool and TRIM's Top-K keys, and materializes
    TRIM releases only for the latter. This keeps the ``rank first, Top-K next,
    release last`` order used by TRIM.
    """

    _RANKING_COLUMNS = (
        "seed",
        "iteration",
        "candidate_type",
        "candidate_id",
        "mlp_lga_score",
        "yang_if_distance_score",
        "random_score",
        "exact_utility_delta",
    )
    _RANKING_RUNTIME_COLUMNS = (
        "seed",
        "iteration",
        "method",
        "time_seconds",
        "candidate_count",
        "timing_scope",
    )
    _ESTIMATION_COLUMNS = (
        "seed",
        "iteration",
        "method",
        "selected_candidate_type",
        "selected_candidate_id",
        "oracle_candidate_type",
        "oracle_candidate_id",
        "estimated_utility_delta",
        "actual_utility_delta",
        "estimated_rho",
        "actual_rho",
        "oracle_actual_utility_delta",
        "oracle_actual_rho",
        "actual_utility_gap_to_oracle",
        "ratio_abs_error",
        "time_seconds",
    )

    def __init__(
        self,
        *,
        mode,
        seed,
        output_csv,
        runtime_output_csv=None,
        oracle_candidate_limit_per_type,
        horizontal_last,
        vertical_all,
    ):
        if mode not in {"ranking", "estimation"}:
            raise ValueError("mode must be 'ranking' or 'estimation'.")
        self.mode = mode
        self.seed = int(seed)
        self.output_csv = Path(output_csv).expanduser()
        if runtime_output_csv is None:
            runtime_output_csv = self.output_csv.with_name(
                f"{self.output_csv.stem}_runtime.csv"
            )
        self.runtime_output_csv = Path(runtime_output_csv).expanduser()
        if (
            self.mode == "ranking"
            and self.runtime_output_csv == self.output_csv
        ):
            raise ValueError(
                "Ranking candidate and runtime CSV paths must be different."
            )
        self.oracle_candidate_limit_per_type = int(
            oracle_candidate_limit_per_type
        )
        if self.oracle_candidate_limit_per_type <= 0:
            raise ValueError("oracle_candidate_limit_per_type must be positive.")
        self.horizontal_last = dict(horizontal_last)
        self.vertical_all = dict(vertical_all)
        required_solver_keys = {"damping", "cg_max_iter", "cg_tolerance"}
        for scope_name, solver in (
            ("horizontal_last", self.horizontal_last),
            ("vertical_all", self.vertical_all),
        ):
            missing = sorted(required_solver_keys - set(solver))
            if missing:
                raise ValueError(f"{scope_name} is missing {missing}.")
        self.rows = []
        self.runtime_rows = []
        self._reference_setup_seconds = {
            "last_layer": 0.0,
            "all": 0.0,
        }
        self._reference_setup_reported_scopes = set()

    def prepare_original(self, model, original_encode, original_y):
        """Freeze the Yang-style reference at the full original-data MLP."""

        started = time.perf_counter()
        self.yang_last = FixedOriginalInfluence(
            model,
            original_encode,
            original_y,
            parameter_scope="last_layer",
            damping=float(self.horizontal_last["damping"]),
            cg_max_iter=int(self.horizontal_last["cg_max_iter"]),
            cg_tolerance=float(self.horizontal_last["cg_tolerance"]),
        )
        self._reference_setup_seconds["last_layer"] = (
            time.perf_counter() - started
        )
        started = time.perf_counter()
        self.yang_all = FixedOriginalInfluence(
            model,
            original_encode,
            original_y,
            parameter_scope="all",
            damping=float(self.vertical_all["damping"]),
            cg_max_iter=int(self.vertical_all["cg_max_iter"]),
            cg_tolerance=float(self.vertical_all["cg_tolerance"]),
        )
        self._reference_setup_seconds["all"] = time.perf_counter() - started

    def begin_iteration(
        self,
        *,
        iteration,
        estimator_model,
        current_encode,
        current_y,
        validation_encode,
        validation_y,
        current_proxy_validation_loss=None,
        current_downstream_validation_loss=None,
    ):
        if not hasattr(self, "yang_last"):
            raise RuntimeError("prepare_original must be called before ranking.")
        self.iteration = int(iteration)
        self.current_encode = current_encode
        self.current_y = current_y
        self.validation_encode = validation_encode
        self.validation_y = validation_y
        self.current_proxy_validation_loss = (
            None
            if current_proxy_validation_loss is None
            else float(current_proxy_validation_loss)
        )
        self.current_downstream_validation_loss = (
            None
            if current_downstream_validation_loss is None
            else float(current_downstream_validation_loss)
        )
        if self.mode == "estimation" and (
            self.current_proxy_validation_loss is None
            or self.current_downstream_validation_loss is None
        ):
            raise ValueError(
                "Exp-33 requires the current proxy and downstream validation "
                "losses."
            )
        started = time.perf_counter()
        self.mlp_lga = MLPLGAScorer.build(
            estimator_model,
            current_encode,
            current_y,
        )
        self._mlp_setup_seconds = time.perf_counter() - started

        self.current_estimated_validation_loss = {}
        self._yang_iteration_setup_seconds = {}
        yang_started = time.perf_counter()
        self.current_yang_last = self.yang_last.state(
            current_encode,
            current_y,
        )
        if self.mode == "estimation":
            self.current_estimated_validation_loss["last_layer"] = (
                self.yang_last.estimated_validation_loss(
                    self.current_yang_last,
                    validation_encode,
                    validation_y,
                )
            )
        self._yang_iteration_setup_seconds["last_layer"] = (
            time.perf_counter() - yang_started
        )
        yang_started = time.perf_counter()
        self.current_yang_all = self.yang_all.state(
            current_encode,
            current_y,
        )
        if self.mode == "estimation":
            self.current_estimated_validation_loss["all"] = (
                self.yang_all.estimated_validation_loss(
                    self.current_yang_all,
                    validation_encode,
                    validation_y,
                )
            )
        self._yang_iteration_setup_seconds["all"] = (
            time.perf_counter() - yang_started
        )
        self.iteration_times = {
            "mlp_lga": 0.0,
            "yang_if": 0.0,
            "random": 0.0,
            "retrain": 0.0,
        }
        self._mlp_shared_setup_seconds = 0.0
        self._retrain_attempt_count = 0
        self.candidates = {}
        self.observations = {}

    @staticmethod
    def _candidate_key(candidate_type, candidate_id):
        return str(candidate_type), str(candidate_id)

    def _finish_score(
        self,
        *,
        candidate_type,
        candidate_id,
        mlp_result,
        next_yang_state,
        yang_reference,
        parameter_scope,
        validation_encode,
        validation_y,
        yang_started,
    ):
        current_state = (
            self.current_yang_last
            if parameter_scope == "last_layer"
            else self.current_yang_all
        )
        distance_score = float(
            current_state.distance - next_yang_state.distance
        )
        if self.mode == "ranking":
            yang_score = distance_score
        else:
            next_estimated_loss = yang_reference.estimated_validation_loss(
                next_yang_state,
                validation_encode,
                validation_y,
            )
            yang_score = float(
                self.current_estimated_validation_loss[parameter_scope]
                - next_estimated_loss
            )
        # Keep candidate-local work separate from setup. Each experiment adds
        # setup once when it summarizes the iteration runtime.
        yang_time = time.perf_counter() - yang_started

        random_started = time.perf_counter()
        random_score = random.Random(
            f"{self.seed}:{self.iteration}:{candidate_type}:{candidate_id}"
        ).random()
        random_time = time.perf_counter() - random_started
        mlp_time = float(mlp_result["time_seconds"])
        mlp_shared_setup = float(mlp_result.get("shared_setup_seconds", 0.0))
        self._mlp_shared_setup_seconds += mlp_shared_setup
        self.iteration_times["mlp_lga"] += mlp_time
        self.iteration_times["yang_if"] += yang_time
        self.iteration_times["random"] += random_time

        key = self._candidate_key(candidate_type, candidate_id)
        if key in self.candidates:
            raise ValueError(
                f"Duplicate candidate key in iteration {self.iteration}: {key!r}."
            )
        result = {
            "candidate_key": key,
            "candidate_type": key[0],
            "candidate_id": key[1],
            "mlp_lga_score": float(mlp_result["score"]),
            "yang_if_distance_score": distance_score,
            "yang_if_utility_delta": float(yang_score),
            "random_score": float(random_score),
            "mlp_lga_time_seconds": mlp_time,
            "mlp_lga_shared_setup_seconds": mlp_shared_setup,
            "yang_if_time_seconds": float(yang_time),
            "random_time_seconds": float(random_time),
            "yang_parameter_scope": parameter_scope,
        }
        self.candidates[key] = result
        return result

    def score_admission(
        self,
        *,
        candidate_type,
        candidate_id,
        admitted_encode,
        admitted_y,
    ):
        mlp_result = self.mlp_lga.score_admission(
            admitted_encode,
            admitted_y,
            self.validation_encode,
            self.validation_y,
        )
        next_encode = torch.cat((self.current_encode, admitted_encode), dim=0)
        next_y = torch.cat((self.current_y, admitted_y), dim=0)
        yang_started = time.perf_counter()
        next_state = self.yang_last.state(next_encode, next_y)
        return self._finish_score(
            candidate_type=candidate_type,
            candidate_id=candidate_id,
            mlp_result=mlp_result,
            next_yang_state=next_state,
            yang_reference=self.yang_last,
            parameter_scope="last_layer",
            validation_encode=self.validation_encode,
            validation_y=self.validation_y,
            yang_started=yang_started,
        )

    def score_transition(
        self,
        *,
        candidate_type,
        candidate_id,
        next_encode,
        next_y,
        validation_encode,
    ):
        mlp_result = self.mlp_lga.score_transition(
            next_encode,
            next_y,
            validation_encode,
            self.validation_y,
            candidate_id=candidate_id,
        )
        yang_started = time.perf_counter()
        next_state = self.yang_all.state(next_encode, next_y)
        return self._finish_score(
            candidate_type=candidate_type,
            candidate_id=candidate_id,
            mlp_result=mlp_result,
            next_yang_state=next_state,
            yang_reference=self.yang_all,
            parameter_scope="all",
            validation_encode=validation_encode,
            validation_y=self.validation_y,
            yang_started=yang_started,
        )

    def evaluation_candidates(self, candidates, *, rank_top_k, include_all=False):
        """Return the measured pool and the TRIM Top-K candidate keys."""

        primary = sorted(
            candidates,
            key=lambda candidate: candidate["mlp_lga_score"],
            reverse=True,
        )[: int(rank_top_k)]
        primary_keys = {candidate["candidate_key"] for candidate in primary}
        evaluation_keys = set(primary_keys)
        if self.mode == "estimation":
            oracle = sorted(
                candidates,
                key=lambda candidate: candidate["mlp_lga_score"],
                reverse=True,
            )[: self.oracle_candidate_limit_per_type]
            oracle_keys = {
                candidate["candidate_key"] for candidate in oracle
            }
            evaluation_keys.update(oracle_keys)
            if include_all:
                evaluation_keys.update(
                    candidate["candidate_key"] for candidate in candidates
                )
            else:
                for score_name in (
                    "mlp_lga_score",
                    "yang_if_utility_delta",
                ):
                    eligible = [
                        candidate for candidate in candidates
                        if float(candidate["privacy_cost"]) > 0.0
                    ]
                    if eligible:
                        selected = max(
                            eligible,
                            key=lambda candidate, name=score_name: (
                                float(candidate[name])
                                / float(candidate["privacy_cost"])
                            ),
                        )
                        evaluation_keys.add(selected["candidate_key"])
            for candidate in candidates:
                candidate["in_proxy_pool"] = (
                    candidate["candidate_key"] in primary_keys
                )
                candidate["in_oracle_pool"] = (
                    candidate["candidate_key"] in oracle_keys
                )
            return (
                [
                    candidate for candidate in candidates
                    if candidate["candidate_key"] in evaluation_keys
                ],
                primary_keys,
            )

        for score_name in (
            "mlp_lga_score",
            "yang_if_distance_score",
            "random_score",
        ):
            ranked = sorted(
                candidates,
                key=lambda candidate, name=score_name: candidate[name],
                reverse=True,
            )[: self.oracle_candidate_limit_per_type]
            evaluation_keys.update(
                candidate["candidate_key"] for candidate in ranked
            )
        return (
            [
                candidate
                for candidate in candidates
                if candidate["candidate_key"] in evaluation_keys
            ],
            primary_keys,
        )

    def record_retrain_attempt(self, time_seconds):
        """Record one exact TRIM-release retraining evaluation, including failures."""

        if self.mode != "ranking":
            raise RuntimeError("Exact candidate retrains are only defined for Exp-32.")
        elapsed = float(time_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("Exact retrain time must be finite and non-negative.")
        self.iteration_times["retrain"] += elapsed
        self._retrain_attempt_count += 1

    def observe_estimation_candidate(
        self,
        candidate,
        *,
        privacy_cost,
        downstream_validation_loss,
        downstream_time_seconds,
        proxy_validation_loss=None,
        proxy_time_seconds=0.0,
    ):
        """Record one Exp-33 candidate using independent model fits."""

        if self.mode != "estimation":
            raise RuntimeError("Estimation observations are only defined for Exp-33.")
        key = candidate["candidate_key"]
        record = dict(candidate)
        if key in self.observations:
            raise ValueError(
                f"Duplicate Exp-33 measurement in iteration {self.iteration}: "
                f"{key!r}."
            )
        privacy_cost = float(privacy_cost)
        downstream_loss = float(downstream_validation_loss)
        downstream_time = float(downstream_time_seconds)
        proxy_time = float(proxy_time_seconds)
        if not math.isfinite(privacy_cost) or privacy_cost <= 0.0:
            raise ValueError("Exp-33 privacy cost must be finite and positive.")
        if not math.isfinite(downstream_loss):
            raise ValueError("Exp-33 downstream validation loss must be finite.")
        if not math.isfinite(downstream_time) or downstream_time < 0.0:
            raise ValueError("Exp-33 downstream time must be finite and non-negative.")
        if not math.isfinite(proxy_time) or proxy_time < 0.0:
            raise ValueError("Exp-33 proxy time must be finite and non-negative.")
        proxy_utility_delta = None
        if proxy_validation_loss is not None:
            proxy_loss = float(proxy_validation_loss)
            if not math.isfinite(proxy_loss):
                raise ValueError("Exp-33 proxy validation loss must be finite.")
            proxy_utility_delta = float(
                self.current_proxy_validation_loss - proxy_loss
            )
            record["proxy_validation_loss"] = proxy_loss
        actual_utility_delta = float(
            self.current_downstream_validation_loss - downstream_loss
        )
        record.update({
            "privacy_cost": privacy_cost,
            "proxy_utility_delta": proxy_utility_delta,
            "proxy_time_seconds": proxy_time,
            "downstream_validation_loss": downstream_loss,
            "downstream_time_seconds": downstream_time,
            "actual_utility_delta": actual_utility_delta,
            "actual_rho": actual_utility_delta / privacy_cost,
        })
        self.observations[key] = record

    @staticmethod
    def _best_by(records, field, *, predicate=lambda _record: True):
        eligible = [
            record for record in records
            if predicate(record)
            and record.get(field) is not None
            and math.isfinite(float(record[field]))
        ]
        return (
            max(eligible, key=lambda record: float(record[field]))
            if eligible
            else None
        )

    def selected_estimation(self, selected_candidate_key):
        """Return the already measured losses for TRIM's selected action."""

        key = self._candidate_key(*selected_candidate_key)
        try:
            return dict(self.observations[key])
        except KeyError as exc:
            raise RuntimeError(
                f"Selected candidate {key!r} has no Exp-33 measurement."
            ) from exc

    def observe(
        self,
        candidate,
        *,
        exact_utility_delta,
        retrain_time_seconds,
        count_retrain_time=True,
    ):
        if self.mode != "ranking":
            raise RuntimeError("Exact candidate observations are only defined for Exp-32.")
        record = dict(candidate)
        key = candidate["candidate_key"]
        if key in self.observations:
            raise ValueError(
                f"Duplicate exact measurement in iteration {self.iteration}: "
                f"{key!r}."
            )
        record["exact_utility_delta"] = float(exact_utility_delta)
        record["retrain_time_seconds"] = float(retrain_time_seconds)
        if count_retrain_time:
            self.record_retrain_attempt(retrain_time_seconds)
        self.observations[key] = record

    def finish_iteration(
        self,
        selected_candidate_key,
        *,
        actual_utility_delta=None,
        downstream_time_seconds=None,
        trim_estimated_utility_delta=None,
        trim_proxy_time_seconds=None,
    ):
        if self.mode == "ranking":
            if any(value is not None for value in (
                actual_utility_delta,
                downstream_time_seconds,
                trim_estimated_utility_delta,
                trim_proxy_time_seconds,
            )):
                raise ValueError(
                    "Exp-32 finish_iteration does not accept Exp-33 downstream "
                    "measurements."
                )
            for record in self.observations.values():
                self.rows.append({
                    column: {
                        "seed": self.seed,
                        "iteration": self.iteration,
                    }.get(column, record.get(column))
                    for column in self._RANKING_COLUMNS
                })
            scored_candidate_count = len(self.candidates)
            if scored_candidate_count < len(self.observations):
                raise RuntimeError(
                    "Exact observations cannot outnumber scored candidates."
                )
            if self._retrain_attempt_count < len(self.observations):
                raise RuntimeError(
                    "Exact observations cannot outnumber retrain attempts."
                )
            mlp_total = (
                self.iteration_times["mlp_lga"]
                + self._mlp_setup_seconds
                + self._mlp_shared_setup_seconds
            )
            yang_total = (
                self.iteration_times["yang_if"]
                + sum(self._yang_iteration_setup_seconds.values())
            )
            for scope, elapsed in self._reference_setup_seconds.items():
                if scope not in self._reference_setup_reported_scopes:
                    yang_total += elapsed
                    self._reference_setup_reported_scopes.add(scope)
            runtime_rows = (
                (
                    "trim",
                    mlp_total,
                    scored_candidate_count,
                    "all_ranked_candidates_including_setup",
                ),
                (
                    "if",
                    yang_total,
                    scored_candidate_count,
                    "all_ranked_candidates_including_setup",
                ),
                (
                    "random",
                    self.iteration_times["random"],
                    scored_candidate_count,
                    "all_ranked_candidates_including_setup",
                ),
                (
                    "retrain",
                    self.iteration_times["retrain"],
                    self._retrain_attempt_count,
                    "all_exact_evaluation_attempts_in_common_pool",
                ),
            )
            for method, elapsed, candidate_count, timing_scope in runtime_rows:
                self.runtime_rows.append({
                    "seed": self.seed,
                    "iteration": self.iteration,
                    "method": method,
                    "time_seconds": float(elapsed),
                    "candidate_count": int(candidate_count),
                    "timing_scope": timing_scope,
                })
            return

        if len(selected_candidate_key) != 2:
            raise ValueError(
                "selected_candidate_key must contain candidate_type and "
                "candidate_id."
            )
        key = self._candidate_key(
            selected_candidate_key[0],
            selected_candidate_key[1],
        )
        if key not in self.observations:
            raise RuntimeError(f"Selected candidate {key!r} has no Exp-33 measurement.")
        if actual_utility_delta is not None or downstream_time_seconds is not None:
            raise ValueError(
                "Exp-33 candidate measurements are recorded before finish_iteration."
            )
        if trim_estimated_utility_delta is None or trim_proxy_time_seconds is None:
            raise ValueError(
                "Exp-33 requires TRIM's selected estimate and recorded proxy time."
            )
        trim_estimate = float(trim_estimated_utility_delta)
        trim_proxy_time = float(trim_proxy_time_seconds)
        if not math.isfinite(trim_estimate):
            raise ValueError("TRIM's selected Utility Delta must be finite.")
        if not math.isfinite(trim_proxy_time) or trim_proxy_time < 0.0:
            raise ValueError("TRIM's proxy time must be finite and non-negative.")
        records = list(self.observations.values())
        trim_selected = self.observations[key]
        trim_selected["trim_estimated_utility_delta"] = trim_estimate
        trim_selected["trim_estimated_rho"] = (
            trim_estimate / trim_selected["privacy_cost"]
        )
        oracle = self._best_by(
            records,
            "actual_rho",
            predicate=lambda record: bool(record["in_oracle_pool"]),
        )
        if oracle is None:
            raise RuntimeError("Exp-33 approximate oracle has no evaluated candidate.")
        mlp_total = (
            self.iteration_times["mlp_lga"]
            + self._mlp_setup_seconds
            + self._mlp_shared_setup_seconds
        )
        yang_total = (
            self.iteration_times["yang_if"]
            + sum(self._yang_iteration_setup_seconds.values())
        )
        for scope, elapsed in self._reference_setup_seconds.items():
            if scope not in self._reference_setup_reported_scopes:
                yang_total += elapsed
                self._reference_setup_reported_scopes.add(scope)
        method_times = {
            "trim": mlp_total + trim_proxy_time,
            "if": yang_total,
            "lga": mlp_total,
            "retrain": mlp_total + sum(
                record["downstream_time_seconds"]
                for record in records if record["in_oracle_pool"]
            ),
        }
        for record in records:
            privacy_cost = record["privacy_cost"]
            record["if_estimated_rho"] = (
                record["yang_if_utility_delta"] / privacy_cost
            )
            record["lga_estimated_rho"] = (
                record["mlp_lga_score"] / privacy_cost
            )
        selections = {
            "trim": trim_selected,
            "if": self._best_by(records, "if_estimated_rho"),
            "lga": self._best_by(records, "lga_estimated_rho"),
            "retrain": oracle,
        }
        estimate_fields = {
            "trim": ("trim_estimated_utility_delta", "trim_estimated_rho"),
            "if": ("yang_if_utility_delta", "if_estimated_rho"),
            "lga": ("mlp_lga_score", "lga_estimated_rho"),
            "retrain": ("actual_utility_delta", "actual_rho"),
        }
        for method in ("trim", "if", "lga", "retrain"):
            selected = selections[method]
            if selected is None:
                raise RuntimeError(f"Exp-33 method {method!r} selected no candidate.")
            utility_field, rho_field = estimate_fields[method]
            self.rows.append({
                "seed": self.seed,
                "iteration": self.iteration,
                "method": method,
                "selected_candidate_type": selected["candidate_type"],
                "selected_candidate_id": selected["candidate_id"],
                "oracle_candidate_type": oracle["candidate_type"],
                "oracle_candidate_id": oracle["candidate_id"],
                "estimated_utility_delta": float(selected[utility_field]),
                "actual_utility_delta": float(selected["actual_utility_delta"]),
                "estimated_rho": float(selected[rho_field]),
                "actual_rho": float(selected["actual_rho"]),
                "oracle_actual_utility_delta": float(
                    oracle["actual_utility_delta"]
                ),
                "oracle_actual_rho": float(oracle["actual_rho"]),
                "actual_utility_gap_to_oracle": float(abs(
                    selected["actual_utility_delta"]
                    - oracle["actual_utility_delta"]
                )),
                "ratio_abs_error": float(abs(
                    selected["actual_rho"] - oracle["actual_rho"]
                )),
                "time_seconds": float(method_times[method]),
            })

    def write(self):
        if not self.rows:
            raise ValueError("The experiment produced no candidate measurements.")
        if self.mode == "ranking" and not self.runtime_rows:
            raise ValueError("The ranking experiment produced no runtime rows.")
        columns = (
            self._RANKING_COLUMNS
            if self.mode == "ranking"
            else self._ESTIMATION_COLUMNS
        )
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        if self.mode == "ranking":
            self.runtime_output_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.runtime_output_csv.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=self._RANKING_RUNTIME_COLUMNS,
                )
                writer.writeheader()
                writer.writerows(self.runtime_rows)


__all__ = [
    "CandidateExperiment",
    "FixedOriginalInfluence",
    "InfluenceState",
    "MLPLGAScorer",
]
