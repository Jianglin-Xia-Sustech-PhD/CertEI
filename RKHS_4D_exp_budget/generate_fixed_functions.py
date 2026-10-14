"""Generate, optimize, visualize, and freeze the two RKHS objectives."""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

from rkhs_functions import (
    FixedRKHSFunction,
    KernelName,
    kernel_matrix,
    save_fixed_function,
)
from study_config import (
    DIMENSION,
    NOISE_SD_MAX_MULTIPLIER,
    NOISE_SD_MIN_MULTIPLIER,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "fixed_functions"
MASTER_SEED = 20260827
N_BASIS = 100
LENGTHSCALE = 0.2


def _evaluate(
    points: np.ndarray,
    basis_points: np.ndarray,
    weights: np.ndarray,
    kernel: KernelName,
) -> np.ndarray:
    values = np.empty(points.shape[0], dtype=float)
    chunk_size = 8192
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        values[start:stop] = (
            kernel_matrix(
                points[start:stop], basis_points, kernel, LENGTHSCALE
            )
            @ weights
        )
    return values


def _global_extreme(
    basis_points: np.ndarray,
    weights: np.ndarray,
    kernel: KernelName,
    *,
    maximize: bool,
    seed: int,
) -> tuple[np.ndarray, float, dict[str, float | int]]:
    sign = -1.0 if maximize else 1.0

    def scalar(point: np.ndarray) -> float:
        value = kernel_matrix(
            np.asarray(point).reshape(1, DIMENSION),
            basis_points,
            kernel,
            LENGTHSCALE,
        ) @ weights
        return sign * float(value[0])

    sobol = qmc.Sobol(d=DIMENSION, scramble=True, seed=seed)
    screening_points = sobol.random_base2(m=18)
    screening_values = _evaluate(
        screening_points, basis_points, weights, kernel
    )
    ordering = np.argsort(sign * screening_values)
    starts = screening_points[ordering[:48]]
    starts = np.vstack(
        (
            starts,
            basis_points,
            np.asarray(list(product((0.0, 1.0), repeat=DIMENSION))),
        )
    )

    de_result = differential_evolution(
        scalar,
        bounds=[(0.0, 1.0)] * DIMENSION,
        seed=seed,
        tol=1e-11,
        atol=1e-12,
        popsize=24,
        maxiter=700,
        polish=True,
        updating="immediate",
        workers=1,
    )
    candidates: list[tuple[float, np.ndarray]] = [
        (float(de_result.fun), np.asarray(de_result.x, dtype=float))
    ]
    for start in starts:
        result = minimize(
            scalar,
            start,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * DIMENSION,
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000},
        )
        candidates.append((float(result.fun), np.asarray(result.x, dtype=float)))
    candidates.sort(key=lambda item: item[0])
    point = np.clip(candidates[0][1], 0.0, 1.0)
    value = float(
        _evaluate(
            point.reshape(1, DIMENSION), basis_points, weights, kernel
        )[0]
    )
    diagnostics: dict[str, float | int] = {
        "sobol_points": int(screening_points.shape[0]),
        "multistart_count": int(starts.shape[0]),
        "differential_evolution_evaluations": int(de_result.nfev),
        "screening_extreme": float(
            np.max(screening_values) if maximize else np.min(screening_values)
        ),
        "refined_extreme": value,
    }
    return point, value, diagnostics


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _plot(function: FixedRKHSFunction) -> None:
    """Plot the x1-x2 slice through the four-dimensional global minimizer."""

    axis = np.linspace(0.0, 1.0, 301)
    xx, yy = np.meshgrid(axis, axis)
    points = np.tile(function.optimum_x, (xx.size, 1))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()
    values = function.evaluate(points).reshape(xx.shape)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    filled = axes[0].contourf(xx, yy, values, levels=40, cmap="viridis")
    axes[0].contour(xx, yy, values, levels=18, colors="black", linewidths=0.25, alpha=0.5)
    axes[0].scatter(
        function.optimum_x[0], function.optimum_x[1], marker="*", s=150,
        c="red", edgecolors="white", linewidths=0.8, label="global minimum",
    )
    axes[0].set(
        xlabel="$x_1$",
        ylabel="$x_2$",
        title=f"{function.name}: $x_1$-$x_2$ slice at $x^*$",
    )
    axes[0].legend(loc="best", fontsize=8)
    figure.colorbar(filled, ax=axes[0], label="$f(x)$")

    noise = function.noise_standard_deviation(points).reshape(xx.shape)
    noise_filled = axes[1].contourf(xx, yy, noise, levels=40, cmap="magma")
    axes[1].scatter(
        function.optimum_x[0], function.optimum_x[1], marker="*", s=150,
        c="cyan", edgecolors="black", linewidths=0.6,
    )
    axes[1].set(
        xlabel="$x_1$", ylabel="$x_2$",
        title="Heteroscedastic noise SD on the same slice",
    )
    figure.colorbar(noise_filled, ax=axes[1], label="$\\sigma(x)$")
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"{function.name}_optimum_slice.{suffix}", dpi=220)
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(MASTER_SEED)
    basis_points = rng.uniform(0.0, 1.0, size=(N_BASIS, DIMENSION))
    weights = rng.standard_normal(N_BASIS)
    shared_hash = _hash_arrays(basis_points, weights)
    manifest: dict[str, object] = {
        "master_seed": MASTER_SEED,
        "dimension": DIMENSION,
        "domain": [[0.0, 1.0]] * DIMENSION,
        "n_basis": N_BASIS,
        "weight_distribution": "standard normal",
        "shared_basis_and_weights": True,
        "basis_weight_sha256": shared_hash,
        "lengthscale": LENGTHSCALE,
        "functions": {},
    }
    for index, (name, kernel) in enumerate(
        (("rkhs_matern52_4d", "matern52"), ("rkhs_se_4d", "squared_exponential"))
    ):
        minimum_x, minimum_value, minimum_diagnostics = _global_extreme(
            basis_points,
            weights,
            kernel,  # type: ignore[arg-type]
            maximize=False,
            seed=MASTER_SEED + 101 + index,
        )
        maximum_x, maximum_value, maximum_diagnostics = _global_extreme(
            basis_points,
            weights,
            kernel,  # type: ignore[arg-type]
            maximize=True,
            seed=MASTER_SEED + 201 + index,
        )
        gram = kernel_matrix(basis_points, basis_points, kernel, LENGTHSCALE)  # type: ignore[arg-type]
        norm_squared = float(weights @ gram @ weights)
        function = FixedRKHSFunction(
            name=name,
            kernel=kernel,  # type: ignore[arg-type]
            lengthscale=LENGTHSCALE,
            basis_points=basis_points.copy(),
            weights=weights.copy(),
            optimum_x=minimum_x,
            optimum_value=minimum_value,
            maximum_x=maximum_x,
            maximum_value=maximum_value,
            rkhs_norm=float(np.sqrt(max(norm_squared, 0.0))),
            seed=MASTER_SEED,
        )
        path = save_fixed_function(function, OUTPUT / f"{name}.npz")
        _plot(function)
        function_manifest = {
            "file": path.name,
            "kernel": kernel,
            "kernel_form": "radial/isotropic scaled Euclidean distance",
            "matern_nu": 2.5 if kernel == "matern52" else None,
            "lengthscale": LENGTHSCALE,
            "optimum_x": minimum_x.tolist(),
            "optimum_value": minimum_value,
            "maximum_x": maximum_x.tolist(),
            "maximum_value": maximum_value,
            "response_range": function.response_range,
            "rkhs_norm": function.rkhs_norm,
            "noise_sd_min_multiplier": NOISE_SD_MIN_MULTIPLIER,
            "noise_sd_max_multiplier": NOISE_SD_MAX_MULTIPLIER,
            "noise_sd_min": NOISE_SD_MIN_MULTIPLIER * function.response_range,
            "noise_sd_max": NOISE_SD_MAX_MULTIPLIER * function.response_range,
            "noise_variance_min": function.variance_min,
            "noise_variance_max": function.variance_max,
            "minimum_search": minimum_diagnostics,
            "maximum_search": maximum_diagnostics,
        }
        manifest["functions"][name] = function_manifest  # type: ignore[index]
        print(
            f"{name}: x*={minimum_x.tolist()}, f*={minimum_value:.12g}, "
            f"range={function.response_range:.12g}, ||f||_H={function.rkhs_norm:.12g}",
            flush=True,
        )

    with (OUTPUT / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    np.savetxt(
        OUTPUT / "shared_basis_points.csv",
        basis_points,
        delimiter=",",
        header=",".join(f"x{index}" for index in range(1, DIMENSION + 1)),
        comments="",
    )
    np.savetxt(
        OUTPUT / "shared_weights.csv",
        weights.reshape(-1, 1),
        delimiter=",",
        header="weight",
        comments="",
    )


if __name__ == "__main__":
    main()
