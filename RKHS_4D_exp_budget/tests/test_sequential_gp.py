from __future__ import annotations

import numpy as np

from rkhs_functions import kernel_matrix
from sequential_gp import SequentialFixedKernelGP


def test_incremental_posterior_matches_direct_formula() -> None:
    universe = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.3, 0.1, 0.4],
            [0.7, 0.8, 0.6, 0.9],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    indices = np.asarray([0, 2, 2])
    outputs = np.asarray([1.0, -0.4, -0.2])
    noise = np.asarray([0.2, 0.1, 0.3])
    process_variance = 1.7
    constant_mean = 0.25
    model = SequentialFixedKernelGP(
        universe,
        "matern52",
        0.2,
        process_variance,
        constant_mean,
        len(indices),
    )
    for index, output, variance in zip(indices, outputs, noise):
        model.add_observation(int(index), float(output), float(variance))

    training = universe[indices]
    covariance = process_variance * kernel_matrix(
        training, training, "matern52", 0.2
    ) + np.diag(noise)
    cross = process_variance * kernel_matrix(
        universe, training, "matern52", 0.2
    )
    residual = outputs - constant_mean
    expected_mean = constant_mean + cross @ np.linalg.solve(covariance, residual)
    expected_variance = process_variance - np.sum(
        cross * np.linalg.solve(covariance, cross.T).T, axis=1
    )
    np.testing.assert_allclose(model.posterior_mean, expected_mean, atol=1e-10)
    np.testing.assert_allclose(model.posterior_variance, expected_variance, atol=1e-10)

    queries = np.asarray(
        [
            [0.1, 0.25, 0.2, 0.35],
            [0.8, 0.7, 0.7, 0.8],
        ]
    )
    query_cross = process_variance * kernel_matrix(
        queries, training, "matern52", 0.2
    )
    query_mean = constant_mean + query_cross @ np.linalg.solve(
        covariance, residual
    )
    query_variance = process_variance - np.sum(
        query_cross * np.linalg.solve(covariance, query_cross.T).T,
        axis=1,
    )
    actual_mean, actual_variance = model.posterior_at(queries)
    np.testing.assert_allclose(actual_mean, query_mean, atol=1e-10)
    np.testing.assert_allclose(actual_variance, query_variance, atol=1e-10)

    appended = model.append_points(queries)
    np.testing.assert_array_equal(appended, np.asarray([4, 5]))
    np.testing.assert_allclose(model.posterior_mean[appended], query_mean, atol=1e-10)
    np.testing.assert_allclose(
        model.posterior_variance[appended], query_variance, atol=1e-10
    )


def test_continuous_posterior_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(13)
    universe = rng.uniform(0.0, 1.0, size=(9, 4))
    model = SequentialFixedKernelGP(
        universe, "squared_exponential", 0.3, 1.2, -0.1, 5
    )
    for index, output, noise in zip(
        (0, 2, 4, 7), (0.4, -0.8, 0.1, -0.2), (0.2, 0.3, 0.1, 0.25)
    ):
        model.add_observation(index, output, noise)
    point = rng.uniform(0.25, 0.75, size=4)
    mean, variance, mean_gradient, variance_gradient = (
        model.posterior_with_gradients(point)
    )
    step = 1e-6
    finite_mean = np.empty(4)
    finite_variance = np.empty(4)
    for coordinate in range(4):
        upper = point.copy()
        lower = point.copy()
        upper[coordinate] += step
        lower[coordinate] -= step
        upper_mean, upper_variance = model.posterior_at(upper)
        lower_mean, lower_variance = model.posterior_at(lower)
        finite_mean[coordinate] = (upper_mean[0] - lower_mean[0]) / (2 * step)
        finite_variance[coordinate] = (
            upper_variance[0] - lower_variance[0]
        ) / (2 * step)
    assert np.isfinite(mean)
    assert variance >= 0.0
    np.testing.assert_allclose(mean_gradient, finite_mean, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(
        variance_gradient, finite_variance, rtol=2e-5, atol=2e-7
    )

    batch_points = np.vstack((point, rng.uniform(0.25, 0.75, size=4)))
    batch = model.posterior_with_gradients_batch(batch_points)
    singles = [model.posterior_with_gradients(item) for item in batch_points]
    np.testing.assert_allclose(batch[0], [item[0] for item in singles], atol=1e-12)
    np.testing.assert_allclose(batch[1], [item[1] for item in singles], atol=1e-12)
    np.testing.assert_allclose(batch[2], [item[2] for item in singles], atol=1e-12)
    np.testing.assert_allclose(batch[3], [item[3] for item in singles], atol=1e-12)
    indices = np.asarray([1, 3, 5])
    regular = model.posterior_with_gradients_batch(model.universe[indices])
    projected = model.posterior_with_gradients_batch(
        model.universe[indices], universe_indices=indices
    )
    for expected, actual in zip(regular, projected):
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_long_same_site_history_is_rebuilt_without_changing_posterior() -> None:
    universe = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.8, 0.7, 0.6, 0.5],
            [0.2, 0.8, 0.4, 0.6],
        ]
    )
    observation_count = 2049
    indices = np.arange(observation_count, dtype=int) % 2
    outputs = np.sin(np.arange(observation_count) / 17.0)
    noise = 0.05 + 0.01 * (np.arange(observation_count) % 3)
    model = SequentialFixedKernelGP(
        universe,
        "squared_exponential",
        0.25,
        1.2,
        -0.1,
        observation_count,
    )
    for index, output, variance in zip(indices, outputs, noise):
        model.add_observation(int(index), float(output), float(variance))

    precision = np.asarray(
        [np.sum(1.0 / noise[indices == site]) for site in (0, 1)]
    )
    equivalent_outputs = np.asarray(
        [
            np.sum(outputs[indices == site] / noise[indices == site])
            / precision[site]
            for site in (0, 1)
        ]
    )
    training = universe[:2]
    covariance = 1.2 * kernel_matrix(
        training, training, "squared_exponential", 0.25
    ) + np.diag(1.0 / precision)
    cross = 1.2 * kernel_matrix(
        universe, training, "squared_exponential", 0.25
    )
    expected_mean = -0.1 + cross @ np.linalg.solve(
        covariance, equivalent_outputs + 0.1
    )
    expected_variance = 1.2 - np.sum(
        cross * np.linalg.solve(covariance, cross.T).T, axis=1
    )
    np.testing.assert_allclose(model.posterior_mean, expected_mean, atol=2e-9)
    np.testing.assert_allclose(
        model.posterior_variance, expected_variance, atol=2e-9
    )
    assert model.count == observation_count
