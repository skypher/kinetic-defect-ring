#!/usr/bin/env python3
"""Short numerical audits for the manuscript formulas."""

from __future__ import annotations

import argparse
import math

import numpy as np


def generator(n: int, omega: float, omega0: float) -> np.ndarray:
    matrix = np.zeros((2 * n, 2 * n), dtype=float)
    for j in range(n):
        rate = omega0 if j == 0 else omega
        matrix[2 * j, 2 * ((j - 1) % n)] += 1.0
        matrix[2 * j, 2 * j] -= 1.0 + rate
        matrix[2 * j, 2 * j + 1] += rate
        matrix[2 * j + 1, 2 * ((j + 1) % n) + 1] += 1.0
        matrix[2 * j + 1, 2 * j + 1] -= 1.0 + rate
        matrix[2 * j + 1, 2 * j] += rate
    return matrix


def d_factor(lam: complex, omega: float, q: float) -> complex:
    c = math.cos(q)
    s = math.sin(q)
    return (lam + 1.0 - c) * (lam + 1.0 + 2.0 * omega - c) + s * s


def finite_green(n: int, omega: float, lam: complex) -> complex:
    wave_numbers = 2.0 * math.pi * np.arange(n) / n
    cosines = np.cos(wave_numbers)
    numerators = lam + 1.0 - cosines
    denominators = numerators * (
        lam + 1.0 + 2.0 * omega - cosines
    ) + np.sin(wave_numbers) ** 2
    return complex(np.sum(numerators / denominators) / n)


def finite_green_derivative(n: int, omega: float, lam: complex) -> complex:
    wave_numbers = 2.0 * math.pi * np.arange(n) / n
    cosines = np.cos(wave_numbers)
    numerators = lam + 1.0 - cosines
    denominators = numerators * (
        lam + 1.0 + 2.0 * omega - cosines
    ) + np.sin(wave_numbers) ** 2
    denominator_derivatives = 2.0 * (lam + 1.0 + omega - cosines)
    return complex(
        np.sum(
            (denominators - numerators * denominator_derivatives)
            / (denominators * denominators)
        )
        / n
    )


def homogeneous_polynomial(n: int, omega: float, lam: complex) -> complex:
    value = 1.0 + 0.0j
    for k in range(n):
        value *= d_factor(lam, omega, 2.0 * math.pi * k / n)
    return value


def exact_polynomial(n: int, omega: float, omega0: float, lam: complex) -> complex:
    factors = [d_factor(lam, omega, 2.0 * math.pi * k / n) for k in range(n)]
    base = np.prod(np.asarray(factors, dtype=complex))
    correction = 0.0 + 0.0j
    for k, factor in enumerate(factors):
        q = 2.0 * math.pi * k / n
        numerator = 2.0 * (lam + 1.0 - math.cos(q))
        correction += numerator * np.prod(
            np.asarray(factors[:k] + factors[k + 1 :], dtype=complex)
        )
    return base + (omega0 - omega) * correction / n


def spectral_gap(n: int, omega: float, omega0: float) -> float:
    values = np.linalg.eigvals(generator(n, omega, omega0))
    if float(np.min(np.abs(values))) > 2.0e-9:
        raise AssertionError("stationary eigenvalue audit failed")
    if float(np.max(values.real)) > 2.0e-9:
        raise AssertionError("Markov spectral half-plane audit failed")
    nonzero = values[np.abs(values) > 1.0e-9]
    return -float(np.max(nonzero.real))


def homogeneous_gap(n: int, omega: float) -> float:
    q = 2.0 * math.pi / n
    return (
        1.0
        - math.cos(q)
        + omega
        - math.sqrt(max(0.0, omega * omega - math.sin(q) ** 2))
    )


def determinant_audit(n: int, omega: float, omega0: float) -> None:
    matrix = generator(n, omega, omega0)
    samples = [0.37 + 0.19j, -0.8 + 0.31j, -2.4 + 0.07j]
    for lam in samples:
        direct = np.linalg.det(lam * np.eye(2 * n) - matrix)
        formula = exact_polynomial(n, omega, omega0, lam)
        scale = max(1.0, abs(direct), abs(formula))
        error = abs(direct - formula) / scale
        print(f"determinant lambda={lam} relative_error={error:.3e}", flush=True)
        if error > 2.0e-9:
            raise AssertionError("characteristic formula audit failed")


def transfer_audit(n: int, omega: float, omega0: float) -> None:
    samples = [0.37 + 0.19j, -0.8 + 0.31j, -2.4 + 0.07j]
    for lam in samples:
        monodromy = np.eye(2, dtype=complex)
        rates = [omega0] + [omega] * (n - 1)
        denominator = 1.0 + 0.0j
        for rate in rates:
            b = lam + 1.0 + rate
            transfer = np.array(
                [[1.0, rate], [-rate, b * b - rate * rate]],
                dtype=complex,
            ) / b
            monodromy = transfer @ monodromy
            denominator *= b
        direct = exact_polynomial(n, omega, omega0, lam)
        formula = denominator * (np.trace(monodromy) - 2.0)
        scale = max(1.0, abs(direct), abs(formula))
        error = abs(direct - formula) / scale
        print(f"transfer lambda={lam} relative_error={error:.3e}", flush=True)
        if error > 2.0e-9:
            raise AssertionError("transfer determinant audit failed")


def small_ring_formula_audit() -> None:
    for n in range(2, 10):
        omega = 0.51 + 0.13 * n
        omega0 = 0.0 if n % 3 == 0 else 1.37
        lam = -0.23 + (0.11 + 0.017 * n) * 1.0j
        matrix = generator(n, omega, omega0)
        direct = np.linalg.det(lam * np.eye(2 * n) - matrix)
        polynomial = exact_polynomial(n, omega, omega0, lam)

        rates = [omega0] + [omega] * (n - 1)
        monodromy = np.eye(2, dtype=complex)
        denominator = 1.0 + 0.0j
        for rate in rates:
            b = lam + 1.0 + rate
            transfer = np.array(
                [[1.0, rate], [-rate, b * b - rate * rate]],
                dtype=complex,
            ) / b
            monodromy = transfer @ monodromy
            denominator *= b
        transfer_value = denominator * (np.trace(monodromy) - 2.0)
        scale = max(1.0, abs(direct), abs(polynomial), abs(transfer_value))
        determinant_error = abs(direct - polynomial) / scale
        transfer_error = abs(direct - transfer_value) / scale
        print(
            f"small_ring_formula n={n}"
            f" determinant_error={determinant_error:.3e}"
            f" transfer_error={transfer_error:.3e}",
            flush=True,
        )
        if max(determinant_error, transfer_error) > 2.0e-9:
            raise AssertionError("small-ring formula audit failed")


def gap_audit(n: int, omega: float, omega0: float) -> None:
    exact_gap = spectral_gap(n, omega, omega0)
    protected_gap = homogeneous_gap(n, omega)
    print(
        "gap"
        f" exact={exact_gap:.12g}"
        f" protected={protected_gap:.12g}"
        f" difference={exact_gap-protected_gap:.3e}",
        flush=True,
    )
    if exact_gap < -2.0e-9:
        raise AssertionError("spectral gap sign audit failed")


def fixed_gap_asymptotic_audit() -> None:
    omega = 1.3
    n = 80

    slow_omega0 = 0.4
    slow_gap = spectral_gap(n, omega, slow_omega0)
    protected_gap = homogeneous_gap(n, omega)
    slow_error = abs(slow_gap - protected_gap)
    print(
        f"fixed_gap_slow n={n} error={slow_error:.3e}",
        flush=True,
    )
    if slow_error > 5.0e-10:
        raise AssertionError("fixed slow-defect gap audit failed")

    fast_omega0 = 2.0
    fast_gap = spectral_gap(n, omega, fast_omega0)
    coefficient = (
        4.0
        * math.pi**2
        * (fast_omega0 - omega)
        * (1.0 + omega)
        / (omega * omega * (1.0 + fast_omega0))
    )
    predicted = protected_gap - coefficient * n ** (-3)
    scaled_error = abs(fast_gap - predicted) * n**4
    print(
        f"fixed_gap_fast n={n} scaled_n4_error={scaled_error:.9g}",
        flush=True,
    )
    if scaled_error > 5.0:
        raise AssertionError("fixed fast-defect gap audit failed")


def localization_audit(omega: float, omega0: float) -> None:
    delta = omega0 - omega
    coefficients = [
        2.0 * delta,
        2.0 * delta * omega + 1.0 - omega * omega,
        4.0 * delta * delta * omega
        + 4.0 * delta * omega * omega
        - 2.0 * delta,
        2.0 * delta * omega + omega * omega - 1.0,
    ]
    while coefficients and abs(coefficients[0]) < 1.0e-14:
        coefficients.pop(0)
    if len(coefficients) < 2:
        print("localization no nontrivial cubic", flush=True)
        return
    for z in np.roots(coefficients):
        if 1.0e-10 < abs(z) < 1.0 - 1.0e-9:
            denominator = 1.0 - z * z + 2.0 * delta * z
            if abs(denominator) < 1.0e-10:
                continue
            a = 2.0 * delta * z * (omega + z) / denominator
            lam = a - 1.0 - omega
            dispersion_defect = a * a + 1.0 - omega * omega - a * (z + 1.0 / z)
            match_defect = a * (1.0 - z * z) + 2.0 * delta * z * (
                a - omega - z
            )
            print(
                "localized_candidate"
                f" z={z} lambda={lam}"
                f" dispersion_defect={abs(dispersion_defect):.3e}"
                f" match_defect={abs(match_defect):.3e}",
                flush=True,
            )
            if max(abs(dispersion_defect), abs(match_defect)) > 2.0e-10:
                raise AssertionError("localization identity audit failed")


def flat_band_audit() -> None:
    cases = []
    for n in range(2, 13):
        homogeneous_expected = (
            [n, n + 2, n + 2] if n % 4 == 0 else [n, n, n]
        )
        cases.append((n, 1.0, homogeneous_expected, "homogeneous"))

        if n % 4 in (1, 2):
            generic_expected = [n - 1, n - 1, n - 1]
        else:
            generic_expected = [n - 1, n, n]
        cases.append((n, 0.4, generic_expected, "generic_defect"))

        if n % 4 == 3:
            tuned = (n - 1.0) / (n + 1.0)
            cases.append((n, tuned, [n - 1, n, n + 1], "tuned_J3"))

    for n, omega0, expected, regime in cases:
        nilpotent = generator(n, 1.0, omega0) + 2.0 * np.eye(2 * n)
        nullities = []
        power = np.eye(2 * n)
        for exponent in range(1, 4):
            power = power @ nilpotent
            rank = np.linalg.matrix_rank(power, tol=1.0e-9)
            nullities.append(2 * n - rank)
        print(
            f"flat_band n={n} regime={regime} omega0={omega0}"
            f" nullities={nullities}",
            flush=True,
        )
        if nullities != expected:
            raise AssertionError("flat-band Jordan audit failed")


def embedded_flat_band_eigenspace_audit() -> None:
    n = 11
    base = np.arange(1.0, n + 1.0)
    for omega0 in [0.0, 0.4, 1.0, 1.8]:
        p = base.astype(complex)
        if omega0 != 1.0:
            p[0] = -p[-1]
        vector = np.empty(2 * n, dtype=complex)
        vector[0::2] = p
        vector[1::2] = -np.roll(p, 1)
        residual = np.linalg.norm(
            (generator(n, 1.0, omega0) + 2.0 * np.eye(2 * n)) @ vector
        )
        constraint = abs((omega0 - 1.0) * (p[0] + p[-1]))
        print(
            "embedded_flat_band"
            f" n={n} omega0={omega0}"
            f" constraint={constraint:.3e} residual={residual:.3e}",
            flush=True,
        )
        if max(constraint, residual) > 2.0e-10:
            raise AssertionError("embedded flat-band eigenspace audit failed")


def compact_localization_audit() -> None:
    omega = 2.0
    omega0 = (omega * omega + 1.0) / (2.0 * omega)
    lam = -1.0 - omega
    for n in [2, 3, 4, 7, 12]:
        shifted = generator(n, omega, omega0) - lam * np.eye(2 * n)
        singular_values = np.linalg.svd(shifted, compute_uv=False)
        residual = float(singular_values[-1])
        print(
            f"compact_localization n={n} lambda={lam}"
            f" smallest_singular_value={residual:.3e}",
            flush=True,
        )
        if residual > 2.0e-9:
            raise AssertionError("compact localization audit failed")


def zero_denominator_localization_audit() -> None:
    omega = 0.5
    omega0 = (omega * omega + 1.0) / (2.0 * omega)
    delta = omega0 - omega
    z = -omega
    recovery_denominator = 1.0 - z * z + 2.0 * delta * z
    discriminant = 5.0 * omega**4 - 2.0 * omega * omega + 1.0
    limiting_eigenvalues = []
    for sign in [-1.0, 1.0]:
        a = (
            -omega * omega
            - 1.0
            + sign * math.sqrt(discriminant)
        ) / (2.0 * omega)
        lam = a - 1.0 - omega
        dispersion_defect = (
            a * a
            + 1.0
            - omega * omega
            - a * (z + 1.0 / z)
        )
        match_defect = (
            a * (1.0 - z * z)
            + 2.0 * delta * z * (a - omega - z)
        )
        green = z * (lam + 1.0 - z) / (a * (1.0 - z * z))
        secular_defect = 1.0 + 2.0 * delta * green
        print(
            "zero_denominator_localization"
            f" z={z} lambda={lam}"
            f" recovery_denominator={recovery_denominator:.3e}"
            f" dispersion_defect={abs(dispersion_defect):.3e}"
            f" match_defect={abs(match_defect):.3e}"
            f" secular_defect={abs(secular_defect):.3e}",
            flush=True,
        )
        if max(
            abs(recovery_denominator),
            abs(dispersion_defect),
            abs(match_defect),
            abs(secular_defect),
        ) > 2.0e-12:
            raise AssertionError(
                "zero-denominator localization identity audit failed"
            )
        limiting_eigenvalues.append(lam)

    n = 24
    finite_values = np.linalg.eigvals(generator(n, omega, omega0))
    for lam in limiting_eigenvalues:
        finite_error = float(np.min(np.abs(finite_values - lam)))
        print(
            "zero_denominator_finite_convergence"
            f" n={n} lambda={lam} nearest_error={finite_error:.3e}",
            flush=True,
        )
        if finite_error > 2.0e-5:
            raise AssertionError(
                "zero-denominator finite convergence audit failed"
            )


def critical_boundary_audit() -> None:
    k = 2.0 * math.pi
    n = 10000
    for beta in [0.0, 3.0]:
        omega = k / n
        delta = (beta - k) / n
        h = k - beta
        center = math.cos(k / n) - 1.0 - omega
        lam = (
            center
            + 2.0j * math.sqrt(k * h) * n ** (-1.5)
            + 2.0 * h * n ** (-2.0)
        )
        for _ in range(20):
            green = finite_green(n, omega, lam)
            green_derivative = finite_green_derivative(n, omega, lam)
            residual = 1.0 + 2.0 * delta * green
            step = residual / (2.0 * delta * green_derivative)
            lam -= step
            if abs(step) < 1.0e-15:
                break
        observed = (-lam.real - k / n) * n * n
        predicted = k * k / 2.0 - 2.0 * h
        error = abs(observed - predicted)
        print(
            f"critical_boundary beta={beta} coefficient={observed:.9g}"
            f" predicted={predicted:.9g} error={error:.3e}",
            flush=True,
        )
        if error > 2.0e-2:
            raise AssertionError("critical boundary audit failed")

    beta = 8.0
    omega = k / n
    delta = (beta - k) / n
    center = math.cos(k / n) - 1.0 - omega
    predicted = 2.0 * math.sqrt(k * (beta - k))
    lam = center + predicted * n ** (-1.5)
    for _ in range(20):
        green = finite_green(n, omega, lam)
        green_derivative = finite_green_derivative(n, omega, lam)
        residual = 1.0 + 2.0 * delta * green
        step = residual / (2.0 * delta * green_derivative)
        lam -= step
        if abs(step) < 1.0e-15:
            break
    observed = (lam.real - center) * n ** 1.5
    error = abs(observed - predicted)
    print(
        f"critical_boundary beta={beta} displacement={observed:.9g}"
        f" predicted={predicted:.9g} error={error:.3e}",
        flush=True,
    )
    if error > 8.0e-2:
        raise AssertionError("critical square-root boundary audit failed")


def critical_grouped_remainder_audit() -> None:
    k = 2.0 * math.pi
    for n in [400, 800, 1600, 3200]:
        omega = k / n
        center = math.cos(k / n) - 1.0 - omega
        radius = math.sqrt((k / n) ** 2 - math.sin(k / n) ** 2)
        delta_lam = (1.1 + 0.7j) * n ** (-1.5)
        lam = center + delta_lam
        grouped = 2.0 * (delta_lam - k / n) / (
            n * (delta_lam * delta_lam - radius * radius)
        )
        remainder = finite_green(n, omega, lam) - grouped
        print(
            f"critical_grouped_remainder n={n} absolute={abs(remainder):.9g}",
            flush=True,
        )
        if abs(remainder) > 10.0:
            raise AssertionError("critical grouped remainder audit failed")


def critical_zero_wave_cancellation_audit() -> None:
    alpha = 2.0 * math.pi
    previous_inverse_distance = 0.0
    for n in [200, 400, 800, 1600]:
        omega = alpha / n
        lam = (0.4 + 0.7j) / n**2
        a = lam + 1.0 + omega
        x_minus_one = lam * (lam + 2.0 * omega) / (2.0 * a)
        w = 2.0 * n * np.arcsinh(np.sqrt(x_minus_one / 2.0))
        theta = lam * (lam + 2.0) / (2.0 * a * np.sinh(w / n))
        canceled = theta / np.tanh(w / 2.0)
        factorized = (
            (lam + 2.0)
            / (lam + 2.0 * omega)
            * np.tanh(w / (2.0 * n))
            / np.tanh(w / 2.0)
        )
        factorization_defect = abs(canceled - factorized)
        leading = (lam + 2.0) / (n * (lam + 2.0 * omega))
        inverse_distance = 1.0 / abs(w)
        scaled_remainder = n * abs(canceled - leading)
        print(
            f"critical_zero_wave n={n}"
            f" inverse_distance={inverse_distance:.9g}"
            f" canceled_absolute={abs(canceled):.9g}"
            f" factorization_defect={factorization_defect:.3e}"
            f" scaled_remainder={scaled_remainder:.9g}",
            flush=True,
        )
        if inverse_distance <= previous_inverse_distance:
            raise AssertionError("zero-wave pole-distance audit failed")
        if factorization_defect > 5.0e-11:
            raise AssertionError("zero-wave exact factorization audit failed")
        if abs(canceled) > 1.0 or scaled_remainder > 0.2:
            raise AssertionError("zero-wave cancellation audit failed")
        previous_inverse_distance = inverse_distance


def critical_side_gap_audit() -> None:
    k = 2.0 * math.pi
    n = 120
    cases = [(3.0, 0.0), (3.0, 6.0), (8.0, 3.0), (8.0, 12.0)]
    for alpha, beta in cases:
        exact_gap = spectral_gap(n, alpha / n, beta / n)
        if alpha < k:
            predicted = (
                alpha / n
                + (k * k / 2.0 - 2.0 * max(alpha - beta, 0.0)) / n**2
            )
        else:
            root = math.sqrt(alpha * alpha - k * k)
            predicted = (
                (alpha - root) / n
                + (
                    k * k / 2.0
                    - 2.0
                    * max(beta - alpha, 0.0)
                    * (alpha - root)
                    / root
                )
                / n**2
            )
        scaled_error = abs(exact_gap - predicted) * n**3
        print(
            f"critical_side_gap alpha={alpha} beta={beta} n={n}"
            f" scaled_n3_error={scaled_error:.9g}",
            flush=True,
        )
        if scaled_error > 60.0:
            raise AssertionError("critical side-gap audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=18)
    parser.add_argument("--omega", type=float, default=1.3)
    parser.add_argument("--omega0", type=float, default=0.4)
    args = parser.parse_args()
    determinant_audit(args.n, args.omega, args.omega0)
    transfer_audit(args.n, args.omega, args.omega0)
    small_ring_formula_audit()
    gap_audit(args.n, args.omega, args.omega0)
    fixed_gap_asymptotic_audit()
    localization_audit(args.omega, args.omega0)
    flat_band_audit()
    embedded_flat_band_eigenspace_audit()
    compact_localization_audit()
    zero_denominator_localization_audit()
    critical_boundary_audit()
    critical_grouped_remainder_audit()
    critical_zero_wave_cancellation_audit()
    critical_side_gap_audit()
    print("all formula audits passed", flush=True)


if __name__ == "__main__":
    main()
