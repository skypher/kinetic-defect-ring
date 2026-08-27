#!/usr/bin/env python3
"""Short numerical and symbolic audits for the manuscript formulas."""

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


def gap_audit(n: int, omega: float, omega0: float) -> None:
    values = np.linalg.eigvals(generator(n, omega, omega0))
    nonzero = values[np.abs(values) > 1.0e-9]
    exact_gap = -float(np.max(nonzero.real))
    q = 2.0 * math.pi / n
    homogeneous_gap = (
        1.0
        - math.cos(q)
        + omega
        - math.sqrt(max(0.0, omega * omega - math.sin(q) ** 2))
    )
    print(
        "gap"
        f" exact={exact_gap:.12g}"
        f" protected={homogeneous_gap:.12g}"
        f" difference={exact_gap-homogeneous_gap:.3e}",
        flush=True,
    )


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


def flat_band_audit() -> None:
    cases = [
        (5, 0.4, [4, 4, 4]),
        (7, 0.4, [6, 7, 7]),
        (7, 0.75, [6, 7, 8]),
        (8, 0.4, [7, 8, 8]),
    ]
    for n, omega0, expected in cases:
        nilpotent = generator(n, 1.0, omega0) + 2.0 * np.eye(2 * n)
        nullities = []
        power = np.eye(2 * n)
        for exponent in range(1, 4):
            power = power @ nilpotent
            rank = np.linalg.matrix_rank(power, tol=1.0e-9)
            nullities.append(2 * n - rank)
        print(
            f"flat_band n={n} omega0={omega0} nullities={nullities}",
            flush=True,
        )
        if nullities != expected:
            raise AssertionError("flat-band Jordan audit failed")


def compact_localization_audit() -> None:
    omega = 2.0
    omega0 = (omega * omega + 1.0) / (2.0 * omega)
    lam = -1.0 - omega
    for n in [4, 7, 12]:
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


def critical_boundary_audit() -> None:
    k = 2.0 * math.pi
    n = 10000
    wave_numbers = 2.0 * math.pi * np.arange(n) / n
    cosines = np.cos(wave_numbers)
    sine_squares = np.sin(wave_numbers) ** 2
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
            numerator = lam + 1.0 - cosines
            denominator = numerator * (
                lam + 1.0 + 2.0 * omega - cosines
            ) + sine_squares
            derivative_denominator = 2.0 * (
                lam + 1.0 + omega - cosines
            )
            green = np.sum(numerator / denominator) / n
            green_derivative = np.sum(
                (
                    denominator
                    - numerator * derivative_denominator
                )
                / (denominator * denominator)
            ) / n
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=18)
    parser.add_argument("--omega", type=float, default=1.3)
    parser.add_argument("--omega0", type=float, default=0.4)
    args = parser.parse_args()
    determinant_audit(args.n, args.omega, args.omega0)
    transfer_audit(args.n, args.omega, args.omega0)
    gap_audit(args.n, args.omega, args.omega0)
    localization_audit(args.omega, args.omega0)
    flat_band_audit()
    compact_localization_audit()
    critical_boundary_audit()


if __name__ == "__main__":
    main()
