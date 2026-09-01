#!/usr/bin/env python3
"""Generate the numerical data used in the manuscript summary figure."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from verify_formulas import generator, homogeneous_gap, spectral_gap


OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "paper" / "figures"


def reflection_matrix(n: int) -> np.ndarray:
    reflection = np.zeros((2 * n, 2 * n), dtype=float)
    for j in range(n):
        reflected = (-j) % n
        reflection[2 * j, 2 * reflected + 1] = 1.0
        reflection[2 * j + 1, 2 * reflected] = 1.0
    return reflection


def parity_spectrum() -> None:
    n = 24
    omega = 0.7
    omega0 = 1.4
    matrix = generator(n, omega, omega0)
    reflection = reflection_matrix(n)
    commutator_error = float(np.linalg.norm(matrix @ reflection - reflection @ matrix))
    if commutator_error > 1.0e-12:
        raise AssertionError("reflection does not commute with the generator")

    parity_values, parity_vectors = np.linalg.eigh(reflection)
    for name, sign in (("protected", 1.0), ("affected", -1.0)):
        basis = parity_vectors[:, np.abs(parity_values - sign) < 1.0e-10]
        if basis.shape != (2 * n, n):
            raise AssertionError("unexpected parity-sector dimension")
        values = np.linalg.eigvals(basis.T @ matrix @ basis)
        values = values[np.lexsort((values.imag, values.real))]
        data = np.column_stack((values.real, values.imag))
        np.savetxt(
            OUTPUT_DIRECTORY / f"spectrum_{name}.dat",
            data,
            header="Re(lambda) Im(lambda)",
            fmt="%.15e",
        )
    print(
        f"parity spectrum: n={n}, omega={omega}, omega0={omega0}, "
        f"commutator_error={commutator_error:.3e}",
        flush=True,
    )


def positive_localization_root(omega: float, omega0: float) -> tuple[float, float]:
    delta = omega0 - omega
    coefficients = [
        2.0 * delta,
        2.0 * delta * omega + 1.0 - omega * omega,
        4.0 * delta * delta * omega
        + 4.0 * delta * omega * omega
        - 2.0 * delta,
        2.0 * delta * omega + omega * omega - 1.0,
    ]
    roots = [
        float(root.real)
        for root in np.roots(coefficients)
        if abs(root.imag) < 1.0e-10 and 0.0 < root.real < 1.0
    ]
    if len(roots) != 1:
        raise AssertionError("expected one positive localization root")
    z = roots[0]
    denominator = 1.0 - z * z + 2.0 * delta * z
    a = 2.0 * delta * z * (omega + z) / denominator
    return z, a - 1.0 - omega


def localization_profile() -> None:
    n = 60
    omega = 0.5
    omega0 = 0.8
    z, limiting_eigenvalue = positive_localization_root(omega, omega0)
    values, vectors = np.linalg.eig(generator(n, omega, omega0))
    index = int(np.argmin(np.abs(values - limiting_eigenvalue)))
    eigenvalue = values[index]
    if abs(eigenvalue - limiting_eigenvalue) > 1.0e-8:
        raise AssertionError("finite localized eigenvalue did not match its limit")

    vector = vectors[:, index]
    vector /= np.linalg.norm(vector)
    site_amplitude = np.asarray(
        [abs(vector[2 * j]) ** 2 + abs(vector[2 * j + 1]) ** 2 for j in range(n)]
    )
    distances = np.arange(n // 2 + 1)
    profile = np.empty_like(distances, dtype=float)
    profile[0] = site_amplitude[0]
    for distance in distances[1:]:
        profile[distance] = 0.5 * (
            site_amplitude[distance] + site_amplitude[-distance]
        )
    reference = profile[1] * z ** (2.0 * (distances - 1.0))
    np.savetxt(
        OUTPUT_DIRECTORY / "localization_profile.dat",
        np.column_stack((distances, profile, reference)),
        header="distance squared_site_amplitude reference_decay",
        fmt=["%d", "%.15e", "%.15e"],
    )
    print(
        f"localized profile: n={n}, z={z:.12f}, "
        f"lambda_limit={limiting_eigenvalue:.12f}, "
        f"finite_error={abs(eigenvalue-limiting_eigenvalue):.3e}",
        flush=True,
    )


def fixed_gap_scaling() -> None:
    omega = 1.3
    omega0 = 2.0
    sizes = np.asarray([20, 24, 30, 40, 50, 64, 80, 100], dtype=int)
    coefficient = (
        4.0
        * math.pi**2
        * (omega0 - omega)
        * (1.0 + omega)
        / (omega * omega * (1.0 + omega0))
    )
    scaled_differences = []
    for n in sizes:
        direct_gap = spectral_gap(int(n), omega, omega0)
        protected_gap = homogeneous_gap(int(n), omega)
        scaled_differences.append(n**3 * (protected_gap - direct_gap))
        print(
            f"fixed-gap data: n={n}, scaled_difference={scaled_differences[-1]:.9f}",
            flush=True,
        )
    data = np.column_stack(
        (
            sizes,
            np.asarray(scaled_differences),
            np.full_like(sizes, coefficient, dtype=float),
        )
    )
    np.savetxt(
        OUTPUT_DIRECTORY / "fixed_gap_scaling.dat",
        data,
        header="n n^3*(gamma_hom-gamma) predicted_coefficient",
        fmt=["%d", "%.15e", "%.15e"],
    )
    print(f"fixed-gap coefficient: {coefficient:.12f}", flush=True)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    parity_spectrum()
    localization_profile()
    fixed_gap_scaling()
    print(f"figure data written to {OUTPUT_DIRECTORY}", flush=True)


if __name__ == "__main__":
    main()
