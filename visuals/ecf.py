"""
Compute and visualize ECF-based C_N^(beta) substructure ratios from
JetClass-format HDF5 files produced by ntuples_to_h5.py.

The C_N ratio is defined as (Larkoski, Salam, Thaler, JHEP 2013):

    C_N^(beta) = ECF(N+1, beta) * ECF(N-1, beta) / ECF(N, beta)^2

ECF definitions using normalized pT fractions z_i = pT_i / sum_k pT_k,
read from the 'constituent_pt_weight' dataset. Using fractions instead of
raw pT is equivalent in the C ratio since all jet-pT factors cancel exactly.

    ECF(0, beta) = 1
    ECF(1, beta) = sum_i z_i  (= 1 by construction for normalized weights)
    ECF(2, beta) = sum_{i<j}     z_i z_j (R_ij)^beta
    ECF(3, beta) = sum_{i<j<k}   z_i z_j z_k (R_ij R_ik R_jk)^beta
    ECF(4, beta) = sum_{i<j<k<l} z_i z_j z_k z_l
                                  (R_ij R_ik R_il R_jk R_jl R_kl)^beta

Required HDF5 datasets (from ntuples_to_h5.py):
  - pair_delta_R          : (N_jets, 100, 100) upper-triangle deltaR matrix
  - constituent_pt_weight : (N_jets, 100) normalized pT fractions

Author: Aritra Bal (ETP)
Date  : 2026-02-25
"""

import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import Optional

import h5py
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index precomputation
# ---------------------------------------------------------------------------

def precompute_indices(J: int) -> dict:
    """
    Precompute sorted combination index arrays for ECF(2), ECF(3), ECF(4).

    Computed once and reused across all jets and all chunks to avoid
    repeated calls to itertools.combinations inside the hot path.

    Parameters
    ----------
    J : int
        Number of constituents per jet.

    Returns
    -------
    dict with keys 'pairs', 'triplets', 'quads', each a tuple of int32
    arrays representing the index axes of each combination type.
    """
    pairs = np.array(list(combinations(range(J), 2)), dtype=np.int32)
    tris  = np.array(list(combinations(range(J), 3)), dtype=np.int32)
    quads = np.array(list(combinations(range(J), 4)), dtype=np.int32)

    result: dict = {
        "pairs":    (pairs[:, 0], pairs[:, 1]),
        "triplets": (tris[:, 0],  tris[:, 1],  tris[:, 2]),
        "quads":    (quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]),
    }
    logger.info(
        "Combination counts for J=%d: %d pairs, %d triplets, %d quads",
        J, len(pairs), len(tris), len(quads),
    )
    return result


# ---------------------------------------------------------------------------
# ECF functions -- all operate on batches of jets (B, J) / (B, J, J)
# ---------------------------------------------------------------------------

def ecf0(
    z: np.ndarray,
    dr: np.ndarray,
    beta: float,
    indices: dict,
) -> np.ndarray:
    """
    ECF(0, beta) = 1 for every jet by definition.

    Parameters
    ----------
    z       : (B, J) normalized pT fractions (unused)
    dr      : (B, J, J) upper-triangle delta-R matrix (unused)
    beta    : angular exponent (unused)
    indices : precomputed index dict (unused)

    Returns
    -------
    np.ndarray of shape (B,) filled with 1.0
    """
    return np.ones(z.shape[0], dtype=np.float64)


def ecf1(
    z: np.ndarray,
    dr: np.ndarray,
    beta: float,
    indices: dict,
) -> np.ndarray:
    """
    ECF(1, beta) = sum_i z_i.

    For globally normalized pT fractions this is 1.0 per jet, but may
    be less than 1 when only J < n_real_constituents are used.

    Parameters
    ----------
    z       : (B, J) normalized pT fractions
    dr      : (B, J, J) upper-triangle delta-R matrix (unused)
    beta    : angular exponent (unused)
    indices : precomputed index dict (unused)

    Returns
    -------
    np.ndarray of shape (B,)
    """
    return z.sum(axis=1)


def ecf2(
    z: np.ndarray,
    dr: np.ndarray,
    beta: float,
    indices: dict,
) -> np.ndarray:
    """
    ECF(2, beta) = sum_{i<j} z_i z_j R_ij^beta.

    Parameters
    ----------
    z       : (B, J) normalized pT fractions
    dr      : (B, J, J) upper-triangle delta-R matrix
    beta    : angular exponent
    indices : precomputed index dict; uses key 'pairs'

    Returns
    -------
    np.ndarray of shape (B,)
    """
    pi, pj = indices["pairs"]
    z_ij = z[:, pi] * z[:, pj]    # (B, n_pairs)
    r_ij = dr[:, pi, pj]           # (B, n_pairs)
    return (z_ij * np.power(r_ij, beta)).sum(axis=1)


def ecf3(
    z: np.ndarray,
    dr: np.ndarray,
    beta: float,
    indices: dict,
) -> np.ndarray:
    """
    ECF(3, beta) = sum_{i<j<k} z_i z_j z_k (R_ij R_ik R_jk)^beta.

    Parameters
    ----------
    z       : (B, J) normalized pT fractions
    dr      : (B, J, J) upper-triangle delta-R matrix
    beta    : angular exponent
    indices : precomputed index dict; uses key 'triplets'

    Returns
    -------
    np.ndarray of shape (B,)
    """
    ti, tj, tk = indices["triplets"]
    z_ijk = z[:, ti] * z[:, tj] * z[:, tk]                 # (B, n_tri)
    r_ijk = dr[:, ti, tj] * dr[:, ti, tk] * dr[:, tj, tk]  # (B, n_tri)
    return (z_ijk * np.power(r_ijk, beta)).sum(axis=1)


def ecf4(
    z: np.ndarray,
    dr: np.ndarray,
    beta: float,
    indices: dict,
) -> np.ndarray:
    """
    ECF(4, beta) = sum_{i<j<k<l}
                    z_i z_j z_k z_l (R_ij R_ik R_il R_jk R_jl R_kl)^beta.

    Parameters
    ----------
    z       : (B, J) normalized pT fractions
    dr      : (B, J, J) upper-triangle delta-R matrix
    beta    : angular exponent
    indices : precomputed index dict; uses key 'quads'

    Returns
    -------
    np.ndarray of shape (B,)
    """
    qi, qj, qk, ql = indices["quads"]
    z_ijkl = z[:, qi] * z[:, qj] * z[:, qk] * z[:, ql]    # (B, n_quad)
    r_ijkl = (
        dr[:, qi, qj] * dr[:, qi, qk] * dr[:, qi, ql] *
        dr[:, qj, qk] * dr[:, qj, ql] * dr[:, qk, ql]
    )                                                        # (B, n_quad)
    return (z_ijkl * np.power(r_ijkl, beta)).sum(axis=1)


# Dispatch table: ECF order -> function
_ECF_FUNCS: dict = {
    0: ecf0,
    1: ecf1,
    2: ecf2,
    3: ecf3,
    4: ecf4,
}


# ---------------------------------------------------------------------------
# C ratio computation
# ---------------------------------------------------------------------------

def compute_c_ratio_batch(
    z: np.ndarray,
    dr: np.ndarray,
    beta: float,
    N: int,
    indices: dict,
) -> np.ndarray:
    """
    Compute C_N^(beta) for a batch of jets.

    C_N = ECF(N+1, beta) * ECF(N-1, beta) / ECF(N, beta)^2

    Jets where ECF(N, beta) = 0 (e.g. single-constituent jets within J)
    are returned as NaN for downstream filtering.

    Parameters
    ----------
    z       : (B, J) normalized pT fractions
    dr      : (B, J, J) upper-triangle delta-R matrix
    beta    : angular exponent
    N       : order of the C ratio
    indices : precomputed index dict

    Returns
    -------
    np.ndarray of shape (B,); NaN where ECF(N)^2 = 0
    """
    ecf_nm1 = _ECF_FUNCS[N - 1](z, dr, beta, indices)
    ecf_n   = _ECF_FUNCS[N    ](z, dr, beta, indices)
    ecf_np1 = _ECF_FUNCS[N + 1](z, dr, beta, indices)

    denom = ecf_n ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(denom > 0.0, ecf_np1 * ecf_nm1 / denom, np.nan)
    return c


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(
    h5_path: str,
    beta: float,
    N: int,
    J: int,
    chunk_size: Optional[int],
    indices: dict,
) -> np.ndarray:
    """
    Load one HDF5 file and return per-jet C_N^(beta) values.

    Memory strategy: when chunk_size is set, pair_delta_R and
    constituent_pt_weight are loaded J-sliced in batches of chunk_size jets.
    The returned C values are one scalar per jet, so concatenating them is
    memory-cheap relative to the full constituent arrays.

    Parameters
    ----------
    h5_path    : path to the HDF5 file
    beta       : angular exponent for ECF
    N          : order of the C ratio
    J          : number of leading constituents per jet
    chunk_size : jet batch size for H5 reads; None means load all at once
    indices    : precomputed combination index dict

    Returns
    -------
    np.ndarray of shape (N_valid,) with NaN-filtered C_N values
    """
    with h5py.File(h5_path, "r") as f:
        n_jets: int = int(f["pair_delta_R"].shape[0])
        max_stored: int = int(f["pair_delta_R"].shape[1])

    if J > max_stored:
        logger.warning(
            "Requested J=%d exceeds stored constituent count %d; clamping to %d.",
            J, max_stored, max_stored,
        )
        J = max_stored

    logger.info(
        "Processing %s | n_jets=%d | J=%d | chunk_size=%s",
        h5_path, n_jets, J, chunk_size,
    )

    c_list: list[np.ndarray] = []

    with h5py.File(h5_path, "r") as f:
        dr_ds = f["pair_delta_R"]           # (N_jets, 100, 100)
        z_ds  = f["constituent_pt_weight"]  # (N_jets, 100)

        ranges: list[tuple[int, int]] = (
            [(s, min(s + chunk_size, n_jets)) for s in range(0, n_jets, chunk_size)]
            if chunk_size is not None
            else [(0, n_jets)]
        )

        for start, end in ranges:
            # Slice only the leading J constituents to reduce I/O and memory.
            # Constituents are pT-sorted descending (JetClass convention),
            # so [:J] selects the J hardest constituents.
            dr_slice = dr_ds[start:end, :J, :J].astype(np.float64)  # (B, J, J)
            z_slice  = z_ds [start:end, :J    ].astype(np.float64)  # (B, J)

            c_batch = compute_c_ratio_batch(z_slice, dr_slice, beta, N, indices)
            c_list.append(c_batch)
            logger.debug("Processed jets %d to %d.", start, end - 1)

    c_all: np.ndarray = np.concatenate(c_list)

    n_nan: int = int(np.isnan(c_all).sum())
    if n_nan > 0:
        logger.warning(
            "%d / %d jets yielded NaN C_%d (ECF(%d)=0, likely fewer than 2 "
            "real constituents within J=%d); excluded from histogram.",
            n_nan, n_jets, N, N, J,
        )

    valid: np.ndarray = c_all[~np.isnan(c_all)]
    logger.info(
        "  Valid jets: %d | C_%d range: [%.5g, %.5g]",
        len(valid), N,
        float(np.min(valid)) if len(valid) else float("nan"),
        float(np.max(valid)) if len(valid) else float("nan"),
    )
    return valid


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot ECF-based C_N^(beta) substructure ratio histograms "
            "from JetClass HDF5 files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5-files",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more HDF5 files produced by ntuples_to_h5.py.",
    )
    parser.add_argument(
        "--plot-labels",
        nargs="*",
        default=None,
        metavar="LABEL",
        help=(
            "Legend labels for each input file. If fewer labels than files "
            "are supplied, missing ones are replaced by 'Placeholder N'."
        ),
    )
    parser.add_argument(
        "--ecf-beta",
        type=float,
        default=0.2,
        metavar="BETA",
        help="Angular exponent beta for ECF (must be > 0).",
    )
    parser.add_argument(
        "--correlation-ratio",
        type=int,
        default=2,
        choices=[1, 2, 3],
        metavar="N",
        help=(
            "Order N of the C_N ratio to plot. "
            "1 uses ECF(0,1,2); 2 uses ECF(1,2,3); 3 uses ECF(2,3,4)."
        ),
    )
    parser.add_argument(
        "--max-constituents",
        type=int,
        default=10,
        metavar="J",
        help=(
            "Number of leading-pT constituents to use per jet. "
            "Assumed to be pT-sorted descending (standard JetClass convention). "
            "Must be >= 2."
        ),
    )
    parser.add_argument(
        "--chunking",
        type=int,
        default=None,
        metavar="X",
        help=(
            "If set, load and process jets in batches of X rows from the HDF5 "
            "file to reduce peak memory. Omit to load all jets at once."
        ),
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=50,
        metavar="BINS",
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ecf_ratio.pdf",
        metavar="FILE",
        help="Output filename for the saved figure.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point."""
    args = parse_args()

    # ---- Input validation -------------------------------------------------
    if args.ecf_beta <= 0.0:
        raise ValueError(f"--ecf-beta must be positive; got {args.ecf_beta}.")
    if args.max_constituents < 2:
        raise ValueError("--max-constituents must be >= 2 to form at least one pair.")

    N: int      = args.correlation_ratio
    J: int      = args.max_constituents
    beta: float = args.ecf_beta

    # ---- Labels -----------------------------------------------------------
    n_files: int = len(args.h5_files)
    raw_labels: list[str] = args.plot_labels or []
    labels: list[str] = [
        raw_labels[i] if i < len(raw_labels) else f"Placeholder {i + 1}"
        for i in range(n_files)
    ]

    # ---- Precompute combination indices once for all files ----------------
    indices: dict = precompute_indices(J)

    # ---- Compute C ratios per file ----------------------------------------
    all_c: list[np.ndarray] = []
    for h5_path in args.h5_files:
        c_vals = process_file(
            h5_path=h5_path,
            beta=beta,
            N=N,
            J=J,
            chunk_size=args.chunking,
            indices=indices,
        )
        all_c.append(c_vals)

    # ---- Determine common histogram range ---------------------------------
    # Joint 0.5th-99.5th percentile across all files suppresses outliers
    # from numerically degenerate jets without distorting the bulk shape.
    combined: np.ndarray = np.concatenate(all_c)
    if len(combined) == 0:
        raise RuntimeError("No valid jets found across all supplied files.")

    lo: float = max(0.0, float(np.percentile(combined, 0.5)))
    hi: float = float(np.percentile(combined, 99.5))
    logger.info("Histogram range (joint 0.5-99.5 percentile): [%.5g, %.5g]", lo, hi)

    bin_edges: np.ndarray  = np.linspace(lo, hi, args.n_bins + 1)
    bin_widths: np.ndarray = np.diff(bin_edges)

    # ---- Plot -------------------------------------------------------------
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(8, 6))

    colors: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (c_vals, label) in enumerate(zip(all_c, labels)):
        counts, _ = np.histogram(c_vals, bins=bin_edges)
        total: int = int(counts.sum())

        if total == 0:
            logger.warning(
                "File index %d ('%s') has no entries in histogram range; skipping.",
                idx, label,
            )
            continue

        # Area-normalised density: integral over bin_widths = 1
        density: np.ndarray = counts / (total * bin_widths)

        ax.stairs(
            density,
            bin_edges,
            color=colors[idx % len(colors)],
            linewidth=1.8,
            label=label,
        )

    # X-axis: e.g. C_2^{(0.2)} -- beta always rendered to 1 decimal place
    x_label: str = rf"$C_{{{N}}}^{{({beta:.1f})}}$"
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel("a.u.", fontsize=16)
    ax.set_xlim(lo, hi)
    ax.legend(fontsize=13, frameon=False)

    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Figure saved to %s", out)
    plt.show()


if __name__ == "__main__":
    main()