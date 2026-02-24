"""
Convert JetClass Delphes-ntuplizer ROOT files into HDF5 format for ML training.

Extracts:
  - Jet-level features: pT, η, φ, energy, nParticles, soft-drop mass,
    N-subjettiness ratios (τ₂₁, τ₃₂, τ₄₃).
  - Constituent-level (padded to 100):
      · jetConstituentsList  : [Δη, Δφ, pT]  (relative to jet axis)
      · jetConstituentsExtra : [px, py, pz, E, charge, PID,
                                d0val, d0err, dzval, dzerr]
  - Precomputed EEC building blocks:
      · pair_delta_R          : upper-triangular ΔR matrix (N, 100, 100)
      · constituent_pt_weight : per-particle pT weight  (N, 100)

Author: Aritra Bal (ETP)
Date  : 2026-02-23

Copy of https://github.com/neutrino-man4/jetsim/blob/master/h5_maker/ntuples_to_h5.py
"""

import argparse
import logging
import os
import pathlib
from multiprocessing import Pool
from typing import Optional

import awkward as ak
import h5py,time
import numpy as np
import uproot

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
# Constants
# ---------------------------------------------------------------------------
MAX_CONSTITUENTS: int = 100
H5_STR_DTYPE = h5py.special_dtype(vlen=str)

PARTICLE_FEATURE_NAMES: list[str] = ["part_deta", "part_dphi", "part_pt"]

EXTRA_FEATURE_NAMES: list[str] = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_charge",
    "part_pid",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
]

JET_FEATURE_NAMES: list[str] = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_energy",
    "jet_nparticles",
    "jet_sdmass",
    "jet_tau21",
    "jet_tau32",
    "jet_tau43",
    "jet_tau1",
    "jet_tau2",
    "jet_tau3",
    "jet_tau4",
]


# ---------------------------------------------------------------------------
# Utility: padding
# ---------------------------------------------------------------------------
def zero_pad_to_numpy(
    arr: ak.Array,
    target_length: int = MAX_CONSTITUENTS,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad a ragged awkward array to a fixed constituent count and return NumPy.

    Parameters
    ----------
    arr : ak.Array
        Ragged array of shape (N, variable).
    target_length : int
        Number of constituents to clip/pad to.
    pad_value : float
        Value inserted at padded positions.

    Returns
    -------
    np.ndarray
        Dense array of shape (N, target_length).
    """
    padded = ak.pad_none(arr, target_length, axis=1, clip=True)
    padded = padded[ak.num(padded, axis=1) <= target_length]
    return ak.fill_none(padded, pad_value).to_numpy()


# ---------------------------------------------------------------------------
# EEC building blocks
# ---------------------------------------------------------------------------
def delta_r_pair(eta_phi: np.ndarray) -> np.ndarray:
    """
    Compute pairwise ΔR between all constituent pairs within each jet.

    Only the strict upper triangle (i < j) is populated; entries where
    i ≥ j are set to zero, avoiding double-counting of symmetric pairs.

    Phi-wrapping note
    -----------------
    The input Δφ values are already computed relative to the AK8 jet axis
    (via deltaPhi in the ntuplizer), so each particle satisfies |Δφ_i| ≤ 0.8.
    The maximum pairwise difference is therefore |Δφ_i − Δφ_j| ≤ 1.6 < π,
    which is safely within the principal range.  No additional φ-wrapping is
    required when forming pairwise differences.

    Role in the 2-point EEC
    -----------------------
    The 2-point differential Energy-Energy Correlator is:

        EEC(R) = SUM_{i < j}  w_i x w_j · delta_func(R − Delta_R_{ij})

    where w_i = pT_i / SUM_k pT_k  (see `pt_weight`).

    This function provides the Delta_R_{ij} matrix.  To obtain the EEC histogram,
    histogram the upper-triangle entries of pair_delta_R, weighted by the
    outer product of the constituent_pt_weight vectors.

    Parameters
    ----------
    eta_phi : np.ndarray
        Shape (N, P, 2), where the last axis is [rel_eta, rel_phi] per constituent relative to the jet axis.
        N = number of jets, P = number of constituents (padded to fixed length).

    Returns
    -------
    np.ndarray
        Shape (N, P, P), float32.  Entry [n, i, j] is Delta_R_{ij} for i < j,
        and 0 otherwise.
    """
    N, P, _ = eta_phi.shape

    eta: np.ndarray = eta_phi[:, :, 0]  # (N, P)
    phi: np.ndarray = eta_phi[:, :, 1]  # (N, P)

    # Broadcasting: expand to (N, P, 1) vs (N, 1, P) → (N, P, P)
    deta: np.ndarray = eta[:, :, np.newaxis] - eta[:, np.newaxis, :]
    dphi: np.ndarray = phi[:, :, np.newaxis] - phi[:, np.newaxis, :]

    dr: np.ndarray = np.sqrt(deta ** 2 + dphi ** 2).astype(np.float32)

    # Zero out diagonal and lower triangle; retain only i < j
    upper_mask: np.ndarray = np.triu(np.ones((P, P), dtype=bool), k=1)
    dr *= upper_mask[np.newaxis, :, :]

    return dr


def pt_weight(pt: np.ndarray) -> np.ndarray:
    """
    Compute the pT fraction weight for each constituent within its jet.

    Role in the 2-point EEC
    -----------------------
    The 2-point differential Energy-Energy Correlator uses pT-based energy
    fractions as proxy weights:

        w_i = pT_i / Σ_{k=1}^{N_part} pT_k

    so that Σ_i w_i = 1 for each jet.  The EEC integrand is then:

        EEC(R) = Σ_{i < j}  w_i · w_j · δ(R − ΔR_{ij})

    This function provides the weight vector w_i.  Padded constituents
    (pT = 0) automatically receive w_i = 0, so they do not contribute
    to the EEC sum even without explicit masking.

    Parameters
    ----------
    pt : np.ndarray
        Shape (N, P), per-constituent transverse momentum [GeV].
        Padded positions should carry pT = 0.

    Returns
    -------
    np.ndarray
        Shape (N, P), float32.  Normalised pT fractions; rows sum to 1
        for jets with at least one constituent with pT > 0.
    """
    pt_sum: np.ndarray = pt.sum(axis=1, keepdims=True)  # (N, 1)
    weights: np.ndarray = np.where(pt_sum > 0.0, pt / pt_sum, 0.0)
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def process_file(task: tuple) -> None:
    """
    Read one ROOT file and write the corresponding HDF5 output.

    Parameters
    ----------
    task : tuple
        (rfile, output_path, max_events)
        · rfile       : str  – path to the input ROOT file
        · output_path : str  – full path of the HDF5 file to create
        · max_events  : Optional[int] – if set, slice first N events only
    """
    rfile: str
    output_path: str
    max_events: Optional[int]
    rfile, output_path, max_events = task

    logger.info("Processing %s --> %s", rfile, output_path)

    entry_stop: Optional[int] = max_events  # uproot interprets None as all

    with uproot.open(rfile) as root_file:
        tree = root_file["tree"]

        # ---- Particle branches (ragged) ------------------------------------
        deta_ak  = tree["part_deta"].array(entry_stop=entry_stop)
        dphi_ak  = tree["part_dphi"].array(entry_stop=entry_stop)
        pt_ak    = tree["part_pt"].array(entry_stop=entry_stop)
        px_ak    = tree["part_px"].array(entry_stop=entry_stop)
        py_ak    = tree["part_py"].array(entry_stop=entry_stop)
        pz_ak    = tree["part_pz"].array(entry_stop=entry_stop)
        E_ak     = tree["part_energy"].array(entry_stop=entry_stop)
        charge_ak = tree["part_charge"].array(entry_stop=entry_stop)
        pid_ak   = tree["part_pid"].array(entry_stop=entry_stop)
        d0val_ak = tree["part_d0val"].array(entry_stop=entry_stop)
        d0err_ak = tree["part_d0err"].array(entry_stop=entry_stop)
        dzval_ak = tree["part_dzval"].array(entry_stop=entry_stop)
        dzerr_ak = tree["part_dzerr"].array(entry_stop=entry_stop)
        truth_label = tree["is_signal"].array(entry_stop=entry_stop).to_numpy()  # (N,)
        # Number of real constituents per jet (before padding)
        nparticles: np.ndarray = ak.num(pt_ak, axis=1).to_numpy()

        # ---- Jet branches (flat) -------------------------------------------
        jet_pt:     np.ndarray = tree["jet_pt"].array(entry_stop=entry_stop).to_numpy()
        jet_eta:    np.ndarray = tree["jet_eta"].array(entry_stop=entry_stop).to_numpy()
        jet_phi:    np.ndarray = tree["jet_phi"].array(entry_stop=entry_stop).to_numpy()
        jet_energy: np.ndarray = tree["jet_energy"].array(entry_stop=entry_stop).to_numpy()
        jet_sdmass: np.ndarray = tree["jet_sdmass"].array(entry_stop=entry_stop).to_numpy()
        tau1: np.ndarray = tree["jet_tau1"].array(entry_stop=entry_stop).to_numpy()
        tau2: np.ndarray = tree["jet_tau2"].array(entry_stop=entry_stop).to_numpy()
        tau3: np.ndarray = tree["jet_tau3"].array(entry_stop=entry_stop).to_numpy()
        tau4: np.ndarray = tree["jet_tau4"].array(entry_stop=entry_stop).to_numpy()

    # ---- Pad constituent arrays -------------------------------------------
    part_deta   = zero_pad_to_numpy(deta_ak,   pad_value=0.0)
    part_dphi   = zero_pad_to_numpy(dphi_ak,   pad_value=0.0)
    part_pt     = zero_pad_to_numpy(pt_ak,     pad_value=0.0)
    part_px     = zero_pad_to_numpy(px_ak,     pad_value=0.0)
    part_py     = zero_pad_to_numpy(py_ak,     pad_value=0.0)
    part_pz     = zero_pad_to_numpy(pz_ak,     pad_value=0.0)
    part_E      = zero_pad_to_numpy(E_ak,      pad_value=0.0)
    part_charge = zero_pad_to_numpy(charge_ak, pad_value=0.0)
    part_pid    = zero_pad_to_numpy(pid_ak,    pad_value=0.0)
    part_d0val  = zero_pad_to_numpy(d0val_ak,  pad_value=-999.0)
    part_d0err  = zero_pad_to_numpy(d0err_ak,  pad_value=-999.0)
    part_dzval  = zero_pad_to_numpy(dzval_ak,  pad_value=-999.0)
    part_dzerr  = zero_pad_to_numpy(dzerr_ak,  pad_value=-999.0)
    part_mask  = part_pt > 0.0  # boolean mask of real vs padded constituents

    # ---- Assemble composite arrays ----------------------------------------

    # jetConstituentsList : (N, 100, 3) - [rel_eta, rel_phi, pT]
    jet_pfc: np.ndarray = np.stack(
        [part_deta, part_dphi, part_pt], axis=-1
    ).astype(np.float32)

    # jetConstituentsExtra : (N, 100, 10)
    jet_extra: np.ndarray = np.stack(
        [
            part_px, part_py, part_pz, part_E,
            part_charge, part_pid,
            part_d0val, part_d0err,
            part_dzval, part_dzerr,
        ],
        axis=-1,
    ).astype(np.float32)

    # N-subjettiness ratios (protected against division by zero)
    tau21: np.ndarray = np.where(tau1 > 0.0, tau2 / tau1, 0.0)
    tau32: np.ndarray = np.where(tau2 > 0.0, tau3 / tau2, 0.0)
    tau43: np.ndarray = np.where(tau3 > 0.0, tau4 / tau3, 0.0)
    logger.info("Computed N-subjettiness ratios (tau21, tau32, tau43)")
    # jetFeatures : (N, 9) – [pT, η, φ, E, nPart, sdMass, τ₂₁, τ₃₂, τ₄₃]
    jet_features: np.ndarray = np.stack(
        [jet_pt, jet_eta, jet_phi, jet_energy,
         nparticles.astype(np.float32),
         jet_sdmass, tau21, tau32, tau43,
         tau1, tau2, tau3, tau4,
        ],
        axis=-1,
    ).astype(np.float32)

    # ---- EEC building blocks -----------------------------------------------
    # eta_phi tensor for delta_r_pair: (N, 100, 2)
    logger.info("Computing pairwise delta_R matrix for all jets (this may take a while)...")
    eta_phi_tensor: np.ndarray = np.stack([part_deta, part_dphi], axis=-1).astype(np.float32)
    pair_dr:   np.ndarray = delta_r_pair(eta_phi_tensor)          # (N, 100, 100)
    pt_weights: np.ndarray = pt_weight(part_pt.astype(np.float32)) # (N, 100)

    # ---- Sanity checks ----------------------------------------------------
    assert not np.isnan(jet_pfc).any(),     "NaN detected in jet_pfc"
    assert not np.isnan(jet_extra).any(),   "NaN detected in jet_extra"
    assert not np.isnan(jet_features).any(),"NaN detected in jet_features"
    assert not np.isnan(pair_dr).any(),     "NaN detected in pair_delta_R"
    assert not np.isnan(pt_weights).any(),  "NaN detected in constituent_pt_weight"

    # ---- Write HDF5 --------------------------------------------------------
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("jetConstituentsList",data=jet_pfc)
        f.create_dataset("jetConstituentsExtra",data=jet_extra)
        f.create_dataset("jetFeatures",data=jet_features)
        f.create_dataset("pair_delta_R",data=pair_dr)
        f.create_dataset("constituent_pt_weight",data=pt_weights)
        f.create_dataset("jetConstituentsMask",data=part_mask.astype(np.bool_))
        f.create_dataset("particleFeatureNames",data=np.array(PARTICLE_FEATURE_NAMES, dtype=H5_STR_DTYPE))
        f.create_dataset("jetConstituentsExtraNames",data=np.array(EXTRA_FEATURE_NAMES, dtype=H5_STR_DTYPE))
        f.create_dataset("jetFeatureNames",data=np.array(JET_FEATURE_NAMES, dtype=H5_STR_DTYPE))
        f.create_dataset("truth_label", data=truth_label.astype(np.int8))
    logger.info("Written: %s  (%d jets)", output_path, len(jet_pt))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert JetClass ROOT files to HDF5 with EEC building blocks."
    )
    parser.add_argument(
        "--root-files",
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "One or more ROOT file paths, OR a single .txt file containing "
            "one ROOT file path per line."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default='/ceph/abal/QFIT/MC/HDF5/',
        metavar="DIR",
        help=(
            "Directory in which HDF5 files are written.  "
            "Ignored when --test-run is set (output goes to /tmp/abal/test.h5)."
        ),
    )
    parser.add_argument(
        "--jet-type",
        type=str,
        choices=["dijet", "TTBar","HToCC","HToBB","WToQQ","ZToQQ"],
        default="dijet",
        help="Type of jets being processed (default: dijet). Required for normal mode to determine output filename. Ignored in test-run mode.", 
    )
    parser.add_argument(
        "--max-cores",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of worker processes for parallel file processing (default: 5).",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help=(
            "Process only the first ROOT file and only the first 100 events. "
            "Output is written to /tmp/abal/test.h5."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def resolve_file_list(raw_paths: list[str]) -> list[str]:
    """
    Resolve the --root-files argument into a flat list of ROOT file paths.

    If a single argument ending in '.txt' is supplied, the file is read
    line-by-line; otherwise the argument list is used directly.

    Parameters
    ----------
    raw_paths : list[str]
        Value of args.root_files from argparse.

    Returns
    -------
    list[str]
        Resolved, deduplicated list of ROOT file paths.
    """
    if len(raw_paths) == 1 and raw_paths[0].endswith(".txt"):
        txt_path = raw_paths[0]
        with open(txt_path, "r") as fh:
            paths = [line.strip() for line in fh if line.strip()]
        logger.info("Read %d paths from %s", len(paths), txt_path)
    else:
        paths = raw_paths

    # Basic existence check
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"ROOT files not found: {missing}")

    return list(dict.fromkeys(paths))  # preserve order, deduplicate


def main() -> None:
    """Main entry point."""
    args = parse_args()

    file_list: list[str] = sorted(resolve_file_list(args.root_files))
    logger.info("Total ROOT files to process: %d", len(file_list))

    if args.test_run:
        # ---- Test-run mode: single file, 100 events -----------------------
        test_out = "/tmp/abal/test.h5"
        pathlib.Path("/tmp/abal").mkdir(parents=True, exist_ok=True)
        logger.info("TEST RUN – processing 1 file / 100 events → %s", test_out)
        process_file((file_list[0], test_out, 100))
        return

    # ---- Normal mode -------------------------------------------------------
    if args.output_dir is None:
        raise ValueError("--output-dir is required when not running in --test-run mode.")

    
    # Build task list: each ROOT file → sibling h5 in output_dir
    tasks: list[tuple] = []
    for i,rfile in enumerate(file_list):
        try:
            run_id = pathlib.Path(rfile).parent.name # extract run_XX from path
        except Exception as e:
            logger.warning("Could not extract run_id from filename '%s': %s", rfile, e)
            run_id = f"run_{i:02d}" # assign a run ID based on index if extraction fails
        if args.jet_type.casefold() not in rfile.casefold():
            logger.warning("Jet type '%s' not found in filename '%s'; output filename may be misleading or overwrite. Is this intended? (waiting 5s)", args.jet_type, rfile)
            time.sleep(5)
        output_dir = pathlib.Path(args.output_dir) / args.jet_type / f"{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(output_dir / f"{args.jet_type}.h5")
        tasks.append((rfile, out_path, None))  # None → all events

    n_cores: int = min(args.max_cores, len(tasks))
    logger.info("Using %d worker(s) for %d file(s)", n_cores, len(tasks))

    if n_cores == 1:
        # Single file or single core: skip Pool overhead
        for task in tasks:
            process_file(task)
    else:
        with Pool(processes=n_cores) as pool:
            pool.map(process_file, tasks)

    logger.info("All files processed.")


if __name__ == "__main__":
    main()