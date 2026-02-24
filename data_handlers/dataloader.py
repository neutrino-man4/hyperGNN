"""
Streaming PyTorch Geometric DataLoader for jet graph / hypergraph classification.

Graph mode  (data_cfg.hyper = False):
    Fully-connected undirected graph on self.cfg.num_constituents=100 nodes.
    Edge feature (ECF2):
        e_{ij} = z_i * z_j * theta_{ij}^{beta}
    where z_i = pT_i / sum_k pT_k  (from constituent_pt_weight)
    and   theta_{ij} = DeltaR_{ij}  (from pair_delta_R, upper triangle).
    Both directed edges (i->j) and (j->i) carry the same scalar feature
    (undirected PyG convention).

Hypergraph mode (data_cfg.hyper = True):
    3-uniform hypergraph.  Hyperedges are 3-combinations of node indices,
    either without repetition (data_cfg.hyperedge_repetition = False,
    C(100,3) = 161,700 hyperedges per jet) or with repetition
    (data_cfg.hyperedge_repetition = True, C(102,3) = 171,700 hyperedges).
    Hyperedge weight (ECF3):
        w_h = z_i * z_j * z_k * (theta_{ij} * theta_{ik} * theta_{jk})^{beta}
    Hyperedge features are not assigned.
    Hyperedge index is in COO incidence format compatible with
    torch_geometric.nn.conv.HypergraphConv:
        hyperedge_index[0] : global node indices
        hyperedge_index[1] : global hyperedge indices

Node features (4 channels per constituent, shape [N_jets*100, 4]):
    [ pT_part / jet_pT,   Delta_eta,   Delta_phi,   E_part / jet_E ]

HDF5 schema (Refer to ntuples_to_h5.py for details (https://github.com/neutrino-man4/jetsim/blob/master/h5_maker/ntuples_to_h5.py)):
    jetConstituentsList    (N, 100,  3)  last axis : [Delta_eta, Delta_phi, pT_part]
    jetConstituentsExtra   (N, 100, 10)  last axis index 3 : E_part
    jetFeatures            (N,      13)  index 0 : jet_pT ;  index 3 : jet_E
    pair_delta_R           (N, 100, 100) strictly upper-triangular pairwise DeltaR
    constituent_pt_weight  (N, 100)      z_i = pT_i / sum_k pT_k
    jetConstituentsMask    (N, 100)      bool, True = real (non-padded) particle
    truth_label            (N,)          int8 class label

Memory note (hypergraph mode):
    In principle: 
    C(100,3) = 161,700 triplets -> hyperedge_index shape (2, B*485100) int64.
    At batch_size=128 this is ~993 MB of index data alone.
    Reduce batch_size (e.g. 8-16) when operating in hypergraph mode on
    memory-constrained systems.
    In practice: 
    We should be using only up to 10 particles per jet (AK8 jets), so the effective hyperedge count is much lower.
    We are limited by how many dimensions of the QFI can be effectively calculated. 

Author: Aritra Bal (ETP)
Date  : 2026-02-24
"""

import argparse
import itertools
import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

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
# Constants (must be consistent with ntuples_to_h5.py)
# ---------------------------------------------------------------------------

# Indices into the jetFeatures array  (JET_FEATURE_NAMES in ntuples_to_h5.py):
#   0: jet_pt   1: jet_eta   2: jet_phi   3: jet_energy   4: jet_nparticles ...
_JET_PT_IDX: int = 0
_JET_ENERGY_IDX: int = 3

# Index of E_part inside the last axis of jetConstituentsExtra
# (EXTRA_FEATURE_NAMES: px=0, py=1, pz=2, energy=3, charge=4, pid=5, ...)
_EXTRA_ENERGY_IDX: int = 3


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """
    Configuration dataclass for JetGraphDataLoader.

    Attributes
    ----------
    batch_size : int
        Jets per batch.
    num_jets : Optional[int]
        Maximum total jets to consume across all files; None means all jets.
    hyper : bool
        Build a 3-uniform hypergraph if True; fully-connected graph if False.
    ecf_beta : float
        Angular exponent beta used in ECF2 / ECF3 edge/hyperedge weights.
    hyperedge_repetition : bool
        If True, triplets are drawn with index repetition
        (combinations_with_replacement); if False, strict combinations
        (i < j < k).  Only relevant when hyper=True.
    """

    batch_size: int = 128
    num_constituents:int = 10
    num_jets: Optional[int] = None
    hyper: bool = False
    ecf_beta: float = 1.0
    hyperedge_repetition: bool = False


# ---------------------------------------------------------------------------
# JetGraphDataLoader
# ---------------------------------------------------------------------------

class JetGraphDataLoader:
    """
    Streaming iterator that yields batched PyG Data objects from HDF5 files.

    The connectivity templates (edge pairs or hyperedge triplets) are computed
    once at construction time and reused for every batch, so per-batch overhead
    is limited to index arithmetic and ECF weight computation.

    Usage
    -----
    >>> cfg = DataConfig(batch_size=64, hyper=False, ecf_beta=1.0)
    >>> loader = JetGraphDataLoader(["/path/to/file.h5"], cfg)
    >>> for batch in loader:
    ...     # batch is a torch_geometric.data.Data object
    ...     out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    """

    def __init__(self, h5_files: List[str], data_cfg: DataConfig) -> None:
        """
        Parameters
        ----------
        h5_files : List[str]
            Paths to HDF5 files produced by ntuples_to_h5.py.  Files are
            streamed in order; all jets within a file are consumed before
            moving to the next.
        data_cfg : DataConfig
            Loader configuration object.
        """
        self.h5_files: List[str] = h5_files
        self.cfg: DataConfig = data_cfg

        # Pre-compute connectivity templates
        if data_cfg.hyper:
            self._init_hyperedge_template()
        else:
            self._init_edge_template()

        # Streaming state (reset by __iter__)
        self._current_file_idx: int = 0
        self._current_jet_idx: int = 0
        self._current_file: Optional[h5py.File] = None
        self._current_file_size: int = 0
        self._jets_yielded: int = 0

        if data_cfg.hyper:
            logger.info(
                "JetGraphDataLoader | mode=HYPERGRAPH | triplets/jet=%d "
                "| repetition=%s | beta=%.3f | files=%d",
                self._num_hyperedges,
                data_cfg.hyperedge_repetition,
                data_cfg.ecf_beta,
                len(h5_files),
            )
        else:
            logger.info(
                "JetGraphDataLoader | mode=GRAPH | edges/jet=%d (undirected) "
                "| beta=%.3f | files=%d",
                self._num_edges,
                data_cfg.ecf_beta,
                len(h5_files),
            )

    # ------------------------------------------------------------------
    # Connectivity template construction
    # ------------------------------------------------------------------

    def _init_edge_template(self) -> None:
        """
        Build the undirected edge index template for a fully-connected graph
        on self.cfg.num_constituents nodes.

        The upper-triangular pair list (i < j, shape [C(P,2), 2]) is stored
        separately for vectorised ECF2 extraction from pair_delta_R.
        The full undirected edge_index (both i->j and j->i) has shape
        (2, 2*C(P,2)) and is built by concatenation.
        """
        pairs_upper = np.array(
            list(itertools.combinations(range(self.cfg.num_constituents), 2)),
            dtype=np.int64,
        )  # (C(100,2) = 4950, 2); guaranteed i < j

        src: np.ndarray = np.concatenate([pairs_upper[:, 0], pairs_upper[:, 1]])
        dst: np.ndarray = np.concatenate([pairs_upper[:, 1], pairs_upper[:, 0]])

        # Shape (2, 9900): full undirected template, no per-jet offset applied yet
        self._edge_index_template: np.ndarray = np.stack([src, dst], axis=0)

        # Upper-triangle pairs for ECF2 lookup (i < j always valid for pair_delta_R)
        self._edge_pairs_upper: np.ndarray = pairs_upper  # (4950, 2)
        self._num_upper_edges: int = pairs_upper.shape[0]  # 4950
        self._num_edges: int = int(src.shape[0])            # 9900

    def _init_hyperedge_template(self) -> None:
        """
        Build the triplet list and COO incidence matrix template for a
        3-uniform hypergraph on self.cfg.num_constituents nodes.

        Triplets satisfy i <= j <= k (with repetition) or i < j < k (without).
        Because pair_delta_R is strictly upper-triangular (triu k=1 in
        ntuples_to_h5.py), all three pairwise accesses theta_{ij}, theta_{ik},
        theta_{jk} are always in the upper triangle, so no index transposition
        is required.
            hyperedge_index[0] : node indices
            hyperedge_index[1] : hyperedge indices
        For triplet (i, j, k) with hyperedge index h, three entries are added:
            (i, h), (j, h), (k, h).
        """
        if self.cfg.hyperedge_repetition:
            gen = itertools.combinations_with_replacement(range(self.cfg.num_constituents), 3)
        else:
            gen = itertools.combinations(range(self.cfg.num_constituents), 3)

        triplets: np.ndarray = np.array(list(gen), dtype=np.int64)
        # Shape: (num_hyper, 3); rows satisfy i <= j <= k or i < j < k

        self._triplets: np.ndarray = triplets
        self._num_hyperedges: int = int(triplets.shape[0])

        num_h: int = self._num_hyperedges
        node_row: np.ndarray  = triplets.flatten()                               # (3*num_h,)
        hedge_row: np.ndarray = np.repeat(np.arange(num_h, dtype=np.int64), 3)  # (3*num_h,)

        # Shape (2, 3*num_h): reused every batch, offsets added per graph
        self._hyperedge_index_template: np.ndarray = np.stack(
            [node_row, hedge_row], axis=0
        )

    # ------------------------------------------------------------------
    # Iterator protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of full (or partial) batches."""
        total: int = self._count_total_jets()
        return max(1, (total + self.cfg.batch_size - 1) // self.cfg.batch_size)

    def __iter__(self) -> "JetGraphDataLoader":
        """Reset streaming state and open the first file."""
        self._current_file_idx = 0
        self._current_jet_idx = 0
        self._jets_yielded = 0
        self._close_file()
        self._open_file()
        return self

    def __next__(self) -> Data:
        """
        Yield the next batched Data object.

        Raises StopIteration when the jet budget (data_cfg.num_jets) is
        exhausted or all files have been consumed.
        """
        # Check global jet budget
        if (
            self.cfg.num_jets is not None
            and self._jets_yielded >= self.cfg.num_jets
        ):
            self._close_file()
            raise StopIteration

        # Advance past exhausted files
        while self._current_jet_idx >= self._current_file_size:
            self._close_file()
            self._current_file_idx += 1
            if self._current_file_idx >= len(self.h5_files):
                raise StopIteration
            self._open_file()
            self._current_jet_idx = 0

        # Determine actual slice end, respecting both file boundary and budget
        batch_end: int = min(
            self._current_jet_idx + self.cfg.batch_size,
            self._current_file_size,
        )
        if self.cfg.num_jets is not None:
            remaining: int = self.cfg.num_jets - self._jets_yielded
            batch_end = min(batch_end, self._current_jet_idx + remaining)

        batch: Data = self._read_batch(self._current_jet_idx, batch_end)

        consumed: int = batch_end - self._current_jet_idx
        self._current_jet_idx = batch_end
        self._jets_yielded += consumed
        return batch

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _open_file(self) -> None:
        """Open the HDF5 file at self._current_file_idx."""
        path: str = self.h5_files[self._current_file_idx]
        self._current_file = h5py.File(path, "r")
        self._current_file_size = int(self._current_file["truth_label"].shape[0])
        logger.debug("Opened  %s  (%d jets)", path, self._current_file_size)

    def _close_file(self) -> None:
        """Close the currently open HDF5 file, if any."""
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

    def __del__(self) -> None:
        self._close_file()

    def _count_total_jets(self) -> int:
        """Count jets across all files, clipped to data_cfg.num_jets."""
        total: int = 0
        for path in self.h5_files:
            with h5py.File(path, "r") as f:
                total += int(f["truth_label"].shape[0])
        if self.cfg.num_jets is not None:
            total = min(total, self.cfg.num_jets)
        return total

    # ------------------------------------------------------------------
    # Core batch construction
    # ------------------------------------------------------------------

    def _read_batch(self, start: int, end: int) -> Data:
        """
        Read jets in [start, end) from the currently open HDF5 file and
        build the corresponding batched PyG Data object.

        Parameters
        ----------
        start, end : int
            Row slice into every HDF5 dataset.

        Returns
        -------
        torch_geometric.data.Data
            Graph or hypergraph batch; field layout depends on data_cfg.hyper.
        """
        f: h5py.File = self._current_file
        B: int = end - start
        N: int = self.cfg.num_constituents
        # ---- HDF5 reads (cast to float32 immediately to reduce memory) ------

        # jetConstituentsList  (B, 100, 3)  last axis = [Delta_eta, Delta_phi, pT_part]
        constituents: np.ndarray = (
            f["jetConstituentsList"][start:end, :N, :].astype(np.float32)
        )

        # jetConstituentsExtra (B, 100, 10) index 3 along last axis = E_part
        # Slice only the energy channel directly to avoid reading 10 channels
        part_E: np.ndarray = (
            f["jetConstituentsExtra"][start:end, :N, _EXTRA_ENERGY_IDX].astype(np.float32)
        )  # (B, NUM_CONSTITUENTS)

        # jetFeatures (B, 13); index 0 = jet_pT, index 3 = jet_E
        jet_features: np.ndarray = (
            f["jetFeatures"][start:end].astype(np.float32)
        )  # (B, 13)

        # pair_delta_R (B, NUM_CONSTITUENTS, NUM_CONSTITUENTS) strictly upper-triangular DeltaR matrix
        pair_dr: np.ndarray = (
            f["pair_delta_R"][start:end, :N, :N].astype(np.float32)
        )  # (B, NUM_CONSTITUENTS, NUM_CONSTITUENTS)

        # constituent_pt_weight (B, NUM_CONSTITUENTS)  z_i = pT_i / sum_k pT_k
        z_weights: np.ndarray = (
            f["constituent_pt_weight"][start:end, :N].astype(np.float32)
        )  # (B, NUM_CONSTITUENTS)

        # jetConstituentsMask (B, NUM_CONSTITUENTS) bool
        mask: np.ndarray = f["jetConstituentsMask"][start:end, :N]
        # truth_label (B,)
        labels: np.ndarray = f["truth_label"][start:end].astype(np.int64)

        # ---- Jet-level normalisers ------------------------------------------

        # jet_pT  : (B,) -> broadcast denominator (B, 1)
        jet_pt: np.ndarray = jet_features[:, _JET_PT_IDX]
        jet_E: np.ndarray  = jet_features[:, _JET_ENERGY_IDX]

        # Guard against degenerate jets (should not occur for valid AK8 jets,
        # but protects against NaN propagation)
        jet_pt_safe: np.ndarray = np.where(jet_pt > 0.0, jet_pt, 1.0)[:, np.newaxis]  # (B,1)
        jet_E_safe: np.ndarray  = np.where(jet_E  > 0.0, jet_E,  1.0)[:, np.newaxis]  # (B,1)

        # ---- Node features (4 channels) ------------------------------------

        # jetConstituentsList last axis:  0=Delta_eta  1=Delta_phi  2=pT_part
        part_deta: np.ndarray = constituents[:, :, 0]   # (B, NUM_CONSTITUENTS)
        part_dphi: np.ndarray = constituents[:, :, 1]   # (B, NUM_CONSTITUENTS)
        part_pt: np.ndarray   = constituents[:, :, 2]   # (B, NUM_CONSTITUENTS) [GeV]

        # Normalise pT and E to jet-level quantities; angular coords are already
        # relative to the jet axis and require no further normalisation
        pt_norm: np.ndarray = part_pt / jet_pt_safe   # (B, NUM_CONSTITUENTS)
        E_norm: np.ndarray  = part_E  / jet_E_safe    # (B, NUM_CONSTITUENTS)

        # (B, NUM_CONSTITUENTS, 4) -> flatten to (B*NUM_CONSTITUENTS, 4); channel order documented in header
        x_np: np.ndarray = np.stack(
            [pt_norm, part_deta, part_dphi, E_norm], axis=-1
        ).reshape(B * self.cfg.num_constituents, 4).astype(np.float32)

        # ---- Common PyG tensors --------------------------------------------
        x_tensor: torch.Tensor    = torch.from_numpy(x_np)
        mask_flat: torch.Tensor   = torch.from_numpy(mask.reshape(B * self.cfg.num_constituents))
        y_tensor: torch.Tensor    = torch.from_numpy(labels)
        # batch assignment: 0,0,...,0,  1,1,...,1,  ...,  (B-1),...,(B-1)
        # each graph contributes self.cfg.num_constituents entries
        batch_tensor: torch.Tensor = torch.repeat_interleave(
            torch.arange(B, dtype=torch.long), self.cfg.num_constituents
        )

        if self.cfg.hyper:
            return self._build_hypergraph_batch(
                B, pair_dr, z_weights,
                x_tensor, mask_flat, y_tensor, batch_tensor,
            )
        return self._build_graph_batch(
            B, pair_dr, z_weights,
            x_tensor, mask_flat, y_tensor, batch_tensor,
        )

    # ------------------------------------------------------------------
    # Ordinary graph batch
    # ------------------------------------------------------------------

    def _build_graph_batch(
        self,
        B: int,
        pair_dr: np.ndarray,
        z_weights: np.ndarray,
        x: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        batch: torch.Tensor,
    ) -> Data:
        """
        Build a fully-connected undirected graph batch.

        Edge scalar feature (ECF2):
            e_{ij} = z_i * z_j * theta_{ij}^{beta}
        Both directed edges (i->j) and (j->i) share the same feature value,
        consistent with the undirected convention used by PyG message-passing
        layers (MessagePassing with flow='source_to_target').

        Parameters
        ----------
        B : int
            Batch size (number of jets).
        pair_dr : np.ndarray, shape (B, 100, 100)
            Strictly upper-triangular DeltaR matrix for each jet.
        z_weights : np.ndarray, shape (B, 100)
            pT-fraction weights z_i per constituent.
        x, mask, y, batch : torch.Tensor
            Pre-built node, mask, label and batch-assignment tensors.

        Returns
        -------
        torch_geometric.data.Data
            Fields: x, edge_index, edge_attr, mask, y, batch
        """
        beta: float = self.cfg.ecf_beta

        # Upper-triangle pair indices (i < j) -> valid for pair_dr lookup
        pairs: np.ndarray = self._edge_pairs_upper  # (4950, 2)
        i_idx: np.ndarray = pairs[:, 0]             # (4950,)
        j_idx: np.ndarray = pairs[:, 1]             # (4950,)

        # Advanced indexing: arr[:, idx] yields (B, 4950) for 1-D idx
        zi: np.ndarray    = z_weights[:, i_idx]           # (B, 4950)
        zj: np.ndarray    = z_weights[:, j_idx]           # (B, 4950)
        theta: np.ndarray = pair_dr[:, i_idx, j_idx]      # (B, 4950)

        ecf2: np.ndarray  = zi * zj * (theta ** beta)     # (B, 4950)

        # Duplicate for both edge directions; both carry the same scalar
        # (B, 4950) cat (B, 4950) -> (B, 9900) -> (B*9900, 1)
        ecf2_sym: np.ndarray = np.concatenate([ecf2, ecf2], axis=1).reshape(
            B * self._num_edges, 1
        ).astype(np.float32)

        # Build batched edge_index ----------------------------------------
        # np.tile(shape (2,9900), B) -> (2, B*9900) via last-axis tiling
        edge_index_batched: np.ndarray = np.tile(self._edge_index_template, B)

        # Add per-jet node offset: jet n -> all its node indices += n*self.cfg.num_constituents
        # node_offsets shape (B*9900,): [0]*9900, [100]*9900, ..., [(B-1)*100]*9900
        node_offsets: np.ndarray = np.repeat(
            np.arange(B, dtype=np.int64) * self.cfg.num_constituents,
            self._num_edges,
        )
        # In-place addition is safe: np.tile returns a new array
        edge_index_batched[0] += node_offsets
        edge_index_batched[1] += node_offsets

        return Data(
            x          = x,
            edge_index = torch.from_numpy(edge_index_batched),
            edge_attr  = torch.from_numpy(ecf2_sym),
            mask       = mask,
            y          = y,
            batch      = batch,
        )

    # ------------------------------------------------------------------
    # Hypergraph batch
    # ------------------------------------------------------------------

    def _build_hypergraph_batch(
        self,
        B: int,
        pair_dr: np.ndarray,
        z_weights: np.ndarray,
        x: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        batch: torch.Tensor,
    ) -> Data:
        """
        Build a batched 3-uniform hypergraph.

        Hyperedge weight (ECF3):
            w_h = z_i * z_j * z_k * (theta_{ij} * theta_{ik} * theta_{jk})^{beta}

        All triplet indices satisfy i <= j <= k (strict i < j < k when
        hyperedge_repetition=False), so every pairwise DeltaR access maps to
        the upper triangle of pair_dr without any transposition.

        Hyperedge features are not assigned per specification.

        Hyperedge index format (PyG HypergraphConv-compatible):
            Row 0: node indices    (global, with per-jet offset)
            Row 1: hyperedge idx   (global, with per-jet offset)
        Each triplet h=(i,j,k) contributes three incidences:
            (node=i, hedge=h), (node=j, hedge=h), (node=k, hedge=h)

        Parameters
        ----------
        B : int
            Batch size.
        pair_dr : np.ndarray, shape (B, 100, 100)
        z_weights : np.ndarray, shape (B, 100)
        x, mask, y, batch : torch.Tensor

        Returns
        -------
        torch_geometric.data.Data
            Fields: x, hyperedge_index, hyperedge_weight, mask, y, batch,
                    num_hyperedges
        """
        beta: float  = self.cfg.ecf_beta
        triplets: np.ndarray = self._triplets   # (num_h, 3); i<=j<=k
        num_h: int   = self._num_hyperedges

        i_idx: np.ndarray = triplets[:, 0]   # (num_h,)
        j_idx: np.ndarray = triplets[:, 1]   # (num_h,)
        k_idx: np.ndarray = triplets[:, 2]   # (num_h,)

        # pT-fraction weights for each vertex of every triplet -> (B, num_h)
        zi: np.ndarray = z_weights[:, i_idx]
        zj: np.ndarray = z_weights[:, j_idx]
        zk: np.ndarray = z_weights[:, k_idx]

        # Pairwise DeltaR for each triplet --------------------------------
        # All accesses are to the upper triangle (guaranteed by i<=j<=k):
        #   theta_{ij}: row=i, col=j, with i<=j  -> upper or diagonal (diag=0)
        #   theta_{ik}: row=i, col=k, with i<=k  -> same
        #   theta_{jk}: row=j, col=k, with j<=k  -> same
        # Diagonal elements of pair_dr are 0 (triu k=1 in ntuples_to_h5.py),
        # so repeated-index triplets automatically contribute zero ECF3.
        theta_ij: np.ndarray = pair_dr[:, i_idx, j_idx]   # (B, num_h)
        theta_ik: np.ndarray = pair_dr[:, i_idx, k_idx]   # (B, num_h)
        theta_jk: np.ndarray = pair_dr[:, j_idx, k_idx]   # (B, num_h)

        # ECF3: scalar product of z-weights times the angle factor
        # Angle factor: (theta_ij * theta_ik * theta_jk)^beta
        # When beta>0: 0^beta=0, so padded-node contributions vanish automatically.
        # When beta=0: z_i=0 for padded nodes suppresses their contribution.
        angle_product: np.ndarray = theta_ij * theta_ik * theta_jk  # (B, num_h)
        ecf3: np.ndarray = zi * zj * zk * (angle_product ** beta)   # (B, num_h)

        # Flatten to (B*num_h,) float32
        hyperedge_weight_flat: np.ndarray = ecf3.reshape(B * num_h).astype(np.float32)

        # Build batched COO hyperedge_index --------------------------------
        # Template rows:
        #   tmpl_node  (3*num_h,): node index within a single jet (0..99)
        #   tmpl_hedge (3*num_h,): hyperedge index within a single jet (0..num_h-1)
        tmpl_node:  np.ndarray = self._hyperedge_index_template[0]   # (3*num_h,)
        tmpl_hedge: np.ndarray = self._hyperedge_index_template[1]   # (3*num_h,)

        # Per-jet offsets: (B, 1) broadcast against (1, 3*num_h)
        # Avoids materialising B full copies of the template before adding offsets.
        node_offsets_2d: np.ndarray = (
            np.arange(B, dtype=np.int64)[:, np.newaxis] * self.cfg.num_constituents
        )  # (B, 1)
        hedge_offsets_2d: np.ndarray = (
            np.arange(B, dtype=np.int64)[:, np.newaxis] * num_h
        )  # (B, 1)

        # Broadcasting: (1, 3*num_h) + (B, 1) -> (B, 3*num_h); then flatten
        node_row_batched: np.ndarray = (
            tmpl_node[np.newaxis, :] + node_offsets_2d
        ).reshape(-1)  # (B*3*num_h,)

        hedge_row_batched: np.ndarray = (
            tmpl_hedge[np.newaxis, :] + hedge_offsets_2d
        ).reshape(-1)  # (B*3*num_h,)

        hyperedge_index: np.ndarray = np.stack(
            [node_row_batched, hedge_row_batched], axis=0
        ).astype(np.int64)  # (2, B*3*num_h)

        return Data(
            x                = x,
            hyperedge_index  = torch.from_numpy(hyperedge_index),
            hyperedge_weight = torch.from_numpy(hyperedge_weight_flat),
            # hyperedge_attr intentionally omitted (per specification)
            mask             = mask,
            y                = y,
            batch            = batch,
            # total hyperedges in this batch; useful for downstream normalisation
            num_hyperedges   = torch.tensor(B * num_h, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Standalone test: load 10 batches of size 128 from a single HDF5 file
    and print shape / summary statistics for both graph and hypergraph modes.
    """
    parser = argparse.ArgumentParser(
        description="Smoke-test JetGraphDataLoader on a single HDF5 file."
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="/ceph/abal/QFIT/MC/HDF5/TTBar/run_01/TTBar.h5",
        help="Path to a single HDF5 file.",
    )
    args = parser.parse_args()

    N_BATCHES: int = 10
    BATCH_SIZE: int = 128
    BETA: float = 1.0

    for hyper in (False, True):
        cfg = DataConfig(
            batch_size=BATCH_SIZE,
            num_jets=N_BATCHES * BATCH_SIZE,
            hyper=hyper,
            ecf_beta=BETA,
            hyperedge_repetition=False,
        )
        loader = JetGraphDataLoader([args.test_file], cfg)

        mode_label: str = "HYPERGRAPH" if hyper else "GRAPH"
        separator: str = "=" * 62
        print(f"\n{separator}")
        print(f"  MODE : {mode_label}  |  beta={BETA}  |  batch_size={BATCH_SIZE}")
        print(separator)

        batches_seen: int = 0
        for batch_idx, batch in enumerate(loader):
            batches_seen += 1

            if batch_idx == 0:
                # Node features
                print(f"  x (node features)   : shape = {tuple(batch.x.shape)}")
                print(f"    channel layout     : [pT/jet_pT, deta, dphi, E/jet_E]")
                print(f"  batch assignment    : shape = {tuple(batch.batch.shape)}")
                print(f"  labels y            : shape = {tuple(batch.y.shape)}")
                print(f"  mask                : shape = {tuple(batch.mask.shape)}")
                print(
                    f"  real particles/batch: "
                    f"{int(batch.mask.sum().item())} / {batch.mask.numel()}"
                )

                if hyper:
                    hi: torch.Tensor = batch.hyperedge_index
                    hw: torch.Tensor = batch.hyperedge_weight
                    print(f"  hyperedge_index     : shape = {tuple(hi.shape)}")
                    print(f"  num_hyperedges      : {batch.num_hyperedges.item()}")
                    print(
                        f"  hyperedge_weight    : range = "
                        f"[{hw.min().item():.4e}, {hw.max().item():.4e}]"
                    )
                    # Sanity: every hyperedge index should be >=0 and < B*num_nodes
                    n_nodes_total: int = batch.x.shape[0]
                    n_hedge_total: int = int(batch.num_hyperedges.item())
                    assert hi[0].max().item() < n_nodes_total, (
                        "Node index out of range in hyperedge_index"
                    )
                    assert hi[1].max().item() < n_hedge_total, (
                        "Hyperedge index out of range in hyperedge_index"
                    )
                    print("  [sanity] hyperedge_index bounds OK")

                else:
                    ei: torch.Tensor = batch.edge_index
                    ea: torch.Tensor = batch.edge_attr
                    print(f"  edge_index          : shape = {tuple(ei.shape)}")
                    print(f"  num edges (total)   : {ei.shape[1]}")
                    print(f"  edge_attr           : shape = {tuple(ea.shape)}")
                    print(
                        f"  edge_attr           : range = "
                        f"[{ea.min().item():.4e}, {ea.max().item():.4e}]"
                    )
                    # Sanity: edge indices should be in [0, B*100)
                    n_nodes_total = batch.x.shape[0]
                    assert ei.max().item() < n_nodes_total, (
                        "Node index out of range in edge_index"
                    )
                    assert ei.min().item() >= 0, (
                        "Negative node index in edge_index"
                    )
                    print("  [sanity] edge_index bounds OK")

        print(f"  Loaded {batches_seen} / {N_BATCHES} batches successfully.")


if __name__ == "__main__":
    main()