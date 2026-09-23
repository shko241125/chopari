"""MaleCNS edge-table adapter. W[post, pre] stores model millivolts per spike."""

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
from typing import Iterator

import numpy as np
from scipy import sparse

OFFICIAL_EDGES = {
    "name": "MaleCNS v1.0 minconf 0.5 connection weights",
    "url": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "sha256": "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
    "bytes": 1051241946,
    "columns": ["body_pre", "body_post", "weight"],
}
OFFICIAL_NEUROTRANSMITTERS = {
    "sha256": "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
    "bytes": 43282834,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_rows(path: Path, columns: list[str]) -> Iterator[dict]:
    """Feather V2 via memory mapped Arrow record batches; CSV for small fixtures."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not set(columns).issubset(reader.fieldnames or []):
                raise ValueError(f"Missing columns {set(columns) - set(reader.fieldnames or [])}")
            yield from reader
    elif path.suffix.lower() == ".feather":
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise RuntimeError("Feather requires pip install '.[feather]'") from exc
        with pa.memory_map(str(path), "r") as source:
            reader = ipc.open_file(source)
            if not set(columns).issubset(reader.schema.names):
                raise ValueError(f"Missing columns {set(columns) - set(reader.schema.names)}")
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i).select(columns).to_pydict()
                for values in zip(*(batch[col] for col in columns)):
                    yield dict(zip(columns, values))
    else:
        raise ValueError("Expected .csv or .feather")


def iter_edge_batches(path: Path, batch_size: int = 250_000):
    """Vectorized official Feather import; CSV is reserved for small smoke fixtures."""
    columns = OFFICIAL_EDGES["columns"]
    if path.suffix.lower() == ".csv":
        arrays = {name: [] for name in columns}
        for row in iter_rows(path, columns):
            for name in columns:
                arrays[name].append(int(row[name]))
            if len(arrays[columns[0]]) >= batch_size:
                yield tuple(np.asarray(arrays[name], dtype=np.int64) for name in columns)
                arrays = {name: [] for name in columns}
        if arrays[columns[0]]:
            yield tuple(np.asarray(arrays[name], dtype=np.int64) for name in columns)
        return
    if path.suffix.lower() != ".feather":
        raise ValueError("Expected .csv or .feather")
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("Feather requires pip install '.[feather]'") from exc
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        if not set(columns).issubset(reader.schema.names):
            raise ValueError("Official edges require body_pre, body_post, weight")
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            for start in range(0, batch.num_rows, batch_size):
                part = batch.slice(start, batch_size)
                yield tuple(part.column(part.schema.get_field_index(name)).to_numpy(
                    zero_copy_only=False).astype(np.int64, copy=False) for name in columns)


def load_ids(path: Path, column: str = "bodyId", filter_column: str | None = None,
             filter_value: str | None = None) -> np.ndarray:
    columns = [column] + ([filter_column] if filter_column else [])
    ids = [int(row[column]) for row in iter_rows(path, columns)
           if not filter_column or str(row[filter_column]) == filter_value]
    if not ids:
        raise ValueError("Selection produced zero IDs")
    return np.array(sorted(set(ids)), dtype=np.int64)


def load_signs(path: Path, id_column: str = "body", sign_column: str = "sign") -> dict[int, int]:
    signs = {}
    for row in iter_rows(path, [id_column, sign_column]):
        sign = int(row[sign_column])
        if sign not in (-1, 1):
            raise ValueError("Signs must be -1 or +1; resolve unknowns explicitly")
        key = int(row[id_column])
        if key in signs and signs[key] != sign:
            raise ValueError(f"Conflicting signs for {key}")
        signs[key] = sign
    return signs


def load_neurotransmitter_signs(path: Path) -> dict[int, int]:
    """Coarse explicit convention. Receptors and modulators remain unmodelled."""
    sign_by_name = {"acetylcholine": 1, "gaba": -1,
                    "glutamate": -1, "histamine": -1}
    signs = {}
    for row in iter_rows(path, ["body", "consensus_nt"]):
        if row["body"] is None:
            continue
        key = int(row["body"])
        name = str(row["consensus_nt"] or "").lower().strip()
        value = sign_by_name.get(name, 1)  # uncertain and modulators: explicit +1 fallback
        if key in signs and signs[key] != value:
            raise ValueError(f"Conflicting transmitter signs for body {key}")
        signs[key] = value
    return signs


@dataclass
class Graph:
    ids: np.ndarray
    weights: sparse.csr_matrix
    manifest: dict

    def __post_init__(self):
        self.ids = np.asarray(self.ids, dtype=np.int64)
        if self.ids.ndim != 1 or not np.all(self.ids[1:] > self.ids[:-1]):
            raise ValueError("Neuron IDs must be strictly increasing int64")
        self.weights = self.weights.tocsr()
        if self.weights.shape != (len(self.ids), len(self.ids)):
            raise ValueError("Weight matrix shape does not match IDs")
        self.weights.sum_duplicates()
        self.weights.sort_indices()
        if not np.all(np.isfinite(self.weights.data)):
            raise ValueError("Weights must be finite")

    def index(self, body_id: int) -> int:
        index = int(np.searchsorted(self.ids, body_id))
        if index == len(self.ids) or self.ids[index] != body_id:
            raise KeyError(body_id)
        return index

    @classmethod
    def from_edges(cls, ids, rows, gain_mv: float, signs: dict[int, int] | None = None,
                   source: dict | None = None):
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        signs = signs or {}
        if not np.isfinite(gain_mv) or gain_mv <= 0:
            raise ValueError("gain_mv must be finite and positive")
        pre, post, values = [], [], []
        seen = kept = contacts = unknown = 0
        for row in rows:
            seen += 1
            a, b, n = int(row["body_pre"]), int(row["body_post"]), int(row["weight"])
            if n <= 0:
                raise ValueError("Contact counts must be positive integers")
            ai, bi = int(np.searchsorted(ids, a)), int(np.searchsorted(ids, b))
            if ai >= len(ids) or bi >= len(ids) or ids[ai] != a or ids[bi] != b:
                continue
            kept += 1
            contacts += n
            if a not in signs:
                unknown += 1
            pre.append(ai); post.append(bi); values.append(n * gain_mv * signs.get(a, 1))
        weights = sparse.coo_matrix((np.asarray(values, dtype=np.float64),
            (np.asarray(post, dtype=np.int32), np.asarray(pre, dtype=np.int32))),
            shape=(len(ids), len(ids))).tocsr()
        manifest = {"dataset": source or {"name": "explicit fixture"},
                    "selection": "explicit neuron IDs; retain edges iff both endpoints selected",
                    "unknown_sign_policy": "positive and counted; no receptor physiology implied",
                    "gain_mv_per_contact": gain_mv, "neuron_count": len(ids),
                    "input_edge_rows": seen, "retained_edge_rows": kept,
                    "aggregated_edge_rows": weights.nnz, "retained_contacts": contacts,
                    "retained_rows_with_unknown_presynaptic_sign": unknown,
                    "orientation": "W[postsynaptic_index, presynaptic_index]"}
        return cls(ids, weights, manifest)

    @classmethod
    def from_edge_batches(cls, ids, batches, gain_mv: float,
                          signs: dict[int, int] | None = None, source: dict | None = None):
        """Filter each batch before collecting; selected graph still needs RAM to sort."""
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        if not len(ids) or len(ids) >= 2**31:
            raise ValueError("Invalid selected neuron count")
        if not np.isfinite(gain_mv) or gain_mv <= 0:
            raise ValueError("gain_mv must be finite and positive")
        signs = signs or {}
        sign_vector = np.array([signs.get(int(id_), 1) for id_ in ids], dtype=np.int8)
        known_vector = np.array([int(id_) in signs for id_ in ids], dtype=bool)
        pre_parts, post_parts, val_parts = [], [], []
        seen = kept = contacts = unknown = 0
        for pre, post, count in batches:
            if not (pre.shape == post.shape == count.shape) or np.any(count <= 0):
                raise ValueError("Invalid batch shapes or non-positive contact count")
            seen += len(pre)
            ai, bi = np.searchsorted(ids, pre), np.searchsorted(ids, post)
            valid = ((ai < len(ids)) & (bi < len(ids)) &
                     (ids[np.minimum(ai, len(ids)-1)] == pre) &
                     (ids[np.minimum(bi, len(ids)-1)] == post))
            a, b, c = ai[valid].astype(np.int32), bi[valid].astype(np.int32), count[valid]
            kept += len(c); contacts += int(c.sum(dtype=np.int64))
            unknown += int((~known_vector[a]).sum())
            pre_parts.append(a); post_parts.append(b)
            val_parts.append(c.astype(np.float64) * gain_mv * sign_vector[a])
        matrix = sparse.coo_matrix((np.concatenate(val_parts) if val_parts else [],
                (np.concatenate(post_parts) if post_parts else [],
                 np.concatenate(pre_parts) if pre_parts else [])),
                shape=(len(ids), len(ids))).tocsr()
        manifest = {"dataset": source or {"name": "explicit fixture"},
                    "selection": "explicit neuron IDs; retain edges iff both endpoints selected",
                    "unknown_sign_policy": "positive and counted; no receptor physiology implied",
                    "gain_mv_per_contact": gain_mv, "neuron_count": len(ids),
                    "input_edge_rows": seen, "retained_edge_rows": kept,
                    "aggregated_edge_rows": matrix.nnz, "retained_contacts": contacts,
                    "retained_rows_with_unknown_presynaptic_sign": unknown,
                    "orientation": "W[postsynaptic_index, presynaptic_index]"}
        return cls(ids, matrix, manifest)

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "ids.npy", self.ids, allow_pickle=False)
        sparse.save_npz(directory / "weights.npz", self.weights, compressed=False)
        manifest = dict(self.manifest)
        manifest["artifacts_sha256"] = {name: sha256_file(directory / name)
             for name in ("ids.npy", "weights.npz")}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, directory: Path):
        manifest = json.loads((directory / "manifest.json").read_text())
        for name, digest in manifest["artifacts_sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError(f"Graph artifact hash mismatch: {name}")
        return cls(np.load(directory / "ids.npy", allow_pickle=False),
                   sparse.load_npz(directory / "weights.npz"), manifest)
