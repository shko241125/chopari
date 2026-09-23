import argparse
import json
from pathlib import Path

import numpy as np

from .graph import (Graph, OFFICIAL_EDGES, OFFICIAL_NEUROTRANSMITTERS,
                    iter_edge_batches, load_ids, load_signs,
                    load_neurotransmitter_signs, sha256_file)
from .lif import LIF, LIFParameters, RateReadout, signed_turn


def build(args):
    source_path = Path(args.edges)
    source_hash = sha256_file(source_path)
    source = {"file": source_path.name, "bytes": source_path.stat().st_size,
              "sha256": source_hash}
    if source_path.suffix.lower() == ".feather":
        if (source_hash != OFFICIAL_EDGES["sha256"] or
                source_path.stat().st_size != OFFICIAL_EDGES["bytes"]):
            raise ValueError("Feather input does not match the locked official MaleCNS v1.0 file")
        source = {**OFFICIAL_EDGES, **source}
        if not args.neurotransmitters:
            raise ValueError("Official MaleCNS import requires --neurotransmitters")
    ids_path = Path(args.ids)
    ids = load_ids(ids_path, args.id_column, args.filter_column, args.filter_value)
    source["ids_source"] = {"file": ids_path.name, "sha256": sha256_file(ids_path),
                            "filter_column": args.filter_column, "filter_value": args.filter_value}
    if args.signs and args.neurotransmitters:
        raise ValueError("Choose exactly one sign source")
    signs = load_signs(Path(args.signs)) if args.signs else None
    if args.neurotransmitters:
        nt_path = Path(args.neurotransmitters)
        nt_hash = sha256_file(nt_path)
        if source_path.suffix.lower() == ".feather" and (
                nt_hash != OFFICIAL_NEUROTRANSMITTERS["sha256"] or
                nt_path.stat().st_size != OFFICIAL_NEUROTRANSMITTERS["bytes"]):
            raise ValueError("Neurotransmitter file does not match official MaleCNS v1.0")
        signs = load_neurotransmitter_signs(nt_path)
        source["neurotransmitters_source"] = {"file": nt_path.name, "sha256": nt_hash,
              "column": "consensus_nt", "sign_policy": "ACh +; GABA/glutamate/histamine -; other/uncertain +"}
    if args.signs:
        source["signs_source"] = {"file": Path(args.signs).name,
                                 "sha256": sha256_file(Path(args.signs))}
    graph = Graph.from_edge_batches(ids, iter_edge_batches(source_path), args.gain_mv,
                                     signs=signs, source=source)
    graph.save(Path(args.output))
    print(json.dumps(graph.manifest, indent=2))


def demo(args):
    graph = Graph.from_edges([101, 102, 201, 202], [
        {"body_pre": 101, "body_post": 201, "weight": 20},
        {"body_pre": 102, "body_post": 202, "weight": 20},
    ], gain_mv=1.5, signs={101: 1, 102: 1})
    params = LIFParameters()
    brain = LIF(graph, params)
    rates = RateReadout(4, params.dt_ms, tau_ms=50)
    trace = []
    for t in range(args.steps):
        drive = np.zeros(4)
        drive[graph.index(101 if t < args.steps // 2 else 102)] = 3.0
        rates.update(brain.step(drive))
        if (t + 1) % 10 == 0:
            trace.append({"tick": t + 1, "left_dn_hz": round(rates.hz[2], 2),
                          "right_dn_hz": round(rates.hz[3], 2),
                          "turn": round(signed_turn(rates, 2, 3, gain=.03), 3)})
    print(json.dumps({"model": "synthetic four-neuron fixture; no fly behavior claim",
                      "graph": graph.manifest, "trace": trace}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="connectome-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="verify, filter and compile MaleCNS graph")
    b.add_argument("--edges", required=True)
    b.add_argument("--ids", required=True, help="CSV/Feather neuron annotations or explicit ID table")
    b.add_argument("--id-column", default="bodyId")
    b.add_argument("--filter-column")
    b.add_argument("--filter-value")
    b.add_argument("--signs", help="optional CSV/Feather with body,sign=(-1|1)")
    b.add_argument("--neurotransmitters", help="official MaleCNS table with body,consensus_nt")
    b.add_argument("--gain-mv", type=float, default=0.275)
    b.add_argument("--output", required=True)
    d = sub.add_parser("demo", help="four-neuron left/right sensorimotor demonstration")
    d.add_argument("--steps", type=int, default=200)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args)
    else:
        demo(args)


if __name__ == "__main__":
    main()
