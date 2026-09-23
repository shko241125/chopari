import tempfile
import unittest
from pathlib import Path

import numpy as np

from connectome_bench.graph import (Graph, iter_edge_batches, load_ids, load_signs,
                                    load_neurotransmitter_signs)
from connectome_bench.lif import LIF, LIFParameters, RateReadout, signed_turn


class CoreTests(unittest.TestCase):
    def test_import_direction_duplicate_filter_sign_and_integrity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ids.csv").write_text("bodyId,superclass\n20,Neuron\n10,Neuron\n99,Glia\n")
            (root / "signs.csv").write_text("body,sign\n20,-1\n10,1\n")
            (root / "edges.csv").write_text("body_pre,body_post,weight\n20,10,2\n20,10,3\n10,20,4\n99,20,8\n")
            ids = load_ids(root / "ids.csv", filter_column="superclass", filter_value="Neuron")
            graph = Graph.from_edge_batches(ids, iter_edge_batches(root / "edges.csv"), .5,
                                             load_signs(root / "signs.csv"))
            self.assertEqual(ids.tolist(), [10, 20])
            self.assertAlmostEqual(graph.weights[0, 1], -2.5)  # post10 <- pre20
            self.assertAlmostEqual(graph.weights[1, 0], 2.0)   # post20 <- pre10
            self.assertEqual(graph.manifest["retained_contacts"], 9)
            self.assertEqual(graph.manifest["aggregated_edge_rows"], 2)
            graph.save(root / "graph")
            self.assertEqual(Graph.load(root / "graph").weights[0, 1], -2.5)
            with (root / "graph" / "ids.npy").open("ab") as output:
                output.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                Graph.load(root / "graph")

    def test_delay_refractory_and_numeric_voltage(self):
        graph = Graph.from_edges([10, 20], [{"body_pre": 10, "body_post": 20, "weight": 1}], 1)
        p = LIFParameters(dt_ms=1, tau_mem_ms=1, tau_syn_ms=100,
                          threshold_mv=1, refractory_ms=2, delay_ms=2)
        sim = LIF(graph, p)
        self.assertTrue(sim.step([4, 0])[0])  # source fires at t=0
        self.assertFalse(sim.step([0, 0]).any())
        self.assertFalse(sim.step([0, 0]).any())  # event reaches target at t=2
        self.assertAlmostEqual(sim.v[1], 1 - np.exp(-1))  # hand-calculated arrival
        self.assertTrue(sim.step([4, 0])[0])  # eligible again at t=3

    def test_refractory_discards_arrivals_and_snapshot(self):
        graph = Graph.from_edges([1], [{"body_pre": 1, "body_post": 1, "weight": 1}], 100)
        p = LIFParameters(dt_ms=1, tau_mem_ms=1, refractory_ms=3, delay_ms=1)
        sim = LIF(graph, p)
        self.assertTrue(sim.step([4])[0])
        self.assertEqual(sim.ring.sum(), 100)
        self.assertFalse(sim.step([0])[0])
        self.assertEqual(sim.ring.sum(), 0)
        self.assertEqual(sim.g[0], 0)
        saved = sim.snapshot()
        a = [bool(sim.step([4])[0]) for _ in range(8)]
        sim.restore(saved)
        b = [bool(sim.step([4])[0]) for _ in range(8)]
        self.assertEqual(a, b)

    def test_rate_readout_does_not_receive_non_neural_game_state(self):
        rates = RateReadout(2, 1, 10)
        rates.update([0, 1])
        self.assertGreater(signed_turn(rates, 0, 1), 0)
        with self.assertRaises(ValueError):
            rates.update([1, 0, 0])

    def test_neurotransmitter_policy_and_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nt.csv"
            path.write_text("body,consensus_nt\n1,acetylcholine\n2,gaba\n3,unclear\n")
            self.assertEqual(load_neurotransmitter_signs(path), {1: 1, 2: -1, 3: 1})
            path.write_text("body,consensus_nt\n2,gaba\n2,acetylcholine\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_neurotransmitter_signs(path)

    def test_simultaneous_excitation_inhibition_cancel_before_threshold(self):
        graph = Graph.from_edges([1, 2, 3], [
            {"body_pre": 1, "body_post": 3, "weight": 1},
            {"body_pre": 2, "body_post": 3, "weight": 1},
        ], 4.0, signs={1: 1, 2: -1})
        sim = LIF(graph, LIFParameters(dt_ms=1, tau_mem_ms=1, delay_ms=1))
        self.assertEqual(sim.step([4, 4, 0]).tolist(), [True, True, False])
        self.assertAlmostEqual(sim.ring.sum(), 0)  # two signed arrivals cancel
        self.assertEqual(sim.step([0, 0, 0]).tolist(), [False, False, False])
        self.assertAlmostEqual(sim.v[2], 0)

    def test_feather_record_batches_and_annotation_selection(self):
        try:
            import pyarrow as pa
            import pyarrow.feather as feather
        except ImportError:
            self.skipTest("Optional pyarrow is unavailable")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            feather.write_feather(pa.table({"bodyId": [9, 5, 7],
                                            "class": ["neuron", "neuron", "glia"]}), root / "ids.feather")
            feather.write_feather(pa.table({"body_pre": [5, 9], "body_post": [9, 7],
                                            "weight": [4, 8]}), root / "edges.feather")
            ids = load_ids(root / "ids.feather", filter_column="class", filter_value="neuron")
            graph = Graph.from_edge_batches(ids, iter_edge_batches(root / "edges.feather", 1), .25)
            self.assertEqual(graph.manifest["input_edge_rows"], 2)
            self.assertEqual(graph.manifest["retained_edge_rows"], 1)
            self.assertEqual(graph.weights[graph.index(9), graph.index(5)], 1.0)


if __name__ == "__main__":
    unittest.main()
