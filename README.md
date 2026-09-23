# Connectome Sensorimotor Bench — phase 1 CPU reference

This is a runnable **engineering prototype**, not a biologically validated fly. It implements a reproducible MaleCNS v1.0 edge import, explicit signed weights, deterministic CPU LIF, synthetic sensor → neural state → rate → action example, and small numerical checks. No GPU backend, visual physiology, trained readout, or real-body/game integration is claimed yet.

## Quick start

Python 3.10+ with NumPy and SciPy:

```bash
python -m pip install -e .
connectome-bench demo --steps 200
python -m unittest discover -s tests -v
```

The demo uses **four invented neurons** (left/right sensors and left/right output proxies). At tick 50 it should show a negative turn, and by tick 200 a positive turn. This is an interface and integration check, not MaleCNS data.

## Compile MaleCNS data (after obtaining the official files)

Download the three Feather files linked on the [official MaleCNS page](https://male-cns.janelia.org/download/): `connectome-weights-male-cns-v1.0-minconf-0.5.feather`, `body-annotations-male-cns-v1.0-minconf-0.5.feather`, and `body-neurotransmitters-male-cns-v1.0.feather`. They are **not bundled**. Inspect the official annotations and explicitly export your desired neuronal body IDs as a single-column `curated-neuron-ids.csv` with header `bodyId`. This command reads only edges whose endpoints are among those IDs:

```bash
python -m pip install -e '.[feather]'
connectome-bench build \
  --edges data/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  --ids data/curated-neuron-ids.csv \
  --id-column bodyId \
  --neurotransmitters data/body-neurotransmitters-male-cns-v1.0.feather \
  --gain-mv 0.275 --output out/malecns
```

You can instead pass the official annotation Feather as `--ids` with `--filter-column FIELD --filter-value VALUE` after inspecting its actual schema and values; this prototype does not prescribe a biological cell filter. The importer will reject an official Feather edges file whose byte count or SHA-256 does not match the pinned v1.0 lock, and checks the neurotransmitter file similarly. The ID-selection file is hashed and recorded but not pinned to one official hash. Graph outputs include `ids.npy`, `weights.npz`, and `manifest.json` with output digests and selection policy. `Graph.load(path)` rechecks the digests.

**Memory limit:** Although reading Feather is batched, this reference builder accumulates *retained* edges in NumPy arrays, then SciPy sorts them into CSR. A 25.6-million-edge retained graph can require several GB of free RAM and build time; a 150-million-edge selection is not supported as a low-memory workflow. Begin with a small ID CSV. The LIF reference currently uses Python-level event loops and is intended for correctness on small graphs; building a full graph does **not** imply a real-time whole-CNS simulation. GPU and production full-graph paths are later milestones.

For a smoke graph without Arrow, prepare `ids.csv` and `edges.csv`:

```text
# ids.csv
bodyId
101
102
201
202

# edges.csv
body_pre,body_post,weight
101,201,20
102,202,20
```

Run `connectome-bench build --edges edges.csv --ids ids.csv --gain-mv 1.5 --output out/smoke`. CSV without `--signs` uses the **explicit positive fallback** and counts missing sign annotations in the manifest. `--signs` accepts `body,sign` with ±1; official neurotransmitters use `body,consensus_nt`. The coarse ACh(+), GABA/glutamate/histamine(-), other(+) rule is a model assumption and is not a receptor-resolved physiological model.

## Model convention and independent work still required

The connection matrix is `W[postsynaptic_index, presynaptic_index]`. Synaptic contacts become `n * gain_mv * sign` mV-equivalent increments. One step decays `g` by `exp(-dt/tau_syn)`, adds scheduled arrivals, then integrates `V` using `exp(-dt/tau_mem)` with sustained drive and current `g`. A spike resets `V,g` and schedules outgoing events at an **integer** delay. Inputs arriving during refractory time are discarded. State persists between `step` calls; `snapshot/restore` reproduces the continuation. This is an explicit reference convention; it is not asserted to exactly replicate Shiu/Brian2 or DOOMFLY, whose operation order and update rules must be compared separately before scientific parity claims.

Current tests cover source/destination orientation, duplicate edge aggregation, endpoint filtering, inhibitory sign, counts, persisted hash verification, hand-calculated delay/voltage, refractory arrival discard, snapshot continuation, and decoder sign. They do not prove whole-connectome parity or biological behavior. Planned next steps: import an official small curated subset; compare with a separately specified Brian2 oracle; validate graded visual frontend on recorded responses; profile/implement fast CPU and GPU backends; run black-screen, shuffled-image, degree-preserving rewiring, and silencing controls.

## Sources

- [MaleCNS official dataset](https://male-cns.janelia.org/download/)
- [DOOMFLY official-data source lock](https://github.com/nftechie/doomfly/blob/main/data-provenance/malecns_v1/source.lock.json) (three file sizes and SHA-256 hashes)
- [Published Shiu et al. LIF reference code](https://github.com/philshiu/Drosophila_brain_model)
- [Visual graded model flyvis](https://github.com/TuragaLab/flyvis)

License for this new prototype: MIT (`LICENSE`). Dataset redistribution and attribution remain governed by the source dataset terms.
# chopari
