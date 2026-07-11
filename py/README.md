# extropic-sim Python bindings

Python bindings for [extropic-sim](..), in the flavor of
[THRML](https://github.com/extropic-ai/thrml): build probabilistic
graphical models out of nodes and blocks, and sample them with GPU
block Gibbs sampling.

```python
from extropic_sim import (
    Block, IsingEBM, IsingSamplingProgram, SamplingSchedule,
    SpinNode, hinton_init, sample_states,
)

nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i + 1]) for i in range(4)]
model = IsingEBM(nodes, edges, [0.0] * 5, [0.5] * 4, 1.0)

free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, [])
schedule = SamplingSchedule(100, 1000, 2)

init_state = hinton_init(0, model, free_blocks)
samples = sample_states(1, program, schedule, init_state, [], [Block(nodes)])
```

Divergences from THRML:

- states are plain nested Python lists instead of JAX arrays;
- the PRNG key is an integer seed;
- batching (parallel chains) is controlled by the shape of the initial
  state — pass `hinton_init(..., n_chains=64)` to run 64 chains — rather
  than by `vmap`;
- `CategoricalNode(num_states)` carries its state count directly;
- sampling takes an optional `device` argument: `"auto"` (default),
  `"gpu"`, or `"cpu"`.

## Building

With [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop --release -m py/Cargo.toml
```

Or copy the raw cdylib next to your script:

```bash
cargo build --release -p extropic-sim-py
cp target/release/libextropic_sim_py.so extropic_sim.so  # .dylib on macOS
```

## Testing

```bash
python3 py/tests/test_compat.py
```
