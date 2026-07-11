"""Smoke tests for the THRML-flavored Python bindings.

Build the module and make it importable, e.g.:

    cargo build --release -p extropic-sim-py
    cp target/release/libextropic_sim_py.so extropic_sim.so

then run with pytest or plain python:

    python3 py/tests/test_compat.py
"""

import math

from extropic_sim import (
    Block,
    IsingEBM,
    IsingSamplingProgram,
    SamplingSchedule,
    SpinNode,
    estimate_moments,
    hinton_init,
    sample_states,
)


def make_chain(n=5, weight=0.5):
    nodes = [SpinNode() for _ in range(n)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
    biases = [0.0] * n
    weights = [weight] * (n - 1)
    model = IsingEBM(nodes, edges, biases, weights, 1.0)
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, [])
    return nodes, model, free_blocks, program


def test_readme_example():
    """The THRML README example, transcribed."""
    nodes, model, free_blocks, program = make_chain()
    schedule = SamplingSchedule(100, 1000, 2)

    init_state = hinton_init(0, model, free_blocks)
    samples = sample_states(1, program, schedule, init_state, [], [Block(nodes)])

    assert len(samples) == 1  # one observed block
    assert len(samples[0]) == 1000  # n_samples
    assert len(samples[0][0]) == len(nodes)  # unbatched
    assert all(value in (0, 1) for value in samples[0][0])


def test_batched_sampling():
    nodes, model, free_blocks, program = make_chain()
    schedule = SamplingSchedule(50, 100, 2)

    init_state = hinton_init(0, model, free_blocks, n_chains=8)
    samples = sample_states(1, program, schedule, init_state, [], [Block(nodes)])

    assert len(samples[0]) == 100
    assert len(samples[0][0]) == 8  # chains
    assert len(samples[0][0][0]) == len(nodes)


def test_moments_match_exact():
    """First moment of a two-spin ferromagnet with a field, vs enumeration."""
    a, b = SpinNode(), SpinNode()
    bias, weight = 0.4, 0.6
    model = IsingEBM([a, b], [(a, b)], [bias, 0.0], [weight], 1.0)
    program = IsingSamplingProgram(model, [Block([a]), Block([b])], [])
    schedule = SamplingSchedule(200, 4000, 2)

    init = hinton_init(2, model, [Block([a]), Block([b])], n_chains=32)
    moments = estimate_moments(3, program, schedule, init, [], [[a], [a, b]])

    # Exact enumeration over the four spin states.
    z = up_a = corr = 0.0
    for sa in (-1, 1):
        for sb in (-1, 1):
            p = math.exp(bias * sa + weight * sa * sb)
            z += p
            up_a += p * sa
            corr += p * sa * sb
    assert abs(moments[0] - up_a / z) < 0.03
    assert abs(moments[1] - corr / z) < 0.03


def test_clamping():
    nodes, model, _, _ = make_chain(4, weight=1.0)
    free_blocks = [Block([nodes[1]]), Block([nodes[2]])]
    clamped = [Block([nodes[0], nodes[3]])]
    program = IsingSamplingProgram(model, free_blocks, clamped)
    schedule = SamplingSchedule(20, 50, 1)

    samples = sample_states(
        4, program, schedule, [[0], [1]], [[1, 0]], [Block(nodes)], device="cpu"
    )
    for frame in samples[0]:
        assert frame[0] == 1 and frame[3] == 0


if __name__ == "__main__":
    test_readme_example()
    test_batched_sampling()
    test_moments_match_exact()
    test_clamping()
    print("All Python binding tests passed")
