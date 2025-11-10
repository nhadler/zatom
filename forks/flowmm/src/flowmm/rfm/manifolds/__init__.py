"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from forks.flowmm.src.flowmm.rfm.manifolds.analog_bits import MultiAtomAnalogBits
from forks.flowmm.src.flowmm.rfm.manifolds.euclidean import EuclideanWithLogProb
from forks.flowmm.src.flowmm.rfm.manifolds.flat_torus import (
    FlatTorus01FixFirstAtomToOrigin,
    FlatTorus01FixFirstAtomToOriginWrappedNormal,
)
from forks.flowmm.src.flowmm.rfm.manifolds.null import NullManifoldWithDeltaRandom
from forks.flowmm.src.flowmm.rfm.manifolds.product import ProductManifoldWithLogProb
from forks.flowmm.src.flowmm.rfm.manifolds.simplex import (
    FlatDirichletSimplex,
    MultiAtomFlatDirichletSimplex,
)
