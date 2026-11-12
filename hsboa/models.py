"""QRNN classifier for the d selected features.

Architecture (one model, two implementations with identical weights):

    x in R^d  ->  sequence of d steps with 1 channel
              ->  QRNN layer, fo-pooling (Bradbury et al. 2017):
                    [Z, F, O] = causal Conv1D(x, window k, 3u filters) + b
                    z = tanh(Z), f = sigmoid(F), o = sigmoid(O)
                    c_t = f_t * c_{t-1} + (1 - f_t) * z_t,   c_0 = 0
                    h_t = o_t * c_t
              ->  pooling: last state h_d, or additive attention
                    s_t = v . tanh(W_a h_t + b_a),  alpha = softmax(s),  h = sum alpha_t h_t
              ->  (dropout, training only) -> Dense(5) -> softmax

The cell state c_t follows the
published fo-pooling recurrence. Gate order in the 3u channels is [z, f, o].

* :class:`QRNNSpec` fixes the shapes and the flat parameter layout.
* :func:`forward` is a numpy forward pass for one flat weight vector, used
  for swarm training (weight-space fitness) and to check the
  Keras model.
* :func:`build_keras_model` builds the same model in Keras for gradient
  training; :func:`set_flat_weights` / :func:`get_flat_weights`
  move weights between the two.

Keras/TensorFlow is imported lazily, so the numpy path works without it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from hsboa.data import CLASSES

N_CLASSES = len(CLASSES)  # UNSW-NB15: 10
POOLING = ("last", "attention")


@dataclass(frozen=True)
class QRNNSpec:
    d: int                          # sequence length = number of selected features
    units: int = 16                 # u
    window: int = 2                 # k, causal convolution width
    pooling: str = "last"           # last | attention
    attention_units: int = 16       # a (attention only)
    n_classes: int = N_CLASSES

    def __post_init__(self):
        if self.pooling not in POOLING:
            raise ValueError(f"pooling must be one of {POOLING}")
        if min(self.d, self.units, self.window, self.attention_units, self.n_classes) < 1:
            raise ValueError("all sizes must be >= 1")

    def shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        """Named parameter shapes in flat-vector order."""
        u, a = self.units, self.attention_units
        s = [("conv_kernel", (self.window, 1, 3 * u)), ("conv_bias", (3 * u,))]
        if self.pooling == "attention":
            s += [("att_kernel", (u, a)), ("att_bias", (a,)), ("att_v", (a, 1))]
        s += [("dense_kernel", (u, self.n_classes)), ("dense_bias", (self.n_classes,))]
        return s

    @property
    def n_params(self) -> int:
        return int(sum(np.prod(sh) for _, sh in self.shapes()))

    def unpack(self, w: np.ndarray) -> dict[str, np.ndarray]:
        w = np.asarray(w)
        if w.shape != (self.n_params,):
            raise ValueError(f"expected {self.n_params} parameters, got {w.shape}")
        out, i = {}, 0
        for name, sh in self.shapes():
            n = int(np.prod(sh))
            out[name] = w[i:i + n].reshape(sh)
            i += n
        return out

    def pack(self, params: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([np.asarray(params[n], dtype=np.float64).ravel()
                               for n, _ in self.shapes()])

    def to_dict(self) -> dict:
        return asdict(self)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))         # overflow-free


def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def forward(spec: QRNNSpec, w: np.ndarray, X: np.ndarray, return_hidden: bool = False,
            dtype=np.float32):
    """Class probabilities (n, n_classes) for inputs X (n, d) and flat weights w.

    float32 by default (as Keras computes); pass ``dtype=np.float64`` for checks.
    """
    p = {k: v.astype(dtype) for k, v in spec.unpack(w).items()}
    X = np.asarray(X, dtype=dtype)
    n, d = X.shape
    if d != spec.d:
        raise ValueError(f"expected {spec.d} features, got {d}")
    k, u = spec.window, spec.units
    Xp = np.concatenate([np.zeros((n, k - 1), dtype=dtype), X], axis=1)               # causal left padding
    K = p["conv_kernel"][:, 0, :]                                         # (k, 3u)
    G = sum(Xp[:, j:j + d, None] * K[j] for j in range(k)) + p["conv_bias"]   # (n, d, 3u)
    z, f, o = np.tanh(G[..., :u]), _sigmoid(G[..., u:2 * u]), _sigmoid(G[..., 2 * u:])
    c = np.zeros((n, u), dtype=dtype)
    H = np.empty((n, d, u), dtype=dtype)
    for t in range(d):
        c = f[:, t] * c + (1.0 - f[:, t]) * z[:, t]
        H[:, t] = o[:, t] * c
    if spec.pooling == "last":
        h = H[:, -1]
    else:
        s = (np.tanh(H @ p["att_kernel"] + p["att_bias"]) @ p["att_v"])[..., 0]   # (n, d)
        h = (_softmax(s, axis=1)[..., None] * H).sum(axis=1)
    proba = _softmax(h @ p["dense_kernel"] + p["dense_bias"])
    return (proba, H) if return_hidden else proba


def init_weights(spec: QRNNSpec, rng) -> np.ndarray:
    """Glorot-uniform kernels and zero biases, as Keras initialises the model."""
    rng = np.random.default_rng(rng)
    parts = {}
    for name, sh in spec.shapes():
        if name.endswith("bias"):
            parts[name] = np.zeros(sh)
        else:
            fan_in, fan_out = (np.prod(sh[:-1]), sh[-1])
            if name == "conv_kernel":
                fan_in, fan_out = sh[0] * sh[1], sh[0] * sh[2]
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            parts[name] = rng.uniform(-lim, lim, sh)
    return spec.pack(parts)


# ---------------------------------------------------------------------------
# Keras implementation
# ---------------------------------------------------------------------------

def _keras():
    import keras
    return keras


def _layers():
    """Define the Keras layers on first use (keeps TensorFlow out of numpy-only imports)."""
    keras = _keras()
    ops = keras.ops

    @keras.saving.register_keras_serializable("hsboa")
    class QRNN(keras.layers.Layer):
        """fo-pooling QRNN layer; returns the full hidden sequence (batch, d, units)."""

        def __init__(self, units, window=2, kernel_regularizer=None, **kw):
            super().__init__(**kw)
            self.units, self.window = units, window
            self.conv = keras.layers.Conv1D(3 * units, window, padding="causal", name="conv",
                                            kernel_regularizer=kernel_regularizer)

        def build(self, input_shape):
            self.conv.build(input_shape)
            self.built = True

        def call(self, x):
            g = self.conv(x)
            u = self.units
            z, f, o = ops.tanh(g[..., :u]), ops.sigmoid(g[..., u:2 * u]), ops.sigmoid(g[..., 2 * u:])
            c = ops.zeros_like(z[:, 0])
            hs = []
            for t in range(x.shape[1]):                  # d is small and static: unrolled loop
                c = f[:, t] * c + (1.0 - f[:, t]) * z[:, t]
                hs.append(o[:, t] * c)
            return ops.stack(hs, axis=1)

        def get_config(self):
            return {**super().get_config(), "units": self.units, "window": self.window}

    @keras.saving.register_keras_serializable("hsboa")
    class AttentionPooling(keras.layers.Layer):
        def __init__(self, units, **kw):
            super().__init__(**kw)
            self.units = units
            self.score = keras.layers.Dense(units, activation="tanh", name="att")
            self.v = keras.layers.Dense(1, use_bias=False, name="att_v")

        def build(self, input_shape):
            self.score.build(input_shape)
            self.v.build((*input_shape[:-1], self.units))
            self.built = True

        def call(self, H):
            alpha = ops.softmax(self.v(self.score(H))[..., 0], axis=1)
            return ops.sum(alpha[..., None] * H, axis=1)

        def get_config(self):
            return {**super().get_config(), "units": self.units}

    @keras.saving.register_keras_serializable("hsboa")
    class LastStep(keras.layers.Layer):
        def call(self, H):
            return H[:, -1]

    return QRNN, AttentionPooling, LastStep


def build_keras_model(spec: QRNNSpec, dropout: float = 0.0, l2: float = 0.0):
    """The QRNN classifier as a Keras model (input (batch, d), output softmax over classes)."""
    keras = _keras()
    QRNN, AttentionPooling, LastStep = _layers()
    reg = keras.regularizers.L2(l2) if l2 > 0 else None
    inp = keras.Input((spec.d,), name="features")
    x = keras.layers.Reshape((spec.d, 1), name="sequence")(inp)
    H = QRNN(spec.units, spec.window, kernel_regularizer=reg, name="qrnn")(x)
    if spec.pooling == "attention":
        h = AttentionPooling(spec.attention_units, name="attention")(H)
    else:
        h = LastStep(name="last_state")(H)
    if dropout > 0:
        h = keras.layers.Dropout(dropout, name="dropout")(h)
    out = keras.layers.Dense(spec.n_classes, activation="softmax", name="dense",
                             kernel_regularizer=reg)(h)
    return keras.Model(inp, out, name="qrnn_classifier")


def get_flat_weights(spec: QRNNSpec, model) -> np.ndarray:
    q = model.get_layer("qrnn").conv
    p = {"conv_kernel": q.kernel.numpy(), "conv_bias": q.bias.numpy()}
    if spec.pooling == "attention":
        a = model.get_layer("attention")
        p.update(att_kernel=a.score.kernel.numpy(), att_bias=a.score.bias.numpy(),
                 att_v=a.v.kernel.numpy())
    dense = model.get_layer("dense")
    p.update(dense_kernel=dense.kernel.numpy(), dense_bias=dense.bias.numpy())
    return spec.pack(p)


def set_flat_weights(spec: QRNNSpec, model, w: np.ndarray) -> None:
    p = spec.unpack(w)
    model.get_layer("qrnn").conv.set_weights([p["conv_kernel"], p["conv_bias"]])
    if spec.pooling == "attention":
        a = model.get_layer("attention")
        a.score.set_weights([p["att_kernel"], p["att_bias"]])
        a.v.set_weights([p["att_v"]])
    model.get_layer("dense").set_weights([p["dense_kernel"], p["dense_bias"]])
