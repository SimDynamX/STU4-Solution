import numpy as np
import spaceteams as st


class _Ctx:
    pass


ctx = _Ctx()


def init(**kwargs):
    """Called once from Controller.py with the objects created in the protected boilerplate."""
    for key, value in kwargs.items():
        setattr(ctx, key, value)


def phobos_to_body(vec) -> np.ndarray:
    """Re-express a vector given in Phobos-frame axes in the shuttle body frame."""
    return np.asarray(
        st.frames.FramedVector(vec, ctx.phobosFrame).ExprIn(ctx.shuttleFrame), dtype=float
    )
