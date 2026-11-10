import numpy as np


def impl(x1, x2):
    orig_dtype = x1.dtype
    is_int = np.issubdtype(orig_dtype, np.integer)
    if is_int:
        a = np.asarray(x1, dtype=np.int64)
        b = np.asarray(x2, dtype=np.int64)
        s = a + b
        s = (s + (1 << 31)) % (1 << 32) - (1 << 31)
        return s.astype(np.int32)
    a = np.asarray(x1, dtype=np.float32)
    b = np.asarray(x2, dtype=np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        y = a + b
    if orig_dtype == np.float16:
        return y.astype(np.float16)
    return y.astype(np.float32)
