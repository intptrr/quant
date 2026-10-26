import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]


def desired_weights(current: FloatArray, target: FloatArray, band: FloatArray) -> FloatArray:
    desired = np.clip(current, target - band, target + band)
    return np.where(current * desired < 0.0, 0.0, desired)
