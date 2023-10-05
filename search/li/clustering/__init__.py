from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

from .faiss_kmeans import cluster as faiss_kmeans
from .scikit_kmeans import cluster as scikit_kmeans

ClusteringAlgorithm = Callable[
    [npt.NDArray[np.float32], int, Optional[Dict[str, Any]]],
    Tuple[Any, npt.NDArray[np.int32]],
]

algorithms: Dict[str, ClusteringAlgorithm] = {
    "faiss_kmeans": faiss_kmeans,
    "scikit_kmeans": scikit_kmeans,
}
