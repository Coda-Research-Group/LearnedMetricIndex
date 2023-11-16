from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from faiss import Kmeans


def cluster(
    data: npt.NDArray[np.float32],
    n_clusters: int,
    parameters: Optional[Dict[str, Any]],
) -> Tuple[Kmeans, npt.NDArray[np.int32]]:
    if parameters is None:
        parameters = {"verbose": False, "seed": 2023}

    _, d = data.shape

    kmeans = Kmeans(d=d, k=n_clusters, **parameters)
    data = np.ascontiguousarray(data.astype(np.float32))
    kmeans.train(data)

    labels = kmeans.index.search(data, 1)[1].T[0]  # type: ignore

    return kmeans, labels.astype(np.int32)
