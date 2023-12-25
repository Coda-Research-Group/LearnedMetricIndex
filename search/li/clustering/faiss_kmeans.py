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
        # TODO: added "nredo" to make KMeans more reliable, parametrize this in LVD if it cause too long index build time
        parameters = {"verbose": False, "seed": 2023, "nredo": 10}

    _, d = data.shape

    print('FAISS Kmeans parameters', parameters)
    kmeans = Kmeans(d=d, k=n_clusters, **parameters)
    kmeans.train(data)

    labels = kmeans.index.search(data, 1)[1].T[0]  # type: ignore

    return kmeans, labels.astype(np.int32)
