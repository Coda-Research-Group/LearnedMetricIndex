from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans


def cluster(
    data: npt.NDArray[np.float32],
    n_clusters: int,
    parameters: Optional[Dict[str, Any]],
) -> Tuple[KMeans, npt.NDArray[np.int32]]:
    if parameters is None:
        parameters = {
            "verbose": 0,
            "random_state": 2023,
            # The same default values as in faiss.Kmeans
            "init": "random",
            "max_iter": 25,
            "n_init": 1,
            #
        }

    kmeans = KMeans(n_clusters=n_clusters, **parameters)
    kmeans.fit(data)

    labels = kmeans.labels_

    return kmeans, labels
