import numpy as np
import warnings

def attribute_filtering(indices, attribute_filter, bucket_obj_indexes):
    """
    Filters the given indices based on the provided attribute filter and bucket object indices mapping.

    Parameters:
    - indices (np.ndarray): A 2D array of indices to be filtered.
    - attribute_filter (np.ndarray): A 2D array containing the filtering attributes corresponding to each row in `indices`.
    - bucket_obj_indexes (list): A mapping from bucket indices to object indices.

    Returns:
    - np.ndarray: A 2D array of filtered indices. Rows that have fewer indices after filtering are padded with -1.

    Notes:
    - The function pads filtered indices with -1. Ensure that this padding does not cause issues in subsequent processing.
    """
    filtered_indices = []

    for i_row, f_row in zip(indices, attribute_filter):
        if i_row.size != bucket_obj_indexes.size:
            warnings.warn("\n WARNING: Ann relative is bigger than the number of objects in the bucket. This usually caused by Threshold optimization!\n")
            i_row = i_row[i_row < bucket_obj_indexes.size]

        # Perform mapping from bucket indices to object indicies
        mapped_values = np.array([bucket_obj_indexes[i] for i in i_row])

        # Performs the filtering
        # TODO use roaring bitmaps for this
        mask = np.isin(mapped_values, f_row, invert=False)

        filtered_row = i_row[mask]

        filtered_indices.append(filtered_row)

    # Constructs the resulting filtered list and pads it with len(bucket_obj_indexes) which maps to non-existing object 0 and distance infinity
    filtered_indices = np.array(
        [np.pad(row, (0, indices.shape[1] - len(row)), 'constant', constant_values=len(bucket_obj_indexes)) for row in filtered_indices],
        dtype=int)

    return filtered_indices
