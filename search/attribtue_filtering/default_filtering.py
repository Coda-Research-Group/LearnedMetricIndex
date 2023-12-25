import numpy as np
import warnings
from itertools import product

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

def precompute_bucket_ids(n_categories):
    """
    Generates all possible bucket IDs given the number of categories at each level of a tree.

    Parameters:
    - n_categories (List[int]): A list of integers where each element represents the number of categories at each level of the tree.

    Returns:
    - np.ndarray: A 2D array where each row is a unique bucket ID represented as a combination of category indices.

    Notes:
    - The function uses a Cartesian product to generate combinations, assuming the tree is perfectly balanced.
    """

    # Generate all possible combinations of categories for each level
    all_combinations = list(product(*[range(n) for n in n_categories]))
    # Convert the combinations to a NumPy array
    bucket_ids = np.array(all_combinations, dtype=np.int32)
    return bucket_ids


def compute_ratios_for_attribute_filters(data_prediction, attribute_filters, n_categories):
    """
    Computes ratios for attribute filters based on the data predictions and the precomputed bucket IDs.

    Parameters:
    - data_prediction (np.ndarray): A 2D array where each row represents an object's bucket assignment in the tree.
    - attribute_filters (np.ndarray): A 2D array of indices indicating which objects to consider for each filter.
    - n_categories (List[int]): A list representing the number of categories at each level of the tree.

    Returns:
    - List[Dict[Tuple[int], float]]: A list of dictionaries, where each dictionary maps bucket IDs (as tuples) to the ratio of objects satisfying the constraint for that bucket.

    Notes:
    - Ensure that `attribute_filters` indices are 1-indexed and are adjusted within the function for 0-based indexing of Python.
    """

    # Precompute all possible bucket IDs
    all_bucket_ids = precompute_bucket_ids(n_categories)

    # Initialize the list to hold NumPy arrays for each attribute filter's ratios
    filters_ratios_list = []

    # Adjust for zero-based indexing in attribute_filter
    shifted_attribute_filters = attribute_filters - 1

    # Iterate through each attribute filter
    for filter_array in shifted_attribute_filters:
        # Get the data for the current filter
        filter_data = data_prediction[filter_array.flatten()]

        # Count occurrences in each bucket for the current filter
        # Create a 2D array of shape (len(filter_data), len(all_bucket_ids)) where each element is True if the
        # filter_data matches the corresponding bucket_id, else False
        matches = np.all(filter_data[:, None] == all_bucket_ids, axis=2)

        # Sum the matches along the first axis to count occurrences in each bucket
        counts = matches.sum(axis=0)

        # Calculate the ratio for each bucket
        total_elements = len(filter_data)
        ratios = counts / total_elements

        # Convert the ratios to a dictionary mapping bucket_id to ratio
        filter_ratios_dict = {tuple(bucket_id): ratio for bucket_id, ratio in zip(all_bucket_ids, ratios)}

        # Add the dictionary to the list
        filters_ratios_list.append(filter_ratios_dict)

    return filters_ratios_list

def combine_probabilities(filters_ratios):
    """
    Combines probabilities of buckets to form new bucket probabilities one level up by replacing the last non-negative item with -1.

    Parameters:
    - filters_ratios (List[Dict[Tuple[int], float]]): A list of dictionaries, where each dictionary maps bucket IDs to the ratio of objects satisfying the constraint.

    Returns:
    - List[Dict[Tuple[int], float]]: A list of dictionaries with updated bucket IDs and combined probabilities.

    Notes:
    - This function is designed to work with tuples of arbitrary length and combines probabilities only one level below.
    - The function maintains the original ratios while adding new combined ratios for the immediate upper level.
    """

    combined_filters_ratios = []

    for filter_ratio in filters_ratios:
        combined_ratio = filter_ratio.copy()  # Copy the original ratios
        temp_dict = {}  # Temporary dictionary for the new combined probabilities

        for bucket_tuple, probability in filter_ratio.items():
            # Create and aggregate probabilities for all modified tuples
            for idx in range(len(bucket_tuple)):
                if bucket_tuple[idx] != -1:  # Identifying a non-negative element
                    modified_tuple = bucket_tuple[:idx] + (-1,) * (len(bucket_tuple) - idx)
                    temp_dict[modified_tuple] = temp_dict.get(modified_tuple, 0) + probability

        # Update the combined ratio dictionary with new tuples and their probabilities
        for new_tuple, new_probability in temp_dict.items():
            combined_ratio[new_tuple] = new_probability

        combined_filters_ratios.append(combined_ratio)  # Append the updated dictionary

    return combined_filters_ratios