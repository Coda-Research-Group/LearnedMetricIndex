import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import os

GRAPHS_FOLDER = "./graphs"

def recall_distance_computations(data: pd.DataFrame, n_buckets: int, nlist: int, nprobe_list: List[int], path: str) -> None:
    filtered_data = data[(data['n_buckets'] == n_buckets) & (data['nlist'] == nlist) & (data['nprobe'].isin(nprobe_list))]
    plt.figure(figsize=(10, 6))
    for nprobe_value, group_data in filtered_data.groupby('nprobe'):
        plt.plot(group_data['recall'], group_data['distance_computations'], label=f'nprobe={nprobe_value}')

    plt.xlabel('Recall')
    plt.ylabel('Distance Computations')
    plt.title('Distance Computations vs. Recall for n_buckets=11, nlist=11')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig(path)
    
def recall_search_time(data: pd.DataFrame, n_buckets: int, nlist: int, nprobe_list: List[int], path: str) -> None:
    filtered_data = data[(data['n_buckets'] == n_buckets) & (data['nlist'] == nlist) & (data['nprobe'].isin(nprobe_list))]
    plt.figure(figsize=(10, 6))
    for nprobe_value, group_data in filtered_data.groupby('nprobe'):
        plt.plot(group_data['recall'], group_data['search'], label=f'nprobe={nprobe_value}')

    plt.xlabel('Recall')
    plt.ylabel('Search time')
    plt.title('Search time vs. Recall for n_buckets=11, nlist=11')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig(path)

if __name__ == '__main__':
    data = pd.read_csv("./test_results/nav=pca32v2search=clip768v2size=100Kbucket=IVFFaissk=10_test.csv")
    recall_search_time(data, 11, 11, [1, 4, 7, 10, 13], os.path.join(GRAPHS_FOLDER, "recall_search_time"))