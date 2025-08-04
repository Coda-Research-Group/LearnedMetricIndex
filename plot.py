from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_recall_vs_candidates(csv_path: Path, title: str = 'Recall vs Candidates') -> None:
    df = pd.read_csv(csv_path)

    # Ensure sorting within each algorithm group
    df = df.sort_values(by=['algo', 'candidates'])

    plt.figure(figsize=(10, 6))

    for algo, group in df.groupby('algo'):
        plt.plot(group['candidates'], group['recall'], label=algo, marker='o')

    plt.xlabel('Number of Candidates')
    plt.ylabel('Recall')
    plt.title(title)
    plt.ylim(bottom=0)
    plt.legend(title='Algorithm')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_recall_vs_candidates(Path("res.csv"), 'Recall vs Candidates - 10 Tasks')