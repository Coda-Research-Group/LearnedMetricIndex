import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

sns.set(font_scale=1.3, style='whitegrid')

# Load data
results = pd.read_csv('res.csv')

# Extract parameters from each df
parameters = {
    't1': r't1-100M-epochs=(?P<epochs>.*)-lr=(?P<lr>.*)-sample=(?P<sample_size>.*)-alpha=(?P<alpha>.*)-chunk_size=(?P<chunk_size>.*)-nprobe=(?P<nprobe>.*)',  # noqa: E501
    't2': r't2-100M-epochs=(?P<epochs>.*)-lr=(?P<lr>.*)-sample=(?P<sample_size>.*)-alpha=(?P<alpha>.*)-chunk_size=(?P<chunk_size>.*)-ncandidates=(?P<ncandidates>.*)-reduced_dim=(?P<reduced_dim>.*)-nprobe=(?P<nprobe>.*)',  # noqa: E501
    't3': r't3-100M-epochs=(?P<epochs>.*)-lr=(?P<lr>.*)-sample=(?P<sample_size>.*)-alpha=(?P<alpha>.*)-chunk_size=(?P<chunk_size>.*)-reduced_dim=(?P<reduced_dim>.*)-nprobe=(?P<nprobe>.*)',  # noqa: E501
}

tasks = {task: results[results.params.str.startswith(task)] for task in ['t1', 't2', 't3']}
parsed = {task: pd.concat([df, df['params'].str.extract(parameters[task], expand=True)], axis=1) for task, df in tasks.items()}

# Convert nprobe to integer and recall to float
for df in parsed.values():
    df['nprobe'] = df['nprobe'].astype(int)
    df['recall'] = df['recall'].astype(float)

# Filter out data points beyond 30 nprobe in Task 2
parsed['t2'] = parsed['t2'][parsed['t2'].nprobe <= 30]  # noqa: PLR2004

# Add task column to each df and concatenate into combined_df
combined_df = pd.concat(
    [
        parsed['t1'].assign(task='Task 1'),
        parsed['t2'].assign(task='Task 2'),
        parsed['t3'].assign(task='Task 3'),
    ],
)

# Swap colors for Task 2 and 3
color_palette = sns.color_palette()[:3]
color_palette[1], color_palette[2] = color_palette[2], color_palette[1]


def plot_curve(df: pd.DataFrame, x: str, y: str, hue: str) -> Axes:
    """Plot a line chart with the specified parameters.

    Args:
    ----
        df (pd.DataFrame): The data to plot.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        hue (str): The column name for the hue (color).

    Returns:
    -------
        plt.Axes: The axis object of the plotted figure.

    """
    return sns.lineplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        markers=True,
        palette=color_palette,
        style=hue,
        dashes=False,
    )


def plot_recall() -> None:
    """Plot a line chart with recall against the number of visited buckets."""
    try:
        plot = plot_curve(combined_df, 'nprobe', 'recall', 'task')
        plot.set(xlabel='Number of visited buckets', ylabel='Average recall')
        plot.get_legend().set_title('')
        plot.axhline(0.4, color='black', linestyle=':')
        plot.axhline(0.8, color='black', linestyle=':')
        plt.ylim([None, 1])
        plt.savefig('nprobe-recall.pdf')
        plt.show()
    except Exception as e:  # noqa: BLE001
        print(f'An error occurred: {e}')


def plot_querytime() -> None:
    """Plot a line chart with query time against the number of visited buckets."""
    try:
        plot = plot_curve(combined_df, 'nprobe', 'querytime', 'task')
        plot.set(xlabel='Number of visited buckets', ylabel='Search time (s) [log]')
        plot.get_legend().set_title('')
        plt.yscale('log')
        plt.ylim([None, 10_000])
        plt.savefig('nprobe-querytime.pdf')
        plt.show()
    except Exception as e:  # noqa: BLE001
        print(f'An error occurred: {e}')


plot_recall()
plot_querytime()
