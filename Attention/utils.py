import torch
import matplotlib.pyplot as plt


def show_heatmaps(matrices, xlabel, ylabel, title=None, figsize=(2.5, 2.5), cmap='red'):
    num_rows, num_cols = matrices[0], matrices[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        pass
