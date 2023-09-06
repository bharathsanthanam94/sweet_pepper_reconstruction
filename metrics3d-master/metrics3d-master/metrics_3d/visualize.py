import matplotlib.pyplot as plt


def threshold_plot_curves(curves, thresholds, ylabel, title, legend=None):
    thresholds = thresholds * 1000
    fig, ax = plt.subplots()
    for c in curves:
        ax.plot(thresholds, c)

    if legend:
        ax.legend(legend)

    ax.set_xlim(thresholds[-1], thresholds[0])
    ax.set_ylabel(ylabel)
    ax.set_xlabel('thresholds [mm]')
    ax.set_title(title)
    ax.grid(True)
    plt.show()


def pr_plot_curves(pr, re, xlabel, ylabel, title, legend=None):
    fig, ax = plt.subplots()
    for p, r in zip(pr, re):
        ax.plot(r, p)

    if legend:
        ax.legend(legend)

    ax.set_ylim(0, 100)
    ax.set_xlim(100, 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    plt.show()