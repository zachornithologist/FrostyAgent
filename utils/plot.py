import matplotlib.pyplot as plt

def plot_results(data, title="Training Results", xlabel="Episodes", ylabel="Rewards"):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()