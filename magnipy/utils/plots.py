import matplotlib.pyplot as plt
import seaborn as sns


def plot_magnitude_function(mag, ts, name=""):
    """
    Plot a magnitude function.
    """
    plt.plot(mag, ts, label="magnitude function " + name)
    plt.xlabel("t")
    plt.ylabel("magnitude function")
    sns.despine()


def plot_magnitude_dimension_profile(mag_dim, ts, log_scale=False, name=""):
    """
    Plot a magnitude dimension profile.
    """
    plt.plot(ts, mag_dim, label="magnitude dimension profile " + name)
    if log_scale:
        plt.xlabel("log(t)")
    else:
        plt.xlabel("t")
    plt.ylabel("magnitude dimension profile")
    sns.despine()