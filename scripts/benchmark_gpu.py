from magnipy import Magnipy
import numpy as np
import time
import pandas as pd
from magnipy.magnitude.approximation import greedy_maximization, magnitude_by_batch_SGD, magnitude_by_SGD, discrete_center_hierarchy

def tes_ts(data, ts, method):
    Mag = Magnipy(X=data, ts=ts, method = method)
    return Mag.get_magnitude_weights()[0]

if __name__ == "__main__":

    methods = [
        "naive",
        "cholesky",
        "scipy",
        "scipy_sym",
        "pinv",
        "conjugate_gradient_iteration",
        "cg",
        "spread",
        #"spread_torch",
        "naive_torch",
        "cholesky_torch",
        "pinv_torch",
        "solve_torch"
        # "krylov",
        #"greedy_maximization",
        #"magnitude_by_batch_SGD",
        #"magnitude_by_SGD",
        #"discrete_center"
    ]
    
    #tss = [[1], np.linspace(0.01, 1, 100), None]
    df_results = pd.DataFrame()

    #for ts in tss:
    methodd = []
    runtimes=[]
    n_obsss = []
    mean_errors = []
    std_errors = []
    mag_errors = []
    for n_obs in [10, 50, 100, 250, 500, 750, 1000, 
                1250, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
        print("n_obs ", n_obs)
        for i in range(10):
            print(i)
            np.random.seed(i)
            data = np.random.uniform(0, 1, (n_obs, 10))
            Mag = Magnipy(X=data, method = "naive", n_ts=2)
            t_conv = Mag.get_t_conv()
            ts = np.linspace(0.01, t_conv, 10)
            w_true = tes_ts(data, ts, "naive")
            mag_true = np.sum(w_true)

            for method in methods:
                print(method)
                start_time = time.perf_counter()
                weights = tes_ts(data, ts, method)
                end_time = time.perf_counter()

                mean_error = np.mean(np.abs(w_true - weights))
                std_error = np.std(np.abs(w_true - weights))
                mag_error = np.abs(mag_true - np.sum(weights))

                runtimes.append(end_time - start_time)
                n_obsss.append(n_obs)
                methodd.append(method)
                mean_errors.append(mean_error)
                std_errors.append(std_error)
                mag_errors.append(mag_error)

    df_results = pd.DataFrame({"method": methodd, "n_obs": n_obsss, "runtime": runtimes, "mag_error": mag_errors, "weight_mean_error": mean_errors, "weight_std_error": std_errors})
    df_results.to_csv("./benchmark_results.csv")

