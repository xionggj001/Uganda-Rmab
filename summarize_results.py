import os
import pandas as pd
import glob

dfs = {}
is_std = False

for filename in glob.glob('./logs/results/rewards_c*.csv'):
    if not os.path.basename(filename).startswith("rewards_"):
        continue

    env = os.path.basename(os.path.dirname(filename))

    parts = os.path.basename(filename)[9:-4].split("_")
    experiment_name = "_".join(parts[:-5])  # The experiment name is everything before the last 5 underscore-separated elements

    N = parts[-4][1:]
    opt_in_rate = parts[-2][3:]
    
    df = pd.read_csv(filename)
    if is_std:
        mean_values = df.std() / int(int(N) * float(opt_in_rate))
    else:
        mean_values = df.mean() / int(int(N) * float(opt_in_rate))
    
    if N not in dfs:
        dfs[N] = pd.DataFrame()
    
    for method, mean_value in mean_values.items():
        dfs[N].loc[method, opt_in_rate] = mean_value

for N, df in dfs.items():
    df.sort_index(axis=1, inplace=True)
    if is_std:
        df.to_csv(f'logs/results/summary__std_{env}_N{N}.csv')
    else:
        df.to_csv(f'logs/results/summary_{env}_N{N}.csv')



