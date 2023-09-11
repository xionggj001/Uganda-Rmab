import os
import pandas as pd
import glob

dfs = {}

for filename in glob.glob('./logs/results/*.csv'):
    parts = os.path.basename(filename).split('_')
    env = parts[1]
    N, opt_in_rate = parts[3][1:], parts[5][3:]
    df = pd.read_csv(filename)
    mean_values = df.mean() / int(N)
    
    if N not in dfs:
        dfs[N] = pd.DataFrame()
    
    for method, mean_value in mean_values.items():
        dfs[N].loc[method, opt_in_rate] = mean_value

for N, df in dfs.items():
    df.sort_index(axis=1, inplace=True)
    df.to_csv(f'logs/results/summary_{env}_N{N}.csv')
