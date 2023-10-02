import os
import pandas as pd
import glob

dfs = {}
is_std = False

##for filename in glob.glob('./logs/results/rewards_contin*.csv'):
for filename in glob.glob('./logs/results/rewards_sis*.csv'):
    if not os.path.basename(filename).startswith("rewards_"):
        continue

    env = os.path.basename(os.path.dirname(filename))

    parts = os.path.basename(filename)[9:-4].split("_")
    experiment_name = "_".join(parts[:-5])  # The experiment name is everything before the last 5 underscore-separated elements

    N = parts[-4][1:]
    B = parts[-3][1:]
    opt_in_rate = parts[-2][3:]
    
    df = pd.read_csv(filename)
    if is_std:
        mean_values = df.std() / int(int(N) * float(opt_in_rate))
    else:
        mean_values = df.mean() / int(int(N) * float(opt_in_rate))

    if B not in dfs:
        dfs[B] = pd.DataFrame()
    
    for method, mean_value in mean_values.items():
        dfs[B].loc[method, opt_in_rate] = mean_value

for B, df in dfs.items():
    df.sort_index(axis=1, inplace=True)
    if is_std:
        df.to_csv(f'logs/results/summary_sis_std_{env}_B{B}.csv')
    else:
        df.to_csv(f'logs/results/summary_sis_{env}_B{B}.csv')

##dfs = {}
##
##for filename in glob.glob('./logs/results/rewards_contin*.csv'):
##    if not os.path.basename(filename).startswith("rewards_"):
##        continue
##
##    env = os.path.basename(os.path.dirname(filename))
##
##    parts = os.path.basename(filename)[9:-4].split("_")
##    experiment_name = "_".join(parts[:-5])  # The experiment name is everything before the last 5 underscore-separated elements
##
##    N = parts[-4][1:]
##    opt_in_rate = parts[-2][3:]
##    new_N = round(float(N) * float(opt_in_rate))
##
##    df = pd.read_csv(filename)
##    mean_values = df.mean() / new_N
##    print(new_N, opt_in_rate, "\n", mean_values, "\n\n")
##    
##    if N not in dfs:
##        dfs[N] = pd.DataFrame()
##    
##    for method, mean_value in mean_values.items():
##        dfs[N].loc[method, opt_in_rate] = mean_value
##
##for N, df in dfs.items():
##    df.sort_index(axis=1, inplace=True)
##    df.to_csv(f'logs/results/summary_counterexample_{env}_N{N}.csv')
