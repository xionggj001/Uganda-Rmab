data="counterexample"
save_string="ce_rmabppo_test"
N=21
B=7.0
robust_keyword="sample_random" # other option is "mid"
n_train_epochs=200
seed=0
cdir="."
no_hawkins=1
tp_transform="linear"

bash run/run_rmabppo_counterexample.sh ${cdir} ${seed} 0 ${data} ${save_string} ${N} ${B} ${robust_keyword} ${n_train_epochs} ${no_hawkins} ${tp_transform}


