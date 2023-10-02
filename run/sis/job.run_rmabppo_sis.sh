data="sis"
save_string="sis_rmabppo_test"
N=20
B=16.0
robust_keyword="sample_random" # other option is "mid"
n_train_epochs=100
seed=0
cdir="."
no_hawkins=1
pop_size=150
opt_in_rate=0.8

bash run/sis/run_rmabppo_sis_test.sh ${cdir} ${seed} 0 ${data} ${save_string} ${N} ${B} ${robust_keyword} ${n_train_epochs} ${no_hawkins} ${pop_size} ${opt_in_rate}



