data="armman"
save_string="armman_rmabppo_test"
N=5
B=1.0
robust_keyword="sample_random" # other option is "mid"
n_train_epochs=10
seed=0
cdir="."
no_hawkins=1

bash run/run_rmabppo_armman.sh ${cdir} ${seed} 0 ${data} ${save_string} ${N} ${B} ${robust_keyword} ${n_train_epochs} ${no_hawkins} 



