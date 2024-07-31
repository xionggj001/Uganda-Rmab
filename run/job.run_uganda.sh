data="uganda"
save_string="uganda"
B=5.0
robust_keyword="sample_random" # other option is "mid"
n_train_epochs=30 
seed=0
cdir="."
no_hawkins=1
tp_transform=None
opt_in_rate=2
N=43

bash run/run_uganda.sh ${cdir} ${seed} 0 ${data} ${save_string} ${N} ${B}  \
    ${robust_keyword} ${n_train_epochs} ${no_hawkins} ${tp_transform} ${opt_in_rate}



