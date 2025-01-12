import numpy as np
import pandas as pd
import time

from itertools import product

from robust_rmab.environments.bandit_env import SISBanditEnv, RandomBanditEnv, RandomBanditResetEnv, CirculantDynamicsEnv, ARMMANEnv
from robust_rmab.environments.bandit_env_robust import ToyRobustEnv, CounterExampleRobustEnv, ARMMANRobustEnv, SISRobustEnv, FakeT, ContinuousStateExampleEnv
from robust_rmab.environments.bandit_env_uganda import UgandaEnv
from robust_rmab.environments.bandit_env_mimiciv import MimicivEnv
from robust_rmab.environments.bandit_env_mimiciii import MimiciiiEnv

import os
import os.path as osp
import argparse
import tqdm
import itertools
import torch

import mdptoolbox


import matplotlib.pyplot as plt

index_policies = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
RL_policies = [101, 102]


def list_valid_action_combinations(N,C,B,options):

    costs = np.zeros(options.shape[0])
    for i in range(options.shape[0]):
        costs[i] = C[options[i]].sum()
    valid_options = costs <= B
    options = options[valid_options]
    return options


def barPlot(labels, values, errors, ylabel='Mean Discounted Reward',
            title='Simulated policy rewards', filename='image.png', root='.',
            bottom=0):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.85  # the width of the bars
    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
    # rects1 = ax.bar(x, values, width, bottom=bottom, label='Intervention benefit')
    
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)   
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    autolabel(rects1)       
    plt.tight_layout() 
    plt.savefig(fname)
    plt.show()


class RobustEnvWrapper():
    def __init__(self, env, params):
        # params is either nature_params or opt_in.
        # when we have changing # arms, params is opt_in
        self.params = params
        self.env = env

        # loop over the attributes of the parent class and create those for the decorator
        env_dict = vars(env)
        for attr in [a for a in env_dict if '__' not in a]:
            self.__dict__[attr] = env_dict[attr]

    def seed(self, seed):
        return self.env.seed(seed)

    def reset(self):
        return self.env.reset_random()

    def step(self, actions):
        return self.env.step(actions, self.params, mode="eval")

class RobustEnvWrapperArmman():
    def __init__(self, env, params):
        # params is either nature_params or opt_in.
        # when we have changing # arms, params is opt_in

        self.params = params
        self.env = env

        # loop over the attributes of the parent class and create those for the decorator
        env_dict = vars(env)
        for attr in [a for a in env_dict if '__' not in a]:
            self.__dict__[attr] = env_dict[attr]

    def seed(self, seed):
        return self.env.seed(seed)

    def reset(self):
        return self.env.reset_random()

    def step(self, actions):
        return self.env.step(actions, self.params, mode="eval")


# def takeAction(current_states, T, actions, random_stream):
#
#     N=len(current_states)
#
#     ###### Get next state
#     next_states=np.zeros(current_states.shape)
#     for i in range(N):
#
#         current_state=current_states[i]
#         next_state=np.argmax(random_stream.multinomial(1, T[i, current_state, int(actions[i]), :]))
#         next_states[i]=next_state
#
#     return next_states


def getActions(N, T, R, C, B, t, policy_option, act_dim, rl_info=None,
                        current_state=None, data_dict=None, env=None,
                        valid_action_combinations=None):

    gamma = data_dict['gamma']

    # Pull no arms
    if policy_option==0:

        return np.zeros(N)

    # Pull all arms
    elif policy_option==1:

        return np.ones(N)

    # Random continuous
    elif policy_option==2:

        # actions = np.zeros(N,dtype=int)
        # arm_inds = np.arange(N)
        # choices = np.random.choice(arm_inds, B, replace=False)
        # actions[choices] = 1
        # return actions
        actions = np.random.dirichlet([1]*N*act_dim)*B
        return actions.reshape(N,-1)

    # Random discrete (binary action only)
    elif policy_option==3:

        actions = np.zeros(N,dtype=int)
        arm_inds = np.arange(N)
        choices = np.random.choice(arm_inds, int(B), replace=False)
        actions[choices] = 1
        return actions




    # Round robin 1
    elif policy_option==5:
        actions = np.zeros(N)
        num_feasible = int(B/C[1])
        last_proc_acted_on = data_dict['last_proc_acted_on_rr']
        ind = 0
        for i in range(last_proc_acted_on+1, last_proc_acted_on+1 + num_feasible):
            ind = i%N
            actions[ind] = 1

        data_dict['last_proc_acted_on_rr'] = ind
        return actions


    # random action
    elif policy_option==6:

        actions = np.zeros(N,dtype=int)

        current_action_cost = 0
        candidate_arms = []
        for i in range(N):
            if rl_info['model'].arm_device_removed[i] < 0.5 and rl_info['model'].opt_in[i] > 0.5:
                if rl_info['model'].opt_in_steps[i] <= rl_info['model'].new_opt_in_guarantee_steps:
                    actions[i] = 1 # we must give newly opt-in arms the device
                    current_action_cost += 1 # assume binary action here
                else:
                    candidate_arms.append(i)
        # print('num eligible arms ', len(candidate_arms) + current_action_cost, ' new optin ', current_action_cost)
        if candidate_arms != []:
            if B - current_action_cost >= len(candidate_arms):
                chosen_arms = np.array(candidate_arms)
            else:
                chosen_arms = np.random.choice(candidate_arms, int(B - current_action_cost), replace=False)
            actions[chosen_arms] = 1
        # if current_action_cost < B: # this is for multi-action
        #     candidate_arms = np.array(candidate_arms)
        #     process_order = np.random.choice(candidate_arms, len(candidate_arms), replace=False)
        #
        #     for arm in process_order: # we only process eligible arms
        #         # select an action at random
        #         num_valid_actions_left = len(C[C<=B-current_action_cost])
        #         p = 1/(C[C<=B-current_action_cost]+1)
        #         p = p/p.sum()
        #         p = None
        #         a = np.random.choice(np.arange(num_valid_actions_left), 1, p=p)[0]
        #         current_action_cost += C[a]
        #         # if the next selection takes us over budget, break
        #         if current_action_cost > B:
        #             break
        #         actions[arm] = a

        # update self.arm_device_removed
        for i in range(N):
            if actions[i] == 0:
                if rl_info['model'].arm_device_usage[i] > 0 or rl_info['model'].opt_in[i] == 1:
                    rl_info['model'].arm_device_removed[i] = 1
                    # this arm used the device, and later we remove the device from this arm.
                    # thus, according to given constraints, from now on, we can no longer give the device to this arm.
            else: # actions[i] == 1:
                rl_info['model'].arm_device_usage[i] += 1
                if rl_info['model'].arm_device_usage[i] >= rl_info['model'].max_device_usage:
                    # this arm used the device for the max amount of steps allowed. we can no longer give this arm the device
                    rl_info['model'].arm_device_removed[i] = 1
        return actions


    # pull lowest state
    elif policy_option==7:

        actions = np.zeros(N,dtype=int)

        current_action_cost = 0
        candidate_arms = []
        for i in range(N):
            if rl_info['model'].arm_device_removed[i] < 0.5 and rl_info['model'].opt_in[i] > 0.5:
                if rl_info['model'].opt_in_steps[i] <= rl_info['model'].new_opt_in_guarantee_steps:
                    actions[i] = 1 # we must give newly opt-in arms the device
                    current_action_cost += 1 # assume binary action here
                else:
                    candidate_arms.append(i)
        # print('num eligible arms ', len(candidate_arms) + current_action_cost, ' new optin ', current_action_cost)
        if candidate_arms != []:
            if B - current_action_cost >= len(candidate_arms):
                chosen_arms = np.array(candidate_arms)
            else:
                # pull arms with lowest states
                normalized_state = -current_state # pull arm with high pulse rate, respiratory rate
                if data_dict['dataset_name'] in ['uganda', 'mimiciii']: # 'uganda', 'mimiciv', 'mimiciii'
                    normalized_state[0] *= -1 # pull arm with low SPO2. for uganda and mimiciii, state[0] is 'SPO2',
                vital_signs_avg = np.average(normalized_state, axis=1)
                chosen_arms = np.argsort(vital_signs_avg)[:int(B - current_action_cost)]
                # chosen_arms = np.random.choice(candidate_arms, int(B - current_action_cost), replace=False)
            actions[chosen_arms] = 1
        for i in range(N):
            if actions[i] == 0:
                if rl_info['model'].arm_device_usage[i] > 0 or rl_info['model'].opt_in[i] == 1:
                    rl_info['model'].arm_device_removed[i] = 1
                    # this arm used the device, and later we remove the device from this arm.
                    # thus, according to given constraints, from now on, we can no longer give the device to this arm.
            else: # actions[i] == 1:
                rl_info['model'].arm_device_usage[i] += 1
                if rl_info['model'].arm_device_usage[i] >= rl_info['model'].max_device_usage:
                    # this arm used the device for the max amount of steps allowed. we can no longer give this arm the device
                    rl_info['model'].arm_device_removed[i] = 1
        return actions

    # LP to compute the index policies (online vs. oracle version)
    elif policy_option==22:

        # print(policy_option)
        # print(T)

        actions = np.zeros(N,dtype=int)

        lambda_lim = R.max()/(C[C>0].min()*(1-gamma))

        indexes = np.zeros(N)

        a_index = 1 # need this because below method is set up to compute multi-action indices
        _, indexes = lp_methods.lp_to_compute_index(T, R, C, B, current_state, a_index, lambda_lim=lambda_lim, gamma=gamma)


        # compute all indexes
        if t==1 and policy_option==22:
            all_indexes = np.zeros((N, T.shape[1]))
            for s in range(T.shape[1]):
                state_vec = np.ones(N,dtype=int)*s
                # print(state_vec)
                _, all_indexes[:,s] = lp_methods.lp_to_compute_index(T, R, C, B, state_vec, 1, lambda_lim=lambda_lim, gamma=gamma)


            data_dict['lp-oracle-index'] = all_indexes



        # print(actions)
        # print(C)

        # action selection
        print('states')
        print(current_state)
        print('indexes')
        print(indexes)
        sorted_inds = np.argsort(indexes)[::-1]
        num_selected_actions = 0
        ind_cursor = 0
        epsilon=1e-6
        while num_selected_actions < B:

            next_best_index_value = indexes[sorted_inds[ind_cursor]]
            inds_of_equal_value = np.argwhere(abs(indexes - next_best_index_value)<epsilon).reshape(-1)
            if len(inds_of_equal_value) + num_selected_actions <= B:
                actions[inds_of_equal_value] = 1
                num_selected_actions += len(inds_of_equal_value)
                ind_cursor += len(inds_of_equal_value)
            else:
                num_actions_remaining = B - num_selected_actions
                randomly_chosen_inds = np.random.choice(inds_of_equal_value, num_actions_remaining, replace=False)
                actions[randomly_chosen_inds] = 1
                num_selected_actions = B



        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment == B: raise ValueError("Wrong budget")


        return actions



    # No cost value funcs
    elif policy_option==24:

        indexes = data_dict['indexes']

        actions = np.zeros(N)

        indexes_per_state = np.zeros((N, C.shape[0]))
        for i in range(N):
            s = current_state[i]
            indexes_per_state[i] = indexes[i,:,s]


        decision_matrix = lp_methods.action_knapsack(indexes_per_state, C, B)

        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")

        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")

        return actions







    # RMAB RL - returns an N-length action vector
    elif policy_option == 102:
        actions = get_action_rl(rl_info['model'], (current_state))
        payment = 0
        for a in actions:
            payment+= C[a]
        EPS = 1e-6
        if payment - EPS > B: raise ValueError("Over budget",payment)
        return actions


# make function for producing an action given a single state
def get_action_rl(model, x):
    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float32)
        # action = model.act_q(x)
        action = model.act_test(x)
        # action = model.act_test_stochastic(x)
    return action

def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    return model

def featurize_tp(transition_probs, transformation=None, out_dim=6):
    N = transition_probs.shape[0]
    output_features = np.zeros((N, out_dim))
    np.random.seed(0)  # Set random seed for reproducibility
    
    if transformation == "linear":
        transformation_matrix = np.random.rand(4, out_dim)
        output_features = np.dot(transition_probs, transformation_matrix)
    elif transformation == "nonlinear":
        transformation_matrix = np.random.rand(4, out_dim)
        output_features = 1 / (1 + np.exp(-np.dot(transition_probs, transformation_matrix)))
    else:
        output_features[:, :min(4, out_dim)] = transition_probs[:, :min(4, out_dim)]
    return output_features

def simulateAdherence(N, L, T, R, C, B, policy_option, start_state, seedbase=None, world_random_seed=None,
                        data_dict=None, file_root=None, env=None, rl_info=None, valid_action_combinations=None):

    gamma = data_dict['gamma']

    # env.seed(world_random_seed)
    start_state = env.reset()

    state_log=np.zeros((N, L, env.observation_dimension))
    action_log=np.zeros((N, L-1, env.action_dimension))


    # indexes = np.zeros((N,C.shape[0]))

    # Round robin setups
    if policy_option in [5]:
        data_dict['last_proc_acted_on_rr'] = N-1




    if policy_option in RL_policies:
        if policy_option == 101:
            model = load_pytorch_policy(rl_info['model_file_path_combinatorial'], "")
            rl_info['model'] = model
        if policy_option == 102:
            model = load_pytorch_policy(rl_info['model_file_path_rmab'], "")
            env.env.update_transition_probs(np.ones(env.env.N), mode='eval') # resample arms for every trial
            if data_dict['dataset_name'] in ['uganda', 'mimiciv', 'mimiciii']:
                T_matrix = env.features
            else:
                T_matrix = env.env.model_input_T if hasattr(env.env, 'model_input_T') else env.env.T
                T_matrix = np.reshape(T_matrix[:, :, :, 1:], (T_matrix[:, :, :, 1:].shape[0], np.prod(T_matrix[:, :, :, 1:].shape[1:])))
            model.transition_param_arr = T_matrix
            model.feature_arr = model.featurize_tp(T_matrix, transformation=model.tp_transform, out_dim=model.out_dim, in_dim=model.feature_input_dim)
            model.opt_in = env.params
            rl_info['model'] =  model

    print('Running simulation w/ policy: %s'%policy_option)
    if policy_option in index_policies:

        lambdas = np.zeros((N,C.shape[0]))
        V = np.zeros((N,T.shape[1]))

        start = time.time()


        if policy_option == 21:
            pass


        # VfNc
        if policy_option == 24:

            start = time.time()
            indexes = np.zeros((N, C.shape[0], T.shape[1]))

            # time to: add variables, add constraints, optimize, extract variable values
            for i in range(N):
                # Go from S,A,S to A,S,S
                T_i = np.swapaxes(T[i],0,1)
                R_i = np.zeros(T_i.shape)
                for x in range(R_i.shape[0]):
                    for y in range(R_i.shape[1]):
                        R_i[x,:,y] = R[i]

                mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion='full', epsilon=data_dict['mdp_epsilon'])
                mdp.run()

                V[i] = np.array(mdp.V)


            for i in range(N):
                for a in range(C.shape[0]):
                    for s in range(T.shape[1]):
                        indexes[i,a,s] = R[i,s] + gamma*V[i].dot(T[i,s,a])



    state_log[:,0] = start_state

    # data_dict['indexes'] = indexes


    #######  Run simulation #######
    print('Running simulation w/ policy: %s'%policy_option)
    print("Policy:", policy_option)

    ep_ret = 0
    # rl_info['model'].arm_device_removed = np.zeros(rl_info['model'].N) # reset tracker (whether we remove the device from an arm)
    # rl_info['model'].arm_device_usage = np.zeros(rl_info['model'].N) # reset tracker (how many steps has am arm used the device)

    rl_info['model'].arm_device_removed = np.zeros(N)  # reset tracker (whether we remove the device from an arm)
    rl_info['model'].arm_device_usage = np.zeros(N)  # reset tracker (how many steps has am arm used the device)
    rl_info['model'].opt_in = np.ones(N)  # reset opt-in status
    rl_info['model'].opt_in_steps = np.zeros(N)  # reset tracker (the amount of steps each opt-in arm stays in the system)
    rl_info['model'].opt_in[int(env.B):] *= 0  # block all arms except for first B arms
    for t in range(1,L):
        if t % 5 == 0 and t > 1:
            release_index = int(env.B + 2 * (t // 5) - 2)
            rl_info['model'].opt_in[release_index:release_index + 2] = 1  # release a blocked arm
        rl_info['model'].opt_in_steps[rl_info['model'].opt_in > 0.5] += 1  # update the amount of steps each opt-in arm stays in the system
        rl_info['model'].opt_in[rl_info['model'].opt_in_steps >= 50] = 0  # block arms that are in the system for 50 steps or more
        # print("Round %s"%t)
        actions=getActions(N, T, R, C, B, t, policy_option, env.action_space,
                            rl_info=rl_info, current_state=state_log[:,t-1],
                            data_dict=data_dict, env=env,
                            valid_action_combinations=valid_action_combinations)
        actions = actions.reshape(N,-1)
        # if policy_option == 6:
            # print('step ', t, 'sum action ', sum(actions))
        # print('policy option', policy_option, 'round ', t, 'complete')

        action_log[:, t-1]=actions
        EPS = 1e-6
        if policy_option != 0 and actions.sum() - B > EPS:
            print(actions.sum())
            raise ValueError('bad num actions')
        # print(policy_option, state_log[:,t-1])
        # print(policy_option, actions)


        next_state, r, d, _ = env.step(actions)
        ep_ret += r.sum()
        state_log[:,t] = next_state
        # state_log[:,t] = takeAction(state_log[:,t-1].reshape(-1), T, actions, random_stream=world_random_stream)

    # if policy_option == 22:
    #     utils.plot_indexes(data_dict['lp-oracle-index'])

    # if policy_option == 24:
    #     utils.plot_vfnc_indexes(data_dict['indexes'])


    return state_log, action_log, ep_ret





if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Run adherence simulations with various methods.')
    parser.add_argument('-N', '--num_arms', default=4, type=int, help='Number of Processes')
    parser.add_argument('-b', '--budget_frac', default=0.5, type=float, help='Budget per day as fraction of n')
    parser.add_argument(      '--budget', default=None, type=float, help='Total budget per day (trumps budget_frac)')
    parser.add_argument('-L', '--simulation_length', default=180, type=int, help='Number of days to run simulation')
    parser.add_argument('-n', '--num_trials', default=3, type=int, help='Number of trials to run')
    parser.add_argument('-S', '--num_states', default=2, type=int, help='Number of states per process')
    parser.add_argument('-A', '--num_actions', default=2, type=int, help='Number of actions per process') # Only two actions implemented for now
    parser.add_argument('-g', '--discount_factor', default=0.9, type=float, help='Discount factor for MDP solvers')
    parser.add_argument('-rb', '--REWARD_BOUND', default=1.0, type=float, help='Maximum reward')
    parser.add_argument('-opt', '--opt_in_rate', default=1.0, type=float, help='Maximum reward')

    parser.add_argument('-d', '--data', default='real', type=str,help='Method for generating transition probabilities',
                            choices=[   'uganda',
                                        'mimiciv',
                                        'mimiciii',
                                        'continuous_state'
                                    ])

    parser.add_argument('-me', '--mdp_epsilon', default=1e-1, type=float, help='Tolerance for Value Iteration')

    parser.add_argument('-s', '--seed_base', type=int, help='Base for the random seed')
    parser.add_argument('-ws','--world_seed_base', default=None, type=int, help='Base for the random seed')

    parser.add_argument('-f', '--file_root', default='./..', type=str,help='Root dir for experiment (should be the dir containing this script)')
    parser.add_argument('-pc', '--policy', default=-1, type=int, help='policy to run, default is all policies')
    parser.add_argument('-tr', '--trial_number', default=None, type=int, help='Trial number')
    parser.add_argument('-sv', '--save_string', default='', type=str, help='special string to include in saved file name')

    parser.add_argument('-sid', '--slurm_array_id', default=None, type=int, help='Specify the index of the parameter combo')
    parser.add_argument('-sva', '--save_actions', default=False, type=bool, help='Whether or not to save action logs')

    parser.add_argument('--init_infection_size', type=int, default=5)
    parser.add_argument('--population_size', type=int, default=100)

    parser.add_argument('-rlmfc', '--rl_combinatorial_model_filepath', default=None, type=str, help='path to Combinatorial RL model file if using')
    parser.add_argument('-rlmfr', '--rl_rmab_model_filepath', default=None, type=str, help='path to RMAB RL model file if using')
    parser.add_argument('-jhl', '--just_hawkins_lambda', default=False, type=bool, help='Just output the Hawkins lambda value')

    parser.add_argument('-nh', '--no_hawkins', default=0, type=int, help='If set, will not run Hawkins')

    parser.add_argument('--robust_keyword', default='mid', type=str,help='Method for picking some T out of robust env',
                        choices=[   
                                    'pess',
                                    'mid',
                                    'opt',
                                    'sample_random'
                                ])
    parser.add_argument('-ps', '--pop_size', default=10, type=int, help='Population size for SIS')

    args = parser.parse_args()

    if not args.no_hawkins:
        from robust_rmab import lp_methods


    # CODE ONLY IMPLEMENTED FOR TWO ACTIONS FOR NOW
    args.num_actions = 2


    # policy names dict
    pname={
            0: 'No Calls',    2: 'RandomC', 3: 'RandomD',
            4: 'MDP Optimal',
            5: 'Round Robin',

            6:'RandomDS',
            7:'LowestState',

            21:'Hawkins',
            22:'Whittle index',

            24:r'$\lambda=0$',
            25:'S-Based Random',

            101:'RLcomb',
            102:'RLRMAB'

    }


    ##### File root
    if args.file_root == '.':
        args.file_root = os.getcwd()

    ##### Save special name
    if args.save_string=='':
        args.save_string=str(time.ctime().replace(' ', '_').replace(':','_'))

    policies_to_plot = None

    ##### Policies to run
    if args.policy<0:
        #**************

        # continuous algos
        # policies = [0, 2, 101, 102]

        # discrete no action, discrete random, hawkins, RMABPPO
        policies = [0, 6, 21, 102]

        if args.no_hawkins:
            policies = [102, 0, 6, 7] # start with our policy to initialize the transition dynamics.
            # policies = [102, 0, 6]

        # policies = [0, 6, 102]

        # policies = [0, 3, 21]

        # discrete algos, rmab combined
        # policies = [0, 3, 21, 101]
        
        policies_to_plot = policies


    else:
        policies=[args.policy]


    if args.just_hawkins_lambda:
        policies = [0, 21]


    ################# This section only for setting up batched jobs to run on FASRC's cannon ########################

    # for i in 0 21 24; do sbatch --array=0-899 job.run_simulation_low_mem.sh $i; done

    NUM_TRIALS = 20
    trial_number_list = [i for i in range(NUM_TRIALS)]
    n_list = [250, 500, 750]
    budget_frac_list = [0.1, 0.2, 0.5]
    state_size_list = [3, 4, 5]
    master_combo_list = list(itertools.product(trial_number_list, n_list, budget_frac_list, state_size_list))

    # print(len(master_combo_list));1/0

    trial_percent_lam0 = 0
    if args.slurm_array_id is not None:
        combo = master_combo_list[args.slurm_array_id]

        args.trial_number = combo[0]
        args.num_arms = combo[1]
        args.budget_frac = combo[2]
        args.num_states = combo[3]

    # If we pass a trial number, that means we are running this as a job
    # and we want jobs/trials to run in parallel so this does some rigging to enable that,
    # while still synchronizing all the seeds
    if args.trial_number is not None:
        args.num_trials=1
        add_to_seed_for_specific_trial=args.trial_number
    else:
        add_to_seed_for_specific_trial=0


    #################################################################################################################


    first_seedbase=np.random.randint(0, high=100000)
    if args.seed_base is not None:
        first_seedbase = args.seed_base+add_to_seed_for_specific_trial

    first_world_seedbase=np.random.randint(0, high=100000)
    if args.world_seed_base is not None:
        first_world_seedbase = args.world_seed_base+add_to_seed_for_specific_trial




    # N=args.num_arms
    N = int(args.opt_in_rate * (args.simulation_length / 5 - 1) + args.budget) # every 5 steps, we have opt_in_rate many arms opt-in
    print('N', N)
    L=args.simulation_length
    savestring=args.save_string
    N_TRIALS=args.num_trials
    S = args.num_states
    A = args.num_actions
    B = int(N*args.budget_frac)
    opt_in_rate = args.opt_in_rate
    if args.budget is not None:
        B = args.budget
    rl_info = {
        'model_file_path_combinatorial':args.rl_combinatorial_model_filepath,
        'model_file_path_rmab':args.rl_rmab_model_filepath,
        'compute_hawkins_lambda':False

    }



    size_limits={
                    0:None, 1:None, 2:1000, 3:1000, 4:8,
                    5:None, 6:None, 7:None,
                    21:None, 22:None,
                    24:None, 25:None,
                    101:1000, 102:None
                }





    # for rapid prototyping
    # use this to avoid updating all the function calls when you need to pass in new
    # algo-specific things or return new data
    data_dict = {}
    data_dict['hawkins_lambda'] = []
    data_dict['gamma'] = args.discount_factor
    data_dict['mdp_epsilon'] = args.mdp_epsilon

    data_dict['hawkins_lambdas_rl_states'] = []
    data_dict['rl_lambdas'] = []
    data_dict['dataset_name'] = args.data




    start=time.time()
    file_root=args.file_root

    # for post-computation
    runtimes = np.zeros((N_TRIALS, len(policies)))
    reward_log=dict([(key,[]) for key in pname.keys()])
    state_log=dict([(key,[]) for key in pname.keys()])




    T = None
    R = None
    C = None
    # B = None gets set above
    start_state = None
    env=None



    # use np global seed for rolling random data, then for random algorithmic choices
    seedbase = first_seedbase
    torch.manual_seed(seedbase)
    np.random.seed(seed=seedbase)

    # Use world seed only for evolving the world (If two algs
    # make the same choices, should create the same world for same seed)
    world_seed_base = first_world_seedbase


    if args.data == 'continuous_state':
        env = ContinuousStateExampleEnv(N, B, seedbase)
        # env.update_transition_probs(np.ones(env.N)) # initialize all transition probs

        T = env.T
        R = env.R
        C = env.C

        current_state = np.random.get_state()
        np.random.seed()  # Or any other seed you'd like to use
        # num_opt_in = int(round(N * opt_in_rate))
        # opt_in_indices = np.random.choice(N, num_opt_in, replace=False)
        # opt_in_status = np.zeros(N)
        # opt_in_status[opt_in_indices] = 1
        np.random.set_state(current_state)
        opt_in_status = np.ones(N)

        env = RobustEnvWrapper(env, opt_in_status) # here the second argument is opt_in decisions.
        env.env.update_transition_probs(np.ones(env.env.N))
        # for now, in testing, assume all arms are opt-in.

    if args.data == 'uganda':
        env = UgandaEnv(N, B, seedbase)
        # env.update_transition_probs(np.ones(env.N)) # initialize all transition probs

        T = 0
        R = 0
        C = env.C

        current_state = np.random.get_state()
        np.random.seed()  # Or any other seed you'd like to use
        # num_opt_in = int(round(N * opt_in_rate))
        # opt_in_indices = np.random.choice(N, num_opt_in, replace=False)
        # opt_in_status = np.zeros(N)
        # opt_in_status[opt_in_indices] = 1
        np.random.set_state(current_state)
        opt_in_status = np.ones(N)

        env = RobustEnvWrapper(env, opt_in_status) # here the second argument is opt_in decisions.
        env.env.update_transition_probs(np.ones(env.env.N))
        # for now, in testing, assume all arms are opt-in.


    if args.data == 'mimiciv':
        env = MimicivEnv(N, B, seedbase)
        # env.update_transition_probs(np.ones(env.N)) # initialize all transition probs

        T = 0
        R = 0
        C = env.C

        current_state = np.random.get_state()
        np.random.seed()  # Or any other seed you'd like to use
        # num_opt_in = int(round(N * opt_in_rate))
        # opt_in_indices = np.random.choice(N, num_opt_in, replace=False)
        # opt_in_status = np.zeros(N)
        # opt_in_status[opt_in_indices] = 1
        np.random.set_state(current_state)
        opt_in_status = np.ones(N)

        env = RobustEnvWrapper(env, opt_in_status) # here the second argument is opt_in decisions.
        env.env.update_transition_probs(np.ones(env.env.N))
        # for now, in testing, assume all arms are opt-in.

    if args.data == 'mimiciii':
        env = MimiciiiEnv(N, B, seedbase)
        # env.update_transition_probs(np.ones(env.N)) # initialize all transition probs

        T = 0
        R = 0
        C = env.C

        current_state = np.random.get_state()
        np.random.seed()  # Or any other seed you'd like to use
        # num_opt_in = int(round(N * opt_in_rate))
        # opt_in_indices = np.random.choice(N, num_opt_in, replace=False)
        # opt_in_status = np.zeros(N)
        # opt_in_status[opt_in_indices] = 1
        np.random.set_state(current_state)
        opt_in_status = np.ones(N)

        env = RobustEnvWrapper(env, opt_in_status) # here the second argument is opt_in decisions.
        env.env.update_transition_probs(np.ones(env.env.N))
        # for now, in testing, assume all arms are opt-in.

    if args.data == 'sis':
        from robust_rmab.baselines.nature_baselines_sis import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, SampledRandomNaturePolicy
                        )
        env_fn = lambda: SISRobustEnv(N,B,args.pop_size,seedbase)
        
        # don't one hot encode this state space...

        POP_SIZE_LIM = 2000


        env = env_fn()
        sampled_nature_parameter_ranges = env.sample_parameter_ranges()
        # important to make sure these are always the same for all instatiations of the env
        env.sampled_parameter_ranges = sampled_nature_parameter_ranges

        nature_strategy = None
        if args.robust_keyword == 'mid':
            nature_strategy = MiddleNaturePolicy(sampled_nature_parameter_ranges, 0)
            middle_nature_params = sampled_nature_parameter_ranges.mean(axis=-1)
            if args.pop_size < POP_SIZE_LIM:
                T = env.get_T_for_a_nature(middle_nature_params)
            else:
                T = env.T # this will be a FakeT with the proper dimensions

        if args.robust_keyword == 'sample_random':
            nature_strategy = SampledRandomNaturePolicy(sampled_nature_parameter_ranges, 0)

            # init the random strategy
            nature_strategy.sample_param_setting(seedbase)
            sampled_nature_params = nature_strategy.param_setting
            if args.pop_size < POP_SIZE_LIM:
                T = env.get_T_for_a_nature(sampled_nature_params)
            else:
                T = env.T


        # N = 3
        # env = CounterExampleRobustEnv(B, seedbase)
        # T = env.T
        R = env.R
        C = env.C

        nature_actions = nature_strategy.get_nature_action(None)

        current_state = np.random.get_state()
        np.random.seed()  # Or any other seed you'd like to use
        num_opt_in = int(round(N * opt_in_rate))
        opt_in_indices = np.random.choice(N, num_opt_in, replace=False)
        opt_in_status = np.zeros(N)
        opt_in_status[opt_in_indices] = 1
        np.random.set_state(current_state)

        env = RobustEnvWrapper(env, opt_in_status)
        env.env.update_transition_probs(np.ones(env.env.N), mode='eval')


    valid_action_combinations = None



    np_seed_states = []
    world_seed_states = []

    # create a bunch of random seed states that can be replicated 
    # if we need to add another policy later
    for i in range(N_TRIALS):
        
        # save the states
        np_seed_states.append(np.random.get_state())
        world_seed_states.append(env.random_stream.get_state())

        # evolve the states
        np.random.rand()
        env.random_stream.rand()
        
    for i in range(N_TRIALS):

        if valid_action_combinations is None:
            combinatorial_policies = set(policies) & set([4, 101])
            if len(combinatorial_policies) > 0:

                at_least_one_size_limit_satisfied = False
                for policy in combinatorial_policies:
                    at_least_one_size_limit_satisfied |= size_limits[policy]>N

                if at_least_one_size_limit_satisfied:
                    options = np.array(list(product(np.arange(C.shape[0]), repeat=N)))
                    valid_action_combinations = list_valid_action_combinations(N,C,B,options)


        


        for p,policy_option in enumerate(policies):


            if size_limits[policy_option]==None or size_limits[policy_option]>N:

                np_seed_state_for_trial = np_seed_states[i]
                world_seed_state_for_trial = world_seed_states[i]


                # reset the seed states for all policies to have same shot
                np.random.set_state(np_seed_state_for_trial)
                env.random_stream.set_state(world_seed_state_for_trial)

                # TODO - recover MDP optimal
                optimal_policy = None
                combined_state_dict = None

                data_dict['optimal_policy'] = optimal_policy
                data_dict['combined_state_dict'] = combined_state_dict


                policy_start_time=time.time()
                state_matrix, action_log, ep_ret = simulateAdherence(N, L, T, R, C, B, policy_option, start_state, seedbase=seedbase, world_random_seed=world_seed_base,
                                                   data_dict=data_dict, file_root=file_root, env=env, rl_info=rl_info, valid_action_combinations=valid_action_combinations)
                policy_end_time=time.time()


                ####### SAVE RELEVANT LOGS #########

                state_log[policy_option].append(state_matrix)

                policy_run_time=policy_end_time-policy_start_time
                np.save(file_root+'/logs/runtimes/runtime_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states), policy_run_time)

                runtimes[i,p] = policy_run_time

                reward_log[policy_option].append(ep_ret)



                # write out action logs
                if args.save_actions:
                    print(action_log.shape)
                    print(action_log)
                    fname = os.path.join(args.file_root,'logs/action_log/action_log_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s.csv'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states))
                    columns = list(map(str, np.arange(L-1)))
                    df = pd.DataFrame(action_log, columns=columns)
                    df.to_csv(fname, index=False)




    end=time.time()
    print ("Time taken: {:2f} s".format(end-start))

    # for i,p in enumerate(policies):
    #     print ('{}: {:.3f} +/- {:.3f}'.format(pname[p], runtimes[:,i].mean(), runtimes[1:,i].std()))

    # print("Max state:",max_state)

    labels = [pname[i] for i in policies_to_plot]
    values_for_df=np.array([reward_log[i] for i in policies_to_plot])
    values_for_df=values_for_df.T

    df = pd.DataFrame(values_for_df, columns=labels)
    fname = file_root+'/logs/results/rewards_%s_n%s_b%s_opt%s.csv'%(args.data, N, int(args.budget), args.opt_in_rate)
    if os.path.exists(fname):
        df.to_csv(fname, mode='a', header=False, index=False)
    else:
        df.to_csv(fname, index=False)
