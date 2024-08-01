

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# from numba import jit
from itertools import product
# from code.lp_methods import action_knapsack

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)



class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPQCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q_net = mlp([obs_dim+act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, x):
        return torch.squeeze(self.q_net(x), -1) # Critical to ensure Q has right shape.


class MLPLambdaNet(nn.Module):

    def __init__(self, input_dim, hidden_sizes, activation):
        super().__init__()

        self.lambda_net = mlp([input_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, input): # input should be obs and transition probabilities
        return torch.squeeze(self.lambda_net(input), -1)


# @jit(nopython=True)
def list_valid_action_combinations(N,C,B,options):

    costs = np.zeros(options.shape[0],dtype=np.float32)
    for i in range(options.shape[0]):
        costs[i] = C[options[i]].sum()
    valid_options = costs <= B
    options = options[valid_options]
    return options

class MLPActorCriticRMAB(nn.Module):


    def __init__(self, obs_dim, action_space, opt_in_rate=1.0,
                 hidden_sizes=(64,64), input_feat_dim=4, C=None, N=None, B=None,
                 strat_ind=0, activation=nn.Tanh, use_feature=0
                 ):
        super().__init__()

        self.use_feature = use_feature
        self.feature_arr = np.zeros((N, input_feat_dim))
        self.opt_in = np.ones(N) # assume all arms opt-in at the start
        self.opt_in_steps = np.zeros(N)
        self.opt_in_rate = opt_in_rate
        self.state_norm = 1

        # the following variables are used for enforcing constraints
        self.arm_device_removed = np.zeros(N) # if element i is 1, then we have removed the device from this person. we can no longer give the same person the device
        self.max_device_usage = 25 # a person can not use the device for more than a fixed number of steps
        self.arm_device_usage = np.zeros(N) # track for how many steps has this arm used the device
        self.new_opt_in_guarantee_steps = 3 # we must give newly opt-in arms the device for this number of steps

        # one-hot-encode the states for now
        self.obs_dim = obs_dim

        self.act_type = 'd' # for discrete

        # we will only work with discrete actions
        self.act_dim = action_space.shape[0]

        # self.pi_list = np.zeros(N,dtype=object)
        # self.v_list = np.zeros(N,dtype=object)
        # self.q_list = np.zeros(N,dtype=object)
        self.N = N
        self.C = C
        self.B = B
        self.hidden_sizes = hidden_sizes
        self.activation = activation

        self.input_feat_dim = input_feat_dim
        self.pi_list = MLPCategoricalActor(self.obs_dim, self.act_dim, hidden_sizes, activation)
        self.v_list  = MLPCritic(self.obs_dim, hidden_sizes, activation)
        if self.use_feature == 1:
            self.pi_list = MLPCategoricalActor(self.obs_dim+1+self.input_feat_dim, self.act_dim, hidden_sizes, activation)
            self.v_list  = MLPCritic(self.obs_dim+1+self.input_feat_dim, hidden_sizes, activation)
        self.q_list = MLPQCritic(self.obs_dim+1+self.input_feat_dim, self.act_dim, hidden_sizes, activation)

        # Lambda_net is currently expected one input per arm, but other
        # networks are one-hot encoding the states...
        # This leads to a dimension difference...
        # need to change this eventually
        lambda_hidden_sizes = [8, 8]
        transition_prob_dim = int(N * input_feat_dim)
        self.lambda_net = MLPLambdaNet(N * 4 + transition_prob_dim, lambda_hidden_sizes, activation)
        # self.lambda_net = MLPLambdaNet(N, lambda_hidden_sizes, activation)

        self.name = "RMABPPO"
        self.ind = strat_ind

    # def update_transition_probs(self, transition_probs_input):
    #     # assume states are (0,1) and actions are (0,1). so transition probabilities can be encode by 4 numbers
    #     # input is 1d array of length N
    #     # output is 2d array of shape (N, 4)
    #     # output[i] should be (0.5, 0.5, 0, p) --see synthetic experiments description in the paper
    #     self.feature_arr = np.zeros((self.N, 4))
    #     for i in range(self.N):
    #         self.feature_arr[i] = (0.5, 0.5, 0.0, transition_probs_input[i])

    # def update_opt_in(self):
    #     # randomly choose arms to opt-out. randomly choose some opt-out states to opt-in again
    #     # opt_in_prob = [0.9, 0.8] # probability that an arm will opt-in given it is currently opt-in / opt-out
    #     # with opt_in_prob = [0.9, 0.8], in expectation, 88.89% of all arms are opt-in, among which 10% are new beneficieries
    #     # next_iter_prob = self.opt_in * opt_in_prob[0] + (1 - self.opt_in) * opt_in_prob[1]
    #     np.random.seed(None)
    #     st0 = np.random.get_state()
    #
    #     next_iter_prob = [self.opt_in_rate] * self.N
    #     new_opt_in = np.random.binomial([1] * self.N, next_iter_prob)
    #     new_arms_indices = ((new_opt_in - self.opt_in) > 0.5).astype(float)
    #     self.opt_in = new_opt_in
    #
    #     np.random.set_state(st0)
    #     return new_arms_indices

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)


    def reset_actor_and_critic_networks(self):
        self.pi_list = MLPCategoricalActor(self.obs_dim, self.act_dim, self.hidden_sizes, self.activation)
        self.v_list  = MLPCritic(self.obs_dim, self.hidden_sizes, self.activation)
        if use_feature == 1:
            self.pi_list = MLPCategoricalActor(self.obs_dim+1+self.input_feat_dim, self.act_dim, self.hidden_sizes, self.activation)
            self.v_list  = MLPCritic(self.obs_dim+1+self.input_feat_dim, self.hidden_sizes, self.activation)
        self.q_list = MLPQCritic(self.obs_dim+1+self.input_feat_dim, self.act_dim, self.hidden_sizes, self.activation)


    def return_large_lambda_loss(self, obs, gamma):

        disc_cost = 2 * self.B/(1-gamma)
        # lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))
        lambda_net_input = np.concatenate((obs, self.feature_arr.flatten()))
        lamb = self.lambda_net(torch.as_tensor(lambda_net_input, dtype=torch.float32))

        loss = lamb*(self.B/(1-gamma) - disc_cost)

        return loss


    def step(self, obs, lamb):
        # modified for Uganda data and Mimic data. Only supports binary actions
        with torch.no_grad():
            arms_eligible = (1 - self.arm_device_removed) * self.opt_in  # these arms can no longer use the device.
            n_arms_eligible = sum(arms_eligible)

            obs = obs/self.state_norm
            a_list = np.zeros(self.N,dtype=int)
            v_list = np.zeros(self.N) # not to confuse with self.v_list
            logp_a_list = np.zeros(self.N)
            pi_list = np.zeros((self.N, self.act_dim), dtype=float)
            for i in range(self.N):
                if arms_eligible[i] > 0.5:
                    if self.opt_in_steps[i] <= self.new_opt_in_guarantee_steps:
                        pi_list[i, 1:] = 100  # we must give newly opt-in arms the device
                    else:
                        full_obs = obs[i]
                        if self.use_feature == 1:
                            full_obs = np.concatenate([obs[i],[lamb],self.feature_arr[i]])
                        full_obs = torch.as_tensor(full_obs,dtype=torch.float32)
                        pi_list[i] = self.pi_list._distribution(full_obs).probs.detach().numpy()

                        v = self.v_list(full_obs)
                        v_list[i] = v.numpy()
                else:
                    pi_list[i, 1:] -= 1

            # play the actions with the largest probs.
            # need to update and take into account opt-in
            ACTION  = 1
            a1_list = pi_list[:,ACTION]

            sorted_inds = np.argsort(a1_list)[::-1]
            i = 0
            budget_spent = 0
            while budget_spent < min(self.B, n_arms_eligible) and i < self.N:
            # while budget_spent < self.B and i < self.N:
                # if taking the next action (greedy) puts over budget, break
                if budget_spent + self.C[ACTION] > self.B:
                    break
                a_list[sorted_inds[i]] = ACTION
                self.arm_device_usage[sorted_inds[i]] += 1
                budget_spent += self.C[ACTION]
                i+=1
            # update trackers
            for i in range(self.N):
                if a_list[i] == 0:
                    if self.arm_device_usage[i] > 0 or self.opt_in[i] == 1:
                    # if we do not give device at the start or remove the device, we can never give it back
                        self.arm_device_removed[i] = 1
                if self.arm_device_usage[i] >= self.max_device_usage:
                    # each arm can only use device for a maximum number of steps
                    self.arm_device_removed[i] = 1


            # compute loss prob of actions
            for i in range(self.N):
                full_obs = obs[i]
                if self.use_feature == 1:
                    full_obs = np.concatenate([obs[i],[lamb],self.feature_arr[i]])
                full_obs = torch.as_tensor(full_obs,dtype=torch.float32)
                pi = self.pi_list._distribution(full_obs)
                logp_a = self.pi_list._log_prob_from_distribution(pi, torch.tensor(a_list[i], dtype=torch.int8))
                # logp_a is log of probability of choosing this action.
                # if a=1, then logp_a=log(a1_probs[i]); else, logp_a=log(1-a1_probs[i])
                # if self.opt_in[i] < 0.5:
                #     logp_a = torch.tensor(np.log(0.1)) # log(0) is undefined, so we use log(0.1)
                logp_a_list[i] = logp_a.numpy()

        return a_list, v_list, logp_a_list

    # def get_probs_for_all(self, obs, lamb):
    #
    #     with torch.no_grad():
    #         prob_a_list = np.zeros(self.N)
    #         obs = obs/self.state_norm
    #
    #         for i in range(self.N):
    #             transition_prob = self.feature_arr[i]
    #             full_obs = None
    #             full_obs = np.concatenate([obs[i],[lamb], transition_prob])
    #             # full_obs = obs[i]
    #
    #             full_obs = torch.as_tensor(full_obs,dtype=torch.float32)
    #             pi = self.pi_list._distribution(full_obs)
    #             prob_a_list[i] = pi.probs[1]
    #
    #
    #     return prob_a_list

        

    def act(self, obs, lamb):
        a = self.step(obs, lamb, self.transition_prob)[0]
        return a

    def act_test(self, obs):
        # obs=obs.reshape(-1)
        # return self.act_test_deterministic(obs)
        return self.act_test_deterministic_multiaction(obs)
        # return self.act_test_stochastic_multi_action(obs)

    def get_lambda(self, obs):
        obs = obs.reshape(-1)
        obs = obs/self.state_norm
        # lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))
        lambda_net_input = np.concatenate((obs, self.feature_arr.flatten()))
        lamb = self.lambda_net(torch.as_tensor(lambda_net_input, dtype=torch.float32))
        return lamb.detach().numpy()

    # # Currently only implemented for binary action
    # def act_test_deterministic(self, obs):
    #     # print("Enforcing budget constraint on action")
    #     ACTION = 1
    #     a_list = np.zeros(self.N,dtype=int)
    #     pi_list = np.zeros((self.N,self.act_dim),dtype=float)
    #     with torch.no_grad():
    #         obs = obs/self.state_norm
    #
    #         # lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))
    #         lambda_net_input = np.concatenate((obs, self.feature_arr.flatten()))
    #         lamb = self.lambda_net(torch.as_tensor(lambda_net_input,dtype=torch.float32))
    #
    #         for i in range(self.N):
    #             transition_prob = self.feature_arr[i]
    #             full_obs = None
    #             full_obs = np.concatenate([obs[i],[lamb],transition_prob])
    #             # full_obs = obs[i]
    #
    #             full_obs = torch.as_tensor(full_obs, dtype=torch.float32)
    #
    #             pi = self.pi_list._distribution(full_obs).probs.detach().numpy()
    #             pi_list[i] = pi
    #
    #         # play the actions with the largest probs
    #         a1_list = pi_list[:,ACTION]
    #         # print(a1_list)
    #
    #         sorted_inds = np.argsort(a1_list)[::-1]
    #
    #         i = 0
    #         budget_spent = 0
    #         while budget_spent < self.B and i < self.N:
    #
    #             # if taking the next action (greedy) puts over budget, break
    #             if budget_spent + self.C[ACTION] > self.B:
    #                 break
    #
    #             a_list[sorted_inds[i]] = ACTION
    #             budget_spent += self.C[ACTION]
    #
    #             i+=1
    #
    #
    #     return a_list

    # Multi-action implementation
    # Naive -- do the same thing, but first take a max over the actions per arm
    def act_test_deterministic_multiaction(self, obs):
        # print("Enforcing budget constraint on action")
        arms_eligible = (1 - self.arm_device_removed) * self.opt_in   # these arms can no longer use the device.
        n_arms_eligible = sum(arms_eligible)

        actions = np.zeros(self.N,dtype=int)
        pi_list = np.zeros((self.N,self.act_dim),dtype=float)

        with torch.no_grad(): 
            # print(obs)   
            obs = obs/self.state_norm

            # lambda_net_input = np.concatenate((obs.reshape(-1), self.feature_arr.flatten()))
            # lamb = self.lambda_net(torch.as_tensor(lambda_net_input,dtype=torch.float32))
            lamb = 0


            for i in range(self.N):
                if arms_eligible[i] > 0.5:
                    if self.opt_in_steps[i] <= self.new_opt_in_guarantee_steps:
                        pi_list[i, 1:] = 100  # we must give newly opt-in arms the device
                    else:
                        full_obs = obs[i]
                        if self.use_feature == 1:
                            full_obs = np.concatenate([obs[i],[lamb],self.feature_arr[i]])
                        # print(full_obs)
                        full_obs = torch.as_tensor(full_obs, dtype=torch.float32)

                        pi = self.pi_list._distribution(full_obs).probs.detach().numpy()
                        pi_list[i] = pi
                else:
                    pi_list[i, 1:] -= 1


            row_maxes = pi_list.max(axis=1)
            row_order = np.argsort(row_maxes)[::-1]

            pi_arg_maxes = np.argsort(pi_list, axis=1)

            actions = np.zeros(self.N, dtype=int)

            budget_spent = 0
            done = False

            # while budget_spent < self.B and not done:
            while budget_spent < min(self.B, n_arms_eligible) and not done:

                i=0

                while i < self.N:
                    arm = row_order[i]
                    arm_a = pi_arg_maxes[row_order[i]][-1]
                    # if self.opt_in[i] < 0.5:
                    #     arm_a = 0 # 'no pull' action for opt-out arms.
                    a_cost = self.C[arm_a]

                    if budget_spent + a_cost - self.C[actions[arm]] > self.B:
                        done = True
                        # print('broke here')
                        break
                    else:
                        i+=1
                        # assign action
                        actions[arm] = arm_a
                        # now hide all cheaper actions
                        pi_list[arm, :arm_a+1] = 0
                        # print(a)

                        budget_spent = sum(self.C[a] for a in actions)
                        # update trackers
                        if arm_a > 0: # we give device to this arm
                            self.arm_device_usage[arm] += 1

                        # also hide all actions that are now too expensive
                        cost_diff_array = np.zeros(pi_list.shape)
                        for j in range(self.N):
                            cost_diff_array[j] = self.C - self.C[actions[j]]
                        overbudget_action_inds = cost_diff_array > self.B - budget_spent
                        
                        if overbudget_action_inds.any():
                            i = 0
                            pi_list[overbudget_action_inds] = 0
                            row_maxes = pi_list.max(axis=1)
                            row_order = np.argsort(row_maxes)[::-1]

                            pi_arg_maxes = np.argsort(pi_list, axis=1)
                        if not pi_list.sum() > 0:
                            done = True
                            break


                row_maxes = pi_list.max(axis=1)
                row_order = np.argsort(row_maxes)[::-1]

                pi_arg_maxes = np.argsort(pi_list, axis=1)

        # update trackers
        for i in range(self.N):
            if actions[i] == 0:
                if self.arm_device_usage[i] > 0 or self.opt_in[i] == 1:
                    # if we do not give device at the start or remove the device, we can never give it back
                    self.arm_device_removed[i] = 1
            if self.arm_device_usage[i] >= self.max_device_usage:
                # each arm can only use device for a maximum number of steps
                self.arm_device_removed[i] = 1

        return actions


    # # Currently only implemented for binary action
    # def act_test_stochastic_binary(self, obs):
    #     # print("Enforcing budget constraint on action")
    #     ACTION = 1
    #     a_list = np.zeros(self.N,dtype=int)
    #     pi_list = np.zeros((self.N,self.act_dim),dtype=float)
    #     obs = obs.reshape(-1)
    #
    #     with torch.no_grad():
    #         obs = obs/self.state_norm
    #
    #         # lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))
    #         lambda_net_input = np.concatenate((obs, self.feature_arr.flatten()))
    #         lamb = self.lambda_net(torch.as_tensor(lambda_net_input,dtype=torch.float32))
    #
    #         for i in range(self.N):
    #             transition_prob = self.feature_arr[i]
    #             full_obs = None
    #             full_obs = np.concatenate([obs[i],[lamb],transition_prob])
    #             full_obs = torch.as_tensor(full_obs, dtype=torch.float32)
    #
    #             pi = self.pi_list._distribution(full_obs).probs.detach().numpy()
    #             pi_list[i] = pi
    #
    #         # play the actions with the largest probs
    #         a1_list = pi_list[:,ACTION]
    #
    #         options = list(np.arange(self.N))
    #
    #         i = 0
    #         budget_spent = 0
    #         while budget_spent < self.B and i < self.N:
    #
    #             # if taking the next action (greedy) puts over budget, break
    #             if budget_spent + self.C[ACTION] > self.B:
    #                 break
    #
    #             normalized_probs = a1_list[options] / a1_list[options].sum()
    #
    #             # might want to give the ac object its own seed?
    #             # Rather than relying on the numpy global seed
    #             # print(options, normalized_probs)
    #             choice = np.random.choice(options, p=normalized_probs)
    #
    #             a_list[choice] = ACTION
    #             budget_spent += self.C[ACTION]
    #
    #             options.remove(choice)
    #
    #             i+=1
    #
    #
    #     return a_list
    #
    #
    # # multi-action implementation
    # def act_test_stochastic_multi_action(self, obs):
    #     # print("Enforcing budget constraint on action")
    #
    #     actions = np.zeros(self.N,dtype=int)
    #     pi_list = np.zeros((self.N,self.act_dim),dtype=float)
    #     obs = obs.reshape(-1)
    #
    #     with torch.no_grad():
    #         obs = obs/self.state_norm
    #
    #         # lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))
    #         lambda_net_input = np.concatenate((obs, self.feature_arr.flatten()))
    #         lamb = self.lambda_net(torch.as_tensor(lambda_net_input,dtype=torch.float32))
    #
    #         for i in range(self.N):
    #             transition_prob = self.feature_arr[i]
    #             full_obs = None
    #             full_obs = np.concatenate([obs[i],[lamb],transition_prob])
    #             full_obs = torch.as_tensor(full_obs, dtype=torch.float32)
    #
    #             pi = self.pi_list._distribution(full_obs).probs.detach().numpy()
    #             pi_list[i] = pi
    #
    #         # play the actions with the largest probs
    #
    #
    #         # options = list(np.arange(self.N))
    #
    #         # actions = np.zeros(N,dtype=int)
    #
    #         current_action_cost = 0
    #         p=pi_list.max(axis=1)
    #         p = p/p.sum()
    #         process_order = np.random.choice(np.arange(self.N), self.N, p=p, replace=False)
    #         for arm in process_order:
    #
    #             # select an action at random
    #             num_valid_actions_left = len(self.C[self.C<=self.B-current_action_cost])
    #             p = pi_list[arm][self.C<=self.B-current_action_cost]
    #             p = p/p.sum()
    #             p = None
    #             a = np.random.choice(np.arange(num_valid_actions_left), p=p)
    #             current_action_cost += self.C[a]
    #             # if the next selection takes us over budget, break
    #             if current_action_cost > self.B:
    #                 break
    #             # print(actions)
    #
    #             actions[arm] = a
    #
    #
    #     return actions


    def act_q(self, obs):
        with torch.no_grad():
            max_q = -10e10
            action = 0
            for ind,row in enumerate(np.eye(self.act_dim)):
                x = torch.as_tensor(np.concatenate([obs, row]), dtype=torch.float32)
                q = self.q(x)
                if q >= max_q:
                    max_q = q
                    action = ind
        #         print(ind, q)
        #     print()
        # print(action)
        return action

    def featurize_tp(self, transition_probs, transformation=None, out_dim=6, in_dim=6):
        N = transition_probs.shape[0]
        output_features = np.zeros((N, out_dim))
        np.random.seed(0)  # Set random seed for reproducibility

        if transformation == "linear":
            transformation_matrix = np.random.rand(in_dim, out_dim)
            output_features = np.dot(transition_probs, transformation_matrix)
        elif transformation == "nonlinear":
            transformation_matrix = np.random.rand(in_dim, out_dim)
            output_features = 1 / (1 + np.exp(-np.dot(transition_probs, transformation_matrix)))
        else:
            output_features[:, :min(in_dim, out_dim)] = transition_probs[:, :min(in_dim, out_dim)]
        return output_features
