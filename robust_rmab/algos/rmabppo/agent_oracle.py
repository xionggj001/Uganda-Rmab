
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import torch
from torch.optim import Adam, SGD
import time
import robust_rmab.algos.rmabppo.rmabppo_core as core
from robust_rmab.utils.logx import EpochLogger
from robust_rmab.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from robust_rmab.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from robust_rmab.environments.bandit_env import RandomBanditEnv, Eng1BanditEnv, RandomBanditResetEnv, CirculantDynamicsEnv, ARMMANEnv
from robust_rmab.environments.bandit_env_robust import ToyRobustEnv, ARMMANRobustEnv, CounterExampleRobustEnv, SISRobustEnv, ContinuousStateExampleEnv
from robust_rmab.environments.bandit_env_uganda import UgandaEnv
from robust_rmab.environments.bandit_env_mimiciv import MimicivEnv
from robust_rmab.environments.bandit_env_mimiciii import MimiciiiEnv
from torch.optim.lr_scheduler import ExponentialLR, StepLR


class RMABPPO_Buffer:
    """
    A buffer for storing trajectories experienced by a RMABPPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, tp_feats, act_dim, N, act_type, size, gamma=0.99, lam_OTHER=0.95):
        self.N = N
        self.obs_dim = obs_dim
        self.transition_probs_buf = np.zeros(core.combined_shape(size, (N, tp_feats)), dtype=np.float32)
        self.opt_in_buf = np.ones(core.combined_shape(size, N), dtype=np.float32)

        self.obs_buf = np.zeros(core.combined_shape(size, (N, obs_dim)), dtype=np.float32)
        
        self.act_buf = np.zeros((size, N), dtype=np.float32)
        self.oha_buf = np.zeros(core.combined_shape(size, (N, act_dim)), dtype=np.float32)

        self.adv_buf = np.zeros((size,N), dtype=np.float32)
        self.rew_buf = np.zeros((size,N), dtype=np.float32)
        self.cost_buf = np.zeros((size,N), dtype=np.float32)
        self.ret_buf = np.zeros((size,N), dtype=np.float32)
        self.val_buf = np.zeros((size,N), dtype=np.float32)
        self.q_buf   = np.zeros((size,N), dtype=np.float32)
        self.logp_buf = np.zeros((size,N), dtype=np.float32)
        self.cdcost_buf = np.zeros(size, dtype=np.float32)
        self.lamb_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam_OTHER = gamma, lam_OTHER
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.arm_opt_in_step = np.zeros(N, dtype=int)
        self.arm_opt_out_step = np.ones(N, dtype=int) * 100
        self.act_type = act_type
        self.act_dim = act_dim


    def store(self, obs, transition_probs, opt_in, act, rew, cost, val, lamb, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.transition_probs_buf[self.ptr] = transition_probs
        self.opt_in_buf[self.ptr] = opt_in

        self.act_buf[self.ptr] = act
        oha = np.zeros((self.N, self.act_dim))
        for i in range(self.N):
            oha[i, int(act[i])] = 1
        self.oha_buf[self.ptr] = oha

        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.val_buf[self.ptr] = val
        self.lamb_buf[self.ptr] = lamb
        self.logp_buf[self.ptr] = logp
        self.ptr += 1


    # TODO: implement last costs rollout if we create a training procedure that 
    # uses every step as a sample to update lambda
    # for now, we only use the first sample of epoch to update lambda, so "future" costs don't matter much
    def finish_path(self, last_vals=0, last_costs=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        arm_summed_costs = np.zeros(self.ptr - self.path_start_idx + 1)

        for i in range(self.N):
            # adjust slice according to arm's opt-in and opt-out step
            path_slice = slice(self.arm_opt_in_step[i], self.arm_opt_out_step[i])
            arm_summed_costs = np.zeros(self.arm_opt_out_step[i] - self.arm_opt_in_step[i] + 1)

            rews = np.append(self.rew_buf[path_slice, i], last_vals[i])
            costs = np.append(self.cost_buf[path_slice, i], 0)
            # print(costs)
            # lambds = np.append(self.lamb_buf[path_slice], 0)

            arm_summed_costs += costs
            # adjust based on action costs

            # rews = rews - lambds*costs

            vals = np.append(self.val_buf[path_slice, i], last_vals[i])
            
            # the next two lines implement GAE-Lambda advantage calculation
            qs = rews[:-1] + self.gamma * vals[1:] # gamma is the beta in the paper
            deltas = qs - vals[:-1]
            self.adv_buf[path_slice, i] = core.discount_cumsum(deltas, self.gamma * self.lam_OTHER)
            
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice, i] = core.discount_cumsum(rews, self.gamma)[:-1]

            # store the learned q functions
            self.q_buf[path_slice, i]   = qs
            
            # self.path_start_idx = self.ptr

        # the next line computes costs-to-go, to be part of the loss for the lambda net
        self.cdcost_buf[path_slice] = core.discount_cumsum(arm_summed_costs, self.gamma)[:-1]




    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        for i in range(self.N):
            # the next two lines implement the advantage normalization trick
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[:, i])
            self.adv_buf[:, i] = (self.adv_buf[:, i] - adv_mean) / adv_std
        
        
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                adv=self.adv_buf, logp=self.logp_buf, qs=self.q_buf, oha=self.oha_buf, 
                costs=self.cdcost_buf, lambdas=self.lamb_buf,
                transition_probs=self.transition_probs_buf,
                opt_in_step=self.arm_opt_in_step, opt_out_step=self.arm_opt_out_step)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class AgentOracle:

    def __init__(self, data, N, S, A, B, seed, REWARD_BOUND, agent_kwargs=dict(),
        home_dir="", exp_name="", sampled_nature_parameter_ranges=None, robust_keyword="",
        pop_size=0, opt_in_rate=None, scheduler_discount=0.99999):

        self.data = data
        self.home_dir = home_dir
        self.exp_name = exp_name
        self.REWARD_BOUND = REWARD_BOUND
        self.N = N
        self.S = S
        self.A = A
        self.B = B
        self.seed=seed
        self.sampled_nature_parameter_ranges = sampled_nature_parameter_ranges
        self.robust_keyword = robust_keyword
        self.opt_in_rate = opt_in_rate
        self.scheduler_discount = scheduler_discount

        self.pop_size = pop_size

        if data == 'continuous_state':
            self.env_fn = lambda: ContinuousStateExampleEnv(N, B, seed)

        if data == 'uganda':
            self.env_fn = lambda: UgandaEnv(N, B, seed)

        if data == 'mimiciv':
            self.env_fn = lambda: MimicivEnv(N, B, seed)

        if data == 'mimiciii':
            self.env_fn = lambda: MimiciiiEnv(N, B, seed)

        self.state_norm = 1
        self.actor_critic=core.MLPActorCriticRMAB
        self.agent_kwargs=agent_kwargs

        self.strat_ind = 0

        # this won't work if we go back to MPI, but doing it now to simplify seeding
        self.env = self.env_fn()
        self.env.seed(seed)
        # self.env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges


    # Todo - figure out parallelization with MPI -- not clear how to do this yet, so restrict to single cpu
    def best_response(self, nature_strats, nature_eq, add_to_seed):

        self.strat_ind += 1

        # mpi_fork(args.cpu, is_cannon=args.cannon)  # run parallel code with mpi

        from robust_rmab.utils.run_utils import setup_logger_kwargs

        exp_name = '%s_n%ib%.1fd%sr%sp%s'%(self.exp_name, self.N, self.B, self.data, self.robust_keyword, self.pop_size)
        data_dir = os.path.join(self.home_dir, 'data')
        logger_kwargs = setup_logger_kwargs(exp_name, self.seed, data_dir=data_dir)
        # logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed+add_to_seed, data_dir=data_dir)

        return self.best_response_per_cpu(nature_strats, nature_eq, add_to_seed, seed=self.seed,  logger_kwargs=logger_kwargs, **self.agent_kwargs)

    # add_to_seed is obsolete
    def best_response_per_cpu(self, nature_strats, nature_eq, add_to_seed, actor_critic=core.MLPActorCriticRMAB, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
            vf_lr=1e-3, qf_lr=1e-3, lm_lr=5e-2, train_pi_iters=80, train_v_iters=80, train_q_iters=80,
            lam_OTHER=0.97,
            start_entropy_coeff=0.0, end_entropy_coeff=0.0,
            max_ep_len=1000,
            target_kl=0.01, logger_kwargs=dict(), save_freq=10,
            lamb_update_freq=4,
            init_lambda_trains=0,
            final_train_lambdas=0,
            tp_transform=None):

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        # Instantiate environment
        # env = self.env_fn()
        env = self.env
        
        # env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges
        obs_dim = env.observation_dimension

        # set input dim size given transformation selected 
        # ac_kwargs["input_feat_dim"] refers to how many features generated from input tps
        if tp_transform is None or tp_transform=="None" or ac_kwargs["input_feat_dim"]==None: 
            print("Defaulting to ground truth transition prob. inputs")
            ac_kwargs["input_feat_dim"] = 4
            tp_transform = None 
        else: 
            print("[tp->feats] Applying {} with dim {}".format(tp_transform,ac_kwargs["input_feat_dim"]))


        # choose whether we want to use feature information in learning policy
        self.use_feature = 0
        # choose the amount of steps an arm stays in the system after opt-in
        # (this is different from ac.max_device_usage, which is the max amount of steps an arm can get the device)
        self.max_opt_in_steps = 50

        # Create actor-critic module
        ac = actor_critic(obs_dim, env.action_space, opt_in_rate=self.opt_in_rate,
            N = env.N, C = env.C, B = env.B, strat_ind=self.strat_ind, use_feature=self.use_feature, **ac_kwargs)

        act_dim = ac.act_dim
        obs_dim = ac.obs_dim

        # Sync params across processes
        sync_params(ac)

        # Set up experience buffer
        local_steps_per_epoch = int(steps_per_epoch / num_procs())

        buf = RMABPPO_Buffer(obs_dim, ac_kwargs["input_feat_dim"], act_dim, env.N, ac.act_type, local_steps_per_epoch,
                        gamma=gamma, lam_OTHER=lam_OTHER)

        FINAL_TRAIN_LAMBDAS = final_train_lambdas

        # `compute_loss_pi` function will need the optimizer
        pi_optimizer = Adam(ac.pi_list.parameters(), lr=pi_lr)
        vf_optimizer = Adam(ac.v_list.parameters(), lr=vf_lr)
        qf_optimizer = Adam(ac.q_list.parameters(), lr=qf_lr)
        lambda_optimizer = SGD(ac.lambda_net.parameters(), lr=lm_lr)
        scheduler_lm = ExponentialLR(lambda_optimizer, gamma=self.scheduler_discount) # 0.95 works. try 0.96
        # scheduler_lm = StepLR(lambda_optimizer, step_size=20, gamma=0.05)

        # Set up model saving
        logger.setup_pytorch_saver(ac)


        def featurize_tp(transition_probs, transformation=None, out_dim=4, in_dim=4):
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

        # Set up function for computing RMABPPO policy loss
        def compute_loss_pi(data, entropy_coeff):
            act, adv, logp_old, lambdas, obs, transition_probs, opt_in_step, opt_out_step = \
                data['act'], data['adv'], data['logp'], data['lambdas'], \
                data['obs'], data['transition_probs'], data['opt_in_step'], data['opt_out_step']
            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            full_obs = None
            # this line below may not be necessary, if the transition_probs are stored as float32
            # transition_probs_tensor =  torch.from_numpy(transition_probs).float()
            obs = obs/self.state_norm
            full_obs = obs
            if self.use_feature == 1:
                full_obs = torch.cat([obs, lamb_to_concat, transition_probs], axis=2)

            loss_pi_list = np.zeros(env.N,dtype=object)
            pi_info_list = np.zeros(env.N,dtype=object)

            # Policy loss
            for i in range(env.N):
                # if sum(act[:,i]) == 0:
                #     continue # all actions are zeros would cause an error in ac.pi_list computation
                # calculate when this arm was released
                if opt_out_step[i] - opt_out_step[i] <= 5:
                    continue # these arms are newly opt-in. don't have useful information from them
                pi_optimizer.zero_grad()

                path_slice = slice(int(opt_in_step[i]), int(opt_out_step[i]))
                pi, logp = ac.pi_list(full_obs[path_slice, i], act[path_slice, i])
                # pi, logp = ac.pi_list(full_obs[:, i], act[:, i])
                ent = pi.entropy().mean()
                ratio = torch.exp(logp - logp_old[path_slice, i])
                clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv[path_slice, i]
                loss_pi = -(torch.min(ratio * adv[path_slice, i], clip_adv)).mean()
                
                # subtract entropy term since we want to encourage it 
                loss_pi -= entropy_coeff*ent
                loss_pi_list[i] = loss_pi

                # Useful extra info
                approx_kl = (logp_old[path_slice, i] - logp).mean().item()
                
                clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                pi_info = dict(kl=approx_kl, ent=ent.item(), cf=clipfrac)
                pi_info_list[i] = pi_info

                # backprop
                loss_pi.backward() # another option is to do backprop after a batch
                pi_optimizer.step()

            return loss_pi_list, pi_info_list

        # Set up function for computing value loss
        def compute_loss_v(data):
            ret, lambdas, obs, transition_probs, opt_in_step, opt_out_step = \
                data['ret'], data['lambdas'], data['obs'], \
                data['transition_probs'], data['opt_in_step'], data['opt_out_step']
            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            full_obs = None
            # transition_probs_tensor = torch.from_numpy(transition_probs_tensor).float()

            obs = obs/self.state_norm
            full_obs = obs
            if self.use_feature == 1:
                full_obs = torch.cat([obs, lamb_to_concat, transition_probs], axis=2)

            loss_list = np.zeros(env.N,dtype=object)
            for i in range(env.N):
                if opt_out_step[i] - opt_in_step[i] <= 5:
                    continue # these arms are newly opt-in. don't have useful information from them

                vf_optimizer.zero_grad()
                path_slice = slice(int(opt_in_step[i]), int(opt_out_step[i]))
                loss_list[i] = ((ac.v_list(full_obs[path_slice, i]) - ret[path_slice, i])**2).mean()
                loss_list[i].backward()
                vf_optimizer.step()
            return loss_list

        # def compute_loss_q(data):
        #     # currently not used
        #     print('compute_loss_q. seems this function is unused')
        #
        #     qs, oha, lambdas, transition_probs  = \
        #         data['qs'], data['oha'], data['lambdas'], data['transition_probs']
        #     lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
        #     full_obs = None
        #     # transition_probs_tensor = torch.from_numpy(transition_probs_tensor).float()
        #
        #     obs = obs/self.state_norm
        #     full_obs = torch.cat([obs, lamb_to_concat, transition_probs], axis=2)
        #
        #     loss_list = np.zeros(env.N,dtype=object)
        #     for i in range(env.N):
        #         x = torch.as_tensor(np.concatenate([full_obs[:, i], oha[:, i]], axis=1), dtype=torch.float32)
        #         loss_list[i] = ((ac.q_list(x) - qs[:, i])**2).mean()
        #     return loss_list

        def compute_loss_lambda(data):

            disc_cost = data['costs'][0]
            # lamb = data['lambdas'][0]
            obs = data['obs'][0]
            obs = obs/self.state_norm
            lambda_net_input = np.concatenate((obs, ac.feature_arr.flatten()))
            lamb = ac.lambda_net(torch.as_tensor(lambda_net_input, dtype=torch.float32))
            # lamb = ac.lambda_net(torch.as_tensor(obs,dtype=torch.float32))
            # print('lamb',lamb, 'term 1', env.B/(1-gamma), 'cost',disc_cost, 'diff', env.B/(1-gamma) - disc_cost)
            # print('term 1', , 'cost',disc_cost)
            # print('term 1',env.B/(1-gamma))
            # print('cost',disc_cost)

            loss = lamb*(env.B/(1-gamma) - disc_cost)
            # print(loss)

            return loss

        def update(epoch, head_entropy_coeff):
            data = buf.get()

            entropy_coeff = 0.0
            if (epochs - epoch) > FINAL_TRAIN_LAMBDAS:
                # cool entropy down as we relearn policy for each lambda
                entropy_coeff_schedule = np.linspace(head_entropy_coeff,0,lamb_update_freq)
                # don't rotate
                # entropy_coeff_schedule = entropy_coeff_schedule[1:] + entropy_coeff_schedule[:1]
                ind = epoch%lamb_update_freq
                entropy_coeff = entropy_coeff_schedule[ind]
            # print('entropy',entropy_coeff)

            # Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):
                # pi_optimizer.zero_grad()
                loss_pi_list, pi_info_list = compute_loss_pi(data, entropy_coeff)
                # loss_pi.backward() # moved inside the function compute_loss_pi
                # mpi_avg_grads(ac.pi_list)    # average grads across MPI processes
                # pi_optimizer.step()

            logger.store(StopIter=i)

            # Value function learning
            for i in range(train_v_iters):
                # vf_optimizer.zero_grad()
                loss_v_list = compute_loss_v(data)
                # loss_v.backward() # moved inside the function compute_loss_v
                # mpi_avg_grads(ac.v_list)    # average grads across MPI processes
                # vf_optimizer.step()


            # Lambda optimization
            # if epoch%lamb_update_freq == 0 and epoch > 0:
            # if epoch%lamb_update_freq == 0 and epoch > 0 and (epochs - epoch) > FINAL_TRAIN_LAMBDAS:

                # lambda_optimizer.zero_grad()
                # loss_lamb = compute_loss_lambda(data)
                #
                # loss_lamb.backward()
                # last_param = list(ac.lambda_net.parameters())[-1]
                # lambda_optimizer.step()
                # scheduler_lm.step()

                # update the opt-in decisions, which will stay the same until the next time we update lambda net
                # new_arms_indices = ac.update_opt_in()
                # env.update_transition_probs(new_arms_indices)


        # Prepare for interaction with environment
        start_time = time.time()
        current_lamb = 0

        o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
        # o = o.reshape(-1)

        INIT_LAMBDA_TRAINS = init_lambda_trains

        # # Initialize lambda to make large predictions
        # for i in range(INIT_LAMBDA_TRAINS):
        #     init_lambda_optimizer = SGD(ac.lambda_net.parameters(), lr=lm_lr)
        #     init_lambda_optimizer.zero_grad()
        #     loss_lamb = ac.return_large_lambda_loss(o.reshape(-1), gamma)
        #
        #     loss_lamb.backward()
        #     last_param = list(ac.lambda_net.parameters())[-1]
        #
        #     # mpi_avg_grads(ac.lambda_net)    # average grads across MPI processes
        #     init_lambda_optimizer.step()

        env.update_transition_probs(np.ones(env.N)) # initialize all transition probs
        # new_arms_indices = ac.update_opt_in()
        # env.update_transition_probs(new_arms_indices)

        # Main loop: collect experience in env and update/log each epoch
        head_entropy_coeff_schedule = np.linspace(start_entropy_coeff, end_entropy_coeff, epochs)
        for epoch in range(epochs):
            env.update_transition_probs(np.ones(env.N)) # resample arms every epoch
            # print("start state",o)
            current_lamb = 0
            with torch.no_grad():
                # this is the version where we only predict lambda once at the top of the epoch...
                if self.data == 'uganda' or self.data == 'mimiciv' or self.data == 'mimiciii':
                    T_matrix = env.features
                if self.data == 'continuous_state':
                    T_matrix = env.model_input_T if hasattr(env, 'model_input_T') else env.T
                    T_matrix = T_matrix[:, :, :, 1:] # since probabilities sum up to 1, can reduce the dim of the last axis by 1
                    T_matrix = np.reshape(T_matrix, (T_matrix.shape[0], np.prod(T_matrix.shape[1:])))
                # featurization
                feature_input_dim = T_matrix.shape[1]
                ac.feature_arr = featurize_tp(T_matrix, transformation=tp_transform, out_dim=ac_kwargs["input_feat_dim"], in_dim=feature_input_dim)


            logger.store(Lamb=current_lamb)


            # # Resample nature policy every time we update lambda
            # if epoch%lamb_update_freq == 0 and epoch > 0:
            #     nature_pol = np.random.choice(nature_strats,p=nature_eq)
            #     # get transition probs from this nature policy

            ac.arm_device_removed = np.zeros(N) # reset tracker (whether we remove the device from an arm)
            ac.arm_device_usage = np.zeros(N) # reset tracker (how many steps has am arm used the device)
            ac.opt_in = np.ones(N) # reset opt-in status
            ac.opt_in_steps = np.zeros(N) # reset tracker (the amount of steps each opt-in arm stays in the system)
            buf.arm_opt_in_step = np.zeros(N, dtype=int) * local_steps_per_epoch # important to initialize as zeros, since first B arms opt-in at step 0

            ac.opt_in[int(env.B):] *= 0 # block all arms except for first B arms
            for t in range(local_steps_per_epoch):
                if t % 5 == 0 and t > 1:
                    release_index = int(env.B + self.opt_in_rate * ((t // 5) - 1))
                    ac.opt_in[release_index:release_index + int(self.opt_in_rate)] = 1 # release a blocked arm
                    buf.arm_opt_in_step[release_index:release_index + int(self.opt_in_rate)] = int(t)
                    new_opt_in_arms = np.zeros(N, dtype=int)
                    new_opt_in_arms[release_index:release_index + int(self.opt_in_rate)] = 1
                    env.update_transition_probs(new_opt_in_arms) # freshly sample the arms that are newly opt in
                ac.opt_in_steps[ac.opt_in > 0.5] += 1 # update the amount of steps each opt-in arm stays in the system
                ac.opt_in[ac.opt_in_steps >= self.max_opt_in_steps] = 0 # block arms that are in the system for more than allowed opt-in steps

                # print('step', t)
                # print('opt-in', ac.opt_in)
                # print('opt-in-steps', ac.opt_in_steps)
                # print('device removed', ac.arm_device_removed)
                # print('step', t, 'usage', ac.arm_device_usage)

                torch_o = torch.as_tensor(o, dtype=torch.float32)
                assert torch_o.shape[0] == env.N and torch_o.shape[1] == env.observation_dimension
                a_agent, v, logp= ac.step(torch_o, current_lamb)
                next_o, r, d, _ = env.step(a_agent, ac.opt_in) # removed a_nature
                # next_o = next_o.reshape(-1)
                
                actual_r = r.sum()
                cost_vec = np.zeros(env.N)
                for i in range(env.N):
                    cost_vec[i] = env.C[a_agent[i]]

                lamb_adjusted_r = r.sum() - current_lamb*cost_vec.sum()

                ep_actual_ret += actual_r.sum()
                ep_lamb_adjusted_ret += lamb_adjusted_r
                ep_len += 1

                # save and log
                buf.store(o, ac.feature_arr, ac.opt_in, a_agent, r, cost_vec, v, current_lamb, logp)
                logger.store(VVals=v)
                
                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    FINAL_ROLLOUT_LENGTH = 50
                    if epoch_ended and not(terminal):
                        # print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                        pass
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        # print('lam',current_lamb,'obs:',o,'a',a_agent,'v:',v)

                        # print('lambda lr', scheduler_lm.get_last_lr()[0])
                        # scheduler_lm.step()
                        # print('# arms pulled', sum(a_agent), '# opt-out arms pulled', sum(a_agent * (1 - ac.opt_in)))
                        _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32), current_lamb)

                        # rollout costs for an imagined 50 steps...
                        
                        last_costs = np.zeros((FINAL_ROLLOUT_LENGTH, env.N))
   

                    else:
                        v = 0
                        last_costs = np.zeros((FINAL_ROLLOUT_LENGTH, env.N))

                    # fill in opt-out steps before updating buffer
                    buf.arm_opt_out_step = np.minimum(local_steps_per_epoch, buf.arm_opt_in_step + self.max_opt_in_steps)
                    buf.finish_path(v, last_costs)

                    # only save EpRet / EpLen if trajectory finished
                    # if terminal:
                    logger.store(EpActualRet=ep_actual_ret, EpLambAdjRet=ep_lamb_adjusted_ret, EpLen=ep_len)
                    o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
                    # o = o.reshape(-1)


            # Save model
            if (epoch == epochs-1):
                print("saving")
                # logger.save_state({'env': env}, None)

            # Perform RMABPPO update!
            head_entropy_coeff = head_entropy_coeff_schedule[epoch]
            update(epoch, head_entropy_coeff)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpActualRet', with_min_and_max=True)
            # logger.log_tabular('EpActualRet', average_only=True)
            logger.log_tabular('EpLambAdjRet', with_min_and_max=True)
            # logger.log_tabular('EpLambAdjRet', average_only=True)
            # logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)

            # logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Lamb', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


        # env.update_transition_probs(np.ones(env.N))
        # if self.data == 'uganda' or self.data == 'mimiciv':
        #     T_matrix = env.features
        # if self.data == 'coontinuous_state':
        #     T_matrix = env.model_input_T if hasattr(env, 'model_input_T') else env.T
        #     T_matrix = np.reshape(T_matrix[:, :, :, 1:], (T_matrix[:, :, :, 1:].shape[0], np.prod(T_matrix[:, :, :, 1:].shape[1:])))
        #
        # ac.transition_param_arr = T_matrix
        ac.tp_transform = tp_transform
        ac.out_dim = ac_kwargs["input_feat_dim"]
        ac.feature_input_dim = feature_input_dim
        # ac.feature_arr = featurize_tp(T_matrix, transformation=tp_transform, out_dim=ac_kwargs["input_feat_dim"], in_dim = feature_input_dim)
        print("saving")
        logger.save_state({'env': env}, None)
        return ac


    # def simulate_reward(self, agent_pol, nature_pol=[], seed=0,
    #         steps_per_epoch=100, epochs=100, gamma=0.99):
    #     env = self.env_fn()
    #     # important to make sure these are always the same for all instatiations of the env
    #     # env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges
    #
    #     env.seed(seed)
    #
    #     o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
    #     # o = o.reshape(-1)
    #
    #     rewards = np.zeros((epochs, steps_per_epoch))
    #     for epoch in range(epochs):
    #         # print("epoch",epoch)
    #         ac.arm_device_removed = np.zeros(N) # reset tracker (whether we remove the device from an arm)
    #         ac.arm_device_usage = np.zeros(N) # reset tracker (how many steps has am arm used the device)
    #
    #         for t in range(steps_per_epoch):
    #             torch_o = torch.as_tensor(o, dtype=torch.float32)
    #
    #             a_agent  = agent_pol.act_test(torch_o)
    #             next_o, r, d, _ = env.step(a_agent, ac.opt_in) # removed a_nature
    #             # next_o = next_o.reshape(-1)
    #             actual_r = r.sum()
    #             ep_actual_ret += actual_r
    #             ep_len += 1
    #
    #             rewards[epoch,t] = actual_r*(gamma**t)
    #             o = next_o
    #
    #         o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
    #         # o = o.reshape(-1)
    #
    #
    #
    #     rewards = rewards.sum(axis=1).mean()
    #
    #
    #     return rewards
    #


# python3 spinup/algos/pytorch/ppo/rmab_rl_lambda_ppo.py --hid 64 -l 2 --gamma 0.9 --cpu 1 --step 100 -N 4 -S 2 -A 2 -B 1 --REWARD_BOUND 2 --exp_name rmab_rl_bandit_n4s2a2b1_r2_lambda -s 0 --epochs 1000 --init_lambda_trains 1
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--hid', type=int, default=64, help="Number of units in each layer of the neural networks used for the Oracles")
    parser.add_argument('-l', type=int, default=2, help="Depth of the neural networks used for Agent and Nature Oracles (i.e., layers)")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--seed', '-s', type=int, default=0, help="Seed")
    parser.add_argument('--cpu', type=int, default=1, help="Number of processes for mpi")
    
    parser.add_argument('--exp_name', type=str, default='experiment', help="Experiment name")
    parser.add_argument('-N', type=int, default=5, help="Number of arms")
    parser.add_argument('-S', type=int, default=4, help="Number of states in each arm (when applicable, e.g., SIS)")
    parser.add_argument('-A', type=int, default=2, help="Number of actions in each arm (not currently implemented)")
    parser.add_argument('-B', type=float, default=1.0, help="Budget per round")
    parser.add_argument('--reward_bound', type=int, default=1, help="Rescale rewards to this value (only some environments)")
    parser.add_argument('--save_string', type=str, default="")
    parser.add_argument('--opt_in_rate', type=float, default=1.0, help="Opt-in rate; p of sampled binomial for each arm")

    parser.add_argument('--agent_steps', type=int, default=10, help="Number of rollout steps between epochs")
    parser.add_argument('--agent_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--agent_init_lambda_trains', type=int, default=0, help="Deprecated, leave at 0")
    parser.add_argument('--agent_clip_ratio', type=float, default=2.0, help="Clip ratio for PPO step")
    parser.add_argument('--agent_final_train_lambdas', type=int, default=10, help="Number of epochs at the end of training to update the policy and critic network, but not the lambda-network")
    parser.add_argument('--agent_start_entropy_coeff', type=float, default=0.0, help="Start entropy coefficient for the cooling procedure")
    parser.add_argument('--agent_end_entropy_coeff', type=float, default=0.0, help="End entropy coefficient for the cooling procedure")
    parser.add_argument('--agent_pi_lr', type=float, default=2e-3, help="Learning rate for policy network")
    parser.add_argument('--agent_vf_lr', type=float, default=2e-3, help="Learning rate for critic network")
    parser.add_argument('--agent_lm_lr', type=float, default=2e-3, help="Learning rate for lambda network") #2e-3
    parser.add_argument('--agent_train_pi_iters', type=int, default=20, help="Training iterations to run per epoch")
    parser.add_argument('--agent_train_vf_iters', type=int, default=20, help="Training iterations to run per epoch")
    parser.add_argument('--agent_lamb_update_freq', type=int, default=4, help="Number of epochs that should pass before updating the lambda network (so really it is a period, not frequency)") # 4
    parser.add_argument('--agent_tp_transform', type=str, default=None, help="Type of transform to apply to transition probabilities, if any") 
    parser.add_argument('--agent_tp_transform_dims', type=int, default=6, help="Number of output features to generate from input tps; only used if tp_transform is True")
    parser.add_argument('--pop_size', type=int, default=0)
    parser.add_argument('--scheduler_discount', type=float, default=0.99999) # 0.99999 is effectively removing scheduler

    parser.add_argument('--home_dir', type=str, default='.', help="Home directory for experiments")
    parser.add_argument('--cannon', type=int, default=0, help="Flag used for running experiments on batched slurm-based HPC resources. Leave at 0 for small experiments.")
    parser.add_argument('-d', '--data', default='continuous_state', type=str, help='Environment selection',
                        choices=[   'continuous_state',
                                    'uganda',
                                    'mimiciv',
                                    'mimiciii'
                                ])

    parser.add_argument('--robust_keyword', default='pess', type=str, help='Method for picking some T out of the uncertain environment',
                        choices=[   
                                    'pess',
                                    'mid',
                                    'opt', # i.e., optimistic
                                    'sample_random'
                                ])

    args = parser.parse_args()

    mpi_fork(args.cpu, is_cannon=args.cannon)  # run parallel code with mpi

    # from spinup.utils.run_utils import setup_logger_kwargs

    # exp_name = '%s_n%is%ia%ib%.2fr%.2f'%(args.exp_name, args.N, args.S, args.A, args.B, args.REWARD_BOUND)
    # print(exp_name)
    # data_dir = os.path.join(args.home_dir, 'data')
    # logger_kwargs = setup_logger_kwargs(exp_name, args.seed, data_dir=data_dir)

    # N = args.N
    N = int(args.opt_in_rate * (args.agent_steps / 5 - 1) + args.B) # every 5 steps, we have opt_in_rate many arms opt-in
    print('N', N)
    S = args.S
    A = args.A
    B = args.B
    budget = B
    reward_bound = args.reward_bound
    seed=args.seed
    data = args.data
    home_dir = args.home_dir
    exp_name=args.exp_name
    gamma = args.gamma

    opt_in_rate = args.opt_in_rate
    assert opt_in_rate <= B # else the constraint that we must give newly opt-in arm device cannot be satisfied
    scheduler_discount = args.scheduler_discount

    torch.manual_seed(seed)
    np.random.seed(seed)

    agent_kwargs = {}
    agent_kwargs['steps_per_epoch'] = args.agent_steps
    agent_kwargs['epochs'] = args.agent_epochs
    agent_kwargs['init_lambda_trains'] = args.agent_init_lambda_trains
    agent_kwargs['clip_ratio'] = args.agent_clip_ratio
    agent_kwargs['final_train_lambdas'] = args.agent_final_train_lambdas
    agent_kwargs['start_entropy_coeff'] = args.agent_start_entropy_coeff
    agent_kwargs['end_entropy_coeff'] = args.agent_end_entropy_coeff
    agent_kwargs['pi_lr'] = args.agent_pi_lr
    agent_kwargs['vf_lr'] = args.agent_vf_lr
    agent_kwargs['lm_lr'] = args.agent_lm_lr
    agent_kwargs['train_pi_iters'] = args.agent_train_pi_iters
    agent_kwargs['train_v_iters'] = args.agent_train_vf_iters
    agent_kwargs['tp_transform'] = args.agent_tp_transform
    agent_kwargs['ac_kwargs'] = dict(hidden_sizes=[args.hid]*args.l,
                                     input_feat_dim=args.agent_tp_transform_dims)
    agent_kwargs['gamma'] = args.gamma
    agent_kwargs['lamb_update_freq'] = args.agent_lamb_update_freq

    env_fn = None

    state_norm = None

    # if args.data == 'continuous_state':
    #     env_fn = lambda : ContinuousStateExampleEnv(N,B,seed)
    #
    # if args.data == 'uganda':
    #     env_fn = lambda : UgandaEnv(N,B,seed)
    #
    #
    #
    # env = env_fn()

    agent_oracle  = AgentOracle(data, N, S, A, budget, seed, reward_bound,
                             agent_kwargs=agent_kwargs, home_dir=home_dir, exp_name=exp_name,
                             robust_keyword=args.robust_keyword,
                             # sampled_nature_parameter_ranges = sampled_nature_parameter_ranges,
                             pop_size=args.pop_size, opt_in_rate=opt_in_rate,
                                scheduler_discount=scheduler_discount)

    nature_strategy = None
    # if args.robust_keyword == 'mid':
    #     nature_strategy = MiddleNaturePolicy(sampled_nature_parameter_ranges, 0)
    #
    # if args.robust_keyword == 'sample_random':
    #     nature_strategy = SampledRandomNaturePolicy(sampled_nature_parameter_ranges, 0)
    #
    #     # init the random strategy
    #     nature_strategy.sample_param_setting(seed)



    add_to_seed = 0
    agent_oracle.best_response([nature_strategy], [1.0], add_to_seed)




