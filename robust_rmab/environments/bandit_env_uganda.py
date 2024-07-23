import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp
import seaborn as sns
import math
import pandas as pd
import gym
import os
import random


#Hyperparameters for 3 settings
setting="uganda"
min_entries=10
time_size=60
# vital_signs = ['COVERED_SKIN_TEMPERATURE', 'PULSE_RATE',  'RESPIRATORY_RATE','SPO2']
vital_signs = ['COVERED_SKIN_TEMPERATURE', 'PULSE_RATE',  'RESPIRATORY_RATE']
num_comp=5
sample_size=-1
num_timesteps=1

'''
setting="mimiciii"
min_entries=10 # Only keep icu stays with 10 measurements in two consecutive hours
time_size=60 # Summarized time windoes
sample_size=5000
vital_signs = ['PULSE_RATE',  'RESPIRATORY_RATE','SPO2'] #'COVERED_SKIN_TEMPERATURE' is available but too sparse
num_comp=5 #Number of components of the Gaussian THIS CAN BE FREELY ADAPTED
num_timesteps=1
'''

'''
setting="mimiciv"
min_entries=10
time_size=60
vital_signs = ['PULSE_RATE',  'RESPIRATORY_RATE','COVERED_SKIN_TEMPERATURE'] #'BLOOD_PRESSURE'
num_comp=5
sample_size=-1
num_timesteps=1
'''

degree_of_arm_noise=0.15

intervention_success_rate=0.7

variability_window=5

vital_alarm = pd.read_csv(setting+'.csv')

if sample_size!=-1:
  all_patient_ids = vital_alarm['patient_id'].unique()
  np.random.seed(42)
  random_patient_ids = np.random.choice(all_patient_ids, size=sample_size, replace=False)
  restricted_df = vital_alarm[vital_alarm['patient_id'].isin(random_patient_ids)]
  vital_alarm=restricted_df

#HELPER
def create_training_dataset3(pivot_df, num_timesteps, time_size, min_entries=1):
    input_data = []
    output_data = []
    patient_entry_counts = {}  # Dictionary to store the count of valid sequences for each patient_id
    valid_patient_ids = set()  # Set to store patient_ids that meet the min_entries threshold
    patient_rewards = {}

    # Group by 'patient_id' and process each group
    for patient_id, group in pivot_df.groupby('patient_id'):
        group = group.sort_values('generatedat')
        valid_sequences = []  # Temporary list to hold valid sequences for the current patient_id
        rewards = []

        # Iterate through the group to find valid sequences
        for i in range(len(group) - num_timesteps):
            valid_sequence = True
            base_time = group.iloc[i]['generatedat']
            for j in range(1, num_timesteps + 1):
                expected_time = base_time + pd.Timedelta(minutes=time_size * j)
                if group.iloc[i + j]['generatedat'] != expected_time:
                    valid_sequence = False
                    break
            if valid_sequence:
                input_values = group.iloc[i:i + num_timesteps][vital_signs].values.flatten()
                output_values = group.iloc[i + num_timesteps][vital_signs].values
                valid_sequences.append((input_values.astype(float), output_values.astype(float)))
                rewards.append(group.iloc[i + num_timesteps]['reward'])


        patient_rewards[patient_id] = np.mean(rewards)
        # Only add to final dataset if valid_sequences count meets the threshold
        if len(valid_sequences) > min_entries:
            valid_patient_ids.add(patient_id)
            for input_values, output_values in valid_sequences:
                input_data.append(input_values)
                output_data.append(output_values)

    return np.hstack((np.array(input_data), np.array(output_data))), list(valid_patient_ids)

def conditional_sample_gmm(gmm, given_values, given_indices,component_index=None):
    """
    Sample from the conditional distribution of a Gaussian Mixture Model.

    Parameters:
    - gmm: Fitted GaussianMixture object
    - given_values: The values of the given variables
    - given_indices: The indices of the given variables

    Returns:
    - Sample from the conditional distribution
    """
    all_indices = np.arange(gmm.means_.shape[1])
    remaining_indices = np.setdiff1d(all_indices, given_indices)

    # Extract the means and covariances of the components
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_


    # Convert to DataFrame
    df = pd.DataFrame(covariances[0])

    # Print nicely formatted
    #print(df.to_string(index=False, float_format="%.8f"))
    #print(means,weights)
    # Calculate conditional means and covariances for each component
    if not component_index is None:
        mean_given = means[component_index, given_indices]
        mean_remaining = means[component_index, remaining_indices]
        cov_given_given = covariances[component_index][np.ix_(given_indices, given_indices)]
        cov_remaining_given = covariances[component_index][np.ix_(remaining_indices, given_indices)]
        cov_given_remaining = covariances[component_index][np.ix_(given_indices, remaining_indices)]
        cov_remaining_remaining = covariances[component_index][np.ix_(remaining_indices, remaining_indices)]
        #print("means",mean_given,mean_remaining)
        #print("covariates",cov_given_given,cov_remaining_given,cov_given_remaining,cov_remaining_remaining)

        cov_inv_given_given = np.linalg.inv(cov_given_given)
        conditional_mean = mean_remaining + cov_remaining_given @ cov_inv_given_given @ (given_values - mean_given)
        conditional_cov = cov_remaining_remaining - cov_remaining_given @ cov_inv_given_given @ cov_given_remaining

        return np.clip(np.random.multivariate_normal(mean=conditional_mean, cov=conditional_cov),0,1)
    else:
      conditional_means = []
      conditional_covs = []
      for k in range(gmm.n_components):
          mean_given = means[k, given_indices]
          mean_remaining = means[k, remaining_indices]
          cov_given_given = covariances[k][np.ix_(given_indices, given_indices)]
          cov_remaining_given = covariances[k][np.ix_(remaining_indices, given_indices)]
          cov_given_remaining = covariances[k][np.ix_(given_indices, remaining_indices)]
          cov_remaining_remaining = covariances[k][np.ix_(remaining_indices, remaining_indices)]
          #print("means",mean_given,mean_remaining)
          #print("covariates",cov_given_given,cov_remaining_given,cov_given_remaining,cov_remaining_remaining)

          cov_inv_given_given = np.linalg.inv(cov_given_given)
          conditional_mean = mean_remaining + cov_remaining_given @ cov_inv_given_given @ (given_values - mean_given)
          conditional_cov = cov_remaining_remaining - cov_remaining_given @ cov_inv_given_given @ cov_given_remaining

          conditional_means.append(conditional_mean)
          conditional_covs.append(conditional_cov)

      # Sample from the conditional distribution of each component
      component_samples = [
          np.random.multivariate_normal(mean=conditional_means[k], cov=conditional_covs[k])
          for k in range(gmm.n_components)
      ]

      # Sample a component based on the weights
      component = np.random.choice(gmm.n_components, p=weights)

    return np.clip(component_samples[component],0,1)



def conditional_sample_mnd(given_values, given_indices, mean, cov):
    """
    Sample from the conditional distribution of a Gaussian Mixture Model.

    Parameters:
    - gmm: Fitted GaussianMixture object
    - given_values: The values of the given variables
    - given_indices: The indices of the given variables

    Returns:
    - Sample from the conditional distribution
    """
    all_indices = np.arange(len(mean))
    remaining_indices = np.setdiff1d(all_indices, given_indices)


    # Convert to DataFrame
    df = pd.DataFrame(cov[0])

    # Print nicely formatted
    #print(df.to_string(index=False, float_format="%.8f"))
    #print(means,weights)
    # Calculate conditional means and covariances for each component
    mean_given = mean[given_indices]
    mean_remaining = mean[remaining_indices]
    cov_given_given = cov[np.ix_(given_indices, given_indices)]
    cov_remaining_given = cov[np.ix_(remaining_indices, given_indices)]
    cov_given_remaining = cov[np.ix_(given_indices, remaining_indices)]
    cov_remaining_remaining = cov[np.ix_(remaining_indices, remaining_indices)]
    #print("means",mean_given,mean_remaining)
    #print("covariates",cov_given_given,cov_remaining_given,cov_given_remaining,cov_remaining_remaining)

    cov_inv_given_given = np.linalg.inv(cov_given_given)
    conditional_mean = mean_remaining + cov_remaining_given @ cov_inv_given_given @ (given_values - mean_given)
    conditional_cov = cov_remaining_remaining - cov_remaining_given @ cov_inv_given_given @ cov_given_remaining

    return np.clip(np.random.multivariate_normal(mean=conditional_mean, cov=conditional_cov),0,1)



# Function to apply the conditional_sample_gmm row-wise
def apply_conditional_sample_gmm(row, gmm, given_indices):
    y_test = []
    for row in X_test:
        given_values = row[given_indices]
        y_test.append(conditional_sample_gmm(gmm, given_values, given_indices))
    return np.array(y_test)

def manual_normalization(x,p_min,p_max):
  return (x-p_min)/(p_max-p_min)

def min_max_normalize(column):
    #print(column.min(),column.max())
    return (column - column.min()) / (column.max() - column.min())

def reverse_min_max_normalize(column, min_val, max_val):
    return column * (max_val - min_val) + min_val

def manual_normalize_data(vital_signs,p_dict,min_max):
  for sign in vital_signs:
    p_dict[sign] = np.clip(manual_normalization(p_dict[sign],min_max[sign][0], min_max[sign][1]),0,1)
  return p_dict

def clean_data(vital_signs,p_df,min_max):
  for sign in vital_signs:
    p_df[sign] = reverse_min_max_normalize(p_df[sign], min_max[sign][0], min_max[sign][1])
  return p_df

def create_model():
  f_name_f=setting+"-"+str(time_size)+"-"+str(min_entries)+"-"+str(sample_size)+"-"+'-'.join(vital_signs)+"_full.csv"
  f_name_p=setting+"-"+str(time_size)+"-"+str(min_entries)+"-"+str(sample_size)+"-"+'-'.join(vital_signs)+".csv"
  if os.path.isfile("./preprocessed_files/"+f_name_f):
    print("File exists")
    df=pd.read_csv("./preprocessed_files/"+f_name_f)
    pivot_df=pd.read_csv("./preprocessed_files/"+f_name_p)
    df['generatedat'] = pd.to_datetime(df['generatedat'])
    pivot_df['generatedat'] = pd.to_datetime(pivot_df['generatedat'])


  else:
    df=vital_alarm


    #df['patient_id'] = df['patient_id'].astype(int)

    df['generatedat'] = pd.to_datetime(df['generatedat'])



    vital_signs_of_interest = vital_signs
    df = df[df['name'].isin(vital_signs_of_interest)]


    def to_float(value):
        try:
            return float(value)
        except ValueError:
            return np.nan
    df.loc[:, 'doublevalue'] = df['doublevalue'].apply(to_float)

    df = df.dropna(subset=['doublevalue','patient_id'])


    condition = (df['name'] == 'PULSE_RATE') & (df['doublevalue'] > 300)
    df = df[~condition]

    condition = (df['name'] == 'RESPIRATORY_RATE') & (df['doublevalue'] > 60)
    df = df[~condition]

    condition = (df['name'] == 'SPO2') & ((df['doublevalue'] < 50) | (df['doublevalue'] > 100))
    df = df[~condition]

    condition = (df['name'] == 'COVERED_SKIN_TEMPERATURE') & (df['doublevalue'] > 45)
    df = df[~condition]





    df['generatedat'] = pd.to_datetime(df['generatedat'], errors='coerce')


    df.set_index('generatedat', inplace=True)

    df.sort_index(inplace=True)


    first_entries = df.groupby('patient_id').head(1).reset_index()





    median_values = df.groupby(['patient_id','name']).resample(str(time_size)+'T')['doublevalue'].median()

    median_values = median_values.reset_index()

    resampled_df = median_values.dropna(subset=['doublevalue'])
    pivot_df = resampled_df.pivot_table(index=['patient_id', 'generatedat'], columns='name', values='doublevalue')
    pivot_df = pivot_df.dropna().reset_index()

    pivot_df['reward'] = pivot_df.apply(lambda row: reward_function(row[vital_signs].to_dict()), axis=1)
    df.to_csv("./preprocessed_files/"+f_name_f)
    pivot_df.to_csv("./preprocessed_files/"+f_name_p)





  # for vital_sign in vital_signs:
  #   sns.histplot(pivot_df[vital_sign], kde=True)
  #   plt.title('Distribution of Column Name')
  #   plt.show()


  min_max={}
  for sign in vital_signs:
      min_max[sign] = [pivot_df[sign].min(),pivot_df[sign].max()]
      pivot_df[sign] = min_max_normalize(pivot_df[sign])

  first_val=['patient_id', 'generatedat', 'reward']
  pivot_df=pivot_df[first_val+vital_signs]
  df_proc = pivot_df[vital_signs]

  idx = pivot_df.groupby('patient_id')['generatedat'].idxmin()

  earliest_entries_df = pivot_df.loc[idx].reset_index(drop=True)


  rev_df = pivot_df.copy()
  clean_data(vital_signs,rev_df,min_max)


  '''
  filtered_df = rev_df[(rev_df['reward'] >= -10) & (rev_df['reward'] < 0)]

  # Plot the filtered data
  sns.histplot(filtered_df['reward'], kde=True)
  plt.title('Distribution of Reward (0 to -100)')
  plt.show()
  '''


  f_name_t=setting+"-"+str(time_size)+"-"+str(min_entries)+"-"+str(sample_size)+"-"+'-'.join(vital_signs)+"_train.npy"
  f_name_v=setting+"-"+str(time_size)+"-"+str(min_entries)+"-"+str(sample_size)+"-"+'-'.join(vital_signs)+"_patients.npy"

  if os.path.isfile("./preprocessed_files/"+f_name_t):
    print("File exists")
    valid_patient_ids=np.load("./preprocessed_files/"+f_name_v)
    combined_training=np.load("./preprocessed_files/"+f_name_t)
  else:
    combined_training,valid_patient_ids = create_training_dataset3(pivot_df, num_timesteps, time_size=time_size, min_entries=min_entries)
    np.save("./preprocessed_files/"+f_name_t, combined_training)
    np.save("./preprocessed_files/"+f_name_v, valid_patient_ids)


  pivot_df=pivot_df[pivot_df['patient_id'].isin(valid_patient_ids)]
  rev_df=rev_df[rev_df['patient_id'].isin(valid_patient_ids)]

  # In this part we intitalize the covariance matrix of the Gaussian. Could be skipped as I am not fully sure how much/whether it helps.
  #x=len(vital_signs)
  #initial_covariances = np.zeros((num_comp, 2*len(vital_signs), 2*len(vital_signs)))
  #for k in range(num_comp):
  #    cov_matrix = np.eye(2*len(vital_signs))  # Identity matrix for variances
  #    for i in range(x):
  #        cov_matrix[i, i + x] = 0.9  # High correlation between ith and (i+x)th features
  #        cov_matrix[i + x, i] = 0.9  # Symmetric entry
  #    initial_covariances[k] = cov_matrix

  #print()


  gmm = GaussianMixture(n_components=num_comp, covariance_type='full', random_state=42)
  #gmm.precisions_init = np.linalg.inv(initial_covariances)


  gmm.fit(combined_training)

  return gmm,min_max

#Exp REWARDS


import math

def temperature_penalty(temperature):
    if temperature <= 38:
        return 0
    else:
        return -math.exp(abs(temperature - 38.0)/2)  # Exponential penalty

def pulse_penalty(pulse):
    if pulse <= 120:
        return 0
    else:
        return -math.exp(abs(pulse - 120) / 17)  # Exponential penalty

def respiratory_penalty(respiratory_rate):
    if respiratory_rate <= 30:
        return 0
    else:
        return -math.exp(abs(respiratory_rate - 30) / 5)  # Exponential penalty

def spo2_penalty(spo2):
    if 90 <= spo2:
        return 0
    else:
        return -math.exp(abs(spo2 - 90) / 4)  # Exponential penalty

def blood_penalty(blood_pressure):
    if blood_pressure <= 127:
        return 0
    else:
        return -math.exp(abs(blood_pressure - 127) / 5)  # Exponential penalty

def reward_function(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)
  reward=0
  for signs in sign_dict:
    if signs=="COVERED_SKIN_TEMPERATURE":
      reward+=temperature_penalty(sign_dict[signs])
    elif signs=="PULSE_RATE":
      reward+=pulse_penalty(sign_dict[signs])
    elif signs=="RESPIRATORY_RATE":
      reward+=respiratory_penalty(sign_dict[signs])
    elif signs=="SPO2":
      reward+=spo2_penalty(sign_dict[signs])
  return reward

def improve_vital_signs(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)

  #print(sign_dict)
  for signs in sign_dict:
    if signs=="COVERED_SKIN_TEMPERATURE":
      if temperature_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-1.5
    elif signs=="PULSE_RATE":
      if pulse_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-15
    elif signs=="RESPIRATORY_RATE":
      if respiratory_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-10
    elif signs=="SPO2":
      if spo2_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]+3
  sign_dict=manual_normalize_data(vital_signs,sign_dict,o_values)
  return sign_dict

def improve_vital_signs1(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)

  #print(sign_dict)
  if random.random() < intervention_success_rate:
    for signs in sign_dict:
      if signs=="COVERED_SKIN_TEMPERATURE":
        if temperature_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-1.5
      elif signs=="PULSE_RATE":
        if pulse_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-15
      elif signs=="RESPIRATORY_RATE":
        if respiratory_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-10
      elif signs=="SPO2":
        if spo2_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]+3
  sign_dict=manual_normalize_data(vital_signs,sign_dict,o_values)
  return sign_dict

def improve_vital_signs2(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)

  #print(sign_dict)
  for signs in sign_dict:
    if signs=="COVERED_SKIN_TEMPERATURE":
      if temperature_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-np.random.normal(1.5, 0.5)
    elif signs=="PULSE_RATE":
      if pulse_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-np.random.normal(15, 5)
    elif signs=="RESPIRATORY_RATE":
      if respiratory_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-np.random.normal(10, 10/3)
    elif signs=="SPO2":
      if spo2_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]+np.random.normal(3, 1)
  sign_dict=manual_normalize_data(vital_signs,sign_dict,o_values)
  return sign_dict

def improve_vital_signs3(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)

  #print(sign_dict)
  if random.random() < intervention_success_rate:
    for signs in sign_dict:
      if signs=="COVERED_SKIN_TEMPERATURE":
        if temperature_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-np.random.normal(1.5, 0.5)
      elif signs=="PULSE_RATE":
        if pulse_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-np.random.normal(15, 5)
      elif signs=="RESPIRATORY_RATE":
        if respiratory_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-np.random.normal(10, 10/3)
      elif signs=="SPO2":
        if spo2_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]+np.random.normal(3, 1)
  sign_dict=manual_normalize_data(vital_signs,sign_dict,o_values)
  return sign_dict

def interventions(current_values,min_max,mean=None,cov=None,given_indices=list(range(len(vital_signs)))):
  if reward_function(dict(zip(vital_signs,current_values)),rev_norm=True,o_values=min_max)>=0:
    return conditional_sample_mnd(current_values, given_indices,mean=mean,cov=cov)
  else:
    #new_signs= conditional_sample_gmm(gmm, current_values, given_indices,component_index=component_index)
    #print("Old", current_values)
    new_signs=improve_vital_signs3(dict(zip(vital_signs,current_values)),rev_norm=True,o_values=min_max)
    #print("NEW",[new_signs[vital] for vital in vital_signs])
    return conditional_sample_mnd([new_signs[vital] for vital in vital_signs], given_indices,mean=mean,cov=cov)
    #return resample_values(gmm,min_max,component_index=component_index)[0]

def simulate_one_step(current_state,min_max,intervention=False,mean=None,cov=None,given_indices=list(range(len(vital_signs)))):
    current_signs=current_state[0]
    signs_history=current_state[2]
    #print(current_signs)

    if intervention:
      next_signs=interventions(current_values=current_signs,min_max=min_max,mean=mean,cov=cov)
    else:
      next_signs=conditional_sample_mnd(current_signs, given_indices,mean=mean,cov=cov)

    for i in range(len(vital_signs)):
      del signs_history[i][0]
      signs_history[i].append(next_signs[i])

    variability=[np.var(l) for l in signs_history]


    reward=reward_function(dict(zip(vital_signs,next_signs)),rev_norm=True,o_values=min_max)
    return [next_signs,variability,signs_history],reward

def sample_agent(gmm,min_max,given_indices):
  weights = gmm.weights_

  # Normalize the weights to ensure they sum to 1
  weights /= np.sum(weights)

  # Sample an index based on the weights
  component = np.random.choice(len(weights), p=weights)

  means = gmm.means_
  covariances = gmm.covariances_
  mean=means[component]
  cov=covariances[component]
  state, _ = resample_values(min_max,given_indices,mean=mean,cov=cov)

  perturb=random.choice([i for i in range(len(weights)) if i != component])

  x=random.uniform(0, degree_of_arm_noise)
  y=random.uniform(0, degree_of_arm_noise)

  mean=(1-x)*mean+x*means[perturb]
  cov=(1-y)*cov+y*covariances[perturb]

  #print(mean,cov)
  #pertubation

  return state,component,mean,cov

def resample_values(min_max,given_indices,mean,cov):
  sample=np.clip(np.random.multivariate_normal(mean=mean,cov=cov),0,1)
  current_signs=[sample[i] for i in given_indices]
  signs_history=[[] for _ in range(len(vital_signs))]
  for i in range(len(vital_signs)):
    signs_history[i].append(sample[i])

  for _ in range(variability_window-1):
    current_signs=conditional_sample_mnd(current_signs, given_indices,mean=mean,cov=cov)
    for i in range(len(vital_signs)):
      signs_history[i].append(current_signs[i])

  #print(signs_history)
  #for l in signs_history:
    #print(l,np.var(l))
  variability=[np.var(l) for l in signs_history]
  #print(variability)
  reward=reward_function(dict(zip(vital_signs,current_signs)),rev_norm=True,o_values=min_max)
  return [current_signs,variability,signs_history],reward





class UgandaEnv(gym.Env):
    def __init__(self, N, B, seed):
        self.N = N
        self.gmm, self.min_max = create_model()
        self.num_vital_signs = len(vital_signs)
        self.features = np.zeros((self.N, 2 * self.num_vital_signs))
        self.arm_component = np.zeros(self.N, dtype=np.int8)
        S = 2 * self.num_vital_signs
        self.given_indices = [j for j in range(len(vital_signs))]
        A = 2
        # initialize arm_component
        for i in range(self.N):
            state, component, mean, cov = sample_agent(self.gmm, self.min_max, self.given_indices)
            self.features[i] = mean
            self.arm_component[i] = component

        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = S
        self.action_dimension = 1
        self.action_dim_nature = N
        self.S = S
        self.A = A
        self.B = B
        self.init_seed = seed

        self.current_full_state = np.zeros((self.N, self.observation_dimension))
        self.current_full_state_raw = [[] for _ in range(self.N)]
        self.random_stream = np.random.RandomState()

        # make sure to set this whenever environment is created, but do it outside so it always the same
        self.sampled_parameter_ranges = None

        self.seed(seed=seed)
        self.C = np.array([0, 1])

    def update_transition_probs(self, arms_to_update, mode='train'):
        # arms_to_update is 1d array of length N. arms_to_update[i] == 1 if transition prob of arm i needs to be resampled
        for i in range(self.N):
            if arms_to_update[i] > 0.5:
                state, component, mean, cov = sample_agent(self.gmm, self.min_max, self.given_indices)
                self.features[i] = mean
                self.arm_component[i] = component

    def random_agent_action(self): # to be updated
        actions = np.zeros(self.N)
        choices = np.random.choice(np.arange(self.N), int(self.B), replace=False)
        actions[choices] = 1
        return actions

    def step(self, a_agent, opt_in, mode="train"):
        ###### Get next state
        next_full_state = np.zeros((self.N, self.observation_dimension), dtype=float)
        next_full_state_raw = [[] for _ in range(self.N)]
        rewards = np.zeros(self.N)
        for i in range(self.N):
            means = self.gmm.means_
            covariances = self.gmm.covariances_
            mean = means[self.arm_component[i]]
            cov = covariances[self.arm_component[i]]
            current_arm_state = self.current_full_state_raw[i]  # want continuous states. not rounded6
            action = int(a_agent[i])
            next_arm_state_raw, reward = simulate_one_step(current_arm_state, self.min_max, intervention=action, mean=mean,
                                                    cov=cov)
            # next_arm_state = np.minimum(1, np.maximum(0, next_arm_state))
            next_full_state_raw[i] = next_arm_state_raw
            next_full_state[i] = np.append(next_full_state_raw[i][0], next_full_state_raw[i][1])
            if opt_in[i] < 0.5:
                next_full_state[i] *= 0  # opt-out arms have dummy state
            rewards[i] = reward

        if mode == "eval":
            rewards[opt_in == 0] = 0 # enforce no reward from opt-out only during test time
        self.current_full_state = next_full_state
        self.current_full_state_raw = next_full_state_raw
        # print('rewards', rewards)

        return next_full_state, rewards, False, None

    def reset_random(self):
        return self.reset()

    def reset(self):
        for i in range(self.N):
            means = self.gmm.means_
            covariances = self.gmm.covariances_
            mean = means[self.arm_component[i]]
            cov = covariances[self.arm_component[i]]
            self.current_full_state_raw[i], _ = resample_values(self.min_max, self.given_indices, mean=mean, cov=cov)
            self.current_full_state[i] = np.append(self.current_full_state_raw[i][0], self.current_full_state_raw[i][1])
        # self.current_full_state = self.random_stream.uniform(low=[0] * self.N, high=[1] * self.N)
        return self.current_full_state

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            # print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]
