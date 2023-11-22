
# data discretization
import numpy as np

def discretize_percentages_between_0_100(value, num_bins=6):
    # for tower capacity and the buffer length

    # Define the range of Tower Available Capacity [0, 100]
    min_value = 0
    max_value = 100

    # Calculate the width of each bin
    delta = (max_value - min_value) / num_bins

    # Determine the bin index for the given value
    bin_index = min(int((value - min_value) / delta), num_bins - 1)

    # Return the discretized value
    discretized_value = min_value + bin_index * delta
    return discretized_value


def discretize_percentages_between_0_3(value, num_bins=3):
    # Define the range of Tower Available Capacity [0, 100]
    min_value = 0
    max_value = 3

    # Calculate the width of each bin
    delta = (max_value - min_value) / num_bins

    # Determine the bin index for the given value
    bin_index = min(int((value - min_value) / delta), num_bins - 1)

    # Return the discretized value
    discretized_value = min_value + bin_index * delta
    return discretized_value


num_bins_cap = 2
num_bins_bl = 2
num_bins_pa = 3
path =  "C:/Users/Windows dunya/Desktop/data_for_DQN_model_tuning_parameters/"

import pickle as pk

all_data = []
with open(f"{path}_reject.pkl",'rb') as file:
    try :
        while True:
            data = pk.load(file)
            all_data.append(data)
    except EOFError:
        pass

with open(f"{path}_serve.pkl",'rb') as file:
    try :
        while True:
            data = pk.load(file)
            all_data.append(data)
    except EOFError:
        pass

with open(f"{path}_time_out.pkl",'rb') as file:
    try :
        while True:
            data = pk.load(file)
            all_data.append(data)
    except EOFError:
        pass

with open(f"{path}_wait_to_serve.pkl",'rb') as file:
    try :
        while True:
            data = pk.load(file)
            all_data.append(data)
    except EOFError:
        pass

set_of_unique_states = set()
for value in all_data:
    state = value[0]
    action = value[1]
    reward = value[2]
    next_state = value[3]

    print("the origin state : ",state)
    remaining_time_out = state[0]
    available_cap = state[1]
    discrete_capacity = round(discretize_percentages_between_0_100(available_cap, num_bins=num_bins_cap))
    power_allocation = state[2]
    discrete_power_allocation = discretize_percentages_between_0_3(power_allocation, num_bins=num_bins_pa)
    buffer_length = state[3]
    discrete_buffer_length = round(discretize_percentages_between_0_100(buffer_length, num_bins=num_bins_bl))
    discrete_state = [remaining_time_out,discrete_capacity,discrete_power_allocation,discrete_buffer_length]
    print("the new state : ", discrete_state)
    set_of_unique_states.add(tuple(discrete_state))

print("len of all_data ",len(all_data))
print("len of unique states : ", len(set_of_unique_states))

#Initialize the Q-table to 0
n_states = num_bins_bl*num_bins_cap*50*num_bins_pa

Q_table = np.zeros((n_states,2))
print(Q_table)