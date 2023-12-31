import math
import os

from collections import deque
import random
# from keras import backend as K
from RL.Agent.IAgent import AbstractAgent
from RL.RLEnvironment.Action.ActionChain import Exploit, Explore, FallbackHandler
import pickle
from Environment.utils.mask_generation import *


class Agent(AbstractAgent):
    _grid_outlets = []
    _action_value = 0
    _outlets_id = []

    def __init__(self, *args):
        super().__init__(*args)
        self._outlets_id = []
        self._grid_outlets = []
        self._action_value = 0
        self._qvalue = 0.0
        self.mask = []
        self.C = 1000
        self.remember = False
        self.action1_negative_reward = deque([], maxlen=250)
        self.action1_positive_reward = deque([], maxlen=250)
        self.action0_negative_reward = deque([], maxlen=250)
        self.minibatch = []
        self.data_for_grid_search = []

    @property
    def outlets_id(self):
        return self._outlets_id

    @outlets_id.setter
    def outlets_id(self, id_):
        self._outlets_id = id_

    @property
    def qvalue(self):
        return self._qvalue

    @qvalue.setter
    def qvalue(self, q):
        self._qvalue = q

    @property
    def grid_outlets(self):
        return self._grid_outlets

    @grid_outlets.setter
    def grid_outlets(self, list_):
        self._grid_outlets = list_

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, a):
        self._action = a

    @property
    def action_value(self):
        return self._action_value

    @action_value.setter
    def action_value(self, a):
        self._action_value = a

    def filter_buffer(self, model):
        def remove_below_threshold(dictionary, threshold):
            return {key: value for key, value in dictionary.items() if value < threshold}

        def find_median_second_half(data):
            sorted_data = sorted(data)
            n = len(sorted_data)
            if n >= 2:
                second_half = sorted_data[n // 2:]
                return np.median(second_half)
            else:
                return None

        dictionary_sample_loss = dict()
        counter = 0
        for index, (exploration, state, action, reward, next_state, prob) in enumerate(self.memory):
            if next_state is not None:
                if prob == 0.0:
                    counter += 1
                    sh = np.array(state).shape
                    state = np.array(state).reshape([1, max(sh)])
                    qvalue_for_state = model.predict(state, verbose=0)
                    model.fit(state, qvalue_for_state, epochs=1, verbose=0)
                    qvalue_for_state_after_fit = model.predict(state, verbose=0)
                    loss = math.pow((qvalue_for_state_after_fit[0][action] - qvalue_for_state[0][action]), 2)
                    # assert loss == 0, f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  {loss}...{state}'
                    dictionary_sample_loss[index] = loss
                    updated_tuple = (exploration, state, action, reward, next_state, loss)
                    self.memory[index] = updated_tuple
        # print("number of samples edit loss for them : ",counter )

        sorted_dict = dict(sorted(dictionary_sample_loss.items(), key=lambda item: item[1]))
        median = find_median_second_half(list(sorted_dict.values()))
        if median != None:
            samples_to_remove = remove_below_threshold(sorted_dict, median)
            return list(samples_to_remove.keys())
        else:
            return None

    def fair_selection_from_memory(self):
        # if len(self.action1_positive_reward)>250:
        #     self.action1_positive_reward = []
        # if len(self.action1_negative_reward)>250:
        #     self.action1_negative_reward = []
        # if len(self.action0_negative_reward)>250:
        #     self.action0_negative_reward = []

        # print("len of memory : ", len(self.memory))
        for index, (exploitation, state, action, reward, next_state, prob) in enumerate(self.memory):
            if action == 1 and reward > 0:
                if self.memory[index] not in self.action1_positive_reward:
                    self.action1_positive_reward.append(self.memory[index])
            if action == 1 and reward < 0:
                if self.memory[index] not in self.action1_negative_reward:
                    self.action1_negative_reward.append(self.memory[index])
            if action == 0:
                if self.memory[index] not in self.action0_negative_reward:
                    self.action0_negative_reward.append(self.memory[index])

    def replay_buffer_decentralize(self, batch_size, model):
        # filtered_samples_indices = self.filter_buffer(model)
        # if filtered_samples_indices != None:
        #     updated_deque = deque(item for i, item in enumerate(self.memory) if i not in filtered_samples_indices)
        #     self.memory = deque(updated_deque, maxlen=750)
        self.minibatch = []
        self.fair_selection_from_memory()
        # print(len(self.action1_positive_reward),len(self.action1_negative_reward),
        #                  len(self.action0_negative_reward))
        batch_size = min(len(self.action1_positive_reward),len(self.action1_negative_reward),
                             len(self.action0_negative_reward))
        if batch_size > 25 :
            batch_size =  25
        minibatch_action1_positive_reward = random.sample(self.action1_positive_reward, batch_size)
        minibatch_action1_negative_reward = random.sample(self.action1_negative_reward, batch_size)
        minibatch_action0_negative_reward = random.sample(self.action0_negative_reward, batch_size)
        self.minibatch.extend(minibatch_action1_positive_reward)
        self.minibatch.extend(minibatch_action0_negative_reward)
        self.minibatch.extend(minibatch_action1_negative_reward)
        # self.minibatch = random.sample(self.memory, batch_size)
        target = 0
        for exploitation, state, action, reward, next_state, prob in self.minibatch:
            target = reward
            if next_state is not None:
                sh = np.array(next_state).shape
                next_state = np.array(next_state).reshape([1, max(sh)])
                # logit_model2 = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
                logit_value = model.predict(next_state, verbose=0)[0]
                target = reward + self.gamma * np.amax(logit_value)
                state = np.array(state).reshape([1, max(sh)])
            target_f = model.predict(state, verbose=0)
            target_f[0][action] = target
            self.data_for_grid_search.append([state,target_f])
            model.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.min_epsilon:
        #     self.epsilon -= self.epsilon * self.epsilon_decay
        return target

    def hard_update_target_network(self, step, model, target_model):
        """ Here the target network is updated every K timesteps
            By update, I mean clone the behavior network.
        """
        if step % self.C == 0:
            pars = model.get_weights()
            target_model.set_weights(pars)

    def replay_buffer_centralize(self, batch_size, model):
        minibatch = random.sample(self.memory, batch_size)
        target = []
        for exploitation, state, action, reward, next_state in minibatch:
            target = reward
            if next_state is not None:
                next_state = np.array(next_state).reshape([1, np.array(next_state).shape[0]])
                model_qvalue = model.predict(next_state, verbose=0)[0]

                if exploitation == 0:
                    # print("exploration : centralize ")
                    target = reward + self.gamma * model_qvalue[action]
                elif exploitation == 1:
                    # print("exploitaion : centralize ")
                    target = reward + self.gamma * np.amax(model_qvalue)

            state = np.array(state).reshape([1, np.array(state).shape[0]])
            target_f = np.round(model.predict(state, verbose=0))
            ########################################### note for rounding
            target_f[0][action] = target
            model.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.min_epsilon:
        #     self.epsilon -= self.epsilon * self.epsilon_decay
        return target

    def free_up_memory(self, deque, filename):
        mode = 'wb' if not os.path.exists(filename) else 'ab'
        with open(filename, mode) as file:
            for item in deque:
                pickle.dump(item, file)
        deque.clear()

    def fill_memory(self, deque, filename):
        with open(filename, 'rb') as file:
            try:
                while True:
                    loaded_value = pickle.load(file)
                    deque.append(loaded_value)
            except EOFError:
                pass

    def remember(self, flag, state, action, reward, next_state):
        self.memory.append((flag, state, action, reward, next_state))

    def remember_decentralize(self, flag, state, action, reward, next_state, probability):

        # print(supported_services, "  \n  ",flag  , "  \n  ", state, "  \n  ", action, "  \n  ", reward, "  \n  ", next_state)
        self.memory.append((flag, state, action, reward, next_state, probability))

    def chain_dec(self, model, state, epsilon):
        "A chain with a default first successor"
        test = np.random.rand()
        "Setting the first successor that will modify the payload"
        action = self.action
        handler = Exploit(action, model, state,
                          Explore(action, FallbackHandler(action)))
        action_Value, flag = handler.handle(test, epsilon)
        # print("action value inside chain : ",action_Value )
        return action_Value

    def chain(self, model, state, epsilon):
        "A chain with a default first successor"
        test = np.random.rand()
        "Setting the first successor that will modify the payload"
        action = self.action
        handler = Exploit(action, model, state, Explore(action, FallbackHandler(action)))
        action_Value, flag = handler.handle(test, epsilon)
        return action, np.where(action_Value > 0.5, 1, 0), flag

    def exploitation(self, model, state):
        action = self.action
        action_value = self.action.exploit(model, state)
        flag = 1
        return action, np.where(action_value > 0.5, 1, 0), flag

    def advisor_for_decentralize(self, current_capacity, power_allocation, time_out, buffer_length):
        action = 0
        if current_capacity >= power_allocation and buffer_length == 0:
            action = 1
        elif current_capacity >= power_allocation and buffer_length <= 20 and time_out >= 12:
            action = 1
        elif current_capacity < power_allocation and buffer_length >= 85 and time_out < 12:
            action = 0
        elif current_capacity < power_allocation and buffer_length <= 20 and time_out >= 12:
            action = 1
        else:
            random_number = np.random.rand()
            if random_number >= 0.5:
                action = 1
            else:
                action = 0
        return action

    def heuristic_action(self, gridcell, current_services_power_allocation, current_services_requested,
                         number_of_periods_until_now):
        outlets = []
        flags = np.zeros(9)
        # flags = 0
        for j, outlet in enumerate(gridcell.agents.grid_outlets):
            outlets.append(outlet)
        list_power = [0, 0, 0]
        list_requested = [0, 0, 0]
        dic_power_with_index = {}
        dec_requested_with_index = {}
        for out in outlets:
            for i in range(3):
                list_power[i] = list_power[i] + current_services_power_allocation[out][i]
                list_requested[i] = list_requested[i] + current_services_requested[out][i]
        for i in range(3):
            dic_power_with_index[i] = list_power[i]
            dec_requested_with_index[i] = list_requested[i]
        the_sorted_current_power = dict(sorted(dic_power_with_index.items(), key=lambda x: x[1]))
        the_sorted_current_power_copy = dict(sorted(dic_power_with_index.items(), key=lambda x: x[1]))

        outlets.reverse()
        for j, out in enumerate(outlets):
            out.supported_services = [0, 0, 0]
            count_zero = 0
            copy_current = out.current_capacity
            for i in range(3):
                key = list(the_sorted_current_power_copy.keys())[i]
                power = the_sorted_current_power_copy[key]
                requested = dec_requested_with_index[key]
                average = 0

                if number_of_periods_until_now > 0:
                    average = power / number_of_periods_until_now

                if copy_current >= average and average > 0.0:
                    out.supported_services[key] = 1
                    copy_current = out.current_capacity - average
                    the_sorted_current_power_copy[key] = 0
                elif average == 0.0:
                    out.supported_services[key] = 0
                elif copy_current == 0:
                    out.supported_services[key] = 0
                elif copy_current < average:
                    if copy_current >= average * 0.4:
                        out.supported_services[key] = 1
                        the_sorted_current_power_copy[key] = abs(the_sorted_current_power_copy[key] - copy_current)
                        copy_current = 0
                        break
            # print(f"out {out.supported_services}")
        return flags
