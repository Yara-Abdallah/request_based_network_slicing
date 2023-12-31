import numpy as np

from Service.IService import Service


class FactoryEntertainment(Service):
    """
    params: *args
        Service init parameters; reference to Iservice.py
    """
    def __init__(self,*args):
        super().__init__(*args)
        self._data_size = 0
        self._network_latency = 0
        # add weights for each parameter


    def calculate_processing_time(self):
        # self._network_latency + self._data_size
        return np.random.choice(np.arange(8,12))

    def calculate_time_out(self):
        return np.random.choice(np.arange(2,60))

    def calculate_arrival_rate(self):
        # TODO: add doc string
        return 1
