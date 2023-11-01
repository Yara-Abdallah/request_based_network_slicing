from typing import List

from Outlet.Cellular.ICellular import Cellular
from Outlet.Sat.sat import Satellite
from Service.FactoryService import FactoryService
from Utils.config import SERVICES_TYPES
from .utils.episodes import Episodes
from .utils.imports import *
from .utils.savingWeights import *
from .utils.loadingWeights import *


class Environment:
    size = 0
    data = {}
    Grids = {}
    steps = 0
    average_qvalue_centralize = []
    # Initialize previous_steps variable
    previous_steps = 0
    frame_rate_for_sending_requests = 1
    previous_steps_sending = 0
    previous_period = 0
    snapshot_time = 5
    previous_steps_centralize = 0
    previous_steps_centralize_action = 0
    previouse_steps_reseting = 0
    prev = 0
    memory_threshold = 1500  # 3.5GB
    temp_outlets: List[Cellular] = []
    gridcells_dqn = []
    flag_processing_old_requests = [False] * 3
    reset_decentralize = False
    previouse_steps_reward32 = 0
    previous_steps_of_update_target_model = 0
    episodes_numbers = 0
    sub_episode_index = 0
    start_end_index = 0
    start = env_variables.sub_episode_length * sub_episode_index
    end = (env_variables.episode_steps * sub_episode_index + 1)

    def __init__(self, period: str):
        # Period(period)
        self.polygon = traci.polygon
        self.route = traci.route
        self.vehicle = traci.vehicle
        self.poi = traci.poi
        self.gui = traci.gui
        self.simulation = traci.simulation

    def get_polygons(self):
        all_polygon_ = self.polygon.getIDList()
        return all_polygon_

    def get_buildings(self):
        all_builds_ = []
        for id_poly in self.get_polygons():
            if self.polygon.getType(id_poly) == "building":
                all_builds_.append(id_poly)
        return all_builds_

    def prepare_route(self):
        """
        add routes to env_variables
        where the routes generated by randomTrips and store in random_routes_path
        """
        tree = ET.parse(env_variables.random_routes_path)
        root = tree.getroot()
        for child_root in root:
            id_ = child_root.attrib["id"]
            for child in child_root:
                # print(child.tag, child.attrib)
                # if child_root.tag == 'route':
                edges_ = list((child.attrib["edges"]).split(" "))
                # print('the id: {}  , edges: {}'.format(id_, edges_))
                self.route.add(id_, edges_)
                env_variables.all_routes.append(id_)
        del tree
        del root

    def update_outlet_color(self, id_, value):
        color_mapping = {
            (9, 10): (64, 64, 64, 255),  # dark grey
            (6, 9): (255, 0, 0, 255),  # red
            (3, 6): (0, 255, 0, 255),  # green
            (1, 3): (255, 255, 0, 255),  # yellow
        }

        for val_range, color in color_mapping.items():
            if value >= val_range[0] and value <= val_range[1]:
                traci.poi.setColor(id_, color)
        del color_mapping

    def get_all_outlets(self, performancelogger):
        """
        get all outlets and add id with position to env variables
        """
        outlets = []
        poi_ids = traci.poi.getIDList()

        def append_outlets(id_):
            type_poi = traci.poi.getType(id_)

            if type_poi in env_variables.types_outlets:
                # print(" type_poi : ", type_poi)
                position_ = traci.poi.getPosition(id_)
                env_variables.outlets[type_poi].append((id_, position_))
                val = 0
                if type_poi == "3G":
                    val = 8500
                elif type_poi == "4G":
                    val = 12500
                elif type_poi == "5G":
                    val = 100000
                elif type_poi == "wifi":
                    val = 1500
                factory = FactoryCellular(
                    outlet_types[str(type_poi)],
                    1,
                    1,
                    [1, 1, 0],
                    id_,
                    [position_[0], position_[1]],
                    10000,
                    [],
                    [10, 10, 10],
                )
                outlet = factory.produce_cellular_outlet(str(type_poi))
                outlet.outlet_id = id_
                outlet.radius = val

                performancelogger.initial_setting(outlet)

                outlets.append(outlet)

        list(map(lambda x: append_outlets(x), poi_ids))

        # satellite = Satellite(1, [1, 1, 0], 0, [0, 0],
        #                       100000, [],
        #                       [10, 10, 10])
        # outlets.append(satellite)

        del poi_ids

        return outlets

    def distance(self, outlet1, outlet2):
        """Returns the Euclidean distance between two outlets"""
        return math.sqrt(
            (outlet1.position[0] - outlet2.position[0]) ** 2
            + (outlet1.position[1] - outlet2.position[1]) ** 2
        )

    def fill_grids_with_the_nearest(self, outlets):
        sub_dis = []
        for j in outlets:
            dis = []
            for i in outlets:
                dis.append(self.distance(j, i))
            if len(dis) >= 4:
                sorted_dis = sorted(dis)
                min_indices = [dis.index(sorted_dis[i]) for i in range(4)]
                elements = [outlets[i] for i in min_indices]
                outlets = [
                    element
                    for index, element in enumerate(outlets)
                    if index not in min_indices
                ]
                sub_dis.append(elements)
        return sub_dis

    @staticmethod
    def fill_grids(grids):
        Grids = {
            "grid1": [],
            "grid2": [],
            "grid3": [],
            "grid4": [],
            "grid5": [],
            "grid6": [],
            "grid7": [],
        }

        def grid_namer(i, grid):
            name = "grid" + str(i + 1)
            Grids[name] = grid

        list(map(lambda x: grid_namer(x[0], x[1]), enumerate(grids)))
        return Grids

    def select_outlets_to_show_in_gui(self):
        """
        select outlets in .network to display type of each outlet
        """
        # for key in env_variables.outlets.keys():
        #     for _id,_ in env_variables.outlets[key]:
        #         self.gui.toggleSelection(_id, 'poi')
        from itertools import chain

        array = list(
            map(
                lambda x: x,
                chain(*list(map(lambda x: x[1], env_variables.outlets.items()))),
            )
        )
        list(
            map(
                lambda x: self.gui.toggleSelection(x[0], "poi"), map(lambda x: x, array)
            )
        )
        del array

    def get_positions_of_outlets(self, outlets):
        positions_of_outlets = []

        list(map(lambda x: positions_of_outlets.append(x.position), outlets))
        return positions_of_outlets

    def generate_vehicles(self, number_vehicles):
        """
        It generates vehicles and adds it to the simulation
        and get random route for each vehicle from routes in env_variables.py
        :param number_vehicles: number of vehicles to be generated
        """

        all_routes = env_variables.all_routes

        def add_vehicle(id_route_):
            uid = str(uuid4())
            self.vehicle.add(vehID=uid, routeID=id_route_)

            env_variables.vehicles[uid] = Car(uid, 0.0, 0.0)

        list(map(add_vehicle, ra.choices(all_routes, k=number_vehicles)))
        del all_routes

    def starting(self):
        """
        The function starts the simulation by calling the sumoBinary, which is the sumo-gui or sumo
        depending on the nogui option
        """

        os.environ["SUMO_NUM_THREADS"] = "8"
        # show gui
        # sumo_cmd = ["sumo-gui", "-c", env_variables.network_path]
        # dont show gui
        sumo_cmd = ["sumo", "-c", env_variables.network_path]
        traci.start(sumo_cmd)

        # end the simulation and d

        self.prepare_route()

    def remove_vehicles_arrived(self):
        """
        Remove vehicles which removed from the road network ((have reached their destination) in this time step
        the add to env_variables.vehicles (dictionary)
        """
        ids_arrived = self.simulation.getArrivedIDList()

        def remove_vehicle(id_):
            # print("del car object ")
            del env_variables.vehicles[id_]

        if len(ids_arrived) != 0:
            list(map(remove_vehicle, ids_arrived))

    def add_new_vehicles(self):
        """
        Add vehicles which inserted into the road network in this time step.
        the add to env_variables.vehicles (dictionary)
        """
        ids_new_vehicles = traci.vehicle.getIDList()

        def create_vehicle(id_):
            env_variables.vehicles[id_] = Car(id_, 0, 0)

        list(map(create_vehicle, ids_new_vehicles))

    def car_distribution(self, step):
        if step == 0:
            number_cars = int(
                nump_rand.normal(
                    loc=env_variables.number_cars_mean_std["mean"],
                    scale=env_variables.number_cars_mean_std["std"],
                )
            )
            self.generate_vehicles(number_cars)

        if traci.vehicle.getIDCount() <= env_variables.threashold_number_veh:
            number_cars = int(
                nump_rand.normal(
                    loc=env_variables.number_cars_mean_std["mean"],
                    scale=env_variables.number_cars_mean_std["std"],
                )
            )
            self.generate_vehicles(number_cars)

    def run(self):
        self.starting()
        performance_logger = PerformanceLogger()
        # performance_logger_for_fifo = PerformanceLoggerFifo()
        outlets = self.get_all_outlets(performance_logger)
        self.Grids = self.fill_grids(self.fill_grids_with_the_nearest(outlets[:4]))
        satellite = Satellite(1,
                              [1, 1, 0],
                              'sat',
                              [100, 100],
                              1000000000000000,
                              [],
                              [10, 10, 10])
        step = 0
        step_for_each_episode_change_period = 0
        print("\n")
        for i in outlets:
            print("out ", i.__class__.__name__)
        outlets_pos = self.get_positions_of_outlets(outlets)
        observer = ConcreteObserver(outlets_pos, outlets)

        # set the maximum amount of memory that the garbage collector is allowed to use to 1 GB
        max_size = 273741824

        gc.set_threshold(700, max_size // gc.get_threshold()[1])
        gc.collect(0)
        build = []
        for i in range(1):
            build.append(RLBuilder())
            self.gridcells_dqn.append(
                build[i]
                .agent.build_agent(ActionAssignment())
                .environment.build_env(CentralizedReward(), CentralizedState())
                .model_.build_model("centralized", 12, 2)
                .build()
            )

            self.gridcells_dqn[i].agents.grid_outlets = self.Grids.get(f"grid{i + 1}")
            self.gridcells_dqn[i].agents.outlets_id = list(
                range(len(self.gridcells_dqn[i].agents.grid_outlets))
            )

        for i in range(1):
            for index, outlet in enumerate(self.gridcells_dqn[i].agents.grid_outlets):
                self.temp_outlets.append(outlet)

        load_weigths_buffer(self.gridcells_dqn[0])
        while step < env_variables.TIME:
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024.0 / 1024.0  # Convert to MB
            if memory_usage > self.memory_threshold:
                gc.collect(0)

            gc.collect(0)

            traci.simulationStep()
            self.car_distribution(step)
            self.remove_vehicles_arrived()
            print("step is ....................................... ", step)
            if step % 320 == 0:
                step_for_each_episode_change_period = 0
            if step == self.start and step != 0:
                for outlet in self.temp_outlets:
                    if outlet.__class__.__name__ == 'Wifi' and len(outlet.dqn.agents.memory) > 75:
                        if performance_logger.queue_ensured_buffer[outlet]== 0 or performance_logger.number_of_requested_requests_buffer[outlet] == 0 :
                            throughput = 0
                        else:
                            throughput = performance_logger.queue_ensured_buffer[outlet] / performance_logger.number_of_requested_requests_buffer[outlet]
                            # print(" throughput >>>>  : ",  throughput )
                        outlet.dqn.environment.reward.throughput = throughput
                        l = list(outlet.dqn.agents.memory[-1])
                        v = outlet.dqn.agents.memory[-1]
                        outlet.dqn.agents.memory.remove(v)
                        l[3] = l[3] + outlet.dqn.environment.reward.throughput
                        # print("the new sampl e :   " , l )
                        outlet.dqn.agents.memory.append(tuple(l))
                        if len(outlet.dqn.agents.memory) > 75:
                            outlet.dqn.agents.qvalue = (
                                outlet.dqn.agents.replay_buffer_decentralize(
                                    75, outlet.dqn.model,
                                )
                            )
                        outlet.dqn.agents.memory = deque([],maxlen= 750)
                        outlet.dqn.agents.action1_positive_reward = deque([],maxlen= 250)
                        outlet.dqn.agents.action1_negative_reward = deque([],maxlen= 250)
                        outlet.dqn.agents.action0_negative_reward = deque([],maxlen= 250)

            if self.sub_episode_index % 18 == 0:
                self.sub_episode_index = 0

            if self.start <= step <= self.end:
                Episodes(f'episode{self.sub_episode_index + 1}')
                print("episode index : ", f'episode{self.sub_episode_index + 1}')
                self.sub_episode_index += 1
                self.start_end_index += 1
                # self.sub_episode_index = 0
                # self.start_end_index += 1
                self.start = env_variables.sub_episode_length * self.start_end_index
                self.end = env_variables.sub_episode_length * (self.start_end_index + 1)

                for outlet in self.temp_outlets:
                    if outlet.__class__.__name__ == 'Wifi':
            #             add_value_to_pickle("C:/Users/Windows dunya/PycharmProjects/pythonProject/Network-Slicing/throughputdata_for_grid_search(0.5,-1,-0.5)",(len(
            # performance_logger.queue_requested_buffer[outlet]),len(
            # performance_logger.queue_ensured_buffer[outlet])))
                        performance_logger.initial_setting(outlet)


                        waited_buffer_max_length = round(nump_rand.normal(
                            loc=env_variables.buffer_length_mean_std["mean"],
                            scale=env_variables.buffer_length_mean_std["std"],
                        ), 2)

                        if waited_buffer_max_length < 0:
                            waited_buffer_max_length = max(0, waited_buffer_max_length)
                        if waited_buffer_max_length > 1:
                            waited_buffer_max_length = min(1, waited_buffer_max_length)

                        number_of_requests_should_generation = int(
                            waited_buffer_max_length * outlet.waited_buffer_max_length)
                        if len(performance_logger.queue_waiting_requests_in_buffer[
                                   outlet]) <= outlet.waited_buffer_max_length:

                            # number_of_requests_should_generation  = 0
                            # print("number_of_requests_should_generation  : ",number_of_requests_should_generation)
                            for i in range(number_of_requests_should_generation):
                                # print("req  : ",i)
                                types = [*SERVICES_TYPES.keys()]
                                type_ = random.choices(types, weights=(
                                    env_variables.ENTERTAINMENT_RATIO, env_variables.SAFETY_RATIO,
                                    env_variables.AUTONOMOUS_RATIO), k=1)
                                realtime_ = random.choice(SERVICES_TYPES[type_[0]]["REALTIME"])
                                bandwidth_ = random.choice(SERVICES_TYPES[type_[0]]["BANDWIDTH"])
                                criticality_ = random.choice(SERVICES_TYPES[type_[0]]["CRITICAL"])
                                factory = FactoryService(realtime_, bandwidth_, criticality_)
                                service = factory.produce_services(type_[0])
                                service.realtime = realtime_
                                request_bandwidth = Bandwidth(service.bandwidth, service.criticality)
                                request_cost = RequestCost(request_bandwidth, service.realtime)
                                service.cost_in_bit_rate = request_cost.cost_setter(outlet)
                                service.service_power_allocate = request_bandwidth.allocated
                                service.total_cost_in_dolar = service.calculate_service_cost_in_Dolar_per_bit()
                                service.time_out = service.calculate_time_out()
                                service.time_execution = service.calculate_processing_time()
                                performance_logger.queue_requested_buffer[outlet] += 1
                                # print("performance_logger.queue_requested_buffer[outlet] inside ...  : ", performance_logger.queue_requested_buffer[outlet])

                                performance_logger.queue_power_for_requested_in_buffer[outlet].append(
                                    [service, False])

                                performance_logger.queue_power_for_requested_in_buffer[outlet][0][1] = False

                                performance_logger.queue_waiting_requests_in_buffer[outlet].appendleft(
                                    [service, True])
                                performance_logger.queue_requests_with_time_out_buffer[outlet][service] = [
                                    step,
                                    service.time_out]
                                logging_important_info_for_testing(performance_logger, 0, outlet, satellite)

                        tower_available_capacity = round(nump_rand.normal(
                            loc=env_variables.capacity_mean_std["mean"],
                            scale=env_variables.capacity_mean_std["std"], ), 2)

                        if tower_available_capacity < 0:
                            tower_available_capacity = max(0, tower_available_capacity)
                        if tower_available_capacity > 1:
                            tower_available_capacity = min(1, tower_available_capacity)
                        outlet.current_capacity = tower_available_capacity * outlet._max_capacity

            seed_value = 1
            # Seed the random number generator
            ra.seed(seed_value)

            number_of_cars_will_send_requests = round(
                len(list(env_variables.vehicles.values())) * 0.5

            )
            vehicles = ra.sample(
                list(env_variables.vehicles.values()), number_of_cars_will_send_requests
            )

            provisioning_time_services(self.gridcells_dqn[0].agents.grid_outlets, performance_logger, self.steps)

            list(map(lambda veh: enable_sending_requests(veh, observer, self.gridcells_dqn, performance_logger,
                                                         self.steps, satellite), vehicles, ))

            buffering_not_served_requests(self.gridcells_dqn[0].agents.grid_outlets, performance_logger, self.steps,
                                          satellite)

            if self.steps - self.previous_steps_centralize_action >= 40:
                self.previous_steps_centralize_action = self.steps
                # centralize_nextstate_reward(self.gridcells_dqn)
                centralize_state_action(self.gridcells_dqn, step, performance_logger)

            if self.steps - self.prev == self.snapshot_time:
                self.prev = self.steps
                take_snapshot_figures()
            else:
                close_figures()
            if self.steps - self.previous_steps >= env_variables.decentralized_replay_buffer:
                # print("here : .... ")
                self.previous_steps = self.steps
                for i, outlet in enumerate(self.temp_outlets):
                    if len(outlet.dqn.agents.memory) > 75:
                        # print(" outlet.dqn.agents.memory : ", len(outlet.dqn.agents.memory))
                        outlet.dqn.agents.qvalue = (
                            outlet.dqn.agents.replay_buffer_decentralize(
                                75, outlet.dqn.model,
                            )
                        )
            # for i, outlet in enumerate(self.temp_outlets):
            #     if outlet.__class__.__name__=="Wifi":
            #         print(outlet.dqn.agents.memory)

            # if self.steps - self.previous_steps_centralize >= env_variables.centralized_replay_buffer:
            #     self.previous_steps_centralize = self.steps
            #     for ind, gridcell_dqn in enumerate(self.gridcells_dqn):
            #         if len(gridcell_dqn.agents.memory) >= 64:
            #             # print("replay buffer of centralize ")
            #             self.average_qvalue_centralize.append(gridcell_dqn.agents.replay_buffer_centralize(32,
            #                                                                                           gridcell_dqn.model))

            if self.steps - self.previouse_steps_reseting >= env_variables.episode_steps:
                self.episodes_numbers += 1
                self.previouse_steps_reseting = self.steps
                # print(" performance_logger.user_requests : ", performance_logger.user_requests)
                for ind, gridcell_dqn in enumerate(self.gridcells_dqn):
                    for i, out in enumerate(gridcell_dqn.agents.grid_outlets):
                        add_value_to_pickle(
                            os.path.join(reward_info_path, f"reward_info.pkl"),
                            (out.__class__.__name__, out.dqn.environment.reward.serving_reward,
                             out.dqn.environment.reward.rejected_reward,
                             out.dqn.environment.reward.wait_to_serve_reward,
                             out.dqn.environment.reward.time_out_reward)
                        )

                add_value_to_pickle(
                    os.path.join(requests_with_execution_time_path, f"requests_with_execution_time.pkl"),
                    performance_logger.queue_requests_with_execution_time_buffer,
                )

                add_value_to_pickle(
                    os.path.join(requests_with_out_time_path, f"requests_with_out_time.pkl"),
                    performance_logger.queue_requests_with_time_out_buffer,
                )
                for ind, gridcell_dqn in enumerate(self.gridcells_dqn):
                    for i, out in enumerate(gridcell_dqn.agents.grid_outlets):
                        states = []
                        for st in out.dqn.agents.memory:
                            states.append(st[1])

                        add_value_to_pickle(
                            os.path.join(states_of_memory_path, f"states_of_memory_path{i}.pkl"),
                            states,
                        )

                list_ = []
                for ind, gridcell_dqn in enumerate(self.gridcells_dqn):
                    for i, out in enumerate(gridcell_dqn.agents.grid_outlets):
                        add_value_to_pickle(
                            os.path.join(decentralize_qvalue_path, f"qvalue{i}.pkl"),
                            out.dqn.agents.qvalue,
                        )

                        add_value_to_pickle(
                            os.path.join(reward_accumilated_decentralize_path, f"accu_reward{i}.pkl"),
                            out.dqn.environment.reward.reward_value_accumilated,
                        )

                        out.dqn.environment.state.resetsate()
                        out.dqn.environment.reward.resetreward()
                        out.dqn.environment.reward.reward_value_accumilated = 0
                        out.current_capacity = out.set_max_capacity(out.__class__.__name__)
                        satellite.rejected_requests_buffer = deque([])
                        satellite.sum_of_costs_of_all_requests = 0
                        out.abort_request = 0

                        # for index , (exploitation, state, action, reward, next_state, prob) in enumerate(out.dqn.agents.memory) :
                        #     updated_tuple = (exploitation, state, action, reward, next_state, 0.0)
                        #     out.dqn.agents.memory[index] = updated_tuple

                    gridcell_dqn.environment.reward.resetreward()
                    gridcell_dqn.environment.state.resetsate(self.temp_outlets)

                performance_logger.reset_state_decentralize_requirement()

            step += 1
            step_for_each_episode_change_period += 1
            self.steps += 1

            if step == 25 * 1152:
                save_weigths_buffer(self.gridcells_dqn[0], 25)
            if step == 35 * 1152:
                save_weigths_buffer(self.gridcells_dqn[0], 35)
            if step == 40 * 1152:
                save_weigths_buffer(self.gridcells_dqn[0], 40)
            if step == 45 * 1152:
                save_weigths_buffer(self.gridcells_dqn[0], 45)
            if step == 50 * 1152:
                save_weigths_buffer(self.gridcells_dqn[0], 50)
            if step == env_variables.TIME:
                save_weigths_buffer(self.gridcells_dqn[0], 70)


        self.close()

    def close(self):
        traci.close()
