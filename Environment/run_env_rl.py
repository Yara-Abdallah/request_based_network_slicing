from typing import List

from Outlet.Cellular.ICellular import Cellular
from Utils.PerformanceLoggerFifo import PerformanceLoggerFifo
from .utils.imports import *
from .utils.period import Period
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
                performancelogger.set_outlet_services_power_allocation(outlet, [0, 0, 0])
                performancelogger.set_queue_requested_buffer(outlet, deque([]))
                performancelogger.set_queue_wasted_req_buffer(outlet, deque([]))
                performancelogger.set_queue_ensured_buffer(outlet, deque([]))
                performancelogger.set_queue_power_for_requested_in_buffer(outlet, deque([]))
                performancelogger.set_queue_waiting_requests_in_buffer(outlet, deque([]))
                performancelogger.set_queue_time_out_from_simulation(outlet, deque([]))
                performancelogger.set_queue_from_wait_to_serve_over_simulation(outlet, deque([]))
                performancelogger.set_outlet_services_requested_number_all_periods(outlet, [0, 0, 0])
                performancelogger.set_outlet_services_requested_number(outlet, [0, 0, 0])
                performancelogger.set_outlet_services_ensured_number(outlet, [0, 0, 0])

                # performancelogger.set_outlet_services_power_allocation_10_TimeStep(outlet, [0, 0, 0])
                if outlet not in performancelogger.queue_requests_with_execution_time_buffer:
                    performancelogger.queue_requests_with_execution_time_buffer[outlet] = dict()

                if outlet not in performancelogger.queue_requests_with_time_out_buffer:
                    performancelogger.queue_requests_with_time_out_buffer[outlet] = dict()

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
        number_of_decentralize_periods = 0
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
            #     print("each refresh episode : ", step_for_each_episode_change_period)
            # print("step_for_each_episode_change_period : ", step_for_each_episode_change_period)

            if 0 <= step_for_each_episode_change_period <= env_variables.period1_episode:
                Period('period1')
            if env_variables.period1_episode < step_for_each_episode_change_period <= env_variables.period2_episode:
                Period('period2')
            if env_variables.period2_episode < step_for_each_episode_change_period <= env_variables.period3_episode:
                Period('period3')
            if env_variables.period3_episode < step_for_each_episode_change_period <= env_variables.period4_episode:
                Period('period4')
            if env_variables.period4_episode < step_for_each_episode_change_period <= env_variables.period5_episode:
                Period('period5')

            # if self.steps - previous_steps_sending == frame_rate_for_sending_requests:
            #     previous_steps_sending = self.steps

            seed_value = 1
            # Seed the random number generator
            ra.seed(seed_value)

            number_of_cars_will_send_requests = round(
                len(list(env_variables.vehicles.values())) * 0.7
            )
            vehicles = ra.sample(
                list(env_variables.vehicles.values()), number_of_cars_will_send_requests
            )

            provisioning_time_services(self.gridcells_dqn[0].agents.grid_outlets, performance_logger, self.steps)

            print("vehicles number : ",len(vehicles))
            list(map(lambda veh: enable_sending_requests(veh, observer, self.gridcells_dqn, performance_logger,
                                                         self.steps), vehicles, ))

            buffering_not_served_requests(self.gridcells_dqn[0].agents.grid_outlets, performance_logger, self.steps)
            if self.steps - self.previous_steps_centralize_action >= 40:
                self.previous_steps_centralize_action = self.steps
                # centralize_nextstate_reward(self.gridcells_dqn)
                centralize_state_action(self.gridcells_dqn, step, performance_logger)

            if self.steps - self.prev == self.snapshot_time:
                self.prev = self.steps
                take_snapshot_figures()

            else:
                close_figures()
            #
            # if self.steps - self.previous_steps >= env_variables.decentralized_replay_buffer:
            #     self.previous_steps = self.steps
            #     for ind, gridcell_dqn in enumerate(self.gridcells_dqn):
            #         for i, outlet in enumerate(gridcell_dqn.agents.grid_outlets):
            #             if len(outlet.dqn.agents.memory) > 31:
            #                 # print("replay buffer of decentralize ")
            #                 outlet.dqn.agents.qvalue = (
            #                     outlet.dqn.agents.replay_buffer_decentralize(
            #                         30, outlet.dqn.model
            #                     )
            #                 )


            # if self.steps - self.previ
            # if self.steps - self.previous_steps_centralize >= env_variables.centralized_replay_buffer:
            #     self.previous_steps_centralize = self.steps
            #     for ind, gridcell_dqn in enumerate(self.gridcells_dqn):
            #         if len(gridcell_dqn.agents.memory) >= 64:
            #             # print("replay buffer of centralize ")
            #             self.average_qvalue_centralize.append(gridcell_dqn.agents.replay_buffer_centralize(32,
            #                                                                                           gridcell_dqn.model))

            if self.steps - self.previouse_steps_reseting >= env_variables.episode_steps:
                self.previouse_steps_reseting = self.steps

                add_value_to_pickle(
                    os.path.join(requests_with_execution_time_path, f"requests_with_execution_time.pkl"),
                    performance_logger.queue_requests_with_execution_time_buffer,
                )

                add_value_to_pickle(
                    os.path.join(requests_with_out_time_path, f"requests_with_out_time.pkl"),
                    performance_logger.queue_requests_with_time_out_buffer,
                )

                list_ = []
                print("resetting ................. ")
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

                        # print(" out : ", out.current_capacity)
                        # out.dqn.environment.state.state_value_decentralize = out.dqn.environment.state.calculate_state()

                    gridcell_dqn.environment.reward.resetreward()
                    gridcell_dqn.environment.state.resetsate(self.temp_outlets)

                performance_logger.reset_state_decentralize_requirement()

            step += 1
            step_for_each_episode_change_period += 1
            self.steps += 1
            if step == 90:
                save_weigths_buffer(self.gridcells_dqn[0], 90)
            if step == 100:
                save_weigths_buffer(self.gridcells_dqn[0], 100)
            if step == 110:
                save_weigths_buffer(self.gridcells_dqn[0], 110)
            if step == 120:
                save_weigths_buffer(self.gridcells_dqn[0], 120)
            if step == 130:
                save_weigths_buffer(self.gridcells_dqn[0], 130)
            if step == 140:
                save_weigths_buffer(self.gridcells_dqn[0], 140)
            if step == env_variables.TIME:
                save_weigths_buffer(self.gridcells_dqn[0], 100)

        self.close()

    def close(self):
        traci.close()
