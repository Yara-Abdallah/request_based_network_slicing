import os
import pickle
import sys

outlet_num  = 0

results_dir_train_results = f"{os.path.join(sys.path[0])}//fair_memory_selection_0.5_m0.5_m1_test"
request_inf =os.path.join(results_dir_train_results, f"request_info//request_info.pkl")

info = []
with open(request_inf, "rb") as file:
    try:
        while True:
            loaded_value = pickle.load(file)
            info.append(loaded_value)
    except EOFError:
        pass
for i in info :
    print(i[0])