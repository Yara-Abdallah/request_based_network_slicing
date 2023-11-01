
import pickle
from keras import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Dense
class DecentralizeModel():
    def __init__(self, state_size = 4, action_size =2, activation_function = "relu", loss_function = "mse", optimization_algorithm = Adam,
                 learning_rate = 0.0001, output_activation = "sigmoid", **kwargs):
        self.state_size = state_size
        self.action_size = action_size
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.output_activation = output_activation

    def build_model(self) -> Sequential:
        model_ = Sequential()
        model_.add(Dense(24, input_dim=self.state_size, activation=self.activation_function))
        model_.add(Dense(24, activation=self.activation_function))
        model_.add(Dense(self.action_size, activation=self.output_activation))
        # model_.add(Reshape((3, 3)))
        model_.compile(loss=self.loss_function,
                       optimizer=self.optimization_algorithm(learning_rate=self.learning_rate))
        return model_
# Assuming you have a pickle file named 'data.pkl'
pickle_file_path = 'C:/Users/Windows dunya/Desktop/data_for_grid_search/data_for_grid_search(0.5,-1,-0.5).pkl'
data = []
# Open the pickle file in read mode
with open(pickle_file_path, "rb") as file:
    try:
        while True:
            loaded_value = pickle.load(file)
            data.append(loaded_value)
    except EOFError:
        pass


dqn = DecentralizeModel()
model = dqn.build_model()
print(len(data[0]))
for i in data[0]:
    print("i[0] : ",i[0],"  i[1] : ",i[1])
    model.fit(i[0],i[1])

model.save_weights("C:/Users/Windows dunya/Desktop/data_for_grid_search/data_for_grid_search(0.5,-1,-0.5).hdf5")
# for key , values in data.items():
#     print(key)
# 'data' now contains the contents of the pickle file
