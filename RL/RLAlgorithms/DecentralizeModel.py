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
    def save(self, filename):
        self.save_model(filename)

    def load(self, filename):
        return load_model(filename)

    def load_weights(self, filename):
        """ Load model from file. """
        self.load_weights(filename)

    def save_weights(self, filename):
        """ Save model to file. """
        self.save_weights(filename)

    def predict(self, state):
        """ Predict value based on state. """
        pass
