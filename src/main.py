# Author: Damjan Denic
import random
import numpy as np
from communication import ArduinoCommunication
import torch

class Arduino_Visualizer:
    def __init__(self, model, baudrate=9600, port='COM3'):
        # the end matrix that will be sent to the arduino is 8X32
        self.model_shapes = []
        # find the shape of the weights
        for i in range(len(model)):
            # get the shape of the weights
            weight_shape = model[i].shape
            if len(weight_shape) > 1:
                self.model_shapes.append(weight_shape[1])
            else:
                self.model_shapes.append(0)

        print("Model weights shapes: ", self.model_shapes)

        self.arduino_communication = ArduinoCommunication(baudrate, port)

        # pick random weights from the model and put them in the matrix
        self.random_weight_indices = []
        expecting_number_of_weights = 8 * 32

        while len(self.random_weight_indices) < expecting_number_of_weights:
            random_shape_index = random.randint(0, len(self.model_shapes) - 1)
            # check if the model layer has more than 1 weight
            if self.model_shapes[random_shape_index] < 2:
                continue
            random_weight_index = random.randint(0, self.model_shapes[random_shape_index] - 1)
            
            self.random_weight_indices.append((random_shape_index, random_weight_index))

    def get_values(self, model) -> list:
        values = []
        parameters = list(model.parameters())
        with torch.no_grad():
            layer_indices = [x[0] for x in self.random_weight_indices]
            param_indices = [x[1] for x in self.random_weight_indices]
            layers = [parameters[i] for i in layer_indices]
            params = torch.tensor([layers[index].flatten()[param_index] for index, param_index in enumerate(param_indices)])
            # for i in range(len(self.random_weight_indices)):
            # layer_index = self.random_weight_indices[i][0]
            # param_index = self.random_weight_indices[i][1]
            # print(layer_index, param_index)
            # layer = parameters[layer_index]
            # print("layer parameters: ", len(layer))
            # param = list(layer.flatten())[param_index]
            # values.append(param)
        values = params.detach().numpy()
        return values

    def convert_to_binary(self, matrix):
        # get the median of the matrix
        median = np.median(matrix)
        # convert the matrix to binary
        binary_matrix = matrix > median
        return binary_matrix

    def hex_values_by_columns(self, matrix):
        # convert the matrix to hex values by columns
        hex_values = []
        for i in range(len(matrix[0])):
            hex_values.append(int(''.join('1' if x else '0' for x in matrix[:, i]), 2))
        return hex_values

    def show(self, model):
        # get all the weights from the model
        weight_list = self.get_values(model)
        weight_matrix = np.array(weight_list).reshape(8, 32)

        # convert to binary true/false depending on the median
        binary_matrix = self.convert_to_binary(weight_matrix)

        # convert to hex values for the arduino
        hex_values = self.hex_values_by_columns(binary_matrix)

        # send the matrix to the arduino
        self.arduino_communication.send_matrix(hex_values)


    

