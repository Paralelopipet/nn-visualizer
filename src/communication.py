import time
import serial

class ArduinoCommunication:
    def __init__(self, baudrate=9600, port='COM3') -> None:
        self.baudrate = baudrate
        self.port = port
        self.serial = serial.Serial(port, baudrate)

        # last time the matrix was sent
        self.last_time = 0

    def send_matrix(self, matrix):
        # send the matrix to the arduino
        if time.time() - self.last_time < 1: # 1 second delay
            return
        serialized_data = bytes(matrix)
        self.serial.write(matrix)
        self.last_time = time.time()


