import serial
import time
import csv
import torch
from NN import NN
from train import m, s

com = "COM3"
baud = 115200

x = serial.Serial(com, baud, timeout = 0.1)

time_start = time.time()

model = NN(7, 64, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()



while x.isOpen() == True:
    data0 = str(x.readline().decode('utf-8')).rstrip()
    print(data0)
    data = str(x.readline().decode('utf-8')).rstrip()
    print(data)
    if data != '':
        data = [dummy.strip() for dummy in data.split(',')]
        shunt = data[0]
        bus = data[1]
        load = data[2]
        curr = data[3]
        power = data[4]
        temp_bat = data[5]
        temp_amb = data[6]


        time_delta = time.time() - time_start




        input_tensor = (torch.tensor([float(shunt), float(bus), float(load), float(curr), float(power), float(temp_bat), float(temp_amb)], dtype=torch.float32) - m) / s
        prediction = model(input_tensor)
        prediction_value = prediction.item()
        x.write(f'{prediction_value}\n'.encode('utf-8'))
        time.sleep(2)
        
        # with open('SensorData.csv', mode='a', newline='') as sensor_file:
        #     sensor_writer = csv.writer(sensor_file)
        #     sensor_writer.writerow([shunt, bus, load, curr, power, temp_bat, temp_amb, time_delta])