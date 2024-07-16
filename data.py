import serial
import time
import csv
import torch
import inspect
from pyfirmata import Arduino, util, STRING_DATA
from NN import NN
from train import m, s

com = 'COM5'
baud = 115200
x = serial.Serial(com, baud, timeout = 0.1) 

time_start = time.time()

model = NN(5, 64, 1)
model.load_state_dict(torch.load('model1.pth'))
model.eval()



while x.isOpen() == True:

    data = str(x.readline().decode('utf-8')).rstrip()
    print(data)
    if data != '' and len(data.split(',')) == 7:
        data = [dummy.strip() for dummy in data.split(',')]
        shunt = data[0]
        bus = data[1]
        load = data[2]
        curr = data[3]
        power = data[4]
        temp_bat = data[5]
        temp_amb = data[6]


        time_delta = time.time() - time_start




        input_tensor = (torch.tensor([float(shunt), float(bus), float(load), float(curr), float(power)], dtype=torch.float32) - m) / s
        prediction = model(input_tensor)
        prediction_value = prediction.item()
        
        time.sleep(1)
        x.write(f'{prediction_value*100:.1f}\n'.encode('utf-8'))
        print(f'Prediction: {prediction_value * 100}%')
        
        # with open('SensorData.csv', mode='a', newline='') as sensor_file:
        #     sensor_writer = csv.writer(sensor_file)
        #     sensor_writer.writerow([shunt, bus, load, curr, power, temp_bat, temp_amb, time_delta])