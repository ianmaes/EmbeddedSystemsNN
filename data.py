import serial
import time
import csv

com = "COM3"
baud = 115200

x = serial.Serial(com, baud, timeout = 0.1)

time_start = time.time()

while x.isOpen() == True:
    data = str(x.readline().decode('utf-8')).rstrip()
    if data != '':
        data = [dummy.strip() for dummy in data.split(',')]
        shunt = data[0]
        bus = data[1]
        curr = data[2]
        load = data[3]
        power = data[4]
        temp_bat = data[5]
        temp_amb = data[6]

        print(data)
        time_delta = time.time() - time_start
        
        with open('SensorData.csv', mode='a', newline='') as sensor_file:
            sensor_writer = csv.writer(sensor_file)
            sensor_writer.writerow([shunt, bus, curr, load, power, temp_bat, temp_amb, time_delta])