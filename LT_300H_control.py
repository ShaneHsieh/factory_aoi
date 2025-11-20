from Serial_port import SerialDevice 
import time

import random 

class LT300HControl(SerialDevice):
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        super().__init__(port, baudrate, timeout)
        self.get_current_position()#100.001,100.002,0.000
        time.sleep(0.5)  # 等待回應
        self.cur_x, self.cur_y, self.cur_z = dev.last_line.split(",")

    def move_to(self, x: float, y: float, z: float):
        print(f"移動到: X={x}, Y={y}, Z={z}")
        command = f"H\r\nMA {x},{y},{z}\r\nEND\r\n"
        self.write(command)
    
    def set_move_speed(self, speed: int):
        command = f"H\r\nSP {speed}\r\nEND\r\n"
        self.write(command)

    def get_current_position(self):
        command = "H\r\nPA\r\nEND\r\n"
        self.write(command)

if __name__ == "__main__":
    dev = LT300HControl(port="COM8", baudrate=115200, timeout=1.0)
    dev.open()
    dev.start_reading()
    dev.set_move_speed(100)
    time.sleep(2)  # 等待連線穩定
    try:
        while True:
            print("Enter command to send (type 'END' on a new line to finish, or 'exit' to quit):")
            lines = []
            while True:
                line = input()
                if line.lower() == "exit":
                    raise KeyboardInterrupt
                # lines.append(line)
                dev.move_to(line.split(",")[0], line.split(",")[1], line.split(",")[2])
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()

    # try:
    #     while True:
    #         x = round(random.uniform(0,200),3)
    #         y = round(random.uniform(0,200),3)
    #         z = round(random.uniform(0,10),3)
    #         print(f"移動到: X={x}, Y={y}, Z={z}")
    #         dev.move_to(x, y, z)
    #         time.sleep(1)

    #         while True:
    #             dev.get_current_position()#100.001,100.002,0.000
    #             time.sleep(0.5)  # 等待回應
    #             cur_x, cur_y, cur_z = dev.last_line.split(",")
    #             cur_x, cur_y, cur_z = float(cur_x), float(cur_y), float(cur_z) 
    #             print(f"目前位置: X={cur_x}, Y={cur_y}, Z={cur_z}")
    #             if abs(cur_x - x) < 0.01 and abs(cur_y - y) < 0.01 and abs(cur_z - z) < 0.01:
    #                 break
    #             else:
    #                 time.sleep(0.5)            

    # except KeyboardInterrupt:
    #     print("Exiting...")
    # finally:
    #     dev.close()
