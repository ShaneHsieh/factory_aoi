from Serial_port import SerialDevice 
import time
import numpy as np
import threading

import random 

class LT300HControl(SerialDevice):
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        super().__init__(port, baudrate, timeout)
        self.open()
        if self.ser is None:
            print("RS232 連接有誤")
            QMessageBox.warning(self, "RS232 沒有接上", "無法控制攝影機")
            sys.exit(1)
        self.start_reading()
        self.set_move_speed(100)
        #self.get_current_position()#100.001,100.002,0.000
        time.sleep(0.5)  # 等待回應
        self.limit_x = np.array([0, 300])
        self.limit_y = np.array([0, 300])
        self.limit_z = np.array([0, 300])
        
        self._arrive_callback = None  # 回調函數
        self._callback_context = None  # 回調上下文參數
        self._checking = False
        self._check_thread = None
    
    def set_start_position(self, x, y, z):
        self.start_x = x
        self.start_y = y
        self.start_z = z
        self.cur_x = self.start_x
        self.cur_y = self.start_y
        self.cur_z = self.start_z
        self.target_x = self.start_x
        self.target_y = self.start_y
        self.target_z = self.start_z

        self.limit_x = np.array([x, 300])
        self.limit_y = np.array([y, 300])
        self.limit_z = np.array([z, 300])

        self.move_to(self.start_x, self.start_y, self.start_z)

    def set_max_limit_position(self, x, y, z):
        self.limit_x[1] = x
        self.limit_y[1] = y
        self.limit_z[1] = z

    def clamp(self, value, limit):
        """自動修正到極限範圍內"""
        return float(np.clip(value, limit[0], limit[1]))

    def move_to(self, x: float, y: float, z: float, callback=None, context=None):
        x = float(x)
        y = float(y)
        z = float(z)

        x_clamped = self.clamp(x, self.limit_x)
        y_clamped = self.clamp(y, self.limit_y)
        z_clamped = self.clamp(z, self.limit_z)

        print(f"✓ 修正後移動: X={x_clamped}, Y={y_clamped}, Z={z_clamped}")

        # 下指令
        command = f"H\r\nMA {x_clamped},{y_clamped},{z_clamped}\r\nEND\r\n"
        self.write(command)

        # 儲存目標位置和回調函數
        self.target_x = x_clamped
        self.target_y = y_clamped
        self.target_z = z_clamped
        self._arrive_callback = callback
        self._callback_context = context

        # 啟動背景檢查執行緒
        if callback:
            self._start_arrival_check()
    
    def _start_arrival_check(self):
        """啟動背景執行緒檢查是否到達"""
        if self._checking:
            return
        self._checking = True
        self._check_thread = threading.Thread(target=self._check_arrival, daemon=True)
        self._check_thread.start()
    
    def _check_arrival(self):
        """背景執行緒：持續檢查是否到達目標位置"""
        max_wait = 5  # 最多等待 5 秒
        start_time = time.time()
        
        while self._checking and (time.time() - start_time) < max_wait:
            self.get_current_position()
            
            if (abs(self.cur_x - self.target_x) < 0.01 and 
                abs(self.cur_y - self.target_y) < 0.01 and 
                abs(self.cur_z - self.target_z) < 0.01):
                
                self._checking = False
                
                # 呼叫回調函數，並傳入上下文參數
                if self._arrive_callback:
                    self._arrive_callback(self._callback_context)
                break
            
            time.sleep(0.1)

        if self._checking:
            print("⚠️ 超時：未在指定時間內到達目標位置")
            self._checking = False
    
    def set_move_speed(self, speed: int):
        command = f"H\r\nSP {speed}\r\nEND\r\n"
        self.write(command)

    def get_current_position(self):
        command = "H\r\nPA\r\nEND\r\n"
        self.write(command)
        time.sleep(0.3)
        if self.last_line:
            try:
                cur_x, cur_y, cur_z = self.last_line.split(",")
                self.cur_x = int(round(float(cur_x)))
                self.cur_y = int(round(float(cur_y)))
                self.cur_z = int(round(float(cur_z)))
            except Exception as e:
                print(f"⚠️ 解析位置資料失敗: {e}")

    def check_current_position(self, x , y, z):
        self.get_current_position()
        if (abs(self.cur_x - x) < 0.01 and 
            abs(self.cur_y - y) < 0.01 and 
            abs(self.cur_z - z) < 0.01):
            return True
        return False

    def move_left(self):
        command = f"H\r\nVX 0\r\nEND\r\n"
        self.write(command)

    def move_right(self):
        command = f"H\r\nVX 300\r\nEND\r\n"
        self.write(command)
    
    def move_top(self):
        command = f"H\r\nVY 0\r\nEND\r\n"
        self.write(command)

    def move_down(self):
        command = f"H\r\nVY 300\r\nEND\r\n"
        self.write(command)

    def move_zoom_in(self):
        command = f"H\r\nVZ 100\r\nEND\r\n"
        self.write(command)

    def move_zoom_out(self):
        command = f"H\r\nVZ 0\r\nEND\r\n"
        self.write(command)
    
    def move_stop(self):
        command = f"H\r\nST\r\nEND\r\n"
        self.write(command)

if __name__ == "__main__":
    dev = LT300HControl(port="COM9", baudrate=115200, timeout=1.0)
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
