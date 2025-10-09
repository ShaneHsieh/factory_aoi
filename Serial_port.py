import serial
import threading
import time


class SerialDevice:
    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1.0):
        """初始化 RS232 連線參數"""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self._stop_event = threading.Event()
        self._read_thread = None
        self._line_callback = None
        self.last_line = None  # 新增屬性

    def open(self):
        """開啟 COM port"""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            print(f"✅ Connected to {self.port} at {self.baudrate} bps")
        except serial.SerialException as e:
            print(f"❌ Failed to open {self.port}: {e}")
            self.ser = None

    def close(self):
        """關閉連線"""
        if self.ser and self.ser.is_open:
            self._stop_event.set()
            if self._read_thread and self._read_thread.is_alive():
                self._read_thread.join()
            self.ser.close()
            print(f"🔌 Disconnected from {self.port}")

    def write(self, data: str):
        """傳送資料到 RS232"""
        if self.ser and self.ser.is_open:
            self.ser.write(data.encode('utf-8'))
            #print(f"➡️ Sent: {data.strip()}")
        else:
            print("⚠️ Serial port not open")

    def _read_loop(self):
        """背景讀取執行緒"""
        while not self._stop_event.is_set():
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if line:
                        #print(f"⬅️ Received: {line}")
                        if "ok!" not in line and "equals sign expected" not in line: 
                            self.last_line = line  # 更新 last_line
                else:
                    time.sleep(0.1)
            except serial.SerialException:
                print("⚠️ Serial disconnected unexpectedly")
                break

    def start_reading(self):
        """開啟背景讀取"""
        if not self.ser or not self.ser.is_open:
            print("⚠️ Port not open, cannot start reading")
            return
        self._stop_event.clear()
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        print("📡 Start listening for incoming data")

# === 使用範例 ===
if __name__ == "__main__":
    # 將這裡的 COM3 改成你的實際 port
    dev = SerialDevice(port="COM7", baudrate=115200)
    dev.open()
    dev.start_reading()

    try:
        while True:
            print("Enter command to send (type 'END' on a new line to finish, or 'exit' to quit):")
            lines = []
            while True:
                line = input()
                if line.lower() == "exit":
                    raise KeyboardInterrupt
                lines.append(line)
                if line == "END":
                    break
            msg = "\n".join(lines)
            if msg:
                dev.write(msg + "\r\n")
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()