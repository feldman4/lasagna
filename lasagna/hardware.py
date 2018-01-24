import ftd2xx
import time
import serial

relay = None
chemyx_0 = None # cleavage
chemyx_1 = None # incorporation
chemyx_names = 'A9K7ZDDH', 'A4008VMp'

class Relay(object):
    def __init__(self):
        device = open_device('A506L0NW')
        # from amazon review
        device.setBitMode(0xFF,0x4)
        device.resetDevice()
        self._initialize(device)
            
    def _update(self):
        binary = ''.join(map(str, self.state[::-1]))
        cmd = '\xFF' + chr(int(binary, 2))
        self.device.write(cmd)
        
    def toggle(self, i):
        state = list(self.state)
        state[i] = 1 - state[i]
        self.state = tuple(state)
        self._update()
        
    def set(self, i, boolean):
        if self.state[i] != boolean:
            self.toggle(i)
        
    def _initialize(self, device):
        self.device = device
        self.state = (0,1,0)
        self._update()

def open_device(name):
    devices = ftd2xx.listDevices()
    ix = devices.index(name)
    return ftd2xx.open(dev=ix)

def open_hardware():
    global relay
    global chemyx_0
    global chemyx_1
    relay = Relay()
    chemyx_0 = open_device(chemyx_names[0])
    chemyx_1 = open_device(chemyx_names[1])

def open_vacuum():
    relay.set(1, False)

def close_vacuum(latency=5):
    # physical latency...
    relay.set(1, True)
    time.sleep(latency)

def open_pump():
    # at speed 30, medium tygon: 10 mL / 27 sec = 370 uL / s
    relay.set(0, True)

def close_pump():
    relay.set(0, False)

def heat_on():
    relay.set(2, True)

def heat_off():
    relay.set(2, False)

def wash(seconds):
    open_pump()
    time.sleep(seconds)
    close_pump()

def wash_cycle(aspirate=5, wash=5):
    open_vacuum()
    time.sleep(aspirate)
    close_vacuum()
    open_pump()
    time.sleep(wash)
    close_pump()

def pump_cleavage():
    chemyx_0.write('start\r\n')
    time.sleep(20)
    chemyx_0.write('stop\r\n')

def pump_incorporation():
    chemyx_0.write('start\r\n')
    time.sleep(20)
    chemyx_0.write('stop\r\n')    

def do_cleavage():
    heat_on()
    open_vacuum()
    # really aspirate
    time.sleep(10)
    close_vacuum(10)
    pump_cleavage()
    time.sleep(60*3)
    for _ in range(20):
        wash_cycle()
    heat_off()

def do_incorporation():
    heat_on()
    open_vacuum()
    # really aspirate
    time.sleep(10)
    close_vacuum(10)
    pump_incorporation()
    time.sleep(60*3)
    for _ in range(20):
        wash_cycle()
    heat_off()

