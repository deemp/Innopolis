import socket
import struct
import os
# import spidev
from multiprocessing import Process, Value, Array, Lock
from time import perf_counter
# TODO:
# Add support of python-can lib
# Add different hardware devices
#       can-hacker
#
# Rename CAN_Bus - to CANSocket

# Implement the object with 5 virtual
# rPyCAN
#


class CANSocket:
    def __init__(self, interface='can0', devices_id=[0x001], serial_port=None, device=None, bitrate=1000000, reset = True):

        self.frame_format = "=IB3x8s"
        self.frame = 0
        self.interface = interface
        self.bitrate = bitrate
        self.devices_reply = dict()
        # self.messages = dict()
        self.id = devices_id
        if reset:
            self.can_down()
            self.can_set()
            self.can_up()

        if serial_port or device == 'can_hacker':
            self.can_hacker_init(port=serial_port, baud_rate=115200)

        self.socket = socket.socket(
            socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.socket.bind((self.interface, ))

    def __del__(self):
        self.can_down()
        print('CAN Devices object was destructed')

    def can_hacker_init(self, port=None, baud_rate=115200):
        if not port:
            port = 'ttyACM0'
            print(
                f'<port> argument is not provided and choosen to be /dev/{port}')
        # self.can_down()
        self.serial_port = '/dev/' + port
        # os.system(f'sudo slcand -o -c -s8 -S {baud_rate} {serial_port} {interface}')
        # TODO:
        # Implent different speeds

        os.system(
            f'sudo slcand -o -c -s8 -S {baud_rate} {self.serial_port} {self.interface}')
        print(
            f'CAN device is connected on {port} / {baud_rate} and up as interface <{self.interface}> with {self.bitrate} bps')

    def can_set(self):
        os.system(
            f'sudo ip link set {self.interface} type can bitrate {self.bitrate}')
        print(
            f'CAN interface <{self.interface}> is setted on {self.bitrate} bps')

    def can_down(self):
        os.system(f'sudo ifconfig {self.interface} down')
        print(f'CAN interface <{self.interface}> is down')

    def can_up(self):
        os.system(f'sudo ifconfig {self.interface} up')
        # os.system(f'sudo ifconfig {self.interface} txqueuelen 65536')
        print(f'CAN interface <{self.interface}> is up')

    def can_reset(self):
        '''Reset the CAN interface'''
        self.can_down()
        self.can_set()
        self.can_up()
        print(f'CAN interface <{self.interface}> was reseted')

    def build_can_frame(self, can_id, data):
        can_dlc = len(data)
        data = data.ljust(8, b'\x00')
        return struct.pack(self.frame_format, can_id, can_dlc, data)

    def parse_can_frame(self, frame):
        can_id, can_dlc, data = struct.unpack(self.frame_format, frame)
        return (can_id, can_dlc, data[:can_dlc])

    def send_bytes(self, can_id, bytes_to_send):
        self.frame = self.build_can_frame(can_id, bytes_to_send)
        self.socket.send(self.frame)
        # print('done')

    def recive_frame(self):
        self.r_msg, _ = self.socket.recvfrom(16)
        can_id, can_dlc, can_data = self.parse_can_frame(self.r_msg)
        return can_id, can_dlc, can_data

    def send_recv(self, messages):
        'Send/recive routine'
        for device_id in self.id:
            self.send_bytes(device_id, messages[device_id])
            can_id, can_dlc, can_data = self.recive_frame()
            self.devices_reply[device_id] = can_data

        # for device_id in self.id:
        #     self.send_bytes(device_id, messages[device_id])

        # for device_id in self.id:
        #     can_id, can_dlc, can_data = self.recive_reply()
        #     self.devices_reply[can_id] = can_data

        return self.devices_reply

    # def send_recive_threads(self, messages):
    #     # TODO: separate threads/processes for send and recive
    #     # possibly separate hardware interfaces
    #     pass


# TODO: CANDevices initialized with bus
# Thnik how to initialize CANDevice for different types of bus
# There is different method to
# Test executer 
# Run executer constantly in different process:
#   - with fixed frequency
#   - with maximal speed

class CANDevice:
    """"""
    # TODO: This class should be created via socket bus or

    def __init__(self, can_bus=None, interface=None, device_id=0x01):

        # TODO: pass dict with reciver/transmitter functions from the specific bus
        if not can_bus:
            print("Provide can_bus as argument")
            self.__del__()

        # TODO: move this to class CANDevice, CANSensors should inherit from CANDevice

        self.interface = interface
        self.bus = can_bus
        self.transmiter = can_bus.send_bytes
        self.reciver = can_bus.recive_frame

        self.device_id = device_id
        # self.
        self.empty = 8 * b"\x00"
        self.command = self.empty
        self.reply = self.empty
        self.processes = []
    
    def __del__(self):
        self.command = self.empty
        self.reply = self.empty
        print('CANDDevice was destructed')

    def run(self):
        self.processes.append(Process(target=self.executer))
        # self.executer = 

        print("Processes are about to start...")
        for process in self.processes:
            process.start()
        try:
            while True:
                self.execute()
        except KeyboardInterrupt:
            print("Exit exicuter")


    def stop(self, delay = 0.0):
        print("Processes are about to stop...")
        if self.processes:
            for process in self.processes:
                process.terminate()
        print("Processes are terminated...")

    def executer(self, freq):
        # TODO:
        # implement frequancy limiter
        self.execute()


    def to_bytes(self, n, integer, signed=True):
        return int(integer).to_bytes(n, byteorder="little", signed=signed)

    def from_bytes(self, byte_string, signed=True):
        return int.from_bytes(byte_string, byteorder="little", signed=signed)

    def send_command(self, command):
        self.transmiter(self.device_id, command)

    def recive_reply(self):
        recived_id, _, reply = self.reciver()

        if recived_id == self.device_id:
            self.reply = reply

        return self.reply

    def execute(self):
        self.send_command(self.command)
        self.recive_reply()

