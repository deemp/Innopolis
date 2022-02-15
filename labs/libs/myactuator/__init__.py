from time import perf_counter
from math import pi
from ..can import CANDevice
# from ..can
# TODO:


# Run motor in several modes in processes:
# i.e: speed, pos, current
# Implement motor modes
# motor.set_mode('s', 'p', 'c')
# via motor.run(mode = 's')
# provide interface to can_bus through tuple of functions
#   tranciver
#   reciver

# Tun command sender in different process

class MyActuator(CANDevice):
    """ This class provide interface to the Gyems BLDC motor driver over CAN socket"""

    def __init__(self, can_bus=None, device_id=0x141, units="rad"):
        super().__init__(can_bus = can_bus, device_id = device_id)

        self.protocol = dict()
        self.protocol = {
            # write PI gains for position, velocity and current loops in to RAM
            "write_pid_ram": b"\x31",
            # write PI gains for position, velocity and current loops in to ROM
            "write_pid_rom": b"\x32",
            "read_accel_lim": b"\x33",  # read the value of the acceleration limit
            "write_accel_lim_ram": b"\x34",  # write accceleration limit to ram
            "read_encoder_data": b"\x90",  # read the encoder data
            "set_encoder_offset": b"\x91",  # set encoder offset to the specofoc value
            "read_encoder_data": b"\x90",
            # set the current position as zero for encoder and save it tp ROM
            "set_encoder_zero_rom": b"\x19",
            "read_multiturn_angle": b"\x92",  # read the encoder data as cumalitive angle
            "read_single_angle": b"\x94",  #
            "read_motor_status_1": b"\x9A",  #
            "read_motor_status_2": b"\x9C",  #
            "read_motor_status_3": b"\x9D",  #
            "clear_error_flags": b"\x9B",  #
            "motor_off": b"\x80",  #
            "motor_stop": b"\x81",  #
            "motor_running": b"\x88",  #
            "set_torque": b"\xA1",  #
            "set_speed": b"\xA2",  #
            "set_pos_1": b"\xA3",  #
            "set_pos_2": b"\xA4",  #
            "set_pos_3": b"\xA5",  #
            "set_pos_4": b"\xA6",  #
        }

        self.command = self.protocol["motor_off"] + 7 * b"\x00"

        self.gains = {
            "pos": {"p": 0, "i": 0},
            "vel": {"p": 0, "i": 0},
            "cur": {"p": 0, "i": 0},
        }

        self.speed_limit = None
        self.accel_limit = 0
        self.current_limit = 500
        self.torque_limit = 500
        self.encoder_offset = 0
        self.error_state = "normal"

        self.torque_constant = 1
        self.encoder_scale = 16384

        self.current_scale = 1
        self.temp_scale = 1
        self.units = units
        self.set_units(self.units)

        self.motor_status = ["on", "off", "error"]

        state_labels = ["temp", "angle", "speed", "torque", "current"]
        self.state = dict(zip(state_labels, [0, 0, 0, 0, 0]))

        self.voltage = 0
        self.temp = 0
        self.angle = 0
        self.pos = 0
        self.speed = 0
        self.current = 0
        self.torque = 0
        self.phases_current = {"A": 0, "B": 0, "C": 0}

        self.raw_state_data = {"temp": 0,
                               "encoder": 0, "speed": 0, "current": 0}

        self.encoder_prev = 0

        self.desired_speed = 0
        self.desired_pos = 0
        self.desired_angle = 0
        self.desired_torque = 0
        self.estimated_speed = 0

        self.reply = 0
        self.time = 0
        self.dt = 0
        self.motor_turns = 0

    
    def request_reply(self, command = None):
        self.execute()                                                                                                                        
        pass


    def clear_errors(self, send = True):
        self.command = self.protocol["clear_error_flags"] + 7 * b"\x00"
        if send:
            self.execute()

    def check_errors(self):
        pass


    def pause(self, clear_errors=False, send = True):
        if clear_errors:
            self.clear_errors()

        self.command = (
            self.protocol["motor_stop"] + 7 * b"\x00"
        )  

        if send:
            self.execute()


    def disable(self, clear_errors=True, send = True):
        if clear_errors:
            self.clear_errors()

        self.command = (
            self.protocol["motor_off"] + 7 * b"\x00"
        )

        if send:
            self.execute()
    

    def enable(self, clear_errors=False, send = True):
        if clear_errors:
            self.clear_errors()

        self.command = (
            self.protocol["motor_running"] + 7 * b"\x00"
        )

        if send:
            self.execute()

    def reset(self, go_to_zero=False):
        self.disable(clear_errors=True)
        # print('test')
        self.execute()
        self.enable()


    def go_to_zero(self):
        """Go to the specific point and set new zero at this point"""
        pass


    def set_as_zero(self):
        """Go to the specific point and set new zero at this point"""
        pass


    def set_degrees(self):
        """Set angle and speed scales for degrees"""
        self.angle_scale = 360 / self.encoder_scale
        self.speed_scale = 1 / 10


    def set_radians(self):
        """Set radians for angle and speed scales"""
        self.angle_scale = 2 * pi / self.encoder_scale
        # print(self.angle_scale)
        self.speed_scale = 2 * pi / 360

    def set_units(self, units="rad"):
        if units == "deg":
            self.units = units
            self.set_degrees()
        else:
            self.units == "rad"
            self.set_radians()

    def parse_sensor_data(self, reply):
        """parse the raw sensor data from the CAN frame"""

        self.raw_state_data["temp"] = reply[1]
        self.raw_state_data["current"] = self.from_bytes(reply[2:4])
        self.raw_state_data["speed"] = self.from_bytes(reply[4:6])
        self.raw_state_data["encoder"] = self.from_bytes(reply[6:])
        return self.raw_state_data

    def multiturn_encoder(self, encoder_data, threshold=8000, velocity_data=0):
        # self.velocity_estimate = self.encoder_prev
        if self.encoder_prev - encoder_data >= threshold:
            self.motor_turns += 1
        elif self.encoder_prev - encoder_data <= -threshold:
            self.motor_turns += -1
        self.encoder_prev = encoder_data
        return encoder_data + (self.encoder_scale) * self.motor_turns

    # Parsing from the CAN frames
    def parse_state(self, reply):
        """parse the motor state from CAN frame"""
        self.parse_sensor_data(
            reply)
        self.state["angle"] = self.angle_scale * self.multiturn_encoder(
            self.raw_state_data["encoder"]
        )
        self.state["temp"] = self.temp_scale * self.raw_state_data["temp"]
        self.state["speed"] = self.speed_scale * self.raw_state_data["speed"]
        self.state["current"] = self.current_scale * \
            self.raw_state_data["current"]
        self.state["torque"] = self.torque_constant * self.state["current"]
        return self.state

    # def

    def check_angle(self, reply):
        t = perf_counter()
        dt = self.time - t
        self.estimated_speed = (
            -self.angle_scale * (self.from_bytes(reply[6:]) - self.angle) / dt
        )
        self.time = t

    def parse_status(self, reply):
        self.temp = reply[1]
        self.voltage = reply[3:5]
        self.error = reply[7]
        pass

    def parse_phases(self, reply):
        pass

    def parse_pos(self, reply):
        self.pos = self.from_bytes(reply[1:])

    def parse_pid(self, reply):
        self.gains = {
            "pos": {"p": reply[2], "i": reply[3]},
            "vel": {"p": reply[4], "i": reply[5]},
            "cur": {"p": reply[6], "i": reply[7]},
        }

    def set_pid(self, gains, persistant=False):

        self.command = self.protocol["write_pid_ram"] + b"\x00"
        memory_type = "RAM"

        if persistant:
            print("New PID gains: will be setted to the ROM, type Y to continue")
            user_input = input()
            memory_type = "ROM"
            if user_input == "Y" or user_input == "y":
                self.command = self.protocol["write_pid_rom"] + b"\x00"
            else:
                print("Canceling, gains will be written to RAM")

        # TODO:
        # convert gains dict to array

        gains = [40, 40, 35, 15, 40, 40]
        for gain in gains:
            self.command += self.to_bytes(1, gain, signed=False)

        self.execute()
        print(f"New gains are written to {memory_type}")

    def set_zero(self, persistant=False):
        """ Set a current position as a zero of encoder"""
        self.command = self.protocol["set_encoder_offset"] + 7 * b"\x00"
        memory_type = "RAM"

        if persistant:
            print("Current encoder value will be written as zero, type Y to continue")
            user_input = input()
            memory_type = "ROM"
            if user_input == "Y" or user_input == "y":
                self.command = self.protocol["set_encoder_zero_rom"] + 7 * b"\x00"
            else:
                print("Canceling, zero will be written to RAM")
            self.execute()
        else:
            self.motor_turns = 0
            self.state["angle"] = 0

    # ///////////////////////////
    # ///// Control Modes ///////
    # ///////////////////////////
    #
    # Protocol:
    #   0xA1 - current control
    #   0xA2 - speed control
    #   0xA3 - position control
    #   0xA4 - position control with speed limit
    #
    # ///////////////////////////

    def limiter(self, value, limit):
        if value > limit:
            value = limit
        if value < -limit:
            value = -limit
        return value

    def set_current(self, current, send = True):
        self.desired_current = self.limiter(current, self.current_limit)
        self.command = (
            self.protocol["set_torque"]
            + 3 * b"\x00"
            + self.to_bytes(2, self.desired_current)
            + 2 * b"\x00"
        )
        if send:
            self.execute()
            self.parse_state(self.reply)

    def set_torque(self, torque, torque_limit=None):
        self.set_current(torque/self.torque_constant)

    def set_speed(self, speed, accel_limit=None, send = True):
        # TODO:
        # implement accel limit_functions
        self.desired_speed = 100 * speed / self.speed_scale
        self.command = (
            self.protocol["set_speed"]
            + 3 * b"\x00"
            + self.to_bytes(4, self.desired_speed)
        )
        if send:
            self.execute()
            self.parse_state(self.reply)

    def update_state(self):
        self.parse_state(self.reply)
        # return

    # TODO:

    def set_angle(self, angle, speed_limit=None, send = True):
        # TODO: Check scales
        self.desired_angle = angle
        if speed_limit:
            self.speed_limit = speed_limit

        if self.speed_limit:
            self.command = (
                self.protocol["set_pos_2"]
                + b"\x00"
                + self.to_bytes(2, self.speed_limit)
                + self.to_bytes(4, self.desired_angle)
            )
        else:
            self.command = (
                self.protocol["set_pos_1"]
                + 3 * b"\x00"
                + self.to_bytes(4, self.desired_angle)
            )
        
        if send:
            self.execute()
            self.parse_state(self.reply)

    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    # \\\\\\\\\\\ FUNCTIONS TO IMPLEMENT \\\\\\\\\\\\\\
    # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    # Measurements
    def get_state(self):
        self.command = self.protocol["set_pos_1"] + 3 * \
            b"\x00" + self.to_bytes(4, self.desired_angle)
        self.execute()

        # pass

    def get_vel(self):
        pass

    def get_angle(self):
        pass

    def get_pos(self):
        pass

    def get_encoder_data(self, send = True):
        self.command = self.protocol["read_encoder_data"] + 7 * b"\x00"
        if send:
            self.execute()

        # for ind, phase in enumerate(self.phases_current.keys()):
        raw_encoder = 2*pi*self.from_bytes(self.reply[4:6])/(2**14-1)
        # raw_encoder = 2*pi*self.from_bytes(self.reply[2:4])/(2**14-1)
            # self.phases_current[phase] = self.from_bytes(self.reply[2 + ind*2:4+ind*2])/64
        
        return raw_encoder

    def get_phases_current(self, send = True):
        self.command = self.protocol["read_motor_status_3"] + 7 * b"\x00"
        if send:
            self.execute()

        for ind, phase in enumerate(self.phases_current.keys()):
            self.phases_current[phase] = self.from_bytes(self.reply[2 + ind*2:4+ind*2])/64
        
        return self.phases_current
