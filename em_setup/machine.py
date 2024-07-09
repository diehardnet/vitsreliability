import errno
import logging
import os
import socket
import subprocess
import telnetlib
import threading
import time
from typing import Optional
from datetime import datetime

import yaml


from enum import Enum, auto


class ErrorCodes(Enum):
    """Error codes used to identify status in this test
    """
    SUCCESS = auto()
    # When all tries of telnet failed
    TELNET_CONNECTION_ERROR = auto()
    # Codes for RebootMachine
    GENERAL_ERROR = auto()
    HTTP_ERROR = auto()
    CONNECTION_ERROR = auto()
    TIMEOUT_ERROR = auto()
    MAXIMUM_APP_REBOOT_REACHED = auto()
    MAXIMUM_OS_REBOOT_REACHED = auto()
    DISABLED_SOFT_OS_REBOOT = auto()
    HOST_UNREACHABLE = auto()
    THREAD_EVENT_IS_SET = auto()

    def __str__(self) -> str:
        """Override the str method
        :return: the name of the enum as string
        """
        return self.name

class EndStatus(Enum):
    NORMAL_END = "#SERVER_END"
    SOFT_APP_REBOOT = "#SERVER_DUE:soft APP reboot"
    SOFT_OS_REBOOT = "#SERVER_DUE:soft OS reboot"
    HARD_REBOOT = "#SERVER_DUE:power cycle"
    UNKNOWN = "#SERVER_UNKNOWN"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

class DUTLogging:
    """ Device Under Test (DUT) logging class.
    This class will replace the local log procedure that
    each device used to perform in the past.
    """

    def __init__(self, log_dir: str, test_name: str, test_header: str, hostname: str, logger_name: str):
        """ DUTLogging create the log file and writes the header on the first line
        :param log_dir: directory of the logfile
        :param test_name: Name of the test that will be performed, ex: cuda_lava_fp16, zedboard_lenet_int8, etc.
        :param test_header: Specific characteristics of the test, extracted from the configuration files
        :param hostname: Device hostname
        """
        self.__log_dir = log_dir
        self.__test_name = test_name
        self.__test_header = test_header
        self.__hostname = hostname
        self.__logger = logging.getLogger(f"{logger_name}.{__name__}")
        # Create the file when the first message arrives
        self.__filename = None

    def __create_file_if_does_not_exist(self, ecc_status: str):
        if self.__filename is None:
            # log example: 2021_11_15_22_08_25_cuda_trip_half_lava_ECC_OFF_fernando.log
            date = datetime.today()
            date_fmt = date.strftime('%Y_%m_%d_%H_%M_%S')
            log_filename = f"{self.__log_dir}/{date_fmt}_{self.__test_name}_ECC_{ecc_status}_{self.__hostname}.log"
            # Writing the header to the file
            try:
                with open(log_filename, "w") as log_file:
                    begin_str = f"#SERVER_BEGIN Y:{date.year} M:{date.month} D:{date.day} "
                    begin_str += f"TIME:{date.hour}:{date.minute}:{date.second}-{date.microsecond}\n"
                    log_file.write(f"#SERVER_HEADER {self.__test_header}\n")
                    log_file.write(begin_str)
                    self.__filename = log_filename
            except (OSError, PermissionError):
                self.__logger.exception(f"Could not create the file {log_filename}")

    def __call__(self, message: bytes, *args, **kwargs) -> None:
        """ Log a message from the DUT
        :param message: a message is composed of
        <first byte ecc status>
        On file_writer defined as:
        #define ECC_ENABLED 0xE
        #define ECC_DISABLED 0xD
        <message of maximum 1023 bytes>
        1 byte for ecc + 1023 maximum message content = 1024 bytes
        """
        ecc_values = {0xD: "OFF", 0xE: "ON"}
        ecc_status = ecc_values[message[0]]
        self.__create_file_if_does_not_exist(ecc_status=ecc_status)
        message_content = message[1:].decode("ascii")

        if self.__filename:
            with open(self.__filename, "a") as log_file:
                message_content += "\n" if "\n" not in message_content else ""
                log_file.write(message_content)
        else:
            self.__logger.exception("[ERROR in __call__(message) Unable to open file]")

    def finish_this_dut_log(self, end_status: EndStatus):
        """ Check if the file exists and put an END in the last line
        :param end_status status of the ending of the log EndStatus
        """
        if self.__filename:
            with open(self.__filename, "a") as log_file:
                date_fmt = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
                log_file.write(f"{end_status} TIME:{date_fmt}\n")
                self.__filename = None

    def __del__(self):
        # If it is not finished it should
        if self.__filename:
            self.finish_this_dut_log(end_status=EndStatus.UNKNOWN)

    @property
    def log_filename(self):
        return self.__filename

class Machine(threading.Thread):
    # Possible connection string
    __ALL_POSSIBLE_CONNECTION_TYPES = [  # Add more if necessary
        '#IT', '#HEADER', '#BEGIN', '#END', '#INF', '#ERR', "#SDC", "#ABORT"
    ]

    __DATA_SIZE = 4096

    def __init__(self, machine_parameters: dict, server_ip: str, logger_name: str, server_log_path: str,
                 *args, **kwargs):
        """ Initialize a new thread that represents a setup machine
        :param configuration_file: YAML file that contains all information from that specific Device Under Test (DUT)
        :param server_ip: IP of the server
        :param logger_name: Main logger name to store the logging information
        :param server_log_path: directory to store the logs for the test
        :param *args: args that will be passed to threading.Thread
        :param *kwargs: kwargs that will be passed to threading.Thread
        """
        self.__logger_name = f"{logger_name}.{__name__}"
        self.__logger = logging.getLogger(self.__logger_name)
        self.__logger.info(f"Creating a new Machine thread for IP {server_ip}")

        self.__dut_ip = machine_parameters["ip"]
        self.__dut_hostname = machine_parameters["hostname"]
        self.__dut_username = machine_parameters["username"]
        self.__dut_password = machine_parameters["password"]
        self.__boot_waiting_time = machine_parameters["boot_waiting_time"]
        self.__max_timeout_time = machine_parameters["max_timeout_time"]
        self.__receiving_port = machine_parameters["receive_port"]
        # self.__cmd_header = machine_parameters["cmd_header"]
        self.__test_name = machine_parameters["test_name"]
        self.__test_header = machine_parameters["header"]
        self.__disable_os_soft_reboot = False
        if "disable_os_soft_reboot" in machine_parameters:
            self.__disable_os_soft_reboot = machine_parameters["disable_os_soft_reboot"] is True

        self.__dut_logging_obj = None
        self.__dut_log_path = f"{server_log_path}/{self.__dut_hostname}"
        # make sure that the path exists
        if os.path.isdir(self.__dut_log_path) is False:
            os.mkdir(self.__dut_log_path)

        # Configure the socket
        self.__messages_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__messages_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.__messages_socket.bind((server_ip, self.__receiving_port))
        self.__messages_socket.settimeout(self.__max_timeout_time)

        super(Machine, self).__init__(*args, **kwargs) 

    def run(self) -> None:
        self.__dut_logging_obj = DUTLogging(log_dir=self.__dut_log_path, test_name=self.__test_name, test_header=self.__test_header, hostname=self.__dut_hostname,
                                                        logger_name=self.__logger_name)
        while 1:
            try:
                data, address = self.__messages_socket.recvfrom(self.__DATA_SIZE)
                self.__dut_logging_obj(message=data)
                data_decoded = data.decode("ascii")[1:]
                connection_type_str = "UnknownConn:" + data_decoded[:10]
                for substring in self.__ALL_POSSIBLE_CONNECTION_TYPES:
                    # It must start from the 1, as the 0 is the ECC defining byte
                    if data_decoded.startswith(substring):
                        connection_type_str = substring
                        break

                self.__logger.debug(f"{connection_type_str} - Connection from {self}")

            except (TimeoutError, socket.timeout):
                self.__logger.debug(f"Connection timeout.")
