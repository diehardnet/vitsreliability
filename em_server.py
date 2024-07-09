#! /usr/bin/env python3

import os
import em_setup.em_configs as em_configs
from em_setup.machine import Machine
# import dnn_log_helper
import paramiko
import pint

import em_setup.fake_avrk4 as AVRK4
# from benches.code.generators import AVRK4
from benches.code.helpers import ip_connection

JETSON_IP = "192.168.1.1" # FIXME: Change for experiment
JETSON_USER = "lucas"
JETSON_PASSWORD = "qwerty0" # FIXME: Change for experiment

assert JETSON_PASSWORD != "qwerty0", "Dummy password not allowed, change for experiment"
assert JETSON_IP != "192.168.1.1", "Dummy ip not allowed, change for experiment"

SERVER_IP = "192.168.1.5"
SERVER_PORT = 1024

PARENT_LOGGER_NAME = "em_server"
LOG_PATH = "/home/lucas/Documents/vitsreliability/logs/em_logs/"

GEMM_1024, GEMM_2048, GEMM_4096, GEMM_8192 = "GEMM_1024", "GEMM_2048", "GEMM_4096", "GEMM_8192"

COMMANDS = {
    GEMM_4096: "LD_LIBRARY_PATH=\"/usr/local/cuda-12/lib64:/home/lucas/git_repo/libLogHelper/build:${LD_LIBRARY_PATH}\" PYTHONPATH=\"/home/lucas/git_repo/libLogHelper/build:${PYTHONPATH}\" PATH=/usr/local/cuda-12/bin:$PATH CUDA_HOME=/usr/local/cuda-12 /home/lucas/git_repo/vitsreliability/main.py --fi_type em --iterations 1000000000000 --goldpath /home/lucas/git_repo/vitsreliability/data/GEMM_4096_fp32.pt --setup_type gemm --floatthreshold 0 --loghelperinterval 5 --precision fp32 --matrix_size 4096 --model gemm --dataset gemm --disableconsolelog",
    GEMM_1024: "LD_LIBRARY_PATH=\"/usr/local/cuda-12/lib64:/home/lucas/git_repo/libLogHelper/build:${LD_LIBRARY_PATH}\" PYTHONPATH=\"/home/lucas/git_repo/libLogHelper/build:${PYTHONPATH}\" PATH=/usr/local/cuda-12/bin:$PATH CUDA_HOME=/usr/local/cuda-12 /home/lucas/git_repo/vitsreliability/main.py --fi_type em --iterations 1000000000000 --goldpath /home/lucas/git_repo/vitsreliability/data/GEMM_1024_fp32.pt --setup_type gemm --floatthreshold 0 --loghelperinterval 5 --precision fp32 --matrix_size 1024 --model gemm --dataset gemm --disableconsolelog",
    GEMM_2048: "LD_LIBRARY_PATH=\"/usr/local/cuda-12/lib64:/home/lucas/git_repo/libLogHelper/build:${LD_LIBRARY_PATH}\" PYTHONPATH=\"/home/lucas/git_repo/libLogHelper/build:${PYTHONPATH}\" PATH=/usr/local/cuda-12/bin:$PATH CUDA_HOME=/usr/local/cuda-12 /home/lucas/git_repo/vitsreliability/main.py --fi_type em --iterations 1000000000000 --goldpath /home/lucas/git_repo/vitsreliability/data/GEMM_2048_fp32.pt --setup_type gemm --floatthreshold 0 --loghelperinterval 5 --precision fp32 --matrix_size 2048 --model gemm --dataset gemm --disableconsolelog",
    GEMM_8192: "LD_LIBRARY_PATH=\"/usr/local/cuda-12/lib64:/home/lucas/git_repo/libLogHelper/build:${LD_LIBRARY_PATH}\" PYTHONPATH=\"/home/lucas/git_repo/libLogHelper/build:${PYTHONPATH}\" PATH=/usr/local/cuda-12/bin:$PATH CUDA_HOME=/usr/local/cuda-12 /home/lucas/git_repo/vitsreliability/main.py --fi_type em --iterations 1000000000000 --goldpath /home/lucas/git_repo/vitsreliability/data/GEMM_8192_fp32.pt --setup_type gemm --floatthreshold 0 --loghelperinterval 5 --precision fp32 --matrix_size 8192 --model gemm --dataset gemm --disableconsolelog",
}

def run_command(command: str, ssh_client: paramiko.SSHClient) -> str:
    _, stdout, stderr = ssh_client.exec_command(command)
    print(stdout.read().decode("utf-8"))
    print(stderr.read().decode("utf-8"))
    print(f"status_code={stdout.channel.recv_exit_status()}")


def main():
    # creating the log directory
    if os.path.isdir(LOG_PATH) is False:
        os.mkdir(LOG_PATH)

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(JETSON_IP, username=JETSON_USER, password=JETSON_PASSWORD, allow_agent=False,look_for_keys=False)

    ureg = pint.UnitRegistry()
    avrk4 = AVRK4.AVRK4(
        connection=ip_connection.IpConnection(em_configs.EM_AVRK4_CONNECTION_IP, em_configs.EM_AVRK4_CONNECTION_PORT),
        configuration=em_configs.EM_AVRK4_CONNECTION_CONF
    )
    # Set amplitude
    avrk4.set_ext_trig()  # Only needed for the external trigger
    # Activate
    avrk4.activate()

    test_name = GEMM_4096
    command = COMMANDS[test_name]
    kill_command = "pkill -9 main.py"

    machine_config = {
        "username": JETSON_USER,
        "password": JETSON_PASSWORD,
        "ip": JETSON_IP,
        "hostname": "lucas-orinnano",
        "boot_waiting_time": 120,
        "max_timeout_time": 20,
        "receive_port": SERVER_PORT,
        "header": command,
        "test_name": test_name,
    }
    machine = Machine(machine_config, SERVER_IP, PARENT_LOGGER_NAME, LOG_PATH)
    machine.start()

    try:
        for delay in em_configs.EM_DELAY_RANGE:
            # Set trigger delay
            delay_ns = pint.Quantity(delay, ureg.ns)
            avrk4.set_trigger_delay(delay_ns)
            for amp_val in em_configs.EM_AMPLITUDE_RANGE:
                amplitude = pint.Quantity(amp_val, ureg.volt)
                avrk4.set_amplitude(amplitude)
                run_command(command, ssh_client)
    finally:
        run_command(kill_command, ssh_client)
        ssh_client.close()
        machine.stop()

    

    
if __name__ == "__main__":
    main()