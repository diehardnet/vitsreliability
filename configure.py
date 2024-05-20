#!/usr/bin/python3
import argparse
import configparser
import json
import os.path
import pathlib
import time
from pathlib import Path
from socket import gethostname

import configs

ALL_DNNS = {
    # PATH TO CHECKPOINT, CONFIG FILE, PRECISIONS, SETUP_TYPE,
    # BATCH SIZE, TEST SAMPLES, hardening types, micro operation
    configs.GROUNDING_DINO_SWINT_OGC: (
        "groundingdino_swint_ogc.pth", "GroundingDINO_SwinT_OGC.py", [configs.FP32], configs.GROUNDING_DINO,
        1, 8, {None, "hardenedid"}
    ),
    configs.GROUNDING_DINO_SWINB_COGCOOR: (
        "groundingdino_swinb_cogcoor.pth", "GroundingDINO_SwinB_cfg.py", [configs.FP32], configs.GROUNDING_DINO,
        1, 8, {None, "hardenedid"}
    ),
    # TODO: other setups configs.SELECTIVE_ECC, configs.VITS
}

MICRO_SETUPS = {

}

LOG_NVML = False
FLOAT_THRESHOLD = 1e-2
SAVE_LOGITS = True
CONFIG_FILE = "/etc/radiation-benchmarks.conf"
ITERATIONS = int(1e12)


def execute_cmd(generate_cmd):
    if os.system(generate_cmd) != 0:
        raise OSError(f"Could not execute command {generate_cmd}")


def general_configure():
    try:
        config = configparser.RawConfigParser()
        config.read(CONFIG_FILE)
        server_ip = config.get('DEFAULT', 'serverip')
    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))
    hostname = gethostname()
    home = str(Path.home())
    jsons_path = f"data/{hostname}_jsons"
    if os.path.isdir(jsons_path) is False:
        os.makedirs(jsons_path, exist_ok=True)
    current_directory = os.getcwd()
    return current_directory, home, jsons_path, server_ip


def configure():
    current_directory, home, jsons_path, server_ip = general_configure()
    script_name = "main.py"

    for dnn_model, dnn_cfg in ALL_DNNS.items():
        weights_file, config_file, precisions, setup_type, batch_size, test_samples, hardened = dnn_cfg
        for hardening in hardened:
            for float_precision in precisions:
                configuration_name = (f"{dnn_model}_{float_precision}_{hardening}_"
                                      f"{setup_type}_{test_samples}_{batch_size}")
                json_file_name = f"{jsons_path}/{configuration_name}.json"
                data_dir = f"{current_directory}/data"
                gold_path = f"{data_dir}/{configuration_name}.pt"
                checkpoint_path = f"{data_dir}/weights_grounding_dino/{weights_file}"
                config_path = f"{current_directory}/GroundingDINO/groundingdino/config/{config_file}"
                parameters = [
                    "CUBLAS_WORKSPACE_CONFIG=:4096:8 ",
                    f"{current_directory}/{script_name}",
                    f"--iterations {ITERATIONS}",
                    f"--testsamples {test_samples}",
                    f"--batchsize {batch_size}",
                    f"--checkpointpath {checkpoint_path}",
                    f"--goldpath {gold_path}",
                    f"--model {dnn_model}",
                    f"--configpath {config_path}",
                    f"--setup_type grounding_dino",
                    f"--floatthreshold {FLOAT_THRESHOLD}",
                    f"--loghelperinterval 1",
                    f"--precision {float_precision}",
                    f"--microop Attention",
                    f"--{hardening}" if hardening else '',
                    f"--savelogits" if SAVE_LOGITS else '',
                    f"--lognvml" if LOG_NVML else ''
                ]
                execute_parameters = parameters + ["--disableconsolelog"]
                command_list = [{
                    "killcmd": f"pkill -9 -f {script_name}",
                    "exec": " ".join(execute_parameters),
                    "codename": dnn_model,
                    "header": " ".join(execute_parameters)
                }]

                generate_cmd = " ".join(parameters + ["--generate"])
                # dump json
                with open(json_file_name, "w") as json_fp:
                    json.dump(obj=command_list, fp=json_fp, indent=4)

                print(f"Executing generate for {generate_cmd}")
                execute_cmd(generate_cmd)

    print("Json creation and golden generation finished")
    print(f"You may run: scp -r {jsons_path} carol@{server_ip}:{home}/radiation-setup/machines_cfgs/")


def test_all_jsons(enable_console_logging, timeout=30):
    hostname = gethostname()
    current_directory = os.getcwd()
    jsons_path = f"{current_directory}/data/{hostname}_jsons"
    print("Looping through all the files in", jsons_path)
    for filename in pathlib.Path(jsons_path).glob('*.json'):
        # checking if it is a file
        if os.path.isfile(filename):
            with open(filename, "r") as fp:
                json_data = json.load(fp)

            for v in json_data:
                exec_str = v["exec"].replace("--disableconsolelog", "") if enable_console_logging else v["exec"]
                print("EXECUTING", exec_str)
                os.system(exec_str + "&")
                time.sleep(timeout)
                os.system(v["killcmd"])


def main():
    parser = argparse.ArgumentParser(description='Configure a setup', add_help=True)
    parser.add_argument('--testjsons', default=0,
                        help="How many seconds to test the jsons, if 0 (default) it does the configure", type=int)
    parser.add_argument('--enableconsole', default=False, action="store_true",
                        help="Enable console logging for testing")

    args = parser.parse_args()

    if args.testjsons != 0:
        test_all_jsons(enable_console_logging=args.enableconsole, timeout=args.testjsons)
    else:
        configure()


if __name__ == "__main__":
    main()
