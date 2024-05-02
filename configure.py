#!/usr/bin/python3
import argparse
import configparser
import json
import os.path
import time
from pathlib import Path
from socket import gethostname

import configs

# It is either false or true
# FIXME: In the future - now it is not possible to save torch compile models
TORCH_COMPILE_CONFIGS = {False}  # torch.cuda.get_device_capability()[0] >= 7}
HARDENING_TYPES = {None , "hardenedid"}
FLOAT_PRECISION = {"fp32"}
FLOAT_THRESHOLD = 1e-3

ALL_DNNS = {
    configs.GROUNDING_DINO_SWINT_OGC: ("groundingdino_swint_ogc.pth", "GroundingDINO_SwinT_OGC.py"),
    configs.GROUNDING_DINO_SWINB_COGCOOR: ("groundingdino_swinb_cogcoor.pth", "GroundingDINO_SwinB_cfg.py")
}

CONFIG_FILE = "/etc/radiation-benchmarks.conf"
ITERATIONS = int(1e12)
BATCH_SIZE = 1

TEST_SAMPLES = {
    **{k: BATCH_SIZE * 16 for k in ALL_DNNS},
}


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
    env_vars = ("PYTHONPATH=/home/carol/vitsreliability/GroundingDINO::/home/carol/libLogHelper/build "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 ")
    script_name = "main.py"
    for torch_compile in TORCH_COMPILE_CONFIGS:
        for hardening in HARDENING_TYPES:
            for float_precision in FLOAT_PRECISION:
                for dnn_model, (weights_file, config_file) in ALL_DNNS.items():
                    configuration_name = (f"{dnn_model}_{float_precision}_torch_compile_{torch_compile}"
                                          f"_hardening_{hardening}")
                    json_file_name = f"{jsons_path}/{configuration_name}.json"
                    data_dir = f"{current_directory}/data"
                    gold_path = f"{data_dir}/{configuration_name}.pt"
                    checkpoint_path = f"{data_dir}/weights_grounding_dino/{weights_file}"
                    config_path = f"{current_directory}/GroundingDINO/groundingdino/config/{config_file}"
                    parameters = [
                        env_vars,
                        f"{current_directory}/{script_name}",
                        f"--iterations {ITERATIONS}",
                        f"--testsamples {TEST_SAMPLES[dnn_model]}",
                        f"--batchsize {BATCH_SIZE}",
                        f"--checkpointpath {checkpoint_path}",
                        f"--goldpath {gold_path}",
                        f"--model {dnn_model}",
                        f"--usetorchcompile" if torch_compile is True else '',
                        f"--{hardening}" if hardening else '',
                        f"--configpath {config_path}",
                        f"--setup_type grounding_dino",
                        f"--floatthreshold {FLOAT_THRESHOLD}",
                        f"--loghelperinterval 1",
                        f"--precision {float_precision}"
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
    for torch_compile in TORCH_COMPILE_CONFIGS:
        for hardening in HARDENING_TYPES:
            for float_precision in FLOAT_PRECISION:
                for dnn_model in ALL_DNNS:
                    configuration_name = (f"{dnn_model}_{float_precision}_torch_compile_{torch_compile}"
                                          f"_hardening_{hardening}")
                    file = f"{current_directory}/data/{hostname}_jsons/{configuration_name}.json"
                    with open(file, "r") as fp:
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
