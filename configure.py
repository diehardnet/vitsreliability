#!/usr/bin/python3
import argparse
import configparser
import json
import os.path
import time
from pathlib import Path
from socket import gethostname

import pandas as pd

import configs

# Micro benchmarks CSV reference
MICROBENCHMARKS_CSV = "/home/carol/maximals/data/microbenchmarks/to_evaluate_microbenchmarks.csv"
# It is either false or true
# FIXME: In the future - now it is not possible to save torch compile models
TORCH_COMPILE_CONFIGS = {False}  # torch.cuda.get_device_capability()[0] >= 7}
HARDENING_TYPES = {None, "hardenedid"}
# don't mess with pointers (this is to avoid changing the original lists)
ALL_DNNS = list()
ALL_DNNS += configs.CNN_CONFIGS
ALL_DNNS += configs.VIT_CLASSIFICATION_CONFIGS

MICRO_CONFIGS = [
    "id_19_name_blocks.0.mlp_class_Mlp_params_8393728_output_size_1052672.json",
    "id_7_name_blocks.0_class_Block_params_12596224_output_size_1052672.json",
    "id_5_name_norm_pre_class_LayerNorm_params_2048_output_size_1052672.json",
    "id_9_name_blocks.0.attn_class_Attention_params_4198400_output_size_1052672.json"
]

CONFIG_FILE = "/etc/radiation-benchmarks.conf"
ITERATIONS = int(1e12)
BATCH_SIZE = 4

TEST_SAMPLES = {
    **{k: BATCH_SIZE * 10 for k in configs.CNN_CONFIGS},
    **{k: BATCH_SIZE * 10 for k in configs.VIT_CLASSIFICATION_CONFIGS},
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

    script_name = "setuppuretorch.py"
    for torch_compile in TORCH_COMPILE_CONFIGS:
        for hardening in HARDENING_TYPES:
            for dnn_model in ALL_DNNS:
                if dnn_model in configs.CNN_CONFIGS and hardening is not None:
                    continue
                configuration_name = f"{dnn_model}_torch_compile_{torch_compile}_hardening_{hardening}"
                json_file_name = f"{jsons_path}/{configuration_name}.json"
                data_dir = f"{current_directory}/data"
                gold_path = f"{data_dir}/{configuration_name}.pt"
                checkpoint_dir = f"{data_dir}/checkpoints"
                parameters = [
                    f"{current_directory}/{script_name}",
                    f"--iterations {ITERATIONS}",
                    f"--testsamples {TEST_SAMPLES[dnn_model]}",
                    f"--batchsize {BATCH_SIZE}",
                    f"--checkpointdir {checkpoint_dir}",
                    f"--goldpath {gold_path}",
                    f"--model {dnn_model}",
                    f"--usetorchcompile" if torch_compile is True else '',
                    f"--{hardening}" if hardening else ''
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


def configure_microbenchmarks():
    not_working = [
        "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k/"
        "id_7_name_blocks.0_class_EvaBlock_params_7092224_output_size_3148800.pt",
        "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k/"
        "id_24_name_blocks.0.mlp.norm_class_LayerNorm_params_4096_output_size_8396800.pt",
        "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k/"
        "id_22_name_blocks.0.mlp.act_class_SiLU_params_0_output_size_8396800.pt",
        "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k/"
        "id_21_name_blocks.0.mlp.act_class_GELU_params_0_output_size_4210688.pt"
    ]
    current_directory, home, jsons_path, server_ip = general_configure()
    script_name = "setupmicrobenchmarks.py"

    # First execute the extract micro benchmarks
    # execute_cmd("./extract_micro_operations.py")
    possible_micros = pd.read_csv(MICROBENCHMARKS_CSV)

    for micro in possible_micros.to_dict(orient="records"):
        gold_path = "/home/carol/maximals/" + micro["full_path"]
        basename = os.path.basename(gold_path).strip().replace(".pt", "")
        if any([basename in ni for ni in not_working]):
            continue
        json_file_name = f"{jsons_path}/{basename}.json"
        parameters = [
            f"{current_directory}/{script_name}",
            f"--iterations {ITERATIONS}",
            f"--testsamples 1",
            f"--batchsize 1",
            f"--checkpointdir 0",
            f"--goldpath {gold_path}",
            # f"--model {dnn_model}",
            # f"--usetorchcompile" if torch_compile is True else ''
        ]
        execute_parameters = parameters + ["--disableconsolelog"]
        command_list = [{
            "killcmd": f"pkill -9 -f {script_name}",
            "exec": " ".join(execute_parameters),
            "codename": basename,
            "header": " ".join(execute_parameters)
        }]

        generate_cmd = " ".join(parameters + ["--generate"])
        # dump json
        with open(json_file_name, "w") as json_fp:
            json.dump(obj=command_list, fp=json_fp, indent=4)

        print(f"Executing generate for {generate_cmd}")
        try:
            execute_cmd(generate_cmd)
        except OSError:
            print(gold_path)
    print("Json creation and golden generation finished")
    print(f"You may run: scp -r {jsons_path} carol@{server_ip}:{home}/radiation-setup/machines_cfgs/")


def test_all_jsons(enable_console_logging, test_micro, timeout=30):
    hostname = gethostname()
    current_directory = os.getcwd()
    if test_micro is True:
        for micro_json in MICRO_CONFIGS:
            file = f"{current_directory}/data/{hostname}_jsons/{micro_json}"
            with open(file, "r") as fp:
                json_data = json.load(fp)

            for v in json_data:
                exec_str = v["exec"].replace("--disableconsolelog", "") if enable_console_logging else v["exec"]
                print("EXECUTING", exec_str)
                os.system(exec_str + "&")
                time.sleep(timeout)
                os.system(v["killcmd"])
    else:
        for torch_compile in TORCH_COMPILE_CONFIGS:
            for hardening in HARDENING_TYPES:
                for dnn_model in ALL_DNNS:
                    if dnn_model in configs.CNN_CONFIGS and hardening is not None:
                        continue
                    file = f"{current_directory}/data/{hostname}_jsons"
                    file += f"/{dnn_model}_torch_compile_{torch_compile}_hardening_{hardening}.json"
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
    parser.add_argument('--micro', default=False, action="store_true",
                        help="Generate jsons for micro instead of DNNs")

    args = parser.parse_args()

    if args.testjsons != 0:
        test_all_jsons(enable_console_logging=args.enableconsole, test_micro=args.micro, timeout=args.testjsons)
    else:
        if args.micro:
            configure_microbenchmarks()
        else:
            configure()


if __name__ == "__main__":
    main()
