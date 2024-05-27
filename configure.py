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

CURRENT_DIR = os.getcwd()

GROUNDING_DINO_SAMPLES = 8

# CHECKPOINT, CFG FILE, PRECISIONS, SETUP_TYPE, BATCH SIZE, TEST SAMPLES, hardening types, micro op, log interval
# If GROUDING_DINO_SETUP for JPL images must also include as last parameter "imgspath" and the dataset
GROUNDING_DINO_SETUPS = {
    configs.GROUNDING_DINO_SWINT_OGC: (
        configs.GROUNDING_DINO_SWINT_OGC,
        f"{CURRENT_DIR}/data/weights_grounding_dino/groundingdino_swint_ogc.pth",
        f"{CURRENT_DIR}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        [configs.FP32], configs.GROUNDING_DINO, 1, GROUNDING_DINO_SAMPLES, {None, "hardenedid"}, "Attention", 1,
        "ignore", configs.COCO
    ),
    configs.GROUNDING_DINO_SWINB_COGCOOR: (
        configs.GROUNDING_DINO_SWINB_COGCOOR,
        f"{CURRENT_DIR}/data/weights_grounding_dino/groundingdino_swinb_cogcoor.pth",
        f"{CURRENT_DIR}/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        [configs.FP32], configs.GROUNDING_DINO, 1, GROUNDING_DINO_SAMPLES, {None, "hardenedid"}, "Attention", 1,
        "ignore", configs.COCO
    ),
}

GROUNDING_DINO_SETUPS_JPL = {
    "jpl_swint_ogc": (
        configs.GROUNDING_DINO_SWINT_OGC,
        f"{CURRENT_DIR}/data/weights_grounding_dino/groundingdino_swint_ogc.pth",
        f"{CURRENT_DIR}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        [configs.FP32], configs.GROUNDING_DINO, 1, 0, {None, "hardenedid"}, "Attention", 1,
        f"{CURRENT_DIR}/data/jpl_samples/jpl_images.txt", configs.CUSTOM_DATASET
    ),
    "jpl_swinb_cogcoor":  (
        configs.GROUNDING_DINO_SWINB_COGCOOR,
        f"{CURRENT_DIR}/data/weights_grounding_dino/groundingdino_swinb_cogcoor.pth",
        f"{CURRENT_DIR}/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        [configs.FP32], configs.GROUNDING_DINO, 1, 0, {None, "hardenedid"}, "Attention", 1,
        f"{CURRENT_DIR}/data/jpl_samples/jpl_images.txt", configs.CUSTOM_DATASET
    ),
}

BATCH_SIZE_VITS = 32
TEST_SAMPLES_VITS = BATCH_SIZE_VITS
LOG_INTERVAL_VITS = 1

VITS_SETUPS = {
    # The parameter micro op for ViTs is ignored
    # TODO: other setups configs.SELECTIVE_ECC, configs.VITS
    # setup for ViTs, TODO: add int8 and hardened ID configs
    model_i: (
        model_i, None, None, [configs.FP32, configs.FP16], configs.VITS, BATCH_SIZE_VITS,
        TEST_SAMPLES_VITS,
        {None, "hardenedid"}, None, LOG_INTERVAL_VITS,
        "ignore", configs.IMAGENET
    ) for model_i in [
        configs.VIT_BASE_PATCH16_224, configs.VIT_BASE_PATCH16_384,
        configs.SWIN_BASE_PATCH4_WINDOW7_224, configs.SWIN_BASE_PATCH4_WINDOW12_384,
        configs.DEIT_BASE_PATCH16_224, configs.DEIT_BASE_PATCH16_384
    ]
}

MICRO_BATCHED_SAMPLES = 32
MICRO_LOG_INTERVAL = 100
MICRO_SETUPS = {
    **{f"swin_{micro_op}": (
        configs.SWIN_BASE_PATCH4_WINDOW12_384, "ignore", "ignore", [configs.FP32, configs.FP16], configs.MICROBENCHMARK,
        MICRO_BATCHED_SAMPLES, MICRO_BATCHED_SAMPLES, {None}, micro_op, MICRO_LOG_INTERVAL, "ignore", configs.IMAGENET
    ) for micro_op in [configs.SWIN_BLOCK, configs.MLP, configs.WINDOW_ATTENTION]},
    **{f"swin_{micro_op}": (
        configs.SWIN_BASE_PATCH4_WINDOW7_224, "ignore", "ignore", [configs.FP32, configs.FP16], configs.MICROBENCHMARK,
        MICRO_BATCHED_SAMPLES, MICRO_BATCHED_SAMPLES, {None}, micro_op, MICRO_LOG_INTERVAL, "ignore", configs.IMAGENET
    ) for micro_op in [configs.SWIN_BLOCK, configs.MLP, configs.WINDOW_ATTENTION]},
    **{f"vit_{micro_op}": (
        configs.VIT_BASE_PATCH16_384, "ignore", "ignore", [configs.FP32, configs.FP16], configs.MICROBENCHMARK,
        MICRO_BATCHED_SAMPLES, MICRO_BATCHED_SAMPLES, {None}, micro_op, MICRO_LOG_INTERVAL, "ignore", configs.IMAGENET
    ) for micro_op in [configs.ATTENTION, configs.BLOCK, configs.MLP]}
}

# Change for configuring
SETUPS = dict()
# SETUPS.update(VITS_SETUPS)
SETUPS.update(GROUNDING_DINO_SETUPS)
# SETUPS.update(GROUNDING_DINO_SETUPS_JPL)
# SETUPS.update(MICRO_SETUPS)

LOG_NVML = False  # FIXME: Logging NVML is not in a good shape
FLOAT_THRESHOLD = 0
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
    return home, jsons_path, server_ip


def configure():
    home, jsons_path, server_ip = general_configure()
    script_name = "main.py"

    for dnn_key, dnn_cfg in SETUPS.items():
        (dnn, weights_file, config_file, precisions, setup_type,
         batch_size, test_samples, hardened, micro_op, log_interval, imgs_path, dataset) = dnn_cfg
        for hardening in hardened:
            for float_precision in precisions:
                configuration_name = f"{dnn}_{float_precision}_{hardening}_"
                configuration_name += f"{setup_type}_{test_samples}_{batch_size}"
                if dnn != dnn_key:
                    configuration_name = f"{dnn_key}_{configuration_name}"
                json_file_name = f"{jsons_path}/{configuration_name}.json"
                gold_path = f"{CURRENT_DIR}/data/{configuration_name}.pt"
                # if float_precision == configs.INT8:
                #     config_file = os.path.join(CURRENT_DIR, "FasterTransformer/examples/pytorch/swin/Swin-Transformer
                #     -Quantization/SwinTransformer/configs/swin/", f"{dnn}.yaml")

                parameters = [
                    # "CUBLAS_WORKSPACE_CONFIG=:4096:8 ",
                    f"{CURRENT_DIR}/{script_name}",
                    f"--iterations {ITERATIONS}",
                    f"--testsamples {test_samples}",
                    f"--batchsize {batch_size}",
                    f"--checkpointpath {weights_file}",
                    f"--goldpath {gold_path}",
                    f"--model {dnn}",
                    f"--configpath {config_file}",
                    f"--setup_type {setup_type}",
                    f"--floatthreshold {FLOAT_THRESHOLD}",
                    f"--loghelperinterval {log_interval}",
                    f"--precision {float_precision}",
                    f"--microop {micro_op}" if micro_op else '',
                    f"--{hardening}" if hardening else '',
                    f"--savelogits" if SAVE_LOGITS else '',
                    f"--lognvml" if LOG_NVML else '',
                    # f"--cfg {cfg_path}" if float_precision == configs.INT8 else '',
                    # f"--int8-mode {1}" if float_precision == configs.INT8 else '',
                    f"--resume {weights_file}" if float_precision == configs.INT8 else '',
                    f"--imgspath {imgs_path}",
                    f"--dataset {dataset}"
                ]
                execute_parameters = parameters + ["--disableconsolelog"]
                command_list = [{
                    "killcmd": f"pkill -9 -f {script_name}",
                    "exec": " ".join(execute_parameters),
                    "codename": dnn,
                    "header": " ".join(execute_parameters)
                }]

                generate_cmd = " ".join(parameters + ["--generate"])
                # dump json
                with open(json_file_name, "w") as json_fp:
                    json.dump(obj=command_list, fp=json_fp, indent=4)

                print(f"Executing generate for {generate_cmd}")
                execute_cmd(generate_cmd)

    print("Json creation and golden generation finished")
    print("Set 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in the .bashrc file")
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
