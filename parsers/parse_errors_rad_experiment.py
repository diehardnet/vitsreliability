#!/usr/bin/python3
import argparse
import copy
import datetime
import os
import re
from typing import List

import pandas as pd


def search_header(lines: List[str], log):
    pattern = None
    if "nmnist" in log:
        pattern = "#HEADER.*testsamples=(\d+).*batchsize=(\d+)"

    get_flags = {
        # "iterations": r"(\d+)",
        "testsamples": r"(\d+)",
        "batchsize": r"(\d+)",
        # "goldpath": r"(\S+)",
        "model": r"(\S+)",
        "setup_type": r"(\S+)",
        "floatthreshold": r"(\S+)",
        "loghelperinterval": r"(\d+)",
        "precision": r"(\S+)",
        "microop": r"(\S+)",
        "hardenedid": r"(\S+)",
        # "savelogits": r"(\S+)",
        "lognvml": r"(\S+)",
        "dataset": r"(\S+)",
    }

    for line in lines:
        parsed_data = dict()
        if "#HEADER" in line and pattern is None:
            for flag, pattern in get_flags.items():
                full_pattern = f".*{flag}:{pattern}.*"
                m = re.search(full_pattern, line)
                if m:
                    parsed_data[flag] = m.group(1)
                else:
                    print(flag, pattern)
                    raise ValueError(line + " " + log)
            if len(parsed_data.keys()) == len(get_flags.keys()):
                return parsed_data
        if pattern:
            if re.search(pattern, line):
                parsed_data["testsamples"] = int(re.match(pattern, line).group(1))
                parsed_data["batch_size"] = int(re.match(pattern, line).group(2))
                parsed_data["model"] = "SNN_NMNIST"

                return parsed_data

    raise ValueError(f"Not possible to parse {log}")


def search_error_criticality(err_str: str) -> re.Match:
    m = re.match(r"#ERR.*critical.*", err_str)
    if not m:
        m = re.match(r"#ERR.*critical.*glb:(\d+) flb:(\d+)", err_str)
    return m


def parse_log_file(log_path: str) -> List[dict]:
    # ...log/2022_09_15_16_00_43_PyTorch-c100_res44_test_02_relu6-bn_200_epochs_ECC_OFF_carolinria.log
    pattern = r".*/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_\S+_ECC_(\S+)_(\S+).log"
    m = re.match(pattern, log_path)
    if m:
        year, month, day, hour, minute, seconds, ecc, hostname = m.groups()
        year, month, day, hour, minute, seconds = [int(i) for i in [year, month, day, hour, minute, seconds]]
        start_dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=seconds)
        data_list = list()

        with open(log_path) as log_fp:
            lines = log_fp.readlines()
        if len(lines) <= 4:
            return []
        data_dict = search_header(lines=lines, log=log_path)
        data_dict["start_dt"] = start_dt
        data_dict["ecc"] = ecc
        data_dict["hostname"] = hostname
        data_dict["logfile"] = os.path.basename(log_path)

        last_acc_time = 0
        critical_sdc = 0
        for line in lines:
            ct_m = search_error_criticality(err_str=line)
            if ct_m:
                critical_sdc += 1
                # golden, output = ct_m.group(1), ct_m.group(2)
                # evil_sdc += int(output != ground_truth)
                # benign_sdc += int(output == ground_truth and golden != ground_truth)
            elif "critical" in line:
                raise ValueError(f"Not a valid line {line}")

            sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)
            if sdc_m:
                it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                last_acc_time = float(acc_time)
                curr_data = copy.deepcopy(data_dict)
                curr_data.update(
                    dict(it=it, ker_time=float(ker_time), acc_time=0, ker_err=ker_err, acc_err=acc_err, sdc=1,
                         critical_sdc=int(critical_sdc != 0), hostname=hostname)
                )
                data_list.append(curr_data)
                critical_sdc = 0

        if data_list:
            data_list[-1]["acc_time"] = last_acc_time
        return data_list


def parse_args() -> argparse.Namespace:
    """ Parse the args and return an args namespace and the tostring from the args    """
    parser = argparse.ArgumentParser(description='PyTorch DNN radiation parser', add_help=False)
    # parser = argparse.ArgumentParser(description='PyTorch DNN radiation setup')
    parser.add_argument('--logdir', help="Path to the directory that contains the logs", required=True)

    args, remaining_argv = parser.parse_known_args()

    return args


def main():
    args = parse_args()
    data_list = list()
    for subdir, dirs, files in os.walk(args.logdir):
        if any([i in subdir for i in ["carola20001", "carola20002", "carolp20002", "carolp22003", "carola20003"]]):
            print("Parsing", subdir)
            for file in files:
                path = os.path.join(subdir, file)
                new_line = parse_log_file(log_path=path)
                if new_line:
                    data_list.extend(new_line)

    df = pd.DataFrame(data_list)
    df = df.fillna(0)
    df.to_csv("../data/parsed_logs_rad_may_2024.csv", index=False)


if __name__ == '__main__':
    main()
