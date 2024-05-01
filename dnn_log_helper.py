"""
This wrapper is only if you don't want to use libLogHelper
"""
import inspect
import sys
import traceback

import log_helper

import configs

__LOGGING_ACTIVE = False


def start_setup_log_file(activate_logging: bool, **kwargs) -> None:
    global __LOGGING_ACTIVE
    __LOGGING_ACTIVE = activate_logging
    if __LOGGING_ACTIVE:
        dnn_name = kwargs.get("model")
        log_helper.start_log_file(dnn_name, " ".join({f"{k}:{v}" for k, v in kwargs.items()}))
        log_helper.set_max_errors_iter(configs.MAXIMUM_ERRORS_PER_ITERATION)
        log_helper.set_max_infos_iter(configs.MAXIMUM_INFOS_PER_ITERATION)
        interval_print = configs.ITERATION_INTERVAL_LOG_HELPER_PRINT[dnn_name]
        log_helper.set_iter_interval_print(interval_print)


def start_iteration() -> None:
    if __LOGGING_ACTIVE:
        log_helper.start_iteration()


def end_iteration() -> None:
    if __LOGGING_ACTIVE:
        log_helper.end_iteration()


def end_log_file() -> None:
    if __LOGGING_ACTIVE:
        log_helper.end_log_file()


def log_info_detail(info_detail: str) -> None:
    if __LOGGING_ACTIVE:
        log_helper.log_info_detail(info_detail)


def log_error_detail(error_detail: str) -> None:
    if __LOGGING_ACTIVE:
        log_helper.log_error_detail(error_detail)


def log_error_count(error_count: int) -> None:
    if __LOGGING_ACTIVE:
        log_helper.log_error_count(error_count)


def __getattr__(name: str) -> str:
    if __LOGGING_ACTIVE:
        if name == 'log_file_name':
            return log_helper.get_log_file_name()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def log_and_crash(fatal_string: str) -> None:
    caller_frame_record = inspect.stack()[1]  # 0 represents this line
    # 1 represents line at caller
    frame = caller_frame_record[0]
    info = inspect.getframeinfo(frame)
    fatal_log_string = f"SETUP_ERROR:{fatal_string} FILE:{info.filename}:{info.lineno} F:{info.function}"
    fatal_log_string += f"\nTRACEBACK:{traceback.format_exc()}"
    # It is better to always show the exception to stdout
    print(fatal_log_string)
    # Also save to tmp in the case of logging is not active
    with open(configs.TMP_CRASH_FILE, "w") as tmp_fp:
        tmp_fp.write(f"{fatal_log_string}\n")
    log_info_detail(info_detail=fatal_log_string)
    end_log_file()
    sys.exit(1)
