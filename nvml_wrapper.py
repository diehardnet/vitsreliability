import time
import pynvml
import dnn_log_helper as dnn_lh


def handle_error(query: str, err: pynvml.NVMLError) -> str:
    err_str = f"{query}-"
    if err.value == pynvml.NVML_ERROR_NOT_SUPPORTED:
        err_str += "NA"
    else:
        err_str += err.__str__()
    return err_str


# class NVMLWrapperThread(threading.Thread):
class NVMLWrapper:
    __device_index = 0
    __enable_query = False

    def __init__(self, enable_query: bool = False):
        # super(NVMLWrapperThread, self).__init__(*args, **kwargs)
        self.__enable_query = enable_query
        if self.__enable_query:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.__device_index)
            except pynvml.NVMLError as err:
                dnn_lh.log_info_detail(handle_error(query="get_handler", err=err))
                return

            try:
                # Check if ECC is enabled
                curr_state, pending_state = pynvml.nvmlDeviceGetEccMode(self.handle)
                self.ecc_available = (curr_state == pynvml.NVML_FEATURE_ENABLED and
                                      pending_state == pynvml.NVML_FEATURE_ENABLED)
                # At the beginning we assume all locations can be traced
            except pynvml.NVMLError:
                self.ecc_available = False

            self.__set_ecc_locations()

    def __set_ecc_locations(self):
        """
        Test only the ECC locations that can be traced
        """
        self.possible_ecc_locations = list()
        self.possible_total_ecc = list()
        if self.ecc_available is False:
            return

        ecc_error_types = {pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED}
        ecc_counter_types = {pynvml.NVML_VOLATILE_ECC, pynvml.NVML_AGGREGATE_ECC}
        ecc_mem_locations = {
            pynvml.NVML_MEMORY_LOCATION_L1_CACHE, pynvml.NVML_MEMORY_LOCATION_L2_CACHE,
            pynvml.NVML_MEMORY_LOCATION_DRAM, pynvml.NVML_MEMORY_LOCATION_DEVICE_MEMORY,
            pynvml.NVML_MEMORY_LOCATION_REGISTER_FILE, pynvml.NVML_MEMORY_LOCATION_TEXTURE_MEMORY,
            pynvml.NVML_MEMORY_LOCATION_TEXTURE_SHM, pynvml.NVML_MEMORY_LOCATION_CBU, pynvml.NVML_MEMORY_LOCATION_SRAM
        }
        not_possible_ecc_locations = ""
        for error_type in ecc_error_types:
            for counter_type in ecc_counter_types:
                for location_type in ecc_mem_locations:
                    try:
                        pynvml.nvmlDeviceGetMemoryErrorCounter(handle=self.handle, errorType=error_type,
                                                               counterType=counter_type, locationType=location_type)
                        # It is possible to get it
                        self.possible_ecc_locations.append((error_type, counter_type, location_type))
                    except pynvml.NVMLError as err:
                        not_possible_ecc_locations += handle_error(query=f"{error_type}-{counter_type}-{location_type}",
                                                                   err=err) + ";"
                try:
                    pynvml.nvmlDeviceGetTotalEccErrors(handle=self.handle, errorType=error_type,
                                                       counterType=counter_type)
                    self.possible_total_ecc.append((error_type, counter_type))
                except pynvml.NVMLError as err:
                    not_possible_ecc_locations += handle_error(query=f"total", err=err) + ";"

        dnn_lh.log_info_detail(f"ImpossibleECCs:{not_possible_ecc_locations}")

    def query(self) -> str:
        if self.__enable_query:
            data_list = self.__query_nvml()
            if data_list:
                return ";".join(map(str, data_list))

    def __query_nvml(self) -> list:
        data_list = list()
        # -----------------------------------------------------------------------
        # Device and application clocks
        try:
            for clock_type in {pynvml.NVML_CLOCK_GRAPHICS, pynvml.NVML_CLOCK_MEM, pynvml.NVML_CLOCK_SM}:
                dev_clock = pynvml.nvmlDeviceGetClockInfo(handle=self.handle, type=clock_type)
                app_clock = pynvml.nvmlDeviceGetApplicationsClock(handle=self.handle, type=clock_type)
                data_list.extend([dev_clock, app_clock])
        except pynvml.NVMLError as err:
            err_str = handle_error(query="clocks", err=err)
            dnn_lh.log_info_detail(err_str)
        # -----------------------------------------------------------------------
        # Get ECC errors -- If ECC is off there will be nothing in the lists
        for error_type, counter_type, location_type in self.possible_ecc_locations:
            try:
                ecc_counts = pynvml.nvmlDeviceGetMemoryErrorCounter(handle=self.handle, errorType=error_type,
                                                                    counterType=counter_type,
                                                                    locationType=location_type)

                data_list.append(ecc_counts)
            except pynvml.NVMLError as err:
                dnn_lh.log_info_detail(handle_error(query=f"{error_type}-{counter_type}-{location_type}", err=err))

        for error_type, counter_type in self.possible_total_ecc:
            try:
                total_eec_errors = pynvml.nvmlDeviceGetTotalEccErrors(handle=self.handle, errorType=error_type,
                                                                      counterType=counter_type)
                data_list.append(total_eec_errors)
            except pynvml.NVMLError as err:
                dnn_lh.log_info_detail(handle_error(query=f"total", err=err))
        # -----------------------------------------------------------------------
        # Get Performance state P0 to P12
        try:
            data_list.append(pynvml.nvmlDeviceGetPerformanceState(self.handle))
        except pynvml.NVMLError as err:
            err_str = handle_error(query="perf_state", err=err)
            dnn_lh.log_info_detail(err_str)
        # -----------------------------------------------------------------------
        # Clocks throttle
        try:
            data_list.append(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(self.handle))
        except pynvml.NVMLError as err:
            err_str = handle_error(query="throttle_reason", err=err)
            dnn_lh.log_info_detail(err_str)
        # -----------------------------------------------------------------------
        # Get utilization on GPU
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            data_list.extend([utilization.gpu, utilization.memory])
        except pynvml.NVMLError as err:
            err_str = handle_error(query="utilization", err=err)
            dnn_lh.log_info_detail(err_str)
        # -----------------------------------------------------------------------
        # Get GPU temperature
        try:
            data_list.append(pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU))
        except pynvml.NVMLError as err:
            err_str = handle_error(query="temperature", err=err)
            dnn_lh.log_info_detail(err_str)
        # -----------------------------------------------------------------------
        # Get GPU power
        try:
            data_list.append(pynvml.nvmlDeviceGetPowerUsage(self.handle))
        except pynvml.NVMLError as err:
            err_str = handle_error(query="power_usage", err=err)
            dnn_lh.log_info_detail(err_str)

        return data_list


def __debug():
    query_gpu = NVMLWrapper(enable_query=True)
    dnn_lh.start_setup_log_file(activate_logging=True, model="DEBUG_NVML", setup_type="TEST", loghelperinterval=1)
    for i in range(10):
        print(f"Testing iteration {i}")
        pre = query_gpu.query()
        dnn_lh.start_iteration()
        time.sleep(1)
        dnn_lh.end_iteration()
        post = query_gpu.query()
        dnn_lh.log_info_detail(f"pre={pre}|post={post}")

    dnn_lh.end_log_file()


if __name__ == '__main__':
    __debug()
