class AVRK4:

    def name(self) -> str:
        return "ARVK4"

    def __init__(self, connection: str = "", configuration: bytes = None):
        pass

    def connect(self, connection: str) -> bool:
        return True

    def read_until(self, p: bytes):
        return True

    def write_bytes(self, bs: bytes):
        """Send bytes to the oscilloscope."""
        return True

    def send_str(self, s: str):
        pass

    def close(self):
        pass

    def try_read_bytes(self, nb: int) -> bytes:
        return bytearray()

    def reset_configuration(self):
        """Reset oscilloscope configuration."""
        self.send_str("*RST\n")

    def read_configuration(self) -> bytes:
        """Read whole configuration from oscilloscope into a binary blob."""
        raise NotImplementedError("")
        # self.__send_str("*LRN?\n")
        # ans = self.__read_until(b"</setup>\r\n\n")
        # return ans

    # """
    # Load a configuration from former read.
    # """
    def load_configuration(self, configuration: bytes):
        # self.__send_bytes(configuration)
        raise NotImplementedError("")

    def set_ext_trig(self):
        pass

    def set_trigger_delay(self, delay: any):
        pass

    def get_trigger_delay(self) -> any:
        return 0

    def activate(self):
        pass

    def shutdown(self):
        pass

    def isActive(self):
        return True

    def set_amplitude(self, voltage: any):
        pass
