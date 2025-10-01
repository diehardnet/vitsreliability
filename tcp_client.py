"""
The idea of the client is that it has to be small and self-contained,
 in order to allow integration without problems with existing scripts
"""
import io
import os
import socket
import time

import torch

FILENAME_OK = "FILE_OK"
FILENAME_NOK = "FILE_NOK"
SOCKET_SEND_OK = "SEND_OK"
SOCKET_ERROR_WHEN_SENDING_FILENAME = "SOCKET_ERROR_FILENAME"
SOCKET_ERROR_WHEN_SENDING_DATA = "SOCKET_ERROR_DATA"
SOCKET_ERROR_WHEN_SHUTTING_DOWN = "SOCKET_ERROR_SHUTTING_DOWN"

__SERVER_PORT = 5001  # must match server
__BUFFER_SIZE = 4096
__SERVER_ADDRESS = "127.0.0.1"


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    # Save tensor into in-memory buffer (pickle-based)
    torch.save(tensor, buffer)
    buffer.seek(0)
    return buffer.read()


def send_tensor_to_server(tensor: torch.Tensor, filename: str) -> str:
    """ Send encoded data to the server address.
    :return: SEND_OK if went ok, otherwise a string with error
    """
    # Make sure we are on the CPU
    if tensor.is_cuda:
        raise ValueError("Tensor must on the CPU")

    try:
        # create TCP socket
        sock = socket.socket()
        sock.connect((__SERVER_ADDRESS, __SERVER_PORT))

        # send filename first
        sock.send(os.path.basename(filename).encode())

        # wait for ack
        ack = sock.recv(__BUFFER_SIZE).decode()
        if ack != FILENAME_OK:
            # print("Server did not accept filename")
            sock.close()
            return FILENAME_NOK
    except socket.error:
        return SOCKET_ERROR_WHEN_SENDING_FILENAME

    try:
        #  send the tensor
        data_bytes = tensor_to_bytes(tensor)
        sock.sendall(data_bytes)
    except socket.error:
        return SOCKET_ERROR_WHEN_SENDING_DATA

    # tell server transfer is done
    try:
        sock.shutdown(socket.SHUT_WR)
        # print(f"File '{filename}' sent successfully.")
        sock.close()
    except socket.error:
        return SOCKET_ERROR_WHEN_SHUTTING_DOWN
    return SOCKET_SEND_OK


def debug():
    dummy = torch.randn(2, 5)
    bytes_test = tensor_to_bytes(dummy)
    bytes_count = len(bytes_test)
    assert bytes_count == 1577, "Not always equal, {bytes_count} != 1577".format(bytes_count=bytes_count)
    for i in range(1, 1000):
        print(dummy)
        returned = send_tensor_to_server(dummy, "dummy_tensor.pt")
        print(returned)
        time.sleep(10)


if __name__ == "__main__":
    debug()
