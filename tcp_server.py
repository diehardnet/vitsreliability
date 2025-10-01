# server
import io
import logging
import os
import socket
import time

import torch

from tcp_client import FILENAME_OK, TENSOR_LENGTH_WIDTH

SERVER_HOST = "0.0.0.0"  # listen on all interfaces
SERVER_PORT = 5001
BUFFER_SIZE = 4096
# Not to be changed
SOCKET_BACKLOG = 5
DOWNLOADED_FILES_DIR = "files_downloaded"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def recv_exact(sock, length):
    buf = bytearray()
    while len(buf) < length:
        chunk = sock.recv(length - len(buf))
        if not chunk:
            # connection closed or error
            raise ConnectionError("Socket closed before receiving all data")
        buf.extend(chunk)
    return bytes(buf)


def recv_tensor(sock: socket.socket) -> torch.Tensor:
    # receive the size
    len_bytes = recv_exact(sock, TENSOR_LENGTH_WIDTH)
    tensor_length = int.from_bytes(len_bytes, 'big')

    # Then read exactly `length` bytes
    tensor_bytes = recv_exact(sock, tensor_length)
    buffer = io.BytesIO(tensor_bytes)
    buffer.seek(0)
    tensor = torch.load(buffer)
    return tensor


def main():
    logger = get_logger("main")

    # Create the files directory
    if not os.path.isdir(DOWNLOADED_FILES_DIR):
        os.makedirs(DOWNLOADED_FILES_DIR)

    tcp_socket = socket.socket()
    tcp_socket.bind((SERVER_HOST, SERVER_PORT))
    tcp_socket.listen(SOCKET_BACKLOG)
    logger.info(f"[*] Listening on {SERVER_HOST}:{SERVER_PORT}")

    while True:
        client_socket, addr = tcp_socket.accept()
        logger.info(f"[+] {addr} connected.")

        # receive filename first
        filename = client_socket.recv(BUFFER_SIZE).decode()
        logger.info(f"Receiving file: {filename} from {addr}")
        client_socket.send(FILENAME_OK.encode())  # send ack

        # open file to write binary data
        current_clock = int(time.time())
        filename = f"{current_clock}_{filename}"

        addr_path = os.path.join(DOWNLOADED_FILES_DIR, addr[0])
        if not os.path.isdir(addr_path):
            os.makedirs(addr_path)
        client_file_path = os.path.join(addr_path, filename)

        tensor_received = recv_tensor(client_socket)
        print(tensor_received)
        torch.save(tensor_received, client_file_path)

        logger.info(f"File '{client_file_path}' received and saved.")
        client_socket.close()


if __name__ == "__main__":
    main()
