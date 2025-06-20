import zmq

#
class CamClient:
    def __init__(self, host="172.16.8.6", port=6667, timeout=5000):
        self.address = f"tcp://{host}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.connect(self.address)

    def send(self, command, cmd_sz=10):
        padded = command + (cmd_sz - 1 - len(command)) * " "
        self.socket.send_string(padded)
        reply = self.socket.recv().decode("ascii")
        return reply

    def close(self):
        self.socket.close()
        self.context.term()

# multi device server (MDS) client 
class MDSClient:
    def __init__(self, host="172.16.8.6", port=5555, timeout=5000):
        self.address = f"tcp://{host}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.connect(self.address)

    def send(self, message):
        self.socket.send_string(message)
        reply = self.socket.recv_string()
        return reply

    def close(self):
        self.socket.close()
        self.context.term()
