import threading
import socket

class ThreadedUDPSocket:
    def __init__(self, server_address):
        self.server_address = server_address
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        self.sock.bind(self.server_address)
        self.sock.settimeout(1) # 1 second wait for blocking calls until timeout
        
        self.thread = threading.Thread(target=self._listen, args=(), daemon=True)
        self.thread.start()

    def _listen(self):
        while(True):
            try:
                bmsg, client_address = self.sock.recvfrom(1024)
                msg = bmsg.decode()
                if msg == 'SHUTDOWN':
                    self.sock.sendto(str.encode('ACK'), client_address)
                    print('Received a command to shutdown from the client. Stopping...')
                    raise SystemExit(0)
                print('Received message from {}:\n{}'.format(client_address, msg))
            except socket.timeout:
                pass


    def send(self, client_address, msg):
        self.sock.sendto(str.encode(msg), client_address)
        if msg == 'SHUTDOWN':
            try:
                bmsg, client_address = self.sock.recvfrom(1024)
                msg = bmsg.decode()
                if msg == 'ACK':
                    print('Client acknowledge receipt of shutdown command. Stopping...')
                    raise SystemExit(0)
                else:
                    print('Received a non-acknowledgement message from client: {}'.format(msg))
            except socket.timeout:
                print('Failed to receive an acknowledgement from client before timeout')
                raise SystemExit(1) 

    def close(self):
        self.sock.close()
        
    def __enter__(self):
        return self
            
    def __exit__(self, *exc):
        self.sock.close()
        return False