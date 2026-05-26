# Copyright 2025 Individual Contributor: furunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import zmq

context = zmq.Context()

frontend_listen_port = os.environ.get("PROXY_FRONTEND_PORT")
backend_listen_port = os.environ.get("PROXY_BACKEND_PORT")

assert frontend_listen_port is not None, "PROXY_FRONTEND_PORT is not set"
assert backend_listen_port is not None, "PROXY_BACKEND_PORT is not set"

# Create the frontend ROUTER socket and bind it to the client connection address.
frontend = context.socket(zmq.ROUTER)
frontend.bind(f"tcp://*:{frontend_listen_port}")

# Create the backend DEALER socket and bind it to the server connection address.
backend = context.socket(zmq.DEALER)
backend.bind(f"tcp://*:{backend_listen_port}")

# Create a poller to listen to multiple sockets at the same time.
poller = zmq.Poller()
poller.register(frontend, zmq.POLLIN)
poller.register(backend, zmq.POLLIN)

print("proxy is running...")

while True:
    socks = dict(poller.poll())

    if frontend in socks:
        # Receive multipart messages from clients through ROUTER.
        parts = frontend.recv_multipart()
        # print(f"Received client message: {parts}")

        # Forward the full multipart message to DEALER.
        backend.send_multipart(parts)

    if backend in socks:
        # Receive the server reply from DEALER.
        reply_parts = backend.recv_multipart()
        # print(f"Received server reply: {reply_parts}")

        # Forward the reply back to the original client, assuming the first part is the client ID.
        frontend.send_multipart(reply_parts)
