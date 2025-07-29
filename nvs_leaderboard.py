import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import modal

TRAIN_COMMAND = "python train.py -s ~/data/{capture_name} -m ~/output/{capture_name}_gaussian_splatting/ --eval --iterations 10"


data_volume = modal.Volume.from_name("data", create_if_missing=True) 
output_volume = modal.Volume.from_name("output", create_if_missing=True)
MODAL_VOLUMES = {
    "/root/data": data_volume,
    "/root/output": output_volume,
}

# app = modal.App("gaussian-splatting", image=modal.Image.from_dockerfile(Path(__file__).parent / "Dockerfile"))
app = modal.App("gaussian-splatting", image=modal.Image.from_dockerfile("Dockerfile"))

LOCAL_PORT = 9090


def wait_for_port(host, port, q):
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection(("localhost", 22), timeout=30.0):
                break
        except OSError as exc:
            time.sleep(0.01)
            if time.monotonic() - start_time >= 30.0:
                raise TimeoutError("Waited too long for port 22 to accept connections") from exc
        q.put((host, port))


@app.function(
    timeout=3600 * 24,
    gpu="T4",
    volumes=MODAL_VOLUMES
)
def run_server(q):
    with modal.forward(22, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        threading.Thread(target=wait_for_port, args=(host, port, q)).start()

        # Added these commands to get the env variables that docker loads in through ENV to show up in my ssh
        import os
        import shlex
        from pathlib import Path

        output_file = Path.home() / "env_variables.sh"

        with open(output_file, "w") as f:
            for key, value in os.environ.items():
                escaped_value = shlex.quote(value)
                f.write(f'export {key}={escaped_value}\n')
        subprocess.run("echo 'source ~/env_variables.sh' >> ~/.bashrc", shell=True)

        subprocess.run(["/usr/sbin/sshd", "-D"])  # TODO: I don't know why I need to start this here


@app.function(
    timeout=3600 * 24,
    gpu="T4",
    volumes=MODAL_VOLUMES
)
def run_shell_script(shell_file_path: str):
    """Run a shell script on the remote Modal instance."""
    # Run the shell script
    print(f"Running shell script: {shell_file_path}")
    subprocess.run("bash " + shell_file_path, 
                  shell=True, 
                  cwd=".")


@app.function(
    timeout=3600,
    gpu="T4",
    volumes=MODAL_VOLUMES,
)
def run(capture_name: str):
    data_volume.reload()
    print(f"Running triangle-splatting on {capture_name}")
    os.system(TRAIN_COMMAND.format(capture_name=capture_name))
    data_volume.commit()


@app.local_entrypoint()
def main(server: bool = False, shell_file: str | None = None):   
    if server:
        import sshtunnel

        with modal.Queue.ephemeral() as q:
            run_server.spawn(q)
            host, port = q.get()
            print(f"SSH server running at {host}:{port}")

            ssh_tunnel = sshtunnel.SSHTunnelForwarder(
                (host, port),
                ssh_username="root",
                ssh_password=" ",
                remote_bind_address=("127.0.0.1", 22),
                local_bind_address=("127.0.0.1", LOCAL_PORT),
                allow_agent=False,
            )

            try:
                ssh_tunnel.start()
                print(f"SSH tunnel forwarded to localhost:{ssh_tunnel.local_bind_port}")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down SSH tunnel...")
            finally:
                ssh_tunnel.stop()

    if shell_file:
        # Run the shell script on the remote instance
        print(f"Running shell script: {shell_file}")
        run_shell_script.remote(shell_file)