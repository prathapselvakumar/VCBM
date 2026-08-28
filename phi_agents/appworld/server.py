#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import signal
import subprocess
import time
from pathlib import Path

from phi_agents.utils.logger import get_phi_logger

DEFAULT_APPWORLD_BIN = "appworld-env/bin/appworld"

logger = get_phi_logger()


def launch_appworld_environment_server(
    port: int | None,
    docker: bool,
    appworld_root: str | None,
    stdout_to_devnull: bool,
) -> subprocess.Popen[bytes]:
    """
    Launch an AppWorld environment server with extra debugging output.
    """

    appworld_cmd = (
        DEFAULT_APPWORLD_BIN
        if Path(DEFAULT_APPWORLD_BIN).exists()
        else "appworld"
    )

    args = [
        appworld_cmd,
        "serve",
        "environment",
        "--no-show-usage",
    ]

    if port:
        args.extend(["--port", str(port)])

    if docker:
        args.append("--docker")

    if appworld_root:
        args.extend(["--root", appworld_root])

    print("=" * 80, flush=True)
    print("DEBUG: Launching AppWorld", flush=True)
    print("PWD          :", Path.cwd(), flush=True)
    print("Executable   :", appworld_cmd, flush=True)
    print("Exists?      :", Path(appworld_cmd).exists(), flush=True)
    print("Arguments    :", args, flush=True)
    print("=" * 80, flush=True)

    p = subprocess.Popen(
        args,
        stdout=None,
        stderr=None,
    )

    print(f"DEBUG: PID = {p.pid}", flush=True)

    time.sleep(2)

    print(f"DEBUG: Return code after 2 seconds = {p.poll()}", flush=True)

    if p.poll() is not None:
        print(
            "DEBUG: AppWorld process exited immediately!",
            flush=True,
        )
    else:
        print(
            "DEBUG: AppWorld process is still running.",
            flush=True,
        )

    return p


def stop_appworld_environment_server(
    popen: subprocess.Popen[bytes],
) -> None:
    logger.info("sending SIGINT command to the appworld subprocess...")
    popen.send_signal(signal.SIGINT)
    popen.wait(timeout=3)
    logger.info("appworld server terminated")


def force_stop_appworld_environment_server(
    popen: subprocess.Popen[bytes],
) -> None:
    logger.info("sending SIGKILL command to the appworld subprocess...")
    popen.send_signal(signal.SIGKILL)
    popen.wait(timeout=10)
    logger.info("appworld server forcefully terminated")
