"""The readiness probe must prove the responder is OURS.

Revert proof, and it runs rather than being asserted in prose. The old probe in every
serving sbatch was ``curl -sf $BASE_URL/models``, i.e. "did anything answer with a 2xx".
``test_revert_proof_old_probe_would_have_passed`` performs exactly that old check
against the wrong-model server of case (ii) and asserts it PASSES: the old logic cannot
fail this case, which is why job 827053 ran against job 827052's server on xgph12 on
2026-09-07. The three cases below then pin the new behaviour:

  (i)   a server answering /v1/models with OUR served name, listening in a process we
        own            -> exit 0
  (ii)  a server answering with a DIFFERENT model name
                       -> exit 11 (foreign), on the first poll
  (iii) a port with nothing listening
                       -> exit 3 (not yet, keep polling)

Case (iv) is the dangerous one the 404 guard cannot catch: a NEIGHBOUR SERVING THE SAME
MODEL. The model list matches, our own pid is alive, and only the port-ownership check
separates the two. It needs a real second process, so it is skipped on a machine where
no listener-naming tool exists (the helper's own graceful fallback path).

No cluster and no GPU: every server here is a stdlib http.server on loopback.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "bcf" / "serve_ready.py"

OUR_MODEL = "Qwen/Qwen3-8B"
NEIGHBOUR_MODEL = "google/gemma-3-27b-it"

EXIT_READY = 0
EXIT_NOT_LISTENING = 3
EXIT_FOREIGN = 11


def _handler_for(model_name: str):
    class ModelsHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server's own naming)
            if not self.path.endswith("/models"):
                self.send_error(404)
                return
            body = json.dumps(
                {"object": "list", "data": [{"id": model_name, "object": "model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # keep pytest output clean
            return

    return ModelsHandler


class LocalServer:
    """A /v1/models endpoint on loopback, listening in THIS process."""

    def __init__(self, model_name: str):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(model_name))
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "LocalServer":
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"


def run_helper(base_url: str, served_name: str, server_pid: int | None, report=None):
    cmd = [
        sys.executable,
        str(HELPER),
        "--base-url",
        base_url,
        "--served-name",
        served_name,
        "--timeout",
        "5",
    ]
    if server_pid is not None:
        cmd += ["--server-pid", str(server_pid)]
    if report is not None:
        cmd += ["--report", str(report)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def listener_tool_available() -> bool:
    from shutil import which

    return (
        which("ss") is not None
        or which("lsof") is not None
        or Path("/proc/net/tcp").exists()
    )


# --- (i) our own server ---------------------------------------------------------


def test_ready_when_the_model_matches_and_we_own_the_port(tmp_path):
    # Arrange
    report = tmp_path / "ready.json"
    with LocalServer(OUR_MODEL) as server:
        # Act
        result = run_helper(server.base_url, OUR_MODEL, os.getpid(), report=report)

    # Assert
    assert result.returncode == EXIT_READY, result.stdout + result.stderr
    written = json.loads(report.read_text())
    assert written["verdict"] == "ready"
    assert written["models"] == [OUR_MODEL]
    # The ownership check either passed or was skipped WITH a reason. Never silent.
    assert written["owner_check"] is not None
    if not listener_tool_available():
        assert written["owner_check"].startswith("skipped")


# --- (ii) a neighbour serving a DIFFERENT model ---------------------------------


def test_foreign_when_the_port_answers_with_another_model(tmp_path):
    # Arrange
    report = tmp_path / "foreign.json"
    with LocalServer(NEIGHBOUR_MODEL) as server:
        # Act
        result = run_helper(server.base_url, OUR_MODEL, os.getpid(), report=report)

    # Assert
    assert result.returncode == EXIT_FOREIGN, result.stdout + result.stderr
    assert "REFUSING" in result.stdout
    written = json.loads(report.read_text())
    assert written["verdict"] == "foreign"
    assert written["models"] == [NEIGHBOUR_MODEL]


def test_revert_proof_old_probe_would_have_passed():
    """The old `curl -sf $BASE_URL/models` cannot fail case (ii). Shown, not claimed."""
    with LocalServer(NEIGHBOUR_MODEL) as server:
        with urllib.request.urlopen(server.base_url + "/models", timeout=5) as response:
            old_probe_passes = response.getcode() == 200
    assert old_probe_passes, (
        "the old probe asked only whether something answered on the port; if this "
        "assertion ever fails the revert proof needs rewriting, not the fix"
    )


# --- (iii) nothing listening ----------------------------------------------------


def test_not_listening_when_the_port_is_dead(tmp_path):
    # Arrange
    port = free_port()
    report = tmp_path / "notyet.json"

    # Act
    result = run_helper(f"http://127.0.0.1:{port}/v1", OUR_MODEL, os.getpid(), report)

    # Assert
    assert result.returncode == EXIT_NOT_LISTENING, result.stdout + result.stderr
    written = json.loads(report.read_text())
    assert written["verdict"] == "not-listening"


# --- (iv) a neighbour serving the SAME model, the case the 404 guard misses ------


@pytest.mark.skipif(
    not listener_tool_available(),
    reason="no ss, no lsof and no /proc/net/tcp: the helper skips ownership with a WARN",
)
def test_foreign_when_a_neighbour_serves_the_same_model(tmp_path):
    # Arrange: the server runs in a process that is NOT our server and NOT a descendant
    # of it, exactly as a second job's vLLM on a shared node would be.
    port = free_port()
    code = (
        "import json,sys\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        b=json.dumps({'object':'list','data':[{'id':sys.argv[2],"
        "'object':'model'}]}).encode()\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Type','application/json')\n"
        "        self.send_header('Content-Length',str(len(b)))\n"
        "        self.end_headers(); self.wfile.write(b)\n"
        "    def log_message(self,*a): pass\n"
        "ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()\n"
    )
    neighbour = subprocess.Popen([sys.executable, "-c", code, str(port), OUR_MODEL])
    # A live process that did not start the neighbour: our own server, as far as the
    # helper is concerned.
    our_server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    report = tmp_path / "same_model.json"
    base_url = f"http://127.0.0.1:{port}/v1"
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                urllib.request.urlopen(base_url + "/models", timeout=2).read()
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("the neighbour server never came up")

        # Act
        result = run_helper(base_url, OUR_MODEL, our_server.pid, report=report)
    finally:
        neighbour.terminate()
        our_server.terminate()
        neighbour.wait(timeout=10)
        our_server.wait(timeout=10)

    # Assert: same model name, our pid alive, and it is STILL refused.
    assert result.returncode == EXIT_FOREIGN, result.stdout + result.stderr
    written = json.loads(report.read_text())
    assert written["models"] == [OUR_MODEL]
    assert written["owner_check"] == "failed"
    assert "REFUSING" in result.stdout


# --- the shell glue: env.sh's wait loop maps the helper's codes ------------------
#
# The helper is only half the fix. bcf_wait_server_ready in bcf/env.sh is what every
# serving sbatch actually calls, and it has to turn "foreign" into the job's own
# refusal code 13, "ready" into 0, and a deadline that has passed into 6. A shell
# mistake there (a lost ${PIPESTATUS[0]}, a swallowed return) would leave the helper
# correct and the fix inert, which is the failure mode this project keeps meeting.


def _env_helpers() -> str:
    text = (REPO / "bcf" / "env.sh").read_text()
    marker = "# --- serving ports and readiness"
    assert marker in text, "env.sh lost the serving-port helpers"
    return text[text.index(marker) :]


def run_wait_loop(base_url: str, served_name: str, pid: int, deadline_offset: int,
                  report: Path) -> tuple[int, str]:
    script = (
        "set -uo pipefail\n"
        f"export BCF_REPO={REPO}\n"
        f"{_env_helpers()}\n"
        f'bcf_wait_server_ready "{base_url}" "{served_name}" {pid} '
        f'$(( $(date +%s) + {deadline_offset} )) /dev/null "{report}"\n'
        'echo "WAIT_RC=$?"\n'
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)
    rc_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("WAIT_RC=")]
    assert rc_line, proc.stdout + proc.stderr
    return int(rc_line[-1].split("=")[1]), proc.stdout


def test_wait_loop_returns_zero_on_our_own_server(tmp_path):
    with LocalServer(OUR_MODEL) as server:
        rc, out = run_wait_loop(server.base_url, OUR_MODEL, os.getpid(), 60, tmp_path / "r.json")
    assert rc == 0, out


def test_wait_loop_refuses_a_foreign_server_with_13(tmp_path):
    with LocalServer(NEIGHBOUR_MODEL) as server:
        rc, out = run_wait_loop(server.base_url, OUR_MODEL, os.getpid(), 60, tmp_path / "r.json")
    assert rc == 13, out
    assert "REFUSING" in out


def test_wait_loop_times_out_with_6_when_nothing_ever_answers(tmp_path):
    port = free_port()
    rc, out = run_wait_loop(
        f"http://127.0.0.1:{port}/v1", OUR_MODEL, os.getpid(), -1, tmp_path / "r.json"
    )
    assert rc == 6, out


def test_pick_port_returns_a_free_port_and_honours_a_pin():
    script = (
        "set -uo pipefail\n"
        f"export BCF_REPO={REPO}\n"
        f"{_env_helpers()}\n"
        'echo "PICKED=$(bcf_pick_port)"\n'
        'echo "PINNED=$(bcf_pick_port 8123)"\n'
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    out = proc.stdout
    picked = int([ln for ln in out.splitlines() if ln.startswith("PICKED=")][0].split("=")[1])
    pinned = [ln for ln in out.splitlines() if ln.startswith("PINNED=")][0].split("=")[1]
    assert 1024 < picked < 65536
    assert pinned == "8123", "an operator pin must still be honoured"
