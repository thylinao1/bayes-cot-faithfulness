"""Readiness check with OWNERSHIP for a vLLM server this job started.

The old probe every serving sbatch used was ``curl -sf $BASE_URL/models``, which asks
only whether SOMETHING answers on the port. On 2026-09-07 two of our jobs landed on
xgph12 (MIG slices of one A100 node share the host loopback) with the fixed port 8000,
the second job's probe was answered by the FIRST job's server, and the run went on
against a server it did not start: cross-model that surfaced as a 404 caught by the
forced-logprob guard, same-model it surfaced as nothing at all, with two clients and up
to 64 requests in flight against a serving mode pre-registered at 32.

So a readiness probe has to prove the responder is OURS, not merely that a responder
exists. Three facts, all required:

  (a) MODEL LIST      GET <base-url>/models parses as JSON and one data[].id is exactly
                      the name we told vLLM to serve. A neighbour serving a different
                      model fails here.
  (b) SERVER ALIVE    the pid we launched is still alive (``kill -0``). If the port
                      answers while our own server is gone, the responder is someone
                      else's by definition.
  (c) PORT OWNERSHIP  the process LISTENING on the port is our server or one of its
                      descendants. This is the check that catches the dangerous case,
                      a neighbour serving the SAME model, which (a) and (b) both pass.

Exit codes, kept to three so a shell loop can branch on them:

    0   ready, and it is ours
    3   nothing is listening yet (connection refused, or the socket answers but is not
        yet a working OpenAI endpoint). The caller keeps polling.
    11  FOREIGN. Something answers and it is not ours. The caller must refuse, fast,
        rather than run against a server it does not own.

Ownership tooling differs per node, so (c) tries, in order, ``ss -ltnp``, ``lsof``, and
a /proc/net/tcp plus /proc/*/fd scan. When none of the three can name the listener the
check is SKIPPED with a reason that is logged and recorded in the run meta, never
passed over in silence.

    python bcf/serve_ready.py --base-url http://127.0.0.1:8000/v1 \
        --served-name Qwen/Qwen3-8B --server-pid 12345 --report ready.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

EXIT_READY = 0
EXIT_NOT_LISTENING = 3
EXIT_FOREIGN = 11

# How long a single /models fetch may take. Short: this runs inside a poll loop, and a
# socket that accepts but never answers is "not ready yet", not "ready".
DEFAULT_TIMEOUT_S = 10.0
# How far up the process tree the descendant walk goes before giving up. A vLLM API
# server sits one or two levels under the `vllm serve` process; 64 is far past any real
# depth and stops a cycle in a malformed ppid chain from spinning.
MAX_ANCESTOR_HOPS = 64


def log(message: str) -> None:
    print(f"[serve] {message}", flush=True)


# --- (a) the model list ---------------------------------------------------------


def fetch_model_ids(base_url: str, timeout: float) -> tuple[str, object]:
    """Return (status, payload). status is ok | not-listening | unready."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            code = response.getcode()
    except urllib.error.HTTPError as exc:
        # A server that is up but answers an error on /models is not a server we can
        # use yet. Not-listening keeps the caller polling; a foreign server stuck on an
        # error will run the caller into its own server-wait timeout, which is a
        # refusal too, just a slower one.
        return "unready", f"HTTP {exc.code} from {url}"
    except (urllib.error.URLError, OSError) as exc:
        return "not-listening", f"{type(exc).__name__}: {exc}"

    if code != 200:
        return "unready", f"HTTP {code} from {url}"
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return "unready", f"{url} did not return JSON ({exc})"
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return "unready", f"{url} returned no data[] list"
    ids = [row.get("id") for row in data if isinstance(row, dict) and row.get("id")]
    return "ok", ids


# --- (b) our server process -----------------------------------------------------


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists and belongs to someone else. For our own launched child this
        # cannot happen; treat "exists" as alive and let the ownership check decide.
        return True
    except OSError:
        return False
    return True


# --- (c) who is listening on the port -------------------------------------------


def _run(cmd: list[str], timeout: float = 15.0) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 and not proc.stdout.strip():
        return None
    return proc.stdout


def _have(tool: str) -> bool:
    from shutil import which

    return which(tool) is not None


def listeners_via_ss(port: int) -> Optional[list[int]]:
    if not _have("ss"):
        return None
    out = _run(["ss", "-ltnp"])
    if out is None:
        return None
    pids: list[int] = []
    for line in out.splitlines():
        # Local Address:Port is the 4th column of `ss -ltnp`; match the port at the end
        # of that field so 18000 is never read as 8000.
        fields = line.split()
        if len(fields) < 4:
            continue
        if not re.search(rf"[:.]{port}$", fields[3]):
            continue
        pids.extend(int(m) for m in re.findall(r"pid=(\d+)", line))
    return pids


def listeners_via_lsof(port: int) -> Optional[list[int]]:
    if not _have("lsof"):
        return None
    out = _run(["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-t", "-n", "-P"])
    if out is None:
        return None
    return [int(tok) for tok in out.split() if tok.isdigit()]


def listeners_via_proc(port: int) -> Optional[list[int]]:
    """/proc/net/tcp inode for the listening socket, then /proc/*/fd back to a pid."""
    inodes: set[str] = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10:
                continue
            local, state, inode = fields[1], fields[3], fields[9]
            if state != "0A":  # TCP_LISTEN
                continue
            try:
                local_port = int(local.rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(inode)
    if not inodes:
        # No /proc/net/tcp at all is "no tool"; an empty result with the files present
        # means nothing is listening, which the model-list step has already ruled out.
        if not Path("/proc/net/tcp").exists():
            return None
        return []
    pids: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            for fd in (proc_dir / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    pids.append(int(proc_dir.name))
                    break
        except OSError:
            # Another user's process, or one that exited under us. Both are fine to
            # skip: we are looking for OUR pid, and it is readable to us.
            continue
    return pids


def find_listeners(port: int) -> tuple[Optional[list[int]], str]:
    """Return (pids, method). pids is None when no tool on this node could look."""
    for finder, name in (
        (listeners_via_ss, "ss -ltnp"),
        (listeners_via_lsof, "lsof -iTCP -sTCP:LISTEN"),
        (listeners_via_proc, "/proc/net/tcp + /proc/*/fd"),
    ):
        pids = finder(port)
        if pids is not None:
            return pids, name
    return None, "no ss, no lsof, no /proc/net/tcp"


def ppid_of(pid: int) -> Optional[int]:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    out = _run(["ps", "-o", "ppid=", "-p", str(pid)], timeout=10.0)
    if out is None:
        return None
    out = out.strip()
    return int(out) if out.isdigit() else None


def is_ours(listener_pid: int, server_pid: int) -> bool:
    """True when listener_pid IS our server or a descendant of it."""
    pid = listener_pid
    for _ in range(MAX_ANCESTOR_HOPS):
        if pid == server_pid:
            return True
        if pid <= 1:
            return False
        parent = ppid_of(pid)
        if parent is None or parent == pid:
            return False
        pid = parent
    return False


def cgroup_of(pid: int) -> Optional[str]:
    try:
        return Path(f"/proc/{pid}/cgroup").read_text().strip()
    except OSError:
        return None


def same_slurm_job(listener_pid: int) -> bool:
    """Second chance: the listener sits in this job's own Slurm cgroup.

    Two jobs on one node are in different cgroups even though they share the host
    loopback, so this cannot pass a neighbour job's server. It exists because a server
    that re-parents (its launcher exits) would otherwise fail the descendant walk and
    cost a good run.
    """
    mine = cgroup_of(os.getpid())
    theirs = cgroup_of(listener_pid)
    if not mine or not theirs or "slurm" not in mine.lower():
        return False
    return mine == theirs


# --- the check ------------------------------------------------------------------


def check(
    base_url: str, served_name: str, server_pid: Optional[int], timeout: float
) -> tuple[int, dict]:
    report: dict = {
        "base_url": base_url,
        "served_name": served_name,
        "server_pid": server_pid,
        "port": port_of(base_url),
        "models": None,
        "owner_check": None,
        "owner_method": None,
        "listener_pids": None,
        "verdict": None,
        "reason": None,
    }

    status, payload = fetch_model_ids(base_url, timeout)
    if status != "ok":
        report["verdict"] = "not-listening"
        report["reason"] = str(payload)
        log(f"not ready yet: {payload}")
        return EXIT_NOT_LISTENING, report

    ids = list(payload)  # type: ignore[arg-type]
    report["models"] = ids
    port = report["port"]
    if served_name not in ids:
        report["verdict"] = "foreign"
        report["reason"] = f"model list {ids} does not contain {served_name!r}"
        log(
            f"REFUSING: port {port} is served by a foreign process / wrong model list "
            f"{ids}; this job serves '{served_name}'."
        )
        return EXIT_FOREIGN, report

    if server_pid is not None and not pid_alive(server_pid):
        report["verdict"] = "foreign"
        report["reason"] = f"our server pid {server_pid} is gone but the port answers"
        log(
            f"REFUSING: port {port} is served by a foreign process / wrong model list "
            f"[our server pid {server_pid} is gone, so the responder is not ours]."
        )
        return EXIT_FOREIGN, report

    if server_pid is None:
        report["owner_check"] = "skipped"
        report["reason"] = "no --server-pid given"
        report["verdict"] = "ready"
        log(f"WARN: port ownership NOT checked on port {port} (no --server-pid given)")
        log(f"ready at {base_url} serving '{served_name}'")
        return EXIT_READY, report

    pids, method = find_listeners(port)
    report["owner_method"] = method
    report["listener_pids"] = pids
    if pids is None:
        report["owner_check"] = f"skipped: {method}"
        report["verdict"] = "ready"
        log(
            f"WARN: port ownership NOT checked on port {port}: {method}. The model list "
            f"and our server pid both check out; the listener could not be named."
        )
        log(f"ready at {base_url} serving '{served_name}'")
        return EXIT_READY, report
    if not pids:
        # The port answered a moment ago and nothing holds it now. Not a foreign
        # server, just a race; keep polling.
        report["owner_check"] = "inconclusive"
        report["verdict"] = "not-listening"
        report["reason"] = f"{method} found no listener on {port}"
        log(f"not ready yet: {method} names no listener on port {port}")
        return EXIT_NOT_LISTENING, report

    ours = [p for p in pids if is_ours(p, server_pid)]
    if ours:
        report["owner_check"] = "passed"
        report["verdict"] = "ready"
        log(
            f"port {port} owned by pid {ours[0]} (our server {server_pid} or a "
            f"descendant), via {method}"
        )
        log(f"ready at {base_url} serving '{served_name}'")
        return EXIT_READY, report

    same_job = [p for p in pids if same_slurm_job(p)]
    if same_job:
        report["owner_check"] = "passed: same slurm cgroup"
        report["verdict"] = "ready"
        log(
            f"port {port} held by pid {same_job[0]}, not a descendant of {server_pid} "
            f"but inside THIS job's slurm cgroup, via {method}"
        )
        log(f"ready at {base_url} serving '{served_name}'")
        return EXIT_READY, report

    report["owner_check"] = "failed"
    report["verdict"] = "foreign"
    report["reason"] = (
        f"listener pid(s) {pids} are not our server {server_pid} nor descendants of it"
    )
    log(
        f"REFUSING: port {port} is served by a foreign process / wrong model list "
        f"[listener pid(s) {pids}, our server pid {server_pid}, via {method}]. The "
        f"model name matches, so this is a NEIGHBOUR SERVING THE SAME MODEL: running "
        f"here would put two clients on one server."
    )
    return EXIT_FOREIGN, report


def port_of(base_url: str) -> int:
    match = re.search(r":(\d+)", base_url.split("//", 1)[-1])
    return int(match.group(1)) if match else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--served-name", required=True, help="--served-model-name value")
    parser.add_argument("--server-pid", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--report", default=None, help="write the check as JSON here")
    args = parser.parse_args(argv)

    code, report = check(
        args.base_url, args.served_name, args.server_pid, args.timeout
    )
    report["exit_code"] = code
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
