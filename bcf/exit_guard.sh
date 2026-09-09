#!/bin/bash
# The exit-code guard for the jury judge jobs.
#
# THE DEFECT THIS CLOSES. Job 826023 was cancelled by explicit id at 04:22:10 on
# 2026-09-07, partway through Q1 variant a, having planned 15,939 votes and written 342 of
# them (2.15 percent). The job's EXIT trap still wrote "0" to
# qwen3-32b-h200/arc_challenge/stated-hint/exit_code.txt, because the trap recorded
# whatever $? happened to be when the shell unwound, and on the cancel path that was 0. A
# results directory holding two percent of a run therefore carried a success code, and
# nothing downstream could tell it apart from a complete run.
#
# THREE RULES, and the first one does most of the work.
#
#   1. exit_code.txt is written NON-ZERO the moment the run starts: 255, meaning "started,
#      no completion recorded". Only a run that reaches its own completion marker
#      overwrites it with a real code. A SIGKILL, a node failure, an OOM killer or a
#      power cut cannot leave a zero behind, because a zero was never written.
#   2. TERM, INT, HUP and USR1 are trapped and become 128 + the signal number, so a
#      scancel reads 143 rather than whatever the last command in the pipeline returned.
#      Slurm sends SIGTERM on scancel and SIGUSR1 when a job asks for --signal.
#   3. An exit of 0 that never passed the completion marker is rewritten to 250. That is
#      the exact 826023 shape: the shell unwound with $? = 0 from a killed pipeline, and
#      no signal trap fired because none was installed.
#
# Codes: 255 started-not-finished, 250 exited-zero-without-completing, 128+n killed by
# signal n. Everything else is the job's own status, and 1 still means a failed threshold,
# which is a result rather than a job failure.
#
# Sourced by bcf/judge_serve.sbatch. Exercised end to end by bcf/test_exit_guard.sh, which
# runs on the Mac in bash 3.2 and on the cluster in bash 5, needs no GPU and no network.

BCF_EXIT_STARTED=255
BCF_EXIT_INCOMPLETE=250

BCF_EXIT_FILE=""
BCF_EXIT_STAGE_FILES=""
BCF_EXIT_COMPLETE=0
BCF_EXIT_SIGNAL=0

# Arm the guard on one exit-code file. Writes the started marker immediately.
bcf_exit_guard_init() {
  BCF_EXIT_FILE="${1:?bcf_exit_guard_init needs the path of an exit-code file}"
  BCF_EXIT_STAGE_FILES=""
  BCF_EXIT_COMPLETE=0
  BCF_EXIT_SIGNAL=0
  mkdir -p "$(dirname "$BCF_EXIT_FILE")"
  printf '%s\n' "$BCF_EXIT_STARTED" > "$BCF_EXIT_FILE"
  trap 'bcf_exit_guard_signal 15' TERM
  trap 'bcf_exit_guard_signal 2' INT
  trap 'bcf_exit_guard_signal 1' HUP
  trap 'bcf_exit_guard_signal 10' USR1
}

# The signal traps. Recording the signal before exiting is what stops a killed pipeline's
# incidental 0 from being believed.
bcf_exit_guard_signal() {
  # Disarm FIRST. Calling exit here runs the EXIT trap, which is what actually records the
  # code, and until 2026-09-09 the signal traps stayed installed for that whole stretch. A
  # second signal arriving in it killed the shell before anything was recorded, so a
  # cancelled run kept its 255 "started, never finished" marker instead of 143. Slurm
  # signals a whole job step, not one pid, so a second signal is the normal case on the
  # cluster rather than a rarity; bcf/test_exit_guard.sh reproduces it by sending TERM
  # twice, and it failed about one run in five.
  trap '' TERM INT HUP USR1
  BCF_EXIT_SIGNAL="$1"
  exit $(( 128 + $1 ))
}

# Mark a per-stage exit-code file started. Each Q1 variant gets one, so a run cancelled in
# variant c leaves variant c non-zero and does not touch a and b.
bcf_exit_guard_open_stage() {
  printf '%s\n' "$BCF_EXIT_STARTED" > "$1"
  BCF_EXIT_STAGE_FILES="${BCF_EXIT_STAGE_FILES} $1"
}

# The completion marker. Called on the one path that means the work finished.
bcf_exit_guard_complete() { BCF_EXIT_COMPLETE=1; }

# Turn the raw $? into the code that gets recorded.
bcf_exit_guard_code() {
  code="$1"
  if [ "${BCF_EXIT_SIGNAL:-0}" -ne 0 ] && [ "$code" -eq 0 ]; then
    code=$(( 128 + BCF_EXIT_SIGNAL ))
  fi
  if [ "${BCF_EXIT_COMPLETE:-0}" -ne 1 ] && [ "$code" -eq 0 ]; then
    code="$BCF_EXIT_INCOMPLETE"
  fi
  printf '%s' "$code"
}

# Write the final code to the run's file, and to any stage file still marked started.
bcf_exit_guard_write() {
  final="$1"
  # ORDER IS LOAD-BEARING, and further signals are ignored while this runs.
  #
  # The run's own file is the authoritative record, so it is written LAST: if it carries a
  # final code, every stage it summarises already carries one. Written the other way round
  # there is a window between the two writes, and the window was wide because the loop
  # forked a `cat` for every stage file. A second signal landing in it left
  # exit_code.txt=143 beside stage_exit_code.txt=255, so one cancelled run read as
  # "cancelled" in one file and "started, never finished" in the other. That reproduced in
  # about one run in five of bcf/test_exit_guard.sh and was live until 2026-09-09.
  #
  # The harness sends TERM twice on purpose (once to the job, once to the pid in the ready
  # file) to mirror how Slurm signals a whole job step, which is exactly how the second
  # signal arrives mid-finalisation. Ignoring TERM/INT/HUP for the few writes it takes to
  # finish is safe: this runs on the EXIT path, the process is already ending, and the
  # writes are a handful of bytes. `read` replaces the `cat` subshell so the loop does not
  # fork per file.
  trap '' TERM INT HUP 2>/dev/null
  for stage_file in $BCF_EXIT_STAGE_FILES; do
    [ -f "$stage_file" ] || continue
    stage_cur=""
    read -r stage_cur < "$stage_file" 2>/dev/null
    if [ "$stage_cur" = "$BCF_EXIT_STARTED" ]; then
      printf '%s\n' "$final" > "$stage_file"
    fi
  done
  [ -n "$BCF_EXIT_FILE" ] && printf '%s\n' "$final" > "$BCF_EXIT_FILE"
  # The trailing test above must not become this function's status: it is called from an
  # EXIT trap, and returning non-zero when BCF_EXIT_FILE is unset would change what the
  # caller records. The old shape ended in the for loop, which always returned 0.
  return 0
}
