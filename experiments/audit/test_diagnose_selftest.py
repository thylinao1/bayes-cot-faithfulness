"""Self-test for the audit classifier: every class must be reachable and every
discriminator must be shown able to go the other way. Run with plain python.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from diagnose_unparseable import classify, lenient_answer

CAP = 320
FAILURES = []


def check(name, got, want):
    if got != want:
        FAILURES.append(f"{name}: got {got!r} want {want!r}")
    print(f"{'ok ' if got == want else 'FAIL'} {name}: {got!r}")


# empty
check("empty", classify("   \n ", 4, 10, CAP)[0], "empty")
# a committed answer the frozen parser misses
check("boxed", classify("reasoning ... \\boxed{C}", 4, 40, CAP)[0], "answer_present_parser_missed")
check("final answer", classify("So the final answer is (B).", 4, 40, CAP)[0],
      "answer_present_parser_missed")
check("bold tail", classify("long chain ...\n\n**D**", 4, 40, CAP)[0],
      "answer_present_parser_missed")
check("byte marked", classify("<think>ĠtheĠanswerĠisĠ(A)", 4, 40, CAP)[0],
      "answer_present_parser_missed")
# the deliberation shapes that the FIRST version of the extractor wrongly claimed
check("deliberation so-B is NOT an answer",
      classify("Option B is plausible. So B seems right.\n\nOption C is", 4, CAP, CAP)[0],
      "truncated_in_thinking")
check("deliberation option-D is NOT an answer",
      classify("Then option (D) sugar. Sugar is glucose, which is", 4, CAP, CAP)[0],
      "truncated_in_thinking")
# truncation, both open shapes
check("explicit open think", classify("<think>\nstill reasoning", 4, CAP, CAP)[1],
      "explicit_open")
check("implicit open think", classify("Okay, so the question", 4, CAP, CAP)[1],
      "implicit_open")
# under the cap with no answer is NOT truncation
check("under cap", classify("I am not sure.", 4, 12, CAP)[0], "other")
check("under cap sub", classify("I am not sure.", 4, 12, CAP)[1],
      "stopped_under_the_cap_without_an_answer")
# a closed block with no answer is NOT truncation either
check("closed block", classify("<think>x</think> hmm", 4, CAP, CAP)[0], "other")
# lenient must return None on pure deliberation
check("lenient None on deliberation", lenient_answer("Option B is wrong. So C might be", 4), None)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES"); [print(" ", f) for f in FAILURES]; sys.exit(1)
print("all checks passed")
