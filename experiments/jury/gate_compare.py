"""Put two or more gate runs side by side: every threshold, every class, every phrasing.

A gate report says whether one configuration passed. It does not say what changed when the
instrument was revised, and that is the question a prompt revision has to answer. This
reads the reports and their vote files and writes one table per view:

  1. the ten thresholds, with numerator, denominator and verdict for each configuration
  2. the seven gate classes, with the Q1 yes/no split and its denominator
  3. the three frozen phrasings inside each planted class, because a class that moves as a
     whole and a class where one phrasing moves are different findings

The phrasing index is recovered the way `synthetic_gate.build_items` assigns it: within a
class, the items are in bank order and the phrasing is that position modulo three. The
corpus file is the input, so this never re-derives the corpus from the builder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gate_thresholds import HIGHER_IS_BETTER, THRESHOLDS
from .synthetic_gate import CLASSES, TRUTH

PHRASED_CLASSES = ("planted_mention", "paraphrased_disclosure", "quoted_denied", "restated_cue_only")


def class_of(item_id: str) -> str:
    """The gate class in an item id of the form gate-<class>-<source>-<index>."""
    body = item_id[len("gate-"):] if item_id.startswith("gate-") else item_id
    for cls in CLASSES:
        if body.startswith(cls + "-"):
            return cls
    raise ValueError(f"no gate class in {item_id!r}")


def phrasing_index(items: list[dict]) -> dict[str, int]:
    """item_id -> which of the three frozen phrasings that item carries."""
    seen: dict[str, int] = {}
    out: dict[str, int] = {}
    for item in items:
        cls = item["meta"]["gate_class"] if "gate_class" in item.get("meta", {}) else class_of(item["item_id"])
        k = seen.get(cls, 0)
        seen[cls] = k + 1
        out[item["item_id"]] = k % 3
    return out


def first_run_q1(votes: list[dict]) -> list[dict]:
    return [v for v in votes
            if v.get("question") == "Q1" and v.get("run_idx") == 0 and not v.get("position_swap")]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class Config:
    """One scored configuration: a report, its votes, and the label it goes under."""

    def __init__(self, name: str, report_path: Path, votes_path: Path, items: list[dict]):
        self.name = name
        self.report = json.loads(report_path.read_text(encoding="utf-8"))
        self.votes = read_jsonl(votes_path)
        self.items = items
        self.phrasing = phrasing_index(items)

    @property
    def judge(self) -> dict:
        return self.report["per_judge"][0]

    def q1_prompt_file(self) -> str:
        """The Q1 file this configuration scored.

        Taken from the report when it carries `prompt_files`, and otherwise from the vote
        records themselves, which have always carried `prompt_file` per row. The report of
        job 825542 predates the field, and reading it off the votes is the stronger source
        anyway: it is what the runner actually rendered, not what a summariser said later.
        """
        named = self.report.get("prompt_files", {}).get("Q1")
        if named:
            return str(named)
        files = {v.get("prompt_file") for v in self.votes if v.get("question") == "Q1"}
        files.discard(None)
        if len(files) == 1:
            return str(files.pop())
        return "|".join(sorted(str(f) for f in files)) or "unknown"

    def serving_line(self) -> str:
        named = self.report.get("serving_line")
        if named:
            return str(named)
        lines = {v.get("serving_line") for v in self.votes}
        lines.discard(None)
        if len(lines) == 1:
            return str(lines.pop())
        return "pinned (section 6.1)"

    def q1_yes_by_class(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for v in first_run_q1(self.votes):
            cls = class_of(v["item_id"])
            bucket = out.setdefault(cls, [0, 0])
            if v["vote"] in ("yes", "no"):
                bucket[1] += 1
                if v["vote"] == "yes":
                    bucket[0] += 1
        return {k: (a, b) for k, (a, b) in out.items()}

    def q1_yes_by_phrasing(self) -> dict[tuple[str, int], tuple[int, int]]:
        out: dict[tuple[str, int], list[int]] = {}
        for v in first_run_q1(self.votes):
            cls = class_of(v["item_id"])
            ph = self.phrasing.get(v["item_id"])
            if ph is None:
                continue
            bucket = out.setdefault((cls, ph), [0, 0])
            if v["vote"] in ("yes", "no"):
                bucket[1] += 1
                if v["vote"] == "yes":
                    bucket[0] += 1
        return {k: (a, b) for k, (a, b) in out.items()}


def fmt(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.4f}" if den else "no data"


def threshold_table(configs: list[Config]) -> list[str]:
    head = "| Threshold | Bar | " + " | ".join(f"{c.name}" for c in configs) + " |"
    rule = "|---|---|" + "---|" * len(configs)
    lines = [head, rule]
    for name, bar in THRESHOLDS.items():
        direction = "at least" if name in HIGHER_IS_BETTER else "at most"
        cells = []
        for c in configs:
            v = c.judge["gate_verdicts"].get(name, {})
            if not v or v.get("rate") is None:
                cells.append("NO DATA")
                continue
            cells.append(f"{v['numerator']}/{v['denominator']} = {v['rate']:.4f} {v['verdict']}")
        lines.append(f"| {name} | {direction} {bar} | " + " | ".join(cells) + " |")
    return lines


def class_table(configs: list[Config]) -> list[str]:
    head = "| Gate class | Q1 truth | " + " | ".join(f"{c.name} Q1 yes" for c in configs) + " |"
    lines = [head, "|---|---|" + "---|" * len(configs)]
    per = [c.q1_yes_by_class() for c in configs]
    for cls in CLASSES:
        truth = TRUTH[cls].get("q1")
        label = {True: "yes", False: "no"}.get(truth, "not scored")
        cells = [fmt(*p.get(cls, (0, 0))) for p in per]
        lines.append(f"| {cls} | {label} | " + " | ".join(cells) + " |")
    return lines


def phrasing_table(configs: list[Config]) -> list[str]:
    head = "| Gate class | Phrasing | " + " | ".join(f"{c.name} Q1 yes" for c in configs) + " |"
    lines = [head, "|---|---|" + "---|" * len(configs)]
    per = [c.q1_yes_by_phrasing() for c in configs]
    for cls in PHRASED_CLASSES:
        for ph in range(3):
            cells = [fmt(*p.get((cls, ph), (0, 0))) for p in per]
            lines.append(f"| {cls} | {ph} | " + " | ".join(cells) + " |")
    return lines


def header_table(configs: list[Config]) -> list[str]:
    rows = [
        ("Q1 prompt file", lambda c: c.q1_prompt_file()),
        ("Q1 prompt sha256", lambda c: c.report["prompt_sha256"]["Q1"]),
        ("gate prompt sha256", lambda c: c.report["prompt_sha256"]["gate"]),
        ("Q2 prompt sha256", lambda c: c.report["prompt_sha256"]["Q2"]),
        ("thresholds sha256", lambda c: c.report["thresholds_sha256"]),
        ("judge", lambda c: c.judge["judge_model"]),
        ("judge revision", lambda c: c.judge["judge_revision"]),
        ("serving line", lambda c: c.serving_line()),
        ("corpus items", lambda c: str(c.report["corpus"]["items"])),
        ("votes", lambda c: str(c.judge["votes"])),
        ("votes per second per server", lambda c: str(c.report.get("votes_per_second_per_server"))),
        ("verdict", lambda c: c.judge["verdict"]),
        ("failed thresholds", lambda c: ", ".join(c.judge["failed_metrics"]) or "none"),
    ]
    lines = ["| Field | " + " | ".join(c.name for c in configs) + " |",
             "|---|" + "---|" * len(configs)]
    for label, get in rows:
        lines.append(f"| {label} | " + " | ".join(str(get(c)) for c in configs) + " |")
    return lines


def build(configs: list[Config], title: str, preamble: str) -> str:
    out = [f"# {title}", "", preamble, "", "## Configurations", ""]
    out += header_table(configs)
    out += ["", "## The ten thresholds", ""]
    out += threshold_table(configs)
    out += ["", "## Q1 yes rate per gate class, run 0, unswapped", ""]
    out += class_table(configs)
    out += ["", "## Q1 yes rate per frozen phrasing", "",
            "Phrasings are the three templates frozen in `synthetic_gate.py`, rotated by "
            "position within each class.", ""]
    out += phrasing_table(configs)
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True, help="the frozen corpus JSONL")
    ap.add_argument("--config", action="append", required=True,
                    help="name=<results dir containing gate_report.json and votes.jsonl>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Gate comparison")
    ap.add_argument("--preamble", default="")
    args = ap.parse_args(argv)
    items = read_jsonl(Path(args.items))
    configs = []
    for spec in args.config:
        name, _, d = spec.partition("=")
        p = Path(d)
        configs.append(Config(name, p / "gate_report.json", p / "votes.jsonl", items))
    Path(args.out).write_text(build(configs, args.title, args.preamble), encoding="utf-8")
    print(f"wrote {args.out} from {len(configs)} configuration(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
