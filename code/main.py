#!/usr/bin/env python3
"""Sentinel - multimodal WhatsApp notification router.

Usage:
    python code/main.py run                 # route messages.csv -> output.csv
    python code/main.py eval                # score against labelled samples
    python code/main.py explain msg_023     # full decision trace for one message
    python code/main.py media               # rebuild the OCR/ASR cache
    python code/main.py ablate              # component ablation study
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router.config import DEFAULT_SETTINGS, Settings  # noqa: E402
from router.dataio import load_dataset, validate_output, write_output  # noqa: E402
from router.evaluate import calibration_report, evaluate  # noqa: E402
from router.metrics import StageTimer  # noqa: E402
from router.metrics import collect as collect_metrics  # noqa: E402
from router.multimodal import build_media_index  # noqa: E402
from router.pipeline import NotificationRouter  # noqa: E402

log = logging.getLogger("sentinel")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("sentence_transformers", "transformers", "urllib3", "httpx", "faster_whisper", "PIL"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def build_settings(args: argparse.Namespace) -> Settings:
    settings = DEFAULT_SETTINGS
    if getattr(args, "dataset", None):
        settings = replace(settings, dataset_dir=Path(args.dataset).resolve())
    if getattr(args, "use_llm", False):
        settings = replace(settings, use_llm=True)
    if getattr(args, "no_llm", False):
        settings = replace(settings, use_llm=False)
    if getattr(args, "no_embeddings", False):
        settings = replace(settings, use_embeddings=False)
    return settings


def _router(settings: Settings):
    dataset = load_dataset(settings)
    media = build_media_index(settings, dataset)
    return dataset, NotificationRouter(settings, dataset, media)


def cmd_run(args: argparse.Namespace) -> int:
    settings = build_settings(args)
    timer = StageTimer()

    timer.start("load_dataset")
    dataset = load_dataset(settings)
    timer.stop("load_dataset")

    timer.start("media_understanding")
    media = build_media_index(settings, dataset)
    timer.stop("media_understanding")

    timer.start("build_router")
    router = NotificationRouter(settings, dataset, media)
    timer.stop("build_router")

    timer.start("routing")
    decisions, stats, critic = router.run()
    timer.stop("routing")

    print(stats.render())
    print(critic.render())

    out_path = Path(args.output).resolve() if args.output else settings.dataset_dir / "output.csv"
    written = write_output(out_path, decisions)
    log.info("wrote %d rows to %s", written, out_path)

    report = validate_output(out_path, [m.message_id for m in dataset.messages])
    print(report.render())

    metrics = collect_metrics(decisions, stats, critic, media, dataset, timer)
    metrics_path = metrics.write(settings.artifact_dir / "run_metrics.json")
    print()
    print(metrics.render())
    log.info("wrote operational metrics to %s", metrics_path)

    if args.trace:
        trace_path = settings.artifact_dir / "decision_traces.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as fh:
            for d in decisions:
                fh.write(json.dumps({
                    "message_id": d.message_id, "action": d.action, "message_type": d.message_type,
                    "confidence": d.confidence, "rationale": d.rationale_key,
                    "priority": d.priority_score, "signals": d.signals,
                    "drivers": d.drivers, "evidence": d.evidence_message_ids, "trace": d.trace,
                }, ensure_ascii=False) + "\n")
        log.info("wrote decision traces to %s", trace_path)

    return 0 if report.ok else 1


def cmd_eval(args: argparse.Namespace) -> int:
    settings = build_settings(args)
    dataset, router = _router(settings)
    if not dataset.samples:
        log.error("no labelled samples found")
        return 1

    decisions, stats, critic = router.run(list(dataset.samples))
    print(stats.render())
    print(critic.render())
    print()
    result = evaluate(decisions, dataset.samples, verbose=True)
    print(result.render(verbose=True))
    print()
    print(calibration_report(decisions, dataset.samples))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    settings = build_settings(args)
    dataset, router = _router(settings)

    pool = list(dataset.messages) + list(dataset.samples)
    target = next((m for m in pool if m.message_id == args.message_id), None)
    if target is None:
        log.error("message not found: %s", args.message_id)
        return 1

    from router.pipeline import RunStats

    routed = router.route_one(target, RunStats())
    d = routed.decision
    print(f"=== {d.message_id} ===")
    print(f"user={target.user_id} type={target.conversation_type} "
          f"group={target.group_id or '-'} business={target.business_id or '-'} "
          f"sender={target.sender_user_id or '-'} media={target.media_type or '-'}")
    print(f"\ncontent ({routed.content.modality}, {routed.content.word_count} words):")
    print("  " + (routed.content.combined[:600] or "(empty)").replace("\n", "\n  "))
    print(f"\nDECISION: {d.action} / {d.message_type}  confidence={d.confidence}")
    print(f"reason:   {d.reason}")
    print(f"rationale key: {d.rationale_key}")
    print(f"evidence: {d.evidence_field()}")
    print("\nsignals:")
    for k, v in d.signals.items():
        print(f"  {k:20s} {v}")
    print(f"\ndrivers: {', '.join(d.drivers) or 'none'}")
    print(f"threats: {', '.join(routed.safety.threats) or 'none'}")
    print(f"trace:   {json.dumps(d.trace, ensure_ascii=False)[:900]}")

    if isinstance(target, type(dataset.samples[0])) if dataset.samples else False:
        print(f"\nGROUND TRUTH: {target.action} / {target.message_type} conf={target.confidence}")
        print(f"  reason: {target.reason}")
        print(f"  evidence: {target.evidence_message_ids}")
    return 0


def cmd_media(args: argparse.Namespace) -> int:
    settings = build_settings(args)
    dataset = load_dataset(settings)
    index = build_media_index(settings, dataset)
    print(f"images: {len(index.images)}  voice notes: {len(index.voices)}")
    for mid, img in sorted(index.images.items()):
        print(f"  {mid:9s} {img.layout:20s} words={img.ocr_word_count:4d} qr={int(img.has_qr)} "
              f"tags={','.join(img.scene_tags)}")
    for mid, vn in sorted(index.voices.items()):
        print(f"  {mid:9s} {vn.duration_s:6.1f}s wpm={vn.words_per_minute:6.1f} :: {vn.transcript[:80]}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """End-to-end submission readiness check.

    Runs the pipeline cold, validates the output contract, scores the labelled
    samples, and confirms no message id is hardcoded into the decision path.
    Prints a PASS/FAIL checklist so the whole submission can be verified with
    one command.
    """
    import re

    settings = build_settings(args)
    checks: list[tuple[str, bool, str]] = []

    dataset = load_dataset(settings)
    checks.append(("dataset loads", bool(dataset.messages), f"{len(dataset.messages)} messages"))

    media = build_media_index(settings, dataset)
    referenced = {m.media_id for m in dataset.messages if m.has_media}
    resolved = set(media.images) | set(media.voices)
    checks.append(("all referenced media resolves", referenced <= resolved,
                   f"{len(referenced & resolved)}/{len(referenced)} resolved"))

    router = NotificationRouter(settings, dataset, media)
    decisions, stats, critic = router.run()
    checks.append(("one decision per message", len(decisions) == len(dataset.messages),
                   f"{len(decisions)} decisions"))
    checks.append(("self-critic found no violations", critic.repaired == 0,
                   f"{critic.repaired} repaired"))

    out_path = Path(args.output).resolve() if args.output else settings.dataset_dir / "output.csv"
    write_output(out_path, decisions)
    report = validate_output(out_path, [m.message_id for m in dataset.messages])
    checks.append(("output.csv matches the required contract", report.ok,
                   "; ".join(report.errors) or str(out_path.name)))

    if dataset.samples:
        sample_decisions, _, _ = router.run(list(dataset.samples))
        r = evaluate(sample_decisions, dataset.samples)
        checks.append(("labelled-sample action accuracy >= 90%", r.action_accuracy >= 0.90,
                       f"{r.action_accuracy:.1%}"))
        checks.append(("labelled-sample type accuracy >= 90%", r.type_accuracy >= 0.90,
                       f"{r.type_accuracy:.1%}"))
        checks.append(("no safety-critical errors", r.critical_error_count == 0,
                       f"{r.critical_error_count} errors"))
        checks.append(("confidence tracks the reference", r.mean_confidence_error <= 0.05,
                       f"mean abs error {r.mean_confidence_error:.3f}"))

    # No message-specific answers may live in the decision path.
    id_pattern = re.compile(r"\b(?:msg_\d+|sample_msg_\d+|message_\d{3,})\b")
    offenders = []
    for path in sorted((Path(__file__).parent / "router").glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if id_pattern.search(line):
                offenders.append(f"{path.name}:{lineno}")
    checks.append(("no hardcoded message ids in decision path", not offenders,
                   ", ".join(offenders) or "clean"))

    width = max(len(name) for name, _, _ in checks)
    print()
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name:<{width}}  {detail}")
    failed = [name for name, passed, _ in checks if not passed]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed"
          f"{'' if not failed else '  FAILED: ' + ', '.join(failed)}")
    print(f"predictions written to {out_path}")
    return 0 if not failed else 1


def cmd_ablate(args: argparse.Namespace) -> int:
    """Measure each component's contribution by disabling it in isolation."""
    base = build_settings(args)
    dataset = load_dataset(base)
    if not dataset.samples:
        log.error("no labelled samples found")
        return 1

    variants: list[tuple[str, Settings]] = [
        ("full system", base),
        ("no embeddings (lexical retrieval)", replace(base, use_embeddings=False)),
        ("no OCR", replace(base, multimodal=replace(base.multimodal, enable_ocr=False))),
        ("no ASR", replace(base, multimodal=replace(base.multimodal, enable_asr=False))),
        ("no media understanding",
         replace(base, multimodal=replace(base.multimodal, enable_ocr=False, enable_asr=False))),
    ]

    # Labelled-sample accuracy alone is too coarse here: only 8 of 30 samples
    # carry media, and those rows have strong redundant sender signals. Decision
    # churn over all 110 messages measures what a component actually changes.
    baseline_full = None
    header = (f"{'variant':34s} {'action':>7s} {'type':>7s} {'joint':>7s} "
              f"{'evid':>6s} {'crit':>5s} {'changed/110':>12s}")
    print(header)
    print("-" * len(header))

    for name, settings in variants:
        media = build_media_index(settings, dataset)
        router = NotificationRouter(settings, dataset, media)

        sample_decisions, _, _ = router.run(list(dataset.samples))
        r = evaluate(sample_decisions, dataset.samples)

        full, _, _ = router.run()
        signature = {d.message_id: (d.action, d.message_type) for d in full}
        if baseline_full is None:
            baseline_full = signature
            changed = "-"
        else:
            diff = sum(1 for k, v in signature.items() if baseline_full.get(k) != v)
            changed = f"{diff}"

        print(f"{name:34s} {r.action_accuracy:6.1%} {r.type_accuracy:6.1%} "
              f"{r.joint_accuracy:6.1%} {r.evidence_recall:5.0%} "
              f"{r.critical_error_count:5d} {changed:>12s}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Global flags live on a parent parser so they are accepted both before and
    # after the subcommand. Argparse otherwise rejects `run --use-llm`, which is
    # the order anyone would naturally type.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    common.add_argument("--dataset", help="path to dataset directory")
    common.add_argument("--use-llm", action="store_true",
                        help="opt in to LLM arbitration (needs OPENAI_API_KEY; measured as a "
                             "net negative on this dataset, so it is off by default)")
    common.add_argument("--no-llm", action="store_true",
                        help="force fully deterministic mode (already the default)")
    common.add_argument("--no-embeddings", action="store_true", help="use lexical retrieval only")

    parser = argparse.ArgumentParser(prog="sentinel", description=__doc__, parents=[common],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser("run", parents=[common],
                           help="route all messages and write output.csv")
    p_run.add_argument("-o", "--output", help="output CSV path")
    p_run.add_argument("--trace", action="store_true", help="also write per-decision JSONL traces")
    p_run.set_defaults(func=cmd_run)

    sub.add_parser("eval", parents=[common],
                   help="evaluate against labelled samples").set_defaults(func=cmd_eval)

    p_explain = sub.add_parser("explain", parents=[common], help="print a full decision trace")
    p_explain.add_argument("message_id")
    p_explain.set_defaults(func=cmd_explain)

    sub.add_parser("media", parents=[common],
                   help="rebuild and dump the media cache").set_defaults(func=cmd_media)
    sub.add_parser("ablate", parents=[common],
                   help="component ablation study").set_defaults(func=cmd_ablate)

    p_verify = sub.add_parser("verify", parents=[common],
                              help="end-to-end submission readiness check")
    p_verify.add_argument("-o", "--output", help="output CSV path")
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
