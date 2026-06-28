"""quizforge command line.

    quizforge grow   BANK [--material-file F] [--mc N ...] [--dry-run] --llm mod:factory
    quizforge sample BANK [--seed N] [--json]
    quizforge stats  BANK

`grow` needs a chat model. Because quizforge is provider-neutral, you point it at
one you construct:

    # a dotted path to a zero-arg callable returning a LangChain-style model
    quizforge grow bank.yaml --llm myproject.models:make_llm

    # or, if langchain-openai is installed and OPENAI_API_KEY is set
    quizforge grow bank.yaml --openai-model gpt-4.1
"""

import argparse
import importlib
import json
import logging
import random
import sys
from collections import Counter
from typing import Optional

from .bank import Bank
from .generate import DEFAULT_TARGETS
from .schemas import FORMAT_KEYS


def _resolve_llm(args):
    """Build the chat model from --llm dotted path or --openai-model."""
    if args.llm:
        module_path, _, attr = args.llm.partition(":")
        if not module_path or not attr:
            raise SystemExit("--llm must be 'module.path:callable' (a zero-arg factory)")
        factory = getattr(importlib.import_module(module_path), attr)
        return factory()
    if args.openai_model:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise SystemExit("--openai-model needs 'pip install langchain-openai'")
        return ChatOpenAI(model=args.openai_model, temperature=args.temperature)
    raise SystemExit("grow needs a model: pass --llm module:factory or --openai-model NAME")


def _cmd_grow(args) -> int:
    bank = Bank.load(args.bank)
    targets = {f: getattr(args, f) for f in FORMAT_KEYS if getattr(args, f) is not None}
    material = None
    if args.material_file:
        material = open(args.material_file).read()

    llm = _resolve_llm(args)
    before = len(bank)
    new = bank.grow(llm, targets=targets or None, material=material,
                    batch_size=args.batch, coverage=args.coverage or None)
    print(f"Generated {len(new)} new questions ({dict(Counter(q['type'] for q in new))}).")
    if args.dry_run:
        print(f"--dry-run: not writing. Bank would grow {before} -> {before + len(new)}.")
        for q in new[:3]:
            print(f"  [{q['type']}/{q['difficulty']}] {q['prompt'][:90]}")
        return 0
    bank.save(args.bank)
    print(f"Wrote {args.bank}. Bank now {len(bank)} (was {before}).")
    return 0


def _cmd_sample(args) -> int:
    bank = Bank.load(args.bank)
    rng = random.Random(args.seed) if args.seed is not None else None
    test = bank.sample(rng=rng)
    if args.json:
        json.dump(test, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return 0
    counts = Counter(q.get("type", "mc") for q in test)
    print(f"Sampled {len(test)} questions: {dict(counts)}")
    for i, q in enumerate(test, 1):
        print(f"  {i:2d}. [{q.get('type')}/{q.get('difficulty')}] {q.get('prompt', '')[:90]}")
    return 0


def _cmd_stats(args) -> int:
    bank = Bank.load(args.bank)
    print(f"Bank: {args.bank}")
    print(f"  total questions : {len(bank)}")
    print(f"  by format       : {bank.counts_by_type()}")
    print(f"  by difficulty   : {bank.counts_by_difficulty()}")
    if bank.material:
        print(f"  material        : {len(bank.material)} chars")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quizforge",
                                description="Generate, sample, and grade mixed-format question banks.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grow", help="generate the shortfall to reach per-format targets")
    g.add_argument("bank", help="path to the bank YAML file")
    g.add_argument("--material-file", help="text file to ground generation (overrides bank 'material:')")
    g.add_argument("--batch", type=int, default=8, help="questions per LLM call (default 8)")
    g.add_argument("--coverage", default=None, help="coverage steering hint for the generator")
    g.add_argument("--dry-run", action="store_true", help="generate and summarize, but don't write")
    g.add_argument("--llm", help="dotted path 'module:callable' to a zero-arg model factory")
    g.add_argument("--openai-model", help="convenience: build ChatOpenAI with this model name")
    g.add_argument("--temperature", type=float, default=0.4, help="temperature for --openai-model")
    for f in FORMAT_KEYS:
        g.add_argument(f"--{f.replace('_', '-')}", dest=f, type=int, default=None,
                       help=f"target TOTAL {f} questions (default {DEFAULT_TARGETS[f]})")
    g.set_defaults(func=_cmd_grow)

    s = sub.add_parser("sample", help="draw one mixed-format test from the bank")
    s.add_argument("bank")
    s.add_argument("--seed", type=int, default=None, help="rng seed for a reproducible draw")
    s.add_argument("--json", action="store_true", help="emit the sampled questions as JSON")
    s.set_defaults(func=_cmd_sample)

    st = sub.add_parser("stats", help="show bank size and format/difficulty mix")
    st.add_argument("bank")
    st.set_defaults(func=_cmd_stats)
    return p


def main(argv: Optional[list] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
