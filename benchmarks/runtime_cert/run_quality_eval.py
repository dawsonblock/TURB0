"""
run_quality_eval.py — TurboQuant quality evaluation gate.

NOT YET IMPLEMENTED.  This script previously fabricated passing metrics
from hard-coded constants.  That path has been removed.

Real quality evaluation requires:
  1. A loaded model + tokenizer (mlx_lm.load)
  2. A representative prompt file
  3. Two generation passes (dense baseline, TurboQuant enabled)
  4. Perplexity and KL-divergence computation via turboquant.eval

Until the real evaluation path is wired, this script exits nonzero to
prevent any CI or certification harness from minting a false PASS artifact.
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant quality evaluation gate")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--prompt-file", required=True, help="Path to a .jsonl prompt file")
    parser.add_argument("--output-dir", required=True, help="Directory for artifacts")
    parser.add_argument("--max-delta-ppl", type=float, default=0.5)
    parser.add_argument("--max-mean-kl", type=float, default=0.1)
    parser.parse_args()  # validate flags; do not use results

    print(
        "ERROR: run_quality_eval.py is not yet implemented.\n"
        "Real evaluation (dense vs TurboQuant perplexity + KL) has not been wired.\n"
        "Do not interpret the absence of a FAIL artifact as a PASS.",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
