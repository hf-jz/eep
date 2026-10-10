#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_prose.py -- flag informal or rhetorical phrasing in the paper source.

    python3 tools/check_prose.py paper/eep.tex

The list is register, not grammar: idioms that read as conversational, promotional
or rhetorical in a technical paper. Each hit prints with a line number so the fix
can be targeted. Exit code is non-zero while any hit remains.
"""
from __future__ import annotations

import re
import sys
# phrase -> suggested academic replacement (None = rewrite the sentence)
BANNED = {
    "before it ships": "at deployment",
    "at great expense": "at substantial computational cost",
    "is the answer": "constitutes the prediction",
    "do none of this": "do not operate in this way",
    "takes\nseriously": "asks",
    "how far a system can go": "what such a system can achieve",
    "literally one operator": "a single operator",
    "what the design has been missing": "what the design has so far lacked",
    "rather than merely possible": None,
    "easy to\nstate and hard to remove": None,
    "The question this paper takes": None,
    "not cosmetic": None,
    "carries exactly this": None,
    "has to match": "must match",
    "pushes outward": "points outward",
    "upgrades\nthis": None,
    "deserve to be stated plainly": None,
    "and it is something else": None,
    "It says the trajectory settles down": None,
    "The mechanism is": None,
    "still a gap": None,
    "shows up": "appears",
    "they answer different questions": None,
    "weakest link": "weakest assumption",
    "place to\npush": "direction for further work",
    "switched plasticity off": "eliminated plasticity",
    "really performs": "performs",
    "cleaned-up idealisation": "idealisation",
    "the signature of a system": None,
    "stopped pronouncing": None,
    "pronouncing on it": "prescribing it",
    "left to a reader": "left implicit",
    "stops being a metaphor": "a precise form",
    "prices how fast": "quantifies how rapid",
    "on top of the same structure": "on the same structure",
    "so that they can fail": "so that they are falsifiable",
    "Nothing here\nsupports": None,
    "A construction can be called": None,
    "one door": "a single channel",
    "It needs an energy": None,
    "describes the implementation that exists": None,
    "has to be": "must be",
    "we insist that": "we require that",
    "can do at birth": "innate fitness",
    "A value of zero would\nmean": None,
    "in the chain": None,
    "is not an analogy": None,
    "is what allows": "permits",
    "which is what makes": "which makes",
    "The improvement is not": None,
    "That is,": "use a comma",
    "Hence,": "therefore",
    "a lot of": "much",
    "not merely biological but adaptive": "adaptive rather than merely biological",
    "get a ": "obtain a ",
}

# Second pass: general conversational / promotional register. Regex, word-bounded.
BANNED_RE = {
    r"\breally\b": "remove",
    r"\bjust\b": "remove or 'only'",
    r"\bsimply\b": "remove",
    r"\bvery\b": "remove",
    r"\bquite\b": "remove",
    r"\bbasically\b": "remove",
    r"\bobviously\b": "remove",
    r"Of course": "remove",
    r"\betc\b": "and so on / name them",
    r"so-called": "remove",
    r"\bhuge\b": "large",
    r"\bnice\b": "remove",
    r"\bpretty\b": "remove",
    r"\bstuff\b": "remove",
    r"\bjob\b": "task",
    r"\bsheds light\b": "clarifies",
    r"\bcutting-edge\b": "recent / state of the art",
    r"\bnovel\b": "new (prefer the specific property)",
    r"\bpowerful\b": "capable / specify",
    r"\bmassive\b": "large",
    r"\bbreakthrough\b": "remove",
    r"within a life\b": "within a lifetime",
    r"\bstate of the art\b": "hyphenate as adjective",
}


def main(path):
    src = open(path, encoding="utf-8").read()
    hits = []
    for phrase, fix in BANNED.items():
        start = 0
        while True:
            idx = src.find(phrase, start)
            if idx < 0:
                break
            hits.append((src.count("\n", 0, idx) + 1, phrase, fix))
            start = idx + len(phrase)
    for pattern, fix in BANNED_RE.items():
        for m in re.finditer(pattern, src, flags=re.IGNORECASE):
            hits.append((src.count("\n", 0, m.start()) + 1, m.group(0), fix))
    hits.sort()
    print(f"file : {path}")
    print(f"hits : {len(hits)}")
    for i, phrase, fix in hits:
        mark = f" -> {fix}" if fix else " -> rewrite"
        print(f"  L{i:<4} {phrase!r}{mark}")
    print("RESULT :", "PASS" if not hits else "FAIL")
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "paper/eep.tex"))
