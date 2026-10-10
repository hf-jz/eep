#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_pdf_text.py -- assert the paper PDF contains no CJK, and locate key pages.

    python3 tools/check_pdf_text.py paper/eep.pdf
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys

HAN = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
NEEDLES = ["An exactly solvable instantiation",
           "Contrast error against the exact gradient",
           "Order of the contrast",
           "Agreement with the reference implementation"]
CJK_FONTS = (b"Songti", b"STSong", b"STFangsong", b"STKaiti", b"STHeiti",
             b"PingFang", b"Hiragino", b"Heiti", b"Fangsong", b"SimSun", b"SimHei",
             b"NotoSansCJK", b"NotoSerifCJK", b"CJK")


def font_check(path):
    """No embedded CJK font means no CJK glyph can be drawn, whatever the source says."""
    raw = open(path, "rb").read()
    found = sorted({m.decode("latin-1") for m in CJK_FONTS if m in raw})
    return found


def main(path):
    fonts = font_check(path)
    print(f"file   : {path}")
    print(f"CJK font names in the PDF : {fonts if fonts else 'none'}")
    if shutil.which("pdftotext"):
        out = subprocess.run(["pdftotext", "-layout", path, "-"],
                             capture_output=True, text=True, check=True).stdout
        pages = out.split("\f")
        print(f"pages  : {len([p for p in pages if p.strip()])}")
        hits = [(i, "".join(dict.fromkeys(HAN.findall(p)))[:60])
                for i, p in enumerate(pages, 1) if HAN.findall(p)]
        print(f"CJK in text layer : {len(hits)} page(s)")
        for i, s in hits:
            print(f"   page {i}: {s}")
        for needle in NEEDLES:
            where = [i for i, p in enumerate(pages, 1) if needle in p]
            print(f"   '{needle[:42]}' -> page {where if where else 'NOT FOUND'}")
    else:
        hits = []
        print("pdftotext      : not installed, byte-level font scan used instead")
    ok = not hits and not fonts
    print("RESULT :", "PASS (no CJK)" if ok else "FAIL (CJK present)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "paper/eep.pdf"))
