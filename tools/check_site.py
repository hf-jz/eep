#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_site.py -- 页面自检: 标签平衡 + 可见文本无 em/en dash + 资源存在

    python3 tools/check_site.py index.html
"""
from __future__ import annotations

import os
import re
import sys
from html.parser import HTMLParser

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr"}


class Balance(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack:
            self.errors.append(f"stray </{tag}> at {self.getpos()}")
            return
        top, pos = self.stack.pop()
        if top != tag:
            self.errors.append(f"</{tag}> at {self.getpos()} closes <{top}> opened at {pos}")


def main(path):
    src = open(path, encoding="utf-8").read()
    p = Balance()
    p.feed(src)
    p.close()
    errs = list(p.errors)
    for tag, pos in p.stack:
        errs.append(f"unclosed <{tag}> opened at {pos}")

    body = src.split("</head>", 1)[-1]
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body)
    dashes = re.findall(r"[\u2014\u2013]", text)

    missing = []
    for m in re.finditer(r'(?:href|src)="([^"]+)"', src):
        ref = m.group(1)
        if ref.startswith(("http", "#", "data:", "mailto:")):
            continue
        if not os.path.exists(os.path.join(os.path.dirname(path) or ".", ref)):
            missing.append(ref)

    print(f"file            : {path}")
    print(f"bytes           : {len(src)}")
    print(f"tag errors      : {len(errs)}")
    for e in errs[:10]:
        print(f"   {e}")
    print(f"em/en dashes    : {len(dashes)}")
    print(f"missing assets  : {missing if missing else 'none'}")
    ok = not errs and not dashes and not missing
    print("RESULT          :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "index.html"))
