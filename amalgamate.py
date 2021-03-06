#!/usr/bin/env python3

# Create amalgamated version of the library
import os
import sys
import re

ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
OUTPUT = os.path.join(ROOT, "amalgamated")
INCLUDE_NAME = "catboost.h"
SRC_NAME = "catboost.cpp"

_INCLUDE_RE = re.compile(r'# *include *[<"]([^">]*)[">]')
class Copy:
    def __init__(self, filename):
        self.filename = filename

    def pre_write(self, out):
        pass

    def post_write(self, out):
        pass

    def process(self, out, includes):
        out.write(f"// FILE: {self.filename}\n")
        self.pre_write(out)
        with open(os.path.join(ROOT, self.filename), "rt", encoding = "utf-8") as f:
            for line in f:
                m = _INCLUDE_RE.match(line)
                if m:
                    inm = os.path.basename(m.group(1))
                    if inm in includes:
                        out.write(f"// {line}")
                    else:
                        out.write(line)
                elif line.find("#pragma once") >= 0:
                    pass
                else:
                    out.write(line)
        self.post_write(out)

class HppCopy(Copy):
    def pre_write(self, out):
        out.write("#ifdef __cplusplus\n")

    def post_write(self, out):
        out.write("#endif // __cplusplus\n")

HEADERS = [
        HppCopy("include/catboost.hpp"),
        Copy("include/cb.h"),
]

SOURCES = [
        Copy("src/vec4.hpp"),
        Copy("src/json.hpp"),
        Copy("src/catboost.cpp"),
        Copy("src/cb.cpp"),
]

def main():
    os.makedirs(OUTPUT, exist_ok = True)

    includes = set()
    for h in HEADERS:
        includes.add(os.path.basename(h.filename))

    for h in SOURCES:
        if h.filename.endswith(".h") or h.filename.endswith(".hpp"):
            includes.add(os.path.basename(h.filename))

    with open(os.path.join(OUTPUT, INCLUDE_NAME), "wt", encoding = "utf-8") as out:
        out.write("#pragma once\n")
        out.write("// This file is generated by amalgamate.py. Do not edit.\n")
        for h in HEADERS:
            h.process(out, includes)

    with open(os.path.join(OUTPUT, SRC_NAME), "wt", encoding = "utf-8") as out:
        out.write("// This file is generated by amalgamate.py. Do not edit.\n")
        out.write(f'#include "{INCLUDE_NAME}"')
        for h in SOURCES:
            h.process(out, includes)

if __name__ == '__main__':
    main()
