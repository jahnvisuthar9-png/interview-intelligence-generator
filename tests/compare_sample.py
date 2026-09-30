"""Render the sample's content through the app and compare with the reference PDF.
Run: python -m tests.compare_sample"""
import re, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.analysis import Inputs, validate
from pipeline.package import build_all
from tests.sample_fixture import SAMPLE, IDS
from pypdf import PdfReader

REF = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "reference_sample.pdf"
OUT = ROOT / "tests" / "compare"

p = validate(SAMPLE, Inputs("Google", "AI/ML Engineer", "Round 2", []), 10, IDS, [], False)
p["meta"].update(transcript_files=[], duplicates=[], groups=[], grouping_source="fixture",
                 notes=[], rejected_files=[], model="sample-fixture")
res = build_all(p, OUT)
for c in res["checks"]:
    print(("PASS " if c["passed"] else "FAIL ") + c["check"] + " - " + c["detail"])

ours = OUT / res["pdf"]
def info(path):
    r = PdfReader(str(path)); pg = r.pages[0]
    return len(r.pages), [float(v) for v in pg.mediabox[2:]], re.sub(r"\s+", " ", pg.extract_text()).upper()
rn, rsize, rtext = info(REF); on, osize, otext = info(ours)
print(f"pages ref={rn} ours={on}; size ref={rsize} ours={osize}")
sections = ["REVIEW", "PRACTICE OUT LOUD", "REMEMBER", "ROUND COVERAGE / TOP TOPICS", "OBSERVED RESUME FOCUS",
            "TOP QUESTIONS", "PREPARATION SIGNAL", "INTERVIEW-DERIVED ROLE FOCUS", "REVISION PRIORITY",
            "30-MINUTE REVISION ORDER", "CANDIDATE INTERVIEWS ANALYZED", "TARGET: ROUND 2"]
for s in sections:
    print(f"{'ok ' if (s in rtext) == (s in otext) else 'DIFF'} {s:32} ref={s in rtext} ours={s in otext}")
counts = lambda t: sorted(re.findall(r"\b\d{1,2}/10\b", t))
print("topic counts ref", counts(rtext)); print("topic counts ours", counts(otext))
subprocess.run(["pdftoppm", "-png", "-r", "90", "-singlefile", str(REF), str(OUT / "ref")], check=True)
subprocess.run(["pdftoppm", "-png", "-r", "90", "-singlefile", str(ours), str(OUT / "ours")], check=True)
from PIL import Image
a, b = Image.open(OUT / "ref.png"), Image.open(OUT / "ours.png")
side = Image.new("RGB", (a.width + b.width + 20, max(a.height, b.height)), (120, 120, 120))
side.paste(a, (0, 0)); side.paste(b, (a.width + 20, 0)); side.save(OUT / "side_by_side.png")
print("saved", OUT / "side_by_side.png")
