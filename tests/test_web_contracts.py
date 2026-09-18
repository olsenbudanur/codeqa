"""The web app mirrors CITATION_RE in TypeScript. Keep the two identical."""
import re
from pathlib import Path

from codeqa.shared.contracts import CITATION_RE

TS = Path(__file__).resolve().parents[1] / "apps" / "web" / "src" / "lib" / "contracts.ts"


def test_web_citation_regex_matches_contract():
    src = TS.read_text()
    m = re.search(r"export const CITATION_RE = /(.*)/g", src)
    assert m, "CITATION_RE not found in apps/web/src/lib/contracts.ts"
    assert m.group(1) == CITATION_RE.pattern
