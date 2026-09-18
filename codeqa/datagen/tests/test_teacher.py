from codeqa.datagen.sources import teacher
from codeqa.shared.contracts import Span


def test_parse_authored_accepts_well_formed_and_rejects_bad():
    good = '''Here it is:
    {"task_type": "trace", "question": "Where is the session cookie signed?", "answer": "In save_session [src/flask/sessions.py:L310-L340].",
     "rubric": ["names SecureCookieSessionInterface", "says the cookie is signed with the secret key", "cites save_session"],
     "evidence": [{"path": "./src/flask/sessions.py", "start": 310, "end": 340}]}'''
    p = teacher.parse_authored(good)
    assert p and p["task_type"] == "trace" and p["evidence"] == [Span(path="src/flask/sessions.py", start=310, end=340)]
    assert teacher.parse_authored('{"question": "x", "answer": "y", "rubric": ["a"], "evidence": []}') is None     # rubric too short, no evidence
    assert teacher.parse_authored("no json here") is None
    weird = '{"task_type": "value", "question": "q", "answer": "a", "rubric": ["1", "2", "3"], "evidence": [{"path": "a.py", "start": 1, "end": 2}]}'
    assert teacher.parse_authored(weird)["task_type"] == "explain"        # unknown type falls back


def test_clip_to_read_keeps_only_lines_actually_read():
    read = [Span(path="a.py", start=100, end=200), Span(path="b.py", start=1, end=10)]
    ev = [Span(path="a.py", start=150, end=260), Span(path="b.py", start=20, end=30), Span(path="c.py", start=1, end=5)]
    out = teacher.clip_to_read(ev, read)
    assert out == [Span(path="a.py", start=150, end=200)]
    long_read = [Span(path="a.py", start=1, end=500)]
    assert teacher.clip_to_read([Span(path="a.py", start=1, end=500)], long_read, max_lines=80) == [Span(path="a.py", start=1, end=80)]


def test_seed_prompt_mentions_symbols():
    s = teacher.seed_prompt("src/x.py", ["A", "A.run", "helper"])
    assert "src/x.py" in s and "A.run" in s
