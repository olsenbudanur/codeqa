from codeqa.evals import cqb_judge as cj


def test_parse_fenced_and_bare_json():
    fenced = 'Analysis... \n```json\n{"accuracy": 4, "completeness": 3, "specificity": 5, "explanation": "ok",}\n```'
    assert cj.parse_response(fenced)["accuracy"] == 4                      # trailing comma sanitised
    bare = "thinking {\"x\": 1} more text {\"accuracy\": 2, \"completeness\": 2, \"specificity\": 1}"
    assert cj.parse_response(bare)["specificity"] == 1                     # last block with 'accuracy' wins


def test_aggregate_median_and_score():
    agg = cj.aggregate([{"accuracy": 5, "completeness": 4, "specificity": 2}, {"accuracy": 3, "completeness": 4, "specificity": 4},
                        {"accuracy": 4, "completeness": 5, "specificity": 3}])
    assert (agg["accuracy"], agg["completeness"], agg["specificity"]) == (4, 4, 3) and abs(agg["score"] - 11 / 15) < 1e-9


def test_prompt_position_randomisation_labels():
    p = cj.build_prompt("Q?", "GOLD", "AGENT", ["r1"], gold_first=False)
    assert "## Answer A\nAGENT" in p and "evaluate **Answer A**" in p and "- r1" in p
    p2 = cj.build_prompt("Q?", "GOLD", "AGENT", [], gold_first=True)
    assert "## Answer B\nAGENT" in p2 and "(no rubric)" in p2
