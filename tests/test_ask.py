from jobs import ask


def test_parse_any_question():
    assert ask.parse("gc 15m vsa trend") == ("GC", "15m", ["vsa", "trend"])
    assert ask.parse("/zoom euro") == ("6E", "15m", ["chart"])
    assert ask.parse("yen daily why swings") == ("6J", "1d", ["why", "swings"])
    assert ask.parse("cable 30m all")[2] == ["chart", "vsa", "trend", "swings", "effort", "why"]
    assert ask.parse("effort aussie weekly") == ("6A", "1w", ["effort"])
    assert ask.parse("kiwi") == ("6N", "4h", ["chart"])
    assert ask.parse("nothing here")[0] is None
