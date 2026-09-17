from app.data import clean_line, clean_text


def test_clean_line_normalizes_text():
    assert clean_line("  Hello,   RNN!  ") == "hello rnn"


def test_clean_text_preserves_non_empty_lines():
    assert clean_text("Hello!\n\nMachine Learning.") == "hello\nmachine learning"
