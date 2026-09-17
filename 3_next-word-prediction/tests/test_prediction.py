import numpy as np

from app.data import (
    create_sequences,
    create_tokenizer,
    get_vocabulary_size,
    split_text_train_validation_test,
    split_text_by_line,
)


def test_sequences_have_expected_shape_and_target():
    tokenizer = create_tokenizer("machine learning uses data")
    X, y = create_sequences(
        "machine learning uses data",
        tokenizer,
        sequence_length=3,
    )

    assert X.shape == (3, 3)
    assert y.shape == (3,)
    assert np.array_equal(X[0], np.array([0, 0, tokenizer.word_index["machine"]]))
    assert y[0] == tokenizer.word_index["learning"]


def test_line_split_is_repeatable_and_keeps_lines_separate():
    text = "one sentence\ntwo sentence\nthree sentence\nfour sentence"
    first_split = split_text_by_line(text, validation_ratio=0.25, random_seed=11)
    second_split = split_text_by_line(text, validation_ratio=0.25, random_seed=11)

    assert first_split == second_split
    train_text, validation_text = first_split
    assert len(train_text.splitlines()) == 3
    assert len(validation_text.splitlines()) == 1


def test_train_validation_test_split_is_repeatable_and_disjoint():
    text = "\n".join(f"sentence {index}" for index in range(10))
    first_split = split_text_train_validation_test(
        text,
        validation_ratio=0.2,
        test_ratio=0.2,
        random_seed=11,
    )
    second_split = split_text_train_validation_test(
        text,
        validation_ratio=0.2,
        test_ratio=0.2,
        random_seed=11,
    )

    assert first_split == second_split
    train_text, validation_text, test_text = first_split
    train_lines = set(train_text.splitlines())
    validation_lines = set(validation_text.splitlines())
    test_lines = set(test_text.splitlines())

    assert len(train_lines) == 6
    assert len(validation_lines) == 2
    assert len(test_lines) == 2
    assert train_lines.isdisjoint(validation_lines)
    assert train_lines.isdisjoint(test_lines)
    assert validation_lines.isdisjoint(test_lines)


def test_rare_words_are_not_used_as_prediction_targets():
    tokenizer = create_tokenizer(
        "common word common next rare",
        max_vocab_size=4,
    )
    _, targets = create_sequences(
        "common word common next rare",
        tokenizer,
        sequence_length=2,
    )

    assert get_vocabulary_size(tokenizer) == 4
    assert tokenizer.word_index["<OOV>"] not in targets
