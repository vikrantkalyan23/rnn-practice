from app.data import create_tokenizer


def test_tokenizer_reserves_an_oov_token():
    tokenizer = create_tokenizer("machine learning uses data")

    assert tokenizer.word_index["<OOV>"] == 1
    assert tokenizer.texts_to_sequences(["unknown"])[0] == [1]
