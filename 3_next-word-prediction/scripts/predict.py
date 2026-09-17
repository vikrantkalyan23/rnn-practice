from app.model.predictor import NextWordPredictor


predictor = NextWordPredictor(
    model_path="models/next_word_model.keras",
    tokenizer_path="models/tokenizer.json",
    sequence_length=5,
)


text = "machine learning"

next_word = predictor.predict_next_word(text)

print(f"Input: {text}")

print(f"Next word: {next_word}")


generated = predictor.generate_text(
    text,
    next_words=5,
)

print(f"Generated: {generated}")
