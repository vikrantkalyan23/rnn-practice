# Temperature Prediction using RNN

uv venv --python 3.12

python -c "import tensorflow as tf; print(tf.__version__)"

uv pip install -r requirements.txt

uv run src/prepare_data.py

uv run src/model.py

uv run src/train.py

uv run src/evaluate.py

