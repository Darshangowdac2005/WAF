# Change these lines in ml/layer2a/__init__.py
from feature_engineering.extractor import extract_features, to_vector, extract_vector, FEATURE_NAMES
from feature_engineering.tokenizer import CharTokenizer, VOCAB_SIZE
from feature_engineering.normalizer import Normalizer  # Note: your file used 'Normalizer' class