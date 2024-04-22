import nltk
from nltk.tokenize import word_tokenize

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download the necessary NLTK models (if not already downloaded)
nltk.download('punkt')

# Example usage of the tokenizer
text = "Hello, how are you doing?"
tokens = word_tokenize(text)
print(tokens)

print(nltk.data.path)