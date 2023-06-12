import requests
import re  # import the regular expression module in Python
import tiktoken

# Great Gatsby:
# response = requests.get("https://www.gutenberg.org/cache/epub/64317/pg64317.txt")

# PETER PAN:
response = requests.get("https://www.gutenberg.org/files/16/16-0.txt")

# Metamorphosis:
# response = requests.get("https://www.gutenberg.org/files/5200/5200-0.txt")

assert response.status_code == 200
book_complete_text = response.text

"""
Carriage return characters are control characters that were used in older typewriters and early computer systems. In modern text data, carriage return characters may still exist as artifacts or remnants. In some systems, a line break is represented as a combination of "\r\n" (carriage return followed by newline), while in others, only "\n" is used.
"""
# eliminate the carriage return characters that might be present in the text:
book_complete_text = book_complete_text.replace("\r", "")

"""
eg:
import re

book_complete_text = "meta data\n*** text1 starts ***\ntext1\n*** text2 starts ***\ntext2\n*** copyright ***\nendding"
print(book_complete_text)

meta data
*** text1 starts ***
text1
*** text2 starts ***
text2
*** copyright ***
endding

split = re.split(r"\*\*\* .+ \*\*\*", book_complete_text)
print(split) # ['meta data\n', '\ntext1\n', '\ntext2\n', '\nendding']
"""
# remove Project Gutenberg's header and footer:
split = re.split(r"\*\*\* .+ \*\*\*", book_complete_text)
print("Divided into parts of length:", [len(part) for part in split])
book = split[1]

model_name = "gpt-3.5-turbo"
# obtain the encoding object (enc) for a specific language model:
enc = tiktoken.encoding_for_model(model_name)
num_tokens = len(enc.encode(book))
print(f"Text contains {num_tokens} tokens")
cost_per_token = 0.002 / 1000
print(
    f"The approximate price of this summary will be: ${num_tokens * cost_per_token:.2f}"
)
