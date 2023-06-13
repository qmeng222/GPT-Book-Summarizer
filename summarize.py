from __future__ import division  # enable the Python 3.x division behavior
import openai
from dotenv import load_dotenv
from openai.error import APIConnectionError, APIError, RateLimitError
import requests

import os  # import the operating system module to interact with the underlying OS
import re  # import the regular expression module to work with regular expressions
import tiktoken
import random
import time
from typing import Dict, List  # import specific types from the typing module
from utilities import (
    num_tokens_from_messages,
    summarization_prompt_messages,
    split_text_into_sections,
    memoize_to_file,
    split_text_into_sections,
)


load_dotenv(".env")
openai.api_key = os.environ["OPENAI_API_KEY"]

# BOOKS:
# Book1: Great Gatsby:
response = requests.get("https://www.gutenberg.org/cache/epub/64317/pg64317.txt")

# # Book2: PETER PAN:
# response = requests.get("https://www.gutenberg.org/files/16/16-0.txt")

# # Book3: Metamorphosis:
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


# estimate price:
model_name = "gpt-3.5-turbo"
# obtain the encoding object (enc) for a specific language model:
enc = tiktoken.encoding_for_model(model_name)
num_tokens = len(enc.encode(book))
print(f"Text contains {num_tokens} tokens")
cost_per_token = 0.002 / 1000
print(
    f"The approximate price of this summary will be: ${num_tokens * cost_per_token:.2f}"
)


# summerize:
actual_tokens = 0
MAX_ATTEMPTS = 3
division_point = "."  # summarize text endding on "."


def gpt_summarize(text: str, target_summary_size: int) -> str:
    global actual_tokens
    # Otherwise, we can just summarize the text directly
    tries = 0
    while True:
        try:
            tries += 1
            result = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=summarization_prompt_messages(text, target_summary_size),
            )
            actual_tokens += result.usage.total_tokens
            return "[[[" + result.choices[0].message.to_dict()["content"] + "]]]"
        except (APIConnectionError, APIError, RateLimitError) as e:
            if tries >= MAX_ATTEMPTS:
                print(f"OpenAI exception after {MAX_ATTEMPTS} tries. Aborting. {e}")
                raise e
            if hasattr(e, "should_retry") and not e.should_retry:
                print(f"OpenAI exception with should_retry false. Aborting. {e}")
                raise e
            else:
                print(f"Summarize failed (Try {tries} of {MAX_ATTEMPTS}). {e}")
                random_wait = (
                    random.random() * 4.0 + 1.0
                )  # Wait between 1 and 5 seconds
                random_wait = (
                    random_wait * tries
                )  # Scale that up by the number of tries (more tries, longer wait)
                time.sleep(random_wait * tries)


from dataclasses import dataclass  # import the dataclass decorator


@dataclass(frozen=True, repr=True)
class SummarizationParameters:
    target_summary_size: int
    summary_input_size: int


def summarization_token_parameters(
    target_summary_size: int, model_context_size: int
) -> SummarizationParameters:
    """
    Compute the number of tokens that should be used for the context window, the target summary, and the base prompt.
    """
    base_prompt_size = num_tokens_from_messages(
        summarization_prompt_messages("", target_summary_size), model=model_name
    )
    summary_input_size = model_context_size - (base_prompt_size + target_summary_size)
    return SummarizationParameters(
        target_summary_size=target_summary_size,
        summary_input_size=summary_input_size,
    )


@memoize_to_file(cache_file="cache.json")
def summarize(
    text: str,
    token_quantities: SummarizationParameters,
    division_point: str,
    model_name: str,
) -> str:
    # Shorten text for our console logging
    text_to_print = re.sub(r" +\|\n\|\t", " ", text).replace("\n", "")
    print(
        f"\nSummarizing {len(enc.encode(text))}-token text: {text_to_print[:60]}{'...' if len(text_to_print) > 60 else ''}"
    )

    if len(enc.encode(text)) <= token_quantities.target_summary_size:
        # If the text is already short enough, just return it
        return text
    elif len(enc.encode(text)) <= token_quantities.summary_input_size:
        summary = gpt_summarize(text, token_quantities.target_summary_size)
        print(
            f"Summarized {len(enc.encode(text))}-token text into {len(enc.encode(summary))}-token summary: {summary[:250]}{'...' if len(summary) > 250 else ''}"
        )
        return summary
    else:
        # The text is too long, split it into sections and summarize each section
        split_input = split_text_into_sections(
            text, token_quantities.summary_input_size, division_point, model_name
        )

        summaries = [
            summarize(x, token_quantities, division_point, model_name)
            for x in split_input
        ]

        return summarize(
            "\n\n".join(summaries), token_quantities, division_point, model_name
        )


@memoize_to_file(cache_file="cache.json")
def synthesize_summaries(summaries: List[str], model: str) -> str:
    """
    Use a more powerful GPT model to synthesize the summaries into a single summary.
    """
    print(f"Synthesizing {len(summaries)} summaries into a single summary.")

    summaries_joined = ""
    for i, summary in enumerate(summaries):
        summaries_joined += f"Summary {i + 1}: {summary}\n\n"

    messages = [
        {
            "role": "user",
            "content": f"""
A less powerful GPT model generated {len(summaries)} summaries of a book.

Because of the way that the summaries are generated, they may not be perfect. Please review them
and synthesize them into a single more detailed summary that you think is best.

The summaries are as follows: {summaries_joined}
""".strip(),
        },
    ]

    # check that the summaries are short enough to be synthesized:
    assert num_tokens_from_messages(messages, model=model_name) <= 8192
    print(messages)

    result = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    return result.choices[0].message.to_dict()["content"]


summaries: Dict[int, str] = {}
target_summary_sizes = [500, 750, 1000]
for target_summary_size in target_summary_sizes:
    actual_tokens = 0
    summaries[target_summary_size] = (
        summarize(
            book,
            summarization_token_parameters(
                target_summary_size=target_summary_size, model_context_size=4097
            ),
            division_point,
            model_name,
        )
        .replace("[[[", "")
        .replace("]]]", "")
    )
print(summaries)

print(synthesize_summaries(list(summaries.values()), "gpt-3.5-turbo"))


# summary = (
#     summarize(
#         book,
#         summarization_token_parameters(
#             target_summary_size=1000, model_context_size=4097
#         ),
#         division_point,
#         model_name,
#     )
#     .replace("[[[", "")
#     .replace("]]]", "")
# )
# print(summary)
