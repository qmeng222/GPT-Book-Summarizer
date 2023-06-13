# GPT Book Summarizer

## About:

- Description: An automated book summarizer using gpt-3.5-turbo model and recursive meta summarization techniques.
- Highlights:
  1. Implemented a token limit management strategy to ensure efficient summarization within model constraints.
  2. Utilized caching to store and retrieve summaries, enhancing performance and handling potential request failures.
  3. Incorporated rate limit control logic to prevent excessive requests.
- Tech stack: GPT, Python
- Overview:
  - Recursive meta summarization:
    ![recursive meta summarization](/images/tree-structure.png)
  - The Great Gatsby as the test case:
    ![test case](/images/GPT-Book-Summarizer_overview.png)

## Setup:

1. Create & activate the virtual environment (and all subsequent steps will be performed within the virtual environment):
   ```
   python -m venv env
   source env/bin/activate
   ```
2. Upgrade the pip package manager to the latest version within the current Python environment: `python -m pip install --upgrade pip`
3. Install libraries/packages:
   - Install the OpenAI package and the python-dotenv package:
     ```
     pip install openai
     pip install python_dotenv
     ```
     or simply `pip install openai python_dotenv` altogether!
   - Install the `requests` library to simplify the process of making HTTP requests and handling their responses by `pip install requests`
   - Install the tiktoken library to count the number of tokens: `pip install tiktoken`
4. Generate a snapshot of the installed packages and their versions to the requirements.txt file (or overwrite the txt file): `pip freeze > requirements.txt`, so others can install all the packages and their respective versions specified in the requirements.txt file with `pip install -r requirements.txt`

## Resources:

1. [Project Gutenberg is a library of over 70,000 free eBooks.](https://www.gutenberg.org/)
2. [Max tokens for each model](https://platform.openai.com/docs/models/continuous-model-upgrades)
3. [OpenAI rate limit](https://platform.openai.com/docs/guides/rate-limits/overview)
