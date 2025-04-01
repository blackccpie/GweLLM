# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# This is a custom implementation othe "Magpie" alignment data synthesis technique:
# https://magpie-align.github.io/

import re

def clean_question(question: str) -> str:
    """
    Clean the question from unwanted formatting
    """
    # regex explanation:
    # ^\d+[\).]?\s* → removes leading numbers (e.g., "1)", "5.", etc.) and spaces
    # ^["«]* → removes leading quotes ("), guillemets («), and dash (-)
    question = re.sub(r'^(?:\d+[\).]?\s*)?["«-]*', '', question).strip()
    # ensure the first letter is capitalized
    return question[0].upper() + question[1:]

def extract_question(model_output):
    """
    Extract the question from model generation
    """
    try:
        extract_question = f"{model_output.split('?')[0].strip()}?"
        print(rf"extracted Q: {extract_question}")
        return clean_question(extract_question)
    except:
        print(f"failed extracting question from output: {model_output}")
        return None

def clean_answer(model_output: str) -> str:
    """
    Clean the answer from unwanted formatting
    """
    # use regex to remove leading spaces and a colon (if present), along with spaces after the colon.
    answer = re.sub(r'^\s*:\s*', '', model_output).strip()
    return answer

def extract_answer(model_output: str) -> str:
    """
    Clean the question from leading unwanted formatting
    """
    try:
        extract_answer = f"{model_output.split('.')[0].strip()}."
        print(rf"extracted A: {extract_answer}")
        return clean_answer(extract_answer)
    except:
        print("failed extracting answer")
        return None

#class magpie_instruct_processor:

