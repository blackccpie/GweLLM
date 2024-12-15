# The MIT License

# Copyright (c) 2025 Albert Murienne

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

import requests

def translate_br2fr(str: str, verbose=False):
    """
    Translates input string from breton to french
    """

    # define the URL you want to send the request to
    tr_url_template = "https://apertium.org/apy/translate?langpair={langpair}&q={input}"

    # build final translation request URL
    lp = 'br|fr'
    tr_url = tr_url_template.format(langpair=lp, input=str)

    if verbose:
        print(tr_url)

    # send a GET request
    response = requests.get(tr_url)

    tr = ""

    # check if the request was successful
    if response.status_code == 200:
        # parse the JSON response
        data = response.json()
        tr = data['responseData']['translatedText']
        if verbose:
         print(f"apertium translation: {data['responseData']['translatedText']}")
    else:
        print(f'failed to retrieve apertium data: {response.status_code}')

    return tr