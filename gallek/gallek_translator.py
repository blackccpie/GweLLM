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

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

import torch

class gallek:
    """
    French to Breton translator based on fine-tuned M2M100 base model
    """

    __checkpoint_base = "facebook/m2m100_418M"
    __checkpoint = "gallek-m2m100-b40"

    def __init__(self, chdir: str='./', max_length: int=400, batch_size: int=1):
        self.__tokenizer = AutoTokenizer.from_pretrained(self.__checkpoint_base)
        self.__model = AutoModelForSeq2SeqLM.from_pretrained(chdir + self.__checkpoint, device_map="auto")
        self.__model.eval()
        self.__model.config.use_cache = True
        self.__translation_pipeline = pipeline("translation", model=self.__model, tokenizer=self.__tokenizer, src_lang='fr', tgt_lang='br', max_length=max_length, batch_size=batch_size)

    def translate_fr2br_batch(self, samples: str):
        with torch.no_grad():
            return self.__translation_pipeline(samples)

    def translate_fr2br(self, text: str):
        with torch.no_grad():
            return self.__translation_pipeline(text)[0]['translation_text']
