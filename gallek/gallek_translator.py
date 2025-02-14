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

class base_translator:
    """
    Base translator class using fine-tuned M2M100 models.
    """
    def __init__(self, checkpoint: str, src_lang: str, tgt_lang: str, chdir: str = './', max_length: int = 400, batch_size: int = 1):
        self._checkpoint = checkpoint
        self._tokenizer = AutoTokenizer.from_pretrained(chdir + self._checkpoint)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(chdir + self._checkpoint, device_map="auto")
        self._model.eval()
        self._model.config.use_cache = True
        self._translation_pipeline = pipeline("translation", model=self._model, tokenizer=self._tokenizer, 
                                             src_lang=src_lang, tgt_lang=tgt_lang, max_length=max_length, batch_size=batch_size)

    def _translate_batch(self, samples: str):
        with torch.no_grad():
            return self._translation_pipeline(samples)

    def _translate(self, text: str):
        with torch.no_grad():
            return self._translation_pipeline(text)[0]['translation_text']

class gallek(base_translator):
    """
    French to Breton translator.
    """
    def __init__(self, chdir: str = './', max_length: int = 400, batch_size: int = 1):
        super().__init__(checkpoint="gallek-m2m100-b40", src_lang='fr', tgt_lang='br', chdir=chdir, max_length=max_length, batch_size=batch_size)

    def translate_fr2br_batch(self, samples: str):
        return self._translate_batch(samples)

    def translate_fr2br(self, text: str):
        return self._translate(text)

class kellag(base_translator):
    """
    Breton to French translator.
    """
    def __init__(self, chdir: str = './', max_length: int = 400, batch_size: int = 1):
        super().__init__(checkpoint="kellag-m2m100-b51", src_lang='br', tgt_lang='fr', chdir=chdir, max_length=max_length, batch_size=batch_size)

    def translate_br2fr_batch(self, samples: str):
        return self._translate_batch(samples)

    def translate_br2fr(self, text: str):
        return self._translate(text)
