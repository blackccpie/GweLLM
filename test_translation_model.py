# The MIT License

# Copyright (c) 2017-2017 Albert Murienne

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

from transformers import pipeline, AutoModelForSeq2SeqLM

text = "traduis de français en breton: j'apprends le breton à l'école."

#translator = pipeline("translation_fr_to_br", model="my_awesome_breton_model", device='cuda')
#print(translator(text, max_length=256))

model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_breton_model")
outputs = model.generate(   text,
                            max_new_tokens=40,
                            #return_dict = False   # this is needed to get a tensor as result
                        )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

