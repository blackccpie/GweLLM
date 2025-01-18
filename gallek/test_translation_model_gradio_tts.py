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

import gradio as gr
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from parler_tts import ParlerTTSForConditionalGeneration

modelcard = "gallek-m2m100-b40"

model = AutoModelForSeq2SeqLM.from_pretrained(modelcard)
tokenizer = AutoTokenizer.from_pretrained(modelcard)

device = 'cuda'
repository = "parler-tts/parler-tts-mini-multilingual-v1.1"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(repository).to(device)
tts_tokenizer = AutoTokenizer.from_pretrained(repository)
tts_description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)
tts_feature_extractor = AutoFeatureExtractor.from_pretrained(repository)

SAMPLE_RATE = tts_feature_extractor.sampling_rate

def synthethize(prompt):
    """
    Synthetizes audio speech from text
    """

    description = "Daniel's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

    input_ids = tts_description_tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tts_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    return (SAMPLE_RATE, audio_arr)

def translate(text):
    """
    Translate the text from source lang to target lang
    """
    translation_pipeline = pipeline("translation", model=model, tokenizer=tokenizer, src_lang='fr', tgt_lang='br', max_length=400, device="cuda")
    translation = translation_pipeline("traduis de français en breton: " + text)[0]['translation_text']
    return translation, synthethize(translation)

demo = gr.Interface(
    fn=translate,
    inputs=[
        gr.components.Textbox(label="French"),
    ],
    outputs=[
        gr.components.Textbox(label="Breton"),
        gr.Audio(label="Synthetized Audio", autoplay=True)
    ],
    cache_examples=False,
    title="Gallek French -> Breton Translation Demo",
    allow_flagging='never'
)

demo.launch()