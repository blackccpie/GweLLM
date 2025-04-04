# GweLLM

[Version française](README-fr.md)

Hint: "Gwell" = "Better" in Breton

![image](gwenadu.png)

## Fine-Tuning a Breton speaking Chat Model

Experiments on adapting a pretrained Conversational LLM to a new language, in this case Breton as I live in sunny Brittany :sunglasses::wink:

GweLLM initial motivation was to build open source lightweight langage models for Breton, allowing:
* Local deployment and execution (even on CPU only)
* Hassle-free use (no external API limitations)

Output models and datasets will be made available on my [HuggingFace repo 🤗](https://huggingface.co/amurienne).

This is a Work in Progress...

## :rocket: Approach

Let's breakdown the problem:
* The global idea is to fine-tune an existing multi-lingual LLM, ideally one that saw some Breton during its tokenizer/model pre-training.
* To proceed to such a fine-tuning, we need a Breton instruction dataset, which doesn't seem to exist out-of-the-box.
* We can start from a french (or english) instruction dataset, and translate it to Breton.
* Finally with that dataset I can fine-tune the foundation LLM of my choice. 

Here is a shortlist of the challenges I first identified:
* Finding a good (and free) French -> Breton translation tool:
  * APIs?
    * [Apertium](https://www.apertium.org/) seems like a very interesting project, but the translation pair involving Breton is only the [br->fr](https://www.apertium.org/index.fra.html#?dir=bre-fra&q=Demat) one, the way back is not available yet :confused:
  * LLMs?
    * Google's [T5](https://huggingface.co/google-t5/t5-small)
      * Mutlilingual, but no Breton in training corpus :-1:
    * Meta's [M2M100_418M](https://huggingface.co/facebook/m2m100_418M)
      * Multilingual, Breton included in training corpus :+1:
      * Raw Breton performance is not good, will need fine-tuning!
* Finding a Breton instruction dataset:
  * Not found yet, will have to build one myself :muscle:

So this project has 3 "by-products":
* A French -> Breton Translation Model called **Gallek** (meaning "French" in Breton)
* A Breton Instruction Dataset called **Goulenn** (meaning "Question" in Breton)
* A Breton Conversational LLM called **GweLLM** ("Gwell" meaning "Good" in Breton)

All code is mainly based on HuggingFace's [Transformers](https://huggingface.co/docs/transformers/index) library.

## ⚙️ Building the _Gallek_ fr->br translation model

For now:
* Based on the [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) base model
* Fine-tuned on the [Bretagne/ofis_publik_br-fr](https://huggingface.co/datasets/Bretagne/ofis_publik_br-fr), [Bretagne/OpenSubtitles_br_fr](https://huggingface.co/datasets/Bretagne/OpenSubtitles_br_fr) & [Bretagne/Autogramm_Breton_translation](https://huggingface.co/datasets/Bretagne/Autogramm_Breton_translation) datasets
* Monodirectionally fr->br fine-tuned
* Reaches an honorable BLEU score of 40 on a 20% train/test split of the dataset

What's inside the `gallek` subdirectory:
* `train_translation_model.py` : used to fine-tune m2m100 model on the aforementionned datasets, with BLEU score evaluation at the end of training
* `evaluate_translation_model.py` : used to compute BLEU score on a finetuned model (no training code)
* `test_translation_model.py` : used to test the fine-tuned _gallek_ model on single input french text (also includes Apertium reverse translation)

:tv: Demos:
* `apps/gallek_breton_translator_gradio.py` : Gradio UI used to evaluate the _gallek_ model (uses _kellag_ for reverse translation for now...)
  * try it online :arrow_right: [Gallek Hugging Face Space](https://huggingface.co/spaces/amurienne/Gallek)
* `apps/translated_breton_chat_radio.py` : Gradio UI used to evaluate the _gallek_ model used as a translator coupled with a chat model (also uses _kellag_ for reverse translation for now...)
  * try it online :arrow_right: [BreizhBot HuggingFace Space](https://huggingface.co/spaces/amurienne/BreizhBot)

TODOs:
- [x] Add new datasets in training corpus (initial one was *ofis_publik*)
- [x] Reach a high quality 50 BLEU score
- [x] Train a separate br->fr backward translation model, waiting for the bidirectional one... it is called _Kellag_ :sweat_smile:
- [ ] Add some gguf conversion/quantization scripts using llama.cpp, _**spoiler alert : m2m100 seems unsupported**_ :scream:
- [ ] Train bidirectional version

## ⚙️ Building the _Goulenn_ Breton Instruct Dataset

For now:
* Based on the original [jpacifico/French-Alpaca-dataset-Instruct-110K](https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-110K?row=9), thanks to the work of Jonathan Pacifico.
* Translated to Breton using the _Gallek_ model

What's inside the `goulenn` subdirectory:
* `dataset_translation.py` : used to batch translate the original _French Alpaca_ instructions dataset into Breton
* `convert_dataset.py` : used to convert the `arrow` formated translated dataset to `json` and `parquet`
* `concatenate_datasets.py` : used to concatenate two `arrow` datasets, in case translation has been fragmented

TODOs:
- [x] Translate 50k samples (available on HF🤗 [here](https://huggingface.co/datasets/amurienne/Goulenn-Alpaca-Instruct-50k))
- [x] Translate the whole 110k (available on HF🤗 [here](https://huggingface.co/datasets/amurienne/Goulenn-Alpaca-Instruct-110k))
- [ ] Generate new instruction data using a ["Magpie"](https://magpie-align.github.io) like synthesis approach (WIP in `goulenn/magpie_instruct_dataset_generation.py`)

## :wrench: Fine-Tuning GweLLM

For now:
* Based on the [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) base model (seems to already know a bit of Breton)
* Trained on _Gallek_ 50k

What's inside the `gwellm` subdirectory:
* `train_model_instruct.py` : used to fine-tune the Breton speaking instruct model
* `test_model_instruct` : used to test the fine-tuned model (unmerged adapter)
* `merge_adapter.py` : used to merge the fine-tuned adapter model to the base model 
* `test_model_instruct_gradio.py` : used to test the quantized gguf model using a gradio chat UI

TODOs:
- [ ] Release an initial beta version
- [ ] Distribute as [LLamafile](https://github.com/Mozilla-Ocho/llamafile)
- [ ] Hybrid Fine-Tuning (start with a pretraining with a raw breton text corpus)

TODO FT Strategy
[Instruction Pre-Training: Language Models are Supervised Multitask Learners]

## :computer: Using GweLLM

### Import in GPT4All

TODO

## Additional Resources

### Finding Breton Datasets

Here are the few resources I found after initial googling:
* [Texts corpus at the French public office for Breton language](https://niverel.brezhoneg.bzh/fr/corpus/text)
* [The "Bretagne" organization on Hugging Face](https://huggingface.co/Bretagne) :thumbsup:

### Publications

* Soon after releasing the first _Gallek_ translator model, I stumbled upon this french paper describing the same m2m100 Breton finetuning approach: [_Loïc Grobol, Mélanie Jouitteau. ARBRES Kenstur: a Breton-French Parallel Corpus Rooted in Field Linguistics. LREC-COLING 2024 - The 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation, ELRA Language Resources Association Language Resources Association; International Committee on Computational Linguistics, May 2024, Torino, Italy. ffhal-04551941_](https://hal.science/hal-04551941/document) 

### Misc

* [Free online translation tool](https://niverel.brezhoneg.bzh/fr/troer/) :thumbsup:
* [Reddit thread about Breton LLM](https://www.reddit.com/r/Bretagne/comments/1d7389i/modèle_génératif_llm_langue_bretonne)

## :dizzy_face: Troubleshooting

### Installing `llama-cpp-python`

Installing `llama-cpp-python` can be a bit tricky, as I really struggled to install it on WSL2 (Ubuntu 22.04):
* The classic `pip install llama-cpp-python` systematically failed as described in [this issue](https://github.com/abetlen/llama-cpp-python/issues/1876)
* The documented way of installing a prebuilt cpu-only wheel `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` also failed
* I finally downloaded the [`llama_cpp_python-0.3.2-cp310-cp310-linux_x86_64.whl`](https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.2/llama_cpp_python-0.3.2-cp310-cp310-linux_x86_64.whl) package from the wheel repository and installed it manually with `pip install llama_cpp_python-0.3.2-cp310-cp310-linux_x86_64.whl`
* As I encountered issues related to `libc.musl` dependency I had to use [this workaround](https://github.com/abetlen/llama-cpp-python/issues/1628#issuecomment-2254571128)