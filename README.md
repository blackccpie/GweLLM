# GweLLM

[Version française](README-fr.md)

Hint: "Gwell" = "Better" in Breton

![image](gwenadu.png)

## Fine-Tuning a Breton speaking Chat Model

Experiments on adapting a pretrained Conversational LLM to a new language, in this case Breton as I live in sunny Brittany :sunglasses::wink:

This is a Work in Progress...

### Approach

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
    * Meta's [M2M100](https://huggingface.co/facebook/m2m100_418M)
      * Multilingual, Breton included in training corpus :+1:
      * Raw Breton performance is not good, will need fine-tuning!
* Finding a Breton instruction dataset:
  * Not found yet, will have to buid one myself :muscle:

So this project has 3 "by-products":
* A French -> Breton Translation Model called **Gallek** (meaning "French" in Breton)
* A Breton Instruction Dataset called **Goulenn** (meaning "Question" in Breton)
* A Breton Conversational LLM called **GweLLM** ("Gwell" meaning "Good" in Breton)

All code is mainly based on HuggingFace's [Transformers](https://huggingface.co/docs/transformers/index) library.

### Building the _Gallek_ fr->br translation model

For now:
* Based on the [facebook/m2m100_418M](https://huggingface.co/facebook/m2m100_418M) base model
* Fine-tuned on the [Bretagne/ofis_publik_br-fr](https://huggingface.co/datasets/Bretagne/ofis_publik_br-fr) & [Bretagne/OpenSubtitles_br_fr](https://huggingface.co/datasets/Bretagne/OpenSubtitles_br_fr) datasets
* Monodirectionally fr->br fine-tuned
* Reaches an honorable BLEU score of 40 on a 20% train/test split of the dataset

What's inside the `gallek` subdirectory:
* `train_translation_model.py` : used to fine-tune m2m100 model on the aforementionned datasets, with BLEU score evaluation at the end of training
* `test_translation_model.py` : used to test the fine-tuned _gallek_ model on single input french text (also includes Apertium reverse translation)
* `test_translation_mode_gradui` : used to test the fine-tuned _gallek_ model using a Gradio UI

TODOs:
- [x] Add new datasets in training corpus (initial one was *ofis_publik*)
- [ ] Add some gguf conversion/quantization scripts using llama.cpp, _**spoiler alert : m2m100 seems unsupported**_ :scream:
- [ ] Reach a high quality 50 BLEU score
- [ ] Train bidirectional version

### Building the _Goulenn_ Breton Instruct Dataset

For now:
* Based on the original [jpacifico/French-Alpaca-dataset-Instruct-110K](https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-110K?row=9), thanks to the work of Jonathan Pacifico.
* Translated to Breton using the _Gallek_ model

What's inside the `goulenn` subdirectory:
* `dataset_translation.py` : used to batch translate the original _French Alpaca_ instructions dataset into Breton

TODOs:
- [x] Translate 50k samples
- [ ] Translate the whole 110k

### Fine-Tuning GweLLM

For now:
* Based on the [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) base model (seems to already know a bit of Breton)

TODOs:
- [x] TITI
- [ ] Distribute as [LLamafile](https://github.com/Mozilla-Ocho/llamafile)
- [ ] Hybrid Fine-Tuning

TODO FT Strategy
[Instruction Pre-Training: Language Models are Supervised Multitask Learners]

TODO scripts

### Using GweLLM

#### Import in GPT4All

TODO

### Additional Resources

#### Finding Breton Datasets

Here are the few resources I found after initial googling:
* [Texts corpus at the French public office for Breton language](https://niverel.brezhoneg.bzh/fr/corpus/text)
* [The "Bretagne" organization on Hugging Face](https://huggingface.co/Bretagne) :thumbsup:

#### Misc

* [Free online translation tool](https://niverel.brezhoneg.bzh/fr/troer/) :thumbsup:
* [Reddit thread about Breton LLM](https://www.reddit.com/r/Bretagne/comments/1d7389i/modèle_génératif_llm_langue_bretonne)

