# GweLLM

Hint: "Gwell" = "Better" in Breton

![image](gwenadu.png)

## Fine-Tuning a Breton speaking Chat Model

Experiments on adapting a pretrained Conversational LLM to a new language, in this case Breton as I live in sunny Brittany :sunglasses::wink:

Work in progress...

### Approach

Let's breakdown the problem:
* The global idea is to fine-tune an existing multi-lingual LLM, ideally one that saw some Breton during its tokenizer/model pre-training.
* To proceed to such a fine-tuning, I need a Breton instruction dataset, which doesn't seem to exist out-of-the-box.
* I can start from a french or english instruction dataset, and translate it to Breton.
* Finally I can fine-tune the foundation LLM of my choice. 

Here is a shortlist of the challenges I first identified:
* Finding a good (and free) French -> Breton translation tool:
  * APIs?
    * [Apertium](https://www.apertium.org/) seems like a very interesting project, but the translation pair involving Breton is only the [BR -> FR](https://www.apertium.org/index.fra.html#?dir=bre-fra&q=Demat) one, the way back is not available yet :confused:
  * LLMs?
    * Google's [T5](https://huggingface.co/google-t5/t5-small)
    * Meta's [M2M100](https://huggingface.co/facebook/m2m100_418M)
* Finding a Breton instruction dataset:
  * TODO

So this project has 3 by-products:
* A French -> Breton Translation Model called **Gallek** (meaning "French" in Breton)
* A Breton Instruction Dataset called **Goulenn** (meaning "Question" in Breton)
* A Breton Conversational LLM called **GweLLM** ("Gwell" meaning "Good" in Breton)

#### Misc Additional Resources

** [Free online translation tool](https://niverel.brezhoneg.bzh/fr/troer/) :thumbsup:
* [Reddit thread about Breton LLM](https://www.reddit.com/r/Bretagne/comments/1d7389i/modèle_génératif_llm_langue_bretonne)

#### Finding Breton Datasets

Here are the few resources I found after initial googling:
* [Texts corpus at the French public office for Breton language](https://niverel.brezhoneg.bzh/fr/corpus/text)
* [The "Bretagne" organization on Hugging Face](https://huggingface.co/Bretagne) :thumbsup:

### Building a FR -> BR translation model

Gallek

TODO

### Building a Breton Instruct Dataset

Goulenn

TODO

### Fine-Tuning GweLLM

TODO