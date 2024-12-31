# GweLLM

[Version anglaise](README.md)

Indice: "Gwell" = "Mieux" en Breton

![image](gwenadu.png)

## Fine-Tuning d'un modèle conversationnel Breton

Expériences sur l'adaptation d'un LLM conversationnel pré-entraîné à une nouvelle langue, en l'occurrence le Breton, vivant pour ma part dans la belle Bretagne ensoleillée :sunglasses::wink:

La motivation initiale derrière GweLLM est de construire des modèles de langage légers et open source pour le Breton, permettant :
* Un déploiement et une exécution en local (même sur CPU uniquement)
* Une utilisation sans contraintes (pas de limitations d'API externes)

Ceci est un travail en cours...

### Approche

Décomposons le problème :
* L'idée générale est d'affiner/fine-tuner un LLM multilingue existant, idéalement un LLM qui a déjà vu un peu de breton pendant le pré-entraînement de son tokenizer/modèle.
* Pour procéder à un tel ajustement, nous avons besoin d'un jeu de données d'instructions/réponses en breton, qui ne semble pas exister en l'état.
* Nous pouvons partir d'un jeu de données d'instructions/réponses existant en français (ou en anglais) et le traduire en breton.
* Enfin, avec cet ensemble de données, on peux affiner le LLM de base de notre choix. 

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

Ce projet a donc 3 "sous-produits" dérivés :
* Un modèle de traduction français -> breton appelé **Gallek** (signifiant « français » en breton)
* Un ensemble de données d'instructions/réponses en breton appelé **Goulenn** (signifiant « Question » en breton)
* Un LLM conversationnel breton appelé **GweLLM** (« Gwell » signifiant « Bon » en breton)

Tout le code est principalement basé sur la bibliothèque HuggingFace [Transformers](https://huggingface.co/docs/transformers/index).

EN COURS...