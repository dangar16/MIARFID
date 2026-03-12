# Tecnologías del Lenguaje Humano (TLH) - Technologies of Human Language
This repository contains the materials and resources for the course "Tecnologías del Lenguaje Humano" (Technologies of Human Language).

# Proyecto
The project for this I implemented a encoder architechture based on the transformer architecture, which is a state-of-the-art model for natural language processing tasks. The model is trained using three books from the first Trilogy of Mistborn by Brandon Sanderson. The model is trained to predict the next word in a sequence of text, which allows it to learn the structure and patterns of the language used in the books. The model is implemented using PyTorch and trained on a GPU for faster training times.
The idea behind this project was to learn how to implement a transformer model from scratch and to understand the inner workings of the model. The project also allowed me to experiment differents types of architectures such as Pre-LN and Post-LN, and to compare their performance on the task of language modeling.

In order to evaluate the performance of the model, I extracted chunks of text to generate the embeddings of the sequences and then for different queries I calculated the cosine similarity between the query and the chunks of text to find the most similar ones. This allowed me to see how well the model was able to capture the meaning and context of the text.

Now I will explain the differents folders and files in the repository:
- BPE: Implementation of the encoder using Byte Pair Encoding (BPE) as the tokenization method. 
- CLS: Added CLS token to see how well it contained the information of the sequence.
- Encoder Bert: It uses the Masking Language Modeling (MLM) objective to train the model as in Bert paper.
- PostNorm: Implementation of the encoder using Post-LN architecture.
- PostNorm CLS: Implementation of the encoder using Post-LN architecture and adding CLS token.
- PreNorm: Implementation of the encoder using Pre-LN architecture.
- PreNorm CLS: Implementation of the encoder using Pre-LN architecture and adding CLS token.
- Stemmer: Implementation of the encoder using a stemmer as the tokenization method.

# TLH Whisper
Set of exercises to understand the inner workings of the Whisper model, which is a state-of-the-art model for automatic speech recognition (ASR) tasks. The exercises include:
- Exercise 1: Create the tokenizer for a dataset of audio files and their corresponding transcriptions. Implement whisper and evaluate the results.
- Exercise 2: Implement another encoder but taking into account that the task is to translate from one language to another.
- Exercise 3: Implement a new tokenizer using data that contained a text and a function definition from python. The idea is to train the model to predict the function definition given the text.
- Additional exercises: Implement KV caching to speed up the inference time of the model and add different methods to do the inference such as top-k sampling and sampling.