# %% [markdown]
# # Prompting

# %% [markdown]
# In this notebook, we are going to use for fine-tuning a dataset set that is already available in the [Datasets repository](https://huggingface.co/datasets) from Hugging Face. However, the [Datasets library](https://huggingface.co/docs/datasets) makes easy to access and load datasets. For example, you can easily load your own dataset following [this tutorial](https://huggingface.co/docs/datasets/loading#local-and-remote-files).
# 
# More precisely, we are going to explain how to perform In-Context Learning with the [Llama2 model](https://huggingface.co/docs/transformers/model_doc/llama2) on the [Europarl-ST dataset](https://huggingface.co/datasets/tj-solergibert/Europarl-ST), but only that [dataset of Europarl-ST focused on the text data for MT from English](https://huggingface.co/datasets/tj-solergibert/Europarl-ST-processed-mt-en).

# %%
from datasets import load_dataset, DatasetDict

# Cargar el dataset original
raw_datasets_original = load_dataset("Helsinki-NLP/opus_elhuyar")

# --- Procesamiento para cambiar la estructura y dividir ---

# 1. Definir la función para reestructurar cada ejemplo (cuando batched=True, recibe listas)
def restructure_examples(examples):
    # examples['translation'] es una LISTA de diccionarios
    # Por ejemplo: [{'es': 'Hola', 'eu': 'Kaixo'}, {'es': 'Mundo', 'eu': 'Mundua'}, ...]
    
    # Extraemos las listas de español y euskera iterando sobre la lista de traducciones
    es_texts = []
    eu_texts = []
    for translation_dict in examples['translation']:
        es_texts.append(translation_dict['es'])
        eu_texts.append(translation_dict['eu'])

    # Creamos el nuevo diccionario con las columnas deseadas
    return {
        'source_text': es_texts,
        'dest_text': eu_texts,
    }

# 2. Aplicar la reestructuración al dataset original completo
# Usamos batched=True para procesar listas de ejemplos de una sola vez, lo cual es más eficiente
restructured_dataset = raw_datasets_original.map(
    restructure_examples,
    batched=True,
    # Conservamos solo las columnas nuevas
    remove_columns=raw_datasets_original['train'].column_names
)

# 3. Dividir el dataset 'train' reestructurado en train, validation y test
train_test_split = restructured_dataset['train'].train_test_split(test_size=0.2, seed=42)
test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)

# 4. Crear el DatasetDict final con los tres splits
raw_datasets = DatasetDict({
    'train': train_test_split['train'],
    'valid': test_val_split['train'],
    'test': test_val_split['test']
})

# 5. Imprimir la estructura final
print(raw_datasets)

print("\nCaracterísticas del dataset de entrenamiento:")
print(raw_datasets["train"].features)

print("\nPrimeros 2 ejemplos del dataset de entrenamiento:")
print(raw_datasets["train"].select(range(2)))


# %% [markdown]
# As shown, the Europarl-ST already comes with a pre-defined partition on the three conventional sets: training, validation and test. Each set is a dictionary with a list of source sentences (source_text), target sentences (dest_text) and the target language (dest_lang).
# 
# Let's take a closer look at the features of the training set:

# %%
raw_datasets["train"].features

# %% [markdown]
# As you can see, the possible target languages are German, English, Spanish, French, Italian, Dutch, Polish, Portuguese and Romanian.
# 
# Let us take a look at the translations of the first two English sentences:

# %%
raw_datasets["train"][:14]["source_text"]

# %%
raw_datasets["train"][:14]["dest_text"]

# %% [markdown]
# As shown, each English sentence is repeated for each of the seven target languages (0: 'de', 2: 'es', 3: 'fr', 4: 'it', 5: 'nl', 6: 'pl', 7: 'pt').
# 
# The Llama2 model is a pretrained Large Language Model (LLM) ready to tackle several NLP tasks, being one of the them the translation from English into Spanish. Let us filter the Europarl-ST only for English into Spanish using a simple [lambda function](https://realpython.com/python-lambda/) with the [Dataset.filter() function](https://huggingface.co/docs/datasets/v2.9.0/en/package_reference/main_classes#datasets.Dataset.filter).

# %% [markdown]
# More precisely, we are going to be using the Llama-2 checkpoint [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) to run our experiments for which you need to accept the LLAMA 2 COMMUNITY LICENSE AGREEMENT. Processing your request may take some time, so please do it in advance.

# %% [markdown]
# Logging in HuggingFace to be granted access to Llama2 with 7B parameters:

# %%
from huggingface_hub import login

login(token="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")

# %% [markdown]
# We can apply the tokenizer function to any dataset taking advantage that Hugging Face Datasets are [Apache Arrow](https://arrow.apache.org) files stored on the disk, so you only keep the samples you ask for loaded in memory.
# 
# To keep the data as a dataset, we will use the [Dataset.map() function](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset.map). This also allows us some extra flexibility, if we need more preprocessing done than just tokenization. The map() method works by applying a function on each element of the dataset.
# 
# In our case, each sample pair is going to be preprocessed according to the needs of the model that is to be prompted. In the case of Llama2, it is recommended to explicitly state a task prompt for each source sentence:

# %%
from transformers import AutoTokenizer

max_tok_length = 16
checkpoint = "HiTZ/Latxa-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    token=True,
    padding=True,
    pad_to_multiple_of=8,
    truncation=True,
    max_length=max_tok_length,
    padding_side='left',
    )
tokenizer.pad_token = tokenizer.eos_token

# %%
def preprocess_function(sample):
    model_inputs = tokenizer(
        sample["source_text"], 
        text_target = sample["dest_text"],
        )
    return model_inputs

# %% [markdown]
# The way the Datasets library applies this processing is by adding new fields to the datasets, one for each key in the dictionary returned by the tokenize function, that is, *input_ids*, *attention_mask* and *labels*. We can check what the preprocess_function is doing with a small sample

# %%
sample = raw_datasets["train"].select(range(2))
model_input = preprocess_function({
    "source_text": list(sample["source_text"]),
    "dest_text": list(sample["dest_text"]),
})
print(model_input)

# %%
for sample in model_input['input_ids']:
    print(tokenizer.convert_ids_to_tokens(sample))

# %% [markdown]
# We can recover the source text by applying [batch_decode](https://huggingface.co/docs/transformers/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_decode) of the tokenizer 

# %%
tokenizer.batch_decode(model_input['input_ids'])

# %% [markdown]
# Now, we can apply the preprocess_function to the raw datasets (training, validation and test):

# %%
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# %% [markdown]
# We are going to filter the tokenized datasets by maximum number of tokens in source and target language:

# %%
tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) <= max_tok_length and len(x["labels"]) <= max_tok_length , desc=f"Discarding source and target sentences with more than {max_tok_length} tokens")

# %% [markdown]
# We can take a quick look at the length histogram in the source language:

# %%
dic = {}
for sample in tokenized_datasets['train']:
    sample_length = len(sample['input_ids'])
    if sample_length not in dic:
        dic[sample_length] = 1
    else:
        dic[sample_length] += 1 

for i in range(1,max_tok_length+1):
    if i in dic:
        print(f"{i:>2} {dic[i]:>3}")

# %% [markdown]
# Checking a sample after filtering by maximum number of tokens:

# %%
for sample in tokenized_datasets['train'].select(range(5)):
    print(sample['input_ids'])
    print(sample['attention_mask'])
    print(sample['labels'])

# %%
src = "es"
tgt = "eu"
task_prefix = f"Translate from {src} to {tgt}:\n"
num_shots = 1
shots = ""
s = ""

prefix_tok_len = len(tokenizer.encode(f"{task_prefix}{shots}{src}: {s} = {tgt}: "))
shot_tok_len   = len(tokenizer.encode(f"{src}: {s} = {tgt}: {s}\n"))
max_tok_len = prefix_tok_len
max_tok_len += num_shots * (shot_tok_len + 2 * max_tok_length) 
max_tok_len += max_tok_length

random_seed = 13
sample = tokenized_datasets['train'].shuffle(seed=random_seed).select(range(num_shots))
for s in sample: shots += f"{src}: {s['source_text']} = {tgt}: {s['dest_text']}\n" 

def preprocess4test_function(sample):
    inputs = [f"{task_prefix}{shots}{src}: {s} = {tgt}: " for s in sample["source_text"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_tok_len, 
        truncation=True, 
        return_tensors="pt", 
        padding=True)
    return model_inputs

# %% [markdown]
# The way the Datasets library applies this processing is by adding new fields to the datasets, one for each key in the dictionary returned by the tokenize function, that is, *input_ids*, *attention_mask* and *labels*:

# %%
sample = tokenized_datasets['test'].select(range(5))
model_input = preprocess4test_function(sample)
print(model_input)
print(tokenizer.batch_decode(model_input['input_ids']))

# %%
preprocessed_test_dataset = tokenized_datasets['test'].map(preprocess4test_function, batched=True)

# %%
for sample in preprocessed_test_dataset.select(range(5)):
    print(sample['input_ids'])
    print(sample['attention_mask'])
    print(sample['labels'])

# %% [markdown]
# bitsandbytes is a quantization library with a Transformers integration. With this integration, you can quantize a model to 8 or 4-bits and enable many other options by configuring the BitsAndBytesConfig class. For example, you can:
# 
# <ul>
# <li>set load_in_4bit=True to quantize the model to 4-bits when you load it</li>
# <li>set bnb_4bit_quant_type="nf4" to use a special 4-bit data type for weights initialized from a normal distribution</li>
# <li>set bnb_4bit_use_double_quant=True to use a nested quantization scheme to quantize the already quantized weights</li>
# <li>set bnb_4bit_compute_dtype=torch.bfloat16 to use bfloat16 for faster computation</li>
# </ul>
# 

# %%
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# %% [markdown]
# Pass the quantization_config to the from_pretrained method.

# %%
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    token=True,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
)


# %% [markdown]
# # Inference

# %% [markdown]
# Loading default inference parameters for the model, so that additional parameters could be added and passed to the [generate function](https://huggingface.co/docs/transformers/main_classes/text_generation):

# %%
from transformers import GenerationConfig

generation_config = GenerationConfig.from_pretrained(
    checkpoint,
    )

print(generation_config)

# %% [markdown]
# As observed, the default search strategy for Llama-2 is Top-p with probability 0.9 and temperature 0.6 ($0<T<1$ amplifies output probability differences and makes output more deterministic). [The search strategy can be selected](https://huggingface.co/docs/transformers/en/generation_strategies) at inference time. 

# %% [markdown]
# First, the test set is divided in small batches to reduce GPU memory comsumption:

# %%
test_batch_size = 32
batch_tokenized_test = preprocessed_test_dataset.batch(test_batch_size)

# %%
number_of_batches = len(batch_tokenized_test["input_ids"])
output_sequences = []
for i in range(number_of_batches):
    # Pad sequences in the batch to the same length
    input_ids = batch_tokenized_test["input_ids"][i]
    attention_mask = batch_tokenized_test["attention_mask"][i]
    
    # Find max length in this batch
    max_len = max(len(seq) for seq in input_ids)
    
    # Pad each sequence
    padded_input_ids = []
    padded_attention_mask = []
    for seq, mask in zip(input_ids, attention_mask):
        padding_length = max_len - len(seq)
        # Pad on the left since padding_side='left'
        padded_seq = [tokenizer.pad_token_id] * padding_length + seq
        padded_mask = [0] * padding_length + mask
        padded_input_ids.append(padded_seq)
        padded_attention_mask.append(padded_mask)
    
    with torch.no_grad():
        output_batch = model.generate(
            generation_config=generation_config, 
            input_ids=torch.tensor(padded_input_ids).cuda(), 
            attention_mask=torch.tensor(padded_attention_mask).cuda(), 
            max_length = max_tok_len, 
            num_beams=1, 
            do_sample=False,)
    output_sequences.extend(output_batch)

# %% [markdown]
# ## Evaluation

# %% [markdown]
# The output of the model is automatically evaluated compared to the reference translations. To this purpose, we use the [Evaluate library](https://huggingface.co/docs/evaluate) which includes the definition of generic and task-specific metrics. In our case, we use the [BLEU metric](https://huggingface.co/spaces/evaluate-metric/bleu), or to be more precise, [sacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu).

# %%
from evaluate import load

metric = load("sacrebleu")

# %% [markdown]
# The example below performs a basic post-processing to decode the predictions and extract the translation:

# %%
import re
from evaluate import load

def compute_metrics(sample, output_sequences):
    inputs = [f"{task_prefix}{shots}{src}: {s} = {tgt}: " for s in sample["source_text"]]
    preds = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    print(inputs)
    print(preds)
    for i, (input,pred) in enumerate(zip(inputs,preds)):
      pred = re.search(r'^.*\n',pred.removeprefix(input).lstrip())
      if pred is not None:
        preds[i] = pred.group()[:-1]
      else:
        preds[i] = ""
    print(sample["source_text"])
    print(sample["dest_text"])
    print(preds)
    result = metric.compute(predictions=preds, references=sample["dest_text"])
    result = {"bleu": result["score"]}

    #code under here
    comet_metric = load("comet")
    comet_result = comet_metric.compute(
        predictions=preds,
        references=sample["dest_text"],
        sources=sample["source_text"] # Usamos el texto fuente del sample original
    )
    result["comet"] = comet_result["mean_score"]

    chrf_metric = load("chrf")
    result_chrf = chrf_metric.compute(predictions=preds, references=sample["dest_text"])
    result["chrf"] = result_chrf["score"]

    return result

# %%
result = compute_metrics(preprocessed_test_dataset, output_sequences)
print(f'BLEU score: {result["bleu"]}')
print(f'COMET score: {result["comet"]}')
print(f'CHRF score: {result["chrf"]}')
