# %% [markdown]
# # Fine-tuning

# %% [markdown]
# Fine-tuning refers to the process in transfer learning in which the parameter values of a model trained on a large dataset are modified when the training process continues on a small dataset (see [Kevin Murphy's book](https://probml.github.io/pml-book/book1.html) Section 19.2 for further details). The main motivation is to adapt a pre-trained model trained on a large amount of data to tackle a specific task providing better performance that would be achieved training on the small task-specific dataset.

# %% [markdown]
# In this notebook, we are going to use for fine-tuning a dataset set that is already available in the [Datasets repository](https://huggingface.co/datasets) from Hugging Face. However, the [Datasets library](https://huggingface.co/docs/datasets) makes easy to access and load datasets. For example, you can easily load your own dataset following [this tutorial](https://huggingface.co/docs/datasets/loading#local-and-remote-files).
#
# More precisely, we are going to explain how to fine-tune the [Llama2 model](https://huggingface.co/docs/transformers/model_doc/llama2) on the [Europarl-ST dataset](https://huggingface.co/datasets/tj-solergibert/Europarl-ST), but only that [dataset of Europarl-ST focused on the text data for MT from English](https://huggingface.co/datasets/tj-solergibert/Europarl-ST-processed-mt-en).

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
# In our case, each sample pair is going to be preprocessed according to the needs of the model that is to be fine-tuned. In the case of Llama2, it is recommended to explicitly state a task prompt for each source sentence:

# %%
from transformers import AutoTokenizer

max_tok_length = 16
checkpoint = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, use_auth_token=True,
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
import torch

src = "es"
tgt = "eu"
task_prefix = f"Translate from {src} to {tgt}:\n"
s = ""

prefix_tok_len = len(tokenizer.encode(f"{task_prefix}{src}: {s} = {tgt}: "))
max_tok_len = prefix_tok_len
# Adding 2 for new line in target sentence and eos_token_id token
max_tok_len += 2 * max_tok_length + 2


def preprocess4training_function(sample):

    sample_size = len(sample["source_text"])

    # Creating the prompt with the task description for each source sentence
    inputs  = [f"{task_prefix}{src}: {s} = {tgt}: " for s in sample["source_text"]]

    # Appending new line after each sample in the batch
    targets = [f"{s}\n" for s in sample["dest_text"]]

    # Applying the Llama2 tokenizer to the inputs and targets
    # to obtain "input_ids" (token_ids) and "attention mask"
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)

    # Each input is appended with its target
    # Each target is prepended with as many special token id (-100) as the original input length
    # Both input and target (label) has the same max_tok_len
    # Attention mask is all 1s
    for i in range(sample_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    # Each input is applied left padding up to max_tok_len
    # Attention mask is 0 for padding
    # Each target (label) is left filled with special token id (-100)
    # Finally inputs, attention_mask and targets (labels) are truncated to max_tok_len
    for i in range(sample_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_tok_len - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_tok_len - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_tok_len - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_tok_len])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_tok_len])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_tok_len])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# %% [markdown]
# We can check what the preprocess4training_function is doing:

# %%
sample = tokenized_datasets['train'].select(range(2))
model_input = preprocess4training_function(sample)
print(model_input)
print(tokenizer.batch_decode(model_input.input_ids))

# %% [markdown]
# We need to replace -100 by 0 to apply batch_decode:

# %%
import numpy as np
for i in range(len(model_input['labels'])):
  print(tokenizer.batch_decode([np.where(model_input['labels'][i] < 0, tokenizer.pad_token_id, model_input['labels'][i])]))

# %% [markdown]
# In the case of the test set, we just preprocess the inputs (source sentences)

# %%
def preprocess4test_function(sample):
    inputs = [f"{task_prefix}{src}: {s} = {tgt}: " for s in sample["source_text"]]
    model_inputs = tokenizer(inputs,padding=True,)
    return model_inputs

# %% [markdown]
# We can check what the preprocess4test_function is doing:

# %%
sample = tokenized_datasets['train'].select(range(2))
model_input = preprocess4test_function(sample)
print(model_input)
print(tokenizer.batch_decode(model_input.input_ids))

# %% [markdown]
# Preprocessing train and dev sets:

# %%
preprocessed_train_dataset = tokenized_datasets['train'].map(preprocess4training_function, batched=True)
preprocessed_dev_dataset = tokenized_datasets['valid'].map(preprocess4training_function, batched=True)

# %%
for sample in preprocessed_train_dataset.select(range(5)):
    print(sample['input_ids'])
    print(sample['attention_mask'])
    print(sample['labels'])

# %% [markdown]
# Preprocessing test set:

# %%
preprocessed_test_dataset = tokenized_datasets['test'].map(preprocess4test_function, batched=True)

# %%
for sample in preprocessed_test_dataset.select(range(5)):
    print(sample['input_ids'])
    print(sample['attention_mask'])
    print(sample['labels'])

# %% [markdown]
# [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index) is a quantization library with a Transformers integration. With this integration, you can quantize a model to 8 or 4-bits and enable many other options by configuring the BitsAndBytesConfig class. For example, you can:
#
# <ul>
# <li>set load_in_4bit=True to quantize the model to 4-bits when you load it</li>
# <li>set bnb_4bit_quant_type="nf4" to use a special 4-bit data type for weights initialized from a normal distribution</li>
# <li>set bnb_4bit_use_double_quant=True to use a nested quantization scheme to quantize the already quantized weights</li>
# <li>set bnb_4bit_compute_dtype=torch.bfloat16 to use bfloat16 for faster computation</li>
# </ul>
#

# %%
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
# Next, you should call the prepare_model_for_kbit_training() function to preprocess the quantized model for training.

# %%
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False, gradient_checkpointing_kwargs={'use_reentrant':False})

# %% [markdown]
# [LoRA (Low-Rank Adaptation of Large Language Models)](https://huggingface.co/docs/peft/task_guides/lora_based_methods) is a [parameter-efficient fine-tuning (PEFT)](https://huggingface.co/docs/peft/index) technique that significantly reduces the number of trainable parameters. It works by inserting a smaller number of new weights into the model and only these are trained. This makes training with LoRA much faster, memory-efficient, and produces smaller model weights (a few hundred MBs), which are easier to store and share.

# %% [markdown]
# Each PEFT method is defined by a PeftConfig class that stores all the important parameters for building a PeftModel. For example, to train with LoRA, load and create a LoraConfig class and specify the following parameters:
#
# <ul>
# <li>task_type: the task to train for (sequence-to-sequence language modeling in this case)</li>
# <li>r: the dimension of the low-rank matrices</li>
# <li>lora_alpha: the scaling factor for the low-rank matrices</li>
# <li>target_modules: determine what set of parameters are adapted</li>
# <li>lora_dropout: the dropout probability of the LoRA layers</li>
# </ul>

# %%
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"] # Mistral specific
)

# %% [markdown]
# Once LoRA and the quantization are setup, create a quantized PeftModel with the get_peft_model() function. It takes a quantized model and the LoraConfig containing the parameters for how to configure a model for training with LoRA.

# %%
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()

# %% [markdown]
# The function that is responsible for putting together samples inside a batch is called a collate function.

# %%
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

# %% [markdown]
# ## Training

# %% [markdown]
# The first step before we can define our [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) is to define a [TrainingArguments class](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) that will contain all the hyperparameters the Trainer will use for training and evaluation. The only compulsory argument you have to provide is a directory where the trained model will be saved, as well as the checkpoints along the way. For all the rest, you can set them depending on the recommendations from the model developers:

# %%
from transformers import TrainingArguments

batch_size = 4
gradient_accumulation_steps = 8
model_name = checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-es-to-eu-mistral",
    eval_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    warmup_steps=100,
    optim="adamw_bnb_8bit",
    prediction_loss_only=True,
    gradient_accumulation_steps = gradient_accumulation_steps,
    bf16=True,
    bf16_full_eval=True,
    group_by_length=True,
)

# %% [markdown]
# Once we have our model, we can define a Trainer by passing it all the objects constructed up to now — the model, the training_args, the training and validation datasets, the tokenizer and the data collator:

# %%
from transformers import Trainer

trainer = Trainer(
    lora_model,
    args,
    train_dataset=preprocessed_train_dataset,
    eval_dataset=preprocessed_dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# %% [markdown]
# To fine-tune the model on our dataset, we just have to call the [train() function](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Trainer.train) of our Trainer. However, the [wandb library](https://docs.wandb.ai/guides) is used and it requires to have a [wandb account and login](https://docs.wandb.ai/guides/integrations/huggingface/).

# %%
trainer.train()

# %% [markdown]
# ## Inference

# %% [markdown]
# At inference time, it is recommended to use [generate()](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate). This method takes care of encoding the input and auto-regressively generates the decoder output. Check out [this blog post](https://huggingface.co/blog/how-to-generate) to know all the details about generating text with Transformers.

# %% [markdown]
# Let us first load the default inference parameters of Llama-2:

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
test_batch_size = 4
batch_tokenized_test = preprocessed_test_dataset.batch(test_batch_size)

# %% [markdown]
# Batches are provided to the [generate()](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate) together with inference parameters to define the search strategy. In this case, num_beams = 1 and do_sample = False means greedy search.

# %%
number_of_batches = len(batch_tokenized_test["input_ids"])
output_sequences = []
for i in range(number_of_batches):
    with torch.no_grad():
        output_batch = lora_model.generate(
            generation_config=generation_config,
            input_ids=torch.tensor(batch_tokenized_test["input_ids"][i]).cuda(),
            attention_mask=torch.tensor(batch_tokenized_test["attention_mask"][i]).cuda(),
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

def compute_metrics(sample, output_sequences):
    inputs = [f"{task_prefix}{src}: {s} = {tgt}: "  for s in sample["source_text"]]
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

    comet_metric = load("comet")
    comet_result = comet_metric.compute(
        predictions=preds,
        references=sample["dest_text"],
        sources=sample["source_text"] # Usamos el texto fuente del sample original
    )
    result["comet"] = comet_result["mean_score"]
    return result

# %%
result = compute_metrics(preprocessed_test_dataset, output_sequences)
print(f'BLEU score: {result["bleu"]}')
print(f'COMET score: {result["comet"]}')
