# %% [markdown]
# # Fine-tuning

# %% [markdown]
# Fine-tuning refers to the process in transfer learning in which the parameter values of a model trained on a large dataset are modified when the training process continues on a small dataset (see [Kevin Murphy's book](https://probml.github.io/pml-book/book1.html) Section 19.2 for further details). The main motivation is to adapt a pre-trained model trained on a large amount of data to tackle a specific task providing better performance that would be achieved training on the small task-specific dataset.

# %% [markdown]
# In this notebook, we are going to use for fine-tuning a dataset set that is already available in the [Datasets repository](https://huggingface.co/datasets) from Hugging Face. However, the [Datasets library](https://huggingface.co/docs/datasets) makes easy to access and load datasets. For example, you can easily load your own dataset following [this tutorial](https://huggingface.co/docs/datasets/loading#local-and-remote-files).
# 
# More precisely, we are going to explain how to fine-tune the [NLLB model](https://huggingface.co/docs/transformers/model_doc/nllb) on the [Europarl-ST dataset](https://huggingface.co/datasets/tj-solergibert/Europarl-ST), but only that [dataset of Europarl-ST focused on the text data for MT from English](https://huggingface.co/datasets/tj-solergibert/Europarl-ST-processed-mt-en).

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
# Now we load the pre-trained tokenizer for the NLLB model and apply it to the English-Spanish pair:

# %%
max_tok_length = 16

from transformers import AutoTokenizer

checkpoint = "facebook/nllb-200-distilled-600M"
# from flores200_codes import flores_codes
src_code = "esp_Latn"
tgt_code = "eus_Latn"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, 
    padding=True, 
    pad_to_multiple_of=8, 
    src_lang=src_code, 
    tgt_lang=tgt_code, 
    truncation=True, 
    max_length=max_tok_length,
    )

# %% [markdown]
# We can apply the tokenizer function to any dataset taking advantage that Hugging Face Datasets are [Apache Arrow](https://arrow.apache.org) files stored on the disk, so you only keep the samples you ask for loaded in memory.
# 
# To keep the data as a dataset, we will use the [Dataset.map() function](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset.map). This also allows us some extra flexibility, if we need more preprocessing done than just tokenization. The map() method works by applying a function on each element of the dataset.
# 
# In our case, each sample pair is going to be preprocessed according to the training needs of the model that is to be finetuned:

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
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    quantization_config=quantization_config
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
    task_type="SEQ_2_SEQ_LM",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# %% [markdown]
# Once LoRA and the quantization are setup, create a quantized PeftModel with the get_peft_model() function. It takes a quantized model and the LoraConfig containing the parameters for how to configure a model for training with LoRA.

# %%
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()

# %% [markdown]
# The function that is responsible for putting together samples inside a batch is called a collate function. It is an argument you can pass when you build a DataLoader, the default being a function that will just convert your samples to PyTorch tensors and concatenate them. This is not possible in our case since the inputs we have are not all of the same size. We have deliberately postponed the padding, to only apply it as necessary on each batch and avoid having over-long inputs with a lot of padding.
# 
# To do this in practice, we have to define a collate function that will apply the correct amount of padding to the items of the dataset we want to batch together. Fortunately, the Transformers library provides us with such a function via DataCollatorForSeq2Seq that takes a tokenizer when you instantiate it (to know which padding token to use, and whether the model expects padding to be on the left or on the right of the inputs), so we will also need to instantiate the model first to provide it to the collate function:

# %%
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=lora_model, 
    pad_to_multiple_of=8
    )

# %% [markdown]
# ## Evaluation

# %% [markdown]
# The last thing to define for our Seq2SeqTrainer is how to compute the metrics to evaluate the predictions of our model with respect to references. To this purpose, we use the [Evaluate library](https://huggingface.co/docs/evaluate) which includes the definition of generic and task-specific metrics. In our case, we use the [BLEU metric](https://huggingface.co/spaces/evaluate-metric/bleu), or to be more precise, [sacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu). You can see a simple example of usage below:
# 
# :

# %%
from evaluate import load

metric = load("sacrebleu")

# %% [markdown]
# We need to define a function compute_metrics to compute BLEU scores at each epoch. The example below performs a basic post-processing to decode the predictions into texts:

# %%
import numpy as np
from evaluate import load

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def compute_metrics(eval_preds, raw_test_dataset=None):
    preds, labels = eval_preds

    # Convert to lists if coming from a datasets.Column
    if not isinstance(labels, list):
        labels = list(labels)
        
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace negative ids in the labels as we can't decode them.
    labels = [
        [tokenizer.pad_token_id if j < 0 else j for j in label]
        for label in labels
    ]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels_for_bleu = postprocess_text(decoded_preds, decoded_labels)

    # Calculate BLEU
    bleu_metric = load("sacrebleu")
    # Sacrebleu expects references as a list of lists of strings
    references_bleu = [[label] for label in decoded_labels_for_bleu]
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=references_bleu)
    bleu_score = round(bleu_result["score"], 4)

    # Initialize result dictionary with BLEU
    result = {"bleu": bleu_score}

    # Calculate COMET if the original dataset is provided
    if raw_test_dataset is not None:
        # Get source texts from the original dataset
        # Ensure the order matches the predictions/labels
        source_texts = raw_test_dataset["source_text"][:len(decoded_preds)]
        reference_texts_for_comet = raw_test_dataset["dest_text"][:len(decoded_preds)] # Use original target text

        comet_metric = load("comet")
        comet_result = comet_metric.compute(
            predictions=decoded_preds,
            references=reference_texts_for_comet, # Use original target text
            sources=source_texts
        )
        comet_score = round(comet_result["mean_score"], 4)
        result["comet"] = comet_score
    else:
        # If raw_test_dataset is not provided, COMET cannot be calculated
        print("Warning: raw_test_dataset not provided. COMET score will not be calculated.")
        result["comet"] = None

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items() if v is not None}
    return result

# %% [markdown]
# ## Training

# %% [markdown]
# The first step before we can define our [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer#trainer) is to define a [Seq2SeqTrainingArguments class](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments) that will contain all the hyperparameters the Trainer will use for training and evaluation. The only compulsory argument you have to provide is a directory where the trained model will be saved, as well as the checkpoints along the way. For all the rest, you can set them depending on the recommendations from the model developers:

# %%
from transformers import Seq2SeqTrainingArguments

batch_size = 32
model_name = checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-es-to-eu",
    eval_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=2,
    predict_with_generate=True,
)

# %% [markdown]
# Once we have our model, we can define a Trainer by passing it all the objects constructed up to now — the model, the training_args, the training and validation datasets, the tokenizer, the data collator and the compute_metrics function:

# %%
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    lora_model,
    args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['valid'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# %% [markdown]
# To fine-tune the model on our dataset, we just have to call the [train() function](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Trainer.train) of our Trainer:

# %%
trainer.train()

# %% [markdown]
# ## Inference

# %% [markdown]
# At inference time, it is recommended to use [generate()](https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/text_generation#transformers.GenerationMixin.generate). This method takes care of encoding the input and feeding the encoded hidden states via cross-attention layers to the decoder and auto-regressively generates the decoder output. Check out [this blog post](https://huggingface.co/blog/how-to-generate) to know all the details about generating text with Transformers. There’s also [this blog post](https://huggingface.co/blog/encoder-decoder#encoder-decoder) which explains how generation works in general in encoder-decoder models.

# %% [markdown]
# Let us first load the default inference parameters of NLLB: 

# %%
from transformers import GenerationConfig

generation_config = GenerationConfig.from_pretrained(
    checkpoint,
)

print(generation_config)

# %% [markdown]
# We prepare the test set in batches to be translated:

# %%
test_batch_size = 32
batch_tokenized_test = tokenized_datasets['test'].batch(test_batch_size)

# %% [markdown]
# Processing in batches to add padding and converting to tensors, then perform inference with num_beams = 1 and do_sample = False, that is, greedy search.

# %%
number_of_batches = len(batch_tokenized_test["source_text"])
output_sequences = []
for i in range(number_of_batches):
    inputs = tokenizer(
        batch_tokenized_test["source_text"][i], 
        max_length=max_tok_length, 
        truncation=True, 
        return_tensors="pt", 
        padding=True)
    with torch.no_grad():
        output_batch = lora_model.generate(
            generation_config=generation_config, 
            input_ids=inputs["input_ids"].cuda(), 
            attention_mask=inputs["attention_mask"].cuda(), 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code), 
            max_length = max_tok_length, 
            num_beams=1, 
            do_sample=False,)
    output_sequences.extend(output_batch.cpu())

# %%
result = compute_metrics((output_sequences,tokenized_datasets['test']["labels"]), raw_test_dataset=raw_datasets['test'])
print(f'BLEU score: {result["bleu"]}')
print(f'COMET score: {result["comet"]}')