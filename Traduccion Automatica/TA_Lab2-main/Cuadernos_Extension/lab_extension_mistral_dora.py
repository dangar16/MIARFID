# ============================================================
# DATASET
# ============================================================
from datasets import load_dataset

DATA_PATH_TRAIN = "/home/alumno.upv.es/dnargar/U4_Speech_Translation/Lab/laboratorio/Cuadernos_4_4/L4.1_v5_large_validation.csv"

raw_datasets = load_dataset(
    "csv",
    data_files=DATA_PATH_TRAIN
)

raw_datasets = raw_datasets["train"].select_columns(
    ["hypothesis", "translation"]
)

raw_datasets = raw_datasets.train_test_split(
    test_size=0.1,
    shuffle=True,
    seed=42
)

print(raw_datasets)

# ============================================================
# TOKENIZER
# ============================================================
from transformers import AutoTokenizer

checkpoint = "mistralai/Mistral-7B-v0.1"
max_tok_length = 275

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    use_fast=True,
    padding=True,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def preprocess_function(batch):
    input_ids_list = []
    labels_list = []

    for src, tgt in zip(batch["hypothesis"], batch["translation"]):
        prompt = f"Translate to English:\n{src}\nAnswer:"
        full_text = prompt + " " + tgt

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=max_tok_length,
        )

        input_ids = tokenized["input_ids"]

        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_tok_length,
        )["input_ids"]

        labels = input_ids.copy()
        labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
    }



tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# ============================================================
# MODEL (4-bit + DoRA)
# ============================================================
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoModelForCausalLM

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    device_map="auto"
)

# Prepare for LoRA
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=False
)

# LoRA config
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    use_dora=True
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# ============================================================
# DATA COLLATOR
# ============================================================
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,                
    pad_to_multiple_of=8         
)


# ============================================================
# METRICS
# ============================================================
import numpy as np
from evaluate import load

bleu_metric = load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(
        preds, skip_special_tokens=True
    )

    labels = np.where(
        labels < 0,
        tokenizer.pad_token_id,
        labels
    )

    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )

    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels
    )

    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )["score"]

    gen_len = np.mean([
        np.count_nonzero(p != tokenizer.pad_token_id) for p in preds
    ])

    return {
        "bleu": round(bleu, 4),
        "gen_len": round(gen_len, 2)
    }

# ============================================================
# TRAINING
# ============================================================
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="mistral-pt-en-dora",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    bf16=True,
    logging_steps=50,
    save_total_limit=2,
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ============================================================
# INFERENCE + FINAL EVALUATION (CORREGIDO)
# ============================================================
from transformers import GenerationConfig
from tqdm.auto import tqdm

lora_model.eval()

# Cargar configuración de generación
generation_config = GenerationConfig.from_model_config(lora_model.config)
# Aseguramos que pare al final de la frase
generation_config.pad_token_id = tokenizer.eos_token_id 

# Cargar dataset de test
DATA_PATH_TEST = "/home/alumno.upv.es/dnargar/U4_Speech_Translation/Lab/laboratorio/Cuadernos_4_4/L4.2_v5_large.csv"
raw_test = load_dataset("csv", data_files=DATA_PATH_TEST)["train"]

def format_prompt_test(sample):
    return f"Translate to English:\n{sample}\nAnswer:"

# Preparamos las entradas con el prompt
inputs_with_prompt = [format_prompt_test(x) for x in raw_test["hypothesis"]]

# Tokenización
inputs_tokenized = tokenizer(
    inputs_with_prompt,
    padding=True,
    truncation=True,
    max_length=max_tok_length,
    return_tensors="pt"
).to(lora_model.device)

# Generación por lotes
batch_size = 16 
decoded_preds = []

print("Generando traducciones...")
for i in tqdm(range(0, len(inputs_with_prompt), batch_size)):
    batch_inputs = {k: v[i:i+batch_size] for k, v in inputs_tokenized.items()}
    
    with torch.no_grad():
        generated_ids = lora_model.generate(
            **batch_inputs,
            generation_config=generation_config,
            max_new_tokens=128,   # Cuántos tokens generar
            num_beams=1,
            do_sample=False
        )
    
    # Decodificamos todo el bloque
    batch_decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # La salida contiene "Translate to... Answer: Traducción". 
    # Nos quedamos solo con lo que va después de "Answer:"
    for text in batch_decoded:
        if "Answer:" in text:
            # Separamos por "Answer:" y tomamos la última parte
            clean_text = text.split("Answer:")[-1].strip()
        else:
            clean_text = text.strip()
        decoded_preds.append(clean_text)

references = raw_test["translation"]

# ============================================================
# BLEU + COMET
# ============================================================
from whisper.normalizers.basic import BasicTextNormalizer

normalizer = BasicTextNormalizer()

decoded_preds_clean = [
    normalizer(t) for t in decoded_preds
]
references_clean = [
    normalizer(t) for t in references
]

# BLEU
bleu_result = bleu_metric.compute(
    predictions=decoded_preds_clean,
    references=[[r] for r in references_clean]
)

print(f"BLEU: {bleu_result['score']:.2f}")

# COMET
comet_metric = load("comet")

comet_result = comet_metric.compute(
    predictions=decoded_preds_clean,
    references=references_clean,
    sources=raw_test["hypothesis"]
)

print(f"COMET: {comet_result['mean_score'] * 100:.2f}%")
