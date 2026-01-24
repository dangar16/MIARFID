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

checkpoint = "facebook/nllb-200-distilled-600M"
src_code = "por_Latn"
tgt_code = "eng_Latn"
max_tok_length = 275

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    src_lang=src_code,
    tgt_lang=tgt_code,
    padding=True,
    pad_to_multiple_of=8,
    truncation=True,
    max_length=max_tok_length,
)

def preprocess_function(batch):
    return tokenizer(
        batch["hypothesis"],
        text_target=batch["translation"],
    )

tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# ============================================================
# MODEL (4-bit + LoRA)
# ============================================================
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
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
    task_type="SEQ_2_SEQ_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# ============================================================
# DATA COLLATOR
# ============================================================
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=lora_model,
    pad_to_multiple_of=8,
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
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="nllb-pt-en-lora",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    bf16=True,
    logging_steps=50,
    save_total_limit=2,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
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
# INFERENCE + FINAL EVALUATION
# ============================================================
from transformers import GenerationConfig

lora_model.eval()

generation_config = GenerationConfig.from_model_config(
    lora_model.config
)
generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

# Load full dataset for final evaluation
DATA_PATH_TEST = "/home/alumno.upv.es/dnargar/U4_Speech_Translation/Lab/laboratorio/Cuadernos_4_4/L4.2_v5_large.csv"

raw_test = load_dataset(
    "csv",
    data_files=DATA_PATH_TEST
)["train"]

def preprocess_test(batch):
    return tokenizer(
        batch["hypothesis"],
        truncation=True,
        max_length=max_tok_length
    )

tokenized_test = raw_test.map(
    preprocess_test,
    batched=True
)

# Batched generation
test_batch_size = 32
tokenized_test = tokenized_test.batch(test_batch_size)

outputs = []

for i in range(len(tokenized_test["hypothesis"])):
    inputs = tokenizer(
        tokenized_test["hypothesis"][i],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tok_length
    ).to(lora_model.device)

    with torch.no_grad():
        generated = lora_model.generate(
            **inputs,
            generation_config=generation_config,
            max_length=max_tok_length,
            num_beams=3,
            do_sample=False,
        )

    outputs.extend(generated.cpu())

decoded_preds = tokenizer.batch_decode(
    outputs,
    skip_special_tokens=True
)

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
