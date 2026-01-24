
from datasets import load_dataset

raw_datasets = load_dataset("csv", data_files="/home/alumno.upv.es/dnargar/U4_Speech_Translation/Lab/laboratorio/Cuadernos_4_3/L4.2_v5_large.csv")

print(raw_datasets)

print(raw_datasets["train"][:1]["hypothesis"])
print(raw_datasets["train"][:1]["reference"])
print(raw_datasets["train"][:1]["translation"])
print(raw_datasets["train"][:1]["hypothesis_clean"])
print(raw_datasets["train"][:1]["reference_clean"])
print(raw_datasets["train"][:1]["translation_clean"])

max_tok_length = 275

from transformers import AutoTokenizer

checkpoint = "facebook/nllb-200-distilled-600M"
# from flores200_codes import flores_codes
src_code = "por_Latn"
tgt_code = "eng_Latn"
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint, 
    padding=True, 
    pad_to_multiple_of=8, 
    src_lang=src_code, 
    tgt_lang=tgt_code, 
    truncation=False, 
    max_length=max_tok_length,
    )

def preprocess_function(sample):
    model_inputs = tokenizer(
        sample["hypothesis"], 
        text_target = sample["translation"],
        )
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    quantization_config=quantization_config
    )

from transformers import GenerationConfig

generation_config = GenerationConfig.from_pretrained(
    checkpoint,
)

print(generation_config)

test_batch_size = 32
batch_tokenized_test = tokenized_datasets['train'].batch(test_batch_size)

number_of_batches = len(batch_tokenized_test["hypothesis"])
output_sequences = []
for i in range(number_of_batches):
    inputs = tokenizer(
        batch_tokenized_test["hypothesis"][i], 
        max_length=max_tok_length, 
        truncation=False, 
        return_tensors="pt", 
        padding=True,
        )
    output_batch = model.generate(
        generation_config=generation_config, 
        input_ids=inputs["input_ids"].cuda(), 
        attention_mask=inputs["attention_mask"].cuda(), 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code), 
        max_length = max_tok_length, 
        num_beams=1, 
        do_sample=False,
        )
    output_sequences.extend(output_batch.cpu())

decoded_preds = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

references = tokenizer.batch_decode(tokenized_datasets["train"]["labels"], skip_special_tokens=True)

decoded_preds[:1]

references[:2]

from whisper.normalizers.basic import BasicTextNormalizer

normalizer = BasicTextNormalizer()

decoded_preds_clean = [normalizer(text) for text in decoded_preds]
references_clean = [normalizer(text) for text in references]

from evaluate import load

metric = load("sacrebleu")

result = metric.compute(predictions=decoded_preds_clean, references=references_clean)
print(f'BLEU score: {result["score"]:.1f}')

from evaluate import load
comet_metric = load('comet')

comet_score = comet_metric.compute(predictions=decoded_preds_clean, references=references_clean, sources=raw_datasets["train"]["hypothesis"])

print(f"COMET: {comet_score['mean_score'] * 100:.2f} %")