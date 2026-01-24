import whisper
from whisper.normalizers.basic import BasicTextNormalizer

from tqdm.notebook import tqdm
import pandas as pd

import jiwer

model = whisper.load_model("large")

from datasets import load_dataset

raw_datasets = load_dataset("facebook/covost2", 'pt_en', data_dir="/home/alumno.upv.es/dnargar/U4_Speech_Translation/Lab/covost2/pt/cv-corpus-23.0-2025-09-05/pt")

print(raw_datasets)

print(raw_datasets["train"].features)

print(raw_datasets["train"][:5]["file"])

print(raw_datasets["train"][:5]["audio"])

print(raw_datasets["train"][:5]["sentence"])

print(raw_datasets["train"][:5]["translation"])

data=raw_datasets["validation"]

hypotheses = []
for sample in data["file"]:
    hypotheses.append((model.transcribe(sample, language="Portuguese"))['text'])

normalizer = BasicTextNormalizer()

hypotheses_clean = [normalizer(text) for text in hypotheses]
sentence_clean = [normalizer(text) for text in data["sentence"]]
translation_clean = [normalizer(text) for text in data["translation"]]

wer = jiwer.wer(sentence_clean, hypotheses_clean)

print(f"WER: {wer * 100:.2f} %")

dataframe = pd.DataFrame(dict(hypothesis=hypotheses, reference=data["sentence"], translation=data["translation"], hypothesis_clean=hypotheses_clean,  reference_clean=sentence_clean, translation_clean=translation_clean ))
pd.set_option('display.max_colwidth', None)
dataframe.head(1)

dataframe.to_csv('./Cuadernos_4_4/L4.1_v5_large_validation.csv', encoding='utf-8', index=False)