import whisper
from whisper.normalizers.basic import BasicTextNormalizer

from tqdm.notebook import tqdm
import pandas as pd

import jiwer

model = whisper.load_model("turbo")

from datasets import load_dataset

raw_datasets = load_dataset("facebook/covost2", 'pt_en', data_dir="/home/alumno.upv.es/dnargar/U4_Speech_Translation/Lab/covost2/pt/cv-corpus-23.0-2025-09-05/pt")

print(raw_datasets)

print(raw_datasets["train"].features)

print(raw_datasets["train"][:5]["file"])

print(raw_datasets["train"][:5]["audio"])

print(raw_datasets["train"][:5]["sentence"])

print(raw_datasets["train"][:5]["translation"])

data=raw_datasets["test"][:1000]

hypotheses = []
for sample in data["file"]:
    hypotheses.append((model.transcribe(sample, language="Portuguese"))['text'])

data["hypothesis"]=hypotheses

print(data["hypothesis"][:5])

normalizer = BasicTextNormalizer()

data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["sentence_clean"] = [normalizer(text) for text in data["sentence"]]
data["translation_clean"] = [normalizer(text) for text in data["translation"]]


wer = jiwer.wer(list(data["sentence_clean"]), list(data["hypothesis_clean"]))

print(f"WER: {wer * 100:.2f} %")

dataframe = pd.DataFrame(dict(transcription=data["hypothesis"], sentence=data["sentence"], translation=data["translation"], transcription_clean=data["hypothesis_clean"],  sentence_clean=data["sentence_clean"], translation_clean=data["translation_clean"] ))
pd.set_option('display.max_colwidth', None)
dataframe.head(1)

dataframe.to_csv('L4.1_v4_turbo.csv', encoding='utf-8')
