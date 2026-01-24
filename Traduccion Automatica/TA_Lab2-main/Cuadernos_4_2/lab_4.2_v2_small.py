import whisper
import jiwer
from whisper.normalizers import BasicTextNormalizer

from tqdm.notebook import tqdm
import pandas as pd

model = whisper.load_model("small")
audios = []

from datasets import load_dataset

raw_datasets = load_dataset("facebook/covost2", 'pt_en', data_dir="/home/alumno.upv.es/dnargar/U4_Speech_Translation/Lab/covost2/pt/cv-corpus-23.0-2025-09-05/pt")

data = raw_datasets["test"][:1000]
translations = []
for sample in data["file"]:
    translations.append((model.transcribe(sample, language="Portuguese", task="translate"))['text'])

normalizer = BasicTextNormalizer()

final_data = {
    "translation": translations,                    # Lo que predijo el modelo (Inglés)
    
    "reference": data["translation"],               
    
    "source": data["sentence"],                     
    
    "translation_clean": [normalizer(text) for text in translations],
    "reference_clean": [normalizer(text) for text in data["translation"]], 
    "source_clean": [normalizer(text) for text in data["sentence"]]
}

from evaluate import load

metric = load("sacrebleu")

# predictions: Lo que predijo el modelo ("translation_clean")
# references: La traducción humana correcta ("reference_clean")
result = metric.compute(
    predictions=final_data["translation_clean"], 
    references=final_data["reference_clean"]
)

print(f'BLEU score: {result["score"]:.1f}')

from evaluate import load
comet_metric = load('comet')

comet_score = comet_metric.compute(
    predictions=final_data["translation_clean"],  # Lo que predijo el modelo (Inglés)
    references=final_data["reference_clean"],     # Humano (Inglés)
    sources=final_data["source_clean"]            # Original (Portugués)
)

print(f"COMET: {comet_score['mean_score'] * 100:.2f} %")

df = pd.DataFrame(final_data)
pd.set_option('display.max_colwidth', None)

# Exportar
df.to_csv('./Cuadernos_4_2/L4.2_v2_small.csv', encoding='utf-8', index=False)