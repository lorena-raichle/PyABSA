import pandas as pd
import ast
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np

df = pd.read_csv("/PyABSA/results/atepc_batch_results_1.csv")


df = df[:100]

# === 2) Explode into one row per aspect mention ===
exploded = df.explode(["aspect", "sentiment", "confidence", "position"])
exploded = exploded.reset_index(drop=True)

# === 3) Drop invalid aspect entries ===
mask_valid = exploded["aspect"].apply(lambda x: isinstance(x, str) and x.strip() != "")
exploded = exploded[mask_valid].reset_index(drop=True)

# === 4) Load multilingual embedding model ===
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)

# === 5) Define German topic descriptions (3 paraphrases per topic) ===
labels = {
    "Sauberkeit": [
        "Die Bemerkung bezieht sich auf Sauberkeit, Hygiene oder sanitäre Einrichtungen."

    ],
    "Service": [
        "Die Bemerkung betrifft den Service oder das Verhalten des Personals oder Bedienungen.",
        "Es geht um die Freundlichkeit oder Hilfsbereitschaft des Teams.",
        "Der Kommentar bezieht sich auf die Qualität des Kundenservice."
    ],
    "Ambiente": [
        "Die Bemerkung betrifft die Atmosphäre oder das Ambiente des Ortes.",
        "Es geht um die Stimmung, Einrichtung oder das Flair."
    ],
    "Preis": [
        "Die Bemerkung erwähnt den Preis, die Kosten oder das Preis-Leistungs-Verhältnis.",
        "Es geht um die Wahrnehmung von teuer oder günstig.",
        "Der Kommentar betrifft die Angemessenheit der Preise."
    ],
    "Lage": [
        "Die Bemerkung bezieht sich auf die Lage, Entfernung zum Zentrum oder die Umgebung.",
        "Es geht um die Nähe zu Sehenswürdigkeiten oder die Anbindung an den Verkehr.",
        "Der Kommentar betrifft die geografische Position."
    ],
    "Essen": [
        "Die Bemerkung bezieht sich auf das Essen, die Speisen oder die Qualität der Mahlzeiten.",
        "Es geht um Geschmack, Vielfalt oder Präsentation der angebotenen Gerichte.",
        "Der Kommentar betrifft das Frühstück, Abendessen oder das gastronomische Angebot allgemein sowie deren Auswahl und Vielfalt."
    ]
}

# === 6) Compute mean embedding for each topic ===
label_vecs = {
    t: model.encode(descs, normalize_embeddings=True).mean(0)
    for t, descs in labels.items()
}

# === 7) Mapping function using similarity and margin ===
def map_aspect(a, thr_sim=0.45, thr_gap=0.05):
    v = model.encode(a, normalize_embeddings=True)
    sims = {t: float(util.dot_score(v, e)) for t, e in label_vecs.items()}
    best, second = sorted(sims.values(), reverse=True)[:2]
    top_topic = max(sims, key=sims.get)
    if best >= thr_sim and best - second >= thr_gap:
        return top_topic, best
    return "Other", best

# === 8) Apply mapping to all aspects ===
mapped = [map_aspect(a) for a in exploded["aspect"]]
exploded["mapped_topic"] = [m[0] for m in mapped]
exploded["topic_score"] = [m[1] for m in mapped]

# === 9) Save results ===
out_path = os.path.expanduser("/PyABSA/results/atepc_with_topics.csv")
exploded.to_csv(out_path, index=False, encoding="utf-8")

print(f"Saved {len(exploded)} mapped aspect-topic rows to {out_path}")
