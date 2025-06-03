import torch
import json
import os
from pyabsa import AspectSentimentTripletExtraction as ASTE

# 1) Pick device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# 2) Point at your local ASTE checkpoint
aste_ckpt = os.path.expanduser(
    "/PyABSA/checkpoints/ASTE_MULTILINGUAL_CHECKPOINT"
)

# 3) Instantiate the extractor once
extractor = ASTE.AspectSentimentTripletExtractor(
    checkpoint=aste_ckpt,
    auto_device=False,
    device=device,
    cal_perplexity=True,   # enables a per-triplet “confidence” if supported
)

# 4) Your multilingual sentences
sentences = [
    "Die Toiletten waren schmutzig, aber das Ambiente ist nett!",
    "J'adore ce film, il est vraiment bien !",
    "Il servizio era lento ma il cibo era ottimo.",
    "I love this movie, it is so great!"
]

# 5) Run inference + restructure in one loop
output = []
for sent in sentences:
    # call predict with a single-item list to preserve alignment
    out = extractor.predict(
        [sent],
        save_result=False,
        print_result=False,
        ignore_error=True
    )[0]

    # out["Triplets"] is a list of dicts: { "Aspect", "Opinion", "Polarity", ... }
    triplets = out.get("Triplets", [])

    # build our per-aspect records, pulling confidence if present
    extractions = []
    for trip in triplets:
        rec = {
            "aspect":   trip.get("Aspect"),
            "opinion":  trip.get("Opinion"),
            "sentiment":trip.get("Polarity"),
        }
        # if the model returned a confidence per trip, include it
        if "confidence" in trip:
            rec["confidence"] = trip["confidence"]
        extractions.append(rec)

    output.append({
        "sentence":   out["sentence"],
        "extractions":extractions
    })

# 6) Write final JSON
with open("/PyABSA/results/aspect_triplet.json", "w", encoding="utf-8") as fp:
    json.dump(output, fp, ensure_ascii=False, indent=2)

print(f"Saved {len(output)} records to triplet_restructured.json")
