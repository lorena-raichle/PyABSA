# -*- coding: utf-8 -*-
# file: make_absa_dataset.py
# author: YANG, HENG (corrected by ChatGPT)
# updated: 2025-06-17

import os
import findfile
from termcolor import colored
from pyabsa.tasks.AspectTermExtraction.prediction.aspect_extractor import AspectExtractor
from pyabsa.utils.pyabsa_utils import fprint
from pyabsa import LabelPaddingOption



def make_ABSA_dataset(dataset_name_or_path, checkpoint="multilingual"):
    fprint(" USING CUSTOM make_ABSA_dataset FUNCTION!")
    """
    Make APC and ATEPC datasets for PyABSA by using the AspectExtractor to label raw sentences.
    Input files must end with '.ignore'. Outputs will be written to .apc and .atepc files.
    """

    if os.path.isdir(dataset_name_or_path):
        fs = findfile.find_files(
            dataset_name_or_path, and_key=[".ignore"], exclude_key=[".apc", ".atepc"]
        )
    elif os.path.isfile(dataset_name_or_path):
        fs = [dataset_name_or_path]
    else:
        fs = findfile.find_cwd_files(
            [dataset_name_or_path, ".dat"], exclude_key=[".apc", ".atepc"]
        )

    if fs:
        aspect_extractor = AspectExtractor(checkpoint=checkpoint)
    else:
        fprint(' No files found! Make sure your dataset names end with ".ignore"')
        return

    fprint(" Start processing dataset: " + colored(dataset_name_or_path, "green"))

    for f in fs:
        with open(f, mode="r", encoding="utf8") as f_in:
            lines = f_in.readlines()

        results = aspect_extractor.batch_predict(lines)

        with open(f.replace(".ignore", "") + ".apc", mode="w", encoding="utf-8") as f_apc_out, \
             open(f.replace(".ignore", "") + ".atepc", mode="w", encoding="utf-8") as f_atepc_out:

            for result in results:
                # Write APC format output
                for aspect, position, sentiment in zip(result["aspect"], result["position"], result["sentiment"]):
                    masked_tokens = result["tokens"][:position[0]] + ["$T$"] + result["tokens"][position[-1] + 1:]
                    f_apc_out.write(" ".join(masked_tokens) + "\n")
                    f_apc_out.write(f"{aspect}\n")
                    f_apc_out.write(f"{sentiment}\n")

                # Write ATEPC format output (corrected sentiment alignment)
                for j, pos in enumerate(result["position"]):
                    sentiment = result["sentiment"][j]
                    aspect_positions = set(pos)  # avoid modifying result["position"] in place
                    for i, (token, IOB) in enumerate(zip(result["tokens"], result["IOB"])):
                        bio_tag = IOB.replace("[CLS]", "O").replace("[SEP]", "O")
                        if i in aspect_positions:
                            f_atepc_out.write(f"{token} {bio_tag} {sentiment}\n")
                        else:
                            f_atepc_out.write(f"{token} {bio_tag} {LabelPaddingOption.LABEL_PADDING}\n")
                    f_atepc_out.write("\n")

    fprint("APC and ATEPC Datasets built for: {}".format(" ".join(fs)))
    fprint(
        colored(
            " You may need to add IDs and move the generated files to "
            "`integrated_datasets/apc_datasets` and `integrated_datasets/atepc_datasets` respectively.",
            "yellow",
        )
    )


make_ABSA_dataset(dataset_name_or_path='/Users/lorenaraichle/Developer/ABSA/PyABSA/docs/6_tutorials/integrated_datasets/review', checkpoint="/Users/lorenaraichle/Developer/ABSA/PyABSA/checkpoints/ATEPC_MULTILINGUAL_CHECKPOINT")
