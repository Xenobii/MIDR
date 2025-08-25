"""
    Convertion from MIDI to note sequences

    Based on: https://github.com/sony/hFT-Transformer
"""

from pathlib import Path 
import numpy as np
import argparse
import json


def midr2label(midr_path):
    """
    Converts a list of midr notes to labels for training
    """
    # (1) Read files
    midr_path = Path(midr_path)
    json_path = Path(json_path)

    if not midr_path.exists():
        raise ValueError(f"Path does not exist: {midr_path}")
    
    with open(str(midr_path), "r") as f:
        midr = json.load(f)

    # (2) Compute settings TEMP OTI NANAI
    nframe = int(500) # temp

    # (3) Create label based on midr

    # 4D matrices for 3D sequential data
    # mpe       : 0 or 1
    # onset     : 0.0-1.0
    # offset    : 0.0-1.0
    # velocity  : 0 - 127

    if midr["type"] == "standard":
        label = std_label()
    else: 
        raise ValueError("Invalid midrep type")
    
    return label


def std_label():
    return 0



if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument("midr_path", help="path to the midi file")
    argumentParser.add_argument("json_path", help="path to the output json")

    args = argumentParser.parse_args()

    midr_path = args.midr_path
    json_path = args.json_path

    midr2label(midr_path)