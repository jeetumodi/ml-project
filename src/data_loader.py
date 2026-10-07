import os
import torch
from torch.utils.data import Dataset, DataLoader

def parse_fasta(filepath):
    """
    Parses FASTA files with 3-line format:
    >protein_id
    sequence
    label (string of 0s and 1s)
    """
    records = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('>'):
            pid = line[1:].strip()
            if i + 1 < len(lines):
                seq = lines[i+1].strip()
                lbl = None
                if i + 2 < len(lines) and not lines[i+2].startswith('>'):
                    lbl = lines[i+2].strip()
                    i += 3
                else:
                    i += 2
                records.append((pid, seq, lbl))
            else:
                i += 1
        else:
            i += 1
    return records

class ProteinRecord:
    def __init__(self, pid, seq, label_str=None):
        self.pid = pid
        self.seq = seq
        self.label_str = label_str
        self.length = len(seq)
        if label_str is not None:
            self.labels = [int(c) for c in label_str]
        else:
            self.labels = None
