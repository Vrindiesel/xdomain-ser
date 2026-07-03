# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""NLI model wrapper for batched RoBERTa-MNLI inference.

Loads ``roberta-large-mnli`` from HuggingFace and provides batched
entailment probability computation for (premise, hypothesis) pairs.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


class NLIModel:
    """Wrapper around RoBERTa-large-MNLI for batched entailment inference."""

    # RoBERTa-MNLI label mapping: 0=contradiction, 1=neutral, 2=entailment
    ENTAILMENT_IDX = 2

    def __init__(self, model_name="roberta-large-mnli", device=None):
        """
        Args:
            model_name: HuggingFace model name/path.
            device: torch device string. Auto-detects GPU if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        print(f"Loading NLI model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("NLI model loaded.")

    def get_entailment_prob(self, premise, hypothesis):
        """
        Get entailment probability for a single (premise, hypothesis) pair.

        Returns:
            float: probability of entailment (0 to 1).
        """
        inputs = self.tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs[0, self.ENTAILMENT_IDX].item()

    def batch_entailment(self, pairs, batch_size=32, show_progress=True):
        """
        Compute entailment probabilities for a batch of (premise, hypothesis) pairs.

        Args:
            pairs: list of (premise_str, hypothesis_str) tuples.
            batch_size: number of pairs per forward pass.
            show_progress: whether to show a tqdm progress bar.

        Returns:
            list of float: entailment probabilities, one per pair.
        """
        all_probs = []
        n_batches = (len(pairs) + batch_size - 1) // batch_size
        iterator = range(0, len(pairs), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches, desc="NLI inference")

        for i in iterator:
            batch = pairs[i:i + batch_size]
            premises = [p for p, h in batch]
            hypotheses = [h for p, h in batch]

            inputs = self.tokenizer(
                premises, hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, self.ENTAILMENT_IDX]
            all_probs.extend(probs.cpu().tolist())

        return all_probs
