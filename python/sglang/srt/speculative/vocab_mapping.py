# SPDX-License-Identifier: Apache-2.0
"""
Vocabulary mapping for Token-Level Intersection (TLI) speculative decoding.

Based on the ICML 2025 oral paper:
  "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for
   Heterogeneous Vocabularies" — Timor et al., https://arxiv.org/abs/2502.05202

This module builds a normalized token intersection between the target and draft
model vocabularies and provides:
- logit masking (constrain draft logits to the intersection)
- target-to-draft token ID mapping
- draft-to-target token ID mapping
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Tokenizer-specific space-prefix characters used by BPE tokenizers.
# BPE vocabularies often encode a leading space as one of these Unicode chars.
_SPACE_PREFIXES = ("\u0120", "\u2581")


def _normalize_token(token: str) -> str:
    """Normalize a BPE token by converting space-prefix characters to a plain space."""
    for prefix in _SPACE_PREFIXES:
        if token.startswith(prefix):
            return " " + token[len(prefix) :]
    return token


class VocabMapping:
    """Maps token IDs between target and draft model vocabularies via intersection.

    The intersection is computed by normalizing token strings (so that
    different BPE space-prefix representations still match), then finding
    tokens that appear in both vocabularies.

    Args:
        target_tokenizer: HuggingFace tokenizer for the target model.
        draft_tokenizer:  HuggingFace tokenizer for the draft model.
        target_vocab_size: Vocabulary size of the target model.
        draft_vocab_size:  Vocabulary size of the draft model.
        device: Torch device to place mapping tensors on.
    """

    def __init__(
        self,
        target_tokenizer,
        draft_tokenizer,
        target_vocab_size: int,
        draft_vocab_size: int,
        device: torch.device,
    ):
        self.target_vocab_size = target_vocab_size
        self.draft_vocab_size = draft_vocab_size
        self.device = device

        # Validate that both tokenizers have an unk_token_id.
        # Silently falling back to token 0 would corrupt the draft KV cache
        # for out-of-intersection tokens on models where token 0 is a special
        # token (e.g. BOS on Llama).
        self.target_unk_token_id: int = target_tokenizer.unk_token_id
        if self.target_unk_token_id is None:
            raise ValueError(
                "Target tokenizer does not have an unk_token_id. "
                "TLI (universal_draft) requires tokenizers with a defined unk_token. "
                "Compatible models include Qwen (all generations), Llama 1/2, Mistral, "
                "and Gemma. Llama 3 and SmolLM2 are not supported."
            )
        self.draft_unk_token_id: int = draft_tokenizer.unk_token_id
        if self.draft_unk_token_id is None:
            raise ValueError(
                "Draft tokenizer does not have an unk_token_id. "
                "TLI (universal_draft) requires tokenizers with a defined unk_token. "
                "Compatible models include Qwen (all generations), Llama 1/2, Mistral, "
                "and Gemma. Llama 3 and SmolLM2 are not supported."
            )

        # Build normalized vocabularies.  When two tokens normalize to the
        # same string we keep only the first occurrence (lowest token ID).
        target_vocab = target_tokenizer.get_vocab()
        draft_vocab = draft_tokenizer.get_vocab()

        target_normalized: dict[str, int] = {}
        for token, tid in target_vocab.items():
            norm = _normalize_token(token)
            if norm not in target_normalized:
                target_normalized[norm] = tid

        draft_normalized: dict[str, int] = {}
        for token, tid in draft_vocab.items():
            norm = _normalize_token(token)
            if norm not in draft_normalized:
                draft_normalized[norm] = tid

        # Intersection: tokens that appear in both vocabularies.
        common_tokens = set(target_normalized.keys()) & set(draft_normalized.keys())

        # Build mapping tensors (initialized to -1 = "not in intersection").
        draft_to_target = torch.full((draft_vocab_size,), -1, dtype=torch.long)
        target_to_draft = torch.full((target_vocab_size,), -1, dtype=torch.long)
        intersection_mask_draft = torch.zeros(draft_vocab_size, dtype=torch.bool)

        for norm_token in common_tokens:
            t_id = target_normalized[norm_token]
            d_id = draft_normalized[norm_token]
            if t_id < target_vocab_size and d_id < draft_vocab_size:
                draft_to_target[d_id] = t_id
                target_to_draft[t_id] = d_id
                intersection_mask_draft[d_id] = True

        self.draft_to_target_ids = draft_to_target.to(device)
        self.target_to_draft_ids = target_to_draft.to(device)
        # Boolean mask over the draft vocabulary: True iff the token is in the intersection.
        self.intersection_mask_draft = intersection_mask_draft.to(device)
        self.intersection_size = int(intersection_mask_draft.sum().item())

        logger.info(
            "VocabMapping initialized: target_vocab=%d, draft_vocab=%d, "
            "intersection=%d (%.1f%% of draft, %.1f%% of target)",
            target_vocab_size,
            draft_vocab_size,
            self.intersection_size,
            100.0 * self.intersection_size / max(draft_vocab_size, 1),
            100.0 * self.intersection_size / max(target_vocab_size, 1),
        )

        if self.intersection_size < 100:
            logger.warning(
                "Very small vocabulary intersection (%d tokens). "
                "TLI acceptance rate will be very low.",
                self.intersection_size,
            )

    def map_target_to_draft_ids(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Map target model token IDs to draft model token IDs.

        Tokens not in the intersection are mapped to ``draft_unk_token_id``.

        Args:
            target_ids: Integer tensor of target token IDs.

        Returns:
            Integer tensor of draft token IDs with the same shape and dtype.
        """
        draft_ids = self.target_to_draft_ids[target_ids]
        not_in_intersection = draft_ids == -1
        if not_in_intersection.any():
            draft_ids = draft_ids.clone()
            draft_ids[not_in_intersection] = self.draft_unk_token_id
        return draft_ids.to(target_ids.dtype)

    def map_draft_to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Map draft model token IDs to target model token IDs.

        Tokens not in the intersection are mapped to ``target_unk_token_id``.

        Args:
            draft_ids: Integer tensor of draft token IDs.

        Returns:
            Integer tensor of target token IDs with the same shape and dtype.
        """
        target_ids = self.draft_to_target_ids[draft_ids]
        not_in_intersection = target_ids == -1
        if not_in_intersection.any():
            target_ids = target_ids.clone()
            target_ids[not_in_intersection] = self.target_unk_token_id
        return target_ids.to(draft_ids.dtype)

    def constrain_draft_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Zero out (−inf) logits for tokens outside the intersection.

        Args:
            logits: Float tensor of shape ``(..., draft_vocab_size)``.

        Returns:
            A cloned tensor with non-intersection logits set to ``-inf``.
        """
        logits = logits.clone()
        logits[..., ~self.intersection_mask_draft] = float("-inf")
        return logits
