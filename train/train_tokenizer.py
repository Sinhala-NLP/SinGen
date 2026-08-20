#!/usr/bin/env python
"""Train a tokenizer from scratch on sinhala-nlp/sinhala-7m-corpus.

bert / electra -> WordPiece (64k) ; roberta -> byte-level BPE (64k).

Normalisation is Sinhala-safe: strip_accents=False and lowercase=False.
The default BERT normaliser NFD-decomposes and strips combining marks, which
destroys Sinhala vowel signs (pilla) and conjunct characters -- so we disable it.
Byte-level BPE operates on raw bytes and needs no normalisation.
"""

import argparse

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_type",
        choices=["bert", "roberta", "electra"],
        required=True,
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--dataset_name",
        default="sinhala-nlp/sinhala-7m-corpus",
    )
    p.add_argument("--text_column", default="text")
    p.add_argument("--vocab_size", type=int, default=64000)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=10000)
    return p.parse_args()


def batch_iterator(ds, text_column, batch_size):
    """Yield batches containing only valid, non-empty strings."""
    for i in range(0, len(ds), batch_size):
        texts = ds[i:i + batch_size][text_column]

        # Remove None, non-string values, and empty/whitespace-only strings.
        texts = [
            text
            for text in texts
            if isinstance(text, str) and text.strip()
        ]

        if texts:
            yield texts


def train_wordpiece(ds, args):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
    from tokenizers.normalizers import BertNormalizer
    from tokenizers.pre_tokenizers import BertPreTokenizer
    from tokenizers.processors import TemplateProcessing
    from tokenizers.decoders import WordPiece as WordPieceDecoder
    from transformers import BertTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tok = Tokenizer(WordPiece(unk_token="[UNK]"))

    # Sinhala-critical settings:
    # Do not lowercase or strip combining marks.
    tok.normalizer = BertNormalizer(
        lowercase=False,
        strip_accents=False,
    )

    tok.pre_tokenizer = BertPreTokenizer()
    tok.decoder = WordPieceDecoder(prefix="##")

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=specials,
        continuing_subword_prefix="##",
    )

    tok.train_from_iterator(
        batch_iterator(
            ds,
            args.text_column,
            args.batch_size,
        ),
        trainer=trainer,
        length=len(ds),
    )

    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tok.token_to_id("[CLS]")),
            ("[SEP]", tok.token_to_id("[SEP]")),
        ],
    )

    fast = BertTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        model_max_length=512,
    )

    fast.save_pretrained(args.output_dir)


def train_bpe(ds, args):
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import RobertaProcessing
    from transformers import RobertaTokenizerFast

    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    tok = ByteLevelBPETokenizer()

    tok.train_from_iterator(
        batch_iterator(
            ds,
            args.text_column,
            args.batch_size,
        ),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=specials,
        length=len(ds),
    )

    tok._tokenizer.post_processor = RobertaProcessing(
        sep=("</s>", tok.token_to_id("</s>")),
        cls=("<s>", tok.token_to_id("<s>")),
    )

    fast = RobertaTokenizerFast(
        tokenizer_object=tok._tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<s>",
        sep_token="</s>",
        mask_token="<mask>",
        model_max_length=512,
    )

    fast.save_pretrained(args.output_dir)


def main():
    args = parse_args()

    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name, split="train")

    print(f"Original dataset size: {len(ds):,}")
    print(f"Dataset columns: {ds.column_names}")

    # Check that the requested text column exists.
    if args.text_column not in ds.column_names:
        raise ValueError(
            f"Text column '{args.text_column}' not found. "
            f"Available columns: {ds.column_names}"
        )

    # Remove None, non-string, empty, and whitespace-only examples.
    print(f"Cleaning column: {args.text_column}")

    original_size = len(ds)

    ds = ds.filter(
        lambda x: (
            isinstance(x[args.text_column], str)
            and bool(x[args.text_column].strip())
        )
    )

    removed = original_size - len(ds)

    print(f"Removed invalid/empty examples: {removed:,}")
    print(f"Remaining examples: {len(ds):,}")

    if len(ds) == 0:
        raise ValueError(
            f"No valid examples remain in column '{args.text_column}'."
        )

    print(f"Training {args.model_type} tokenizer...")
    print(f"Vocabulary size: {args.vocab_size:,}")
    print(f"Minimum frequency: {args.min_frequency}")
    print(f"Output directory: {args.output_dir}")

    if args.model_type in ("bert", "electra"):
        train_wordpiece(ds, args)
    else:
        train_bpe(ds, args)

    print()
    print("=" * 60)
    print("Tokenizer training completed successfully.")
    print(f"Model type : {args.model_type}")
    print(f"Vocabulary : {args.vocab_size:,}")
    print(f"Examples   : {len(ds):,}")
    print(f"Saved to   : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()