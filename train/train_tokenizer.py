#!/usr/bin/env python
"""Train a tokeniser from scratch on sinhala-nlp/sinhala-7m-corpus.

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
    p.add_argument("--model_type", choices=["bert", "roberta", "electra"], required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_name", default="sinhala-nlp/sinhala-7m-corpus")
    p.add_argument("--text_column", default="text")
    p.add_argument("--vocab_size", type=int, default=64000)
    p.add_argument("--min_frequency", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=10000)
    return p.parse_args()


def batch_iterator(ds, text_column, batch_size):
    # drop None / non-string / empty rows; the Rust trainer errors on a null
    # ('NoneType' object cannot be converted to 'PyString').
    for i in range(0, len(ds), batch_size):
        texts = ds[i:i + batch_size][text_column]
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
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
    # strip_accents=False is the Sinhala-critical setting.
    tok.normalizer = BertNormalizer(lowercase=False, strip_accents=False)
    tok.pre_tokenizer = BertPreTokenizer()
    tok.decoder = WordPieceDecoder(prefix="##")

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=specials,
        continuing_subword_prefix="##",
    )
    tok.train_from_iterator(
        batch_iterator(ds, args.text_column, args.batch_size),
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
        unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]",
        cls_token="[CLS]", mask_token="[MASK]", model_max_length=512,
        # CRITICAL: without these, BertTokenizerFast rebuilds the normalizer from
        # its defaults (lowercase=True, strip_accents=None) and NFD-strips Sinhala
        # combining marks -- dropping hal kirima and decomposing vowel signs.
        do_lower_case=False, strip_accents=False, tokenize_chinese_chars=False,
    )
    fast.save_pretrained(args.output_dir)


def train_bpe(ds, args):
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import RobertaProcessing
    from transformers import RobertaTokenizerFast

    specials = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        batch_iterator(ds, args.text_column, args.batch_size),
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
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
        cls_token="<s>", sep_token="</s>", mask_token="<mask>", model_max_length=512,
    )
    fast.save_pretrained(args.output_dir)


def main():
    args = parse_args()
    ds = load_dataset(args.dataset_name, split="train")
    if args.model_type in ("bert", "electra"):
        train_wordpiece(ds, args)
    else:
        train_bpe(ds, args)
    print(f"saved {args.model_type} tokenizer (vocab={args.vocab_size}) -> {args.output_dir}")


if __name__ == "__main__":
    main()