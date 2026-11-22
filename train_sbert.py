"""Fine-tune Sentence-BERT on verified schema correspondences.

Example: python train_sbert.py --train-pairs train_pairs.json --output models/schema-matcher
Input is a JSON list of {"anchor": "...", "positive": "..."} records.
"""
import argparse
import json
from pathlib import Path


def prepare_pairs(path):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        raise ValueError('Training data must be a JSON array')
    seen = set()
    cleaned = []
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip() for k in ('anchor', 'positive')):
            raise ValueError('Each pair must have non-empty anchor and positive strings')
        pair = (row['anchor'].strip(), row['positive'].strip())
        if pair not in seen:
            seen.add(pair)
            cleaned.append({'anchor': pair[0], 'positive': pair[1]})
    if len(cleaned) < 2:
        raise ValueError('At least two distinct training pairs are required')
    return cleaned


def train(args):
    pairs = prepare_pairs(args.train_pairs)
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        raise ValueError('epochs >= 1, batch-size >= 2, and learning-rate > 0 are required')
    try:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
        from sentence_transformers.training_args import BatchSamplers
    except ImportError as exc:
        raise RuntimeError('Install neural dependencies: pip install -r requirements-neural.txt') from exc
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Output directory is not empty: {output}')
    model = SentenceTransformer(args.model)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, learning_rate=args.learning_rate,
        batch_sampler=BatchSamplers.NO_DUPLICATES, seed=args.seed,
        report_to='none', save_strategy='no',
    )
    trainer = SentenceTransformerTrainer(model=model, args=training_args,
                                        train_dataset=Dataset.from_list(pairs),
                                        loss=losses.MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save_pretrained(str(output))
    (output / 'training_metadata.json').write_text(json.dumps({
        'base_model': args.model, 'pairs': len(pairs), 'epochs': args.epochs,
        'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
        'seed': args.seed, 'loss': 'MultipleNegativesRankingLoss',
        'warning': 'Distinct descriptions can still be semantically equivalent; audit false negatives and split databases.'
    }, indent=2) + '\n')
    return {'model': str(output), 'training_pairs': len(pairs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-pairs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == '__main__':
    main()
