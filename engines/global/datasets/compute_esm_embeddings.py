"""Compute ESM2 per-token embeddings for protein sequences in a FASTA file.

Replaces: python -m esm.scripts.extract esm2_t33_650M_UR50D input.fasta output_dir/ --repr_layers 33 --include per_tok

Usage:
    python -m datasets.compute_esm_embeddings \
        --fasta_file data/em_prepared_for_esm.fasta \
        --output_dir data/esm_embeddings_raw/ \
        --batch_size 4
"""

import os
import argparse
import torch
from tqdm import tqdm


def read_fasta(fasta_path):
    """Parse FASTA file without BioPython."""
    sequences = []
    current_id = None
    current_seq = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                current_id = line[1:].split()[0]  # take first word as ID
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_id is not None:
        sequences.append((current_id, ''.join(current_seq)))
    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_file', type=str, required=True,
                        help='Input FASTA file with protein sequences')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for per-sequence .pt files')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for ESM inference')
    parser.add_argument('--repr_layer', type=int, default=33,
                        help='Which ESM2 layer to extract (default: 33, the last layer)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load ESM2 model
    print("Loading ESM2 model (esm2_t33_650M_UR50D)...")
    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # Read sequences from FASTA
    records = read_fasta(args.fasta_file)
    print(f"Loaded {len(records)} sequences from {args.fasta_file}")

    # Skip already computed
    existing = {f.replace('.pt', '') for f in os.listdir(args.output_dir) if f.endswith('.pt')}
    to_process = [(name, seq) for name, seq in records if name not in existing]
    if len(existing) > 0:
        print(f"Skipping {len(existing)} already computed, {len(to_process)} remaining")

    # Process in batches
    for i in tqdm(range(0, len(to_process), args.batch_size), desc="Computing ESM embeddings"):
        batch_data = to_process[i:i + args.batch_size]

        # Truncate very long sequences to avoid OOM (ESM2 max ~1022 tokens recommended)
        batch_data_truncated = [(name, seq[:1022]) for name, seq in batch_data]

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data_truncated)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[args.repr_layer], return_contacts=False)

        representations = results["representations"][args.repr_layer]

        # Save each sequence's embedding (excluding BOS/EOS tokens)
        for j, (name, seq) in enumerate(batch_data_truncated):
            seq_len = len(seq)
            # tokens: [BOS, ...seq..., EOS], so embedding is at positions 1:seq_len+1
            embedding = representations[j, 1:seq_len + 1].cpu()

            output = {'label': name, 'representations': {args.repr_layer: embedding}}
            torch.save(output, os.path.join(args.output_dir, f'{name}.pt'))

    print(f"Done! Saved {len(to_process)} embeddings to {args.output_dir}")


if __name__ == '__main__':
    main()
