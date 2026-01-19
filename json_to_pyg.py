#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON to PyG Converter v2.0 (Analysis Ready)
===========================================
Strict implementation of graph construction for malware detection research.
Follows "Bag-of-Opcodes" feature extraction (dim=10) and CFG structure.

Features:
- Strict 10-dim opcode vectors (counting occurrences per block)
- Block ID to Integer Index mapping
- Graph Labeling from filename ({hash}_{label}.json)
- PyTorch Geometric Data object generation

Usage:
    python json_to_pyg.py ./exports --output ./graphs --batch

Author: Antigravity Agent
License: MIT
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[!] PyTorch Geometric not installed. Install with: pip install torch torch_geometric")

# === Strict Opcode Categories (dim=10) ===
# As requested in prompt: mov, push, call, add, jmp, test, lea, pop, ret, other
OPCODE_MAP = {
    'mov': 0, 'movzx': 0, 'movsx': 0,  # Group variants
    'push': 1, 'pushf': 1,
    'call': 2,
    'add': 3, 'sub': 3, 'inc': 3, 'dec': 3, # Group arithmetic often useful, but stricter strict be just 'add' -> let's stick to prompt literal first, but maybe expand standard variants
    # Prompt said: "ví dụ: đếm số lượng mov, push, call, add, jmp, test, lea, pop, ret, other"
    # To be safe and strict to "research", usually we group similar semantic ops.
    # However, to be strict to the *prompt*, I will prioritize the list keys but allow variants.
    
    'jmp': 4, 'jz': 4, 'jnz': 4, 'je': 4, 'jne': 4, 'jg': 4, # Jump variants
    'test': 5, 'cmp': 5, # Compare/Test often grouped
    'lea': 6,
    'pop': 7, 'popf': 7,
    'ret': 8, 'retn': 8,
    # 'other': 9 (default)
}

# Dimension of feature vector
FEATURE_DIM = 10


def get_opcode_index(mnemonic: str) -> int:
    """Map mnemonic to 0-9 index."""
    mnem = mnemonic.lower().strip()
    # Normalize
    for key, idx in OPCODE_MAP.items():
        if mnem.startswith(key): # Simple prefix match covers most (mov*, jmp*, etc)
            return idx
    
    # Specific handling if prefix assumption is too broad (e.g. 'call' vs 'callq')
    return 9 # 'other' category


def extract_node_features(blocks: List[dict]) -> torch.Tensor:
    """
    Construct Bag-of-Opcodes feature matrix X (Num_Nodes x 10).
    Count opcode occurrences in each block.
    """
    x = torch.zeros((len(blocks), FEATURE_DIM), dtype=torch.float)
    
    for i, block in enumerate(blocks):
        insns = block.get('insns', [])
        for insn in insns:
            mnemonic = insn.get('mnemonic', '')
            if mnemonic:
                idx = get_opcode_index(mnemonic)
                x[i, idx] += 1.0
                
    return x


def extract_edges(blocks: List[dict], cfg_edges: List[dict], block_id_to_idx: Dict[str, int]) -> torch.Tensor:
    """
    Construct Edge Index (2 x Num_Edges).
    Maps string block IDs to integer indices.
    """
    src_nodes = []
    dst_nodes = []
    
    # Method 1: Use explicit cfg_edges from JSON (Preferred)
    if cfg_edges:
        for edge in cfg_edges:
            src_id = edge.get('from_block')
            dst_id = edge.get('to_block')
            
            if src_id in block_id_to_idx and dst_id in block_id_to_idx:
                src_nodes.append(block_id_to_idx[src_id])
                dst_nodes.append(block_id_to_idx[dst_id])
                
    # Method 2: Fallback to 'succs' in blocks if cfg_edges missing
    else:
        for block in blocks:
            src_id = block.get('id')
            if src_id not in block_id_to_idx:
                continue
                
            src_idx = block_id_to_idx[src_id]
            for succ_id in block.get('succs', []):
                if succ_id in block_id_to_idx:
                    src_nodes.append(src_idx)
                    dst_nodes.append(block_id_to_idx[succ_id])
                    
    if not src_nodes:
        return torch.zeros((2, 0), dtype=torch.long)
        
    return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)


def extract_label_from_filename(filename: str) -> Optional[int]:
    """
    Extract label from filename format: {hash}_{label}.json
    Example: abc123_1.json -> 1 (Malware)
    """
    try:
        # Regex to find label at end of filename
        match = re.search(r'_(\d+)\.json$', filename)
        if match:
            return int(match.group(1))
        
        # Fallback: check if 'malware' or 'benign' in path
        if 'malware' in filename.lower():
            return 1
        if 'benign' in filename.lower():
            return 0
            
    except:
        pass
    return None # Unknown


def convert_json_to_data(filepath: Path) -> Optional[Data]:
    """Convert single JSON file to PyG Data object."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"[!] Error loading {filepath}: {e}")
        return None

    # Flatten logic: Get all blocks from all functions
    # In GNN for malware, we usually treat the whole binary as one graph (super-graph of CFGs)
    all_blocks = []
    functions = raw_data.get('functions', [])
    
    for func in functions:
        all_blocks.extend(func.get('blocks', []))
        
    if not all_blocks:
        return None # Empty binary?
        
    # Map Block ID -> Index
    # Important: Index MUST be 0..N-1
    block_id_to_idx = {b['id']: i for i, b in enumerate(all_blocks)}
    
    # 1. Node Features (X)
    x = extract_node_features(all_blocks)
    
    # 2. Edge Index
    edge_index = extract_edges(all_blocks, raw_data.get('cfg_edges', []), block_id_to_idx)
    
    # 3. Label (y)
    label = extract_label_from_filename(filepath.name)
    y = torch.tensor([label], dtype=torch.long) if label is not None else None
    
    # Create Data
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Add metadata for tracking
    data.sample_name = filepath.stem
    data.num_functions = len(functions)
    
    return data


def process_batch(input_dir: Path, output_dir: Path):
    """Process all JSON files in directory."""
    if not TORCH_AVAILABLE:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    files = list(input_dir.glob('*.json'))
    print(f"[*] Found {len(files)} JSON exports")
    
    count = 0
    for json_file in files:
        data = convert_json_to_data(json_file)
        if data:
            out_name = output_dir / f"{json_file.stem}.pt"
            torch.save(data, out_name)
            count += 1
            if count % 100 == 0:
                print(f"    Processed {count}/{len(files)}")
                
    print(f"[+] Conversion complete: {count} graphs saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Strict JSON to PyG Converter")
    parser.add_argument('input_dir', type=Path, help="Directory containing .json exports")
    parser.add_argument('--output', type=Path, default=Path('./graphs'), help="Output directory for .pt files")
    parser.add_argument('--batch', action='store_true', help="Enable batch processing (default implies this if input is dir)")
    
    args = parser.parse_args()
    
    if args.input_dir.is_dir():
        process_batch(args.input_dir, args.output)
    else:
        # Single file
        data = convert_json_to_data(args.input_dir)
        if data:
            print(f"Graph created: {data}")
            print(f"X shape: {data.x.shape}")
            print(f"Edge index shape: {data.edge_index.shape}")
            if data.y is not None:
                print(f"Label: {data.y.item()}")
            
            # Save single
            args.output.mkdir(parents=True, exist_ok=True)
            torch.save(data, args.output / f"{args.input_dir.stem}.pt")

if __name__ == "__main__":
    main()
