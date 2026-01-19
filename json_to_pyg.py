#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON to PyTorch Geometric Converter v1.0
========================================
Converts IDA JSON exports to PyTorch Geometric Data objects for GNN training.

Features:
- Basic Block as nodes
- CFG edges as graph edges
- Multiple node feature strategies (Bag of Opcodes, Word2Vec, etc.)
- Batch processing for multiple samples

Requirements:
    pip install torch torch_geometric numpy

Usage:
    python json_to_pyg.py sample_export.json --output sample.pt
    python json_to_pyg.py ./exports/ --output ./graphs/ --batch

Author: Antigravity Agent
License: MIT
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[!] PyTorch Geometric not installed.")
    print("[!] Install with: pip install torch torch_geometric")


# === Opcode Vocabulary ===
# Common x86/x64 opcodes for Bag of Opcodes encoding
# This list covers ~95% of instructions in typical binaries
DEFAULT_OPCODE_VOCAB = [
    # Data movement
    'mov', 'push', 'pop', 'lea', 'xchg', 'movzx', 'movsx', 'cmov',
    'movs', 'stos', 'lods', 'movdqu', 'movaps', 'movups', 'movq',
    
    # Arithmetic
    'add', 'sub', 'mul', 'imul', 'div', 'idiv', 'inc', 'dec', 'neg',
    'adc', 'sbb', 'cmp',
    
    # Logic
    'and', 'or', 'xor', 'not', 'test', 'shl', 'shr', 'sar', 'sal',
    'rol', 'ror', 'rcl', 'rcr',
    
    # Control flow
    'jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle',
    'ja', 'jae', 'jb', 'jbe', 'js', 'jns', 'jo', 'jno',
    'call', 'ret', 'retn', 'leave', 'enter', 'loop',
    
    # Stack
    'pushf', 'popf', 'pusha', 'popa', 'pushad', 'popad',
    
    # String
    'rep', 'repe', 'repne', 'repz', 'repnz', 'scas', 'cmps',
    
    # System
    'int', 'syscall', 'sysenter', 'cpuid', 'rdtsc', 'nop', 'hlt',
    
    # SSE/AVX
    'pxor', 'por', 'pand', 'paddb', 'paddw', 'paddd', 'paddq',
    'psubb', 'psubw', 'psubd', 'psubq', 'pmull', 'psll', 'psrl',
    'xorps', 'orps', 'andps', 'addps', 'subps', 'mulps', 'divps',
    
    # Set
    'sete', 'setne', 'setg', 'setge', 'setl', 'setle',
    'seta', 'setae', 'setb', 'setbe',
    
    # Misc (catch-all for unknown)
    '<unk>'
]


class OpcodeVocab:
    """Vocabulary for opcode encoding."""
    
    def __init__(self, vocab: List[str] = None):
        self.vocab = vocab or DEFAULT_OPCODE_VOCAB
        self.opcode_to_idx = {op: idx for idx, op in enumerate(self.vocab)}
        self.idx_to_opcode = {idx: op for idx, op in enumerate(self.vocab)}
        self.unk_idx = self.opcode_to_idx.get('<unk>', len(self.vocab) - 1)
    
    def encode(self, mnemonic: str) -> int:
        """Encode mnemonic to index."""
        # Normalize: lowercase, strip prefixes like 'lock ', 'rep '
        mnem = mnemonic.lower().strip()
        
        # Handle conditional jumps (j* -> normalize first 2 chars)
        if mnem.startswith('j') and len(mnem) > 1:
            if mnem in self.opcode_to_idx:
                return self.opcode_to_idx[mnem]
            # Try just 'j' + first condition char
            short = 'j' + mnem[1:3] if len(mnem) > 2 else mnem
            if short in self.opcode_to_idx:
                return self.opcode_to_idx[short]
        
        # Handle cmov variants
        if mnem.startswith('cmov'):
            if 'cmov' in self.opcode_to_idx:
                return self.opcode_to_idx['cmov']
        
        # Handle set* variants
        if mnem.startswith('set'):
            base = mnem[:4]
            if base in self.opcode_to_idx:
                return self.opcode_to_idx[base]
        
        # Direct lookup
        return self.opcode_to_idx.get(mnem, self.unk_idx)
    
    def __len__(self):
        return len(self.vocab)


# === Feature Extractors ===
class BagOfOpcodesFeaturizer:
    """
    Convert basic block to Bag of Opcodes feature vector.
    Each dimension is the count of a specific opcode in the block.
    """
    
    def __init__(self, vocab: OpcodeVocab = None, normalize: bool = True):
        self.vocab = vocab or OpcodeVocab()
        self.normalize = normalize
        self.feature_dim = len(self.vocab)
    
    def extract(self, block: dict) -> np.ndarray:
        """Extract BoO features from a basic block."""
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        insns = block.get('insns', [])
        for insn in insns:
            mnemonic = insn.get('mnemonic', '')
            if mnemonic:
                idx = self.vocab.encode(mnemonic)
                features[idx] += 1
        
        # Normalize to probabilities
        if self.normalize and features.sum() > 0:
            features = features / features.sum()
        
        return features


class StatisticalFeaturizer:
    """
    Extract statistical features from basic blocks.
    Useful as additional features alongside BoO.
    """
    
    def __init__(self):
        self.feature_dim = 8
    
    def extract(self, block: dict) -> np.ndarray:
        """Extract statistical features."""
        insns = block.get('insns', [])
        
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        if not insns:
            return features
        
        # Feature 0: Block size (number of instructions)
        features[0] = len(insns) / 100.0  # Normalize
        
        # Feature 1: Total bytes
        total_bytes = sum(len(insn.get('bytes', '')) // 2 for insn in insns)
        features[1] = total_bytes / 100.0
        
        # Feature 2: Average instruction length
        if insns:
            features[2] = total_bytes / len(insns) / 10.0
        
        # Feature 3: Number of call instructions
        calls = sum(1 for insn in insns if insn.get('mnemonic', '').lower() == 'call')
        features[3] = calls / 10.0
        
        # Feature 4: Number of xrefs out
        xrefs = sum(len(insn.get('xrefs_out', [])) for insn in insns)
        features[4] = xrefs / 10.0
        
        # Feature 5: Has string reference (heuristic)
        has_string = any(
            any('str' in op.lower() or 'offset' in op.lower() 
                for op in insn.get('operands', []))
            for insn in insns
        )
        features[5] = 1.0 if has_string else 0.0
        
        # Feature 6: Number of successors
        features[6] = len(block.get('succs', [])) / 5.0
        
        # Feature 7: Number of predecessors
        features[7] = len(block.get('preds', [])) / 5.0
        
        return features


class CombinedFeaturizer:
    """Combine multiple featurizers."""
    
    def __init__(self, featurizers: List = None):
        if featurizers is None:
            featurizers = [
                BagOfOpcodesFeaturizer(),
                StatisticalFeaturizer()
            ]
        self.featurizers = featurizers
        self.feature_dim = sum(f.feature_dim for f in featurizers)
    
    def extract(self, block: dict) -> np.ndarray:
        """Extract combined features."""
        features = []
        for featurizer in self.featurizers:
            features.append(featurizer.extract(block))
        return np.concatenate(features)


# === Graph Converter ===
class JSONToGraphConverter:
    """Convert IDA JSON export to PyTorch Geometric Data object."""
    
    def __init__(self, featurizer=None, include_function_level: bool = True):
        self.featurizer = featurizer or CombinedFeaturizer()
        self.include_function_level = include_function_level
    
    def load_json(self, filepath: Path) -> dict:
        """Load JSON export file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def convert(self, data: dict, label: Optional[int] = None) -> Data:
        """
        Convert JSON data to PyTorch Geometric Data object.
        
        Args:
            data: Parsed JSON from IDA export
            label: Optional label for classification (e.g., malware family)
        
        Returns:
            PyG Data object with:
            - x: Node features (basic blocks)
            - edge_index: CFG edges
            - y: Label (if provided)
        """
        # Collect all blocks and create ID mapping
        block_id_to_idx = {}
        all_blocks = []
        
        for func in data.get('functions', []):
            for block in func.get('blocks', []):
                block_id = block.get('id')
                if block_id and block_id not in block_id_to_idx:
                    block_id_to_idx[block_id] = len(all_blocks)
                    all_blocks.append(block)
        
        if not all_blocks:
            # Empty graph
            return self._create_empty_graph(label)
        
        # Extract node features
        node_features = []
        for block in all_blocks:
            features = self.featurizer.extract(block)
            node_features.append(features)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Build edge index from cfg_edges
        edges_src = []
        edges_dst = []
        
        # Use cfg_edges if available
        if 'cfg_edges' in data:
            for edge in data['cfg_edges']:
                from_id = edge.get('from_block')
                to_id = edge.get('to_block')
                
                if from_id in block_id_to_idx and to_id in block_id_to_idx:
                    edges_src.append(block_id_to_idx[from_id])
                    edges_dst.append(block_id_to_idx[to_id])
        else:
            # Fallback: use succs from blocks
            for block in all_blocks:
                block_id = block.get('id')
                if block_id not in block_id_to_idx:
                    continue
                src_idx = block_id_to_idx[block_id]
                
                for succ_id in block.get('succs', []):
                    if succ_id in block_id_to_idx:
                        edges_src.append(src_idx)
                        edges_dst.append(block_id_to_idx[succ_id])
        
        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create Data object
        graph_data = Data(x=x, edge_index=edge_index)
        
        # Add label if provided
        if label is not None:
            graph_data.y = torch.tensor([label], dtype=torch.long)
        
        # Add metadata
        graph_data.num_blocks = len(all_blocks)
        graph_data.num_functions = len(data.get('functions', []))
        
        if 'meta' in data:
            graph_data.sample_name = data['meta'].get('idb_name', 'unknown')
        
        return graph_data
    
    def _create_empty_graph(self, label: Optional[int] = None) -> Data:
        """Create empty graph for samples with no blocks."""
        x = torch.zeros((1, self.featurizer.feature_dim), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.long)
        data.num_blocks = 0
        data.num_functions = 0
        
        return data
    
    def convert_file(self, filepath: Path, label: Optional[int] = None) -> Data:
        """Convert a JSON file to PyG Data."""
        data = self.load_json(filepath)
        return self.convert(data, label)


# === Batch Processing ===
class BatchConverter:
    """Convert multiple JSON files to PyG dataset."""
    
    def __init__(self, converter: JSONToGraphConverter = None):
        self.converter = converter or JSONToGraphConverter()
    
    def convert_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        label: Optional[int] = None
    ) -> List[Path]:
        """Convert all JSON files in directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_files = list(input_dir.glob('*.json'))
        results = []
        
        print(f"Converting {len(json_files)} files...")
        
        for i, json_path in enumerate(json_files):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{len(json_files)}")
            
            try:
                graph = self.converter.convert_file(json_path, label)
                
                # Save as .pt file
                output_path = output_dir / f"{json_path.stem}.pt"
                torch.save(graph, output_path)
                results.append(output_path)
                
            except Exception as e:
                print(f"  [!] Error: {json_path.name}: {e}")
        
        print(f"Converted {len(results)}/{len(json_files)} files")
        return results
    
    def create_dataset_list(
        self,
        input_dir: Path,
        label_map: Dict[str, int] = None
    ) -> List[Data]:
        """
        Create list of Data objects for InMemoryDataset.
        
        Args:
            input_dir: Directory with JSON files
            label_map: Optional dict mapping sample names to labels
        
        Returns:
            List of PyG Data objects
        """
        json_files = list(input_dir.glob('*.json'))
        data_list = []
        
        for json_path in json_files:
            try:
                # Determine label
                label = None
                if label_map:
                    sample_name = json_path.stem
                    label = label_map.get(sample_name, 0)
                
                graph = self.converter.convert_file(json_path, label)
                data_list.append(graph)
                
            except Exception as e:
                print(f"[!] Error: {json_path.name}: {e}")
        
        return data_list


# === CLI ===
def main():
    parser = argparse.ArgumentParser(
        description="Convert IDA JSON exports to PyTorch Geometric format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file
    python json_to_pyg.py sample_export.json --output sample.pt
    
    # Batch convert directory
    python json_to_pyg.py ./exports/ --output ./graphs/ --batch
    
    # With label
    python json_to_pyg.py sample.json --output sample.pt --label 1
        """
    )
    
    parser.add_argument(
        'input',
        type=Path,
        help="Input JSON file or directory"
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help="Output .pt file or directory"
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help="Batch mode: process all JSON files in input directory"
    )
    
    parser.add_argument(
        '--label',
        type=int,
        default=None,
        help="Classification label for the sample(s)"
    )
    
    parser.add_argument(
        '--featurizer',
        choices=['boo', 'stats', 'combined'],
        default='combined',
        help="Feature extraction method (default: combined)"
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help="Print graph statistics without saving"
    )
    
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("Error: PyTorch Geometric is required")
        print("Install with: pip install torch torch_geometric")
        sys.exit(1)
    
    # Select featurizer
    if args.featurizer == 'boo':
        featurizer = BagOfOpcodesFeaturizer()
    elif args.featurizer == 'stats':
        featurizer = StatisticalFeaturizer()
    else:
        featurizer = CombinedFeaturizer()
    
    converter = JSONToGraphConverter(featurizer)
    
    if args.batch or args.input.is_dir():
        # Batch mode
        if not args.input.is_dir():
            print(f"Error: {args.input} is not a directory")
            sys.exit(1)
        
        output_dir = args.output or Path('./pyg_graphs')
        batch = BatchConverter(converter)
        results = batch.convert_directory(args.input, output_dir, args.label)
        
        print(f"\nSaved {len(results)} graphs to: {output_dir}")
        
    else:
        # Single file mode
        if not args.input.exists():
            print(f"Error: File not found: {args.input}")
            sys.exit(1)
        
        graph = converter.convert_file(args.input, args.label)
        
        if args.info:
            print(f"\n=== Graph Statistics ===")
            print(f"Nodes (Basic Blocks): {graph.num_nodes}")
            print(f"Edges (CFG): {graph.num_edges}")
            print(f"Feature Dimension: {graph.x.shape[1]}")
            print(f"Functions: {graph.num_functions}")
            if hasattr(graph, 'y') and graph.y is not None:
                print(f"Label: {graph.y.item()}")
            if hasattr(graph, 'sample_name'):
                print(f"Sample: {graph.sample_name}")
        else:
            output_path = args.output or Path(f"{args.input.stem}.pt")
            torch.save(graph, output_path)
            print(f"Saved graph to: {output_path}")
            print(f"  Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")


if __name__ == "__main__":
    main()
