# IDA JSON Exporter Suite

A comprehensive toolkit for exporting IDA Pro database content to structured JSON format, suitable for malware analysis, visualization tools, and data archival.

## Quick Start

### Running in IDA Pro

1. Open your IDB in IDA Pro 7.x+
2. **File → Script file...** → select `export_ida_to_json.py`
3. Output saved to your home directory as `<idb_name>.ida_export.json`

```python
# Or paste in IDA Python console:
exec(open(r"d:\examinate\json\export_ida_to_json.py").read())
```

### Output Modes

| Mode | Function | Use Case |
|------|----------|----------|
| **JSON** | `export_json()` | Small/medium IDBs, single file output |
| **NDJSON** | `export_ndjson()` | Large IDBs (10k+ functions), streaming |

```python
# Force NDJSON for large databases:
from export_ida_to_json import export_ndjson
export_ndjson("my_large_sample.ndjson")
```

## Files

| File | Description |
|------|-------------|
| `export_ida_to_json.py` | Main IDA exporter script |
| `validate_export.py` | Schema validator |
| `package_export.py` | Archive packager with signing |
| `ida_export_schema.json` | JSON Schema Draft-07 |
| `sample_export.json` | Example output |

### Dataset Reference

The [`sample_export.json`](sample_export.json) file serves as a **template for your malware dataset**. It demonstrates:

- **2 functions** with realistic x86 disassembly
- **7 basic blocks** showing CFG structure
- **26 instructions** with opcodes, operands, and bytes
- **6 xrefs** (code calls, jumps, data reads/writes)
- **7 CFG edges** for graph construction

Use this as a reference when building your GNN training pipeline.

## Validation

```bash
python validate_export.py sample_export.json
```

**Expected output:**
```
============================================================
  IDA Export Validation Report
============================================================

[Metadata]
  Exporter: ida-json-exporter v1.0.0
  IDA Version: 8.3
  ...

[Statistics]
  Functions: 2
  Basic Blocks: 7
  Instructions: 26
  ...

============================================================
  ✓ Schema validation PASSED
============================================================
```

## Packaging

Create a tamper-evident archive:

```bash
# Basic package
python package_export.py sample_export.json

# With Ed25519 signature (requires: pip install pynacl)
python package_export.py sample_export.json --sign --keyfile mykey.json
```

**Archive structure:**
```
sample_export-20260119.ida_export.tar.gz
├── manifest.json      # File hashes, metadata
├── provenance.json    # Extraction environment info
├── data/
│   └── sample_export.json
└── signature.sig      # Ed25519 signature (optional)
```

## JSON Schema Overview

```
{
  "meta": { ... },           // Export metadata & provenance
  "functions": [             // Array of function objects
    {
      "name": "sub_401000",
      "start": "0x401000",
      "end": "0x4010F0",
      "size": 240,
      "blocks": [ ... ],     // Basic blocks with instructions
      "calls_out": [...],    // Functions called
      "called_by": [...]     // Callers
    }
  ],
  "xrefs": [ ... ],          // Global cross-reference list
  "cfg_edges": [ ... ]       // Flattened CFG for visualization
}
```

## Configuration

Edit `ExporterConfig` class in `export_ida_to_json.py`:

```python
class ExporterConfig:
    INCLUDE_BYTES = True          # Instruction bytes
    INCLUDE_COMMENTS = True       # IDA comments
    INCLUDE_GLOBAL_XREFS = True   # Global xref list
    INCLUDE_CFG_EDGES = True      # CFG edge list
    MAX_FUNCTIONS_BEFORE_NDJSON = 10000  # Auto-switch threshold
```

## IDA Version Compatibility

Tested on IDA 7.x and 8.x. The script includes a compatibility layer (`IDACompat`) that handles API variations between versions:

- Automatic API detection
- Fallbacks for renamed functions
- Safe operation on unknown versions

## Integration Examples

### Load in Python
```python
import json
with open("export.json") as f:
    data = json.load(f)
    
for func in data["functions"]:
    print(f"{func['name']}: {len(func['blocks'])} blocks")
```

### Stream NDJSON
```python
import json
with open("export.ndjson") as f:
    for line in f:
        obj = json.loads(line)
        if "function" in obj:
            print(obj["function"]["name"])
```

### Build call graph
```python
import networkx as nx
G = nx.DiGraph()

for func in data["functions"]:
    for target in func["calls_out"]:
        G.add_edge(func["start"], target)
```

---

## GNN Pipeline (v1.1)

### Batch Processing 10,000+ Samples

```bash
# Install dependencies
pip install tqdm

# Run batch processor
python batch_runner.py \
    --samples-dir ./malware \
    --ida-path "C:/Program Files/IDA Pro 8.3/idat64.exe" \
    --output-dir ./exports \
    --workers 8
```

**Features:**
- Parallel IDA instances (configurable workers)
- Auto-skip processed samples
- Progress bar (tqdm)
- Error logging

### Convert to PyTorch Geometric

```bash
# Install PyTorch Geometric
pip install torch torch_geometric

# Single file
python json_to_pyg.py export.json --output graph.pt --info

# Batch convert
python json_to_pyg.py ./exports/ --output ./graphs/ --batch
```

**Node Features (Basic Blocks):**
- Bag of Opcodes (95 common x86/x64 mnemonics)
- Statistical features (block size, call count, xrefs)

---

## Graph Construction Deep Dive

This pipeline implements a scientifically rigorous **Control Flow Graph (CFG)** extraction for GNNs.

### 1. Data Structure (`sample_export.json`)
The JSON structure maps directly to GNN components:

```json
{
  "blocks": [
    { "id": "bb0", "insns": [{"mnemonic": "push"}, {"mnemonic": "mov"}] },  // Node 0
    { "id": "bb1", "insns": [{"mnemonic": "call"}] },                        // Node 1
    { "id": "bb2", "insns": [{"mnemonic": "xor"}] }                          // Node 2
  ],
  "cfg_edges": [
    { "from_block": "bb0", "to_block": "bb1" },  // Edge 0->1
    { "from_block": "bb0", "to_block": "bb2" }   // Edge 0->2
  ]
}
```

### 2. Node Features (X)
We use a **Bag-of-Opcodes** approach (10 dimensions) as specified for high-performance malware detection.
Each basic block becomes a node feature vector counting opcode occurrences.

**Strict Opcode Mapping (dim=10):**
0: `mov` (Data)    | 1: `push` (Stack) | 2: `call` (Flow) | 3: `add` (Arith)
4: `jmp` (Jump) | 5: `test` (Logic) | 6: `lea` (Addr)  | 7: `pop` (Stack)
8: `ret` (Exit) | 9: `other` (Misc)

**Example from Sample:**
- **Node 0 (`bb0`)**: `push`, `mov`, `sub`, `mov`, `test`, `jz`
  - Vector: `[2, 1, 0, 1, 1, 1, 0, 0, 0, 0]`
  - (2 movs, 1 push, 1 sub->add, 1 test, 1 jz->jump)

### 3. Edge Index (A)
We map string IDs (`bb0`) to integer indices (`0`) to create the Sparse Edge Tensor (COO format).

**Structure:**
- `from_block` -> Source Node Index
- `to_block`   -> Target Node Index

**Resulting Tensor:**
```python
edge_index = tensor([
    [0, 0, 1, 2],  # Source Nodes
    [1, 2, 3, 3]   # Target Nodes
])
```

### 4. Labels (y)
Extracted from filename: `hash_1.json` -> `y = [1]` (Malware).

---


## License

MIT

