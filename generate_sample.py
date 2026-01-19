import json
import time

def generate_strict_sample():
    # Synthetic sample data imitating a small malware function structure
    # Graph structure:
    # 401000 (Entry) --> 401020 (True)
    #                --> 401015 (False) --> 401030
    # 401020 --> 401030 (Merge)
    # 401030 --> Ret
    
    data = {
        "meta": {
            "exporter": "ida-json-exporter",
            "exporter_version": "1.1.0",
            "ida_version": "8.3",
            "timestamp": "2026-01-19T12:00:00Z",
            "idb_name": "malware_sample_hash_1.idb", 
            "idb_sha256": "abc1234567890abcdef1234567890abc",
            "notes": "Synthetic sample for GNN Graph Construction reference"
        },
        "functions": [
            {
                "name": "sub_401000",
                "start": "0x401000",
                "end": "0x401040",
                "size": 64,
                "blocks": [
                    {
                        "id": "401000_bb0", # Node 0
                        "start": "0x401000",
                        "end": "0x401015",
                        "insns": [
                            {"mnemonic": "push", "operands": ["ebp"], "bytes": "55"},
                            {"mnemonic": "mov", "operands": ["ebp", "esp"], "bytes": "89E5"},
                            {"mnemonic": "sub", "operands": ["esp", "0x10"], "bytes": "83EC10"},
                            {"mnemonic": "mov", "operands": ["eax", "[ebp+8]"], "bytes": "8B4508"},
                            {"mnemonic": "test", "operands": ["eax", "eax"], "bytes": "85C0"},
                            {"mnemonic": "jz", "operands": ["loc_401020"], "bytes": "7409"}
                        ],
                        "succs": ["401000_bb1", "401000_bb2"]
                    },
                    {
                        "id": "401000_bb1", # Node 1 (False branch)
                        "start": "0x401015",
                        "end": "0x401020",
                        "insns": [
                            {"mnemonic": "push", "operands": ["1"], "bytes": "6A01"},
                            {"mnemonic": "call", "operands": ["sub_402000"], "bytes": "E8..."},
                            {"mnemonic": "add", "operands": ["esp", "4"], "bytes": "83C404"}
                        ],
                        "succs": ["401000_bb3"]
                    },
                    {
                        "id": "401000_bb2", # Node 2 (True branch / Jump target coverage)
                        "start": "0x401020",
                        "end": "0x401030",
                        "insns": [
                            {"mnemonic": "xor", "operands": ["eax", "eax"], "bytes": "31C0"},
                            {"mnemonic": "lea", "operands": ["eax", "[ebp-4]"], "bytes": "8D45FC"}
                        ],
                        "succs": ["401000_bb3"]
                    },
                    {
                        "id": "401000_bb3", # Node 3 (Exit block)
                        "start": "0x401030",
                        "end": "0x401040",
                        "insns": [
                            {"mnemonic": "pop", "operands": ["ebp"], "bytes": "5D"},
                            {"mnemonic": "ret", "operands": [], "bytes": "C3"}
                        ],
                        "succs": []
                    }
                ]
            }
        ],
        "cfg_edges": [
            {"from_block": "401000_bb0", "to_block": "401000_bb1", "type": "fall_through"}, # 0 -> 1
            {"from_block": "401000_bb0", "to_block": "401000_bb2", "type": "cond_jump"},    # 0 -> 2
            {"from_block": "401000_bb1", "to_block": "401000_bb3", "type": "fall_through"}, # 1 -> 3
            {"from_block": "401000_bb2", "to_block": "401000_bb3", "type": "fall_through"}  # 2 -> 3
        ]
    }
    
    with open("sample_export.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Generated sample_export.json")

if __name__ == "__main__":
    generate_strict_sample()
