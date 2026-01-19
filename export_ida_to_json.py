#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IDA JSON Exporter v1.0
======================
Exports IDA database content (functions, blocks, instructions, xrefs, CFG)
to a structured JSON format suitable for malware analysis and visualization.

Supports:
- IDA 7.x (with fallbacks for API variations)
- Bulk JSON output (single file)
- NDJSON streaming output (one function per line, memory-efficient)

Usage in IDA:
    File -> Script file... -> select this script
    OR paste into IDA Python console

Author: Antigravity Agent
License: MIT
"""

import json
import time
import hashlib
import os
import sys
from collections import defaultdict

# === IDA API Imports with Version Detection ===
try:
    import idaapi
    import idautils
    import idc
    IDA_AVAILABLE = True
except ImportError:
    IDA_AVAILABLE = False
    print("[!] IDA API not available - running in standalone mode for testing")


# === Configuration ===
class ExporterConfig:
    """Configuration options for the exporter."""
    EXPORTER_NAME = "ida-json-exporter"
    EXPORTER_VERSION = "1.1.0"  # Updated for batch mode support
    
    # Output modes
    MODE_JSON = "json"       # Single JSON file
    MODE_NDJSON = "ndjson"   # Newline-delimited JSON (streaming)
    
    # Limits for large databases
    MAX_FUNCTIONS_BEFORE_NDJSON = 10000  # Auto-switch to NDJSON above this
    MAX_INSNS_PER_BLOCK = 5000           # Safety limit per basic block
    
    # Feature toggles
    INCLUDE_BYTES = True          # Include instruction bytes
    INCLUDE_COMMENTS = True       # Include IDA comments
    INCLUDE_GLOBAL_XREFS = False  # Disabled for batch (saves time)
    INCLUDE_CFG_EDGES = True      # Include flattened CFG edges
    
    # Batch/Headless mode settings
    OUTPUT_DIR = None             # Set via environment or argument
    AUTO_EXIT = True              # Exit IDA after export in batch mode


def is_batch_mode():
    """Check if IDA is running in batch/headless mode."""
    if not IDA_AVAILABLE:
        return False
    
    # Method 1: Check idc.BATCH (most reliable for IDA 7.x+)
    try:
        if hasattr(idc, 'BATCH') and idc.BATCH:
            return True
    except:
        pass
    
    # Method 2: Check idc.ARGV (script arguments)
    try:
        if hasattr(idc, 'ARGV') and idc.ARGV:
            return True
    except:
        pass
    
    # Method 3: Check environment variable
    if os.environ.get('IDA_BATCH_MODE', '').lower() in ('1', 'true', 'yes'):
        return True
    
    # Method 4: Check if running under idat/idat64 (no GUI)
    try:
        if hasattr(idaapi, 'cvar') and hasattr(idaapi.cvar, 'batch'):
            return bool(idaapi.cvar.batch)
    except:
        pass
    
    # Method 5: Check for -A flag indicator
    try:
        import ida_kernwin
        if not ida_kernwin.is_ida_gui_present():
            return True
    except:
        pass
    
    return False


def get_output_path():
    """Get output path from environment or generate default."""
    # Check environment variable first (set by batch_runner.py)
    env_path = os.environ.get('IDA_EXPORT_OUTPUT')
    if env_path:
        return env_path
    
    # Check for output directory
    output_dir = os.environ.get('IDA_EXPORT_DIR', '')
    
    if IDA_AVAILABLE:
        idb_name = idaapi.get_root_filename() or "unknown"
        filename = f"{idb_name}.ida_export.json"
        
        if output_dir and os.path.isdir(output_dir):
            return os.path.join(output_dir, filename)
        else:
            return os.path.join(os.path.expanduser("~"), filename)
    
    return "export.json"


# === Utility Functions ===
def hex_ea(ea):
    """Convert an effective address to hex string format."""
    return "0x{:X}".format(ea)


def get_ida_version_info():
    """Detect IDA version and available API features."""
    if not IDA_AVAILABLE:
        return {"version": "N/A", "api_level": 0}
    
    try:
        version_str = idaapi.get_ida_version()
        # Parse major.minor from version string like "7.8" or "8.3"
        parts = version_str.split('.')
        major = int(parts[0]) if parts else 7
        minor = int(parts[1]) if len(parts) > 1 else 0
        api_level = major * 100 + minor
        return {
            "version": version_str,
            "major": major,
            "minor": minor,
            "api_level": api_level
        }
    except Exception:
        return {"version": "unknown", "api_level": 700}


def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(1 << 20)  # 1MB chunks
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


# === API Compatibility Layer ===
class IDACompat:
    """
    Compatibility layer for IDA API variations across versions.
    Provides unified method calls with fallbacks.
    """
    
    def __init__(self):
        self.version_info = get_ida_version_info()
        self.api_level = self.version_info.get("api_level", 700)
    
    def get_func_name(self, ea):
        """Get function name at address."""
        try:
            return idc.get_func_name(ea) or ""
        except AttributeError:
            # Older IDA: GetFunctionName
            return idc.GetFunctionName(ea) or ""
    
    def get_operand(self, ea, n):
        """Get operand text at address, operand index n."""
        try:
            return idc.print_operand(ea, n) or ""
        except AttributeError:
            try:
                return idc.GetOpnd(ea, n) or ""
            except:
                return ""
    
    def get_mnemonic(self, ea):
        """Get instruction mnemonic at address."""
        try:
            return idc.print_insn_mnem(ea) or ""
        except AttributeError:
            try:
                return idc.GetMnem(ea) or ""
            except:
                return ""
    
    def get_bytes(self, ea, size):
        """Get bytes at address."""
        try:
            return idc.get_bytes(ea, size)
        except AttributeError:
            try:
                return idaapi.get_bytes(ea, size)
            except:
                return b""
    
    def get_item_size(self, ea):
        """Get size of item (instruction) at address."""
        try:
            return idc.get_item_size(ea)
        except AttributeError:
            return idc.ItemSize(ea)
    
    def next_head(self, ea, end):
        """Get next head (instruction/data) address."""
        try:
            return idc.next_head(ea, end)
        except AttributeError:
            return idc.NextHead(ea, end)
    
    def get_comment(self, ea, repeatable=0):
        """Get comment at address."""
        try:
            return idc.get_cmt(ea, repeatable) or ""
        except AttributeError:
            try:
                return idc.GetCommentEx(ea, repeatable) or ""
            except:
                return ""
    
    def demangle_name(self, name):
        """Demangle a C++ mangled name."""
        try:
            result = idc.demangle_name(name, idc.get_inf_attr(idc.INF_SHORT_DEMNAMES))
            return result or ""
        except:
            try:
                return idaapi.demangle_name(name, 0) or ""
            except:
                return ""
    
    def get_func_type(self, ea):
        """Get function prototype/type string."""
        try:
            tinfo = idaapi.tinfo_t()
            if idaapi.get_tinfo(tinfo, ea):
                return str(tinfo)
        except:
            pass
        # Fallback to guess
        try:
            return idc.guess_type(ea) or ""
        except:
            return ""
    
    def get_segm_end(self, seg_ea):
        """Get segment end address."""
        try:
            return idc.get_segm_end(seg_ea)
        except AttributeError:
            return idc.SegEnd(seg_ea)
    
    def get_root_filename(self):
        """Get IDB root filename."""
        try:
            return idaapi.get_root_filename() or "unknown"
        except:
            return "unknown"
    
    def get_input_file_path(self):
        """Get original input file path."""
        try:
            return idaapi.get_input_file_path() or None
        except:
            return None


# === Core Extraction Classes ===
class InstructionExtractor:
    """Extracts instruction-level data including operands and xrefs."""
    
    def __init__(self, compat, config):
        self.compat = compat
        self.config = config
    
    def extract(self, ea):
        """Extract instruction data at given address."""
        mnem = self.compat.get_mnemonic(ea)
        
        # Gather operands (up to 6)
        operands = []
        for i in range(6):
            op = self.compat.get_operand(ea, i)
            if op:
                operands.append(op)
            else:
                break
        
        # Get instruction bytes
        insn_bytes = ""
        if self.config.INCLUDE_BYTES:
            try:
                size = self.compat.get_item_size(ea)
                raw = self.compat.get_bytes(ea, size)
                if raw:
                    insn_bytes = raw.hex().upper()
            except:
                pass
        
        # Extract xrefs from this instruction
        xrefs_out = []
        try:
            for xr in idautils.XrefsFrom(ea, 0):
                xref_type = self._classify_xref(xr)
                xrefs_out.append({
                    "from": hex_ea(ea),
                    "to": hex_ea(xr.to),
                    "type": xref_type,
                    "operand_index": 0,  # Could be refined
                    "insn_ea": hex_ea(ea),
                    "is_flow": bool(xr.iscode)
                })
        except:
            pass
        
        insn_data = {
            "ea": hex_ea(ea),
            "mnemonic": mnem,
            "operands": operands,
            "bytes": insn_bytes,
            "xrefs_out": xrefs_out
        }
        
        # Add comments if enabled
        if self.config.INCLUDE_COMMENTS:
            comment = self.compat.get_comment(ea, 0)
            rep_comment = self.compat.get_comment(ea, 1)
            combined = comment
            if rep_comment:
                combined = (comment + " | " + rep_comment) if comment else rep_comment
            insn_data["comments"] = combined
        
        return insn_data
    
    def _classify_xref(self, xref):
        """Classify xref type based on IDA xref type constants."""
        if xref.iscode:
            # Code xrefs
            xtype = xref.type
            try:
                if xtype == idaapi.fl_CF or xtype == idaapi.fl_CN:
                    return "code_call"
                elif xtype == idaapi.fl_JF or xtype == idaapi.fl_JN:
                    return "code_jump"
                else:
                    return "code_flow"
            except:
                return "code"
        else:
            # Data xrefs
            try:
                xtype = xref.type
                if xtype == idaapi.dr_R:
                    return "data_read"
                elif xtype == idaapi.dr_W:
                    return "data_write"
                elif xtype == idaapi.dr_O:
                    return "data_offset"
                else:
                    return "data"
            except:
                return "data"


class BasicBlockExtractor:
    """Extracts basic block data from a function's flowchart."""
    
    def __init__(self, compat, config, insn_extractor):
        self.compat = compat
        self.config = config
        self.insn_extractor = insn_extractor
    
    def extract(self, func):
        """Extract all basic blocks from a function."""
        blocks = []
        cfg_edges = []
        
        try:
            flowchart = idaapi.FlowChart(func)
        except Exception as e:
            print(f"[!] FlowChart failed for {hex_ea(func.start_ea)}: {e}")
            return blocks, cfg_edges
        
        for bb in flowchart:
            block_id = "{:X}_bb{}".format(func.start_ea, bb.id)
            
            # Extract instructions in this block
            insns = []
            ea = bb.start_ea
            insn_count = 0
            
            while ea < bb.end_ea and ea != idaapi.BADADDR:
                if insn_count >= self.config.MAX_INSNS_PER_BLOCK:
                    break
                
                try:
                    insn_data = self.insn_extractor.extract(ea)
                    insns.append(insn_data)
                    insn_count += 1
                except Exception as e:
                    pass
                
                ea = self.compat.next_head(ea, bb.end_ea)
                if ea == idaapi.BADADDR:
                    break
            
            # Get successors and predecessors
            succs = []
            preds = []
            
            try:
                for succ in bb.succs():
                    succ_id = "{:X}_bb{}".format(func.start_ea, succ.id)
                    succs.append(succ_id)
                    
                    # Add CFG edge
                    if self.config.INCLUDE_CFG_EDGES:
                        edge_type = self._classify_edge(bb, succ)
                        cfg_edges.append({
                            "from_block": block_id,
                            "to_block": succ_id,
                            "type": edge_type
                        })
            except:
                pass
            
            try:
                for pred in bb.preds():
                    pred_id = "{:X}_bb{}".format(func.start_ea, pred.id)
                    preds.append(pred_id)
            except:
                pass
            
            blocks.append({
                "id": block_id,
                "start": hex_ea(bb.start_ea),
                "end": hex_ea(bb.end_ea),
                "insns": insns,
                "succs": succs,
                "preds": preds
            })
        
        return blocks, cfg_edges
    
    def _classify_edge(self, from_bb, to_bb):
        """Classify CFG edge type."""
        # Simple heuristic: if successor start == from_bb end, it's fall-through
        if to_bb.start_ea == from_bb.end_ea:
            return "fall_through"
        else:
            return "branch"


class FunctionExtractor:
    """Extracts complete function data including blocks and xrefs."""
    
    def __init__(self, compat, config):
        self.compat = compat
        self.config = config
        self.insn_extractor = InstructionExtractor(compat, config)
        self.block_extractor = BasicBlockExtractor(compat, config, self.insn_extractor)
    
    def extract(self, func_ea):
        """Extract complete function data."""
        func = idaapi.get_func(func_ea)
        if not func:
            return None, []
        
        fname = self.compat.get_func_name(func_ea)
        demangled = self.compat.demangle_name(fname) if fname else ""
        prototype = self.compat.get_func_type(func_ea)
        
        # Extract basic blocks and CFG edges
        blocks, cfg_edges = self.block_extractor.extract(func)
        
        # Gather call targets (functions this function calls)
        calls_out = set()
        called_by = set()
        function_xrefs = []
        
        # Get calls FROM this function
        try:
            for block in blocks:
                for insn in block.get("insns", []):
                    for xref in insn.get("xrefs_out", []):
                        if xref.get("type") == "code_call":
                            calls_out.add(xref.get("to"))
                            function_xrefs.append({
                                "from": xref.get("from"),
                                "to": xref.get("to"),
                                "type": "code_call"
                            })
        except:
            pass
        
        # Get calls TO this function (callers)
        try:
            for xr in idautils.XrefsTo(func_ea, 0):
                if xr.iscode:
                    called_by.add(hex_ea(xr.frm))
        except:
            pass
        
        func_data = {
            "name": fname,
            "start": hex_ea(func.start_ea),
            "end": hex_ea(func.end_ea),
            "size": func.size(),
            "ordinal": func.start_ea,  # Using start address as ordinal
            "demangled": demangled,
            "prototype": prototype,
            "blocks": blocks,
            "calls_out": list(calls_out),
            "called_by": list(called_by),
            "function_xrefs": function_xrefs
        }
        
        return func_data, cfg_edges


class GlobalXRefCollector:
    """Collects all cross-references from the database."""
    
    def __init__(self, compat, config):
        self.compat = compat
        self.config = config
        self.insn_extractor = InstructionExtractor(compat, config)
    
    def collect(self):
        """Collect all xrefs from all segments."""
        xrefs = []
        
        if not self.config.INCLUDE_GLOBAL_XREFS:
            return xrefs
        
        print("[*] Collecting global xrefs...")
        
        try:
            for seg_ea in idautils.Segments():
                seg_end = self.compat.get_segm_end(seg_ea)
                ea = seg_ea
                
                while ea < seg_end and ea != idaapi.BADADDR:
                    try:
                        for xr in idautils.XrefsFrom(ea, 0):
                            xref_type = self.insn_extractor._classify_xref(xr)
                            xrefs.append({
                                "from": hex_ea(ea),
                                "to": hex_ea(xr.to),
                                "type": xref_type,
                                "insn_ea": hex_ea(ea),
                                "is_flow": bool(xr.iscode)
                            })
                    except:
                        pass
                    
                    ea = self.compat.next_head(ea, seg_end)
                    if ea == idaapi.BADADDR:
                        break
        except Exception as e:
            print(f"[!] Global xref collection error: {e}")
        
        print(f"[*] Collected {len(xrefs)} global xrefs")
        return xrefs


# === Output Writers ===
class JSONWriter:
    """Writes complete JSON output."""
    
    @staticmethod
    def write(filepath, payload):
        """Write payload to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[+] Exported JSON to: {filepath}")


class NDJSONWriter:
    """Writes NDJSON (newline-delimited JSON) for streaming."""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filepath, "w", encoding="utf-8")
        return self
    
    def __exit__(self, *args):
        if self.file:
            self.file.close()
    
    def write_meta(self, meta):
        """Write metadata as first line."""
        self.file.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
    
    def write_function(self, func_data):
        """Write single function as one line."""
        self.file.write(json.dumps({"function": func_data}, ensure_ascii=False) + "\n")
    
    def write_xrefs(self, xrefs):
        """Write xrefs as one line."""
        self.file.write(json.dumps({"xrefs": xrefs}, ensure_ascii=False) + "\n")
    
    def write_cfg_edges(self, edges):
        """Write CFG edges as one line."""
        self.file.write(json.dumps({"cfg_edges": edges}, ensure_ascii=False) + "\n")


# === Main Exporter ===
class IDAExporter:
    """Main exporter orchestrating the extraction and output."""
    
    def __init__(self, config=None):
        self.config = config or ExporterConfig()
        self.compat = IDACompat()
        self.func_extractor = FunctionExtractor(self.compat, self.config)
        self.xref_collector = GlobalXRefCollector(self.compat, self.config)
    
    def build_metadata(self):
        """Build export metadata."""
        idb_path = self.compat.get_input_file_path()
        idb_hash = compute_file_hash(idb_path) if idb_path else None
        
        return {
            "exporter": self.config.EXPORTER_NAME,
            "exporter_version": self.config.EXPORTER_VERSION,
            "ida_version": self.compat.version_info.get("version", "unknown"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "idb_name": self.compat.get_root_filename(),
            "idb_sha256": idb_hash,
            "notes": ""
        }
    
    def export_json(self, outpath=None):
        """Export to single JSON file."""
        if not outpath:
            idb_name = self.compat.get_root_filename()
            outpath = os.path.join(os.path.expanduser("~"), f"{idb_name}.ida_export.json")
        
        print(f"[*] Starting JSON export to: {outpath}")
        print(f"[*] IDA Version: {self.compat.version_info}")
        
        # Build metadata
        meta = self.build_metadata()
        
        # Extract all functions
        functions = []
        all_cfg_edges = []
        
        func_eas = list(idautils.Functions())
        total = len(func_eas)
        
        # Check if we should auto-switch to NDJSON
        if total > self.config.MAX_FUNCTIONS_BEFORE_NDJSON:
            print(f"[!] Large database ({total} functions). Consider using NDJSON mode.")
            print(f"[!] Call export_ndjson() instead for memory efficiency.")
        
        for idx, func_ea in enumerate(func_eas):
            if (idx + 1) % 100 == 0:
                print(f"[*] Processing function {idx + 1}/{total}...")
            
            try:
                func_data, cfg_edges = self.func_extractor.extract(func_ea)
                if func_data:
                    functions.append(func_data)
                    all_cfg_edges.extend(cfg_edges)
            except Exception as e:
                print(f"[!] Failed to extract {hex_ea(func_ea)}: {e}")
        
        # Collect global xrefs
        xrefs = self.xref_collector.collect()
        
        # Build final payload
        payload = {
            "meta": meta,
            "functions": functions,
            "xrefs": xrefs
        }
        
        if self.config.INCLUDE_CFG_EDGES:
            payload["cfg_edges"] = all_cfg_edges
        
        # Write output
        JSONWriter.write(outpath, payload)
        
        # Print summary
        print(f"\n[+] Export complete!")
        print(f"    Functions: {len(functions)}")
        print(f"    XRefs: {len(xrefs)}")
        print(f"    CFG Edges: {len(all_cfg_edges)}")
        
        return outpath
    
    def export_ndjson(self, outpath=None):
        """Export to NDJSON (streaming, memory-efficient)."""
        if not outpath:
            idb_name = self.compat.get_root_filename()
            outpath = os.path.join(os.path.expanduser("~"), f"{idb_name}.ida_export.ndjson")
        
        print(f"[*] Starting NDJSON streaming export to: {outpath}")
        print(f"[*] IDA Version: {self.compat.version_info}")
        
        meta = self.build_metadata()
        func_eas = list(idautils.Functions())
        total = len(func_eas)
        
        all_cfg_edges = []
        func_count = 0
        
        with NDJSONWriter(outpath) as writer:
            # Write metadata first
            writer.write_meta(meta)
            
            # Stream functions
            for idx, func_ea in enumerate(func_eas):
                if (idx + 1) % 100 == 0:
                    print(f"[*] Streaming function {idx + 1}/{total}...")
                
                try:
                    func_data, cfg_edges = self.func_extractor.extract(func_ea)
                    if func_data:
                        writer.write_function(func_data)
                        func_count += 1
                        all_cfg_edges.extend(cfg_edges)
                except Exception as e:
                    print(f"[!] Failed: {hex_ea(func_ea)}: {e}")
            
            # Write xrefs
            xrefs = self.xref_collector.collect()
            writer.write_xrefs(xrefs)
            
            # Write CFG edges
            if self.config.INCLUDE_CFG_EDGES and all_cfg_edges:
                writer.write_cfg_edges(all_cfg_edges)
        
        print(f"\n[+] NDJSON export complete!")
        print(f"    Functions: {func_count}")
        print(f"    XRefs: {len(xrefs)}")
        print(f"    CFG Edges: {len(all_cfg_edges)}")
        
        return outpath


# === Entry Points ===
def export_json(outpath=None):
    """Quick export to JSON file."""
    exporter = IDAExporter()
    return exporter.export_json(outpath)


def export_ndjson(outpath=None):
    """Quick export to NDJSON (streaming) file."""
    exporter = IDAExporter()
    return exporter.export_ndjson(outpath)


def main():
    """Main entry point when run as script."""
    if not IDA_AVAILABLE:
        print("[!] This script must be run inside IDA Pro.")
        print("[*] Use: File -> Script file... or paste into IDA Python console")
        return None
    
    batch = is_batch_mode()
    
    print("=" * 60)
    print("  IDA JSON Exporter v1.1")
    print("  Exporting functions, xrefs, and CFG edges...")
    if batch:
        print("  [BATCH MODE DETECTED]")
    print("=" * 60)
    
    # Get output path (from environment or default)
    outpath = get_output_path()
    
    try:
        # Default: use JSON mode
        # For large databases, call export_ndjson() instead
        result = export_json(outpath)
        
        if batch and ExporterConfig.AUTO_EXIT:
            print("[*] Batch mode: Exiting IDA...")
            time.sleep(1)  # Wait for file buffers to flush
            idc.qexit(0)
        
        return result
        
    except Exception as e:
        print(f"[!] Export failed: {e}")
        if batch and ExporterConfig.AUTO_EXIT:
            print("[*] Batch mode: Exiting IDA with error...")
            time.sleep(1)
            idc.qexit(1)
        raise


# Run if executed directly
if __name__ == "__main__":
    main()

