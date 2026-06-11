import os
import json
import sys
import argparse
import glob

# Ensure we can import from scripts/
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
try:
    from constant import *
except ImportError:
    # Hardcoded fallbacks in case of import path issues
    CREATIVITY_LIST = ["Composition", "Diversity"]
    COMMONSENSE_LIST = ["Instance Preservation", "Motion Rationality"]
    CONTROLLABILITY_LIST = [
        "Camera Motion", "Complex Landscape", "Complex Plot", 
        "Dynamic Attribute", "Dynamic Spatial Relationship", 
        "Human Interaction", "Motion Order Understanding"
    ]
    HUMAN_FIDELITY_LIST = ["Human Anatomy", "Human Clothes", "Human Identity"]
    PHYSICS_LIST = ["Material", "Mechanics", "Multi-View Consistency", "Thermotics"]
    TASK_INFO = CREATIVITY_LIST + COMMONSENSE_LIST + CONTROLLABILITY_LIST + HUMAN_FIDELITY_LIST + PHYSICS_LIST

def get_latest_eval_results(dir_path):
    """Finds all latest eval_results.json across dimension subdirs."""
    score_dict = {}
    dimension_files = {}
    
    if not os.path.exists(dir_path):
        print(f"[ERROR] Path {dir_path} does not exist.")
        return score_dict, dimension_files

    print(f"Scanning evaluation root: {dir_path}")
    
    for name in os.listdir(dir_path):
        sub_dir = os.path.join(dir_path, name)
        if not os.path.isdir(sub_dir):
            continue
            
        # Dimension key mapped from Folder_Name (with underscore) to Display Name (with space)
        dim_name = name.replace('_', ' ')
        
        # Search for eval_results.json files in this subdirectory
        pattern = os.path.join(sub_dir, "*eval_results.json")
        eval_files = glob.glob(pattern)
        
        if not eval_files:
            continue
            
        # Sort lexicographically to get the latest date based on timestamp inside name:
        # e.g., results_2026-05-13-14:01:05_eval_results.json
        eval_files.sort()
        latest_file = eval_files[-1]
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            # Robustly locate the numerical score
            # The standard structure is { "Dimension": [ score, [ ... ] ] }
            found_score = None
            for k, v in data.items():
                if isinstance(v, list) and len(v) > 0:
                    found_score = v[0]
                    break
                    
            if found_score is not None:
                score_dict[dim_name] = float(found_score)
                dimension_files[dim_name] = os.path.basename(latest_file)
        except Exception as e:
            print(f"[WARNING] Failed to process {latest_file}: {e}")
            
    return score_dict, dimension_files

def compute_category(score_map, keys):
    scores = []
    for k in keys:
        if k in score_map:
            scores.append(score_map[k])
        else:
            # Fallback if missing
            scores.append(0.0)
    return sum(scores) / len(keys) if keys else 0.0


class _TeeStdout:
    """Mirror stdout to a UTF-8 text file (same bytes as console for typical ASCII/UTF-8 prints)."""

    def __init__(self, original, file_obj):
        self._original = original
        self._file = file_obj

    def write(self, data):
        self._original.write(data)
        self._file.write(data)
        return len(data)

    def flush(self):
        self._original.flush()
        self._file.flush()

    def isatty(self):
        return self._original.isatty()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate VBench-2.0 Dimension Scores Locally')
    parser.add_argument('--dir', type=str, 
                        default='/kwkj-k8s/YANG_LTX/workspace/RESULT/Evaluation/0512_3Wcheck_720P',
                        help='Root directory containing dimension evaluation folders')
    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Also write the full table (same as terminal) to this UTF-8 text file.',
    )
    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Optional first line(s) printed at top (terminal + report if --report set).',
    )
    args = parser.parse_args()

    report_fp = None
    original_stdout = sys.stdout
    if args.report:
        report_dir = os.path.dirname(os.path.abspath(args.report))
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        report_fp = open(args.report, 'w', encoding='utf-8')
        sys.stdout = _TeeStdout(original_stdout, report_fp)

    try:
        if args.title:
            print(args.title)
            print()

        scores, file_refs = get_latest_eval_results(args.dir)

        print("\n" + "="*80)
        print(f"{'Dimension Name':<30} | {'Score':<10} | {'Latest File Source':<40}")
        print("-"*80)

        for dim in sorted(TASK_INFO):
            val = scores.get(dim, 0.0)
            src = file_refs.get(dim, "MISSING")
            print(f"{dim:<30} | {val:<10.6f} | {src:<40}")

        print("="*80)

        creativity = compute_category(scores, CREATIVITY_LIST)
        commonsense = compute_category(scores, COMMONSENSE_LIST)
        controllability = compute_category(scores, CONTROLLABILITY_LIST)
        human_fidelity = compute_category(scores, HUMAN_FIDELITY_LIST)
        physics = compute_category(scores, PHYSICS_LIST)

        total_score = (creativity + commonsense + controllability + human_fidelity + physics) / 5.0

        print("\n" + "+----------------------------+--------------------+")
        print(f"| {'Category Name':<26} | {'Aggregate Score':<18} |")
        print("+----------------------------+--------------------+")
        print(f"| {'Creativity Score':<26} | {creativity:<18.6f} |")
        print(f"| {'Commonsense Score':<26} | {commonsense:<18.6f} |")
        print(f"| {'Controllability Score':<26} | {controllability:<18.6f} |")
        print(f"| {'Human Fidelity Score':<26} | {human_fidelity:<18.6f} |")
        print(f"| {'Physics Score':<26} | {physics:<18.6f} |")
        print("+----------------------------+--------------------+")
        print(f"| {'OVERALL FINAL SCORE':<26} | {total_score:<18.6f} |")
        print("+----------------------------+--------------------+")
        if args.report:
            print(f"\n[Report also saved to: {args.report}]")
    finally:
        if report_fp is not None:
            sys.stdout = original_stdout
            report_fp.close()
