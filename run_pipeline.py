import subprocess 
import sys 
import time 
from pathlib import Path

STEPS = [
    ("Pull data from Neo4j", "pull_data.py"),
    ("Train GNN model", "train_gnn.py"),
    ("Evaluate model", "eval.py")
]

def run_pipeline(name: str, script: str) -> bool:

    print("\n" + "=" * 70)
    print(f"▶  {name}  ({script})")
    print("=" * 70) 

    start = time.time() 
    res = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )

    end = time.time() - start 
    if res.returncode == 0:
        print(f"✅ {name} completed in {end:.2f} seconds")
        return True 
    else:
        print(f"❌ {name} failed with return code {res.returncode}")
        return False

def main(): 
    print("\n" + "=" * 35)
    print("  FRAUD DETECTION WITH KNOWLEDGE GRAPH PIPELINE")
    print("=" * 35)

    Path('data').mkdir(exist_ok = True) 

    total_start = time.time() 

    for name, script in STEPS: 
        success = run_pipeline(name, script) 
        if not success:
            print(f"Pipeline stopped due to failure in step: {name}")
            sys.exit(1) 


    total = time.time() - total_start 
    print("\n" + "=" * 70)
    print(f"Pipeline completed successfully in {total:.2f} seconds")
    print("=" * 70)

if __name__ == "__main__":
    main()  


