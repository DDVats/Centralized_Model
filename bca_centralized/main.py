import sys
import time


def run_step(name, fn):
    print(f"\n=== {name} ===")
    start = time.time()
    try:
        fn()
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return
    print(f"[DONE] {name} | Time: {time.time() - start:.2f}s")


def main():
    if len(sys.argv) < 2:
        print("""
Usage:
  python main.py preprocess
  python main.py split
  python main.py train_multi
  python main.py evaluate
  python main.py all
""")
        return

    cmd = sys.argv[1]

    if cmd == "preprocess":
        from utils.preprocessing import run_preprocessing
        run_step("Preprocessing", run_preprocessing)

    elif cmd == "split":
        from utils.splits_data import create_splits
        run_step("Splitting", create_splits)

    elif cmd == "train_multi":
        from training.train_multitask import train_multitask
        run_step("Multitask Training", lambda: train_multitask(use_cv=True))

    elif cmd == "evaluate":
        from training.evaluate import evaluate_multitask
        run_step("Multitask Evaluation", evaluate_multitask)

    elif cmd == "all":
        from utils.preprocessing import run_preprocessing
        from utils.splits_data import create_splits
        from training.train_multitask import train_multitask
        from training.evaluate import evaluate_multitask

        run_step("Preprocessing", run_preprocessing)
        run_step("Splitting", create_splits)
        run_step("Multitask Training", lambda: train_multitask(use_cv=True))
        run_step("Multitask Evaluation", evaluate_multitask)

    else:
        print(f"Unknown command: {cmd}")
        print("""
Commands:
  preprocess
  split
  train_multi
  evaluate
  all
""")


if __name__ == "__main__":
    main()