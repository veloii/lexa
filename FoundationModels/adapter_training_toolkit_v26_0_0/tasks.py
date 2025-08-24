from invoke import task, Collection
import sys
import shlex
import shutil
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.resolve()
REPO_ROOT = PROJECT_ROOT.parent.resolve()

# Ensure both toolkit root and repo root are on sys.path (toolkit first so local overrides win)
for p in (PROJECT_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from utils import run_in_env
except ModuleNotFoundError as e:
    raise RuntimeError(
        "Could not import 'utils'. Expected utils.py at repository root: "
        f"{REPO_ROOT / 'utils.py'}"
    ) from e

from constants import (
    SYSTEM_MESSAGE,
    INPUT_JSON_PATH,
    TRAIN_JSONL_PATH,
    EVAL_JSONL_PATHS,
    LEARNING_RATE,
    CHECKPOINTS_BASE_DIR,
    EXPORTS_BASE_DIR,
    MAX_ENTRIES_PER_FILE 
)

ENV_NAME = "adapter_training_toolkit_v26_0_0"

def _abs(p: Path) -> Path:
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


@task
def build(c):
    """Build the toolkit (placeholder)."""
    pass  # Extend if/when needed


@task
def sample_data(c):
    """
    Generate train/eval JSONL files from INPUT_JSON_PATH into TRAIN_JSONL_PATH + EVAL_JSONL_PATHS.
    After generation, duplicate the primary eval file to any additional eval paths.
    """
    if not EVAL_JSONL_PATHS:
        raise RuntimeError("EVAL_JSONL_PATHS is empty; configure at least one path in constants.py")
    # Normalize all paths (absolute) right away
    input_path = _abs(Path(INPUT_JSON_PATH))
    train_path = _abs(Path(TRAIN_JSONL_PATH))
    primary_eval_path = _abs(Path(EVAL_JSONL_PATHS[0]))
    duplicate_eval_paths = [_abs(Path(p)) for p in EVAL_JSONL_PATHS[1:]]
    # Ensure parent dirs exist before running script
    for p in [train_path, primary_eval_path, *duplicate_eval_paths]:
        _ensure_dir(p.parent)
    script = PROJECT_ROOT / "sample_data.py"
    if not script.is_file():
        raise RuntimeError(f"sample_data.py not found at {script}")
    system_msg_arg = SYSTEM_MESSAGE
    script_arg = str(script)
    argv = [
        "python",
        script_arg,
        "--input", str(input_path),
        "--system-message", system_msg_arg,
        "--train-out", str(train_path),
        "--eval-out", str(primary_eval_path),
    ]
    
    # Add max-entries argument if specified
    if MAX_ENTRIES_PER_FILE is not None:
        argv.extend(["--max-entries", str(MAX_ENTRIES_PER_FILE)])
    
    cmd_display = shlex.join(argv)
    print("=== sample_data debug ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"script: {script}")
    print(f"input_path: {input_path} (exists={input_path.exists()})")
    print(f"train_path (target): {train_path}")
    print(f"primary_eval_path (target): {primary_eval_path}")
    if duplicate_eval_paths:
        for i, dp in enumerate(duplicate_eval_paths, 1):
            print(f"dup_eval_path[{i}]: {dp}")
    if MAX_ENTRIES_PER_FILE is not None:
        print(f"max_entries_per_file: {MAX_ENTRIES_PER_FILE:,}")
    print("Command:")
    print(cmd_display)
    print("=========================")
    start = time.time()
    result = run_in_env(c, ENV_NAME, cmd_display)
    elapsed = time.time() - start
    print(f"Command finished in {elapsed:.2f}s (ok={getattr(result,'ok',True)})")
    exists_now = primary_eval_path.exists()
    print(f"Primary eval exists after run? {exists_now} -> {primary_eval_path}")
    if not exists_now:
        parent = primary_eval_path.parent
        if parent.is_dir():
            listing = ", ".join(sorted(p.name for p in parent.iterdir()))
        else:
            listing = "<parent directory missing>"
        raise RuntimeError(
            f"Primary eval file not found after generation: {primary_eval_path}\n"
            f"Directory listing of {parent}:\n{listing}"
        )
    # Duplicate
    for dup_path in duplicate_eval_paths:
        if dup_path == primary_eval_path:
            continue
        _ensure_dir(dup_path.parent)
        shutil.copyfile(primary_eval_path, dup_path)
        print(f"Copied eval file to duplicate path: {dup_path}")
    print("sample_data task complete.")

@task(help={
    "name": "Unique run name (required).",
    "epochs": "Number of training epochs (default 5).",
    "batch_size": "Mini-batch size (default 4).",
    "force": "Allow reuse of an existing checkpoint directory (default False).",
    "max_sequence_length": "Maximum sequence length (default: model default).",
    "gradient_accumulation_steps": "Gradient accumulation steps (default 1).",
    "activation_checkpointing": "Enable activation checkpointing to save memory (default False).",
    "precision": "Training precision: f32, bf16, bf16-mixed, f16-mixed (default: bf16-mixed).",
    "compile_model": "Enable model compilation for faster training (default False).",
})
def train_adapter(c, name, epochs=5, batch_size=4, force=False,
                 max_sequence_length=None, gradient_accumulation_steps=1, 
                 activation_checkpointing=False, precision="bf16-mixed", compile_model=False):
    """
    Train the adapter model.
    Uses:
      train data: TRAIN_JSONL_PATH
      eval data: first of EVAL_JSONL_PATHS
      learning rate: LEARNING_RATE (from constants)
      checkpoint dir: CHECKPOINTS_BASE_DIR / name
    """
    if not name:
        raise RuntimeError("Parameter 'name' is required.")

    if not EVAL_JSONL_PATHS:
        raise RuntimeError("EVAL_JSONL_PATHS empty; cannot proceed.")

    train_data = _abs(Path(TRAIN_JSONL_PATH))
    eval_data = _abs(Path(EVAL_JSONL_PATHS[0]))
    checkpoints_base = _abs(Path(CHECKPOINTS_BASE_DIR))
    checkpoint_dir = checkpoints_base / name

    if checkpoint_dir.exists() and not force:
        print(f"❌ Checkpoint dir already exists: {checkpoint_dir} (use --force to reuse)")
        return
    _ensure_dir(checkpoint_dir)

    # Build command arguments
    argv = [
        "python", "-m", "src.train.train_adapter",
        "--train-data", str(train_data),
        "--eval-data", str(eval_data),
        "--epochs", str(epochs),
        "--learning-rate", str(LEARNING_RATE),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--precision", precision,
        "--checkpoint-dir", str(checkpoint_dir),
    ]
    
    # Add optional arguments
    if max_sequence_length:
        argv.extend(["--max-sequence-length", str(max_sequence_length)])
        argv.append("--fixed-sized-sequences")
    
    if activation_checkpointing:
        argv.append("--activation-checkpointing")
        
    if compile_model:
        argv.append("--compile-model")

    cmd = shlex.join(argv)

    print("=== train_adapter ===")
    print(f"name: {name}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"train_data: {train_data}")
    print(f"eval_data: {eval_data}")
    print(f"learning_rate: {LEARNING_RATE}")
    print(f"epochs: {epochs} | batch_size: {batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"activation_checkpointing: {activation_checkpointing}")
    print(f"precision: {precision}")
    print(f"max_sequence_length: {max_sequence_length}")
    print(f"compile_model: {compile_model}")
    print(f"Command: {cmd}")
    print("=====================")

    run_in_env(c, ENV_NAME, cmd)
    print("train_adapter complete.")


@task(help={
    "name": "Unique run name (must match prior train_adapter run).",
    "epochs": "Number of training epochs (default 5).",
    "batch_size": "Mini-batch size (default 4).",
    "force": "Allow training even if draft checkpoint dir already exists.",
    "max_sequence_length": "Maximum sequence length (default: model default).",
    "gradient_accumulation_steps": "Gradient accumulation steps (default 1).",
    "activation_checkpointing": "Enable activation checkpointing to save memory (default False).",
    "target_precision": "Target model precision: f32, bf16, bf16-mixed, f16-mixed (default: bf16-mixed).",
    "draft_precision": "Draft model precision: f32, bf16, bf16-mixed, f16-mixed (default: bf16-mixed).",
    "compile_target_model": "Enable target model compilation (default False).",
    "compile_draft_model": "Enable draft model compilation (default False).",
})
def train_draft_model(c, name, epochs=5, batch_size=4, force=False,
                     max_sequence_length=None, gradient_accumulation_steps=1,
                     activation_checkpointing=False, target_precision="bf16-mixed", 
                     draft_precision="bf16-mixed", compile_target_model=False, 
                     compile_draft_model=False):
    """
    Train the draft model using the adapter final checkpoint.
    Expects adapter-final.pt to exist in checkpoint dir (CHECKPOINTS_BASE_DIR / name).
    Uses:
      train data: TRAIN_JSONL_PATH
      eval data: first of EVAL_JSONL_PATHS
      learning rate: LEARNING_RATE (from constants)
      checkpoint dir: CHECKPOINTS_BASE_DIR / name
    """
    if not name:
        raise RuntimeError("Parameter 'name' is required.")

    if not EVAL_JSONL_PATHS:
        raise RuntimeError("EVAL_JSONL_PATHS empty; cannot proceed.")

    train_data = _abs(Path(TRAIN_JSONL_PATH))
    eval_data = _abs(Path(EVAL_JSONL_PATHS[0]))
    checkpoints_base = _abs(Path(CHECKPOINTS_BASE_DIR))
    checkpoint_dir = checkpoints_base / name
    adapter_final = checkpoint_dir / "adapter-final.pt"

    if not adapter_final.exists():
        raise RuntimeError(f"Required adapter checkpoint missing: {adapter_final}")

    if checkpoint_dir.exists() and not force:
        # Still proceed; we just warn (directory shared)
        print(f"ℹ️ Reusing existing checkpoint directory: {checkpoint_dir}")

    # Build command arguments
    argv = [
        "python", "-m", "src.train.train_draft_model",
        "--checkpoint", str(adapter_final),
        "--train-data", str(train_data),
        "--eval-data", str(eval_data),
        "--epochs", str(epochs),
        "--learning-rate", str(LEARNING_RATE),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--target-precision", target_precision,
        "--draft-precision", draft_precision,
        "--checkpoint-dir", str(checkpoint_dir),
    ]
    
    # Add optional arguments
    if max_sequence_length:
        argv.extend(["--max-sequence-length", str(max_sequence_length)])
        argv.append("--fixed-sized-sequences")
    
    if activation_checkpointing:
        argv.append("--activation-checkpointing")

    cmd = shlex.join(argv)

    print("=== train_draft_model ===")
    print(f"name: {name}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adapter_final: {adapter_final}")
    print(f"train_data: {train_data}")
    print(f"eval_data: {eval_data}")
    print(f"learning_rate: {LEARNING_RATE}")
    print(f"epochs: {epochs} | batch_size: {batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"activation_checkpointing: {activation_checkpointing}")
    print(f"target_precision: {target_precision}")
    print(f"draft_precision: {draft_precision}")
    print(f"max_sequence_length: {max_sequence_length}")
    print(f"compile_target_model: {compile_target_model}")
    print(f"compile_draft_model: {compile_draft_model}")
    print(f"Command: {cmd}")
    print("========================")

    run_in_env(c, ENV_NAME, cmd)
    print("train_draft_model complete.")


@task(help={
    "name": "Run name used for locating checkpoints.",
    "prompt": "Prompt text to generate from.",
})
def generate(c, name, prompt):
    """
    Generate output using trained adapter + draft model.
    """
    if not name:
        raise RuntimeError("Parameter 'name' is required.")
    if not prompt:
        raise RuntimeError("Parameter 'prompt' is required.")

    checkpoints_base = _abs(Path(CHECKPOINTS_BASE_DIR))
    checkpoint_dir = checkpoints_base / name
    adapter_final = checkpoint_dir / "adapter-final.pt"
    draft_final = checkpoint_dir / "draft-model-final.pt"

    for p in [adapter_final, draft_final]:
        if not p.exists():
            raise RuntimeError(f"Missing required checkpoint: {p}")

    argv = [
        "python", "-m", "src.train.generate",
        "--prompt", prompt,
        "--checkpoint", str(adapter_final),
        "--draft-checkpoint", str(draft_final),
    ]
    # Need careful quoting because of prompt
    cmd = shlex.join(argv)

    print("=== generate ===")
    print(f"name: {name}")
    print(f"adapter_final: {adapter_final}")
    print(f"draft_final: {draft_final}")
    print(f"prompt: {prompt}")
    print(f"Command: {cmd}")
    print("===============")

    run_in_env(c, ENV_NAME, cmd)
    print("generate complete.")



@task(help={
    "name": "Run name used for locating checkpoints & naming the adapter.",
    "force": "Overwrite export directory if it exists.",
})
def export(c, name, force=False):
    """
    Export the FM adapter + draft model artifacts.
    Produces outputs in EXPORTS_BASE_DIR / name
    """
    if not name:
        raise RuntimeError("Parameter 'name' is required.")

    checkpoints_base = _abs(Path(CHECKPOINTS_BASE_DIR))
    exports_base = _abs(Path(EXPORTS_BASE_DIR))
    checkpoint_dir = checkpoints_base / name
    export_dir = exports_base / name

    adapter_final = checkpoint_dir / "adapter-final.pt"
    draft_final = checkpoint_dir / "draft-model-final.pt"

    for p in [adapter_final, draft_final]:
        if not p.exists():
            raise RuntimeError(f"Missing required checkpoint: {p}")

    if export_dir.exists():
        if not force:
            print(f"❌ Export dir already exists: {export_dir} (use --force to overwrite)")
            return
    _ensure_dir(export_dir)

    argv = [
        "python", "-m", "src.export.export_fmadapter",
        "--adapter-name", name,
        "--checkpoint", str(adapter_final),
        "--draft-checkpoint", str(draft_final),
        "--output-dir", str(export_dir),
    ]
    cmd = shlex.join(argv)

    print("=== export ===")
    print(f"name: {name}")
    print(f"adapter_final: {adapter_final}")
    print(f"draft_final: {draft_final}")
    print(f"export_dir: {export_dir}")
    print(f"Command: {cmd}")
    print("=============")

    run_in_env(c, ENV_NAME, cmd)
    print("export complete.")


ns = Collection(
    build,
    sample_data,
    train_adapter,
    train_draft_model,
    generate,
    export,
)
