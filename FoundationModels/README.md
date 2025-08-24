# Foundation Models

Train adapters for Apple Foundation Models.

This directory contains all the adapter training toolkits. Each toolkit supports a specific set of OS versions. To support all devices, all of the toolkits will need to be built.

## Setup

Requires `conda`, `micromamba` or `mamba`

```bash
pip install invoke pyyaml
invoke setup-all
```

## Usage

```bash
# Run across all toolkits
invoke e2e  

# Environment management
invoke setup adapter_training_toolkit_v26_0_0
invoke status
```

## Commands
- `e2e` - Splits sample data, trains models, exports adapters

## Add New Toolkit

1. Create directory with `environment.yml`
2. Add `tasks.py` with toolkit commands
3. Run `invoke setup <toolkit-name>`

## Structure

```
FoundationModels/
├── tasks.py           # Main orchestration
├── utils.py           # Shared utilities  
└── adapter_training_toolkit_*/
    ├── environment.yml
    ├── tasks.py
    └── ...
```

Auto-detects micromamba/mamba/conda.
