# rageval

Code-only release for the `rageval` experiment framework used to compare
`VectorRAG` and `LightRAG` across documents of different lengths.

## Dataset location

The experiment datasets are hosted separately on Hugging Face:

- https://huggingface.co/datasets/bangwu/rageval_qa

This dataset repository includes:

- raw datasets currently available in the local workspace
- processed unified datasets
- experiment split files used in the thesis

## Expected local dataset layout

This code expects datasets under:

```text
dataset/
  DocFinQA/
  FinanceBench/
  LeCaRDv2/
  Loong/
  PeerQA/
  narrativeqa/
  processed/
```

If you only want to reproduce the final experiments, the most important processed files are:

- `dataset/processed/experiment/main_experiment.jsonl`
- `dataset/processed/experiment/stress_test.jsonl`
- `dataset/processed/experiment/ultra_long_finance.jsonl`

## Quick start

```bash
uv sync
python run_experiment.py --help
python scripts/evaluate_with_ragas.py --help
python scripts/generate_analysis_outputs.py --help
```
