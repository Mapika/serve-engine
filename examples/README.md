# Examples

Realistic agent workloads that run end-to-end against a `serve-engine`
daemon. Each recipe is self-contained:

```
NN-<name>/
├── README.md          # what this recipe shows, expected output
├── setup.sh           # pulls models, starts deployments
├── client.py          # OpenAI Python SDK only — no serve-engine imports
└── sample-output.txt  # checked-in output from a real run, for verification
```

| Recipe | Demonstrates | VRAM | Models |
|---|---|---|---|
| [01-router-reasoner](01-router-reasoner/) | Multi-model auto-swap with a cost-saving routing pattern | ~6 GB | Qwen2.5-0.5B + Qwen2.5-1.5B |
| [02-rag-embed-chat](02-rag-embed-chat/) | Embeddings + chat from one daemon — RAG over serve-engine's own README | ~5 GB | bge-small-en-v1.5 + Qwen2.5-1.5B |
| [03-lora-per-task](03-lora-per-task/) | LoRA hot-load — switch task adapters without restarting the engine | ~6 GB | Qwen2.5-1.5B + 2 public LoRAs |

## Prerequisites

- `serve-engine` installed and a daemon running (`serve daemon start`).
- An admin API key (`serve key create demo --tier admin`) exported as `OPENAI_API_KEY`.
- `pip install openai` (recipe 02 also needs `faiss-cpu numpy`).

## Recommended reading order

If you're new: start with **01** — it's the cleanest demonstration of the
"many models, one endpoint" story. **02** shows heterogeneous model
families. **03** is the differentiator most other tools can't do at all.
