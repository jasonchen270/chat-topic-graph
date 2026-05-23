# chat-topic-graph

Turns your Claude Code chat history into an interactive topic map, grouped by meaning. It reads assistant responses from `~/.claude/projects/**/*.jsonl`, embeds them locally with `all-MiniLM-L6-v2`, clusters them into topics, and renders an interactive force-directed graph (vis-network) with search.

## Run

```
pip install sentence-transformers numpy
python3 build_semantic.py
python3 -m http.server 8731
# open http://localhost:8731/graph_semantic.html
```

## Config (top of build_semantic.py)

- `ROLE`: `"assistant"` (Claude's output) or `"user"` (your prompts)
- `MODEL_NAME`: embedding model; similarity thresholds auto-calibrate to whatever model you pick

Embeddings are cached in `emb_<role>_<model>.npz`, so reruns skip re-embedding. The generated graph and caches are git-ignored because they are regenerable and derived from your private chat history.
