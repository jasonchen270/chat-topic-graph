# chat-topic-graph

Turns your Claude Code chat history into an interactive topic map, grouped by meaning.

## How it works

1. Reads assistant responses from `~/.claude/projects/**/*.jsonl` (strips code blocks, file paths, terminal output so topics reflect intent, not noise).
2. Embeds each response locally with `all-MiniLM-L6-v2` (sentence-transformers).
3. Clusters them by cosine similarity into topics (nodes), each sized by how many responses fall in it.
4. Links topics whose embedding centroids are similar (edges).
5. Renders an interactive force-directed graph (vis-network) with a blue theme and search.

## Run

```
pip install sentence-transformers numpy
python3 build_semantic.py
python3 -m http.server 8731
# open http://localhost:8731/graph_semantic.html
```

## Config (top of build_semantic.py)

- `ROLE` — `"assistant"` (Claude's output) or `"user"` (your prompts)
- `MODEL_NAME` — embedding model; similarity thresholds auto-calibrate to whatever model you pick

Embeddings are cached in `emb_<role>_<model>.npz`, so reruns skip re-embedding. The generated graph and caches are git-ignored because they are regenerable and derived from your private chat history.
```
