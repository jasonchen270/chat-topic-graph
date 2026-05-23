# chat-topic-graph

Turns your Claude Code chat history into an interactive topic map, grouped by meaning. This Python tool reads assistant responses from `~/.claude/projects/**/*.jsonl`, embeds them locally with `all-MiniLM-L6-v2`, clusters them into topics, and renders an interactive force-directed graph (vis-network) with search.

## Prerequisites

- Python 3
- The `sentence-transformers` and `numpy` packages (`pip install sentence-transformers numpy`)

## Build & run

```
pip install sentence-transformers numpy
python3 build_semantic.py
python3 -m http.server 8731
# open http://localhost:8731/graph_semantic.html
```
