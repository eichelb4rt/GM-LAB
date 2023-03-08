#!/bin/bash
# SHEBANG

render_dir="renders"
dot_dir="dot_encodings"

graph_file="$1"
graph_base="$(basename "$graph_file")"
graph_stem="${graph_base%.*}"
dot_file="$dot_dir/$graph_stem.dot"
render_output="$render_dir/$graph_stem.png"

python src/visualize.py "$graph_file" || exit 1
dot -Tpng "$dot_file" -o "$render_output" || exit 1

echo "Graph successfully rendered."
