import argparse
import json

import numpy as np
import pandas as pd

import clock
import graph_utils
from src.greedy_searcher import GreedySearcher
from test_likelihood import print_report


def main():
    parser = argparse.ArgumentParser(description="Learn an adjacency matrix from trainset.csv.")
    parser.add_argument("lambda_reg",
                        type=float,
                        help="Regularization constant in the objective function.")
    parser.add_argument("-c",
                        "--config",
                        default="configs/small.json",
                        help="JSON file with the greedy search parameters.")
    parser.add_argument("--trainset",
                        default="trainset.csv",
                        help="Dataset the structure will be learned with.")
    parser.add_argument("--output_name",
                        default="learned_structure",
                        help="Name (stem) of the .npy file that the adjacency matrix will be saved to.")
    parser.add_argument("--log",
                        action='store_true',
                        help="Enables logging.")
    args = parser.parse_args()
    
    # build the detective
    dataset = pd.read_csv(args.trainset).to_numpy()
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    n = dataset.shape[1]
    empty_adjacency_matrix = np.full((n, n), False)
    detective = GreedySearcher.from_config(config, empty_adjacency_matrix, args.lambda_reg, args.log)

    # search for the top structure
    clock.start("hill climb")
    print("climbing...")
    top_adjacency_matrix = detective.fit(dataset)
    clock.stop("hill climb")
    clock.print_total("hill climb")
    graph_utils.save(top_adjacency_matrix, name=args.output_name)
    print(f"learned structure saved to graphs/{args.output_name}.npy")
    if args.log:
        log_file = f"logs/{args.output_name}.npy"
        np.save(log_file, np.array((detective.score_history, detective.method_starts), dtype=object))
        print(f"logs saved to {log_file}")
    print_report(top_adjacency_matrix, dataset)


if __name__ == "__main__":
    main()
