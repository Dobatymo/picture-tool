import csv
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator, Set

import networkx as nx
import pandas as pd
from genutility.args import is_file
from genutility.file import StdoutFile


def pairs_to_groups(df: pd.DataFrame) -> Iterator[Set[str]]:
    if len(df.columns) != 2:
        raise ValueError("DataFrame is expected to have two columns (one pair of paths)")

    G = nx.Graph()
    for a, b in df.itertuples(index=False):
        G.add_edge(a, b)

    yield from nx.connected_components(G)


def main():
    parser = ArgumentParser()
    parser.add_argument("path", type=is_file)
    parser.add_argument(
        "--out",
        metavar="PATH",
        type=Path,
        default=None,
        help="Write results to file. Otherwise they are written to stdout.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.path)

    with StdoutFile(args.out, "wt", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(["group", "path"])
        for i, group in enumerate(pairs_to_groups(df)):
            for path in group:
                writer.writerow([i, path])


if __name__ == "__main__":
    main()
