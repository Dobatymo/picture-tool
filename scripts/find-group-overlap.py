from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from picturetool.utils import DEFAULT_HASHDB, HashDB


def get_groups_with_same_paths(df):
    df_groups = df.reset_index().groupby("path")["group"].agg(list)
    groups = np.sort(df_groups[df_groups.str.len() > 1].explode("group").unique())
    df = df.loc[groups]
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument("--in-file", type=Path, required=True)
    parser.add_argument("--hash-db", default=DEFAULT_HASHDB)
    parser.add_argument("--print-meta", action="store_true")
    args = parser.parse_args()

    if args.in_file.suffix == ".parquet":
        df = pd.read_parquet(args.in_file)
    else:
        parser.error("Invalid file type")

    db = HashDB(args.hash_db)

    df = get_groups_with_same_paths(df)
    print(f"Found {len(df)} overlapping rows in {df.groupby('group').ngroups} groups")

    df.to_csv(args.in_file.with_suffix(".bad.csv"))

    if args.print_meta:
        for row in df.itertuples():
            print(row.Index, row.path, row.filesize, row.mod_date.value)
            db.get_latest(row.path, row.filesize, row.mod_date.value)


if __name__ == "__main__":
    main()
