import networkx as nx
import pandas as pd

ds_path = "../alaska_cameras/dataset.csv"
gt_path = "../alaska_cameras/matches.csv"
small_ds_path = "dataset.csv"
small_gt_path = "matches.csv"


def main():
    # Load the complete dataset and ground truth
    ds = pd.read_csv(ds_path)
    gt = pd.read_csv(gt_path)

    # Keep only the records with the price (called seeds) and their neighbors
    seeds = set(list(ds[ds["price"] > 0]["_id"]))
    edges = list(gt[(gt["l_id"].isin(seeds)) | (gt["r_id"].isin(seeds))].itertuples(index=False, name=None))
    g = nx.Graph()
    g.add_nodes_from(list(seeds))
    g.add_edges_from(edges)
    ids = set(g.nodes)
    small_ds = ds[ds["_id"].isin(ids)]
    small_ds = small_ds.reset_index(drop=True)
    print(small_ds.dtypes)
    print(small_ds)

    # Save the obtained subset
    small_ds.to_csv(small_ds_path, index=False)

    # Save the related ground truth
    small_gt = pd.DataFrame(edges, columns=["l_id", "r_id"])
    print(small_gt.dtypes)
    print(small_gt)
    small_gt.to_csv(small_gt_path, index=False)


if __name__ == "__main__":
    main()
