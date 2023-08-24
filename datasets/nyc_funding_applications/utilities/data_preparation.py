import itertools as it
import os
import pandas as pd
import pickle as pkl

pd.set_option("display.max_columns", None)

raw_ds_path = "../nyc_funding_applications/raw_data/dataset.csv"
ds_name = "funding"
ds_path = "dataset.csv"
match_path = "matches.csv"


def prepare_dataset():
    if os.path.exists(ds_path):
        ds = pd.read_csv(ds_path)
        print(ds.dtypes)
        print(ds)

        with open("mapping.pkl", "rb") as in_file:
            mapping = pkl.load(in_file)
            in_file.close()

    else:
        # Load the dataset and prepare its identifiers
        ds = pd.read_csv(raw_ds_path, encoding="latin")
        ds = ds[ds["cluster_id"].notna()]
        ds = ds[ds["cluster_id"] != "136400434"]
        ds = ds.reset_index(drop=True)
        ds_len = len(ds)
        ds["_id"] = ds.index
        ds["_id"] = ds["_id"].apply(lambda x: ds_name + "_" + str(x).zfill(len(str(ds_len))))
        ids = ds["_id"]
        clusters = ds["cluster_id"]
        mapping = {ids[i]: clusters[i] for i in range(0, ds_len)}
        # print(ds.dtypes)
        # print(ds)

        # Rename the columns and sort them, then set the index
        ds = ds.rename(columns={"Legal Name ": "name", "Fiscal Year ": "year", "Agency ": "agency", "Source ": "source",
                                "Council Member ": "counselor", "Amount ": "amount", "Status ": "status"})
        ds = ds[["_id", "name", "address", "year", "agency", "source", "counselor", "amount", "status"]]
        ds.set_index("_id")
        # print(ds.dtypes)
        # print(ds)

        # Preprocess numeric columns
        ds = ds.astype({"year": "int32"})
        ds["amount"] = ds["amount"].str.lstrip("$")
        ds["amount"] = pd.to_numeric(ds["amount"], errors="coerce")
        print(ds.dtypes)
        print(ds)

        # Save the obtained dataset
        ds.to_csv(ds_path, index=False)

        with open("mapping.pkl", "wb") as out_file:
            pkl.dump(mapping, out_file)
            out_file.close()

    return mapping


def prepare_ground_truth(mapping):
    if os.path.exists(match_path):
        gt = pd.read_csv(match_path)
        print(gt.dtypes)
        print(gt)

    else:
        # Define the clusters
        clusters = dict()
        for mp in mapping.items():
            if mp[1] in clusters.keys():
                clusters[mp[1]].append(mp[0])
            else:
                clusters[mp[1]] = [mp[0]]

        # Compute the matches
        matches = list()
        for cl in clusters.items():
            if len(cl[1]) > 1:
                for mp in list(it.combinations(cl[1], 2)):
                    matches.append(mp)

        # Load the matches into a dataframe
        gt = pd.DataFrame(matches, columns=["l_id", "r_id"])
        print(gt.columns)
        print(gt)

        # Save the obtained matches
        gt.to_csv(match_path, index=False)


if __name__ == "__main__":
    m = prepare_dataset()
    prepare_ground_truth(m)
