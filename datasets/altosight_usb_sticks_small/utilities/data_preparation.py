import os
import pandas as pd
import pickle as pkl

pd.set_option("display.max_columns", None)

raw_ds_path = "raw_data/dataset.csv"
raw_lab_path = "raw_data/labels.csv"
ds_name = "altosight"
ds_path = "dataset.csv"
lab_path = "labels.csv"
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
        ds_len = len(ds)
        ds["_id"] = ds.index
        ds["_id"] = ds["_id"].apply(lambda x: ds_name + "_" + str(x).zfill(len(str(ds_len))))
        old_ids = list(ds["instance_id"])
        new_ids = list(ds["_id"])
        mapping = {old_ids[i]: new_ids[i] for i in range(0, len(ds))}
        ds = ds.drop(columns=["instance_id"])
        # print(ds.dtypes)
        # print(ds)

        # Filter the columns and sort them, then set the index
        ds = ds[["_id", "name", "brand", "size", "price"]]
        ds.set_index("_id")
        # print(ds.dtypes)
        # print(ds)

        # Preprocess numeric columns
        ds["size"] = ds["size"].str.rstrip(" GB")
        ds["size"] = ds["size"].apply(lambda x: "1024" if x == "1 T" else x)
        ds["size"] = ds["size"].apply(lambda x: int(x) if pd.notna(x) else x)
        ds["size"] = pd.to_numeric(ds["size"], errors="coerce")
        ds["size"] = ds["size"].astype("Int64")
        print(ds.dtypes)
        print(ds)

        # Save the obtained dataset
        ds.to_csv(ds_path, index=False)

        with open("mapping.pkl", "wb") as out_file:
            pkl.dump(mapping, out_file)
            out_file.close()

    return mapping


def prepare_ground_truth(mapping):
    if os.path.exists(lab_path) and os.path.exists(match_path):
        lab = pd.read_csv(lab_path)
        print(lab.dtypes)
        print(lab)
        gt = pd.read_csv(match_path)
        print(gt.dtypes)
        print(gt)

    else:
        # Load the labeled data
        lab = pd.read_csv(raw_lab_path)
        # print(lab.columns)
        # print(lab)

        # Rename identifier columns
        lab = lab.rename(columns={"left_instance_id": "l_id", "right_instance_id": "r_id"})
        # print(lab.columns)
        # print(lab)

        # Map the identifiers
        l_old_ids = list(lab["l_id"])
        r_old_ids = list(lab["r_id"])
        labels = list(lab["label"])
        new_ids = [(mapping[l_old_ids[i]], mapping[r_old_ids[i]], labels[i]) for i in range(0, len(lab))]
        lab = pd.DataFrame(new_ids, columns=["l_id", "r_id", "label"])
        lab = lab.reset_index(drop=True)
        print(lab.columns)
        print(lab)

        # Save the obtained labels
        lab.to_csv(lab_path, index=False)

        # Keep only the matches and filter out the labels
        gt = lab[lab["label"] == 1][["l_id", "r_id"]]
        gt = gt.reset_index(drop=True)
        print(gt.columns)
        print(gt)

        # Save the obtained matches
        gt.to_csv(match_path, index=False)


if __name__ == "__main__":
    m = prepare_dataset()
    prepare_ground_truth(m)
