import os
import pandas as pd
import pickle as pkl

pd.set_option("display.max_columns", None)

raw_ds_path = "raw_data/datasets/"
l_ds_name = "bikedekho"
r_ds_name = "bikewale"
raw_lab_path = "raw_data/labels.csv"
ds_path = "dataset.csv"
match_path = "matches.csv"
lab_path = "labels.csv"


def prepare_dataset():
    if os.path.exists(ds_path):
        ds = pd.read_csv(ds_path)
        print(ds.dtypes)
        print(ds)

        with open("l_map.pkl", "rb") as in_file:
            l_map = pkl.load(in_file)
            in_file.close()

        with open("r_map.pkl", "rb") as in_file:
            r_map = pkl.load(in_file)
            in_file.close()

    else:
        # Load the first dataset and prepare its identifiers
        l_ds = pd.read_csv(raw_ds_path + l_ds_name + ".csv", encoding="latin")
        l_ds_len = len(l_ds)
        l_ds["_id"] = l_ds.index
        l_ds["_id"] = l_ds["_id"].apply(lambda x: l_ds_name + "_" + str(x).zfill(len(str(l_ds_len))))
        old_ids = list(l_ds["id"])
        new_ids = list(l_ds["_id"])
        l_map = {old_ids[i]: new_ids[i] for i in range(0, len(l_ds))}
        l_ds = l_ds.drop(columns=["id"])
        # print(l_ds.dtypes)
        # print(l_ds)

        # Load the second dataset and prepare its identifiers
        r_ds = pd.read_csv(raw_ds_path + r_ds_name + ".csv", encoding="latin")
        r_ds_len = len(r_ds)
        r_ds["_id"] = r_ds.index
        r_ds["_id"] = r_ds["_id"].apply(lambda x: r_ds_name + "_" + str(x).zfill(len(str(r_ds_len))))
        old_ids = list(r_ds["id"])
        new_ids = list(r_ds["_id"])
        r_map = {old_ids[i]: new_ids[i] for i in range(0, len(r_ds))}
        r_ds = r_ds.drop(columns=["id"])
        # print(r_ds.dtypes)
        # print(r_ds)

        # Append the two datasets
        ds = pd.concat([l_ds, r_ds], ignore_index=True)
        # print(ds.dtypes)
        # print(ds)

        # Rename the columns and sort them, then set the index
        ds = ds.rename(columns={"bike_name": "name", "city_posted": "city", "km_driven": "km", "fuel_type": "fuel",
                                "model_year": "year", "owner_type": "owner"})
        ds = ds[["_id", "name", "color", "fuel", "year", "km", "city", "owner", "price"]]
        ds.set_index("_id")
        print(ds.dtypes)
        print(ds)

        # Save the obtained dataset
        ds.to_csv(ds_path, index=False)

        with open("l_map.pkl", "wb") as out_file:
            pkl.dump(l_map, out_file)
            out_file.close()

        with open("r_map.pkl", "wb") as out_file:
            pkl.dump(r_map, out_file)
            out_file.close()

    return l_map, r_map


def prepare_ground_truth(l_map, r_map):
    if os.path.exists(lab_path) and os.path.exists(match_path):
        lab = pd.read_csv(lab_path)
        print(lab.dtypes)
        print(lab)
        gt = pd.read_csv(match_path)
        print(gt.dtypes)
        print(gt)

    else:
        # Load the labeled data
        lab = pd.read_csv(raw_lab_path, skiprows=5)
        # print(lab.columns)
        # print(lab)

        # Rename identifier columns
        lab = lab.rename(columns={"ltable.id": "l_id", "rtable.id": "r_id"})
        # print(lab.columns)
        # print(lab)

        # Filter out useless columns and map the identifiers
        l_old_ids = list(lab["l_id"])
        r_old_ids = list(lab["r_id"])
        labels = list(lab["gold"])
        new_ids = [(l_map[l_old_ids[i]], r_map[r_old_ids[i]], labels[i]) for i in range(0, len(lab))]
        lab = pd.DataFrame(new_ids, columns=["l_id", "r_id", "label"])
        lab = lab.reset_index(drop=True)
        print(lab.columns)
        print(lab)

        # Save the labeled data
        lab.to_csv(lab_path, index=False)

        # Keep only the matches and filter out the labels
        gt = lab[lab["label"] == 1][["l_id", "r_id"]]
        gt = gt.reset_index(drop=True)
        print(gt.columns)
        print(gt)

        # Save the obtained matches
        gt.to_csv(match_path, index=False)


if __name__ == "__main__":
    lm, rm = prepare_dataset()
    prepare_ground_truth(lm, rm)
