import pandas as pd
import pickle as pkl

ds_name = "altosight_usb_sticks"
blocker = "pyjedai"
matcher = "random_forest"

ds_path = "/".join(["datasets", ds_name, "dataset.csv"])
gt_path = "/".join(["datasets", ds_name, "matches.csv"])
cand_path = "/".join(["datasets", ds_name, "blocking_functions", "candidates_" + blocker + ".pkl"])
match_path = "/".join(["datasets", ds_name, "matching_functions", "matches_" + matcher + ".csv"])


def main():
    ds = pd.read_csv(ds_path)
    ds_ids = list(ds["_id"])
    print(" ".join(["Number of records:", str(len(ds_ids))]))

    gt_matches = pd.read_csv(gt_path)
    gt_matches = {(match[0], match[1]) if match[0] < match[1] else (match[1], match[0]) for match in
                  list(gt_matches.itertuples(index=False, name=None))}

    with open(cand_path, "rb") as input_file:
        candidates = list(pkl.load(input_file))
        input_file.close()
    candidates = {(cand[0], cand[1]) if cand[0] < cand[1] else (cand[1], cand[0]) for cand in candidates}

    matches = pd.read_csv(match_path)
    matches = {(match[0], match[1]) if match[0] < match[1] else (match[1], match[0]) for match in
               list(matches.itertuples(index=False, name=None))}

    found_matches = candidates.intersection(matches)
    print(" ".join(["Number of matches:", str(len(matches))]))

    tp = len(found_matches.intersection(gt_matches))
    print(" ".join(["True positives:", str(tp)]))
    fn = len(gt_matches.difference(found_matches))
    print(" ".join(["False negatives:", str(fn)]))
    fp = len(found_matches.difference(gt_matches))
    print(" ".join(["False positives:", str(fp)]))
    precision = tp / (tp + fp)
    print(" ".join(["Precision:", str(precision)]))
    recall = tp / (tp + fn)
    print(" ".join(["Recall:", str(recall)]))


if __name__ == "__main__":
    main()
