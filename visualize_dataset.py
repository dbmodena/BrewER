import pandas as pd
pd.set_option("display.max_columns", None)

dataset = "altosight_usb_sticks_small"


def main():
    ds = pd.read_csv("datasets/" + dataset + "/dataset.csv")
    print(ds)


if __name__ == "__main__":
    main()
