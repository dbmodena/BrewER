import brewer
import variables as var


def parser():

    query = dict()
    query["ds"] = "camera"
    query["attributes"] = ["brand", "model", "megapixels"]
    for attribute in query["attributes"]:
        if attribute not in var.datasets[query["ds"]]["attributes"]:
            query["attributes"] = ["*"]
            break
    query["aggregation_functions"] = {"brand": "vote",
                                      "model": "vote",
                                      "megapixels": "min"}
    query["blocking_function"] = "SparkER"
    query["matching_function"] = "Ground Truth"
    query["conditions"] = [("brand", "'%dahua%'", "like"), ("model", "'%dh'", "like")]
    query["operator"] = "and"
    query["ordering_key"] = "megapixels"
    query["ordering_mode"] = "desc"

    return query


def main():
    brewer.run(parser(), "scratch")


if __name__ == "__main__":
    main()
