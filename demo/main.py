import brewer_ultimate
import variables as var


def parser():

    query = dict()
    query["ds"] = "camera"
    query["attributes"] = ["brand", "model", "mp", "price"]
    for attribute in query["attributes"]:
        if attribute not in var.datasets[query["ds"]]["attributes"]:
            query["attributes"] = ["*"]
            break
    query["aggregation_functions"] = {"brand": "vote",
                                      "model": "vote",
                                      "mp": "max",
                                      "price": "min"}
    query["blocking_function"] = "SparkER Meta-Blocking"
    query["matching_function"] = "Ground Truth"
    query["conditions"] = [("brand", "'%dahua%'", "like"), ("model", "'%dh%'", "like")]
    query["operator"] = "and"
    query["ordering_key"] = "mp"
    query["ordering_mode"] = "asc"
    query["top_k"] = -1

    # query = dict()  # this dictionary emulates the one generated through the parser in the demo notebook interface
    # query["ds"] = "usb"  # dataset
    # query["attributes"] = ["brand", "size_gb", "price"]  # attributes to be returned
    # for attribute in query["attributes"]:
    #     if attribute not in var.datasets[query["ds"]]["attributes"]:
    #         query["attributes"] = ["*"]
    #         break
    # query["aggregation_functions"] = {"brand": "vote",
    #                                   "size_gb": "avg",
    #                                   "price": "min"}
    # query["blocking_function"] = "SparkER Meta-Blocking"
    # query["matching_function"] = "Ground Truth"
    # query["conditions"] = [("brand", "'%intenso%'", "like"), ("size_gb", "16", "<=")]  # having clause
    # query["operator"] = "and"  # having clause
    # query["ordering_key"] = "price"  # order by clause
    # query["ordering_mode"] = "asc"  # order by clause
    # query["top_k"] = 3

    return query


def main():
    query = parser()
    brewer_ultimate.run(query, mode="scratch", top_k=query["top_k"])


if __name__ == "__main__":
    main()
