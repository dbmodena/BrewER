import json
import math
import numpy as np
import os
import pandas as pd
import pickle as pkl
import pprint
import random
import time
import variables as var


class Task:
    """
    The object representing the query driving the entity resolution on-demand process
    """

    def __init__(self, query):

        # FROM clause
        self.ds = query["ds"]
        self.ds_path = var.datasets[self.ds]["ds_path"]

        # SELECT clause
        self.top_k = query["top_k"]

        # Define for every attribute the aggregation function to be used in data fusion
        self.aggregation_functions = query["aggregation_functions"]

        # Define the attributes to be shown in the final result
        if query["attributes"][0] == "*":
            self.attributes = var.datasets[self.ds]["attributes"]
        else:
            self.attributes = query["attributes"]

        # Blocking function
        self.blocking_function = query["blocking_function"]
        self.candidates_path = var.datasets[self.ds]["blocking_functions"][self.blocking_function]["candidates_path"]
        self.blocks_path = var.datasets[self.ds]["blocking_functions"][self.blocking_function]["blocks_path"]

        # Matching function
        self.matching_function = query["matching_function"]
        self.gold_path = var.datasets[self.ds]["matching_functions"][self.matching_function]["gold_path"]

        # HAVING clause
        self.having = [self.clause_generator(clause_tokens) for clause_tokens in query["conditions"]]
        self.operator = query["operator"]

        # ORDER BY clause
        self.ordering_key = query["ordering_key"]
        self.ordering_mode = query["ordering_mode"]

    @staticmethod
    def clause_generator(clause_tokens):
        if clause_tokens[2] in ['!=', '>', '>=', '<', '<=']:
            return clause_tokens[0] + ' ' + clause_tokens[2] + ' ' + clause_tokens[1]
        elif clause_tokens[2] == '=':
            return clause_tokens[0] + " == " + clause_tokens[1]
        elif clause_tokens[2] == "like":
            return clause_tokens[0] + ".str.contains(" + clause_tokens[1].replace('%', '') + ", na=False)"


def blocking(blocking_function, candidates_path):
    """
    Perform the blocking step: if no blocking function has been defined, perform the Cartesian product on all records;
    otherwise, load the candidate pairs previously obtained on the dataset using the selected blocking function
    :param blocking_function: the selected blocking function
    :param candidates_path: the path of the pickle file containing the candidate pairs for that blocking function
    :return: the set of the candidate pairs to be compared in the matching step
    """

    if blocking_function == "None (Cartesian Product)":
        candidates = None  # perform the Cartesian product on all records - not needed for the demo
    else:
        candidates = set(pkl.load(open(candidates_path, "rb")))

    return candidates


def matching(left_id, right_id, gold):
    """
    Perform the matching step: check if the current candidate pair is present or not in the list of matches previously
    obtained on the dataset using the selected matching function
    :param left_id: the identifier of the left record in the current candidate pair
    :param right_id: the identifier of the right record in the current candidate pair
    :param gold: the list of the matches obtained using the selected matching function
    :return: a boolean value denoting if the current candidate pair is a match or not
    """

    return (left_id, right_id) in gold or (right_id, left_id) in gold


def find_matching_neighbors(current_id, neighborhood, neighbors, matches, done, compared, counter, gold):
    """
    Find all the matches of the current record (proceed recursively by "following the matches")
    :param current_id: the identifier of the current record
    :param neighborhood: the neighborhood of the current record
    :param neighbors: the dictionary of the neighborhoods
    :param matches: the set of the matches of the current record
    :param done: the set of the identifiers of the already solved records
    :param compared: the dictionary to keep track of the performed comparisons
    :param counter: the number of performed comparisons
    :param gold: the list of the matches obtained using the selected matching function
    :return: the updated versions of matches, compared and counter
    """

    # Look for the matches among the neighbors
    for neighbor in neighborhood:

        # Do not compare with itself and with the elements already inserted in a solved entity or already compared
        if neighbor not in matches and neighbor not in done and not compared[neighbor]:

            # Increment the comparison counter and register the neighbor as already compared
            counter += 1
            compared[neighbor] = True

            # Apply the matching function
            if matching(current_id, neighbor, gold):
                matches.add(neighbor)
                matches, compared, counter = find_matching_neighbors(neighbor, neighbors[neighbor][0].union(
                    neighbors[neighbor][1]), neighbors, matches, done, compared, counter, gold)

    return matches, compared, counter


def fusion(ds, cluster, aggregation_functions):
    """
    Perform the fusion step: locate in the dataset the matching records in the current cluster and obtain from them the
    representative record for the current entity using the selected aggregation functions
    :param ds: the dataset in the pandas DataFrame format
    :param cluster: the list of the identifiers of the matching records in the current cluster
    :param aggregation_functions: the dictionary defining the aggregation function for every attribute to be included
    in the representative record for the current entity
    :return: the representative record for the current entity in the dictionary format
    """

    # Locate in the dataset the matching records in the current cluster
    matching_records = ds.loc[ds["id"].isin(cluster)]

    # Obtain the representative record for the current entity using the selected aggregation functions
    entity = dict()
    for attribute, aggregation_function in aggregation_functions.items():
        if aggregation_function == "min":
            entity[attribute] = matching_records[attribute].min()
        elif aggregation_function == "max":
            entity[attribute] = matching_records[attribute].max()
        elif aggregation_function == "avg":
            entity[attribute] = round(matching_records[attribute].mean(), 2)
        elif aggregation_function == "sum":
            entity[attribute] = round(matching_records[attribute].sum(), 2)
        elif aggregation_function == "vote":
            try:
                entity[attribute] = matching_records[attribute].mode(dropna=False).iloc[0]
            except ValueError:
                entity[attribute] = np.random.choice(matching_records[attribute])  # should be only among the tied ones
        elif aggregation_function == "random":
            entity[attribute] = np.random.choice(matching_records[attribute])
        elif aggregation_function == "concat":
            entity[attribute] = " ; ".join(matching_records[attribute])

    return entity


def pre_filtering(task, block_records, solved):
    """
    Detect the seed records inside every transitively closed block to perform the preliminary filtering of the blocks
    :param task: the object representing the query driving the entity resolution on-demand process
    :param block_records: the records in the current block in the pandas DataFrame format
    :param solved: a boolean value stating if the block already contains only one record (i.e., no need for ER) or not
    :return: the seed records in the current block in the pandas DataFrame format
    """

    # If the conditions are conjunctive, check that they are separately satisfied by at least one record in the block
    if task.operator == "and":

        # For already solved records (no neighbors), simply filter them using the conditions in and
        if solved:
            sql_statement = " and ".join(task.having)
            return block_records.query(sql_statement, engine="python")

        # Otherwise, check that all conditions are separately satisfied (if not, return an empty DataFrame)
        else:
            for clause in task.having:
                sql_statement = clause
                condition = block_records.query(sql_statement, engine="python")
                if len(condition) == 0:
                    return condition
            # If the conditions are all satisfied, proceed as in the disjunctive case

    # If the conditions are disjunctive, check that at least one of them is satisfied by the records in the block
    sql_statement = " or ".join(task.having)
    return block_records.query(sql_statement, engine="python")


def post_filtering(task, entity):
    """
    Run the query on the current entity to determine its emission
    :param task: the object representing the query driving the entity resolution on-demand process
    :param entity: the current entity which needs to be checked for its emission
    :return: a boolean value stating if the entity has to be emitted or not
    """

    entity = pd.DataFrame(entity, index=[0])
    connector = " " + task.operator + " "
    sql_statement = connector.join(task.having)

    return len(entity.query(sql_statement, engine="python")) > 0


def setup(task, ds, optimize):
    """
    Filter the transitively closed blocks and initialize the priority queue according to the query to be performed
    :param task: the object representing the query driving the entity resolution on-demand process
    :param ds: the dataset in the pandas DataFrame format
    :param optimize: a boolean value denoting if the current task can be optimized or not
    :return: the priority queue, the list of the identifiers of the seed records, and the list of the identifiers of all
    the records whose transitively closed blocks passed the filtering
    """

    priority_queue = list()  # the priority queue
    seeds = set()  # the set of the identifiers of the seed records
    filtered = set()  # the set of the identifiers of all the records whose blocks passed the filtering

    # Load the transitively closed blocks previously created from the list of candidate pairs
    blocks = pkl.load(open(task.blocks_path, "rb"))

    # Perform the preliminary filtering of the transitively closed blocks
    for block in blocks:
        block_records = ds.loc[ds["id"].isin(block)]
        solved = len(block) == 1

        # Perform preliminary filtering on the records of the block
        block_seeds = pre_filtering(task, block_records, solved)

        # Check if the block survives the filtering (i.e., if the list of seed records is not empty)
        if len(block_seeds.index) > 0:
            seeds = seeds.union(set(block_seeds["id"]))
            filtered = filtered.union(set(block_records["id"]))

            # If the task can be optimized, insert in the priority queue only the seed records
            if optimize:
                block_records = block_seeds

            # Initialize the priority queue
            for index, record in block_records.iterrows():
                element = dict()
                element["id"] = record["id"]
                element["matches"] = {record["id"]}  # the set of the identifier of the matching records
                element["ordering_key"] = float(record[task.ordering_key])  # must be a numeric value (cast to float)
                element["solved"] = solved
                priority_queue.append(element)

    print("\nSetup completed... let's go!\n")
    time.sleep(1)

    return priority_queue, seeds, filtered


def brewer(task, ds, gold, candidates, mode, results):
    """
    The BrewER algorithm to perform entity resolution on-demand
    :param task: the object representing the query driving the entity resolution on-demand process
    :param ds: the dataset in the pandas DataFrame format
    :param gold: the list of the matches obtained using the selected matching function
    :param candidates: the set of the candidate pairs to be compared in the matching step
    :param mode: the operating mode (i.e., "scratch" or "resume")
    :param results: the list of the already emitted resulting entities (for "resume" mode)
    :return: the resulting entities in the pandas DataFrame format
    """
    start_time = time.time()

    if mode == "scratch":

        # Check if the current task can be optimized
        optimize = (task.aggregation_functions[task.ordering_key], task.ordering_mode) in [("max", "asc"),
                                                                                           ("min", "desc")]

        # Initialize the priority queue through the preliminary filtering of the transitively closed blocks
        priority_queue, seeds, filtered = setup(task, ds, optimize)
        neighbors = dict()  # the dictionary to track the neighborhoods of the records
        done = set()  # the set of the identifiers of the already solved records

        # Define the neighborhoods using the list of candidate pairs (considering only filtered records)
        for candidate in candidates:

            if candidate[0] in filtered and candidate[1] in filtered:

                # If the records are not in the dictionary yet, insert them (a set for seeds and a set for non-seeds)
                for i in range(0, 2):
                    if candidate[i] not in neighbors.keys():
                        neighbors[candidate[i]] = [set(), set()]

                # Insert the records in one of the two sets
                for i in range(0, 2):
                    record = candidate[0] if i == 0 else candidate[1]
                    other = candidate[1] if i == 0 else candidate[0]
                    set_id = 0 if record in seeds else 1
                    neighbors[record][set_id].add(record)
                    neighbors[other][set_id].add(record)

    else:
        with open(var.cache_priority_queue_path, "rb") as input_file:
            priority_queue = pkl.load(input_file)
        with open(var.cache_neighbors_path, "rb") as input_file:
            neighbors = pkl.load(input_file)
        with open(var.cache_done_path, "rb") as input_file:
            done = pkl.load(input_file)

    # Perform progressive entity resolution and count the number of comparisons before each emission
    counter = 0  # the number of performed comparisons
    # results = list()  # the list of the emitted entities
    compared = {n: False for n in neighbors}  # the dictionary to keep track of the performed comparisons
    num_emitted = 0  # the counter of the newly emitted entities
    top_k = task.top_k if task.top_k > 0 else -1

    while len(priority_queue) > 0:

        # At each iteration, check the head of the priority queue
        if task.ordering_mode == "asc":
            head = min(priority_queue,
                       key=lambda x: x["ordering_key"] if not math.isnan(x["ordering_key"]) else float("inf"))
        else:
            head = max(priority_queue,
                       key=lambda x: x["ordering_key"] if not math.isnan(x["ordering_key"]) else float("-inf"))

        # If the head is already solved, generate the entity and emit it if it satisfies the query
        if head["solved"]:

            # Generate the representative record for the entity
            entity = fusion(ds, head["matches"], task.aggregation_functions)

            # Run the query on the entity
            if post_filtering(task, entity):

                # Emit the entity tracking the comparisons performed before its emission (and the elapsed time)
                entity["id"] = len(results)
                entity["comparisons"] = counter
                entity["time"] = time.time() - start_time
                # entity["matches"] = head["matches"]
                pprint.pprint(entity)
                results.append(entity)
                num_emitted += 1

            # Remove the head from the priority queue
            head_id = head["id"]
            for i in range(0, len(priority_queue)):
                if priority_queue[i]["id"] == head_id:
                    del priority_queue[i]
                    break

            # Check if the top-K query is already satisfied
            if num_emitted == top_k:
                with open(var.cache_priority_queue_path, "wb") as output_file:
                    pkl.dump(priority_queue, output_file, pkl.HIGHEST_PROTOCOL)
                with open(var.cache_neighbors_path, "wb") as output_file:
                    pkl.dump(neighbors, output_file, pkl.HIGHEST_PROTOCOL)
                with open(var.cache_done_path, "wb") as output_file:
                    pkl.dump(done, output_file, pkl.HIGHEST_PROTOCOL)
                with open(var.cache_results_path, "wb") as output_file:
                    pkl.dump(results, output_file, pkl.HIGHEST_PROTOCOL)
                return pd.DataFrame(results)

        # If the head is not solved yet, find the matching neighbours and insert a new element representing them
        else:

            # Set all the elements in compared to False
            compared = dict.fromkeys(compared, False)

            # Look for the matches among the seeds
            head["matches"], compared, counter = find_matching_neighbors(head["id"], neighbors[head["id"]][0],
                                                                         neighbors, head["matches"], done, compared,
                                                                         counter, gold)

            # Check the presence of at least a seed record among the matches
            if len(head["matches"].intersection(neighbors[head["id"]][0])) > 0:

                # Look for the matches also among the non-seeds
                head["matches"], compared, counter = find_matching_neighbors(head["id"], neighbors[head["id"]][1],
                                                                             neighbors, head["matches"], done, compared,
                                                                             counter, gold)

                # Create the representative record (the ordering key is the aggregation of the ones of the matches)
                key_aggregation = {task.ordering_key: task.aggregation_functions[task.ordering_key]}
                entity = fusion(ds, head["matches"], key_aggregation)

                # Define the new element of the priority queue representing the group of matching elements
                solved = dict()
                solved["id"] = head["id"]
                solved["matches"] = head["matches"]
                solved["ordering_key"] = float(entity[task.ordering_key])
                del neighbors[head["id"]]
                solved["solved"] = True
                solved["seed"] = True

                # Insert the matching records in the list of solved records
                done = done.union(head["matches"])

                # Delete the matching records from the priority queue
                priority_queue = [item for item in priority_queue if item["id"] not in head["matches"]]

                # Insert the priority queue the new element representing the matching records
                priority_queue.append(solved)

            # If no seed record is present, delete the current element and insert it in the list of solved records
            else:
                done = done.union(head["matches"])
                priority_queue = [item for item in priority_queue if item["id"] not in head["matches"]]

    return pd.DataFrame(results)


def run(query, mode, top_k):
    # Select the operating mode: "scratch" (i.e., from scratch) or "resume" (i.e., completing a top-k query)
    if mode not in ["scratch", "resume"]:
        mode = "scratch"
    else:
        mode = mode
    if mode == "resume":
        if not os.path.exists(var.cache_task_path) or not os.path.exists(var.cache_priority_queue_path) \
                or not os.path.exists(var.cache_neighbors_path) or not os.path.exists(var.cache_done_path) \
                or not os.path.exists(var.cache_results_path):
            mode = "scratch"

    # Define the query driving the entity resolution on-demand process
    if mode == "scratch":
        task = Task(query)
        with open(var.cache_task_path, "wb") as output_file:
            pkl.dump(task, output_file, pkl.HIGHEST_PROTOCOL)
        results = list()
    else:
        with open(var.cache_task_path, "rb") as input_file:
            task = pkl.load(input_file)
        task.top_k = top_k
        with open(var.cache_results_path, "rb") as input_file:
            results = pkl.load(input_file)

    # Load the dataset in the DataFrame format
    ds = pd.read_csv(task.ds_path)
    ds[task.ordering_key] = pd.to_numeric(ds[task.ordering_key], errors="coerce")
    for column in ds.columns:
        if ds[column].dtype == "object":
            ds[column] = ds[column].fillna("NaN")

    # If the matcher is defined as None, return the dirty results without performing ER (do not call BrewER)
    if task.matching_function == "None (Dirty)":
        if len(task.having) > 0:
            connector = " " + task.operator + " "
            sql_statement = connector.join(task.having)
            ds = ds.query(sql_statement, engine="python")
        ordering_mode = (task.ordering_mode == "asc")
        if task.top_k > 0:
            results = ds.sort_values(by=[task.ordering_key], ascending=ordering_mode).head(task.top_k)
        else:
            results = ds.sort_values(by=[task.ordering_key], ascending=ordering_mode)

    # Otherwise, better call BrewER
    else:

        # Load the ground truth in the DataFrame format and transform it into the list of the matching pairs
        gold = pd.read_csv(task.gold_path)
        gold = list(gold.itertuples(index=False, name=None))

        # Perform blocking according to the selected blocking function
        candidates = blocking(task.blocking_function, task.candidates_path)

        # Perform entity resolution on-demand using BrewER
        results = brewer(task, ds, gold, candidates, mode, results)

    if len(results.index) > 0:
        if task.matching_function == "None (Dirty)":
            attributes = task.attributes
        else:
            attributes = task.attributes + ["comparisons", "time"]
        print(results[attributes])
    else:
        print("\nNo entities satisfied the query.")

    return results
