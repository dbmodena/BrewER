import html
import ipywidgets as widgets
import itertools as it
import math
import numpy as np
import os
import pandas as pd
import pickle as pkl
import pprint
import re
import time
import variables as var
from IPython.display import clear_output, display, HTML


class Task:
    """
    The object representing the features of the query that drives the entity resolution on-demand process
    """

    def __init__(self, query):

        # FROM clause
        self.ds = query["ds"]
        self.ds_path = var.datasets[self.ds]["ds_path"]

        # SELECT clause
        self.top_k = query["top_k"]

        # Define for every attribute the aggregation function to be used for data fusion
        self.aggregation_functions = query["aggregation_functions"]

        # Define the projection of the attributes appearing in the final result
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
        if clause_tokens[2] in ["!=", ">", ">=", "<", "<="]:
            return clause_tokens[0] + " " + clause_tokens[2] + " " + clause_tokens[1]
        elif clause_tokens[2] == "=":
            return clause_tokens[0] + " == " + clause_tokens[1]
        elif clause_tokens[2] == "like":
            return clause_tokens[0] + ".str.contains(" + clause_tokens[1].replace("%", "") + ", na=False)"


def replace_substring(string, substring, substring_new, n):
    """
    Replace the n-th occurrence of a substring in a string
    :param string: the string where the replacement has to be performed
    :param substring: the substring to be searched in the given string
    :param substring_new: the substring used to perform the replacement
    :param n: the occurrence of the substring to be replaced
    :return: the string with the performed replacement
    """

    n_index = [match.start() for match in re.finditer(substring, string)][n]
    before = string[:n_index]
    after = string[n_index:]
    after = after.replace(substring, substring_new, 1)
    return before + after


def parser(sql):
    """
    Validate the SQL query received as input and extract its features as required by the Task object
    :param sql: the SQL query received as input
    :return: the dictionary containing the extracted query features
    """

    # Initialize the dictionary containing the extracted query features
    query = dict()
    query["complete"] = False

    # Put the query in lowercase and tokenize it, checking that the obtained list of tokens is not empty
    sql_string = sql.value.lower()
    sql_tokens = [token for token in re.split(r"\s|;|,|\n", sql_string) if token != ""]
    if len(sql_tokens) == 0:
        return query

    # Check if the query requires to perform entity resolution on-demand
    brewer_query = True if "group by entity" in sql_string else False

    # Scan the tokens to extract the query features
    i = 0

    # SELECT clause
    if sql_tokens[i] != "select":
        return query

    if i < len(sql_tokens) - 1:
        i += 1
    else:
        return query

    if sql_tokens[i] == "top":
        if i < len(sql_tokens) - 1:
            i += 1
        else:
            return query
        try:
            query["top_k"] = int(sql_tokens[i])
            if query["top_k"] <= 0:
                return query
            if i < len(sql_tokens) - 1:
                i += 1
            else:
                return query
        except ValueError:
            return query
    else:
        query["top_k"] = -1

    if sql_tokens[i] == "*":
        if brewer_query:
            return query
        else:
            attributes = ["*"]
            if i < len(sql_tokens) - 1:
                i += 1
            else:
                return query
    else:
        attributes = []
        while sql_tokens[i] != "from":
            attributes.append(sql_tokens[i])
            if i < len(sql_tokens) - 1:
                i += 1
            else:
                return query

    # FROM clause
    if sql_tokens[i] != "from":
        return query
    else:
        if i < len(sql_tokens) - 1:
            i += 1
        else:
            return query

    if sql_tokens[i] in var.datasets.keys():
        query["ds"] = sql_tokens[i]
        if i < len(sql_tokens) - 1:
            i += 1
        else:
            query["complete"] = True
    else:
        return query

    query["ordering_key"] = var.datasets[query["ds"]]["default_ordering_key"]
    query["ordering_mode"] = var.datasets[query["ds"]]["default_ordering_mode"]

    if attributes == ["*"]:
        query["attributes"] = var.datasets[query["ds"]]["attributes"]
    else:
        query["attributes"] = []
        if not brewer_query:
            for attribute in attributes:
                if attribute not in var.datasets[query["ds"]]["attributes"] or attribute in query["attributes"]:
                    query["complete"] = False
                    return query
                else:
                    query["attributes"].append(attribute)
        else:
            query["aggregation_functions"] = dict()
            for attribute in attributes:
                attribute_tokens = attribute.split("(", 1)
                if len(attribute_tokens) != 2:
                    query["complete"] = False
                    return query
                if attribute_tokens[1][:-1] not in var.datasets[query["ds"]]["attributes"] \
                        or attribute_tokens[1][:-1] in query["attributes"]:
                    query["complete"] = False
                    return query
                if attribute_tokens[0].upper() not in var.aggregation_functions:
                    query["complete"] = False
                    return query
                else:
                    query["attributes"].append(attribute_tokens[1][:-1])
                    query["aggregation_functions"][attribute_tokens[1][:-1]] = attribute_tokens[0]

    if query["complete"]:
        return query

    # GROUP BY ENTITY clause
    if brewer_query:
        if i + 5 < len(sql_tokens):
            if [sql_tokens[j] for j in range(i, i + 5)] != ["group", "by", "entity", "with", "matcher"]:
                query["complete"] = False
                return query
            if sql_tokens[i + 5] in var.datasets[query["ds"]]["pipelines"].keys():
                query["pipeline"] = sql_tokens[i + 5]
                query["blocking_function"] = var.datasets[query["ds"]]["pipelines"][query["pipeline"]][
                    "blocking_function"]
                query["matching_function"] = var.datasets[query["ds"]]["pipelines"][query["pipeline"]][
                    "matching_function"]
            else:
                query["complete"] = False
                return query
            if i + 6 < len(sql_tokens) - 1:
                i += 6
            else:
                query["complete"] = True
                return query
        else:
            query["complete"] = False
            return query
    else:
        query["aggregation_functions"] = dict()
        query["blocking_function"] = "None (Cartesian Product)"
        query["matching_function"] = "None (Dirty)"

    # ORDER BY clause
    if "order" in sql_tokens[i:]:
        index_order = sql_tokens[i:].index("order")
        where_clause = sql_tokens[i: i + index_order]
        order_by_clause = sql_tokens[i + index_order:]
    else:
        where_clause = sql_tokens[i:]
        order_by_clause = list()

    if len(order_by_clause) > 0 and len(order_by_clause) != 4:
        query["complete"] = False
        return query
    else:
        if order_by_clause[0:2] != ["order", "by"]:
            query["complete"] = False
            return query
        if order_by_clause[3] not in ["asc", "desc"]:
            query["complete"] = False
            return query
        else:
            query["ordering_mode"] = order_by_clause[3]
        if not brewer_query:
            if order_by_clause[2] in var.datasets[query["ds"]]["attributes"]:
                query["ordering_key"] = order_by_clause[2]
            else:
                query["complete"] = False
                return query
        else:
            attribute_tokens = order_by_clause[2].split("(", 1)
            if len(attribute_tokens) != 2:
                query["complete"] = False
                return query
            if attribute_tokens[0].upper() not in var.aggregation_functions:
                query["complete"] = False
                return query
            if attribute_tokens[1][:-1] not in var.datasets[query["ds"]]["attributes"]:
                query["complete"] = False
                return query
            if attribute_tokens[1][:-1] in query["aggregation_functions"].keys():
                if query["aggregation_functions"][attribute_tokens[1][:-1]] != attribute_tokens[0]:
                    query["complete"] = False
                    return query
            else:
                query["aggregation_functions"][attribute_tokens[1][:-1]] = attribute_tokens[0]
            query["ordering_key"] = attribute_tokens[1][:-1]

    # WHERE/HAVING clause
    if len(where_clause) > 0:
        if brewer_query and where_clause[0] != "having":
            query["complete"] = False
            return query
        elif not brewer_query and where_clause[0] != "where":
            query["complete"] = False
            return query

        where_clause = where_clause[1:]
        conditions = list()
        j = 0

        while j < len(where_clause):
            if where_clause[j + 1] not in ["like", "=", "!=", ">", ">=", "<", "<="]:
                query["complete"] = False
                return query
            if j + 3 < len(where_clause):
                if where_clause[j + 3] not in ["and", "or"]:
                    query["complete"] = False
                    return query
            j += 4
        j = 0
        while j < len(where_clause):
            conditions.append((where_clause[j], where_clause[j + 2], where_clause[j + 1]))
            j += 4
        if len(where_clause) > 3:
            query["operator"] = where_clause[3]
        else:
            query["operator"] = "or"

        query["conditions"] = list()
        for condition in conditions:
            if not brewer_query:
                if condition[0] not in var.datasets[query["ds"]]["attributes"]:
                    query["complete"] = False
                    return query
                else:
                    query["conditions"].append(condition)
            else:
                attribute_tokens = condition[0].split("(", 1)
                if len(attribute_tokens) != 2:
                    query["complete"] = False
                    return query
                if attribute_tokens[0].upper() not in var.aggregation_functions:
                    query["complete"] = False
                    return query
                if attribute_tokens[1][:-1] not in var.datasets[query["ds"]]["attributes"]:
                    query["complete"] = False
                    return query
                if attribute_tokens[1][:-1] in query["aggregation_functions"].keys():
                    if query["aggregation_functions"][attribute_tokens[1][:-1]] != attribute_tokens[0]:
                        query["complete"] = False
                        return query
                else:
                    query["aggregation_functions"][attribute_tokens[1][:-1]] = attribute_tokens[0]
                query["conditions"].append((attribute_tokens[1][:-1], condition[1], condition[2]))

    query["complete"] = True

    return query


def top_k_parser(sql):
    """
    Extract from the SQL query received as input the number of entities to emit (k)
    :param sql: the SQL query received as input
    :return: the number of entities to emit (k)
    """

    top_k = -1

    # Put the query in lowercase and tokenize it, checking that the obtained list of tokens is not empty
    sql_string = sql.value.lower()
    sql_tokens = [token for token in re.split(r"\s|;|,|\n", sql_string) if token != ""]
    if len(sql_tokens) == 0:
        return top_k

    # Extract the number of entities to emit (k) from the list of tokens
    alert = False  # if set to True, the current token represents the number of entities to emit (k)
    for token in sql_tokens:
        if alert:
            top_k = token
            break
        if token == "top":
            alert = True

    # Check that the extracted value is a positive integer
    try:
        top_k = int(top_k)
        if top_k <= 0:
            top_k = -1
    except ValueError:
        top_k = -1

    return top_k


def blocking(blocking_function, candidates_path, ds_ids):
    """
    Perform the blocking step: if no blocking function has been defined, compute the Cartesian product of all records;
    otherwise, load the candidate pairs previously obtained on the dataset using the selected blocking function
    :param blocking_function: the selected blocking function
    :param candidates_path: the path of the Pickle file containing the candidate pairs for that blocking function
    :param ds_ids: the list of the identifiers of all records in the dataset (to compute the Cartesian product)
    :return: the set of the candidate pairs to be compared in the matching step
    """

    if blocking_function == "None (Cartesian Product)":
        candidates = set(list(it.combinations(ds_ids, 2)))
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
    :return: a Boolean value denoting if the current candidate pair is a match
    """

    return (left_id, right_id) in gold or (right_id, left_id) in gold


def find_matching_neighbors(current_id, neighborhood, neighbors, matches, done, compared, counter, gold):
    """
    Find all the matches of the current record (proceed recursively by following the matches)
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

            # Apply the matching function and follow the match
            if matching(current_id, neighbor, gold):
                matches.add(neighbor)
                matches, compared, counter = find_matching_neighbors(neighbor, neighbors[neighbor][0].union(
                    neighbors[neighbor][1]), neighbors, matches, done, compared, counter, gold)

    return matches, compared, counter


def fusion(ds, cluster, aggregation_functions):
    """
    Perform the fusion step: locate in the dataset the matching records in the current cluster and obtain from them the
    representative record for the current entity using the selected aggregation functions
    :param ds: the dataset in the dataframe format
    :param cluster: the list of the identifiers of the matching records in the current cluster
    :param aggregation_functions: the dictionary defining the aggregation function for every attribute to be included
    in the representative record for the current entity
    :return: the representative record for the current entity in the dictionary format
    """

    # Locate in the dataset the matching records in the current cluster
    matching_records = ds.loc[ds["_id"].isin(cluster)]

    # Obtain the representative record for the current entity using the selected aggregation functions
    entity = dict()
    for attribute, aggregation_function in aggregation_functions.items():
        if aggregation_function == "min":
            entity[attribute] = matching_records[attribute].min()
        elif aggregation_function == "max":
            entity[attribute] = matching_records[attribute].max()
        elif aggregation_function == "avg":
            entity[attribute] = round(matching_records[attribute].mean(), 2)
        elif aggregation_function == "sum":  # cannot be applied to the ordering key (unbounded aggregation function)
            entity[attribute] = round(matching_records[attribute].sum(), 2)
        elif aggregation_function == "vote":
            try:
                entity[attribute] = matching_records[attribute].mode(dropna=False).iloc[0]
            except ValueError:
                entity[attribute] = np.random.choice(matching_records[attribute])
        elif aggregation_function == "random":
            entity[attribute] = np.random.choice(matching_records[attribute])
        elif aggregation_function == "concat":  # cannot be applied to the ordering key (result not numeric)
            entity[attribute] = " ; ".join(matching_records[attribute])

    return entity


def pre_filtering(task, block_records, solved):
    """
    Detect the seed records inside every transitively closed block to perform the preliminary filtering of the blocks
    :param task: the object representing the query that drives the entity resolution on-demand process
    :param block_records: the records in the current block in the dataframe format
    :param solved: a Boolean value stating if the block already contains only one record (i.e., no need for ER) or not
    :return: the seed records in the current block in the dataframe format
    """

    # If the conditions are conjunctive, check that they are separately satisfied by at least one record in the block
    if task.operator == "and":

        # For already solved records (no neighbors), simply filter them using the conditions in and
        if solved:
            sql_statement = " and ".join(task.having)
            return block_records.query(sql_statement, engine="python")

        # Otherwise, check that all conditions are separately satisfied (if not, return an empty dataframe)
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
    :param task: the object representing the query that drives the entity resolution on-demand process
    :param entity: the current entity that needs to be checked for its emission
    :return: a Boolean value stating if the entity has to be emitted or not
    """

    entity = pd.DataFrame(entity, index=[0])
    connector = " " + task.operator + " "
    sql_statement = connector.join(task.having)

    return len(entity.query(sql_statement, engine="python")) > 0


def setup(task, ds, optimize, demo):
    """
    Filter the transitively closed blocks and initialize the priority queue according to the query to be performed
    :param task: the object representing the query that drives the entity resolution on-demand process
    :param ds: the dataset in the dataframe format
    :param optimize: a Boolean value denoting if the current task can be optimized or not
    :param demo: a Boolean value denoting if the execution is happening in the demo scenario
    :return: the priority queue, the list of the identifiers of the seed records, and the list of the identifiers of all
    the records whose transitively closed blocks passed the filtering
    """

    priority_queue = list()  # the priority queue
    seeds = set()  # the set of the identifiers of the seed records
    filtered = set()  # the set of the identifiers of all the records whose blocks passed the filtering

    # Load the transitively closed blocks previously created from the list of candidate pairs
    if task.blocking_function == "None (Cartesian Product)":
        blocks = [list(ds["_id"])]
    else:
        with open(task.blocks_path, "rb") as input_file:
            blocks = pkl.load(input_file)
            input_file.close()

    # Perform the preliminary filtering of the transitively closed blocks
    for block in blocks:
        block_records = ds.loc[ds["_id"].isin(block)]
        solved = len(block) == 1

        # Perform preliminary filtering on the records of the block
        block_seeds = pre_filtering(task, block_records, solved)

        # Check if the block survives the filtering (i.e., if the list of seed records is not empty)
        if len(block_seeds.index) > 0:
            seeds = seeds.union(set(block_seeds["_id"]))
            filtered = filtered.union(set(block_records["_id"]))

            # If the task can be optimized, insert in the priority queue only the seed records
            if optimize:
                block_records = block_seeds

            # Initialize the priority queue
            for index, record in block_records.iterrows():
                element = dict()
                element["_id"] = record["_id"]
                element["matches"] = {record["_id"]}  # the set of the identifier of the matching records
                element["ordering_key"] = float(record[task.ordering_key])  # must be a numeric value (cast to float)
                element["solved"] = solved
                priority_queue.append(element)

    if demo:
        clear_output()
    print("\nSetup completed... let's go!\n")

    if demo:
        time.sleep(1)

    return priority_queue, seeds, filtered


def brewer(task, ds, gold, candidates, demo, mode, results):
    """
    The BrewER algorithm to perform entity resolution on-demand
    :param task: the object representing the query that drives the entity resolution on-demand process
    :param ds: the dataset in the dataframe format
    :param gold: the list of the matches obtained using the selected matching function
    :param candidates: the set of the candidate pairs to be compared in the matching step
    :param demo: a Boolean value denoting if the execution is happening in the demo scenario
    :param mode: the operating mode for the demo scenario (choose between "scratch" and "resume")
    :param results: the list of the already emitted resulting entities (for "resume" mode)
    :return: the dataframe containing the emitted resulting entities
    """

    start_time = time.time()

    if mode == "scratch":
        print("\nBrewER is running: setup started.")
        if demo:
            time.sleep(1)

    if mode == "resume":
        with open(var.cache_priority_queue_path, "rb") as input_file:
            priority_queue = pkl.load(input_file)
            input_file.close()
        with open(var.cache_neighbors_path, "rb") as input_file:
            neighbors = pkl.load(input_file)
            input_file.close()
        with open(var.cache_done_path, "rb") as input_file:
            done = pkl.load(input_file)
            input_file.close()
        with open(var.cache_emitted_entities_path, "rb") as input_file:
            emitted_entities = pkl.load(input_file)
            input_file.close()
        with open(var.cache_entity_indices_path, "rb") as input_file:
            entity_indices = pkl.load(input_file)
            input_file.close()

    else:

        # Check if the current task can be optimized
        optimize = (task.aggregation_functions[task.ordering_key],
                    task.ordering_mode) in [("max", "asc"), ("min", "desc")]

        # Initialize the priority queue through the preliminary filtering of the transitively closed blocks
        priority_queue, seeds, filtered = setup(task, ds, optimize, demo)
        neighbors = dict()  # the dictionary to track the neighborhoods of the records
        done = set()  # the set of the identifiers of the already solved records
        emitted_entities = list()  # the list that contains the emitted entities and their records as dictionaries
        entity_indices = list()  # the list to track the indices of the entities in the previous list

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

    # Perform progressive entity resolution and count the number of comparisons before each emission
    counter = 0  # the number of performed comparisons
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

                # Emit the entity tracking the number of comparisons performed before its emission and the elapsed time
                entity["_id"] = len(results)

                if not demo:
                    entity["comparisons"] = counter
                    entity["time"] = time.time() - start_time
                    pprint.pprint(entity)

                else:

                    # Store the emitted entities and their records in the dedicated list
                    matching_records = ds.loc[ds["_id"].isin(head["matches"])]
                    matching_records[" "] = ""
                    ascending = True if task.ordering_mode == "asc" else False
                    matching_records = matching_records.sort_values(by=[task.ordering_key], ascending=ascending)
                    entity_indices.append(len(emitted_entities))
                    entity[" "] = ""
                    emitted_entities.append(entity)
                    emitted_entities += matching_records.to_dict("records")

                    # Display the dataframe containing the emitted entities and their records
                    table = pd.DataFrame.from_records(emitted_entities)[[" ", "_id"] + task.attributes]
                    table_html = table.to_html(index=False, header=True)
                    reversed_entity_indices = list(reversed(entity_indices))
                    for idx in reversed_entity_indices:
                        table_html = replace_substring(table_html, "<tr>", '<tr class="entity">', idx)
                        table_html = replace_substring(table_html, "<td>",
                                                       "<td align='center'><b><span>∧</span></b>",
                                                       idx * (len(task.attributes) + 2))
                    table_html = table_html.replace("<tr>", '<tr class="record">')
                    table_html = var.html_format + table_html
                    clear_output()
                    print("\n")
                    display(HTML(table_html))
                    time.sleep(1)

                results.append(entity)
                num_emitted += 1

            # Remove the head from the priority queue
            head_id = head["_id"]
            for i in range(0, len(priority_queue)):
                if priority_queue[i]["_id"] == head_id:
                    del priority_queue[i]
                    break

            # Check if the top-k query is already satisfied
            if num_emitted == top_k:
                with open(var.cache_priority_queue_path, "wb") as output_file:
                    pkl.dump(priority_queue, output_file, pkl.HIGHEST_PROTOCOL)
                    output_file.close()
                with open(var.cache_neighbors_path, "wb") as output_file:
                    pkl.dump(neighbors, output_file, pkl.HIGHEST_PROTOCOL)
                    output_file.close()
                with open(var.cache_done_path, "wb") as output_file:
                    pkl.dump(done, output_file, pkl.HIGHEST_PROTOCOL)
                    output_file.close()
                with open(var.cache_results_path, "wb") as output_file:
                    pkl.dump(results, output_file, pkl.HIGHEST_PROTOCOL)
                    output_file.close()
                with open(var.cache_emitted_entities_path, "wb") as output_file:
                    pkl.dump(emitted_entities, output_file, pkl.HIGHEST_PROTOCOL)
                    output_file.close()
                with open(var.cache_entity_indices_path, "wb") as output_file:
                    pkl.dump(entity_indices, output_file, pkl.HIGHEST_PROTOCOL)
                    output_file.close()
                return pd.DataFrame(results)

        # If the head is not solved yet, find the matching neighbors and insert a new element representing them
        else:

            # Set all the elements in compared to False
            compared = dict.fromkeys(compared, False)

            # Look for the matches among the seeds
            head["matches"], compared, counter = find_matching_neighbors(head["_id"], neighbors[head["_id"]][0],
                                                                         neighbors, head["matches"], done, compared,
                                                                         counter, gold)

            # Check the presence of at least a seed record among the matches
            if len(head["matches"].intersection(neighbors[head["_id"]][0])) > 0:

                # Look for the matches also among the non-seeds
                head["matches"], compared, counter = find_matching_neighbors(head["_id"], neighbors[head["_id"]][1],
                                                                             neighbors, head["matches"], done, compared,
                                                                             counter, gold)

                # Create the representative record (the ordering key is the aggregation of the ones of the matches)
                key_aggregation = {task.ordering_key: task.aggregation_functions[task.ordering_key]}
                entity = fusion(ds, head["matches"], key_aggregation)

                # Define the new element of the priority queue representing the group of matching elements
                solved = dict()
                solved["_id"] = head["_id"]
                solved["matches"] = head["matches"]
                solved["ordering_key"] = float(entity[task.ordering_key])
                del neighbors[head["_id"]]
                solved["solved"] = True
                solved["seed"] = True

                # Insert the matching records in the list of solved records
                done = done.union(head["matches"])

                # Delete the matching records from the priority queue
                priority_queue = [item for item in priority_queue if item["_id"] not in head["matches"]]

                # Insert the new element representing the matching records in the priority queue
                priority_queue.append(solved)

            # If no seed record is present, delete the current element and insert it in the list of solved records
            else:
                done = done.union(head["matches"])
                priority_queue = [item for item in priority_queue if item["_id"] not in head["matches"]]

    return pd.DataFrame(results)


def run(query, demo=False, mode="scratch", top_k=-1):
    """
    Run BrewER to perform entity resolution on-demand according to the query defined by the user
    :param query: the dictionary containing the features of the query, as extracted by the parser
    :param demo: a Boolean value denoting if the execution is happening in the demo scenario
    :param mode: the operating mode for the demo scenario (choose between "scratch" and "resume")
    :param top_k: the number of resulting entities to be returned (by default -1, i.e., return all entities)
    :return: the dataframe containing the emitted resulting entities
    """

    # Select the operating mode (demo): "scratch" (i.e., from scratch) or "resume" (i.e., completing a top-k query)
    if demo and mode in ["scratch", "resume"]:
        mode = mode
    else:
        mode = "scratch"

    # Check the existence of the cache files required by the resume mode
    if mode == "resume":
        if not os.path.exists(var.cache_task_path) or not os.path.exists(var.cache_priority_queue_path) \
                or not os.path.exists(var.cache_neighbors_path) or not os.path.exists(var.cache_done_path) \
                or not os.path.exists(var.cache_results_path):
            mode = "scratch"

    # Initialize the Task object for the query that drives the entity resolution on-demand process
    if mode == "scratch":
        task = Task(query)
        with open(var.cache_task_path, "wb") as output_file:
            pkl.dump(task, output_file, pkl.HIGHEST_PROTOCOL)
            output_file.close()
        results = list()
    else:
        with open(var.cache_task_path, "rb") as input_file:
            task = pkl.load(input_file)
            input_file.close()
        task.top_k = top_k
        with open(var.cache_results_path, "rb") as input_file:
            results = pkl.load(input_file)
            input_file.close()

    # Load the dataset in the dataframe format
    ds = pd.read_csv(task.ds_path)
    ds.columns = ds.columns.str.lower()
    ds[task.ordering_key] = pd.to_numeric(ds[task.ordering_key], errors="coerce")
    for column in ds.columns:
        if ds[column].dtype == "object":
            ds[column] = ds[column].str.lower()
            ds[column] = ds[column].fillna("NULL")
            ds[column] = ds[column].apply(lambda x: html.unescape(x))

    # If the matcher is defined as None, return the dirty results without performing ER (i.e., do not call BrewER)
    if task.matching_function == "None (Dirty)":
        if len(task.having) > 0:
            connector = " " + task.operator + " "
            sql_statement = connector.join(task.having)
            ds = ds.query(sql_statement, engine="python")
        ordering_mode = (task.ordering_mode == "asc")
        results = ds.sort_values(by=[task.ordering_key], ascending=ordering_mode)
        if task.top_k > 0:
            results = results.head(task.top_k)

    # Otherwise, better call BrewER
    else:

        # Load the gold standard defined by the matcher as a dataframe and produce the set of the matching pairs
        gold = pd.read_csv(task.gold_path)
        gold = set(list(gold.itertuples(index=False, name=None)))

        # Perform blocking according to the selected blocking function
        ds_ids = list(ds["_id"])
        candidates = blocking(task.blocking_function, task.candidates_path, ds_ids)

        # Perform entity resolution on-demand using BrewER
        results = brewer(task, ds, gold, candidates, demo, mode, results)

    if len(results.index) > 0:
        if task.matching_function == "None (Dirty)":
            attributes = task.attributes
        else:
            attributes = task.attributes + ["comparisons", "time"]

        # Print the dataframe containing the resulting entities
        if not demo:
            print("\n")
            print(results[attributes])
        elif task.matching_function == "None (Dirty)":
            results_html = results[attributes].to_html(index=False)
            display(HTML(results_html))

    else:
        print("\nNo entities satisfied the query.")

    return results


output = widgets.Output()

sql_query = widgets.Textarea(
    placeholder="Write your query here",
    disabled=False,
    layout=widgets.Layout(width="99%", height="125px")
)


def on_run_button_clicked(b):
    with output:
        parsed_query = parser(sql_query)
        parsed_top_k = top_k_parser(sql_query)
        if parsed_query["complete"]:
            run(parsed_query, demo=True, mode="scratch", top_k=parsed_top_k)
        else:
            print("Invalid query syntax.")


run_button = widgets.Button(
    description="Run",
    disabled=False,
    button_style="",
    style=dict(button_color="springgreen", font_weight="bold")
)

run_button.on_click(on_run_button_clicked)


def on_resume_button_clicked(b):
    with output:
        parsed_query = parser(sql_query)
        parsed_top_k = top_k_parser(sql_query)
        if parsed_query["complete"]:
            run(parsed_query, demo=True, mode="resume", top_k=parsed_top_k)
        else:
            print("Invalid query syntax.")


resume_button = widgets.Button(
    description="Resume",
    disabled=False,
    button_style="",
    style=dict(button_color="red", font_weight="bold")
)

resume_button.on_click(on_resume_button_clicked)


def on_clear_button_clicked(b):
    with output:
        clear_output()


clear_button = widgets.Button(
    description="Clear",
    disabled=False,
    button_style="",
    style=dict(button_color="lightskyblue", font_weight="bold")
)

clear_button.on_click(on_clear_button_clicked)

buttons = widgets.HBox(
    [run_button, resume_button, clear_button],
    layout=widgets.Layout(
        display="flex",
        flex_flow="row",
        align_items="center",
        justify_content="center",
        width="99%"
    )
)
