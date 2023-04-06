data_dir_path = "./data/"
cache_dir_path = "./cache/"
cache_task_path = cache_dir_path + "task.pkl"
cache_priority_queue_path = cache_dir_path + "priority_queue.pkl"
cache_neighbors_path = cache_dir_path + "neighbors.pkl"
cache_done_path = cache_dir_path + "done.pkl"
cache_results_path = cache_dir_path + "results.pkl"

pipelines = {"bf_man_dev_mf_gt": {"blocking_function": "Manually-Devised Blocking",
                                  "matching_function": "Ground Truth"},
             "bf_sparker_mf_gt": {"blocking_function": "SparkER Meta-Blocking",
                                  "matching_function": "Ground Truth"}}

aggregation_functions = ["MAX", "MIN", "AVG", "VOTE", "RANDOM"]
default_dataset = "camera"
datasets = {"beer": dict(),
            "camera": {"ds_path": data_dir_path + "camera_dataset.csv",
                       "attributes":  # ["brand", "model", "type", "mp", "screen_size", "price"],
                           ["id", "description", "brand", "model", "type", "mp", "optical_zoom", "digital_zoom",
                            "screen_size", "price"],
                       "default_aggregation_function": "VOTE",
                       "default_ordering_key": "mp",
                       "default_ordering_mode": "desc",
                       "blocking_functions":
                           {"None (Cartesian Product)":
                                {"candidates_path": None,
                                 "blocks_path": None},
                            "Manually-Devised Blocking":
                                {"candidates_path": data_dir_path + "camera_candidates_manual.pkl",
                                 "blocks_path": data_dir_path + "camera_blocks_manual.pkl"},
                            "SparkER Meta-Blocking":
                                {"candidates_path": data_dir_path + "camera_candidates_sparker.pkl",
                                 "blocks_path": data_dir_path + "camera_blocks_sparker.pkl"}},
                       "default_blocking_function": "SparkER Meta-Blocking",
                       "matching_functions":
                           {"None (Dirty)": {"gold_path": None},
                            "Ground Truth": {"gold_path": data_dir_path + "camera_gold.csv"}},
                       "default_matching_function": "Ground Truth"},
            "funding": dict(),
            "notebook": dict(),
            "usb": {"ds_path": data_dir_path + "usb_dataset.csv",
                    "attributes": ["id", "name", "brand", "size_gb", "price"],
                    "default_aggregation_function": "VOTE",
                    "default_ordering_key": "price",
                    "default_ordering_mode": "asc",
                    "blocking_functions":
                        {"SparkER Meta-Blocking":
                             {"candidates_path": data_dir_path + "usb_candidates_sparker.pkl",
                              "blocks_path": data_dir_path + "usb_blocks_sparker.pkl"}},
                    "default_blocking_function": "SparkER Meta-Blocking",
                    "matching_functions":
                        {"None (Dirty)": {"gold_path": None},
                         "Ground Truth": {"gold_path": data_dir_path + "usb_gold.csv"}},
                    "default_matching_function": "Ground Truth"}
            }

html_format = """
<script src="https://code.jquery.com/jquery-latest.min.js"></script>
<script type="text/javascript">
    $(document).ready(function(){
        $('tr.entity').click(function(){
            $(this).nextUntil('tr.entity').slideToggle(100, function(){
            });
        });
        $(".record").hide();
    });
</script>

<style type="text/css">
    table.dataframe td, table.dataframe th
    {
        text-align: left;
        word-wrap: break-word;
        column-width: 150px;
        max-width: 150px;
    }
    table.dataframe tr.entity
    {
        cursor:pointer;
        background-color: #E0E0E0;
        font-weight: 500;
    }
    table.dataframe tr.record:nth-child(odd) td
    {
        background-color: #F6F6F6;
    }
    table.dataframe tr.record:nth-child(even) td
    {
        background-color: #FFFFFF;
    }
</style>
"""
#  Note: quando stampa un gruppo di entità, per la prima e per l'ultima funziona bene,
#  per quelle di mezzo invece ricollassa automaticamente i record dopo tot)
