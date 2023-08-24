ds_dir_path = "./datasets/"
ds_file_name = "dataset.csv"
gold_file_name = "matches.csv"
candidates_file_name = "blocking_functions/candidates_"
blocks_file_name = "blocking_functions/blocks_"
matches_file_name = "matching_functions/matches_"

cache_dir_path = "./cache/"
cache_task_path = cache_dir_path + "task.pkl"
cache_priority_queue_path = cache_dir_path + "priority_queue.pkl"
cache_neighbors_path = cache_dir_path + "neighbors.pkl"
cache_done_path = cache_dir_path + "done.pkl"
cache_results_path = cache_dir_path + "results.pkl"
cache_emitted_entities_path = cache_dir_path + "emitted_entities.pkl"
cache_entity_indices_path = cache_dir_path + "entity_indices.pkl"

aggregation_functions = ["MIN", "MAX", "AVG", "VOTE", "RANDOM"]
datasets = {"alaska_cameras":
                {"ds_path": ds_dir_path + "alaska_cameras/" + ds_file_name,
                 "attributes": ["_id", "description", "brand", "model", "type", "mp", "optical_zoom",
                                "digital_zoom", "screen_size", "price"],
                 "default_aggregation_function": "VOTE",
                 "default_ordering_key": "price",
                 "default_ordering_mode": "asc",
                 "blocking_functions":
                     {
                         "None (Cartesian Product)":
                             {"candidates_path": None,
                              "blocks_path": None},
                         "SparkER (Meta-Blocking)":
                             {"candidates_path":
                                  ds_dir_path + "alaska_cameras/" + candidates_file_name + "sparker.pkl",
                              "blocks_path":
                                  ds_dir_path + "alaska_cameras/" + blocks_file_name + "sparker.pkl"},
                         "Manually-Devised Blocking":
                             {"candidates_path":
                                  ds_dir_path + "alaska_cameras/" + candidates_file_name + "manual.pkl",
                              "blocks_path":
                                  ds_dir_path + "alaska_cameras/" + blocks_file_name + "manual.pkl"}
                     },
                 "matching_functions":
                     {
                         "None (Dirty)":
                             {"gold_path": None},
                         "Ground Truth":
                             {"gold_path": ds_dir_path + "alaska_cameras/" + gold_file_name},
                         "Magellan (Decision Tree)":
                             {"gold_path": ds_dir_path + "alaska_cameras/" + matches_file_name + "magellan_decision_tree.csv"},
                         "DeepMatcher (RNN)":
                             {"gold_path": ds_dir_path + "alaska_cameras/" + matches_file_name + "deepmatcher_rnn.csv"}
                     },
                 "pipelines":
                     {
                         "bf_sparker_mf_gt":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "Ground Truth"},
                         "bf_sparker_mf_mag_dt":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "Magellan (Decision Tree)"},
                         "bf_sparker_mf_dm_rnn":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "DeepMatcher (RNN)"},
                         "bf_manual_mf_gt":
                             {"blocking_function": "Manually-Devised Blocking",
                              "matching_function": "Ground Truth"},
                         "bf_manual_mf_mag_dt":
                             {"blocking_function": "Manually-Devised Blocking",
                              "matching_function": "Magellan (Decision Tree)"},
                         "bf_manual_mf_dm_rnn":
                             {"blocking_function": "Manually-Devised Blocking",
                              "matching_function": "DeepMatcher (RNN)"},
                     }
                 },
            "alaska_cameras_small":
                {"ds_path": ds_dir_path + "alaska_cameras_small/" + ds_file_name,
                 "attributes": ["_id", "description", "brand", "model", "type", "mp", "optical_zoom",
                                "digital_zoom", "screen_size", "price"],
                 "default_aggregation_function": "VOTE",
                 "default_ordering_key": "price",
                 "default_ordering_mode": "asc",
                 "blocking_functions":
                     {
                         "None (Cartesian Product)":
                             {"candidates_path": None,
                              "blocks_path": None},
                         "SparkER (Meta-Blocking)":
                             {"candidates_path":
                                  ds_dir_path + "alaska_cameras_small/" + candidates_file_name + "sparker.pkl",
                              "blocks_path":
                                  ds_dir_path + "alaska_cameras_small/" + blocks_file_name + "sparker.pkl"},
                         "Manually-Devised Blocking":
                             {"candidates_path":
                                  ds_dir_path + "alaska_cameras_small/" + candidates_file_name + "manual.pkl",
                              "blocks_path":
                                  ds_dir_path + "alaska_cameras_small/" + blocks_file_name + "manual.pkl"}
                     },
                 "matching_functions":
                     {
                         "None (Dirty)":
                             {"gold_path": None},
                         "Ground Truth":
                             {"gold_path": ds_dir_path + "alaska_cameras_small/" + gold_file_name},
                         "Magellan (Decision Tree)":
                             {"gold_path": ds_dir_path + "alaska_cameras_small/" + matches_file_name +
                                           "magellan_decision_tree.csv"}
                     },
                 "pipelines":
                     {
                         "bf_sparker_mf_gt":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "Ground Truth"},
                         "bf_sparker_mf_mag_dt":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "Magellan (Decision Tree)"},
                         "bf_manual_mf_gt":
                             {"blocking_function": "Manually-Devised Blocking",
                              "matching_function": "Ground Truth"},
                         "bf_manual_mf_mag_dt":
                             {"blocking_function": "Manually-Devised Blocking",
                              "matching_function": "Magellan (Decision Tree)"}
                     }
                 },
            "altosight_usb_sticks":
                {"ds_path": ds_dir_path + "altosight_usb_sticks/" + ds_file_name,
                 "attributes": ["_id", "name", "brand", "size", "price"],
                 "default_aggregation_function": "VOTE",
                 "default_ordering_key": "price",
                 "default_ordering_mode": "asc",
                 "blocking_functions":
                     {
                         "None (Cartesian Product)":
                             {"candidates_path": None,
                              "blocks_path": None},
                         "PyJedAI (Similarity Join)":
                             {"candidates_path":
                                  ds_dir_path + "altosight_usb_sticks/" + candidates_file_name + "pyjedai.pkl",
                              "blocks_path":
                                  ds_dir_path + "altosight_usb_sticks/" + blocks_file_name + "pyjedai.pkl"}
                     },
                 "matching_functions":
                     {
                         "None (Dirty)":
                             {"gold_path": None},
                         "Ground Truth":
                             {"gold_path": ds_dir_path + "altosight_usb_sticks/" + gold_file_name},
                         "Magellan (Random Forest)":
                             {"gold_path": ds_dir_path + "altosight_usb_sticks/" + matches_file_name +
                                           "magellan_random_forest.csv"}
                     },
                 "pipelines":
                     {
                         "bf_pyjedai_mf_gt":
                             {"blocking_function": "PyJedAI (Similarity Join)",
                              "matching_function": "Ground Truth"},
                         "bf_pyjedai_mf_mag_rf":
                             {"blocking_function": "PyJedAI (Similarity Join)",
                              "matching_function": "Magellan (Random Forest)"}
                     }
                 },
            "altosight_usb_sticks_small":
                {"ds_path": ds_dir_path + "altosight_usb_sticks_small/" + ds_file_name,
                 "attributes": ["_id", "name", "brand", "size", "price"],
                 "default_aggregation_function": "VOTE",
                 "default_ordering_key": "price",
                 "default_ordering_mode": "asc",
                 "blocking_functions":
                     {
                         "None (Cartesian Product)":
                             {"candidates_path": None,
                              "blocks_path": None},
                         "PyJedAI (Similarity Join)":
                             {"candidates_path":
                                  ds_dir_path + "altosight_usb_sticks_small/" + candidates_file_name + "pyjedai.pkl",
                              "blocks_path":
                                  ds_dir_path + "altosight_usb_sticks_small/" + blocks_file_name + "pyjedai.pkl"}
                     },
                 "matching_functions":
                     {
                         "None (Dirty)":
                             {"gold_path": None},
                         "Ground Truth":
                             {"gold_path": ds_dir_path + "altosight_usb_sticks_small/" + gold_file_name},
                         "Magellan (Random Forest)":
                             {"gold_path": ds_dir_path + "altosight_usb_sticks_small/" + matches_file_name +
                                           "magellan_random_forest.csv"},
                         "Magellan (SVM)":
                             {"gold_path": ds_dir_path + "altosight_usb_sticks_small/" + matches_file_name + "magellan_svm.csv"}
                     },
                 "pipelines":
                     {
                         "bf_pyjedai_mf_gt":
                             {"blocking_function": "PyJedAI (Similarity Join)",
                              "matching_function": "Ground Truth"},
                         "bf_pyjedai_mf_mag_rf":
                             {"blocking_function": "PyJedAI (Similarity Join)",
                              "matching_function": "Magellan (Random Forest)"},
                         "bf_pyjedai_mf_mag_svm":
                             {"blocking_function": "PyJedAI (Similarity Join)",
                              "matching_function": "Magellan (SVM)"}
                     }
                 },
            "magellan_beers":
                {"ds_path": ds_dir_path + "magellan_beers/" + ds_file_name,
                 "attributes": ["_id", "name", "factory", "style", "abv"],
                 "default_aggregation_function": "VOTE",
                 "default_ordering_key": "abv",
                 "default_ordering_mode": "desc",
                 "blocking_functions":
                     {
                         "None (Cartesian Product)":
                             {"candidates_path": None,
                              "blocks_path": None},
                         "SparkER (Meta-Blocking)":
                             {"candidates_path":
                                  ds_dir_path + "magellan_beers/" + candidates_file_name + "sparker.pkl",
                              "blocks_path":
                                  ds_dir_path + "magellan_beers/" + blocks_file_name + "sparker.pkl"}
                     },
                 "matching_functions":
                     {
                         "None (Dirty)":
                             {"gold_path": None},
                         "Ground Truth":
                             {"gold_path": ds_dir_path + "magellan_beers/" + gold_file_name},
                         "Magellan (Random Forest)":
                             {"gold_path": ds_dir_path + "magellan_beers/" + matches_file_name + "magellan_random_forest.csv"},
                         "Magellan (SVM)":
                             {"gold_path": ds_dir_path + "magellan_beers/" + matches_file_name + "magellan_svm.csv"}
                     },
                 "pipelines":
                     {
                         "bf_sparker_mf_gt":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "Ground Truth"},
                         "bf_sparker_mf_mag_rf":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "Magellan (Random Forest)"},
                         "bf_sparker_mf_mag_svm":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "Magellan (SVM)"}
                     }
                 },
            "magellan_bikes":
                {"ds_path": ds_dir_path + "magellan_bikes/" + ds_file_name,
                 "attributes": ["_id", "name", "color", "fuel", "year", "km", "city", "owner", "price"],
                 "blocking_functions":
                     {
                         "None (Cartesian Product)":
                             {"candidates_path": None,
                              "blocks_path": None}
                     },
                 "matching_functions":
                     {
                         "None (Dirty)":
                             {"gold_path": None},
                         "Ground Truth":
                             {"gold_path": ds_dir_path + "magellan_bikes/" + gold_file_name}
                     }
                 },
            "nyc_funding_applications":
                {"ds_path": ds_dir_path + "nyc_funding_applications/" + ds_file_name,
                 "attributes": ["_id", "name", "address", "year", "agency", "source", "counselor", "amount", "status"],
                 "default_aggregation_function": "VOTE",
                 "default_ordering_key": "amount",
                 "default_ordering_mode": "desc",
                 "blocking_functions":
                     {
                         "None (Cartesian Product)":
                             {"candidates_path": None,
                              "blocks_path": None},
                         "SparkER (Meta-Blocking)":
                             {"candidates_path":
                                  ds_dir_path + "nyc_funding_applications/" + candidates_file_name + "sparker.pkl",
                              "blocks_path":
                                  ds_dir_path + "nyc_funding_applications/" + blocks_file_name + "sparker.pkl"}
                     },
                 "matching_functions":
                     {
                         "None (Dirty)":
                             {"gold_path": None},
                         "Ground Truth":
                             {"gold_path": ds_dir_path + "nyc_funding_applications/" + gold_file_name},
                         "Magellan (Random Forest)":
                             {"gold_path": ds_dir_path + "nyc_funding_applications/" + matches_file_name +
                                           "magellan_random_forest.csv"},
                         "DeepMatcher (RNN)":
                             {"gold_path": ds_dir_path + "nyc_funding_applications/" + matches_file_name +
                                           "deepmatcher_rnn.csv"}
                     },
                 "pipelines":
                     {
                         "bf_sparker_mf_gt":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "Ground Truth"},
                         "bf_sparker_mf_mag_rf":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "Magellan (Random Forest)"},
                         "bf_sparker_mf_dm_rnn":
                             {"blocking_function": "SparkER (Meta-Blocking)",
                              "matching_function": "DeepMatcher (RNN)"},
                     }
                 }
            }

html_format = """
    <script src="https://code.jquery.com/jquery-latest.min.js"></script>
    <script type="text/javascript">
        $(document).ready(function(){
            $('tr.entity').click(function(){
                $(this).find('span').text(function(_, value){return value=='∧'?'∨':'∧'});
                $(this).nextUntil('tr.entity').slideToggle(100, function(){
                });
            });
            $(".record").hide();
        });
    </script>
    
    <style type="text/css">
        table.dataframe
         {
            border-collapse: separate;
            border-spacing: 0 1px;
        }
        table.dataframe td, table.dataframe th
        {
            max-width: 300px;
            word-wrap: break-word;
            padding-left: 15px;
            padding-right: 15px;
        }
        table.dataframe tr.entity
        {
            cursor:pointer;
            background-color: #EBF5FB;
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
