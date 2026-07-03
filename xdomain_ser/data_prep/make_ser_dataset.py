# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Frame per-domain examples as text->MR ``ds`` data and emit per-topic hint maps.

Second step of the dataset-construction pipeline (see ``scripts/build_dataset.sh``),
run once per domain on the unified JSON produced by ``preprocess.py``. Assigns each
example a ``hint_map_id`` (and, for the Taskmaster / RNNLG families, splits by topic),
then writes the examples plus a ``hint-map-<id>.json`` file per topic. The resulting
``*-ds.json`` and ``hint-map-*.json`` files are listed in the ``files.json`` manifest
that ``merge_select.py`` consumes.

``dataset_name`` selects the per-domain branch: ``e2e_nlg``, ``viggo``, ``rnnlg``,
``tm1``, ``tm2``, ``tm3``. (The earlier length-split ``main1`` variant is dropped.)
"""
import argparse
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("hint_map_path", type=str)
    parser.add_argument("dataset_name", type=str)
    args = parser.parse_args()
    print("args:", args)

    with open(args.input_path, "r") as fin:
        data = json.load(fin)
    #lengths = [(len(ex["mr"]), j, ex) for j,ex in enumerate(data)]

    with open(args.hint_map_path, "r") as fin:
        hint_map = json.load(fin)

    if args.dataset_name == "e2e_nlg":
        hint_map_id = "hm_e2e_nlg"
        for ex in data:
            ex["hint_map_id"] = hint_map_id
    elif args.dataset_name == "viggo":
        hint_map_id = "hm_viggo"
        for ex in data:
            ex["hint_map_id"] = hint_map_id
    elif args.dataset_name == "rnnlg":
        if "hotel" in args.input_path:
            topic = "hotel"
        elif "laptop" in args.input_path:
            topic = "laptop"
        elif "restaurant" in args.input_path:
            topic = "restaurant"
        elif "tv" in args.input_path:
            topic = "tv"
        else:
            raise ValueError(f"unknown topic {args.input_path}")
        hint_map_id = f"hm_rnnlg_{topic}"

        for ex in data:
            ex["hint_map_id"] = hint_map_id
            ex["topic"] = topic

        hint_map = hint_map[topic]

    elif args.dataset_name == "tm1":
        hm_lookup = {
            "movie_ticket": "movie",
            "movie_tickets": "movie",
            "movie_finder": "movie",
            "uber_lyft": "uber",
            "restaurant_table": "restaurant",
            "pizza_ordering": "pizza",
            "coffee_ordering": "coffee",
            "auto_repair_appt": "auto",
            "auto_repair": "auto",
        }

        topic_examples = defaultdict(list)
        for ex in data:
            topic = ex["topic"]
            hint_map_id = f"hm_tm1_{topic}"
            ex["hint_map_id"] = hint_map_id
            topic_examples[topic].append(ex)

        out_dir = os.path.dirname(args.output_path)
        part = "test" if "test" in args.input_path else "train"
        fname = f"self-dialogs-{part}" if "self-dialogs" in args.input_path else f"woz-dialogs-{part}"

        for topic, examples in topic_examples.items():
            outpath = os.path.join(out_dir, fname+f"-{topic}.json")
            print("  * saving examples to", outpath)
            with open(outpath, "w") as fout:
                json.dump(examples, fout, indent=2)

            hm = hint_map[hm_lookup[topic]]

            hint_map_id = f"hm_tm1_{topic}"
            out_dir = os.path.dirname(args.output_path)
            out_path = os.path.join(out_dir, f"hint-map-{hint_map_id}.json")
            print("  * saving hint-map to {}".format(out_path))
            with open(out_path, "w") as fout:
                topic_hint_map = {
                    "hint_map_id": hint_map_id,
                    "hint_map": hm,
                }
                json.dump(topic_hint_map, fout, indent=2)

    elif args.dataset_name == "tm2":
        topic = os.path.basename(args.input_path).replace(".json", "").replace("-", "_")
        hint_map_id = f"hm_tm2_{topic}"
        topic_hint_map = {}

        for ex in data:
            if topic == "sports":
                ex_topic = ex["topic"]
                hint_map_id = f"hm_tm2_sports_{ex_topic}"
                this_topic = f"sports_{ex_topic}"
                topic_hint_map[this_topic] = hint_map["sports"][ex_topic]
            else:
                this_topic = topic
            ex["hint_map_id"] = hint_map_id
            ex["topic"] = this_topic

        if topic == "sports":
            for topic, hm in topic_hint_map.items():
                hint_map_id = f"hm_tm2_{topic}"
                out_dir = os.path.dirname(args.output_path)
                out_path = os.path.join(out_dir, f"hint-map-{hint_map_id}.json")
                print("  * saving hint-map to {}".format(out_path))
                with open(out_path, "w") as fout:
                    hint_map = {
                        "hint_map_id": hint_map_id,
                        "hint_map": hm,
                    }
                    json.dump(hint_map, fout, indent=2)
        else:
            # Non-sports tm2 topics also need their per-topic hint-map written.
            # The generic save block below excludes all of dataset_name == "tm2",
            # so without this the merge step is missing hint-map-hm_tm2_<topic>.json.
            out_dir = os.path.dirname(args.output_path)
            out_path = os.path.join(out_dir, f"hint-map-{hint_map_id}.json")
            print("  * saving hint-map to {}".format(out_path))
            with open(out_path, "w") as fout:
                json.dump({
                    "hint_map_id": hint_map_id,
                    "hint_map": hint_map[topic],
                }, fout, indent=2)

    elif args.dataset_name == "tm3":
        this_topic = "Movie_Tickets"
        hint_map = hint_map[this_topic]
        hint_map_id = f"hm_tm3_{this_topic}"

        for ex in data:
            ex["hint_map_id"] = hint_map_id
            ex["topic"] = this_topic

    else:
        raise ValueError(f"unknown dataset name: {args.dataset_name}")


    if args.dataset_name not in {"tm1"}:
        out_dir = os.path.dirname(args.output_path)
        if not os.path.exists(out_dir):
            print(f" * making {args.output_path}")
            os.makedirs(out_dir)


        print("  * saving data to {}".format(args.output_path))
        with open(args.output_path, "w") as fout:
            json.dump(data, fout, indent=2)

        if args.dataset_name != "tm2" and "sports" not in args.input_path:
            out_dir = os.path.dirname(args.output_path)
            out_path = os.path.join(out_dir, f"hint-map-{hint_map_id}.json")
            print("  * saving hint-map to {}".format(out_path))
            with open(out_path, "w") as fout:
                hint_map = {
                    "hint_map_id": hint_map_id,
                    "hint_map": hint_map,
                }
                json.dump(hint_map, fout, indent=2)


if __name__ == "__main__":
    main()
