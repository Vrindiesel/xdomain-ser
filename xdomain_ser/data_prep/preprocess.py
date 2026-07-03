# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Convert raw per-domain corpora into the unified intermediate JSON format.

First step of the dataset-construction pipeline (see ``scripts/build_dataset.sh``).
Reads a raw E2E, ViGGO, RNNLG, or Taskmaster file and emits one JSON record per
example with a normalised meaning representation (``mr``), the reference text
(``surface_form``), and a ``topic``. The per-domain ``unpack_*`` helpers perform
the slot-name / value normalisation (e.g. RNNLG ``kids-allowed=yes`` ->
``child_friendly``, ViGGO ``has_multiplayer=yes`` -> ``multiplayer``); the
Taskmaster branch additionally applies the train/dev/test partition file and
splits annotated assistant turns into slot mentions.

Output feeds ``make_ser_dataset.py``, which adds hint-map ids and the text->MR
``ds`` framing. Part of the optional ``[datasets]`` extra.
"""
import argparse
import json
import os
from collections import defaultdict

import pandas as pd


def unpack_mr_val(val, yes, no):
    if val in {"yes", "true"}:
        new_val = yes
    elif val in {"no", "false"}:
        new_val = no
    elif val in {"dontcare", "dont_care"}:
        new_val = "dontcare"
    else:
        new_val = val
    return new_val


def unpack_rnnlg_mr(mr_str, topic):
    KEY_MAP = {
        "kids-allowed": "child_friendly",
        "hasinternet": "has_internet",
        "acceptscreditcards": "accepts_credit_cards",
        "acceptscards": "accepts_credit_cards",
        "dogs-allowed": "pets_allowed",
        "dogs_allowed": "pets_allowed",
        "dogsallowed": "pets_allowed",
        "isforbusinesscomputing": "is_for_business_computing",
        "hasusbport": "has_usb_port",
        "screensizerange": "screen_size_range",
        "ecorating": "eco_rating",
        "pricerange": "price_range",
        "powerconsumption": "power_consumption",
        "screensize": "screen_size",
        "hdmiport": "hdmi_port",
        "weightrange": "weight_range",
        "batteryrating": "battery_rating",
        "driverange": "drive_range",
        "goodformeal": "good_for_meal"

    }

    #print("mr_str:", mr_str)
    j = mr_str.index("(")
    dact = mr_str[:j]
    attrs = mr_str[j+1:-1].split(";")
    #print("attrs:", attrs)
    slots = defaultdict(list)
    for slot_val in attrs:
        #print(slot_val)
        if "=" in slot_val:
            j = slot_val.index("=")
            name = slot_val[:j]
            value = slot_val[j+1:]
            if value.endswith(")"):
                value = value[:-1]
            if value.startswith("'"):
                value = value[1:]
            if value.endswith("'"):
                value = value[:-1]

            new_value = value

            if name == "kids-allowed":
                new_name = KEY_MAP[name]
                new_value = unpack_mr_val(value, "child_friendly", "not_child_friendly")
            elif name == "hasinternet":
                new_value = unpack_mr_val(value, "has_internet", "no_internet")
                new_name = KEY_MAP.get(name, name)
            elif name in {"acceptscards", "acceptscreditcards"}:
                new_value = unpack_mr_val(value, "accepts_credit_cards", "no_credit_cards")
                new_name = KEY_MAP.get(name, name)
            elif name in {"dogs-allowed", "dogs_allowed", "dogsallowed"}:
                new_value = unpack_mr_val(value, "pets_allowed", "no_pets_allowed")
                new_name = KEY_MAP.get(name, name)
            elif name == "isforbusinesscomputing":
                new_value = unpack_mr_val(value, "business_oriented", "not_business_oriented")
                new_name = KEY_MAP.get(name, name)
            elif name == "hasusbport":
                new_value = unpack_mr_val(value, "has_usb_port", "no_usb_port")
                new_name = KEY_MAP.get(name, name)
            else:
                new_name = KEY_MAP.get(name, name)

        else:
            new_name = slot_val
            if new_name.endswith(")"):
                new_name = new_name[:-1]
            new_value = None

        # skipp empty string attributes
        if new_value is None or new_value:
            new_name = f"{topic}_{new_name}"
            if isinstance(new_value, list):
                slots[new_name].extend(new_value)
            else:
                slots[new_name].append(new_value)

    d = {
        "dact": dact,
        "slots": slots
    }
    return d


def unpack_viggo_mr(mr_str):
    """
    give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])
    :param mr_str:
    :return:
    """
    #print(mr_str)
    #dact, attrs = mr_str.split()
    KEY_MAP = {
        "rating": "review_rating",
        "esrb": "esrb_rating",
        "player_perspective": "perspective",
        "release_year": "released",  # or keep as "release_year" if you prefer
        "has_multiplayer": "multiplayer_mode",
        "available_on_steam": "steam_availability",  # only used when --steam-to-storefronts is set
    }


    #print("mr_str:", mr_str)
    j = mr_str.index("(")
    dact = mr_str[:j]
    attrs = mr_str[j+1:-1].split("], ")

    slots = defaultdict(list)
    for slot_val in attrs:
        #print(slot_val)
        j = slot_val.index("[")
        name = slot_val[:j]
        value = slot_val[j+1:]
        if value.endswith("]"):
            value = value[:-1]
        if ", " in value:
            value = value.split(", ")

        new_value = value
        if name == "available_on_steam":
            new_name = KEY_MAP[name]
            # on_steam/not_on_steam
            new_value = "on_steam" if value == "yes" else "not_on_steam"
        elif name == "has_linux_release":
            new_value = "Linux" if value == "yes" else "not_released_on_Linux"
            new_name = "pc_os_support"
        elif name == "has_mac_release":
            new_value = "macOS" if value == "yes" else "not_released_on_macOS"
            new_name = "pc_os_support"
        elif name == "has_multiplayer":
            new_value = "multiplayer" if value == "yes" else "single-player"
            new_name = KEY_MAP.get(name, name)
        else:
            new_name = KEY_MAP.get(name, name)

        # skipp empty string attributes
        if new_value:
            if isinstance(new_value, list):
                slots[new_name].extend(new_value)
            else:
                slots[new_name].append(new_value)
        #except Exception as e:
        #    print("attrs:", attrs)
        #    print("slot_val:", slot_val)
        #    raise e
    #print("slots:", slots)
    #if "pc_os_support" in slots:
    #    slots["pc_os_support"].append("Windows")

    d = {
        "dact": dact,
        "slots": slots
    }
    return d


def unpack_e2e_nlg_mr(mr_str):
    """
    Family suitability: look for negations (“not family friendly”, “no kids”, “adults only”) before positives. Emit enum, not boolean.
    Venue vs cuisine conflict: if both a venue-type noun (“coffee shop”, “pub”) and a cuisine word appear, assign each to its slot; don’t overwrite one with the other.
    Area vs nearby: area should be a neighborhood/zone (“riverside”, “city centre”); nearby_landmark should stay free-text.
    Missing values: if you train abstention, fill absent requested slots with "not_mentioned".
    """
    KEY_MAP = {
        "customer rating": "customerRating",
        "familyFriendly": "family_suitability",
        "eatType": "venue_type",
        "food": "cuisine_type",
        "near": "nearby_landmark",
        "area": "area_zone",  # only used when --steam-to-storefronts is set
    }

    attrs = mr_str.split("], ")
    slots = defaultdict(list)
    for slot_val in attrs:
        #print(slot_val)
        j = slot_val.index("[")
        name = slot_val[:j]
        value = slot_val[j+1:]
        if value.endswith("]"):
            value = value[:-1]
        if ", " in value:
            value = value.split(", ")

        new_value = value
        if name == "familyFriendly":
            new_value = "family-friendly" if value == "yes" else "not-family-friendly"
            new_name = KEY_MAP.get(name, name)
        else:
            new_name = KEY_MAP.get(name, name)

        # skipp empty string attributes
        if new_value:
            if isinstance(new_value, list):
                slots[new_name].extend(new_value)
            else:
                slots[new_name].append(new_value)

    d = {
        "slots": slots
    }
    return d

def make_taskmaster_slot_name(old_name):
    new_name = old_name
    topic = None
    if "." in new_name:
        toks = new_name.split(".")
        topic, toks = toks[0], toks[1:]
        if toks[-1] in {"accept", "reject"}:
            toks = toks[:-1]
        if len(toks) <= 2 and len(set(toks)) == len(toks):
            toks = [topic] + toks
        new_name = ".".join(toks)
    return new_name, topic

"""
restaurant.offical_description
restaurant.other_description
"""
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument('--dataset', choices=["viggo", "taskmaster", "e2e_nlg", "rnnlg"])
    parser.add_argument("--partition_file", type=str, default=None)
    #parser.add_argument("output_path")
    args = parser.parse_args()
    print("args:", args)

    #with open(args.input_file, "r") as fin:
    #    data = json.load(fin)

    if args.dataset == "viggo":
        df = pd.read_csv(args.input_path, sep=args.sep)
        data = []
        skipp_count = 0
        for row in df.itertuples():
            mr = unpack_viggo_mr(row.mr)
            for k,v in mr["slots"].items():
                if len(v) == 1:
                    mr["slots"][k] = v[0]
            if len(mr["slots"]) > 1:
                data.append({
                    "mr": mr,
                    "surface_form": row.ref,
                    "topic": "video_games",
                    "dataset": args.dataset,
                })
            else:
                skipp_count += 1
        print("skipped:", skipp_count, "empty MR examples")
        print(f"data size:{len(data)}")

    elif args.dataset == "rnnlg":
        with open(args.input_path) as fin:
            raw_data = json.load(fin)
        print("raw_data:", len(raw_data))

        if "restaurant" in args.input_path:
            topic = "restaurant"
        elif "hotel" in args.input_path:
            topic = "hotel"
        elif "laptop" in args.input_path:
            topic = "laptop"
        else:
            topic = "tv"

        data = []
        skipp_count = 0
        for row in raw_data:
            try:
                mr = unpack_rnnlg_mr(row[0], topic)
            except Exception as e:
                print("mr string:", row[0])
                raise e

            refs = row[1:]
            for k,v in mr["slots"].items():
                if len(v) == 1:
                    mr["slots"][k] = v[0]
            if len(mr["slots"]) > 1:
                data.append({
                    "mr": mr,
                    "surface_form": refs,
                    "dataset": args.dataset,
                })
            else:
                skipp_count += 1
        print("skipped:", skipp_count, "empty MR examples")
        print(f"data size:{len(data)}")



        #exit(0)
        #unpack_rnnlg_mr("")

    elif args.dataset == "e2e_personality":
        """
        Family suitability: look for negations (“not family friendly”, “no kids”, “adults only”) before positives. Emit enum, not boolean.
        Venue vs cuisine conflict: if both a venue-type noun (“coffee shop”, “pub”) and a cuisine word appear, assign each to its slot; don’t overwrite one with the other.
        Area vs nearby: area should be a neighborhood/zone (“riverside”, “city centre”); nearby_landmark should stay free-text.
        Missing values: if you train abstention, fill absent requested slots with "not_mentioned".
        """

    elif args.dataset == "e2e_nlg":
        df = pd.read_csv(args.input_path, sep=args.sep)
        data = []
        skipp_count = 0
        for row in df.itertuples():
            mr = unpack_e2e_nlg_mr(row.mr)
            for k,v in mr["slots"].items():
                if len(v) == 1:
                    mr["slots"][k] = v[0]
            if len(mr["slots"]) > 1:
                data.append({
                    "mr": mr,
                    "surface_form": row.ref,
                    "topic": "restaurants",
                    "dataset": args.dataset,
                })
            else:
                skipp_count += 1
        print("skipped:", skipp_count, "empty MR examples")
        print(f"data size:{len(data)}")


    elif args.dataset == "taskmaster":
        partitions = defaultdict(list)
        if args.partition_file is not None and os.path.exists(args.partition_file):
            with open(args.partition_file, "r") as fin:
                partitions = json.load(fin)
                for name, conv_ids in partitions.items():
                    partitions[name] = set(conv_ids)
                partitions["test"] = partitions["test"] - partitions["train"]
                partitions["dev"] = partitions["dev"] - partitions["test"]
                partitions["dev"] = partitions["dev"] - partitions["train"]



                for name, conv_ids in partitions.items():
                    partitions[name] = set(conv_ids)
                    print(f"partition {name}: {len(conv_ids)} conv_ids")
                #print("train - test:", len(partitions["train"] - partitions["test"]))
                #print("train - dev:", len(partitions["train"] - partitions["dev"]))

        #exit(0)
        history_size = 4
        with open(args.input_path) as fin:
            raw_data = json.load(fin)

        print("raw_data:", len(raw_data))
        #
        data = defaultdict(list)
        for j, conv in enumerate(raw_data):
            # calculate data partition
            if conv["conversation_id"] in partitions["test"]:
                part = "test"
                if conv["conversation_id"] in partitions["train"]:
                    continue
                assert conv["conversation_id"] not in partitions["train"]
            elif conv["conversation_id"] in partitions["dev"]:
                part = "dev"
                if conv["conversation_id"] in partitions["train"]:
                    continue
                assert conv["conversation_id"] not in partitions["train"]
            else:
                part = "train"

            if "instruction_id" in conv:
                conv_topic = "_".join([t for t in conv["instruction_id"].split("-") if t.isalpha()])
            elif "vertical" in conv:
                conv_topic = conv["vertical"].replace(" ", "_")
            history = []
            for utterance in conv["utterances"]:
                history.append({
                    "text": utterance["text"],
                    "speaker": utterance["speaker"],
                })

                # only used utterances by the automated agent
                if ("segments" in utterance and len(utterance["segments"]) > 0 and
                        utterance["speaker"] in {"ASSISTANT", "assistant"}):
                    utt_topic = None

                    # mapping from slot names to list of mentions
                    slots = defaultdict(list)
                    for seg in utterance["segments"]:
                        if "annotations" not in seg or len(seg["annotations"])  < 1: continue
                        name, topic = make_taskmaster_slot_name(seg["annotations"][0]["name"])
                        if utt_topic is None and topic:
                            utt_topic = topic
                        slots[name].append(seg)
                    # collapse multiple references to the same character span in the utterance
                    new_slots = defaultdict(list)
                    for slot_name, slot_mentions in slots.items():
                        # reduce duplicates of slot span starts
                        same_starts = defaultdict(list)
                        for men in slot_mentions:
                            same_starts[men["start_index"]].append((len(men["text"]), men))
                        reduced = []
                        for vlist in same_starts.values():
                            vlist.sort()
                            seg = vlist[0][1]
                            reduced.append(seg)
                        # reduce duplicates of slot span endings
                        same_ends = defaultdict(list)
                        for men in reduced:
                            same_ends[men["end_index"]].append((len(men["text"]), men))
                        for vlist in same_ends.values():
                            vlist.sort()
                            seg = vlist[0][1]
                            new_slots[slot_name].append(seg["text"])

                    # squeeze list of strings when only has 1 item.
                    for slot_name, slot_mentions in new_slots.items():
                        if isinstance(slot_mentions, list) and len(slot_mentions) == 1:
                            new_slots[slot_name] = slot_mentions[0]
                        else:
                            new_slots[slot_name] = slot_mentions

                    entry = {
                        "conversation_id": conv["conversation_id"],
                        "surface_form": utterance["text"],
                        "mr": new_slots,
                        "topic": conv_topic,
                        #"conv_history": history[-history_size:],
                    }
                    if "instruction_id" in entry:
                        entry["instruction_id"] = conv["instruction_id"]
                    data[part].append(entry)
        print(f"processed {j} convs")

    if not os.path.exists(args.output_path):
        print("create output directory ...", args.output_path)
        os.makedirs(args.output_path)

    name = os.path.basename(args.input_path)
    if name.endswith(".csv"):
        name = name.replace(".csv", ".json")
    elif name.endswith(".tsv"):
        name = name.replace(".tsv", ".json")
    assert name.endswith(".json")
    outpath = os.path.join(args.output_path, name)



    if isinstance(data, list):
        # viggo
        with open(outpath, "w") as fout:
            print(f"saving {len(data)} examples to {outpath}")
            json.dump(data, fout, indent=2)
    else:
        has_dev = False
        for part in [ "test", "dev",  "train",]:
            if len(data.get(part, [])) < 1:
                print(f"skip {part} ... no data")
                continue
            if part == "dev":
                has_dev = True

            if (part == "train" and has_dev) or part == "test":
                save_path = outpath.replace(".json", f"-{part}.json")
            else:
                save_path = outpath
            print(f"saving {len(data[part])} examples to {save_path}")
            with open(save_path, "w") as fout:
                json.dump(data[part], fout, indent=2)


if __name__ == "__main__":
    main()
