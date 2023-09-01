# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE_CC-BY-NC4.0 file in the root directory of this source tree.

import argparse
import glob, json, os, re
import tarfile, zipfile
import urllib.request
import xml.etree.ElementTree as et

import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk_data.init import init_nltk_data


def download_files(directory, urls, unzipped_filename):
    """download files from the given URLs to a local directory"""
    # Create a directory to store the downloaded files
    download_directory = os.path.join(directory, "downloaded_files")

    if not os.path.exists(download_directory):
        os.mkdir(download_directory)

    # Loop through the URLs and download each file
    for dataset_name, url in urls.items():
        filename = url.split("/")[-1]
        filepath = os.path.join(download_directory, filename)

        # Download the file
        if os.path.exists(filepath):
            print(f"Skipping downloading {dataset_name} as it already exists.")
        else:
            urllib.request.urlretrieve(url, filepath)
            print(f"Successfully downloaded {dataset_name}")

        if os.path.exists(
            os.path.join(download_directory, unzipped_filename[dataset_name])
        ):
            print(f"Skipping extracting {dataset_name} as it already has been done.")
        else:
            if dataset_name == "ReClor":
                # Unzip the password-protected file
                with zipfile.ZipFile(filepath, "r") as z:
                    z.extractall(
                        os.path.join(download_directory, "reclor"),
                        pwd=bytes("for_non-commercial_research_purpose_only", "utf-8"),
                    )
            elif dataset_name == "MCScript2.0":
                with zipfile.ZipFile(filepath, "r") as z:
                    z.extractall(os.path.join(download_directory, "mcscript"))
            elif url[-3:] == "zip":
                # Unzip the file
                with zipfile.ZipFile(filepath, "r") as z:
                    z.extractall(download_directory)
            elif url[-3:] == ".gz":
                # Extract the archive to the same folder
                with tarfile.open(filepath, "r") as t:
                    t.extractall(download_directory)
            print(f"Successfully extracted {dataset_name}")

    return download_directory


def process_sciq(download_directory):
    """process the SciQ json files and return Pandas df"""
    train = pd.read_json(
        os.path.join(download_directory, "SciQ dataset-2 3/train.json")
    )
    val = pd.read_json(os.path.join(download_directory, "SciQ dataset-2 3/valid.json"))
    joined = pd.concat([train, val], keys=["train", "val"])

    # remove fill-in-the-blank
    sciQ = joined.loc[~joined.question.str.contains("_")]

    # use NLTK sent tokenizer to count the number of sentences in the passage
    sciQ["num_sentences"] = sciQ.support.apply(lambda x: sent_tokenize(x)).str.len()
    sciQ["passage_id"] = sciQ.support.apply(hash)

    # randomly shuffle answers
    newcolnames = ["answer1", "answer2", "answer3", "answer4"]
    np.random.seed(0)
    sciQ[newcolnames] = sciQ.apply(
        lambda x: pd.Series(
            np.random.choice(
                x[["distractor1", "distractor2", "distractor3", "correct_answer"]],
                4,
                replace=False,
            ),
            index=newcolnames,
        ),
        axis=1,
    )

    # retrieve correct answer
    def get_correct_answer_num(row):
        for i in [1, 2, 3, 4]:
            if row["correct_answer"] == row["answer" + str(i)]:
                return i

    # finalize format and filter out long passages
    sciQ["correct_answer_num"] = sciQ.apply(get_correct_answer_num, axis=1)
    sciQ["passage_id"] = sciQ.groupby("support").ngroup()
    sciQ_reset = (
        sciQ.loc[sciQ.support.str.len() >= 1]
        .reset_index()
        .rename(columns={"support": "passage", "level_1": "question_id"})
    )
    sciQ_reset["split"] = sciQ_reset.level_0.apply(lambda x: "dev" if x == "val" else x)
    sciQ_reset["dataset"] = "sciQ"
    return sciQ_reset.loc[sciQ_reset.num_sentences <= 25][final_table_columns]


def process_multirc(download_directory):
    """process the MultiRC json files and return Pandas df"""
    with open(os.path.join(download_directory, "splitv2/dev_83-fixedIds.json")) as f:
        multirc_dev = json.load(f)["data"]
    with open(os.path.join(download_directory, "splitv2/train_456-fixedIds.json")) as f:
        multirc_train = json.load(f)["data"]

    # unpack json format to pandas table
    i = 0
    multirc_dict = {}
    reg_str = "</b>(.*?)<br>"
    for split, data in {"dev": multirc_dev, "train": multirc_train}.items():
        for para in data:
            res = re.findall(reg_str, para["paragraph"]["text"])
            para_text = " ".join(res)
            num_sents = len(res)
            for q in para["paragraph"]["questions"]:
                multirc_dict[i] = {
                    "split": split,
                    "passage_id": para["id"],
                    "passage": para_text,
                    "num_sentences": num_sents,
                    "question_dict": q,
                }
                i += 1
    unpacked = pd.DataFrame.from_dict(multirc_dict, orient="index")

    # get number of answers and correct answers
    def get_num_correct(q):
        return sum(a["isAnswer"] for a in q["answers"])

    unpacked["num_correct_answers"] = unpacked.question_dict.apply(get_num_correct)
    unpacked["num_answers"] = unpacked.apply(
        lambda x: len(x["question_dict"]["answers"]), axis=1
    )

    # filter questions that match Belebele format and where passages aren't too long
    one_answer = unpacked.loc[
        (unpacked.num_correct_answers == 1)
        & (unpacked.num_answers >= 4)
        & (unpacked.num_sentences <= 25)
    ].copy()

    # randomly shuffle answers and reformat
    np.random.seed(0)
    newcolnames = [
        "question",
        "question_id",
        "answer1",
        "answer2",
        "answer3",
        "answer4",
        "correct_answer",
        "correct_answer_num",
    ]

    def process_question(question):
        newcols = {"question": question["question"], "question_id": question["idx"]}
        answers = question["answers"]
        while len(answers) != 4 or (not any(a["isAnswer"] for a in answers)):
            answers = np.random.choice(question["answers"], 4, replace=False)

        for i in [1, 2, 3, 4]:
            newcols["answer" + str(i)] = answers[i - 1]["text"]
            if answers[i - 1]["isAnswer"]:
                newcols["correct_answer"] = answers[i - 1]["text"]
                newcols["correct_answer_num"] = i

        return pd.Series(newcols)

    one_answer[newcolnames] = one_answer.question_dict.apply(process_question)

    one_answer["dataset"] = "MultiRC"
    return one_answer[final_table_columns]


def process_mcscript(download_directory):
    """process the MCScript xml files and return Pandas df"""
    # unpack xml format to pandas table
    mc_script_dict = {}
    i = 0

    # only using train data, not taking dev or test set.
    xtree = et.parse(os.path.join(download_directory, f"mcscript/train-data.xml"))
    xroot = xtree.getroot()
    for node in xroot:
        passage_id = node.attrib.get("id")
        text = node.find("text").text
        # use NLTK sent tokenizer to count the number of sentences in the passage
        num_sentences = len(sent_tokenize(text))
        for q in node.find("questions"):
            mc_script_dict[i] = {
                "split": "train",
                "passage_id": passage_id,
                "passage": text,
                "question_id": q.attrib.get("id"),
                "question": q.attrib.get("text"),
                "num_sentences": num_sentences,
            }
            correct_answer = ""
            correct_ans_id = -1
            for ans in q:
                ans_id = ans.attrib.get("id")
                mc_script_dict[i]["answer_" + ans_id] = ans.attrib.get("text")
                if ans.attrib.get("correct") == "True":
                    correct_answer = mc_script_dict[i]["answer_" + ans_id]
                    correct_ans_id = ans_id
            if correct_ans_id == -1:
                print(mc_script_dict[i])
            mc_script_dict[i]["correct_answer"] = correct_answer
            mc_script_dict[i]["correct_answer_id"] = "answer_" + correct_ans_id
            i += 1
    mc_script_unpacked = pd.DataFrame.from_dict(mc_script_dict, orient="index")
    mc_script_unpacked = mc_script_unpacked.loc[mc_script_unpacked.num_sentences <= 25]

    # shuffle and reformat questions
    newcols = ["answer1", "answer2", "answer3", "answer4", "correct_answer_num"]

    def process_mcscript_row(row):
        new_dict = {}
        similar_rows = mc_script_unpacked.loc[
            (mc_script_unpacked.split == row.split)
            & (mc_script_unpacked.passage_id == row.passage_id)
            & (mc_script_unpacked.question_id != row.question_id)
        ]

        similar_answers = similar_rows[["answer_0", "answer_1"]].to_numpy().flatten()
        while len(new_dict.keys()) == 0:
            if len(similar_rows) == 0:
                two_ans = np.random.choice(
                    mc_script_unpacked.correct_answer, 2, replace=False
                )
            else:
                two_ans = np.random.choice(similar_answers, 2, replace=False)

            if (two_ans[0] in row[["answer_0", "answer_1"]]) or (
                two_ans[1] in row[["answer_0", "answer_1"]]
            ):
                continue
            new_ans = np.random.choice(
                np.concatenate([two_ans, row[["answer_0", "answer_1"]]]),
                4,
                replace=False,
            )
            for i in [1, 2, 3, 4]:
                new_dict["answer" + str(i)] = new_ans[i - 1]
                if new_ans[i - 1] == row["correct_answer"]:
                    new_dict["correct_answer_num"] = i
        return pd.Series(new_dict)

    np.random.seed(0)
    mc_script_unpacked[newcols] = mc_script_unpacked.apply(process_mcscript_row, axis=1)

    mc_script_unpacked["dataset"] = "MCScript2.0"
    return mc_script_unpacked[final_table_columns]


def process_mctest(download_directory):
    """process the MCTest tsv files and return Pandas df"""
    mc500_raw = {}
    # not using test split
    for split in ["train", "dev"]:
        raw_df = pd.read_csv(
            os.path.join(download_directory, f"MCTest/mc500.{split}.tsv"),
            sep="\t",
            names=[
                "mc500_id",
                "metadata",
                "passage",
                "question1",
                "MC_answer1.1",
                "MC_answer1.2",
                "MC_answer1.3",
                "MC_answer1.4",
                "question2",
                "MC_answer2.1",
                "MC_answer2.2",
                "MC_answer2.3",
                "MC_answer2.4",
                "question3",
                "MC_answer3.1",
                "MC_answer3.2",
                "MC_answer3.3",
                "MC_answer3.4",
                "question4",
                "MC_answer4.1",
                "MC_answer4.2",
                "MC_answer4.3",
                "MC_answer4.4",
            ],
        )
        ans_df = pd.read_csv(
            os.path.join(download_directory, f"MCTest/mc500.{split}.ans"),
            sep="\t",
            names=[
                "question1_answer",
                "question2_answer",
                "question3_answer",
                "question4_answer",
            ],
        )

        joined_df = raw_df.merge(ans_df, left_index=True, right_index=True)
        mc500_raw[split] = joined_df
    mc500_all_raw = pd.concat(mc500_raw.values())

    # extract answer values to correct format
    def get_answer_values(row, num):
        conversion = {"A": "1", "B": "2", "C": "3", "D": "4"}
        answer_column = (
            "MC_answer" + str(num) + "." + conversion[row[f"question{str(num)}_answer"]]
        )
        return row[answer_column]

    for num in [1, 2, 3, 4]:
        mc500_all_raw[f"question{str(num)}_answer"] = mc500_all_raw.apply(
            get_answer_values, args=[num], axis=1
        )

    # melt to get question and answer columns in one dataframe
    dfq = mc500_all_raw.melt(
        id_vars=["mc500_id", "passage"],
        value_vars=["question1", "question2", "question3", "question4"],
        value_name="question",
        var_name="question_number",
    )
    dfa1 = mc500_all_raw.melt(
        id_vars=["mc500_id", "passage"],
        value_vars=["MC_answer1.1", "MC_answer2.1", "MC_answer3.1", "MC_answer4.1"],
        value_name="MC_answer1",
    )
    dfa2 = mc500_all_raw.melt(
        id_vars=["mc500_id", "passage"],
        value_vars=["MC_answer1.2", "MC_answer2.2", "MC_answer3.2", "MC_answer4.2"],
        value_name="MC_answer2",
    )
    dfa3 = mc500_all_raw.melt(
        id_vars=["mc500_id", "passage"],
        value_vars=["MC_answer1.3", "MC_answer2.3", "MC_answer3.3", "MC_answer4.3"],
        value_name="MC_answer3",
    )
    dfa4 = mc500_all_raw.melt(
        id_vars=["mc500_id", "passage"],
        value_vars=["MC_answer1.4", "MC_answer2.4", "MC_answer3.4", "MC_answer4.4"],
        value_name="MC_answer4",
    )
    dfca = mc500_all_raw.melt(
        id_vars=["mc500_id", "passage"],
        value_vars=[
            "question1_answer",
            "question2_answer",
            "question3_answer",
            "question4_answer",
        ],
        value_name="correct_answer",
    )
    mc500_all = pd.concat(
        [
            dfq,
            dfa1.drop(["mc500_id", "passage", "variable"], axis=1),
            dfa2.drop(["mc500_id", "passage", "variable"], axis=1),
            dfa3.drop(["mc500_id", "passage", "variable"], axis=1),
            dfa4.drop(["mc500_id", "passage", "variable"], axis=1),
            dfca.drop(["mc500_id", "passage", "variable"], axis=1),
        ],
        axis=1,
    )

    # extract the prefix to the questions which details the number of sentences required in the passage to answer
    mc500_all["sent_required"] = mc500_all.question.str.split(":").str[0].str.strip()
    mc500_all["question"] = mc500_all.question.str.split(":").str[1].str.strip()

    # use NLTK sent tokenizer to count the number of sentences in the passage
    mc500_all["num_sentences"] = mc500_all.passage.apply(
        lambda x: sent_tokenize(x)
    ).str.len()

    def get_correct_answer_num(row):
        for i in [1, 2, 3, 4]:
            if row["MC_answer" + str(i)] == row["correct_answer"]:
                return i

    mc500_all["correct_answer_num"] = mc500_all.apply(get_correct_answer_num, axis=1)

    mc500_all["passage_id"] = mc500_all.mc500_id.apply(lambda x: x.split(".")[-1])
    mc500_all["question_id"] = mc500_all.question_number.str.replace("question", "")
    mc500_all["dataset"] = "MCTest_500"
    mc500_all["split"] = [a[1] for a in mc500_all.mc500_id.str.split(".")]
    return mc500_all.loc[mc500_all.num_sentences <= 25].rename(
        mapper=(lambda x: x.replace("MC_", "")), axis=1
    )[final_table_columns]


def process_race(download_directory):
    """process the RACE txt files and return Pandas df"""

    # unpack all the .txt files of the dataset into a single pandas table
    race_dict = {}
    i = 0
    for split in ["dev", "train"]:
        for level in ["middle", "high"]:
            for file in glob.glob(
                os.path.join(download_directory, f"RACE/{split}/{level}/*.txt")
            ):
                with open(file) as f:
                    file_str = f.read()
                file_dict = json.loads(file_str)
                num_sentences = len(sent_tokenize(file_dict["article"]))
                num_qs = len(file_dict["answers"])
                for q in range(num_qs):
                    race_dict[i] = {
                        "split": split,
                        "level": level,
                        "passage_id": file_dict["id"],
                        "passage": file_dict["article"],
                        "question_id": q,
                        "question": file_dict["questions"][q],
                        "num_sentences": num_sentences,
                    }

                    # rename answer columns
                    for j in range(len(file_dict["options"][q])):
                        race_dict[i]["answer" + str(j + 1)] = file_dict["options"][q][j]
                    race_dict[i]["correct_answer_num"] = (
                        ord(file_dict["answers"][q]) - 64
                    )
                    race_dict[i]["correct_answer"] = file_dict["options"][q][
                        race_dict[i]["correct_answer_num"] - 1
                    ]
                    i += 1
    race_unpacked = pd.DataFrame.from_dict(race_dict, orient="index")

    # remove fill-in-the-blank questions
    race_unpacked = race_unpacked.loc[~race_unpacked.question.str.contains("_")]

    race_unpacked["dataset"] = "RACE"
    return race_unpacked.loc[race_unpacked.num_sentences <= 25][final_table_columns]


def process_reclor(download_directory):
    """process the ReClor json files and return Pandas df"""
    # unpack the json format to into a pandas table
    reclor_dict = {}
    i = 0
    for split in ["train", "val"]:  # did not include test
        with open(os.path.join(download_directory, f"reclor/{split}.json")) as f:
            file_str = f.read()
        file_dict = json.loads(file_str)
        if split == "val":
            split = "dev"
        for item in file_dict:
            idx = item["id_string"].split("_")[-1]
            reclor_dict[i] = {
                "split": split,
                "passage_id": idx,
                "question_id": idx,
                "passage": item["context"],
                "question": item["question"],
            }
            for j in range(len(item["answers"])):
                reclor_dict[i]["answer" + str(j + 1)] = item["answers"][j]
            reclor_dict[i]["correct_answer_num"] = item["label"] + 1
            reclor_dict[i]["correct_answer"] = item["answers"][item["label"]]
            i += 1
    reclor_unpacked = pd.DataFrame.from_dict(reclor_dict, orient="index")

    reclor_unpacked["dataset"] = "ReClor"
    return reclor_unpacked[final_table_columns]


if __name__ == "__main__":
    os.environ["HTTPS_PROXY"] = "http://fwdproxy:8080"
    parser = argparse.ArgumentParser(
        description="Assemble samples from numerous datasets and generate a JSON to serve as the training set for Belebele"
    )
    parser.add_argument(
        "--data_path",
        help="Path to the json dataset",
    )
    parser.add_argument(
        "--downloads_path",
        help="Path to folder where all the files required to assemble the training set will be downloaded",
        default=".",
    )
    parser.add_argument(
        "--output_file",
        help="Path to file with the final training set (in tsv format)",
        default="belebele_training_set.tsv",
    )

    args = parser.parse_args()

    # the URLs to download
    urls = {
        "MultiRC": "https://cogcomp.seas.upenn.edu/multirc/data/mutlirc-v2.zip",
        "MCScript2.0": "https://fedora.clarin-d.uni-saarland.de/sfb1102/MCScript-2.0.zip",
        "MCTest": "https://mattr1.github.io/mctest/data/MCTest.zip",
        "RACE": "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz",
        "SciQ": "https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip",
        "ReClor": "https://github.com/yuweihao/reclor/releases/download/v1/reclor_data.zip",
    }

    # the name of the files once unzipped
    unzipped_filenames = {
        "MultiRC": "splitv2",
        "ReClor": "reclor",
        "RACE": "RACE",
        "SciQ": "SciQ dataset-2 3",
        "MCScript2.0": "mcscript",
        "MCTest": "MCTest",
    }

    downloads_repo = download_files(args.downloads_path, urls, unzipped_filenames)

    final_table_columns = [
        "dataset",
        "split",
        "passage_id",
        "question_id",
        "passage",
        "question",
        "answer1",
        "answer2",
        "answer3",
        "answer4",
        "correct_answer",
        "correct_answer_num",
    ]
    init_nltk_data()

    multirc_ready = process_multirc(downloads_repo)
    print("Finished processing MultiRC.")
    print("Starting to process MCScript2.0... this may take around 5 minutes")
    mcscript_ready = process_mcscript(downloads_repo)
    print("Finished processing MCScript2.0.")
    mctest_ready = process_mctest(downloads_repo)
    print("Finished processing MCTest.")
    sciq_ready = process_sciq(downloads_repo)
    print("Finished processing SciQ.")
    reclor_ready = process_reclor(downloads_repo)
    print("Finished processing ReClor.")
    race_ready = process_race(downloads_repo)
    print("Finished processing RACE... now joining them altogether.")

    combined = pd.concat(
        [
            sciq_ready,
            mcscript_ready,
            mctest_ready,
            multirc_ready,
            race_ready,
            reclor_ready,
        ]
    )
    combined.to_csv(args.output_file, sep="\t")
    print(f"Finished creating training set and dumped into {args.output_file}")
    print(
        "Beware when loading the data from the tsv, there are many newline characters, double quotes, single quotes, etc., especially in the RACE passages."
    )
