# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE_CC-BY-SA4.0 file in the root directory of this source tree.

#############################################################################
#
# Alignment script for belebelfleurs
#
# run with uvx --with datasets --with numpy --with pandas --with stopes align.py
#
#############################################################################

import asyncio
from datasets import load_dataset
import numpy as np
import pandas as pd
import collections

from stopes.core.launcher import Launcher

from stopes.modules.partitioned_data_mapper import (
    BatchMapper,
    PartitionedDataMapper,
)

from stopes.modules.partitioned_data_mapper import PartitionedDataMapperConfig
from stopes.utils.sharding.parquet_shards import (
    ParquetShardingConfig,
    ParquetOutputConfig,
)
from stopes.core.stopes_module import Requirements

import typing as tp

from stopes.utils.sharding.abstract_shards import (
    BatchType,
)

# Load stopes launcher config
from stopes.pipelines import config_registry  # type: ignore


# lang mapping
flores2fleurs = {
    "afr_Latn": "af_za",
    "amh_Ethi": "am_et",
    "arz_Arab": "ar_eg",
    "asm_Beng": "as_in",
    "azj_Latn": "az_az",
    "ben_Beng": "bn_in",
    "bul_Cyrl": "bg_bg",
    "cat_Latn": "ca_es",
    "ceb_Latn": "ceb_ph",
    "ces_Latn": "cs_cz",
    "ckb_Arab": "ckb_iq",
    "dan_Latn": "da_dk",
    "deu_Latn": "de_de",
    "ell_Grek": "el_gr",
    "eng_Latn": "en_us",
    "est_Latn": "et_ee",
    "fin_Latn": "fi_fi",
    "fra_Latn": "fr_fr",
    "gaz_Latn": "om_et",
    "guj_Gujr": "gu_in",
    "hau_Latn": "ha_ng",
    "heb_Hebr": "he_il",
    "hin_Deva": "hi_in",
    "hrv_Latn": "hr_hr",
    "hun_Latn": "hu_hu",
    "hye_Armn": "hy_am",
    "ibo_Latn": "ig_ng",
    "ind_Latn": "id_id",
    "isl_Latn": "is_is",
    "ita_Latn": "it_it",
    "jav_Latn": "jv_id",
    "jpn_Jpan": "ja_jp",
    "kan_Knda": "kn_in",
    "kat_Geor": "ka_ge",
    "kaz_Cyrl": "kk_kz",
    "kea_Latn": "kea_cv",
    "khk_Cyrl": "mn_mn",
    "khm_Khmr": "km_kh",
    "kir_Cyrl": "ky_kg",
    "kor_Hang": "ko_kr",
    "lao_Laoo": "lo_la",
    "lin_Latn": "ln_cd",
    "lit_Latn": "lt_lt",
    "lug_Latn": "lg_ug",
    "luo_Latn": "luo_ke",
    "lvs_Latn": "lv_lv",
    "mal_Mlym": "ml_in",
    "mar_Deva": "mr_in",
    "mkd_Cyrl": "mk_mk",
    "mlt_Latn": "mt_mt",
    "mri_Latn": "mi_nz",
    "mya_Mymr": "my_mm",
    "nld_Latn": "nl_nl",
    "npi_Deva": "ne_np",
    "nya_Latn": "ny_mw",
    "ory_Orya": "or_in",
    "pan_Guru": "pa_in",
    "pbt_Arab": "ps_af",
    "pes_Arab": "fa_ir",
    "pol_Latn": "pl_pl",
    "por_Latn": "pt_br",
    "ron_Latn": "ro_ro",
    "rus_Cyrl": "ru_ru",
    "slk_Latn": "sk_sk",
    "slv_Latn": "sl_si",
    "sna_Latn": "sn_zw",
    "snd_Arab": "sd_in",
    "som_Latn": "so_so",
    "spa_Latn": "es_419",
    "srp_Cyrl": "sr_rs",
    "swe_Latn": "sv_se",
    "swh_Latn": "sw_ke",
    "tam_Taml": "ta_in",
    "tel_Telu": "te_in",
    "tgk_Cyrl": "tg_tj",
    "tha_Thai": "th_th",
    "tur_Latn": "tr_tr",
    "ukr_Cyrl": "uk_ua",
    "urd_Arab": "ur_pk",
    "uzn_Latn": "uz_uz",
    "vie_Latn": "vi_vn",
    "wol_Latn": "wo_sn",
    "xho_Latn": "xh_za",
    "yor_Latn": "yo_ng",
    "zul_Latn": "zu_za",
    "fuv_Latn": "ff_sn",
    "nob_Latn": "nb_no",
    "nso_Latn": "nso_za",
    "tgl_Latn": "fil_ph",
    "zho_Hans": "cmn_hans_cn",
    "zsm_Latn": "ms_my",
}


def load_fleurs(lang: str):
    """
    load fleurs and index entries by id.

    Args:
        lang: the language to load, in flores lang code format

    Returns:
        a dict mapping the fleurs id (same as flores) to a dict with:
            - audio: the audio wav nparray
            - sampling_rate: sr of that wav
            - gender: the fleurs gender of the speaker
            - fleurs_split: what fleurs split this entry comes from
    """
    flang = flores2fleurs[lang]

    print(f"loading fleurs for {flang}")
    try:
        fleurs_data = {
            "test ": load_dataset("google/fleurs", flang, split="test"),
            "valid": load_dataset("google/fleurs", flang, split="validation"),
            "train": load_dataset("google/fleurs", flang, split="train"),
        }
    except:
        print(f"download failed for {flang}")
        return None

    print(f"creating fleurs mapping {flang}")
    fleurs_entries = collections.defaultdict(list)
    for spl, ds in fleurs_data.items():
        for e in ds:
            simple = {
                "audio": np.array(e["audio"]["array"]),
                "sampling_rate": e["audio"]["sampling_rate"],
                "gender": e["gender"],
                "fleurs_split": spl,
            }
            fleurs_entries[e["id"]].append(simple)

    return fleurs_entries


def load_flores_ids(lang):
    flores_devtest, flores_dev = load_dataset(
        "facebook/flores", lang, split=["devtest", "dev"]
    )

    id2entry = collections.defaultdict(list)

    for entry in flores_devtest:
        id2entry[entry["id"]] = entry
    for entry in flores_dev:
        id2entry[entry["id"]] = entry

    return id2entry


def load_flores_urls(lang):
    """
    load flores and index it by urls/links

    Returns:
        a dict linking flores urls/link to a tupple with the flores entry and what split it's from
    """
    flores_devtest, flores_dev = load_dataset(
        "facebook/flores", lang, split=["devtest", "dev"]
    )

    url2flores = collections.defaultdict(list)

    for entry in flores_devtest:
        url2flores[entry["URL"]].append((entry, "devtest"))
    for entry in flores_dev:
        url2flores[entry["URL"]].append((entry, "dev"))

    return url2flores


def find_flores(row, url2flores):
    """
    given a belebele dataframe row, align the passage to flores

    returns a dict for the row with the following entries:
        - flores_ids = list of ids from flores, in order of the passage
        - flores_sentences = text of each flores segment, in order of the passage
        - flores_split = what flores split this came from
    """
    flores = list(url2flores[row["link"]])
    passage = row["flores_passage"].lower().replace('""', '"').lstrip(r"\W+").lstrip()
    row["flores_entries_unordered"] = flores
    i = 0
    found = []
    ids = []
    split = "none"
    while len(flores) and len(passage) and i < len(flores):
        entry, split = flores[i]
        sent = entry["sentence"].lower()

        if passage.startswith(sent) or passage.startswith(f'"{sent}'):
            # we found the first sentence of the passage
            del flores[i]
            passage_len = len(passage)
            passage = passage.removeprefix(sent)
            if passage_len == len(passage):
                passage = passage.removeprefix(f'"{sent}')

            passage = passage.lstrip(r"\W+").lstrip()
            ids.append(entry["id"])
            found.append(sent)
            i = 0
            if not len(passage):
                return {
                    "flores_ids": ids,
                    "flores_sentences": found,
                    "flores_split": split,
                }
        else:
            i += 1

    if passage.lstrip('"'):
        print(f"notfound: {passage}")
    return {
        "flores_ids": ids,
        "flores_sentences": found,
        "flores_split": split,
    }


def remap_sentences(matched_flores, id2flores):
    """
    matched_flores has english sentences, we want to replace them by the corret lang sentences
    """
    sentences = [id2flores.get(id)["sentence"] for id in matched_flores["flores_ids"]]
    matched_flores["flores_sentences"] = sentences
    return matched_flores


def match_fleurs(row, fleurs_entries, id2flores, link2flores):
    """
    given a belebele row, find the fleures entries, if some are missing, set the `fleurs_entries` to [] otherwise
    it will contain the list of entries (in correct order). The column `has_fleurs_matched`
    """
    flores = link2flores[row["link"]]
    flores_ids = flores["flores_ids"]
    row["flores"] = remap_sentences(flores, id2flores)
    matches = [fleurs_entries.get(id, None) for id in flores_ids]
    fail = any(element is None for element in matches)

    if fail:
        row["fleurs_entries"] = []
        row["has_fleurs_matched"] = False
        return row

    row["fleurs_entries"] = matches
    row["has_fleurs_matched"] = True
    return row


def align_flores(belebele):
    """
    Aligns the corresponding flores information to each row in the belbele dataframe. We only do it for English as all the other
    languages have the same alignment.

    Args:
        belebele (pd.DataFrame): A dataframe containing the original  belebele dataset.

    Returns:
        dict: A dictionary with belebele links as keys and corresponding flores information as values."""
    print("aligning flores for english")
    eng = belebele[belebele["dialect"] == "eng_Latn"]
    url2flores = load_flores_urls("eng_Latn")
    link2flores = {}
    for _idx, row in eng.iterrows():
        flores_info = find_flores(row, url2flores)
        link2flores[row["link"]] = flores_info

    return link2flores


class AlignFleurs(BatchMapper):
    def __init__(self, link2flores) -> None:
        self.link2flores = link2flores

    def __call__(self, batch: BatchType) -> pd.DataFrame:
        """
        Add Fleurs entries to the given batch of data.

        Args:
            batch (BatchType): The batch of data to add fleurs entries to.

        Returns:
            pd.DataFrame: A dataframe containing the original and newly added columns for each row."""
        if not isinstance(batch, pd.DataFrame):
            batch = batch.to_pandas()

        # make sure there is only one language in the batch and get that language
        lang = batch["dialect"].unique()
        assert len(lang) == 1, f"too many langs in shard: {lang}"
        lang = lang[0]

        # we do not have fleurs data for this language
        if lang not in flores2fleurs:
            batch["fleurs_entries"] = batch["link"].apply(lambda _: list())
            return batch

        # try to download the fleurs dataset
        fleurs_entries = load_fleurs(lang)
        if not fleurs_entries:
            print(f"couldn't download fleurs for {lang}")
            batch["fleurs_entries"] = batch["link"].apply(lambda _: list())
            return batch

        id2flores = load_flores_ids(lang)

        # apply match_fleurs to the dataframe
        return batch.apply(
            match_fleurs,
            axis=1,
            fleurs_entries=fleurs_entries,
            id2flores=id2flores,
            link2flores=self.link2flores,
        )


class Belebele2FleursMapper(PartitionedDataMapper):
    """
    a Data mapper that we can launch on slurm and will add
    fleurs and flores columns to the belebele dataset.
    """

    def get_custom_metadata(self, *args, **kwargs) -> tp.Dict[str, tp.Any]:
        return {}

    def requirements(self):
        return Requirements(
            nodes=1,
            mem_gb=30,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=4,
        )

    def get_batch_mapper(self):
        belebele = pd.read_parquet(
            "/checkpoint/artyomko/meres/datasets/belebele",
            filters=[("dialect", "in", ["eng_Latn"])],
        )
        link2flores = align_flores(belebele)
        return AlignFleurs(link2flores)


def main():
    launcher = Launcher(
        cache=None,
        cluster="local",
    )
    langs = ",".join([f'"{lang}"' for lang in flores2fleurs])
    config = PartitionedDataMapperConfig(
        input_dataset_config=ParquetShardingConfig(
            input_file="~/datasets/belebele",
            filters_expr=f"pc.field('dialect').isin([{langs}])",
        ),
        output_dataset_config=ParquetOutputConfig(
            dataset_path="~/datasets/belebele_fleurs",
            keep_same_partitioning=True,
        ),
    )
    asyncio.run(launcher.schedule(Belebele2FleursMapper(config)))


if __name__ == "__main__":
    main()