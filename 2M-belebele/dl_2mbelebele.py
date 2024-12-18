# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE_CC-BY-SA4.0 file in the root directory of this source tree.

from pathlib import Path
import argparse
import sys
import time
import urllib.request

ROOT = 'https://dl.fbaipublicfiles.com/2M-belebele/speech/version=1'

speech_langs = [
    "azj_Latn",
    "afr_Latn",
    "amh_Ethi",
    "asm_Beng",
    "arz_Arab",
    "bul_Cyrl",
    "ben_Beng",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "deu_Latn",
    "dan_Latn",
    "eng_Latn",
    "ell_Grek",
    "est_Latn",
    "fin_Latn",
    "guj_Gujr",
    "fra_Latn",
    "heb_Hebr",
    "hau_Latn",
    "hin_Deva",
    "hun_Latn",
    "hye_Armn",
    "ibo_Latn",
    "ind_Latn",
    "isl_Latn",
    "jav_Latn",
    "ita_Latn",
    "jpn_Jpan",
    "kat_Geor",
    "kaz_Cyrl",
    "khk_Cyrl",
    "khm_Khmr",
    "kea_Latn",
    "kor_Hang",
    "lug_Latn",
    "luo_Latn",
    "lit_Latn",
    "lvs_Latn",
    "mal_Mlym",
    "mkd_Cyrl",
    "nld_Latn",
    "mya_Mymr",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "pbt_Arab",
    "pes_Arab",
    "pol_Latn",
    "ron_Latn",
    "por_Latn",
    "rus_Cyrl",
    "slk_Latn",
    "slv_Latn",
    "sna_Latn",
    "snd_Arab",
    "spa_Latn",
    "srp_Cyrl",
    "swe_Latn",
    "swh_Latn",
    "tam_Taml",
    "tel_Telu",
    "tgk_Cyrl",
    "tha_Thai",
    "tur_Latn",
    "urd_Arab",
    "wol_Latn",
    "vie_Latn",
    "yor_Latn",
    "xho_Latn",
    "nob_Latn",
    "tgl_Latn",
    "zho_Hans",
]

def download_lang(lang: str, output_dir: Path):
    """
    download a parquet file for this lang, will be placed under output_dir/lang={lang}/{lang}.parquet
    """

    start_time = time.perf_counter()

    def reporthook(count, block_size, total_size):
        if count == 0:
            return
        duration = time.perf_counter() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r%s...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (lang, percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()

    uri = f'{ROOT}/lang={lang}/{lang}.parquet'
    dir = (output_dir / f'lang={lang}')
    dir.mkdir(parents=True, exist_ok=True)
    filename = dir / f'{lang}.parquet'
    urllib.request.urlretrieve(uri, filename, reporthook)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download parquet files for 2M-belebele speech.')
    parser.add_argument('langs', nargs='+', help='List of languages', default=speech_langs)
    parser.add_argument('-o', '--output-dir', required=True, help='Output directory')
    args = parser.parse_args()    
    out = Path(args.output_dir)
    for lang in args.langs:
        assert lang in speech_langs, f'{lang} is not available, you can download any or all of {", ".join(speech_langs)}'
        download_lang(lang, out)