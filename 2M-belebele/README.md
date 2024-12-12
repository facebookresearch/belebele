# 2M-Belebele

## Highly-Multilingual Speech and American Sign Language Comprehension Dataset

We introduce **2M-Belebele** as the first highly multilingual speech and American Sign Language (ASL) comprehension dataset. Our dataset, which is an extension of the existing Belebele only-text dataset, covers 74 spoken languages at the intersection of Belebele and Fleurs, and one sign language (ASL). 

The speech dataset is built from aligning Belebele, Flores200 and Fleurs datasets as well as recording completely new audio for the sentences missing in Fleurs. We also provide new recordings for the Belebele question and answers as these are not in the original Flores200 dataset. 

Therefore, as a by-product, we also extend the Fleurs dataset (which is widely used to benchmark language identification and automatic speech recognition) by providing recordings for more Flores200 sentences than were previously available and adding sign language, creating a new **2M-Flores**. This 2M-Flores extends Fleurs by +20%.

The ASL dataset is built with completely new controlled recordings of ASL signers and each flores sentence as well as questions and answers are available in video format.

## Speech Dataset

The speech dataset can be downloaded with the provided script. The data is partitioned by language and stored in parquet format. You will need to download each partition:

`python dl_2mbelebele.py fra_Latn -o /tmp/belebele`

Here is sample code that you can use to load this dataset:

```
import pyarrow.parquet as pq  
import tempfile  
import polars as pl # polars is more efficient than pandas to load the audio data

from IPython.display import Audio  
from IPython.display import display as d  
import numpy as np

lang = 'eng_Latn'  
df = pl.read_parquet(f’/your2mbebelebeledownload/lang={lang}/{lang}.parquet’)

for r in df.sample(1).iter_rows(named=True):

   d(r['flores_passage'])  
   for seg, sent in zip(r['audio_segments'], r['flores_sentences']):  
       d(sent)  
       for a in seg:  
           d(Audio(data=np.array(a['audio']['wav'], dtype=np.float64), rate=a['audio']['sampling_rate']))    
       d('-----------------')

   d('QUESTION')  
   d(r['question'])  
   for a in r['question_audio']:  
       d(Audio(data=np.array(a['audio']['wav'], dtype=np.float64), rate=a['audio']['sampling_rate']))

    
   d('ANSWER 1')  
   d(r['mc_answer1'])  
   for a in r['answer_1_audio']:  
       d(Audio(data=np.array(a['audio']['wav'], dtype=np.float64), rate=a['audio']['sampling_rate']))  
        
   d('ANSWER 2')  
   d(r['mc_answer2'])  
   for a in r['answer_2_audio']:  
       d(Audio(data=np.array(a['audio']['wav'], dtype=np.float64), rate=a['audio']['sampling_rate']))

   d('ANSWER 3')  
   d(r['mc_answer3'])  
   for a in r['answer_3_audio']:  
       d(Audio(data=np.array(a['audio']['wav'], dtype=np.float64), rate=a['audio']['sampling_rate']))

    
   d('ANSWER 4')  
   d(r['mc_answer4'])  
   for a in r['answer_4_audio']:  
       d(Audio(data=np.array(a['audio']['wav'], dtype=np.float64), rate=a['audio']['sampling_rate']))
```

We are actively working to make this dataset available on huggingface.

### Languages in Belebele-speech

Note that for the speech version of 2M-Belebele, we have kept the original Flores200 dialect codes even if we are only talking about speech, this is to make it easier to align with Belebele and Flores.

| FLORES-200 Code | English Name | Family | Belebele | Belebele-Speech |
| :---- | :---- | :---- | :---- | :---- |
| acm_Arab | Mesopotamian Arabic | Afro-Asiatic | x |  |
| afr_Latn | Afrikaans | Germanic | x | x |
| als_Latn | Tosk Albanian | Paleo-Balkanic | x |  |
| amh_Ethi | Amharic | Afro-Asiatic | x | x |
| apc_Arab | North Levantine Arabic | Afro-Asiatic | x |  |
| arb_Arab | Modern Standard Arabic | Afro-Asiatic | x |  |
| arb_Latn | Modern Standard Arabic (Romanized) | Afro-Asiatic | x |  |
| ars_Arab | Najdi Arabic | Afro-Asiatic | x |  |
| ary_arab | Moroccan Arabic | Afro-Asiatic | x |  |
| arz_Arab | Egyptian Arabic | Afro-Asiatic | x | x |
| asm_Beng | Assamese | Indo-Aryan | x | x |
| azj_Latn | North Azerbaijani | Turkic | x | x |
| bam_Latn | Bambara | Mande | x |  |
| ben_Beng | Bengali | Indo-Aryan | x | x |
| ben_Latn^ | Bengali (Romanized) | Indo-Aryan | x |  |
| bod_Tibt | Standard Tibetan | Sino-Tibetan | x |  |
| bul_Cyrl | Bulgarian | Balto-Slavic | x | x |
| cat_Latn | Catalan | Romance | x | x |
| ceb_Latn | Cebuano | Austronesian | x | x |
| ces_Latn | Czech | Balto-Slavic | x | x |
| ckb_Arab | Central Kurdish | Iranian | x |  |
| dan_Latn | Danish | Germanic | x | x |
| deu_Latn | German | Germanic | x | x |
| ell_Grek | Greek | Hellenic | x | x |
| eng_Latn | English | Germanic | x | x |
| est_Latn | Estonian | Uralic | x |  |
| eus_Latn | Basque | Basque | x |  |
| fin_Latn | Finnish | Uralic | x | x |
| fra_Latn | French | Romance | x | x |
| fuv_Latn | Nigerian Fulfulde | Atlantic-Congo | x |  |
| gaz_Latn | West Central Oromo | Afro-Asiatic | x |  |
| grn_Latn | Guarani | Tupian | x |  |
| guj_Gujr | Gujarati | Indo-Aryan | x | x |
| hat_Latn | Haitian Creole | Atlantic-Congo | x |  |
| hau_Latn | Hausa | Afro-Asiatic | x | x |
| heb_Hebr | Hebrew | Afro-Asiatic | x | x |
| hin_Deva | Hindi | Indo-Aryan | x | x |
| hin_Latn^ | Hindi (Romanized) | Indo-Aryan | x |  |
| hrv_Latn | Croatian | Balto-Slavic | x | x |
| hun_Latn | Hungarian | Uralic | x | x |
| hye_Armn | Armenian | Armenian | x | x |
| ibo_Latn | Igbo | Atlantic-Congo | x |  |
| ilo_Latn | Ilocano | Austronesian | x |  |
| ind_Latn | Indonesian | Austronesian | x | x |
| isl_Latn | Icelandic | Germanic | x | x |
| ita_Latn | Italian | Romance | x | x |
| jav_Latn | Javanese | Austronesian | x | x |
| jpn_Jpan | Japanese | Japonic | x | x |
| kac_Latn | Jingpho | Sino-Tibetan | x |  |
| kan_Knda | Kannada | Dravidian | x |  |
| kat_Geor | Georgian | kartvelian | x | x |
| kaz_Cyrl | Kazakh | Turkic | x | x |
| kea_Latn | Kabuverdianu | Portuguese Creole | x | x |
| khk_Cyrl | Halh Mongolian | Mongolic | x | x |
| khm_Khmr | Khmer | Austroasiatic | x | x |
| kin_Latn | Kinyarwanda | Atlantic-Congo | x |  |
| kir_Cyrl | Kyrgyz | Turkic | x |  |
| kor_Hang | Korean | Koreanic | x | x |
| lao_Laoo | Lao | Kra-Dai | x |  |
| lin_Latn | Lingala | Atlantic-Congo | x |  |
| lit_Latn | Lithuanian | Balto-Slavic | x | x |
| lug_Latn | Ganda | Atlantic-Congo | x | x |
| luo_Latn | Luo | Nilo-Saharan | x | x |
| lvs_Latn | Standard Latvian | Balto-Slavic | x | x |
| mal_Mlym | Malayalam | Dravidian | x | x |
| mar_Deva | Marathi | Indo-Aryan | x |  |
| mkd_Cyrl | Macedonian | Balto-Slavic | x | x |
| mlt_Latn | Maltese | Afro-Asiatic | x |  |
| mri_Latn | Maori | Austronesian | x |  |
| mya_Mymr | Burmese | Sino-Tibetan | x | x |
| nld_Latn | Dutch | Germanic | x | x |
| nob_Latn | Norwegian Bokmål | Germanic | x | x |
| npi_Deva | Nepali | Indo-Aryan | x | x |
| npi_Latn^ | Nepali (Romanized) | Indo-Aryan | x | x |
| nso_Latn | Northern Sotho | Atlantic-Congo | x |  |
| nya_Latn | Nyanja | Afro-Asiatic | x |  |
| ory_Orya | Odia | Indo-Aryan | x | x |
| pan_Guru | Eastern Panjabi | Indo-Aryan | x | x |
| pbt_Arab | Southern Pashto | Indo-Aryan | x | x |
| pes_Arab | Western Persian | Iranian | x | x |
| plt_Latn | Plateau Malagasy | Austronesian | x |  |
| pol_Latn | Polish | Balto-Slavic | x | x |
| por_Latn | Portuguese | Romance | x |  |
| ron_Latn | Romanian | Romance | x |  |
| rus_Cyrl | Russian | Balto-Slavic | x |  |
| shn_Mymr | Shan | Kra-Dai | x |  |
| sin_Latn^ | Sinhala (Romanized) | Indo-Aryan | x |  |
| sin_Sinh | Sinhala | Indo-Aryan | x |  |
| slk_Latn | Slovak | Balto-Slavic | x | x |
| slv_Latn | Slovenian | Balto-Slavic | x | x |
| sna_Latn | Shona | Atlantic-Congo | x | x |
| snd_Arab | Sindhi | Indo-Aryan | x | x |
| som_Latn | Somali | Afro-Asiatic | x |  |
| sot_Latn | Southern Sotho | Atlantic-Congo | x |  |
| spa_Latn | Spanish | Romance | x | x |
| srp_Cyrl | Serbian | Balto-Slavic | x | x |
| ssw_Latn | Swati | Atlantic-Congo | x |  |
| sun_Latn | Sundanese | Austronesian | x |  |
| swe_Latn | Swedish | Germanic | x | x |
| swh_Latn | Swahili | Atlantic-Congo | x | x |
| tam_Taml | Tamil | Dravidian | x | x |
| tel_Telu | Telugu | Dravidian | x | x |
| tgk_Cyrl | Tajik | Iranian | x | x |
| tgl_Latn | Tagalog | Austronesian | x | x |
| tha_Thai | Thai | Kra-Dai | x | x |
| tir_Ethi | Tigrinya | Afro-Asiatic | x |  |
| tsn_Latn | Tswana | Atlantic-Congo | x |  |
| tso_Latn | Tsonga | Afro-Asiatic | x |  |
| tur_Latn | Turkish | Turkic | x | x |
| ukr_Cyrl | Ukrainian | Balto-Slavic | x |  |
| urd_Arab | Urdu | Indo-Aryan | x |  |
| urd_Latn^ | Urdu (Romanized) | Indo-Aryan | x | x |
| uzn_Latn | Northern Uzbek | Turkic | x |  |
| vie_Latn | Vietnamese | Austroasiatic | x | x |
| war_Latn | Waray | Austronesian | x |  |
| wol_Latn | Wolof | Atlantic-Congo | x | x |
| xho_Latn | Xhosa | Atlantic-Congo | x | x |
| yor_Latn | Yoruba | Atlantic-Congo | x | x |
| zho_Hans | Chinese (Simplified) | Sino-Tibetan | x | x |
| zho_Hant | Chinese (Traditional) | Sino-Tibetan | x |  |
| zsm_Latn | Standard Malay | Austronesian | x |  |
| zul_Latn | Zulu | Atlantic-Congo | x |  |

## ASL Belebele

We are currently preparing the ASL version of Belebele for download, it should be online before the end of 2024. If you are interested, contact [mortimer@meta.com](mailto:mortimer@meta.com) to be notified.

## Citation

If you use this data in your work, please cite 2M-Belebele paper as well as the original Belebele paper:

```
@article{2mbelebele,  
  author =        {Marta R. Costa-jussà and Bokai Yu and Pierre Andrews and Belen Alastruey and Necati Cihan Camgoz and Joe Chuang and Jean Maillard and Christophe Ropers and Arina Turkantenko and Carleigh Wood},  
  journal =       {Arxiv},  
url = {https://arxiv.org/abs/2412.08274},  
  title =         {{2M-BELEBELE}: Highly-Multilingual Speech and American Sign Language  
Comprehension Dataset},  
  year =          {2024},  
}

@inproceedings{bandarkar-etal-2024-belebele,  
    title = "The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants",  
    author = "Bandarkar, Lucas  and  
      Liang, Davis  and  
      Muller, Benjamin  and  
      Artetxe, Mikel  and  
      Shukla, Satya Narayan  and  
      Husa, Donald  and  
      Goyal, Naman  and  
      Krishnan, Abhinandan  and  
      Zettlemoyer, Luke  and  
      Khabsa, Madian",  
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",  
    month = aug,  
    year = "2024",  
    address = "Bangkok, Thailand and virtual meeting",  
    publisher = "Association for Computational Linguistics",  
    url = "https://aclanthology.org/2024.acl-long.44",  
    pages = "749--775",  
}
```

## License

2M-Belebele is released under CC-BY-SA4.0, it is composed of Flores200 (CC-BY-SA 4.0), belebele (CC-BY-SA4.0) and fleurs (cc-by-4.0).

