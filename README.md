--------------------------------------------------------------------------------

# The Belebele Benchmark for Massively Multilingual NLU Evaluation

Belebele is a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. This dataset enables the evaluation of mono- and multi-lingual models in high-, medium-, and low-resource languages. Each question has four multiple-choice answers and is linked to a short passage from the [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200) dataset. The human annotation procedure was carefully curated to create questions that discriminate between different levels of generalizable language comprehension and is reinforced by extensive quality checks. While all questions directly relate to the passage, the English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. Belebele opens up new avenues for evaluating and analyzing the multilingual abilities of language models and NLP systems.

Please refer to our paper for more details, [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants](https://arxiv.org/abs/2308.16884).


## Composition

- 900 questions per language variant
- 488 distinct passages, there are 1-2 associated questions for each.
- For each question, there is 4 multiple-choice answers, exactly 1 of which is correct.
- 122 language/language variants (including English).
- 900 x 122 = 109,800 total questions.


--------------------------------------------------------------------------------

## Download

Belebele can be downloaded [here](https://dl.fbaipublicfiles.com/belebele/Belebele.zip) which you can download with the following command:

```bash
wget --trust-server-names https://dl.fbaipublicfiles.com/belebele/Belebele.zip
```

## Formatting details

- The `link` and `split` uniquely identifies a passage.
- The combination of passage (`link` and `split`) and `question_number` (either 1 or 2) uniquely identifies a question.
- The language of each row is denoted in the `dialect` column with the FLORES-200 code (see Languages below)
- The `correct_answer_num` is one-indexed (e.g. a value of `2` means `mc_answer2` is correct)

--------------------------------------------------------------------------------

## Pausible Evaluation Settings

Thanks to the parallel nature of the dataset and the simplicity of the task, there are many possible settings in which we can evaluate language models. In all evaluation settings, the metric of interest is simple accuracy (# correct / total).

Evaluating models on Belebele in English can be done via finetuning, few-shot, or zero-shot. For other target languages, we propose the incomprehensive list of evaluation settings below. Settings that are compatible with evaluating non-English models (monolingual or cross-lingual) are denoted with `^`.

#### No finetuning
- **Zero-shot with natural language instructions (English instructions)**
    - For chat-finetuned models, we give it English instructions for the task and the sample in the target language in the same input.
    - For our experiments, we instruct the model to provide the letter `A`, `B`, `C`, or `D`. We perform post-processing steps and accept answers predicted as e.g. `(A)` instead of `A`. We sometimes additionally remove the prefix `The correct answer is` for predictions that do not start with one of the four accepted answers.
- **Zero-shot with natural language instructions (translated instructions)**^
    - Same as above, except the instructions are translated to the target language so that the instructions and samples are in the same language. The instructions can be human or machine-translated.
- **Few-shot in-context learning (English examples)**
    - A few samples (e.g. 5) are taken from the English training set (see below) and prompted to the model. Then, the model is evaluated with the same template but with the passages, questions, and answers in the target language.
    - For our experiments, we use the template: ```P: <passage> \n Q: <question> \n A: <mc answer 1> \n B: <mc answer 2> \n  C: <mc answer 3> \n  D: <mc answer 4> \n  Answer: <Correct answer letter>```. We perform prediction by picking the answer within `[A, B, C, D]` that has the highest probability relatively to the others.
- **Few-shot in-context learning (translated examples)**^
    - Same as above, except the samples from the training set are translated to the target language so that the examples and evaluation data are in the same language. The training samples can be human or machine-translated.


#### With finetuning
- **English finetune & multilingual evaluation**
    - The model is finetuned to the task using the English training set, probably with a sequence classification head. Then the model is evaluated in all the target languages individually.
- **English finetune & cross-lingual evaluation**
    - Same as above, except the model is evaluated in a cross-lingual setting, where for each question, the passage & answers could be provided in a different language. For example, passage could be in language `x`, question in language `y`, and answers in language `z`.
- **Translate-train**^
    - For each target language, the model is individually finetuned on training samples that have been machine-translated from English to that language. Each model is then evaluated in the respective target language.
- **Translate-train-all**
    - Similar to above, except here the model is trained on translated samples from all target languages at once. The single finetuned model is then evaluated on all target languages.
- **Translate-train-all & cross-lingual evaluation**
    - Same as above, except the single finetuned model is evaluated in a cross-lingual setting, where for each question, the passage & answers could be provided in a different language.
- **Translate-test**
    - The model is finetuned using the English training data and then the evaluation dataset is machine-translated to English and evaluated on the English.
    - This setting is primarily a reflection of the quality of the machine translation system, but is useful for comparison to multilingual models.



In addition, there are 83 additional languages in FLORES-200 for which questions were not translated for Belebele. Since the passages exist in those target languages, machine-translating the questions & answers may enable decent evaluation of machine reading comprehension in those languages.

--------------------------------------------------------------------------------

## Training Set

As discussed in the paper, we also provide an assembled training set consisting of samples

The Belebele dataset is intended to be used only as a test set, and not for training or validation. Therefore, for models that require additional task-specific training, we instead propose using an assembled training set consisting of samples from pre-existing multiple-choice QA datasets in English. We considered diverse datasets, and determine the most compatible to be [RACE](https://www.cs.cmu.edu/~glai1/data/race/), [SciQ](https://allenai.org/data/sciq), [MultiRC](https://cogcomp.seas.upenn.edu/multirc/), [MCTest](https://mattr1.github.io/mctest/), [MCScript2.0](https://aclanthology.org/S19-1012/), and [ReClor](https://whyu.me/reclor/).

For each of the six datasets, we unpack and restructure the passages and questions from their respective formats. We then filter out less suitable samples (e.g. questions with multiple correct answers). In the end, the dataset comprises 67.5k training samples and 3.7k development samples, more than half of which are from RACE. We provide a script (`assemble_training_set.py`) to reconstruct this dataset for anyone to perform task finetuning.

Since the training set is a joint sample of other datasets, it is governed by a different license. We do not claim any of that work or datasets to be our own. See the Licenses section.

--------------------------------------------------------------------------------

## Languages in Belebele

FLORES-200 Code | English Name | Script | Family
---|---|---|---
acm_Arab | Mesopotamian Arabic | Arab | Afro-Asiatic
afr_Latn | Afrikaans | Latn | Germanic
als_Latn | Tosk Albanian | Latn | Paleo-Balkanic
amh_Ethi | Amharic | Ethi | Afro-Asiatic
apc_Arab | North Levantine Arabic | Arab | Afro-Asiatic
arb_Arab | Modern Standard Arabic | Arab | Afro-Asiatic
arb_Latn | Modern Standard Arabic (Romanized) | Latn | Afro-Asiatic
ars_Arab | Najdi Arabic | Arab | Afro-Asiatic
ary_arab | Moroccan Arabic | Arab | Afro-Asiatic
arz_Arab | Egyptian Arabic | Arab | Afro-Asiatic
asm_Beng | Assamese | Beng | Indo-Aryan
azj_Latn | North Azerbaijani | Latn | Turkic
bam_Latn | Bambara | Latn | Mande
ben_Beng | Bengali | Beng | Indo-Aryan
ben_Latn^ | Bengali (Romanized) | Latn | Indo-Aryan
bod_Tibt | Standard Tibetan | Tibt | Sino-Tibetan
bul_Cyrl | Bulgarian | Cyrl | Balto-Slavic
cat_Latn | Catalan | Latn | Romance
ceb_Latn | Cebuano | Latn | Austronesian
ces_Latn | Czech | Latn | Balto-Slavic
ckb_Arab | Central Kurdish | Arab | Iranian
dan_Latn | Danish | Latn | Germanic
deu_Latn | German | Latn | Germanic
ell_Grek | Greek | Grek | Hellenic
eng_Latn | English | Latn | Germanic
est_Latn | Estonian | Latn | Uralic
eus_Latn | Basque | Latn | Basque
fin_Latn | Finnish | Latn | Uralic
fra_Latn | French | Latn | Romance
fuv_Latn | Nigerian Fulfulde | Latn | Atlantic-Congo
gaz_Latn | West Central Oromo | Latn | Afro-Asiatic
grn_Latn | Guarani | Latn | Tupian
guj_Gujr | Gujarati | Gujr | Indo-Aryan
hat_Latn | Haitian Creole | Latn | Atlantic-Congo
hau_Latn | Hausa | Latn | Afro-Asiatic
heb_Hebr | Hebrew | Hebr | Afro-Asiatic
hin_Deva | Hindi | Deva | Indo-Aryan
hin_Latn^ | Hindi (Romanized) | Latn | Indo-Aryan
hrv_Latn | Croatian | Latn | Balto-Slavic
hun_Latn | Hungarian | Latn | Uralic
hye_Armn | Armenian | Armn | Armenian
ibo_Latn | Igbo | Latn | Atlantic-Congo
ilo_Latn | Ilocano | Latn | Austronesian
ind_Latn | Indonesian | Latn | Austronesian
isl_Latn | Icelandic | Latn | Germanic
ita_Latn | Italian | Latn | Romance
jav_Latn | Javanese | Latn | Austronesian
jpn_Jpan | Japanese | Jpan | Japonic
kac_Latn | Jingpho | Latn | Sino-Tibetan
kan_Knda | Kannada | Knda | Dravidian
kat_Geor | Georgian | Geor | kartvelian
kaz_Cyrl | Kazakh | Cyrl | Turkic
kea_Latn | Kabuverdianu | Latn | Portuguese Creole
khk_Cyrl | Halh Mongolian | Cyrl | Mongolic
khm_Khmr | Khmer | Khmr | Austroasiatic
kin_Latn | Kinyarwanda | Latn | Atlantic-Congo
kir_Cyrl | Kyrgyz | Cyrl | Turkic
kor_Hang | Korean | Hang | Koreanic
lao_Laoo | Lao | Laoo | Kra-Dai
lin_Latn | Lingala | Latn | Atlantic-Congo
lit_Latn | Lithuanian | Latn | Balto-Slavic
lug_Latn | Ganda | Latn | Atlantic-Congo
luo_Latn | Luo | Latn | Nilo-Saharan
lvs_Latn | Standard Latvian | Latn | Balto-Slavic
mal_Mlym | Malayalam | Mlym | Dravidian
mar_Deva | Marathi | Deva | Indo-Aryan
mkd_Cyrl | Macedonian | Cyrl | Balto-Slavic
mlt_Latn | Maltese | Latn | Afro-Asiatic
mri_Latn | Maori | Latn | Austronesian
mya_Mymr | Burmese | Mymr | Sino-Tibetan
nld_Latn | Dutch | Latn | Germanic
nob_Latn | Norwegian Bokmål | Latn | Germanic
npi_Deva | Nepali | Deva | Indo-Aryan
npi_Latn^ | Nepali (Romanized) | Latn | Indo-Aryan
nso_Latn | Northern Sotho | Latn | Atlantic-Congo
nya_Latn | Nyanja | Latn | Afro-Asiatic
ory_Orya | Odia | Orya | Indo-Aryan
pan_Guru | Eastern Panjabi | Guru | Indo-Aryan
pbt_Arab | Southern Pashto | Arab | Indo-Aryan
pes_Arab | Western Persian | Arab | Iranian
plt_Latn | Plateau Malagasy | Latn | Austronesian
pol_Latn | Polish | Latn | Balto-Slavic
por_Latn | Portuguese | Latn | Romance
ron_Latn | Romanian | Latn | Romance
rus_Cyrl | Russian | Cyrl | Balto-Slavic
shn_Mymr | Shan | Mymr | Kra-Dai
sin_Latn^ | Sinhala (Romanized) | Latn | Indo-Aryan
sin_Sinh | Sinhala | Sinh | Indo-Aryan
slk_Latn | Slovak | Latn | Balto-Slavic
slv_Latn | Slovenian | Latn | Balto-Slavic
sna_Latn | Shona | Latn | Atlantic-Congo
snd_Arab | Sindhi | Arab | Indo-Aryan
som_Latn | Somali | Latn | Afro-Asiatic
sot_Latn | Southern Sotho | Latn | Atlantic-Congo
spa_Latn | Spanish | Latn | Romance
srp_Cyrl | Serbian | Cyrl | Balto-Slavic
ssw_Latn | Swati | Latn | Atlantic-Congo
sun_Latn | Sundanese | Latn | Austronesian
swe_Latn | Swedish | Latn | Germanic
swh_Latn | Swahili | Latn | Atlantic-Congo
tam_Taml | Tamil | Taml | Dravidian
tel_Telu | Telugu | Telu | Dravidian
tgk_Cyrl | Tajik | Cyrl | Iranian
tgl_Latn | Tagalog | Latn | Austronesian
tha_Thai | Thai | Thai | Kra-Dai
tir_Ethi | Tigrinya | Ethi | Afro-Asiatic
tsn_Latn | Tswana | Latn | Atlantic-Congo
tso_Latn | Tsonga | Latn | Afro-Asiatic
tur_Latn | Turkish | Latn | Turkic
ukr_Cyrl | Ukrainian | Cyrl | Balto-Slavic
urd_Arab | Urdu | Arab | Indo-Aryan
urd_Latn^ | Urdu (Romanized) | Latn | Indo-Aryan
uzn_Latn | Northern Uzbek | Latn | Turkic
vie_Latn | Vietnamese | Latn | Austroasiatic
war_Latn | Waray | Latn | Austronesian
wol_Latn | Wolof | Latn | Atlantic-Congo
xho_Latn | Xhosa | Latn | Atlantic-Congo
yor_Latn | Yoruba | Latn | Atlantic-Congo
zho_Hans | Chinese (Simplified) | Hans | Sino-Tibetan
zho_Hant | Chinese (Traditional) | Hant | Sino-Tibetan
zsm_Latn | Standard Malay | Latn | Austronesian
zul_Latn | Zulu | Latn | Atlantic-Congo

^ denotes a language variant not in FLORES-200

## Further Stats

- 122 language variants, but 115 distinct languages (ignoring scripts)
- 27 language families
- 29 scripts
- Avg. words per passage = 79.1 (std = 26.2)
- Avg. sentences per passage = 4.1 (std = 1.4)
- Avg. words per question = 12.9(std = 4.0)
- Avg. words per answer = 4.2 (std = 2.9)

--------------------------------------------------------------------------------

## License

The Belebele dataset is licensed under the license found in the LICENSE_CC-BY-SA4.0 file in the root directory of this source tree.

The training set and assembly code is, however, licensed differently. The majority of the training set (data and code) is licensed under CC-BY-NC, however portions of the project are available under separate license terms: NLTK is licensed under the Apache 2.0 license; pandas and NumPy are licensed under the BSD 3-Clause License.

## Citation

If you use this data in your work, please cite:

```bibtex
@article{bandarkar2023belebele,
      title={The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants}, 
      author={Lucas Bandarkar and Davis Liang and Benjamin Muller and Mikel Artetxe and Satya Narayan Shukla and Donald Husa and Naman Goyal and Abhinandan Krishnan and Luke Zettlemoyer and Madian Khabsa},
      year={2023},
      journal={arXiv preprint arXiv:2308.16884}
}
```
