# External data: what is used, where it comes from, and what is shipped

Four kinds of input are not redistributed:
- model weights;
- generated structures;
- large public deposits that can be fetched by URL and hash (the RMDB RDAT files);
- third-party files whose licence for redistribution is not stated.

Each is identified here by accession or URL and by sha256. Every file that belongs to a shipped
directory but is not in the execution pack is also listed one by one in the pack's
`EXCLUDED_FILES.json`, with its pack path, sha256, bytes, reason and source.

## RNAbpFlow pretrained checkpoint (not shipped: weights)

| Item | Value |
|---|---|
| File | `RNA3DB.ckpt`, 203,034,292 bytes |
| sha256 | `bb598a736193734541f63c59dae75d45db8c04f01a8414a758677367c4b1e213` |
| Source | Member `checkpoint/RNA3DB.ckpt` of `model.tar.gz` in Zenodo record 18305861 (https://zenodo.org/records/18305861, doi:10.5281/zenodo.18305861, "RNAbpFlow trained checkpoints", licence CC-BY-4.0 per record metadata) |
| Archive | `model.tar.gz`, 558,025,173 bytes; sha256 `7f78050f7869c99511310d86a2a8de0249797967ded8de8893db0f31cb705d6f`; MD5 `f79ce1555126ee77651594b65a95cb4b` |
| Download | https://zenodo.org/api/records/18305861/files/model.tar.gz/content |
| Used by | All training packets (generator initialisation) and Wave D generation |

## RNAbpFlow training archive and feature files (feature files shipped, CC-BY-4.0)

| Item | Value |
|---|---|
| Archive | `RNAbpFlow-Train.tar.gz`, 681,589,956 bytes |
| sha256 | `89280784c053410aeba3255701ea46c0a33b70d89a1c92966d7271212a1b5b44` (MD5 `4b7015fe0fbdd0ba88c752cdb19c0758`) |
| Source | Zenodo record 19388910, "RNAbpFlow training data and code" (https://zenodo.org/records/19388910, doi:10.5281/zenodo.19388910) |
| Licence | CC-BY-4.0, per the record metadata. Attribution and citation are in `NOTICE.md` |

The training packets contain 20 distinct feature pickles from this archive, 64 copies across the
packets. They are **shipped** at their packet paths, so the packets are complete. Each was checked
byte-identical to the archive member when the pack was built:

| Archive member (under `RNAbpFlow_training/`) | Bytes | sha256 | Packets |
|---|---:|---|---|
| `Feature/train_pkl/1e8o_E/1e8o_E.pkl` | 115403 | `09a0c9e1eda657ac1ac73cf717209f0ac779044df5aaf42eea2c1bcaa55dab4d` | e8ff330ee775 |
| `Feature/train_pkl/1g59_B/1g59_B.pkl` | 217437 | `82de333976e2ed37a2712449efa15f0fdde28b4740a5bbe71c21bee730db87a7` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 |
| `Feature/train_pkl/1grz_B/1grz_B.pkl` | 1732594 | `bac16e1c1683807f1d4e8f2d9eaf5de6e4ac054ca6160504f605f5867526f62e` | e8ff330ee775 |
| `Feature/train_pkl/1l8v_B/1l8v_B.pkl` | 762664 | `b9eca106feded27a59a85ef173105cbf2cef3c60ce0dd74ea00ba3fdce3abdb6` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 |
| `Feature/train_pkl/2r8s_R/2r8s_R.pkl` | 779994 | `8b415065241cd72ddff2db64394c8650fc8ea8aa22ac1c40fcbbf5b46268362d` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 |
| `Feature/train_pkl/5zwm_B/5zwm_B.pkl` | 925546 | `dfa9ee5ba11643fcb733bfbe6660245372667abb5ee2c018e049f02e4e0a0b77` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 |
| `Feature/train_pkl/6cc1_B/6cc1_B.pkl` | 349462 | `510f11f2f544ae534dc199237668c00da38197d0a6f7ea246fdee76b91caad56` | e8ff330ee775 |
| `Feature/train_pkl/6chr_A/6chr_A.pkl` | 9928083 | `b7d61eb4e8547af0d637681669d25329c6e19e78ae6d439be8657b9fdf41a40e` | e8ff330ee775 b30ccfbccde2 |
| `Feature/train_pkl/6t3n_A/6t3n_A.pkl` | 4152976 | `d40bdf464b8b96487c4b019cbfd77e3eba7f3a31a654c325432def6d061631ed` | e8ff330ee775 |
| `Feature/train_pkl/7ptk_B/7ptk_B.pkl` | 13221327 | `993d8dd5bfff7d5722f00dda8b5cb20cc1553b386d0ef5652b3704d6f332bb16` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 b30ccfbccde2 |
| `pkls/1e8o_E_base/1e8o_E_feat.pkl` | 1779310 | `9d848fee4794dc658f1c66017eebc116aea9ce0fb756875cfeffb651e3af5853` | e8ff330ee775 |
| `pkls/1g59_B_base/1g59_B_feat.pkl` | 2674327 | `8b43fad29f0eee1cd2f9c4893ea53b0af81823c8d288a4773a9f2a3ee608dcf8` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 |
| `pkls/1grz_B_base/1grz_B_feat.pkl` | 9103066 | `fe1077924763c211e5e3c4e1283607f34e02d82e6a022592afc4f533d9066113` | e8ff330ee775 |
| `pkls/1l8v_B_base/1l8v_B_feat.pkl` | 5680177 | `168e2e50c1e8e832c44dfc352bb1d92cf9d9c270b6c254899b735fe2c1c38d06` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 |
| `pkls/2r8s_R_base/2r8s_R_feat.pkl` | 5754833 | `199df705a8f6b4ada70e660b098a9de4d487693301392ceb8e1eaa7d47cc93ce` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 |
| `pkls/5zwm_B_base/5zwm_B_feat.pkl` | 6354385 | `507f12c89dfbc76b54c5b5e50087801c8b0e6ffd989df054b5a5fb04ad1d1d51` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 |
| `pkls/6cc1_B_base/6cc1_B_feat.pkl` | 3579348 | `bfb5cc541e4532a618ce23021144d7e908190043db6fcfa30b9988ec1a9d16ff` | e8ff330ee775 |
| `pkls/6chr_A_base/6chr_A_feat.pkl` | 24715451 | `e888d73c199f16f519f626ed2df383615751b85d2063621e85a59149cea1c4c3` | e8ff330ee775 b30ccfbccde2 |
| `pkls/6t3n_A_base/6t3n_A_feat.pkl` | 14972574 | `898200c249cf070224656b66d5bd92da248fed5b76d3667c12b7bf448079aa62` | e8ff330ee775 |
| `pkls/7ptk_B_base/7ptk_B_feat.pkl` | 29222750 | `15847c489a4c2781a1d35a35f0d57aceb1805adbe7ec7175bd1dc8e64289e73b` | b63550bd42ee e8ff330ee775 32e2cc0f8974 670450638bf1 aa8af773fd10 b30ccfbccde2 |

The packets also contain small metadata tables taken from the same archive.
`RNAbpFlow_training/metadata/pilot_ensemble.csv` holds rows of `metadata/RNA3DB_train.csv`, whose
sha256 is `9e966524e5ee30347f3e303886c0a104683d446d9a845a8675634669c1171f1a`.
`waveD_cases_metadata.csv` holds rows of `metadata/CASP16_train.csv`.

## Wave D case inputs (shipped; derived from the archive above)

`waveD_efec09f1317a/cases/<id>/case_input.npz` (256 files, about 13 MB) holds:
- the three training base-pair channels;
- the native atom28 coordinates and their mask.

They were extracted with a restricted unpickler from `Feature/train_pkl/<id>/<id>.pkl` in the
archive. The extractor is `provenance/prepare/build_packet.py`. `case_table.json` records, for
every case:
- the archive member names and sha256 of both source pickles;
- the length, bin, origin and node/GPU slot.

## RNA-FM (not shipped: code installed from PyPI, weights)

| Item | Value |
|---|---|
| Code | PyPI `rna-fm` 0.2.2, wheel sha256 `4fff72892ad02559be969b7b3263483e7f3730132025ea05c9927c3675a5e0b4` (MIT); `fm/` byte-identical to https://github.com/ml4bio/RNA-FM commit `348951516e0963d22bbb33b3c9fc18c89081d38e` |
| Import-time dependency | `ptflops` 0.7.5, wheel sha256 `c47543d5209a21e0805b39a50d452c8f8f6f5b648684777490db6f9459bdda07` |
| Weights | `RNA-FM_pretrained.pth`, 1,194,424,423 bytes, sha256 `5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99` |
| Weights source | https://huggingface.co/cuhkaih/rnafm/resolve/91d4a46d28d8054a7b429955e8fc0c253ba0afd6/RNA-FM_pretrained.pth (repository named in the RNA-FM README; model card declares Apache-2.0) |
| Record | `src/sequence_prior/rnafm_sources.json` |

## Rfam 15.1 (shipped as the derived Wave E split; CC0)

| Item | Value |
|---|---|
| File | `Rfam.seed.gz`, 5,928,432 bytes, sha256 `41f014f4ab5628620935f089b1bb77987ebc162fda7c72bff368a32c9cccabac` |
| Source | https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.seed.gz, retrieved 12 Oct 2026 when CURRENT was release 15.1; the file should match the release-15.1 directory by sha256 |
| Licence | Creative Commons Zero (Rfam README and COPYING) |
| Derived | `waveE_dataset.tsv`, sha256 `fc34fe0aba27a57036a79890dc26dc5a3d905a3873c23f8747696c8aa240efb2`, inside the Wave E packet in the execution pack; rebuilt byte-for-byte by `build_waveE_split.py` |

## RMDB chemical-mapping deposits (not shipped: 10.9 GB of public release assets)

All sixty RDAT files are public release assets of https://github.com/DasLab/rmdb.github.io, with
metadata pinned at commit `0a267edb82a30119a0af3b3dfc8fcbf6dbf07357`. Every download was verified
against the published size and sha256 (`admission_run/download_manifest.json`, 60/60).

The unchanged admission parser (`parse_actual_rdat.py`, byte-identical to the 12 Oct copy) turns all
60 files into:
- 3,985,378 profiles;
- 1,708,211 unique candidates.

The parse outputs are shipped in the execution pack under
`work/resource_costing/expansion_admission_20261016/admission_run/`. That directory's `raw/`
subdirectory is not shipped. It holds the 60 RDAT files (10.9 GB), the parser's 6.1 GB SQLite
working index and the fetch log. All three are listed in `EXCLUDED_FILES.json`.

To restore the raw files, run `bash fetch_expansion.sh <dest> download_list.tsv` for the 52 new
files and use the URLs below for the 8 original ones. The script checks size and published sha256.

### The eight original files (12 Oct 2026; source of the first 747,902 profiles)

| RMDB ID | URL | Bytes | sha256 |
|---|---|---:|---|
| OK2TRN_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK2TRN_2A3_0000.rdat | 302,708,515 | `3389dd3e086787be6b9492376250a9050663814e7bd274d01ca617e13d325a56` |
| OK2TRN_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK2TRN_DMS_0000.rdat | 302,546,984 | `26b3f56daede90b3481c341772bdb908b65bbe95f62a2e5b086b8a25834d60e0` |
| M2PK90_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2PK90_2A3_0000.rdat | 219,475,544 | `e139e7918b6c984e41e6ed0bf4273e57630732686834e6eac5e140e79a0e2ef0` |
| M2PK90_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2PK90_DMS_0000.rdat | 219,412,452 | `efc209f453c2e9fe1f15feb7fe1082598723c0b1a8dfa408a938e18a35e1afad` |
| OK1LIB_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK1LIB_2A3_0000.rdat | 197,372,727 | `7ee6f4b3aa0f21eaf2ffc78a7d9ef5dae0feb3cca8a647c8ec3f79e3e46ffc5f` |
| OK1LIB_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK1LIB_DMS_0000.rdat | 197,592,090 | `24f7adb9dd9dfac243ea6581a376bad41cf74c981165ecf332114e52da59b31c` |
| RFMREP_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/RFMREP_2A3_0000.rdat | 188,752,972 | `42100a265b4e10b2a1d7f7c07a7d009e37f6850372e89f963dc0963e281ea954` |
| RFMREP_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/RFMREP_DMS_0000.rdat | 188,698,366 | `d7ecf57d2717f5e2616dddd54125df584a54668ae34529268e57afda40637846` |

### The 52 files acquired on 16 Oct 2026 (9.06 GB)

These are the 30 largest DMS/2A3 counterpart pairs not already held, selected from `rmdb_metadata/deposit_audit/selected_deposit_metadata.json`.

| RMDB ID | URL | Bytes | sha256 |
|---|---|---:|---|
| OK2TST_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK2TST_2A3_0000.rdat | 304,461,599 | `da4d380decf1637814d21c952c6453fa444e34daad19fa1d549834fb51d29c63` |
| OK2TST_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK2TST_DMS_0000.rdat | 304,334,898 | `b36ecc99dd804f419027e266362436e0d3c595b151864153889a80ab5bc0adc2` |
| M2RFPK_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2RFPK_2A3_0000.rdat | 229,159,636 | `f78c6dea3a48fde9ab3c348060b1102eabd612f981dddeb1a16c2f0ca8d4ceb2` |
| M2RFPK_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2RFPK_DMS_0000.rdat | 229,320,598 | `5ec8ea15e5d44933a7f27d1b0212b3f70f2ee49f008991d0db372264b12976d3` |
| M2RFPK_2A3_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2RFPK_2A3_0001.rdat | 229,152,477 | `25821f8f307d682896a81c9be698bd3f02ead8df0d98f12c6d0aa585e406d0d3` |
| M2RFPK_DMS_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2RFPK_DMS_0001.rdat | 229,322,020 | `721b9fcd2bea1109184ea931012c0b28466c17129f5fd022346c92afe3ed0112` |
| OK3TST_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_2A3_0000.rdat | 210,550,022 | `b2c7ea793843c22e8cce218660c6c5d6030fa774683b2aab653873430b36d84c` |
| OK3TST_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_DMS_0000.rdat | 210,658,052 | `e55c94c2059f909a8f80540bd43367619367be570bf1a3100245f1f59f03ec0d` |
| SRSBCV_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_2A3_0000.rdat | 171,784,775 | `b8ceb8cf83f0817f94fe06d3a466ca77d7f8e8404a04cb89d5ce4e7a6523a3a7` |
| SRSBCV_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_DMS_0000.rdat | 171,777,216 | `1e49ce267bb4dc5da74feb925223b22e241022af98402617f98e980b84e720a3` |
| SRSBCV_2A3_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_2A3_0001.rdat | 171,785,488 | `60793fcf7fd9162e39ef9464643847f0f6506b1a498b917d697e501f2fff3d6b` |
| SRSBCV_DMS_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_DMS_0001.rdat | 171,773,113 | `86ff798e52302f7ca65d95d343299e32d1b717a6fb3b58d3dfac7ae194355da1` |
| OK3TST_2A3_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_2A3_0001.rdat | 210,549,110 | `d81247b2d8686326474ccae5dce8ac9a80b5c82587e607d17d96b7730819ca1e` |
| OK3TST_DMS_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_DMS_0001.rdat | 210,651,788 | `da469027e78f8f88f5b67725336e0d5acaa439ab16c2cc0ea31d6971abc40255` |
| OK3TST_2A3_0002 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_2A3_0002.rdat | 210,549,161 | `e5ec41adcc132cef89abdf375b542393345c222a5de7f6064786c9f31b5d6224` |
| OK3TST_DMS_0002 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_DMS_0002.rdat | 210,649,959 | `210028e8b5801f422d7d49b5876da19aa548ecae7e298b00c067b1d414ab857e` |
| OK3TST_2A3_0003 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_2A3_0003.rdat | 210,553,899 | `9b397cfa7d03d0b8053c856de0dab81c0014763f1f233e7b382b2387b4b6f85e` |
| OK3TST_DMS_0003 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_DMS_0003.rdat | 210,655,053 | `2e58309bac210a69e4edd08351bc4f5424a29043521d41ecbfe4926ab8cd6677` |
| SRSBCV_2A3_0003 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_2A3_0003.rdat | 216,579,276 | `a67a1d2b93ece93201e9011916cc313ef53df8126399d61bc16657f025954d44` |
| SRSBCV_DMS_0003 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_DMS_0003.rdat | 216,758,373 | `e0e5ef9a9384151c4de828b478e8fd14e8a668670d2aeb598208270311186764` |
| SRSBCV_2A3_0004 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_2A3_0004.rdat | 216,709,612 | `233d5e0316b945cd62dc6f760b86eb6e1dc1ae8ef874dbb417266639835eb940` |
| SRSBCV_DMS_0004 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_DMS_0004.rdat | 216,894,845 | `86040cf78275eb17fbba0f10556725ce3ff740062101db31cec8eb956a01c344` |
| SRSBCV_2A3_0005 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_2A3_0005.rdat | 216,606,719 | `0341a4bd637c5c61dfab97133882e289f6c9de6320081dfe83eb2d7b02ddd3d3` |
| SRSBCV_DMS_0005 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_DMS_0005.rdat | 216,797,336 | `301e0d8ac2e11a3949847020cf555ec418bd1eb8066073d02b1a46ab316c7351` |
| OK3TST_2A3_0004 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_2A3_0004.rdat | 199,708,304 | `43fac021e1d910b3e67a626505716fb8fd01483236494dcbbffcc58c84ddef8f` |
| OK3TST_DMS_0004 | https://github.com/DasLab/rmdb.github.io/releases/download/data-eterna/OK3TST_DMS_0004.rdat | 199,814,596 | `9e8eaa38e0b2fbb3648bd1a52c777394ebf07501af0e085be6199264463664e6` |
| SRSBCV_2A3_0002 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_2A3_0002.rdat | 155,455,065 | `a1fee405781415aea84ed059f353b72933c6cd18d000bb853ab9dafd8ee9fc10` |
| SRSBCV_DMS_0002 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_DMS_0002.rdat | 155,446,805 | `d17f4a00d507cf81f98cdbc6f15e015b041fa61fd707739db39126f35125495e` |
| 15KREP_2A3_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/15KREP_2A3_0001.rdat | 144,543,440 | `0a38dcc2b0e7feab2cb429be9e5ed649daf4f8035ff5f2a4f91e6a0c3ff149ee` |
| 15KREP_DMS_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/15KREP_DMS_0001.rdat | 144,533,693 | `b30e7d1008c7307558c7986710941b160056a67680f0a3e39e33260fb185e35e` |
| VIRWIN_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/VIRWIN_2A3_0000.rdat | 139,580,287 | `a23b64f05cb709b6ca400752491b0df2f18eaea38f014f850c53f85ba724ea6d` |
| VIRWIN_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/VIRWIN_DMS_0000.rdat | 139,552,987 | `f79c6aa5fa873c03d883fb1c51fa1692c05a7ff9b1526a371bfd0286a652932e` |
| VIRWIN_2A3_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/VIRWIN_2A3_0001.rdat | 172,978,829 | `5f8d9288e3aa82906ec691e8ebb63e39c88f8fd96e73208c224c3262d368e592` |
| VIRWIN_DMS_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/VIRWIN_DMS_0001.rdat | 173,122,387 | `f4ce0a358486f642d826b0de765d96fdb9a1496e426783146260f098eb147cd2` |
| TTRDEL_2A3_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/TTRDEL_2A3_0001.rdat | 133,884,245 | `7ea8a72835883f9b207f7ed3459b732e2dd2fd81cae8f2edfd6fd303da9863f3` |
| TTRDEL_DMS_0001 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/TTRDEL_DMS_0001.rdat | 133,823,831 | `466f8cdbf8bb5473534dcfa14d1c6c6522a80eece3d5364486a1ef51e9efeba8` |
| TTRDES_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/TTRDES_2A3_0000.rdat | 130,179,914 | `1116ceb7f7ec8e69bd40890fb68c42ee4b616eaa5e8a7501bac337c024c4c554` |
| TTRDES_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/TTRDES_DMS_0000.rdat | 130,123,618 | `b9a75cf45b051a21d87e4cabd88f716eb381d893cd753a06ceecc6b4d0dbd562` |
| TTRINS_2A3_0002 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/TTRINS_2A3_0002.rdat | 134,994,702 | `d34f7eccee6d494f73432b719d97560b913b18d23efa9950ba2b3e060b4c3ef6` |
| TTRINS_DMS_0002 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/TTRINS_DMS_0002.rdat | 134,994,611 | `830644b34d2f0d77a6888aebbb635a2cf3cd0c02c49b8d3bc77e32e230fa1522` |
| M2PK50_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2PK50_2A3_0000.rdat | 123,500,789 | `4b08347b74cf72d8d15e4dc1befac4a9065e907342ea268a0cca0cfdf2f767d2` |
| M2PK50_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2PK50_DMS_0000.rdat | 123,466,580 | `410d26b8ba309baba95bdfb6058641d93443301dfb0e7f53847cb0ed0ea5655f` |
| RNAMAK_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/RNAMAK_2A3_0000.rdat | 155,110,616 | `7d620f06fdda3484aae1b9ffc3aad4c0013394e4ae83c8b966cb00f270087a68` |
| RNAMAK_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/RNAMAK_DMS_0000.rdat | 155,212,703 | `1a9c06e10ea4ce481c7db3de8facb275ebec2c5a08717015ffb4f870d458a339` |
| M2RFPK_2A3_0002 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2RFPK_2A3_0002.rdat | 142,050,785 | `ab07b8c4b3bae6c651cadf66370d60dc7132855ff5af3732c9f1bd5989c9bb11` |
| M2RFPK_DMS_0002 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/M2RFPK_DMS_0002.rdat | 142,173,191 | `9014f4b0b08df21764e9dbaf7f125ca1b4f8c14191fedd1ad91587613c5f3f75` |
| SRSBCV_2A3_0006 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_2A3_0006.rdat | 126,973,122 | `d84f28a9a08671fcb7b4e55f7e588a76df43583aeb65a157827cf78f21561337` |
| SRSBCV_DMS_0006 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/SRSBCV_DMS_0006.rdat | 127,082,392 | `3c650db1df66b083b2f55aa6b8f9b4ad9ae5c8249cb0dcb70fe8508c0392b7e9` |
| RFM100_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/RFM100_2A3_0000.rdat | 91,482,159 | `e1e6a29ba1680a9516aff529b30ea675bb485249a0b9caf0e36403262ced0dbf` |
| RFM100_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-rna-structures/RFM100_DMS_0000.rdat | 91,558,264 | `f40d729801582cfe359c43002b2eb642a23b25e9ab1f301073b18db9d021e3b2` |
| ARCHII_2A3_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-general/ARCHII_2A3_0000.rdat | 81,343,842 | `f7cfcf45782444c0bda0e12f80fa96f3cc4fd4c3714cd8dc22027873f71c91c6` |
| ARCHII_DMS_0000 | https://github.com/DasLab/rmdb.github.io/releases/download/data-general/ARCHII_DMS_0000.rdat | 81,396,424 | `e938c8c8cb22372914b6cb08147c035c54f1d1fcda9eb9608088e238e4ff18de` |

## Wave F forward-head pilot sources

**PDB coordinate files and RCSB entry metadata (shipped).** These are under
`biological_validation_pilot_v2/cohort/sources/`. wwPDB archive data are CC0.

**RMDB entries (not shipped: licence for redistribution not stated).** The 14 RMDB RDAT files and
their 14 RMDB entry `.md` files in the same directory come from DasLab/rmdb.github.io:

- The release assets are at `releases/download/<release>/<ID>.rdat`.
- The entries are at `raw.githubusercontent.com/DasLab/rmdb.github.io/0a267edb82a30119a0af3b3dfc8fcbf6dbf07357/_entries/<ID>.md`.

URLs and sha256 are in `cohort/source_download_manifest.json`,
`expanded_source_download_manifest.json` and `liganded_additional_source_manifest.json`. Six of
the entry `.md` files are not in those manifests; for these, `EXCLUDED_FILES.json` gives the
pattern URL and the sha256 of the copy used.

The cohort built from these sources is shipped: `paired_core_cohort.json` and
`paired_core_observations.csv`, with the frozen splits and the results. So is the raw-source
verifier `verify_extraction_from_raw.py`, which re-derives the prepared values from the RDAT and PDB
files; its record `raw_extraction_verification.json` reports 103 checks with 0 failures.

## OpenMM force fields (Wave D refinement)

Wave D refinement uses the OpenMM 8.4 built-in files `amber14/RNA.OL3.xml` (sha256
`61b793a49ded3c9c3bd9fbab328bc1b20b2b4ce7bc2358d537892ca64b085127`) and `implicit/gbn2.xml`
(sha256 `d12c14820a413146eb008758c405b8690898811b4e08827b00a75a163395bde3`). They are recorded
in the Wave D manifest.

## Generated outputs not shipped

- **Checkpoint states:** model, AdamW moments and RNG states, about 56 GB per 32-node restart test.
- **Wave D structures:** generated and refined PDB and NPY files.
- **Wave E model:** the final continued-pretraining model, 384 MB.

All of these remain on JUPITER. Their sha256 values are recorded in the shipped rank reports and
case results.
