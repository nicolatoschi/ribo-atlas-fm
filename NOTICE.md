# Third-party code, attribution and modifications

This repository is distributed under the GNU General Public License v3.0 (see `LICENSE`, a verbatim copy of the licence file in RNAbpFlow's GitHub repository).

## RNAbpFlow: inference code (`src/evaluation/RNAbpFlow/`)

- **Upstream:** https://github.com/Bhattacharya-Lab/RNAbpFlow at commit `e8b1c077f7951d4d6ebe00c851d22c37777dfc9f`, licensed GPL-3.0. The release zip has sha256 `193a01c7f3e54be7710af83ab4e2dbf9fe4f95ff82a2fea4f566da9bb650ff08`.
- **Unchanged:** every file in `src/evaluation/RNAbpFlow/` is byte-identical to that commit, except for the following.

| File | Status | Upstream sha256 | Shipped sha256 | Change |
|---|---|---|---|---|
| `src/data/utils.py` | modified | `b84d7ae02e2e3f899cced42ccee7407a2324f90608eefe3ac6afa3d6dd820a8a` | `543aa6938176aa15378a97c606c6e917e55dcc948d3b86bcd114568f57aba6a6` | One import line: `torch_scatter` is replaced by the Torch-native adapter below. The diff is `src/evaluation/provenance/inference_packet/portability.diff` |
| `src/data/torch_native_scatter.py` | added (this project) | — | see `PROVENANCE.json` | Grouped mean/sum via `index_add` |

## RNAbpFlow: training code (`src/training/RNAbpFlow_training/`)

- **Upstream:** the `RNAbpFlow_training/` tree of `RNAbpFlow-Train.tar.gz`, from the Zenodo record "RNAbpFlow training data and code" (https://zenodo.org/records/19388910, doi:10.5281/zenodo.19388910).
  - The record metadata gives licence CC-BY-4.0.
  - The archive has sha256 `89280784c053410aeba3255701ea46c0a33b70d89a1c92966d7271212a1b5b44` and MD5 `4b7015fe0fbdd0ba88c752cdb19c0758`.
- **Licence file:** the archive itself contains no licence file. `RNAbpFlow_training/LICENSE` is the GPL-3.0 file of the RNAbpFlow GitHub repository, added so that the licence travels with the code.
- **Origin of the training code:** the archive's embedded git metadata names https://github.com/rish-16/rna-backbone-design (commit `d7a2532fb082b1f4500e1f4272f7e0b83427f858`) as the origin; see that repository for its licence and upstream notices.
- **Unchanged:** every file under `src/` and `configs/`, and `RNAbpFlow.yml`, is byte-identical to the archive member, except for the following.

| File | Status | Archive sha256 | Shipped sha256 | Change |
|---|---|---|---|---|
| `src/data/utils.py` | modified | `3eec1fc72b1f6f33ebf8bca2fb220ee69b6cd143697a6affdecadd469c41806d` | `7880ac8141432a016725998329e585490c810295535bb86a470d67f94d8256a3` | Only the grouped mean/sum `torch_scatter` import is replaced by `torch_native_scatter`; the model is untouched |
| `src/models/flow_module.py` | modified | `068d660d155c136ef69262051fd7f740b76e77511d6283d520868e296905bcd9` | `47b8d389ab8d19842d01dae8651e80ad16443476e3a6057dfbb08e80b60520ff` | The NaN-to-zero fallback becomes a `RuntimeError` for any non-finite auxiliary or SE(3) loss; the finite-loss path is unchanged. Note that this file still calls `torch.autograd.set_detect_anomaly(True)` at import. The drivers switch it off after model construction (see `docs/anomaly_fix.diff`) |
| `src/data/torch_native_scatter.py` | added (this project) | — | see `PROVENANCE.json` | Torch-native grouped reduction |
| `LICENSE` | added | — | `3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986` | Copied from RNAbpFlow GitHub `e8b1c07` |

These hashes are also recorded in the frozen-packet manifests (`src/training/packet_manifests/`) and in `training_source_manifest.json` inside each training packet.

## RNAbpFlow training feature files (execution pack only), CC-BY-4.0

The training packets in the execution pack contain 20 distinct RNAbpFlow feature pickles, 64 copies in all. The packet paths are `RNAbpFlow_training/Feature/train_pkl/<id>/<id>.pkl` and `RNAbpFlow_training/pkls/<id>_base/<id>_feat.pkl`. Each one is byte-identical to the member of the same path in the archive described below; this was checked when the pack was built.

| Field | Value |
|---|---|
| Source | Zenodo record 19388910, "RNAbpFlow training data and code", by Sumit Tarafder and Debswapna Bhattacharya, published 2026-04-04 |
| Location | https://zenodo.org/records/19388910, doi:10.5281/zenodo.19388910 |
| Archive | `RNAbpFlow-Train.tar.gz`, 681,589,956 bytes; sha256 `89280784c053410aeba3255701ea46c0a33b70d89a1c92966d7271212a1b5b44` |
| Licence | Creative Commons Attribution 4.0 International (CC-BY-4.0), as stated in the record metadata: https://creativecommons.org/licenses/by/4.0/ |
| Changes | None. The files are redistributed unmodified |
| Please cite | Tarafder S and Bhattacharya D. RNAbpFlow: base pair-augmented SE(3) flow matching for conditional RNA 3D structure generation. *Nature Methods* 23, 1349–1358 (2026). doi:10.1038/s41592-026-03128-4. Also cite the data record: Tarafder S and Bhattacharya D. RNAbpFlow training data and code. Zenodo (2026). doi:10.5281/zenodo.19388910 |

**Derived data.** The Wave D case inputs (`cases/<id>/case_input.npz`) and the teacher fixtures are derived from these files and are shipped under the same attribution. The npz files hold base-pair channels and native coordinates. The fixtures hold seeded corruptions of the native coordinates.

**Upstream structures.** The structures behind the features come from the Protein Data Bank (wwPDB, CC0).

## RNA-FM (`src/sequence_prior/`)

- **Not redistributed:** neither the RNA-FM code nor its weights are included here. Wave E installs PyPI `rna-fm` 0.2.2 (MIT, wheel sha256 `4fff72892ad02559be969b7b3263483e7f3730132025ea05c9927c3675a5e0b4`). Its `fm/` package is byte-identical to https://github.com/ml4bio/RNA-FM at `348951516e0963d22bbb33b3c9fc18c89081d38e`.
- **Weights:** the weights come from the Hugging Face repository `cuhkaih/rnafm` (see `docs/DATA.md`); its model card declares Apache-2.0.

## Other upstream material

- **Force fields:** OpenMM force-field files are referenced by name and hash only.
- **External data:** all other external data (Rfam, RMDB, PDB, the RNAbpFlow checkpoint, RNA-FM weights) is referenced by accession, URL and sha256 in `docs/DATA.md`. Rfam (CC0) and PDB (CC0) files used by the pilots are shipped in the execution pack. RMDB RDAT files are not shipped; they are public release assets, listed with their hashes.

## Project code

Everything not listed above was written for RIBO-ATLAS-FM and is released under GPL-3.0. `PROVENANCE.json` gives the origin path and sha256 of every copied file.
