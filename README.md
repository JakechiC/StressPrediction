# StressPrediction

Equivariant neural-network source for per-atom energy-valued virial tensors.

This release contains model implementations, data loading, model training and raw prediction, SOAP/FPS input preprocessing, and Dueling DQN training and candidate generation for graphene. The E+W/readout and E+F(+W) implementations are separate installable subprojects:

- `ew_readout/StressPrediction`: PaiNN-Direct, PaiNN-Edge, PaiNN-Derivative, TensorNet-Direct, TensorNet-Derivative, TensorPaiNN and PaiNNHeadOnly.
- `force_controls/StressPrediction`: the separate energy/force/virial training implementation.

Use separate virtual environments because both subprojects use the same Python package and command name. Each contains a short README and example configurations. The root is not an installable package.

The release scope excludes plotting, offline metric analysis, result aggregation, AIREBO candidate verification, and historical result archives. Training retains the weighted validation loss needed for checkpoint selection. Graphene hotspot scoring remains part of the policy's reward and candidate generation. No `docs` directory is included.

Raw datasets, split arrays and trained weights are external inputs. Configuration examples are not certified historical run records. This source release does not reproduce or certify historical manuscript figures or resolve their outstanding source/metric/unit correspondence.

Author-controlled contributions are licensed under MIT, copyright (c) 2026 JIAN CHANG. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for retained upstream notices.
