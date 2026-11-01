# Third-party notices

These notices apply to the identified source portions in ew_readout/StressPrediction/ and force_controls/StressPrediction/, including all copies distributed in this release. The notices below must accompany redistribution. The project LICENSE grants rights only to contributions the project authors are entitled to license. Dependencies and external data retain their own terms.

## TorchMD-Net-derived portions

The files named `tensornet.py` and `tensornet_utils.py` retain source headers identifying Universitat Pompeu Fabra, 2020–2023, and the MIT License. The `GatedEquivariantBlock` in files named `painn_utils.py` is also explicitly marked as borrowed from TorchMD-Net. The exact historical import commit was not recorded; the identified source must not be presented as a verified import of the current upstream branch.

Upstream: [TorchMD-Net](https://github.com/torchmd/torchmd-net), [license](https://github.com/torchmd/torchmd-net/blob/main/LICENSE).

```text
MIT License

Copyright (c) 2021-2023 Universitat Pompeu Fabra, https://www.compscience.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OCP / FAIR-Chem PaiNN and GemNet implementation portions

A static source audit identified complete class AST matches between the local `painn_utils.py` and official FAIR-Chem source at comparison revision `977a80328f2be44649b414a9907a1d6ef2f81e95`. This establishes substantial implementation correspondence, not the date or exact revision originally imported by this project. These portions must retain the upstream notices even where the archived local files omitted their headers.

| Local portion | Official comparison source |
|---|---|
| `PaiNNMessage`, `PaiNNUpdate`, `GatedEquivariantBlock`; closely corresponding `PaiNNOutput` | [painn/painn.py](https://github.com/facebookresearch/fairchem/blob/977a80328f2be44649b414a9907a1d6ef2f81e95/src/fairchem/core/models/painn/painn.py) |
| `PolynomialEnvelope`, `ExponentialEnvelope`, `SphericalBesselBasis`, `BernsteinBasis`, `RadialBasis` | [gemnet/layers/radial_basis.py](https://github.com/facebookresearch/fairchem/blob/977a80328f2be44649b414a9907a1d6ef2f81e95/src/fairchem/core/models/gemnet/layers/radial_basis.py) |
| `AtomEmbedding` | [gemnet/layers/embedding_block.py](https://github.com/facebookresearch/fairchem/blob/977a80328f2be44649b414a9907a1d6ef2f81e95/src/fairchem/core/models/gemnet/layers/embedding_block.py) |
| `ScaledSiLU` | [gemnet/layers/base_layers.py](https://github.com/facebookresearch/fairchem/blob/977a80328f2be44649b414a9907a1d6ef2f81e95/src/fairchem/core/models/gemnet/layers/base_layers.py) |

The local PaiNN model wrappers use and adapt these building blocks. The notices apply to corresponding derived portions of those wrappers and archival copies as well. The [FAIR-Chem license at this revision](https://github.com/facebookresearch/fairchem/blob/977a80328f2be44649b414a9907a1d6ef2f81e95/LICENSE.md) is reproduced here:

```text
MIT License

Copyright (c) Meta, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The upstream PaiNN file also carries the following additional notice, retained here in full:

```text
MIT License

Copyright (c) 2021 www.compscience.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PyTorch Geometric GaussianSmearing

The `GaussianSmearing` class in `painn_utils.py` has a complete AST match to the official [PyTorch Geometric 2.6.1 SchNet source](https://github.com/pyg-team/pytorch_geometric/blob/2.6.1/torch_geometric/nn/models/schnet.py). This is an identified vendored implementation portion, in addition to the runtime dependency. The exact historical import revision is unknown. The [upstream 2.6.1 license](https://github.com/pyg-team/pytorch_geometric/blob/2.6.1/LICENSE) is reproduced below:

```text
Copyright (c) 2023 PyG Team <team@pyg.org>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Other material and scope

Scientific citations identify methods; they do not by themselves assign a software license. The original author has confirmed the intended release of author-controlled code under MIT. This does not replace the identified upstream notices above, or establish unrecorded historical import revisions.

The optional SOAP/FPS workflow credits the project's `soap_select_fps.py` by Ting Liang. DScribe and scikit-matter are runtime dependencies rather than vendored selection-library source here. Credit must be preserved.

Large datasets, trained weights, figures and LAMMPS potential files are not bundled in this repository and are not licensed through these notices. Their availability, exact provenance and redistribution terms must be established separately. Archived numerical summaries and protocol/source metadata are distinguished from the external datasets and model weights in the repository README.
