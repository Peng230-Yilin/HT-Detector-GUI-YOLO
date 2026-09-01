# Third-party notices

HT-Detector-GUI-YOLO 1.0.0 combines original project changes with third-party components. Except where a component carries a separate notice, the project's original code and modifications are released under the GNU Affero General Public License version 3 (`AGPL-3.0`); see [LICENSE](LICENSE).

This notice does not replace any component's license text or copyright notice. Historical versions already made available under the MIT License remain available under the terms granted for those versions.

## Project original changes

The GUI integration, result workflows, configuration handling, release tooling, and modifications to vendored components are distributed under AGPL-3.0 except where a file states otherwise. The rights holder for the project's original code and modifications is Peng Yilin.

## hmy-repo MIT components

Code imported from the historical `hmy-repo` components retains its MIT license. The original license texts are preserved without replacement at:

- `Peng1.0_GUI/LICENSE`
- `HT-Detector_Peng/LICENSE`

The root AGPL-3.0 license does not revoke or retroactively alter rights already granted under those MIT license texts.

## Ultralytics 8.1.47

`HT-Detector_Peng/ultralytics/` contains vendored and modified Ultralytics 8.1.47 source code. Its source headers and package metadata identify AGPL-3.0. The complete AGPL-3.0 text is the repository root [LICENSE](LICENSE). Upstream attribution is also retained in `HT-Detector_Peng/CITATION.cff` and the vendored source headers.

Upstream project: <https://github.com/ultralytics/ultralytics>

## YOLOv8n base model and best.pt

The fixed `best.pt` checkpoint is based on Ultralytics YOLOv8n and is distributed as AGPL-3.0 according to [MODEL_CARD.md](MODEL_CARD.md). Its joint trainers and rights holders are Peng Yilin and Yue Hengmao. The checkpoint was trained on project-collected and project-annotated cuvette images. Its permitted purpose, limitations, exact byte size, and SHA-256 are recorded in the model card. The Ultralytics base model and all other third-party code and resources retain their respective licenses, copyright notices, and rights statements.

## Qt and PySide6 example-derived code

`Peng1.0_GUI/camera.py` retains the Qt Company copyright and its file-level SPDX notice, `LicenseRef-Qt-Commercial OR BSD-3-Clause`. PySide6 itself is installed as an external dependency and is not vendored into the source release. Recipients must review the license files supplied with the installed PySide6 distribution and Qt's licensing terms: <https://www.qt.io/licensing/>.

## Icons and resources

The public-domain notice for the identified third-party icons is preserved at `Peng1.0_GUI/resource/3rdparty/COPYING`. The associated Qt attribution metadata is preserved at `Peng1.0_GUI/resource/3rdparty/qt_attribution.json`. Other resource files retain any embedded or adjacent notices supplied with them.

## pip-installed runtime dependencies

PyTorch, TorchVision, PySide6, OpenCV, NumPy, pandas, openpyxl, Matplotlib, SciPy, Pillow, PyYAML, Requests, tqdm, psutil, py-cpuinfo, thop, and seaborn are installed separately from their respective package indexes. They are not copied into the source-form runtime package produced by `release/build_release.ps1`.

Each dependency remains under its own license. License texts and notices are supplied by the installed wheel/source distribution and its upstream project. If a future release bundles Python or any dependency binaries, the release process must collect and ship those exact distribution license files; this source release notice alone is not a substitute.

See [LICENSES/README.md](LICENSES/README.md) for the license-file map included in the runtime package.
