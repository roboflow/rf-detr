# RF-DETR: Real-Time SOTA Object Detection, Instance Segmentation, and Keypoint Detection

# RF-DETR: リアルタイム SOTA 物体検出・インスタンスセグメンテーション・キーポイント検出

[![version](https://badge.fury.io/py/rfdetr.svg)](https://badge.fury.io/py/rfdetr)
[![downloads](https://img.shields.io/pypi/dm/rfdetr)](https://pypistats.org/packages/rfdetr)
[![codecov](https://codecov.io/gh/roboflow/rf-detr/graph/badge.svg?token=K8V4ARR3XV)](https://codecov.io/gh/roboflow/rf-detr)
[![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/roboflow/rfdetr/blob/main/LICENSE)

[![arXiv](https://img.shields.io/badge/arXiv-2511.09554-b31b1b.svg)](https://arxiv.org/abs/2511.09554)
[![hf space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SkalskiP/RF-DETR)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb)
[![roboflow](https://raw.githubusercontent.com/roboflow-ai/notebooks/main/assets/badges/roboflow-blogpost.svg)](https://blog.roboflow.com/rf-detr)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

RF-DETR is a real-time transformer architecture for object detection, instance segmentation, and keypoint detection (preview) developed by Roboflow. Built on a DINOv2 vision transformer backbone, RF-DETR delivers state-of-the-art accuracy and latency trade-offs on [Microsoft COCO](https://cocodataset.org/#home) and [RF100-VL](https://github.com/roboflow/rf100-vl).

> **日本語:** RF-DETR は Roboflow が開発した、物体検出・インスタンスセグメンテーション・キーポイント検出（プレビュー）向けのリアルタイム Transformer アーキテクチャです。DINOv2 ビジョン Transformer バックボーン上に構築され、[Microsoft COCO](https://cocodataset.org/#home) および [RF100-VL](https://github.com/roboflow/rf100-vl) において SOTA 級の精度とレイテンシのトレードオフを実現します。専門用語は [用語解説](#用語解説--glossary) を参照してください。

RF-DETR uses a DINOv2 vision transformer backbone and supports object detection, instance segmentation, and keypoint detection (preview) in a single, consistent API. The open-source `rfdetr` package and Apache-designated models are released under Apache 2.0, while Plus components (`rfdetr_plus`, including RF-DETR-XL/2XL detection models) are licensed under PML 1.0.

> **日本語:** RF-DETR は DINOv2 バックボーンを用い、単一の一貫した API で物体検出・インスタンスセグメンテーション・キーポイント検出（プレビュー）をサポートします。オープンソースの `rfdetr` パッケージおよび Apache 指定モデルは Apache 2.0、Plus コンポーネント（`rfdetr_plus`、RF-DETR-XL/2XL 検出モデルを含む）は PML 1.0 の下で提供されます。

https://github.com/user-attachments/assets/add23fd1-266f-4538-8809-d7dd5767e8e6

## 用語解説 / Glossary

README 内で使われる専門用語の解説です（日本語）。初めて RF-DETR やコンピュータビジョンに触れる方の参照用です。

<details>
<summary>用語解説を開く / Open glossary</summary>

<br>

### タスク・機能

| 用語 | 解説 |
| ---- | ---- |
| **物体検出** (Object Detection) | 画像内の対象物の位置（通常は矩形のバウンディングボックス）とクラス（種類）を同時に推定するタスク。例: 「犬がここにある」と座標付きで出力する。 |
| **インスタンスセグメンテーション** (Instance Segmentation) | 物体検出に加え、各物体の**輪郭（ピクセル単位のマスク）**まで推定するタスク。同じクラスが複数あっても、個体ごとに領域を分離する。 |
| **キーポイント検出** (Keypoint Detection) | 人体などの**関節・特徴点**（肩・肘・膝など）の座標を推定するタスク。姿勢推定（ポーズ推定）の基礎になる。 |
| **プレビュー** (Preview) | 機能は提供されているが、API や精度・安定性が今後変わる可能性がある開発段階の機能。本番利用前にリリースノートの確認が推奨される。 |
| **ファインチューニング** (Fine-tuning) | 事前学習済みモデルを、自社・自用途のデータセットで追加学習し、特定ドメインの精度を高めること。 |
| **事前学習** (Pretraining) | 大規模データ（例: COCO）で最初に学習した重み。ファインチューニングの出発点になる。 |

### モデル・アーキテクチャ

| 用語 | 解説 |
| ---- | ---- |
| **RF-DETR** | Roboflow が開発した **Detection Transformer** 系のリアルタイムモデルファミリー。YOLO 系と比べ、Transformer ベースで高精度と低レイテンシの両立を目指す。 |
| **DETR** (DEtection TRansformer) | Transformer を物体検出に応用した方式。従来の CNN + アンカー方式と異なり、画像全体を一度に処理して物体を集合として予測する。 |
| **Transformer** | Attention 機構で系列（または画像パッチ）間の関係を学習するニューラルネットワーク。RF-DETR の中核。 |
| **Vision Transformer (ViT)** | 画像をパッチに分割し Transformer で処理する方式。 |
| **DINOv2** | Meta が公開した自己教師あり学習の ViT バックボーン。汎用的な視覚特徴の抽出に強く、RF-DETR の**特徴抽出の土台**として使われる。 |
| **バックボーン** (Backbone) | モデル内で画像から特徴マップを抽出する部分。DETR 系では DINOv2 がこの役割。後段の検出ヘッドがこの特徴を使って物体を予測する。 |
| **アーキテクチャ** (Architecture) | モデルの層構成・全体設計。表の RF-DETR-N / S / M などはサイズ違いのアーキテクチャバリアント。 |
| **Deformable DETR** | 計算効率を改善した DETR の派生。RF-DETR の技術的基盤の一つ。 |
| **LW-DETR** | 軽量・リアルタイム DETR の一系統。RF-DETR のベース研究の一つ。 |

### 評価指標（ベンチマーク表の読み方）

| 用語 | 解説 |
| ---- | ---- |
| **AP** (Average Precision) | 検出・セグメンテーションの精度指標。**高いほど正確**。COCO ではクラスごとの AP を平均した **mAP**（mean AP）として報告されることが多い。 |
| **IoU** (Intersection over Union) | 予測領域と正解領域の重なり度（0〜1）。AP の計算で「どれだけ位置が合っているか」を判定する基準になる。 |
| **AP<sub>50</sub>** | IoU 閾値 0.5（予測と正解の重なりが 50% 以上）での AP。**やや緩い基準**で、大まかな検出性能の比較に使われる。 |
| **AP<sub>50:95</sub>** | IoU 0.5〜0.95 を 0.05 刻みで変えた AP の平均。**COCO の標準指標**で、位置合いの精度も厳しく評価する。数値は AP<sub>50</sub> より低く出るのが普通。 |
| **OKS** (Object Keypoint Similarity) | キーポイント同士の一致度。人物のスケールや部位ごとの許容誤差を考慮する。キーポイントの AP は OKS ベースで計算される。 |
| **レイテンシ** (Latency) | 1 枚の画像を推論するのにかかる時間（表では ms = ミリ秒）。**小さいほど速い**。リアルタイム用途では重要。 |
| **SOTA** (State Of The Art) | 当時点で公開されている手法の中で**最高水準**に近い性能を指す表現。 |
| **Params (M)** | 学習可能パラメータ数（百万単位）。**大きいほどモデル容量が大きく**、一般に精度は上がりやすいが計算コストも増える。 |
| **Resolution** | 推論時にモデルへ入力する画像サイズ（幅×高さ）。RF-DETR はサイズごとに最適解像度が異なる。 |
| **トレードオフ** (Trade-off) | 精度と速度（またはモデルサイズ）の**両立関係**。一方を上げると他方が犠牲になりやすい。 |

### データセット・ベンチマーク環境

| 用語 | 解説 |
| ---- | ---- |
| **Microsoft COCO** | 物体検出・セグメンテーション・キーポイントの標準ベンチマークデータセット。80 クラス、多数の検証画像で性能比較に広く使われる。 |
| **RF100-VL** | Roboflow 系の 100 データセットを集めた**実務寄り**のベンチマーク。ドメイン多様性の下での検出性能評価に使われる（README では検出のみ）。 |
| **TensorRT** | NVIDIA の推論最適化エンジン。学習済みモデルを GPU 上で高速実行するために使われる。表のレイテンシは TensorRT 経由の計測。 |
| **FP16** | 16 ビット浮動小数点。FP32 より速く・省メモリだが、ごくわずかに精度が落ちる場合がある。GPU 推論の定番設定。 |
| **バッチサイズ** (Batch size) | 一度に処理する画像枚数。表では **1**（1 枚ずつ）で計測しており、ストリーミング推論に近い条件。 |

### モデルサイズ表記

| 表記 | 意味 |
| ---- | ---- |
| **N** (Nano) | 最小・最速。エッジ端末向け。 |
| **S** (Small) | 小型。速度優先。 |
| **M** (Medium) | 中サイズ。精度と速度のバランス。 |
| **L** (Large) | 大型。精度寄り。 |
| **XL / 2XL** | 超大規模。最高精度だが Plus ライセンスが必要な検出モデルあり（△ マーク）。 |
| **Seg** | Segmentation（セグメンテーション）用バリアント。 |
| **threshold** | 推論結果の信頼度しきい値。`threshold=0.5` なら 50% 未満の検出は捨てる。高くすると誤検出は減るが取りこぼしが増える。 |

### ライブラリ・API

| 用語 | 解説 |
| ---- | ---- |
| **`rfdetr` パッケージ** | RF-DETR を Python から使うための公式ライブラリ。`RFDETRMedium()` のようにクラス名でモデルを選ぶ。 |
| **API** | プログラムから機能を呼び出すためのインターフェース。ここでは `predict()` などのメソッド群を指す。 |
| **supervision** | 検出結果の可視化・後処理用ライブラリ（`Detections`、`BoxAnnotator` など）。Roboflow 系エコシステムでよく使われる。 |
| **Inference ライブラリ** | Roboflow の統合推論 SDK。`get_model("rfdetr-medium")` のように**エイリアス名**でモデルを取得できる。 |
| **エイリアス** (package alias) | 人間が読みやすいモデル ID（例: `rfdetr-medium`）。内部では対応する RF-DETR クラス・重みに解決される。 |

### 対応クラス・モデル

| 用語 | 解説 |
| ---- | ---- |
| **COCO 80 クラス** | 検出・セグ事前学習モデルが識別できる 80 種類（人・車・犬・りんご等）。**potato は含まれない**。一覧は [models-and-coco-classes.md](docs/ja/models-and-coco-classes.md) |
| **別チェックポイント** | 検出（`RFDETRNano` 等）・セグ（`RFDETRSegMedium` 等）・キーポイント（`RFDETRKeypointPreview`）は **重みファイルが別** |
| **ファインチューニング** | COCO 外クラスを使うには自社データで再学習が必要 |

### ライセンス

| 用語 | 解説 |
| ---- | ---- |
| **Apache 2.0** | 商用利用も可能なオープンソースライセンス。`rfdetr` 本体と N/S/M/L などの Apache 指定モデルに適用。 |
| **PML 1.0** | Roboflow の Plus モデルライセンス。`rfdetr_plus` 拡張と XL / 2XL 検出モデルに適用。**利用条件が Apache 2.0 と異なる**ため導入前に LICENSE を確認すること。 |
| **`rfdetr_plus`** | XL / 2XL など Plus モデルを使うための追加パッケージ。`pip install rfdetr[plus]` で導入。 |
| **AGPL-3.0** | 比較対象の YOLO 系などに付くコピーレフト系ライセンス。ネットワーク経由提供時のソース公開義務など、Apache 2.0 より制約が強い場合がある。 |

</details>

## Supported Classes & Models / 対応クラスとモデル

RF-DETR pretrained models detect **Microsoft COCO 80 classes** (detection and segmentation) or **17 person keypoints** (keypoint preview). They are **separate model checkpoints**, not one model with modes.

> **日本語:** 事前学習モデルがそのまま識別できるのは **COCO 80 クラス**（検出・セグ）または **人体 17 関節**（キーポイント Preview）です。検出・セグ・キーポイントは **別の重みファイル** です。**ジャガイモ（potato）など COCO 外の物体は含まれません** — 自社用途にはファインチューニングが必要です。
>
> **詳細リファレンス（クラス一覧・モデル対照表・ER-FlowScan 使い分け）:** [docs/ja/models-and-coco-classes.md](docs/ja/models-and-coco-classes.md)

| Task / タスク | Python class (example) | Detects / 検出対象 | Output / 出力 |
|---------------|------------------------|-------------------|---------------|
| Detection / 検出 | `RFDETRNano` … `RFDETRLarge` | COCO 80 classes / COCO 80 クラス | Bounding boxes / 矩形 |
| Segmentation / セグ | `RFDETRSegNano` … `RFDETRSeg2XLarge` | Same 80 classes / 同じ 80 クラス | Boxes + masks / 矩形 + マスク |
| Keypoints / キーポイント | `RFDETRKeypointPreview` | **Person only** / **人のみ** (17 joints) | Skeleton / 骨格 |

<details>
<summary>COCO 80 class names (English) / COCO 80 クラス名（英語）</summary>

<br>

`person`, `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`, `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`, `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`, `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`, `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`, `banana`, `apple`, `sandwich`, `orange`, `broccoli`, `carrot`, `hot dog`, `pizza`, `donut`, `cake`, `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`, `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`, `microwave`, `oven`, `toaster`, `sink`, `refrigerator`, `book`, `clock`, `vase`, `scissors`, `teddy bear`, `hair drier`, `toothbrush`

> Full table with COCO IDs and Japanese labels: [models-and-coco-classes.md](docs/ja/models-and-coco-classes.md)

</details>

## Install / インストール

To install RF-DETR, install the `rfdetr` package in a [**Python>=3.10**](https://www.python.org/) environment with `pip`.

> **日本語:** RF-DETR をインストールするには、[**Python>=3.10**](https://www.python.org/) 環境で `pip` を使い `rfdetr` パッケージをインストールします。

```bash
pip install rfdetr
```

<details>
<summary>Install from source / ソースからインストール</summary>

<br>

By installing RF-DETR from source, you can explore the most recent features and enhancements that have not yet been officially released. **Please note that these updates are still in development and may not be as stable as the latest published release.**

> **日本語:** ソースからインストールすると、まだ正式リリースされていない最新機能を試せます。**これらの更新は開発中であり、最新の安定版ほど安定しない場合がある点にご注意ください。**

```bash
pip install https://github.com/roboflow/rf-detr/archive/refs/heads/develop.zip
```

</details>

## Local Video Demo / ローカル動画デモ

Run object detection or keypoint inference on a video and export an annotated MP4.

**Recommended sample (local only):** place a dance / person video at `sample/mzoo.mov` on your machine — **do not commit sample videos** (gitignored).  
Fallback: FlashFind `potato_conveyor.mov` (conveyor demo only; **not** a good fit for COCO pretrained models).

> **日本語:** 動画に RF-DETR の検出框または骨格を描画し MP4 を出力します。**人物デモ用動画は `sample/mzoo.mov` を各自の PC に置いて使います（GitHub 等には載せません — `.gitignore` 対象）**。ジャガイモ動画は COCO 事前学習には不向きです。詳細は [sample/README.md](sample/README.md) と [対応クラス解説](docs/ja/models-and-coco-classes.md)。
>
> **重要（Windows）:** **コマンドプロンプト（cmd）やエクスプローラーから `.ps1` を直接実行しないでください**（「アプリを選択」ダイアログが出ます）。代わりに下記の **`.cmd`** を使うか、PowerShell ターミナルで `.ps1` を実行してください。

**Command Prompt (cmd) — recommended / コマンドプロンプト（推奨）:**

```bat
cd rf-detr
rem Person detection on dance sample (default source: sample/mzoo.mov)
scripts\run_demo_video.cmd --task detect --person-only --frame-stride 2

rem Person skeleton (keypoint preview; slow on CPU — use --max-frames first)
scripts\run_demo_video.cmd --task keypoint --frame-stride 2 --max-frames 60
```

**PowerShell:**

```powershell
cd rf-detr
.\scripts\run_demo_video.ps1 -MaxFrames 30
```

**Do not** double-click `run_demo_video.ps1` in Explorer / エクスプローラーで `.ps1` をダブルクリックしないこと。

Manual invocation / 手動実行:

```powershell
$env:UV_TORCH_BACKEND = "cpu"
.\.venv\Scripts\python.exe scripts\run_video_demo.py `
  --model nano `
  --output artifacts\demo\potato_conveyor_detected.mp4
```

| Option | Description / 説明 |
| ------ | -------------------- |
| `--source` | Input video / 入力動画（既定: `sample/mzoo.mov` → FlashFind 動画） |
| `--output` | Output MP4 / 出力 MP4（既定: `artifacts/demo/<stem>_<task>.mp4`） |
| `--task` | `detect` or `keypoint` / 検出 or 骨格 |
| `--model` | `nano` … `large`（検出のみ; CPU では `nano` 推奨） |
| `--person-only` | Detection: COCO `person` only / 人物のみ（mzoo では自動 ON） |
| `--threshold` | Confidence threshold / 信頼度しきい値（既定: 0.5） |
| `--frame-stride` | Infer every N frames / N フレームごとに推論（既定: 2） |
| `--max-frames` | Limit inferred frames / 試行用フレーム上限 |

If the demo video is missing, sync it from the FlashFind scripts:

> **日本語:** 動画がない場合は FlashFind 側で同期してください。

```powershell
cd ..\FlashFind\scripts
.\prepare_demo_video.ps1
```

## Video Demo GUI / 動画デモ GUI

Tkinter GUI で動画デモを実行できます。**ローカル CPU/GPU** または **Vast.ai 外部 GPU** を選べます。進捗バー・7 ステップ起動表示・事前チェック（Preflight）付き。

> **日本語:** CPU のみの PC でも GUI から Vast.ai 上の GPU でキーポイント推論を実行できます。API キーは FlashFind の `.env`（`FLASHFIND_VAST_API_KEY`）を自動共有できます。他プロジェクトへの組み込み手順は [Vast.ai 組み込みガイド](docs/ja/vast-ai-integration-guide.md) を参照してください。

**起動（cmd 推奨）:**

```bat
cd rf-detr
scripts\run_demo_gui.cmd
```

**Vast.ai 利用の初回セットアップ:**

```bat
uv pip install vastai
vastai set api-key YOUR_API_KEY
vastai create ssh-key
```

| 機能 | 説明 |
| ---- | ---- |
| 実行先 | ローカル / 外部 GPU (Vast.ai) |
| タスク | 検出 / キーポイント / 不確実性ヒートマップ |
| 解析プレビュー | パラパラ漫画風（直近 6 枚サムネ＋メイン画面、約 8 FPS 更新上限） |
| GPU 検索 | オファー一覧から選択 |
| 安全装置 | 最大稼働時間、destroy リトライ、orphan 回収 |
| 手動 orphan 回収 | `scripts\vast_cleanup_orphans.cmd` |

関連ドキュメント:

- [docs/ja/vast-ai-integration-guide.md](docs/ja/vast-ai-integration-guide.md) — ER-FlowScan 共通の Vast.ai 組み込み設計書
- [docs/ja/models-and-coco-classes.md](docs/ja/models-and-coco-classes.md) — モデル・COCO クラス解説
- [sample/README.md](sample/README.md) — ローカル動画の置き方

## Benchmarks / ベンチマーク

RF-DETR achieves state-of-the-art results in both object detection and instance segmentation, with benchmarks reported on Microsoft COCO and RF100-VL (RF100-VL for detection only). The charts and tables below compare RF-DETR against other top real-time models across accuracy and latency for detection and segmentation. All latency numbers were measured on an NVIDIA T4 using TensorRT, FP16, and batch size 1. For full benchmarking methodology and reproducibility details, see [roboflow/sab](https://github.com/roboflow/single_artifact_benchmarking).

> **日本語:** RF-DETR は物体検出とインスタンスセグメンテーションの両方で SOTA 級の結果を達成しています（ベンチマークは Microsoft COCO および RF100-VL で報告。RF100-VL は検出のみ）。以下のグラフと表は、検出・セグメンテーションにおける精度とレイテンシの面で、他の主要リアルタイムモデルと RF-DETR を比較したものです。レイテンシはすべて NVIDIA T4、TensorRT、FP16、バッチサイズ 1 で計測しています。ベンチマーク手法と再現性の詳細は [roboflow/sab](https://github.com/roboflow/single_artifact_benchmarking) を参照してください。表の列（AP、Latency など）の説明は [用語解説](#用語解説--glossary) を参照してください。

### Detection / 物体検出

<img alt="rf_detr_1-4_latency_accuracy_object_detection" src="https://storage.googleapis.com/com-roboflow-marketing/rf-detr/rf_detr_1-4_latency_accuracy_object_detection.png" />

<details>
<summary>See object detection benchmark numbers / 物体検出ベンチマーク数値を見る</summary>

<br>

| Architecture  | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | RF100VL AP<sub>50</sub> | RF100VL AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |  License   |
| :-----------: | :------------------: | :---------------------: | :---------------------: | :------------------------: | :----------: | :--------: | :--------: | :--------: |
|   RF-DETR-N   |         67.6         |          48.4           |          85.0           |            57.7            |     2.3      |    30.5    |  384x384   | Apache 2.0 |
|   RF-DETR-S   |         72.1         |          53.0           |          86.7           |            60.2            |     3.5      |    32.1    |  512x512   | Apache 2.0 |
|   RF-DETR-M   |         73.6         |          54.7           |          87.4           |            61.2            |     4.4      |    33.7    |  576x576   | Apache 2.0 |
|   RF-DETR-L   |         75.1         |          56.5           |          88.2           |            62.2            |     6.8      |    33.9    |  704x704   | Apache 2.0 |
| RF-DETR-XL △  |         77.4         |          58.6           |          88.5           |            62.9            |     11.5     |   126.4    |  700x700   |  PML 1.0   |
| RF-DETR-2XL △ |         78.5         |          60.1           |          89.0           |            63.2            |     17.2     |   126.9    |  880x880   |  PML 1.0   |
|   YOLO11-N    |         52.0         |          37.4           |          81.4           |            55.3            |     2.5      |    2.6     |  640x640   |  AGPL-3.0  |
|   YOLO11-S    |         59.7         |          44.4           |          82.3           |            56.2            |     3.2      |    9.4     |  640x640   |  AGPL-3.0  |
|   YOLO11-M    |         64.1         |          48.6           |          82.5           |            56.5            |     5.1      |    20.1    |  640x640   |  AGPL-3.0  |
|   YOLO11-L    |         64.9         |          49.9           |          82.2           |            56.5            |     6.5      |    25.3    |  640x640   |  AGPL-3.0  |
|   YOLO11-X    |         66.1         |          50.9           |          81.7           |            56.2            |     10.5     |    56.9    |  640x640   |  AGPL-3.0  |
|   YOLO26-N    |         55.8         |          40.3           |          76.7           |            52.0            |     1.7      |    2.6     |  640x640   |  AGPL-3.0  |
|   YOLO26-S    |         64.3         |          47.7           |          82.7           |            57.0            |     2.6      |    9.4     |  640x640   |  AGPL-3.0  |
|   YOLO26-M    |         69.7         |          52.5           |          84.4           |            58.7            |     4.4      |    20.1    |  640x640   |  AGPL-3.0  |
|   YOLO26-L    |         71.1         |          54.1           |          85.0           |            59.3            |     5.7      |    25.3    |  640x640   |  AGPL-3.0  |
|   YOLO26-X    |         74.0         |          56.9           |          85.6           |            60.0            |     9.6      |    56.9    |  640x640   |  AGPL-3.0  |
|   LW-DETR-T   |         60.7         |          42.9           |          84.7           |            57.1            |     1.9      |    12.1    |  640x640   | Apache 2.0 |
|   LW-DETR-S   |         66.8         |          48.0           |          85.0           |            57.4            |     2.6      |    14.6    |  640x640   | Apache 2.0 |
|   LW-DETR-M   |         72.0         |          52.6           |          86.8           |            59.8            |     4.4      |    28.2    |  640x640   | Apache 2.0 |
|   LW-DETR-L   |         74.6         |          56.1           |          87.4           |            61.5            |     6.9      |    46.8    |  640x640   | Apache 2.0 |
|   LW-DETR-X   |         76.9         |          58.3           |          87.9           |            62.1            |     13.0     |   118.0    |  640x640   | Apache 2.0 |
|   D-FINE-N    |         60.2         |          42.7           |          84.4           |            58.2            |     2.1      |    3.8     |  640x640   | Apache 2.0 |
|   D-FINE-S    |         67.6         |          50.6           |          85.3           |            60.3            |     3.5      |    10.2    |  640x640   | Apache 2.0 |
|   D-FINE-M    |         72.6         |          55.0           |          85.5           |            60.6            |     5.4      |    19.2    |  640x640   | Apache 2.0 |
|   D-FINE-L    |         74.9         |          57.2           |          86.4           |            61.6            |     7.5      |    31.0    |  640x640   | Apache 2.0 |
|   D-FINE-X    |         76.8         |          59.3           |          86.9           |            62.2            |     11.5     |    62.0    |  640x640   | Apache 2.0 |

</details>

### Segmentation / セグメンテーション

<img alt="rf_detr_1-4_latency_accuracy_instance_segmentation" src="https://storage.googleapis.com/com-roboflow-marketing/rf-detr/rf_detr_1-4_latency_accuracy_instance_segmentation.png" />

<details>
<summary>See instance segmentation benchmark numbers / インスタンスセグメンテーションベンチマーク数値を見る</summary>

<br>

|  Architecture   | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |  License   |
| :-------------: | :------------------: | :---------------------: | :----------: | :--------: | :--------: | :--------: |
|  RF-DETR-Seg-N  |         63.0         |          40.3           |     3.4      |    33.6    |  312x312   | Apache 2.0 |
|  RF-DETR-Seg-S  |         66.2         |          43.1           |     4.4      |    33.7    |  384x384   | Apache 2.0 |
|  RF-DETR-Seg-M  |         68.4         |          45.3           |     5.9      |    35.7    |  432x432   | Apache 2.0 |
|  RF-DETR-Seg-L  |         70.5         |          47.1           |     8.8      |    36.2    |  504x504   | Apache 2.0 |
| RF-DETR-Seg-XL  |         72.2         |          48.8           |     13.5     |    38.1    |  624x624   | Apache 2.0 |
| RF-DETR-Seg-2XL |         73.1         |          49.9           |     21.8     |    38.6    |  768x768   | Apache 2.0 |
|  YOLOv8-N-Seg   |         45.6         |          28.3           |     3.5      |    3.4     |  640x640   |  AGPL-3.0  |
|  YOLOv8-S-Seg   |         53.8         |          34.0           |     4.2      |    11.8    |  640x640   |  AGPL-3.0  |
|  YOLOv8-M-Seg   |         58.2         |          37.3           |     7.0      |    27.3    |  640x640   |  AGPL-3.0  |
|  YOLOv8-L-Seg   |         60.5         |          39.0           |     9.7      |    46.0    |  640x640   |  AGPL-3.0  |
|  YOLOv8-XL-Seg  |         61.3         |          39.5           |     14.0     |    71.8    |  640x640   |  AGPL-3.0  |
|  YOLOv11-N-Seg  |         47.8         |          30.0           |     3.6      |    2.9     |  640x640   |  AGPL-3.0  |
|  YOLOv11-S-Seg  |         55.4         |          35.0           |     4.6      |    10.1    |  640x640   |  AGPL-3.0  |
|  YOLOv11-M-Seg  |         60.0         |          38.5           |     6.9      |    22.4    |  640x640   |  AGPL-3.0  |
|  YOLOv11-L-Seg  |         61.5         |          39.5           |     8.3      |    27.6    |  640x640   |  AGPL-3.0  |
| YOLOv11-XL-Seg  |         62.4         |          40.1           |     13.7     |    62.1    |  640x640   |  AGPL-3.0  |
|  YOLO26-N-Seg   |         54.3         |          34.7           |     2.31     |    2.7     |  640x640   |  AGPL-3.0  |
|  YOLO26-S-Seg   |         62.4         |          40.2           |     3.47     |    10.4    |  640x640   |  AGPL-3.0  |
|  YOLO26-M-Seg   |         67.8         |          44.0           |     6.32     |    23.6    |  640x640   |  AGPL-3.0  |
|  YOLO26-L-Seg   |         69.8         |          45.5           |     7.58     |    28.0    |  640x640   |  AGPL-3.0  |
|  YOLO26-X-Seg   |         71.6         |          46.8           |    12.92     |    62.8    |  640x640   |  AGPL-3.0  |

</details>

### Keypoints / キーポイント

<img alt="RF-DETR Keypoint mAP vs latency chart comparing against YOLO26-pose and YOLO11-pose on MS COCO" src="https://raw.githubusercontent.com/roboflow/rf-detr/develop/docs/assets/keypoints/kp-map-latency.png" />

<details>
<summary>See keypoint detection benchmark numbers / キーポイント検出ベンチマーク数値を見る</summary>

<br>

|        Architecture        | COCO AP<sub>50:95</sub> | Latency (ms) |  License   |
| :------------------------: | :---------------------: | :----------: | :--------: |
| RF-DETR Keypoint (Preview) |          71.8           |     9.7      | Apache 2.0 |
|       YOLO11-pose N        |          48.9           |     3.2      |  AGPL-3.0  |
|       YOLO11-pose S        |          57.5           |     3.4      |  AGPL-3.0  |
|       YOLO11-pose M        |          64.2           |     5.2      |  AGPL-3.0  |
|       YOLO11-pose L        |          65.2           |     6.6      |  AGPL-3.0  |
|       YOLO11-pose X        |          68.6           |     10.6     |  AGPL-3.0  |
|       YOLO26-pose N        |          55.9           |     1.9      |  AGPL-3.0  |
|       YOLO26-pose S        |          62.0           |     2.7      |  AGPL-3.0  |
|       YOLO26-pose M        |          68.0           |     4.6      |  AGPL-3.0  |
|       YOLO26-pose L        |          69.2           |     5.9      |  AGPL-3.0  |
|       YOLO26-pose X        |          71.0           |     9.8      |  AGPL-3.0  |

</details>

> Keypoint benchmarks report AP<sub>50:95</sub> (OKS-based); this is the standard COCO keypoint comparison metric.
>
> **日本語:** キーポイントベンチマークは AP<sub>50:95</sub>（OKS ベース）を報告します。これは COCO キーポイント比較の標準指標です。AP・OKS の意味は [用語解説](#用語解説--glossary) を参照してください。

## Run Models / モデルの実行

### Detection / 物体検出

RF-DETR provides multiple model sizes, ranging from Nano to 2XLarge. To use a different model size, replace the class name in the code snippet below with another class from the table.

> **日本語:** RF-DETR には Nano から 2XLarge まで複数のモデルサイズがあります。別サイズを使う場合は、下記コードのクラス名を表の別クラスに置き換えてください。

```python
import supervision as sv
from rfdetr import RFDETRMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRMedium()

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
```

<details>
<summary>Run RF-DETR with Inference / Inference ライブラリで RF-DETR を実行</summary>

<br>

You can also run RF-DETR models using the Inference library. To switch model size, select the appropriate inference package alias from the table below.

> **日本語:** Inference ライブラリを使って RF-DETR モデルを実行することもできます。モデルサイズを切り替えるには、下表の Inference パッケージエイリアスを選択してください。

```python
import requests
import supervision as sv
from PIL import Image
from inference import get_model

model = get_model("rfdetr-medium")

image = Image.open(requests.get("https://media.roboflow.com/dog.jpg", stream=True).raw)
predictions = model.infer(image, confidence=0.5)[0]
detections = sv.Detections.from_inference(predictions)

annotated_image = sv.BoxAnnotator().annotate(image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)
```

</details>

| Size | RF-DETR package class | Inference package alias | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |  License   |
| :--: | :-------------------: | :---------------------- | :------------------: | :---------------------: | :----------: | :--------: | :--------: | :--------: |
|  N   |     `RFDETRNano`      | `rfdetr-nano`           |         67.6         |          48.4           |     2.3      |    30.5    |  384x384   | Apache 2.0 |
|  S   |     `RFDETRSmall`     | `rfdetr-small`          |         72.1         |          53.0           |     3.5      |    32.1    |  512x512   | Apache 2.0 |
|  M   |    `RFDETRMedium`     | `rfdetr-medium`         |         73.6         |          54.7           |     4.4      |    33.7    |  576x576   | Apache 2.0 |
|  L   |     `RFDETRLarge`     | `rfdetr-large`          |         75.1         |          56.5           |     6.8      |    33.9    |  704x704   | Apache 2.0 |
|  XL  |   `RFDETRXLarge` △    | `rfdetr-xlarge`         |         77.4         |          58.6           |     11.5     |   126.4    |  700x700   |  PML 1.0   |
| 2XL  |   `RFDETR2XLarge` △   | `rfdetr-2xlarge`        |         78.5         |          60.1           |     17.2     |   126.9    |  880x880   |  PML 1.0   |

> △ Requires the `rfdetr_plus` extension: `pip install rfdetr[plus]`. See [License](#license--ライセンス) for details.
>
> **日本語:** △ は `rfdetr_plus` 拡張が必要です: `pip install rfdetr[plus]`。詳細は [License / ライセンス](#license--ライセンス) を参照してください。

### Segmentation / セグメンテーション

RF-DETR supports instance segmentation with model sizes from Nano to 2XLarge. To use a different model size, replace the class name in the code snippet below with another class from the table.

> **日本語:** RF-DETR は Nano から 2XLarge までのサイズでインスタンスセグメンテーションをサポートします。別サイズを使う場合は、下記コードのクラス名を表の別クラスに置き換えてください。

```python
import supervision as sv
from rfdetr import RFDETRSegMedium
from rfdetr.assets.coco_classes import COCO_CLASSES

model = RFDETRSegMedium()

detections = model.predict("https://media.roboflow.com/dog.jpg", threshold=0.5)

labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.MaskAnnotator().annotate(detections.metadata["source_image"], detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
```

<details>
<summary>Run RF-DETR-Seg with Inference / Inference ライブラリで RF-DETR-Seg を実行</summary>

<br>

You can also run RF-DETR-Seg models using the Inference library. To switch model size, select the appropriate inference package alias from the table below.

> **日本語:** Inference ライブラリを使って RF-DETR-Seg モデルを実行することもできます。モデルサイズを切り替えるには、下表の Inference パッケージエイリアスを選択してください。

```python
import requests
import supervision as sv
from PIL import Image
from inference import get_model

model = get_model("rfdetr-seg-medium")

image = Image.open(requests.get("https://media.roboflow.com/dog.jpg", stream=True).raw)
predictions = model.infer(image, confidence=0.5)[0]
detections = sv.Detections.from_inference(predictions)

annotated_image = sv.MaskAnnotator().annotate(image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)
```

</details>

| Size | RF-DETR package class | Inference package alias | COCO AP<sub>50</sub> | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |  License   |
| :--: | :-------------------: | :---------------------- | :------------------: | :---------------------: | :----------: | :--------: | :--------: | :--------: |
|  N   |    `RFDETRSegNano`    | `rfdetr-seg-nano`       |         63.0         |          40.3           |     3.4      |    33.6    |  312x312   | Apache 2.0 |
|  S   |   `RFDETRSegSmall`    | `rfdetr-seg-small`      |         66.2         |          43.1           |     4.4      |    33.7    |  384x384   | Apache 2.0 |
|  M   |   `RFDETRSegMedium`   | `rfdetr-seg-medium`     |         68.4         |          45.3           |     5.9      |    35.7    |  432x432   | Apache 2.0 |
|  L   |   `RFDETRSegLarge`    | `rfdetr-seg-large`      |         70.5         |          47.1           |     8.8      |    36.2    |  504x504   | Apache 2.0 |
|  XL  |   `RFDETRSegXLarge`   | `rfdetr-seg-xlarge`     |         72.2         |          48.8           |     13.5     |    38.1    |  624x624   | Apache 2.0 |
| 2XL  |  `RFDETRSeg2XLarge`   | `rfdetr-seg-2xlarge`    |         73.1         |          49.9           |     21.8     |    38.6    |  768x768   | Apache 2.0 |

### Keypoints / キーポイント

RF-DETR supports keypoint detection (preview) with `RFDETRKeypointPreview`, pretrained on COCO person keypoints.

> **日本語:** RF-DETR は `RFDETRKeypointPreview` によるキーポイント検出（プレビュー）をサポートしています。COCO 人物キーポイントで事前学習済みです。

```python
from rfdetr import RFDETRKeypointPreview

model = RFDETRKeypointPreview()
key_points = model.predict("image.jpg", threshold=0.5)
```

|        Size        |  RF-DETR package class  | COCO AP<sub>50:95</sub> | Latency (ms) | Params (M) | Resolution |  License   |
| :----------------: | :---------------------: | :---------------------: | :----------: | :--------: | :--------: | :--------: |
| Keypoint (Preview) | `RFDETRKeypointPreview` |          71.8           |     9.7      |   126.4    |  576x576   | Apache 2.0 |

### Train Models / モデルの学習

RF-DETR supports training for object detection, instance segmentation, and keypoint detection (preview). You can train models in [Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb) or directly on the Roboflow platform. Below you will find a step-by-step video fine-tuning tutorial.

> **日本語:** RF-DETR は物体検出・インスタンスセグメンテーション・キーポイント検出（プレビュー）の学習をサポートしています。[Google Colab](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb) または Roboflow プラットフォーム上で学習できます。以下にファインチューニングのステップバイステップ動画チュートリアルがあります。

[![rf-detr-tutorial-banner](https://github.com/user-attachments/assets/555a45c3-96e8-4d8a-ad29-f23403c8edfd)](https://youtu.be/-OvpdLAElFA)

## Documentation / ドキュメント

Visit our [documentation website](https://rfdetr.roboflow.com) to learn more about how to use RF-DETR.

> **日本語:** RF-DETR の使い方の詳細は [ドキュメントサイト](https://rfdetr.roboflow.com) をご覧ください。

**ER-FlowScan フォーク向け（日本語）:**

| 資料 | 内容 |
|------|------|
| [docs/ja/README.md](docs/ja/README.md) | 日本語ドキュメント索引 |
| [docs/ja/models-and-coco-classes.md](docs/ja/models-and-coco-classes.md) | モデル・COCO クラス解説 |
| [docs/ja/vast-ai-integration-guide.md](docs/ja/vast-ai-integration-guide.md) | Vast.ai 組み込みガイド（他プロジェクト共通化） |

## License / ライセンス

Licensing is split by component:

> **日本語:** ライセンスはコンポーネントごとに分かれています:

- The open-source `rfdetr` package and Apache-designated model weights are licensed under Apache License 2.0. See [`LICENSE`](LICENSE).
- Plus components, including the `rfdetr_plus` extension and RF-DETR-XL / RF-DETR-2XL detection models, are licensed under PML 1.0.

> **日本語:**
>
> - オープンソースの `rfdetr` パッケージおよび Apache 指定モデルウェイトは Apache License 2.0 です。詳細は [`LICENSE`](LICENSE) を参照してください。
> - Plus コンポーネント（`rfdetr_plus` 拡張、RF-DETR-XL / RF-DETR-2XL 検出モデルを含む）は PML 1.0 です。

## Acknowledgements / 謝辞

Our work is built upon [LW-DETR](https://arxiv.org/pdf/2406.03459), [DINOv2](https://arxiv.org/pdf/2304.07193), and [Deformable DETR](https://arxiv.org/pdf/2010.04159). Thanks to their authors for their excellent work!

> **日本語:** 本プロジェクトは [LW-DETR](https://arxiv.org/pdf/2406.03459)、[DINOv2](https://arxiv.org/pdf/2304.07193)、[Deformable DETR](https://arxiv.org/pdf/2010.04159) を基盤としています。素晴らしい研究を公開してくださった著者の皆様に感謝します。

## Citation / 引用

If you find our work helpful for your research, please consider citing the following BibTeX entry.

> **日本語:** 研究に役立つ場合は、以下の BibTeX エントリの引用をご検討ください。

```bibtex
@misc{rf-detr,
    title={RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
    author={Isaac Robinson and Peter Robicheaux and Matvei Popov and Deva Ramanan and Neehar Peri},
    year={2025},
    eprint={2511.09554},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2511.09554},
}
```

## Contribute / コントリビュート

We welcome and appreciate all contributions! If you notice any issues or bugs, have questions, or would like to suggest new features, please [open an issue](https://github.com/roboflow/rf-detr/issues/new) or pull request. By sharing your ideas and improvements, you help make RF-DETR better for everyone.

> **日本語:** すべてのコントリビューションを歓迎します。不具合や質問、新機能の提案がある場合は [issue を作成](https://github.com/roboflow/rf-detr/issues/new) するか pull request を送ってください。皆さんのアイデアと改善が RF-DETR をより良くします。

<p align="center">
    <a href="https://youtube.com/roboflow"><img src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://roboflow.com"><img src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://www.linkedin.com/company/roboflow-ai/"><img src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://docs.roboflow.com"><img src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://discuss.roboflow.com"><img src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584" width="3%"/></a>
    <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
    <a href="https://blog.roboflow.com"><img src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605" width="3%"/></a>
</p>
