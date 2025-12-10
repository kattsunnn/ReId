## 概要
- **Project**: `ReId` — Re-Identificationツール
- **スクリプト**: `osnet_reid.py` — ディレクトリ内の画像を特徴量に変換し、DBSCAN でクラスタリングします。

## セットアップ
- 仮想環境を作成して依存ライブラリ(requirements.txt)を入れてください。
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 使い方（CLI）
- **コマンド**: `python osnet_reid.py INPUT_DIR`
- **引数**: `INPUT_DIR` は画像が置かれたディレクトリ（`jpg|jpeg|png` を探索）

例:
```bash
python osnet_reid.py ./images/
```
**出力のイメージ**
- スクリプトを実行すると、各クラスタのラベルとメンバー画像パスが標準出力に表示されます。
```
Cluster 0:
	./images/img_0001.jpg
	./images/img_0042.jpg
Cluster 1:
	./images/img_0007.jpg
	./images/img_0103.jpg
Cluster -1:  # ノイズ（どのクラスタにも属さない画像）
	./images/img_0099.jpg
```

## 使い方（モジュール）
- `OSNetReID` クラスをインポートして使用できます。初回でモデルが遅延初期化されます。

```python
from osnet_reid import OSNetReID
groups = OSNetReID.cluster_imgs(list_of_image_paths)
```

**注意点 / 補足**
- `OSNetReID` はクラス変数で `FeatureExtractor` を保持する設計になっています（遅延初期化）。これにより不要なときにモデルをロードせずに済みます。
- マルチスレッド環境で同時に初回初期化を行う可能性がある場合は、ロックで保護することを検討してください。

