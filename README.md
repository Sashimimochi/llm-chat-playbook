# LLM Chat Playbook

ローカルLLMを使って各種機能を試すサンプルコード

## Quick Start

```bash
$ sh setup.sh
$ docker-compose up -d
```

open http://localhost:8503

## Multi Turn Chat

![](./images/chat01.png)
![](./images/chat02.png)
![](./images/chat03.png)

## RAG

参考情報を1以上にしてチャットを打つと、閾値以上の類似度のドキュメントがあれば、そのテキストを参照して回答を生成する。

![](./images/rag.png)

### インデックスデータの作成

`data` ディレクトリ配下にインデックスさせたい `.txt` や `.pdf` を配置する。

```bash
$ tree data/
data/
└── easyeasy.txt
```

Create Indexボタンを押す。
しばらく待つとインデックスデータが作成される。

![](./images/create_index.png)

※大量のデータやサイズの大きいPDFファイルをインデックスしようとすると、処理に失敗してシステムがクラッシュすることがあるので注意

## Function Calling

ModelタイプでFunctionCallingを選択すると外部APIを呼んで回答を生成する。
本サンプルコードでは天気APIを参照して、指定した地域の天気を回答する。

![](./images/function_calling.png)

## Special Thanks

- https://weather.tsukumijima.net/
- https://easy2.connpass.com/
