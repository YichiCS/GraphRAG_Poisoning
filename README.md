# PoisonGraphRAG


**Environment**


```shell
git clone https://github.com/microsoft/graphrag # 3.0.0 other
cd graphrag
pip install -e.
pip install ollama
pip install plotly
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers
```

*Attention:* This work is based on graphrag=3.0.0
*Attention:* You are supposed to replace the files in `.graph\hackfile` to make `GraphRAG` support `ollama`.
*TODO:* One-Step environment set-up.

**Test**

```shell
python main.py --mini 2 --normal    # Construct Clean RAG System, neccesary for following steps.
python main.py --mini 2             # Perform Attack(default, PoisonedRAG) and Evaluation
```