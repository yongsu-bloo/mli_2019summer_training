# Seq2Seq Implementation

### Original Paper
Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).[[link](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks)]
```
@inproceedings{sutskever2014sequence,
  title={Sequence to sequence learning with neural networks},
  author={Sutskever, Ilya and Vinyals, Oriol and Le, Quoc V},
  booktitle={Advances in neural information processing systems},
  pages={3104--3112},
  year={2014}
}
```
### Requirements
***! Use python3, pip3 instead depending on your system***
##### Dependencies

```bash
pip install -r requirements.txt
```

##### Datasets

```bash
python -m spacy download en
python -m spacy download de
```
### Usage
##### Train and Evaluation
```bash
python nmt [options]
```
##### Option Description
```
usage: nmt.py [-h] [-seed SEED] [-b BATCH_SIZE] [-num-layers NUM_LAYERS]
              [-emd-dim EMD_DIM] [-hidden-dim HIDDEN_DIM] [--no-reverse]
              [--bidirectional] [-lr LR] [-rnn-type {LSTM,GRU}]
              [-opt {adam,sgd}] [-epochs EPOCHS] [-dropout DROPOUT] [--cpu]
              [-resume RESUME] [--evaluate] [-v VERBOSE]
              [--local_rank LOCAL_RANK] [--no-multi]

optional arguments:
  -h, --help            show this help message and exit
  -seed SEED
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size(default=128)
  -num-layers NUM_LAYERS
  -emd-dim EMD_DIM
  -hidden-dim HIDDEN_DIM
  --no-reverse          not to reverse input seq
  --bidirectional       bidirectional rnn
  -lr LR
  -rnn-type {LSTM,GRU}  LSTM or GRU
  -opt {adam,sgd}
  -epochs EPOCHS
  -dropout DROPOUT      dropout rate
  --cpu                 forcing to use cpu
  -resume RESUME        load model from checkpoint(input: path of ckpt)
  --evaluate            Not train, Only evaluate
  -v VERBOSE, --verbose VERBOSE
                        0: nothing, 1: test only, else: eval and test
  --local_rank LOCAL_RANK
  --no-multi            use single gpu
```

### Outline

[**Task**] Translation: German -> English

[**Process**]
1. Preprocessing
```
[Sentence] -> [TorchText Field]
```

2. Model
```
[Source Sentence] -> (Encoder) -> [Context Vector] -> (Decoder) -> [Target Sentence]
```

3.

### Implementation Reference
- [CS224n Lecture 8](http://web.stanford.edu/class/cs224n/)
- [TorchText Docs](https://torchtext.readthedocs.io/en/latest/)
- [TorchText Tutorial (블로그)](https://simonjisu.github.io/nlp/2018/07/18/torchtext.html)
- [A Tutorial on Torchtext (Blog)](http://anie.me/On-Torchtext/)
