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

### Usage
@TODO

### Outline

[**Task**] Translation: German -> English

[**Process**]
1. Preprocessing
```
[Sentence] -> [TorchText Field]
```

2. Model
```
[Source Sentence] -> (Encoder) -> [Context Vector]
```

3.

### Implementation Reference
- [CS224n Lecture 8](http://web.stanford.edu/class/cs224n/)
- [TorchText Docs](https://torchtext.readthedocs.io/en/latest/)
- [TorchText Tutorial (블로그)](https://simonjisu.github.io/nlp/2018/07/18/torchtext.html)
- [A Tutorial on Torchtext (Blog)](http://anie.me/On-Torchtext/)
