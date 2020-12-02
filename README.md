# TGCN: A Novel Deep Learning Model for Text Classification

This project contains 

- Re-implementation of ["Graph Convolutional Networks for Text Classification"](https://arxiv.org/abs/1809.05679) in tensorflow 2.1.
- Some baseline models mentioned in original paper.


## Requirement

- python 3.6
- tensorflow 2.1.0
- nltk 3.4.5
- fasttext 0.9.2 (Optional)

## Run

### Preprocess

```bash
cd ./preprocess
python remove_words.py <dataset>
python build_graph.py <dataset>
```

### Training

```bash
cd ..
python train.py --dataset <dataset>
```
## Reference

- The official implementation: https://github.com/yao8839836/text_gcn
- PyTorch version: https://github.com/iworldtong/text_gcn.pytorch
- Paper: https://arxiv.org/abs/1809.05679


## Contributors:
* [Yuan Li](yl6606@nyu.edu)
* [Dongzi Qi](dq394@nyu.edu)
