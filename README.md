UNdreaMT: Unsupervised Neural Machine Translation
==============

This is an open source implementation of our unsupervised neural machine translation system, described in the following paper:

Mikel Artetxe, Gorka Labaka, Eneko Agirre, and Kyunghyun Cho. 2018. **[Unsupervised Neural Machine Translation](https://arxiv.org/pdf/1710.11041.pdf)**. In *Proceedings of the Sixth International Conference on Learning Representations (ICLR 2018)*.

If you use this software for academic research, please cite the paper in question:
```
@inproceedings{artetxe2018iclr,
  author    = {Artetxe, Mikel  and  Labaka, Gorka  and  Agirre, Eneko  and  Cho, Kyunghyun},
  title     = {Unsupervised neural machine translation},
  booktitle = {Proceedings of the Sixth International Conference on Learning Representations},
  month     = {April},
  year      = {2018}
}
```

**NOTE**: This software has been superseded by [Monoses](https://github.com/artetxem/monoses), our unsupervised statistical machine translation system. [Monoses](https://github.com/artetxem/monoses) obtains substantially better results (e.g. 26.2 vs 15.1 BLEU in English-French WMT14), so we strongly recommend that you switch to it.


Requirements
--------
- Python 3
- PyTorch (tested with v0.3)


Usage
--------

The following command trains an unsupervised NMT system from monolingual corpora using the exact same settings described in the paper:

```
python3 train.py --src SRC.MONO.TXT --trg TRG.MONO.TXT --src_embeddings SRC.EMB.TXT --trg_embeddings TRG.EMB.TXT --save MODEL_PREFIX --cuda
```

The data in the above command should be provided as follows:
- `SRC.MONO.TXT` and `TRG.MONO.TXT` are the source and target language monolingual corpora. They should both be pre-processed so atomic symbols (either tokens or BPE units) are separated by whitespaces. For that purpose, we recommend using [Moses](http://www.statmt.org/moses/) to tokenize and truecase the corpora and, optionally, [Subword-NMT](https://github.com/rsennrich/subword-nmt) if you want to use BPE.
- `SRC.EMB.TXT` and `TRG.EMB.TXT` are the source and target language cross-lingual embeddings. In order to obtain them, we recommend training monolingual embeddings in the corpora above using either [word2vec](https://github.com/tmikolov/word2vec) or [fasttext](https://github.com/facebookresearch/fastText), and then map them to a shared space using [VecMap](https://github.com/artetxem/vecmap). Please make sure to cutoff the vocabulary as desired before mapping the embeddings.
- `MODEL_PREFIX` is the prefix of the output model.

Using the above settings, training takes about 3 days in a single Titan Xp. Once training is done, you can use the resulting model for translation as follows:

```
python3 translate.py MODEL_PREFIX.final.src2trg.pth < INPUT.TXT > OUTPUT.TXT
```

For more details and additional options, run the above scripts with the `--help` flag.


FAQ
--------

##### I have seen that you have a separate unsupervised SMT system called [Monoses](https://github.com/artetxem/monoses). Which one should I use?

You should definitely use [Monoses](https://github.com/artetxem/monoses). It is newer and obtains substantially better results (e.g. 26.2 vs 15.1 BLEU in English-French WMT14), so we strongly recommend that you switch to it.


##### You claim that your unsupervised NMT system is trained on monolingual corpora alone, but it also requires bilingual embeddings... Isn't that cheating?

Not really, because we also learn the bilingual embeddings from monolingual corpora alone. We use our companion tool [VecMap](https://github.com/artetxem/vecmap) for that.


##### Can I use this software to train a regular NMT system on parallel corpora?

Yes! You can use the following arguments to make UNdreaMT behave like a regular NMT system:

```
python3 train.py --src2trg SRC.PARALLEL.TXT TRG.PARALLEL.TXT --src_vocabulary SRC.VOCAB.TXT --trg_vocabulary TRG.VOCAB.TXT --embedding_size 300 --learn_encoder_embeddings --disable_denoising --save MODEL_PREFIX --cuda
```


License
-------

Copyright (C) 2018, Mikel Artetxe

Licensed under the terms of the GNU General Public License, either version 3 or (at your option) any later version. A full copy of the license can be found in LICENSE.txt.
