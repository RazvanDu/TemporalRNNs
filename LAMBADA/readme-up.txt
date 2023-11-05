------------------LAMBADA DATASET------------------

This archive contains the LAMBADA dataset (LAnguage Modeling Broadened to
Account for Discourse Aspects) described in D. Paperno, G. Kruszewski, A.
Lazaridou, Q. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda and R.
Fernandez. 2016. The LAMBADA dataset: Word prediction requiring a broad
discourse context. Proceedings of ACL 2016 (54th Annual Meeting of the
Association for Computational Linguistics), East Stroudsburg PA: ACL, pages
1525-1534. The source data come from the Book Corpus, made in turn of
unpublished novels (see Y. Zhu, R. Kiros, R.f Zemel, R. Salakhutdinov, R.
Urtasun, A. Torralba and S. Fidler. Aligning books and movies: Towards
story-like visual explanations by watching movies and reading books. ICCV
2015, pages 19-27).



You will find 5 files besides this readme in the archive:

1. lambada_development_plain_text.txt

	The development data include 4,869 test passages (extracted from 1,331
novels, disjoint from the rest).

2. lambada_test_plain_text.txt

	The test data include 5,153 test passages (extracted from 1,332
novels, disjoint from the rest).

3. lambada_control_test_data_plain_text.txt

	The control data is a set of sentences randomly sampled from the same
novels, and of the same shape and size as the ones used to build the test
dataset, but not filtered in any way. This is the set referred to as the
"control" set in the paper.

---NOTE: In these 3 files each line corresponds to a passage, including
context, target sentence, and target word. For each passage, the word to be
guessed is the last one.

4. train-novels.tar

	The training data include the full text of 2,662 novels (disjoint from
those in dev+test), comprising more than 200M words. It consists of text from
the same domain as the dev+test passages, but not filtered in any way.

---NOTE: Development/test/control (1-3) and train (4) sentences have been
tokenized in the same way.

5. lambada-vocab-2.txt

	This is the alphabetically sorted list of words from which the one to
be guessed must be picked. It includes 112,745 types.



If you use the dataset in published work, PLEASE CITE THE LAMBADA PAPER:

@InProceedings{paperno-EtAl:2016:P16-1,
  author    = {Paperno, Denis  and  Kruszewski, Germ\'{a}n  and  Lazaridou,
Angeliki  and  Pham, Ngoc Quan  and  Bernardi, Raffaella  and  Pezzelle,
Sandro  and  Baroni, Marco  and  Boleda, Gemma  and  Fernandez, Raquel},
  title     = {The {LAMBADA} dataset: Word prediction requiring a broad
discourse context},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers)},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics},
  pages     = {1525--1534},
  url       = {http://www.aclweb.org/anthology/P16-1144}
}



First released on 2016, September 26.
