# Where to start in the NLP jungle?

_(This document is intended for students either writing their (nlp-related) bachelor or master thesis or working on their (nlp-related) consulting project under our supervision.)_

_Author:_ Matthias Aßenmacher // Christian Heumann

_(Please note that this document is subject to continuous change. Every time we find a new, nice source we will add it. Some stuff might also be deleted over time if not considered useful anymore.)_

## I. Set up all necessary communication channels

- [ ] Join Mattermost (ask Matthias for the invite link)
- [ ] Ask Matthias to add you to the “NLP”-channel on Mattermost
- [ ] Ask Matthias to add you to our internal “NLP” mailing list
- [ ] You can reach Matthias via Mattermost or E-Mail, Christian prefers to be contacted via E-Mail  
     (In case of e-mails related to your thesis/project, make sure to cc the respective other in order to create no information asymmetries)
- [ ] We will have a so-called “NLP Colloquium” every now and then (intended 4 times a year) where all of our BA-/MA-/Consulting- students present their work to the others. This meeting is of rather informal character (mostly intended to connect you to each other), so no need for high-gloss slides, or anything like that. Everything like jupyter notebooks/interesting figures/slides is fine.  
     We will announce this via the mailing list and via Mattermost.  
     Dates for 2021: 26.03., 25.06., 15.10., 17.12. at 13h s.t.
- [ ] The mailing list will be mostly used for announcements, while in the Mattermost channel we will occasionally also post (nlp-related) stuff we consider interesting.
- [ ] TALK TO US rather sooner than later if any problems occur which you are not able to solve on your own. Open (and timely) communication is (in our opinion) key to a successful supervision/cooperation during theses or consulting projects.

## II. Useful materials for starting with NLP

### 1. Get familiar with the basics:

- [ ] Pre-Processing (e.g. in Python with NLTK or spacy)
- [ ] The bag-of-words approach, its applications in ML, drawbacks & limitations (just google this stuff, you will find enough material).
- [ ] Extensions of the bow approach, like n-grams or tf-idf (just google this stuff, you will find enough material).

### 2. Get familiar with the Python environment:

- [ ] In general
- [ ] Notebooks
- [ ] Colab 

### 3. Start looking into neural networks and deep learning for NLP

- [ ] Pretty nice book for an a broad overview on everything until self-attention, useful for covering the basics: [Goldberg (2017)](https://github.com/Michael2Tang/ML_Doc/blob/master/Neural%20Network%20Methods%20in%20Natural%20Language%20Processing-Morgan%20%26%20Claypool%20Publishers%20(2017)%20-%20Yoav%20Goldberg%2C%20Graeme%20Hirst.pdf)
- [ ] Good overview on Embeddings: [Pilehvar & Camacho-Collados (2021)](http://pilevar.com/taher/files/Embeddings_in_Natural_Language_Processing.pdf)
- [ ] Overview on DL in general: [Deep Learning book](https://www.deeplearningbook.org/)
- [ ] Or more basic: [Intro to Statistical learning](http://faculty.marshall.usc.edu/gareth-james/ISL/)
- [ ] Our lecture from WS 20/21: https://moodle.lmu.de/course/view.php?id=10268 

### 4. Milestone papers:

- [ ] __Important conceptual foundation:__ Bengio, Y., et al. (2003) ["A neural probabilistic language model."](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) Journal of machine learning research: 1137-1155.
- [ ] Mikolov, Tomas, et al. "Efficient estimation of word representations in vector space." arXiv preprint arXiv:1301.3781(2013).  
      Mikolov, Tomas, et al. "Distributed representations of words and phrases and their compositionality." Advances in neural information processing systems. 2013.  
      Modify the idea of Bengio et al., use the internal representations of the neural net as the primary objective. They are called word embeddings.
- [ ] Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.  
      Alternative to word2vec.
- [ ] Le, Quoc, and Tomas Mikolov. "Distributed representations of sentences and documents." International conference on machine learning. 2014.  
      Extend embedding idea to sentence and document level.
- [ ] Bojanowski, Piotr, et al. "Enriching word vectors with subword information." Transactions of the Association for Computational Linguistics 5 (2017): 135-146.  
      Joulin, Armand, et al. "Bag of tricks for efficient text classification." arXiv preprint arXiv:1607.01759 (2016).  
      Extend the embedding idea to subword tokens.
- [ ] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." arXiv preprint arXiv:1409.0473 (2014).
- [ ] Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective approaches to attention-based neural machine translation." arXiv preprint arXiv:1508.04025 (2015).  
      Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017.  
      Important foundations for the so-called (Self-)Attention mechanism.
- [ ] Some of the most famous models that learn (a) contextualized embeddings and (b) can be used for transfer learning.
    - Radford, Alec, et al. "Improving language understanding by generative pre-training." pdf (2018).
    - Peters, Matthew E., et al. "Deep contextualized word representations." arXiv preprint arXiv:1802.05365 (2018).
    - Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." arXiv preprint arXiv:1810.04805 (2018).
    - Radford, Alec, et al. "Language models are unsupervised multitask learners." OpenAI Blog 1.8 (2019).
    - Yang, Zhilin, et al. "XLNet: Generalized Autoregressive Pretraining for Language Understanding." arXiv preprint arXiv:1906.08237 (2019).
    - Liu, Yinhan, et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." arXiv preprint arXiv:1907.11692 (2019).
    - Lan, Zhenzhong, et al. "Albert: A lite bert for self-supervised learning of language representations." arXiv preprint arXiv:1909.11942 (2019).
    - Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." arXiv preprint arXiv:1910.10683 (2019).
    - Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).


### 5. Make use of the overwhelming offer of blogs, tutorials (or the internet in general): Here are some nice online resources

- [ ] https://github.com/ivan-bilan/The-NLP-Pandect (Exhaustive overview on basically everything)
- [ ] https://mccormickml.com/ (+ https://www.youtube.com/channel/UCoRX98PLOsaN8PtekB9kWrw)
- [ ] https://jalammar.github.io/ (+ https://www.youtube.com/channel/UCmOwsoHty5PrmE-3QhUBfPQ) 
- [ ] https://ruder.io/nlp-news/ (+ his thesis: https://ruder.io/thesis/neural_transfer_learning_for_nlp.pdf)
- [ ] https://dair.ai/ (pretty nice blog)
- [ ] https://github.com/tomohideshibata/BERT-related-papers (Exhaustive list of BERT related papers)
- [ ] Booklet from our NLP seminar (summer semester 2020): https://compstat-lmu.github.io/seminar_nlp_ss20/ 
- [ ] Summaries by Yannik Kilcher: https://www.youtube.com/channel/UCZHmQk67mSJgfCCTn7xBfew 
- [ ] Course about ABSA from a student consulting project: https://lisa-wm.github.io/nlp-twitter-r-bert/ 


### 6. Software

- [ ] https://github.com/huggingface/transformers
- [ ] https://github.com/ThilinaRajapakse/simpletransformers (+ Doku: https://simpletransformers.ai/)
- [ ] https://radimrehurek.com/gensim/
