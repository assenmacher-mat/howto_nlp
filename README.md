# Where to start in the NLP jungle?

__Disclaimer I:__ _This document is intended for students either writing their (nlp-related) bachelor or master thesis or working on their (nlp-related) student consulting project under our supervision._

__Disclaimer II:__ _Please note that this document is subject to continuous change. Every time we find a new, nice source we will add it. Some stuff might also be deleted over time if not considered useful anymore._

_Authors:_ [Matthias Aßenmacher](https://www.misoda.statistik.uni-muenchen.de/personen/mitarbeiter/assenmacher/index.html) // [Christian Heumann](https://www.misoda.statistik.uni-muenchen.de/personen/professoren/heumann/index.html)

_Note:_ Most important resources are marked by a :warning:

_Last change:_ 03-03-2022

## I. Set up all necessary communication channels

- [ ] Join Mattermost (ask Matthias for the invite link)
- [ ] Ask Matthias to add you to the “NLP”-channel on Mattermost
- [ ] Ask Matthias to add you to our internal “NLP” mailing list
- [ ] You can reach Matthias via Mattermost or E-Mail, Christian prefers to be contacted via E-Mail  
     (In case of e-mails related to your thesis/project, make sure to cc the respective other in order to create no information asymmetries)
- [ ] We have made very good experience with (approx.) bi-weekly meetings for short status updates and prefer to work together with you in this fashion  
      (Nevertheless this is not mandadory; we just think it helps you to (i) get started and (ii) stay on track)
- [ ] We will have a so-called “NLP Colloquium” every now and then (intended 4 times a year) where all of our BA-/MA-/Consulting- students present their work to the others. This meeting is of rather informal character (mostly intended to connect you to each other), so no need for high-gloss slides, or anything like that. Everything like jupyter notebooks/interesting figures/slides is fine.  
     We will announce this via the mailing list and via Mattermost.  
     Dates for 2021: 
     - [ ] 01.04. at 13h s.t.
     - [ ] 01.07. at 13h s.t.
     - [ ] 14.10. at 13h s.t.
     - [ ] 16.12. at 13h s.t.
- [ ] The mailing list will be mostly used for announcements, while in the Mattermost channel we will occasionally also post (nlp-related) stuff we consider interesting.
- [ ] TALK TO US rather sooner than later if any problems occur which you are not able to solve on your own. Open (and timely) communication is (in our opinion) key to a successful supervision/cooperation during theses or consulting projects.

## II. Useful materials for starting with NLP

### 1. Get familiar with the basics:

- [ ] Pre-Processing (e.g. in Python with [NLTK](https://www.nltk.org/) or [spaCy](https://spacy.io/))
- [ ] One-hot-encoding of words, the bag-of-words (bow) approach, its applications in ML, drawbacks & limitations (just google this stuff, you will find enough material).
- [ ] Extensions of the bow approach, like n-grams or tf-idf (also just google this).

### 2. Get familiar with the Python environment:

- [ ] In general (it's a little different from the just "plug-and-play" style in which you can install R and R-Studio)
- [ ] Find a comfortable setup:  
     - [ ] One alternative could be using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for Python, package management and virtualenvs together with e.g. [VS Code](https://code.visualstudio.com/) as IDE  
     - [ ] Another alternative (nice for beginners) is [Anaconda](https://www.anaconda.com/), a all-in-one solution that comes with various IDEs (like e.g. Spyder, which very closely resembles R-Studio)
- [ ] [Jupyter Notebooks / Lab](https://jupyter.org/)
- [ ] [Google Colaboratory](https://research.google.com/colaboratory/)

### 3. Start looking into neural networks and deep learning for NLP

- [ ] Pretty nice book for an a broad overview on everything until self-attention, useful for covering the basics: [Goldberg (2017)](https://github.com/Michael2Tang/ML_Doc/blob/master/Neural%20Network%20Methods%20in%20Natural%20Language%20Processing-Morgan%20%26%20Claypool%20Publishers%20(2017)%20-%20Yoav%20Goldberg%2C%20Graeme%20Hirst.pdf)
- [ ] Good overview on Embeddings: [Pilehvar & Camacho-Collados (2021)](https://sites.google.com/view/embeddings-in-nlp)
- [ ] Overview on DL in general: [Deep Learning book](https://www.deeplearningbook.org/)
- [ ] Or more basic: [Intro to Statistical learning](https://www.statlearning.com/)
- [ ] :warning: [Hugging Face Transformer course](https://huggingface.co/course/chapter0/1?fw=pt) :warning:
- [ ] Internal teaching resources (LMU):
     - [ ] Our lecture from WS 20/21: https://moodle.lmu.de/course/view.php?id=10268 
     - [ ] Same lecture, slightly updated (WS 21/22): https://moodle.lmu.de/course/view.php?id=17645
     - [ ] Booklet from our NLP seminar (summer semester 2020): https://slds-lmu.github.io/seminar_nlp_ss20/ 
     - [ ] (Exemples of) Supervised theses: https://www.misoda.statistik.uni-muenchen.de/studium_lehre/theses_old/index.html
     - [ ] Course about ABSA from a student consulting project: https://lisa-wm.github.io/nlp-twitter-r-bert/ 

### 4. Milestone papers:

- __Important conceptual foundation__ 
     - [ ] Bengio, Y., et al. (2003) ["A neural probabilistic language model."](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) Journal of machine learning research: 1137-1155.
- __Modification of the idea of Bengio et al.; use the internal representations of the neural net as the primary objective; Their architecture is called word2vec and is able to learn static word embeddings__
     - [ ] Mikolov, Tomas, et al. ["Efficient estimation of word representations in vector space."](https://arxiv.org/abs/1301.3781) arXiv preprint arXiv:1301.3781(2013).  
     - [ ] Mikolov, Tomas, et al. ["Distributed representations of words and phrases and their compositionality."](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) Advances in neural information processing systems. 2013.  
- __Alternative framework to word2vec__
     - [ ] Pennington, Jeffrey, Richard Socher, and Christopher Manning. ["Glove: Global vectors for word representation."](https://aclanthology.org/D14-1162.pdf) Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.  
- __Extending the embedding idea from word2vec to sentence/paragraph/document level__
     - [ ] Le, Quoc, and Tomas Mikolov. ["Distributed representations of sentences and documents."](http://proceedings.mlr.press/v32/le14.pdf) International conference on machine learning. 2014.  
- __Sequence-to-sequence models__
     - [ ] Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. ["Sequence to sequence learning with neural networks."](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf) Advances in neural information processing systems. 2014.
- __Extending the embedding idea to subword tokens__
     - [ ] Bojanowski, Piotr, et al. ["Enriching word vectors with subword information."](https://arxiv.org/pdf/1607.04606.pdf) Transactions of the Association for Computational Linguistics 5 (2017): 135-146.  
     - [ ] Joulin, Armand, et al. ["Bag of tricks for efficient text classification."](https://arxiv.org/pdf/1607.01759.pdf?fbclid=IwAR1wttEXho2gqk3BasKDuncgftN5I5lmH2TbIgvGuHxfutM3IavbateHH9A) arXiv preprint arXiv:1607.01759 (2016).  
- __Important foundations for the so-called Attention & Self-Attention mechanism__
     - [ ] Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. ["Neural machine translation by jointly learning to align and translate."](https://arxiv.org/pdf/1409.0473.pdf) arXiv preprint arXiv:1409.0473 (2014).
     - [ ] Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. ["Effective approaches to attention-based neural machine translation."](https://arxiv.org/pdf/1508.04025)) arXiv preprint arXiv:1508.04025 (2015).  
     - [ ] Vaswani, Ashish, et al. ["Attention is all you need."](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) Advances in neural information processing systems. 2017.  
- __Some of the most famous models that learn (a) contextualized embeddings and (b) can be used for transfer learning__
     - [ ] Radford, Alec, et al. ["Improving language understanding by generative pre-training."](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) pdf (2018).
     - [ ] Peters, Matthew E., et al. "Deep contextualized word representations." arXiv preprint arXiv:1802.05365 (2018).
     - [ ] Devlin, Jacob, et al. ["Bert: Pre-training of deep bidirectional transformers for language understanding."](https://aclanthology.org/N19-1423.pdf) Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers).
     - [ ] Radford, Alec, et al. ["Language models are unsupervised multitask learners."](http://www.persagen.com/files/misc/radford2019language.pdf) OpenAI Blog 1.8 (2019).
     - [ ] Yang, Zhilin, et al. ["XLNet: Generalized Autoregressive Pretraining for Language Understanding."](https://proceedings.neurips.cc/paper/2019/file/dc6a7e655d7e5840e66733e9ee67cc69-Paper.pdf) Advances in neural information processing systems 32 (2019).
     - [ ] Liu, Yinhan, et al. ["RoBERTa: A Robustly Optimized BERT Pretraining Approach."](https://arxiv.org/pdf/1907.11692.pdf) arXiv preprint arXiv:1907.11692 (2019).
     - [ ] Lan, Zhenzhong, et al. ["Albert: A lite bert for self-supervised learning of language representations."](https://arxiv.org/pdf/1909.11942.pdf) arXiv preprint arXiv:1909.11942 (2019).
     - [ ] Raffel, Colin, et al. ["Exploring the limits of transfer learning with a unified text-to-text transformer."](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf) arXiv preprint arXiv:1910.10683 (2019).
     - [ ] Sanh, Victor, et al. ["DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter."](https://arxiv.org/pdf/1910.01108.pdf?ref=hackernoon.com) arXiv preprint arXiv:1910.01108 (2019).
     - [ ] Clark, Kevin, et al. ["Electra: Pre-training text encoders as discriminators rather than generators."](https://arxiv.org/pdf/2003.10555.pdf%3C/p%3E) arXiv preprint arXiv:2003.10555 (2020).
     - [ ] Brown, Tom B., et al. ["Language models are few-shot learners."](https://arxiv.org/pdf/2005.14165.pdf) arXiv preprint arXiv:2005.14165 (2020).
- __Heavily used benchmark data sets__ 
     - [ ] Wang, Alex, et al. ["GLUE: A multi-task benchmark and analysis platform for natural language understanding."](https://arxiv.org/pdf/1804.07461.pdf) arXiv preprint arXiv:1804.07461 (2018).
     - [ ] Wang, Alex, et al. ["Superglue: A stickier benchmark for general-purpose language understanding systems."](https://arxiv.org/pdf/1905.00537.pdf) arXiv preprint arXiv:1905.00537 (2019).
     - [ ] Pranav Rajpurkar, et al. ["SQuAD: 100,000+ questions for machine comprehension of text"](https://aclanthology.org/D16-1264.pdf) Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.
     - [ ] Pranav Rajpurkar, et al. ["Know what you don’t know: Unanswerable questions for SQuAD"](https://aclanthology.org/P18-2124.pdf) In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 784–789, Melbourne, Australia. 
- __Zero-/Few-Shot Learning__ _TO DO_
     - [ ] Brown, Tom B., et al. ["Language models are few-shot learners."](https://arxiv.org/pdf/2005.14165.pdf) arXiv preprint arXiv:2005.14165 (2020).
     - [ ] Schick, Timo, and Hinrich Schütze. ["It's not just size that matters: Small language models are also few-shot learners."](https://arxiv.org/pdf/2009.07118.pdf) arXiv preprint arXiv:2009.07118 (2020).
- __Prompting/Prompt-Engineering__ 
     - [ ] Liu, Pengfei, et al. ["Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing."](http://pretrain.nlpedia.ai/) arXiv preprint arXiv:2107.13586 (2021).
     - [ ] Wei, Jason, et al. ["Finetuned language models are zero-shot learners."](https://arxiv.org/pdf/2109.01652.pdf) arXiv preprint arXiv:2109.01652 (2021).
     - [ ] Lester, Brian, et al. ["The Power of Scale for Parameter-Efficient Prompt Tuning"](https://aclanthology.org/2021.emnlp-main.243.pdf) In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

### 5. Make use of the overwhelming offer of blogs, tutorials (or the internet in general): Here are some nice online resources

- [ ] https://github.com/ivan-bilan/The-NLP-Pandect (Exhaustive overview on basically everything)
- [ ] https://mccormickml.com/ (+ https://www.youtube.com/channel/UCoRX98PLOsaN8PtekB9kWrw)
- [ ] https://jalammar.github.io/ (+ https://www.youtube.com/channel/UCmOwsoHty5PrmE-3QhUBfPQ) 
- [ ] https://ruder.io/nlp-news/ (+ his thesis: https://ruder.io/thesis/neural_transfer_learning_for_nlp.pdf)
- [ ] https://dair.ai/ (pretty nice blog)
- [ ] https://github.com/tomohideshibata/BERT-related-papers (Exhaustive list of BERT related papers)
- [ ] Summaries by Yannik Kilcher: https://www.youtube.com/channel/UCZHmQk67mSJgfCCTn7xBfew 


### 6. Software

- [ ] https://github.com/huggingface/transformers
- [ ] https://github.com/ThilinaRajapakse/simpletransformers (+ Doku: https://simpletransformers.ai/)
- [ ] https://radimrehurek.com/gensim/
- [ ] https://www.nltk.org/
- [ ] https://spacy.io/
