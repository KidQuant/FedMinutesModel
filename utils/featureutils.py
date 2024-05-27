from re import L

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity

from utils.corpusutils import Corpus


class FeatureProcessor(object):
    """
    FeatureProcessor takes in a Corpus instance
    and extracts the following features:
    - Sentiment
    - Topic distributions
    - Document embeddings

    Methods
    ----------
    get_features : Update Document Features attribute with
                    sentiment, embedding or topic info
    """

    sent_dict = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(
        self,
        corpus,
        transformer_model,
        transformer_tokenizer,
        tokenizer_settings,
        lda_model,
        lda_vec,
        lda_topic_dict,
        batch_size=20,
    ):
        """
        :param corpus: Corpus instance -> Corpus([Documents(["text"])])
        :param transformer_model: transformer model e.g. FinBERT
        :param transformer_tokenizer: transformer tokenizer
        :param lda_model: Latent Dirichlet Allocation
        :param lda_vec: CountVectorizer used to train lda_model
        :param lda_topic_dict: final topic dictionary
        :param batch_size: size of Documents to generate features from

        :returns: Corpus instance with features
        """

        if corpus.__class__.__name__ != Corpus.__name__:
            raise ValueError("corpus variable must be an Corpus instance")

        self.corpus = corpus
        self.transformer_model = transformer_model
        self.transformer_tokenizer = transformer_tokenizer
        self.tokenizer_settings = tokenizer_settings
        self.lda_model = lda_model
        self.lda_vec = lda_vec
        self.lda_topic_dict = lda_topic_dict
        self.batch_size = batch_size

    def _get_topics(self, batch):
        """Takes in a batch of stemeed tokens
        returns topic distrubution and argmax"""

        bag_of_words = self.lda_vec.transform(batch)
        topic_dist = self.lda_model.transform(bag_of_words)

        topics_pred = list(map(self.lda_topic_dict.get, topic_dist.argmax(axis=1)))
        return topic_dist.tolist(), topics_pred

    def _get_sentiment(self, model_logits):
        """
        Takes in logits and returns
        sentiment prediction
        """

        logits = self.softmax(np.array(model_logits))
        if logits.shape[0] == 1:
            sent_pred = np.argmax(logits, axis=1)
        else:
            sent_pred = np.squeeze(np.argmax(logits, axis=1))

        sent_pred = list(map(self.__class__.sent_dict.get, sent_pred))

    def _get_model_output(self, sentiment, embedding, topic):
        """
        Loops through each batch
        of tokens and updates
        features attribute of each
        Document instance
        """

        for i in range(0, len(self.corpus), self.batch_size):
            batch = self.corpus[i : i + self.batch_size]

            batch_size = len(batch)
            logits = [None] * batch_size
            sent_pred = [None] * batch_size
            doc_embedding = [None] * batch_size
            topic_dist = [None] * batch_size
            topic_pred = [None] * batch_size

            feature_tensor = self.transformer_tokenizer(
                [b.tokens for b in batch], **self.transformer_settings
            )

            with torch.no_grad():
                output = self.transformer_model(**feature_tensor)

                if sentiment:
                    model_logits = output[0]
                    logits, sent_pred = self._get_sentiment(model_logits)

                if embedding:
                    embedding_from_last = output.hidden_states[-1]
                    attention_mask = feature_tensor["attention_mask"]
                    doc_embedding = self._embedding_mean_pool(
                        embedding_from_last, attention_mask
                    )

                if topic:
                    topic_dist, topic_pred = self._get_topics(batch=batch)

                yield (
                    batch,
                    (logits, sent_pred),
                    (doc_embedding),
                    (topic_dist, topic_pred),
                )

    def get_features(self, sentiment=True, embedding=True, topic=True):
        """
        Extract sentiment, embedding or topic features
        from corpus

        Parameters
        -----------
        sentiment: boolean
            updates Documents attribute features with
            sentiment logits and prediction

        embedding: boolean
            updates
        """

        if not (sentiment or embedding or topic):
            raise ValueError("At least one feature needs to be True")

        corpus = []
        model_output = self._get_model_output(sentiment, embedding, topic)
        for batch, sent_batch, embed_batch, topic_batch in model_output:

            logits, sent_pred = sent_batch
            doc_embedding = embed_batch
            topic_dist, topic_pred = topic_batch

            for n, doc in enumerate(batch):
                batch_n = batch[n]
                logit_n = logits[n]
                sent_pred_n = sent_pred[n]
                doc_embed_n = doc_embedding[n]
                topic_dist_n = topic_dist[n]
                topic_pred_n = topic_pred[n]

                feature_dict = {
                    "sentiment": {"logits": logit_n, "predictions": sent_pred_n},
                    "embedding": doc_embed_n,
                    "topics": {"topic_dist": topic_dist_n, "topic_pred": topic_pred_n},
                }

                batch_n.features = feature_dict
                corpus.append(batch_n)

        return Corpus(corpus)

    @staticmethod
    def _embedding_mean_pool(embedding_from_last, attention_mask):
        """
        Extract vector representation of token
        """

        mask = attention_mask.unsqueezed(-1).expand(embedding_from_last.shape).float()
        mask_embeddings = embedding_from_last * mask
        mean_embeddings = torch.sum(mask_embeddings, 1) / torch.clamp(
            mask.sum(1), min=1e-8
        )

        if mean_embeddings.shape[0] == 1:
            return mean_embeddings.tolist()
        else:
            return mean_embeddings.squeeze().tolist()

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1)[:, None])
        return e_x / np.sum(e_x, axis=1)[:, None]

    def find_corpus_idx(corpus):
        """
        Extract file_ids positional index in corpus
        """

        ids = [(n, f.category_id, f.file_id) for n, f in enumerate(corpus)]
        df_ids = pd.DataFrame(ids, columns=["idx", "category_id", "file_id"])
        start_idx = df_ids.drop_duplicates(["category_id", "file_id"], keep="first")
        end_idx = df_ids.drop_duplicates(["category_id", "file_id"], keep="last")

        idx = start_idx.merge(
            end_idx, on=["category_id", "file_id"], suffixes=["_start", "end"]
        )
        return idx

    def find_closest(base_corpus, compare_corpus, by_file_is=True):
        """
        Return list of embedding consine similiarity
        and jensenshannon distance for each Document
        instance in base_corpus, compared with
        compared_corpus

        parameters
        -----------
        base_corpus: Corpus instance
            out of sample corpus
            e.g. A new paragraph

        compare_corpus: Corpus instance
            Corpus to be compared with

        by_file_id: boolean
            returnm results by file_id

        returns:
            list of DataFrames with scores
        """

        idx = find_corpus_idx(compare_corpus)
        stats = []

        compare_embed = np.asarray
