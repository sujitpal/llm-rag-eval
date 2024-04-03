import nltk
import numpy as np

from langchain_core.embeddings import Embeddings


def compute_answer_similarity(predicted_answer: str,
                              ideal_answer: str,
                              encoder: Embeddings,
                              logger,
                              cross_encoder: bool = True) -> float:

    logger.debug(f"predicted answer: {predicted_answer}")
    logger.debug(f"ideal answer: {ideal_answer}")

    if cross_encoder:
        # use cross encoder similarity scoring (token level)
        pa_words = [w for w in nltk.word_tokenize(predicted_answer)]
        ia_words = [w for w in nltk.word_tokenize(ideal_answer)]
        words = pa_words + ia_words
        embs = encoder.embed_documents(words)
        pa_vecs = np.array(embs[0:len(pa_words)])
        ia_vecs = np.array(embs[len(pa_words):])
        similarity = (np.sum(
            np.max(
                np.dot(pa_vecs, ia_vecs.T),
                axis=1
            )
        ) / len(pa_words))
    else:
        # use cosine similarity
        # NOTE: RAGAS docs mention using cross-encoder similarity but
        # the code uses cosine similarity
        answers = [predicted_answer, ideal_answer]
        embeddings = np.array(encoder.embed_documents(answers))
        source, target = embeddings[0, :], embeddings[1, :]
        similarity = np.dot(source, target) / (
            np.linalg.norm(source) * np.linalg.norm(target))

    return similarity
