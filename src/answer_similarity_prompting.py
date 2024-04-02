import numpy as np

from langchain_core.embeddings import Embeddings


def compute_answer_similarity(predicted_answer: str,
                              ideal_answer: str,
                              encoder: Embeddings,
                              logger) -> float:
    
    # NOTE: RAGAS docs indicate that the comparison is done using
    # a CrossEncoder but this is currently not implemented. The
    # implementation here mimics what is currently implemented in RAGAS.
    logger.debug(f"predicted answer: {predicted_answer}")
    logger.debug(f"ideal answer: {ideal_answer}")
    answers = [predicted_answer, ideal_answer]
    embeddings = np.array(encoder.embed_documents(answers))
    source, target = embeddings[0, :], embeddings[1, :]
    similarity = np.dot(source, target) / (
        np.linalg.norm(source) * np.linalg.norm(target))
    return similarity
