from sentence_transformers import SentenceTransformer
import logging
logger = logging.getLogger(__name__)

def generate_embedding(prompts, model):
    sentence_embeddings = model.encode(prompts)
    return list(sentence_embeddings.tolist())


def get_embedding_model(embedding_model_name, cache_folder=None):
    logger.debug(f"Getting embedding model {embedding_model_name} to cache_folder: {cache_folder}")
    model = SentenceTransformer(embedding_model_name, cache_folder=cache_folder)
    return model
