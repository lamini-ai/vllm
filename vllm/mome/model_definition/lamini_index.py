import json
import logging
import os
from typing import Iterator, TypeVar, Union

import faiss
import numpy as np

from tqdm import tqdm

from vllm.mome.model_definition.embedding import (
    generate_embedding,
    get_embedding_model,
)
from vllm.mome.model_definition.constants import SENTENCE_TRANSFORMER_DIM


logger = logging.getLogger(__name__)


class LaminiIndex:
    def __init__(
        self,
        cache_dir,
        dataset=None,
        clamp_max_embedding_dimension=SENTENCE_TRANSFORMER_DIM,
    ):
        self.dataset = dataset

        self.embedding_model = get_embedding_model(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir
        )
        self.embedding_dimension = min(
            self.embedding_model.get_sentence_embedding_dimension(),
            clamp_max_embedding_dimension,
        )

    def destroy_embedding_model(self):
        del self.embedding_model

    @staticmethod
    def load_index(path, values_path, cache_dir):
        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")
        config_path = os.path.join(path, "index_config.json")

        lamini_index = LaminiIndex(cache_dir)

        with open(config_path, "r") as f:
            config = json.load(f)
            lamini_index.embedding_dimension = config["embedding_dimension"]

        # Load the index from a file
        lamini_index.index = faiss.read_index(faiss_path)

        # Load the splits from a file
        with open(splits_path, "r") as f:
            lamini_index.splits = json.load(f)

        # Load the key embeddings from a file
        if os.path.exists(os.path.join(path, "keys.json")):
            with open(os.path.join(path, "keys.json"), "r") as f:
                lamini_index.keys = json.load(f)
        elif os.path.exists(os.path.join(path, "keys.npy")):
            lamini_index.keys = np.load(os.path.join(path, "keys.npy"))
        else:
            raise ValueError("Keys file not found")

        # Load the value embeddings from a file
        if os.path.exists(os.path.join(values_path, "values.json")):
            with open(os.path.join(values_path, "values.json"), "r") as f:
                lamini_index.values = json.load(f)
        elif os.path.exists(os.path.join(values_path, "values.npy")):
            lamini_index.values = np.load(os.path.join(values_path, "values.npy"))
        else:
            raise ValueError("Values file not found")

        return lamini_index

    def build_index(
        self,
        args,
    ):
        self.splits = []
        self.index = None
        self.keys = []
        self.values = []
        limit = args["index_max_size"] // args["index_ivf_nlist"]
        # load a batch of splits from a generator
        group_batches = self.group_batches(args["index_ivf_nlist"])
        pbar = tqdm(desc="Generating index", unit="batches", total=limit)
        for batch in group_batches:
            if limit == 0:
                break
            embeddings = self.get_embeddings(batch)

            if self.index is None:
                # initialize the index
                logger.info(
                    f"Creating index with d={len(embeddings[0])}, method={args['index_method']}, embedding_length={len(embeddings)}"
                )
                if args["index_method"] == "IndexIVFPQ":

                    clamped_ivf_nlist = min(
                        args["index_ivf_nlist"], get_power_of_two(len(embeddings))
                    )

                    clamped_pq_nbits = min(
                        args["index_pq_nbits"], get_max_bits(len(embeddings))
                    )

                    clamped_m = min(args["index_pq_m"], len(embeddings[0]))

                    logger.info(
                        f"Creating IVFPQ index with nlist={clamped_ivf_nlist}, m={args['index_pq_m']}, nbits={args['index_pq_nbits']}"
                    )
                    self.index = faiss.IndexFlatL2(len(embeddings[0]))
                    self.index = faiss.IndexIVFPQ(
                        self.index,
                        len(embeddings[0]),
                        clamped_ivf_nlist,
                        clamped_m,
                        clamped_pq_nbits,
                    )
                    self.index.train(np.array(embeddings))
                elif args["index_method"] == "IndexHNSWPQ":
                    self.index = faiss.IndexHNSWPQ(
                        len(embeddings[0]),
                        args["index_pq_m"],
                        args["index_hnsw_m"],
                    )
                    self.index.hnsw.efConstruction = args["index_hnsw_efConstruction"]
                    self.index.hnsw.efSearch = args["index_hnsw_efSearch"]
                    self.index.train(np.array(embeddings))
                elif args["index_method"] == "IndexHNSWFlat":
                    self.index = faiss.IndexHNSWFlat(
                        len(embeddings[0]), args["index_hnsw_m"]
                    )
                    self.index.hnsw.efConstruction = args["index_hnsw_efConstruction"]
                    self.index.hnsw.efSearch = args["index_hnsw_efSearch"]
                elif args["index_method"] == "IndexPQ":
                    self.index = faiss.IndexPQ(
                        len(embeddings[0]),
                        args["index_pq_m"],
                        args["index_pq_nbits"],
                    )
                    self.index.train(np.array(embeddings))
                else:
                    self.index = faiss.IndexFlatL2(len(embeddings[0]))

                if self.embedding_dimension > len(embeddings[0]):
                    self.embedding_dimension = len(embeddings[0])

            # add the embeddings to the index
            self.index.add(np.array(embeddings))

            # save the splits
            self.splits.extend(batch)

            # save the key embeddings
            self.keys.extend(embeddings)

            # save the value embeddings
            self.values.extend(embeddings)
            pbar.update()
            limit -= 1

    def group_batches(self, batch_size):
        iterator = iter(self.dataset)
        for batch in next_n(iterator, batch_size):
            yield batch

    def get_embeddings(self, examples):
        embedding_list = generate_embedding(examples, self.embedding_model)
        embedding_list = self.adjust_embedding_dimension(embedding_list)
        return embedding_list

    def adjust_embedding_dimension(self, embeddings):
        start_dim = len(embeddings[0])
        if start_dim > self.embedding_dimension:
            return [embedding[: self.embedding_dimension] for embedding in embeddings]
        elif start_dim == self.embedding_dimension:
            return embeddings
        else:
            raise ValueError("Embedding dimension is less than the clamp value")

    def get_key_and_value(self, query_embeddings, k):
        # get the k nearest neighbors
        distances, indices = self.index.search(query_embeddings, k)

        returned_keys = []
        returned_values = []
        returned_indices = []
        for i in indices:
            for j in i:
                assert j < len(
                    self.keys
                ), f"Index {j} not in keys, indices: {indices}, query_embeddings: {query_embeddings}"
                assert j < len(
                    self.values
                ), f"Index {j} not in values, indices: {indices}"
                if j == -1:
                    returned_keys.extend(
                        [np.zeros_like(self.keys[j]) for _ in range(len(indices))]
                    )
                    returned_values.extend(
                        [np.zeros_like(self.values[j]) for _ in range(len(indices))]
                    )
                    returned_indices.append(j)
                else:
                    returned_keys.extend([self.keys[j] for _ in range(len(indices))])
                    returned_values.extend(
                        [self.values[j] for _ in range(len(indices))]
                    )
                    returned_indices.append(j)
            break
        return (
            returned_keys,
            returned_values,
            returned_indices,
        )

    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]

        embedding_array = np.array([embedding])

        # get the k nearest neighbors
        distances, indices = self.index.search(embedding_array, k)

        return [self.splits[i] for i in indices[0]]

    def mmr_query(self, query, k=20, n=5):
        embedding = self.get_embeddings([query])[0]

        embedding_array = np.array([embedding])

        # get the k nearest neighbors
        distances, indices = self.index.search(embedding_array, k)

        # get the n most diverse results
        most_diverse = self.most_diverse_results(embedding, indices[0], n)

        return most_diverse

    def most_diverse_results(self, query_embedding, indices, n):
        # get the embeddings for the indices
        split_batch = [self.splits[i] for i in indices]

        embeddings = self.get_embeddings(split_batch)

        # calculate the similarity between the query and the results
        similarities = [np.dot(query_embedding, embedding) for embedding in embeddings]

        # initialize the results
        results = [indices[0]]

        # iterate through the results
        for i in range(1, n):
            # initialize the best result
            best_result = None
            best_result_similarity = 0

            # iterate through the remaining results
            for j in range(len(indices)):
                # skip the result if it is already in the results
                if indices[j] in results:
                    continue

                # calculate the similarity between the result and the other results
                similarity = np.mean(
                    [np.dot(embeddings[j], embeddings[k]) for k in range(len(results))]
                )

                # update the best result
                if similarity > best_result_similarity:
                    best_result = indices[j]
                    best_result_similarity = similarity

            # add the best result to the results
            results.append(best_result)

        return [self.splits[i] for i in results]

    def update(self, index, value):
        self.values[index] = value.tolist()

    def save_index(self, path):
        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")
        keys_path = os.path.join(path, "keys.npy")
        config_path = os.path.join(path, "index_config.json")

        logger.debug("Saving index to %s", faiss_path)
        logger.debug("Saving splits to %s", splits_path)
        logger.debug("Saving key embeddings to %s", keys_path)

        logger.debug("Index size: %d", self.index.ntotal)

        # Save the index to a file
        faiss.write_index(self.index, faiss_path)

        # Save the splits to a file
        with open(splits_path, "w") as f:
            json.dump(self.splits, f)

        # Save the keys to a file
        np.save(keys_path, self.keys)

        # Save the config to a file
        with open(config_path, "w") as f:
            json.dump({"embedding_dimension": self.embedding_dimension}, f)

    def save_index_values(self, path):
        values_path = os.path.join(path, "values.npy")

        logger.debug("Saving only value embeddings to %s", values_path)

        # Save the values to a file
        np.save(values_path, self.values)


def get_power_of_two(n):
    # Get the highest power of 2 that is less than n
    return 2 ** int(np.log2(n))


def get_max_bits(n):
    # Get the number of bits needed to represent n
    return max(int(np.log2(n)), 1)


def next_n(iterator: Union[Iterator], n: int):
    for x in chunks(iterator, n):
        yield x


T = TypeVar("T")


def chunks(
    iterator: Iterator[T],
    size: int,
) -> Iterator[list[T]]:
    """Yield successive n-sized chunks from lst."""
    finished = False

    while not finished:
        results: list[T] = []

        for _ in range(size):
            try:
                result = None
                while result is None:
                    result = next(iterator)
            except StopIteration:
                finished = True
            else:
                results.append(result)

        if results:
            yield results
