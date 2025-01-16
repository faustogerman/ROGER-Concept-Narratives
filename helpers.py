import pickle
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, DetrForObjectDetection


def extract_image_embeddings(images, names, filename):
    filename += "-detr"

    try:
        with open(f"{filename}.pickle", 'rb') as handle:
            obj = pickle.load(handle)
            image_embeddings = obj["image_embeddings"]
            image_names = obj["names"]
            del obj

            print(f"File '{filename}.pickle' loaded successfully.")
    except FileNotFoundError:
        image_names = names

        print(
            f"Could not find file '{filename}.pickle'. "
            "Regenerating the embeddings."
        )
        
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm"
        )

        processor = AutoImageProcessor.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm"
        )

        batch_size = 8
        image_embeddings = []

        for start in tqdm(range(0, len(images), batch_size)):
            ims = list(images[start:start + batch_size])

            # Generate the embeddings
            inputs = processor(ims, return_tensors="pt")
            with torch.no_grad():
                embeds = model(**inputs).last_hidden_state

            # Mean Pooling across the sequence dimension
            mean_pooled = embeds.mean(dim=1)  # Shape: (batch_size, hidden_size)
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)  # Shape: (batch_size, hidden_size)
            image_embeddings.append(normalized.cpu())

            # The input and outputs of the models take a lot of memory,
            # so we delete them here since we are not using them again
            del inputs, embeds

        image_embeddings = torch.cat(image_embeddings)

        with open(f"{filename}.pickle", 'wb') as handle:
            pickle.dump(
                obj={"image_embeddings": image_embeddings, "names": names},
                file=handle,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    return image_embeddings, image_names


def embedding_based_dtw(sequence_a, sequence_b):
    """
    Compute the DTW distance between two sequences of embeddings.

    Parameters:
    - sequence_a: array-like of shape (N, D)
    - sequence_b: array-like of shape (M, D)
    - dist_metric: string specifying the distance metric

    Returns:
    - dtw_distance: float, the DTW distance between the two sequences
    - dtw_matrix: array, the accumulated cost matrix
    - path: list of tuples, the optimal warping path
    """
    N = len(sequence_a)
    M = len(sequence_b)

    # Compute the cost matrix
    cost_matrix = distance.cdist(sequence_a, sequence_b, metric='euclidean')

    # Initialize the accumulated cost matrix with infinities
    dtw_matrix = np.full((N + 1, M + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Compute the accumulated cost matrix
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = cost_matrix[i - 1, j - 1]

            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],     # insertion
                dtw_matrix[i, j - 1],     # deletion
                dtw_matrix[i - 1, j - 1]  # match
            )

    # Backtrace to find the optimal path
    i, j = N, M
    path = []
    while (i > 0) or (j > 0):
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            argmin = np.argmin([
                dtw_matrix[i - 1, j - 1],  # match
                dtw_matrix[i - 1, j],     # insertion
                dtw_matrix[i, j - 1]      # deletion
            ])
            if argmin == 0:
                i -= 1
                j -= 1
            elif argmin == 1:
                i -= 1
            else:
                j -= 1
    path.reverse()

    dtw_distance = dtw_matrix[N, M]
    return dtw_distance / len(path), dtw_matrix[1:, 1:], path


def dtw_distance_metric(embeds, true_storyline, extracted_storyline):
    true_emb = [embeds[i] for i in true_storyline]
    extracted_emb = [embeds[i] for i in extracted_storyline]

    dist, _, path = embedding_based_dtw(true_emb, extracted_emb)
    return dist, path


def avg_cosine_similarity_metric(embeds, true_storyline, extracted_storyline, path):
    true_emb = [embeds[i] for i in true_storyline]
    extracted_emb = [embeds[i] for i in extracted_storyline]

    # Compute similarities between documents, except for the start and end documents
    similarities = [
        np.clip(true_emb[a] @ extracted_emb[b] / (norm(true_emb[a]) * norm(extracted_emb[b])), -1, 1)
        for a, b in path[1:-1]
    ]

    # Compute the geometric mean of the similarities
    return np.array(similarities).mean()
