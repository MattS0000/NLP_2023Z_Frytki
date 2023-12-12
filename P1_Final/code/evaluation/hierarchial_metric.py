from tqdm import tqdm
from comparison.product_comparator import ProductComparator
from preprocessing.dataset_loader import PletsDataset
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import mean_squared_error


class HierachialMetric:
    def __init__(self) -> None:
        self.alpha = 0.5

    def evaluate_dataset(self, model: ProductComparator, test_data: PletsDataset, limit: int = 150) -> None:
        count, score = 0, 0
        for batch in tqdm(test_data, total=limit):
            batch = np.array(batch).transpose()
            for instance in batch:
                similarities = [model.similarity(instance[0], instance[i+1]) for i in range(len(instance)-1)]
                count += 1
                score += self.evaluate_similarities(similarities)
                if count >= limit:
                    break
            if count >= limit:
                break
        return score/count

    def evaluate_similarities(self, similarities: np.array) -> float:
        target = np.linspace(1, 0, len(similarities))
        target_order = np.linspace(1, len(similarities), len(similarities))
        similarity_order = len(similarities)-np.argsort(similarities).argsort()
        return (kendalltau(target_order, similarity_order).statistic+1)/2*self.alpha + (1-mean_squared_error(similarities, target))*(1-self.alpha)