from __future__ import annotations
from typing import List, Tuple, Dict, Callable
from collections import Counter
from .datastructures import UserGraph, cosine_similarity, top_k

SimilarityFn = Callable[[Counter, Counter], float]


class Recommender:
    def __init__(self, graph: UserGraph, similarity: SimilarityFn = cosine_similarity) -> None:
        self.graph = graph
        self.similarity = similarity

    def similar_users(self, user: str, topn: int = 20) -> List[Tuple[str, float]]:
        target_vec = self.graph.get_user_vector(user)
        sims = []
        for other in self.graph.users():
            if other == user:
                continue
            s = self.similarity(target_vec, self.graph.get_user_vector(other))
            if s > 0.0:
                sims.append((other, s))
        return sorted(sims, key=lambda x: x[1], reverse=True)[:topn]

    def recommend(self, user: str, k: int = 10, neighbors: int = 20, min_sim: float = 0.0) -> List[Tuple[str, float]]:
        user_vec = self.graph.get_user_vector(user)
        user_items = set(user_vec.keys())
        nbrs = self.similar_users(user, topn=neighbors)
        scores: Dict[str, float] = {}
        for n_user, sim in nbrs:
            if sim < min_sim:
                continue
            for item, w in self.graph.products_for_user(n_user):
                if item in user_items:
                    continue
                scores[item] = scores.get(item, 0.0) + sim * w
        return top_k(scores.items(), k)


def demo_small() -> None:
    g = UserGraph()
    g.add_interaction("u1", "p1", 3)
    g.add_interaction("u1", "p2", 1)
    g.add_interaction("u2", "p2", 2)
    g.add_interaction("u2", "p3", 5)
    g.add_interaction("u3", "p1", 1)
    g.add_interaction("u3", "p3", 1)
    g.add_interaction("u3", "p4", 2)
    rec = Recommender(g)
    print("Similar to u1:", rec.similar_users("u1"))
    print("Recommend for u1:", rec.recommend("u1", k=3))


if __name__ == "__main__":
    demo_small()
