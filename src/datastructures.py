from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Iterable
import heapq
import math
from collections import defaultdict, Counter


class TrieNode:
    __slots__ = ("children", "is_end", "payloads")

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.payloads: List[str] = []


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str, payload: Optional[str] = None) -> None:
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
        if payload is not None:
            node.payloads.append(payload)

    def _collect(self, node: TrieNode, prefix: str, out: List[Tuple[str, List[str]]]) -> None:
        if node.is_end:
            out.append((prefix, list(node.payloads)))
        for ch, nxt in node.children.items():
            self._collect(nxt, prefix + ch, out)

    def search_prefix(self, prefix: str, limit: Optional[int] = None) -> List[Tuple[str, List[str]]]:
        node = self.root
        for ch in prefix.lower():
            if ch not in node.children:
                return []
            node = node.children[ch]
        out: List[Tuple[str, List[str]]] = []
        self._collect(node, prefix.lower(), out)
        return out[:limit] if limit is not None else out


class UserGraph:
    def __init__(self) -> None:
        self.graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.user_hist: Dict[str, Counter] = defaultdict(Counter)

    def add_interaction(self, user: str, product: str, weight: float = 1.0) -> None:
        self.graph[user].append((product, weight))
        self.user_hist[user][product] += weight

    def get_user_vector(self, user: str) -> Counter:
        return self.user_hist.get(user, Counter())

    def users(self) -> Iterable[str]:
        return self.graph.keys()

    def products_for_user(self, user: str) -> List[Tuple[str, float]]:
        return list(self.graph.get(user, []))


def cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    if len(a) > len(b):
        a, b = b, a
    for k, va in a.items():
        vb = b.get(k, 0.0)
        if vb:
            dot += va * vb
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def jaccard_similarity(a: Counter, b: Counter) -> float:
    sa, sb = set(a.keys()), set(b.keys())
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def top_k(items: Iterable[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    if k <= 0:
        return []
    return heapq.nlargest(k, items, key=lambda x: x[1])
