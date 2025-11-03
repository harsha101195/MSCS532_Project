import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datastructures import Trie, UserGraph
from src.recommender import Recommender


def test_trie():
    t = Trie()
    t.insert("apple", payload="p_apple")
    t.insert("app", payload="p_app")
    t.insert("banana", payload="p_banana")
    results = t.search_prefix("app")
    flat = [w for w, _ in results]
    assert "app" in flat and "apple" in flat


def test_graph_and_recs():
    g = UserGraph()
    g.add_interaction("u1", "p1", 2)
    g.add_interaction("u1", "p2", 1)
    g.add_interaction("u2", "p2", 1)
    g.add_interaction("u2", "p3", 2)
    rec = Recommender(g)
    sims = rec.similar_users("u1")
    assert sims and sims[0][0] == "u2"
    r = rec.recommend("u1", k=2)
    assert any(pid == "p3" for pid, _ in r)