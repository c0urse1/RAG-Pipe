from bu_superagent.domain.services.relevance_scoring import cosine_similarity


def test_cosine_similarity_basic():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.5  # mapped to [0,1]
    assert 0.75 < cosine_similarity([1, 1], [1, 0]) < 0.9
