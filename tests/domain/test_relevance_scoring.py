import pytest
from bu_superagent.domain.services.relevance_scoring import cosine_similarity


def test_cosine_similarity_placeholder_raises():
    with pytest.raises(NotImplementedError):
        cosine_similarity([1.0], [1.0])
