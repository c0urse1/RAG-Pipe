"""
# Composition Root: injiziert konkrete Adapter in Use-Cases.
# Wichtig: keine Geschäftslogik, nur Wiring.
"""


def build_query_use_case():
    raise NotImplementedError
