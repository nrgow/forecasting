import dspy


class SelectRelevantArticles(dspy.Signature):
    """Select articles that are directly or indirectly relevant to the query or reference article."""

    query_or_article: str = dspy.InputField()
    articles: list[str] = dspy.InputField()
    relevant_articles: list[str] = dspy.OutputField()