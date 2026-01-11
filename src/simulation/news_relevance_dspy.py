import dspy


class SelectRelevantArticles(dspy.Signature):
    """Select articles that are directly or indirectly relevant to the query or reference article."""

    query_or_article: str = dspy.InputField()
    articles: list[str] = dspy.InputField()
    relevant_articles: list[str] = dspy.OutputField()


class DeepSearchRelevantArticles(dspy.Signature):
    """Use BM25 search to find relevant articles for an event group."""

    event_group_prompt: str = dspy.InputField(
        desc="Event group description, including title and key event context."
    )
    present_timeline: str = dspy.InputField(
        desc="Current timeline context to guide query expansion."
    )
    max_results: int = dspy.InputField(
        desc="Maximum number of relevant article ids to return."
    )
    relevant_article_ids: list[str] = dspy.OutputField(
        desc="Article ids judged relevant."
    )
