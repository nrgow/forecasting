So currently the LLM based news market relevance calculation is performed together with the future timeline generation. What we would prefer to do is to separate those two steps out so that they can be performed independently and in run pipeline, they should be called by separate steps.

