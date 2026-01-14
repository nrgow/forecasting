So I'd like you to um implement one more user relevance approach.
So we as a reminder we currently have two news relevance approaches. The first one is uh the zero shot classifier, followed by the LLM based um relevance classifier.
And the second one is the agentic lazy uh approach, which constructs sparse and dense indices and then allows an agent to query those indices.
So the third approach that should now be implemented is closer to the first um method. So actually it will be an eager approach, so it will touch every document. It will also have the same zero shot classifier at the beginning. And then there'll be a second step to use a local re-ranking model.
https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
So there's the re ranking model. Um and you can also uh uh uh take a look at that page to uh see uh how it should be used, how the prompts should be formatted, and how the actual relevance score should be calculated from the logits.
Right, um so the same as the first approach, this re ranker should um be applied for every document and for every event group that is active.