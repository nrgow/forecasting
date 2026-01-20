Build a simple tool calling agent to construct the "present timeline" as it is called in this project.
Is should be based on the openai sdk, but it should call open router.
It should basically be a deep research agent.
It should have access to the two Wikipedia search tools available 
It should trace all llm calls to mlflow.
It should implement a basic react tool calling loop.

The desidirata of the agent:
- the research agent should construct a chronological timeline from the past two years until now 
- the research agent should find all events related to and giving background context to the topic surrounding the polymarket event
- the timeline should simply provide factual information in its output, not focussing on analysis or outlook
- the timeline should be information dense
- the timeline should be formatted in markdown with the date as the subsection headers
- the timeline may also incorporate information from related parallel events if they potentially influence .
- the agent should log ho much of its context it has used during its execution.
- the agent can call special tools "think()" and "final_answer()"

This present timeline agent can be used as an alternative to the perplexity "current timeline agent" and the "dspy" timeline agent.
