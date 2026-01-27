We'd like to set up evaluations for the so called present timeline that's implemented in various forms within this project.
In fact, we would like to evaluate a reduced version of this present timeline, let's call it a one-day summary.
So the goal is simply to take a topic and a date and to summarise the news or the relevant updates for that topic.
So the evaluation will be agnostic to the specific model, the prompt, and the tools available, and the agent topology.
the only inputs to this task will be the date of interest and the topic of interest.
And we will use an LLM as a judge with a rubric based evaluation.
And then the goal will be to evaluate different setups, including the various setups that we already have, and to evaluate them with respect to the rubric as judged by the LLM, or perhaps even as judged by multiple LLMs. And secondly, the second dimension that would need to be tracked is the cost. That will include cost of LLM inference for the agent, total cost, as well as any costs that are associated with the tool calls.
So your goal is basically to first look at the different implementations of the present timeline. There are a few. There's a dspy based and there's perplexity based. I think those are the main ones. And in terms of the tools, right, there's Wikipedia and then there's Exa. And we can also implement a hybrid index based on gdelt.
The agents for the present timeline as they're currently implemented, of course, do not strictly speaking follow this this pattern. They're currently implemented to summarise all of the news maybe from the past two years to the present day. But we want to restrict this, so we may need to you know modify those agents.
For the rubric generation, lets take this article as an example of a good one-day summary: ./data/rubric_source_ukraine_conflict_2026_01_11.md
So we go from that article, and the rubric should reward this article.
So make a plan to evaluate this. Store information about your implementation in EVALS.md.

Once the evals are in place, we want to start optimizing the system to improve the agents in terms of quality and cost. So make sure we have sufficient tracing data to enable introspection.