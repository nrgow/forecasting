from functools import partial
from multiprocessing import Pool

import dspy
import wikipedia


class FutureTimeline(dspy.Signature):
    """Generate a realistic chronological timeline related to the scenario or topic from the current date to the foreseeable future"""

    timeline_scenario: str = dspy.InputField()
    contexts: list[str] = dspy.InputField()
    current_date: str = dspy.InputField()
    simulated_timeline: str = dspy.OutputField()


class TimelineImplication(dspy.Signature):
    """Given a simulated future timeline and a question related to the timeline, return whether time timeline as stated implies the answer to the question being true"""

    timeline: str = dspy.InputField()
    question_to_answer: str = dspy.InputField()
    implied_answer: bool = dspy.OutputField(
        desc="The answer to the question, given the timeline"
    )


def run_model(
    model,
    scenario,
    contexts,
    current_date,
    final_question,
    temps=[0.1, 0.3, 0.5, 0.7, 0.9],
):
    dspy.configure(lm=dspy.LM(model))
    rval = []
    for rollout_id, temp in enumerate(temps):
        try:
            timeline_prediction = dspy.Predict(
                FutureTimeline, rollout_id=rollout_id, temperature=temp
            )
            question_answering = dspy.ChainOfThought(TimelineImplication, temperature=0)

            timeline = timeline_prediction(
                timeline_scenario=scenario,
                contexts=contexts,
                current_date=current_date,
            )
            timeline_implication = question_answering(
                timeline=timeline.simulated_timeline, question_to_answer=final_question
            )
            rval.append(
                {
                    "model": model,
                    "rollout_id": rollout_id,
                    "temp": temp,
                    "simulated_timeline": timeline.simulated_timeline,
                    "choice": timeline_implication.implied_answer,
                }
            )
        except:
            rval.append({"model": model, "rollout_id": rollout_id, "error": True})

    return rval


def generate_future_timeline(
    scenario: str, wiki_context_pages: list[str], current_date: str, final_question: str
):
    contexts = []
    for page_title in wiki_context_pages:
        page = wikipedia.page(page_title)
        context = page.title + "\n" + page.content
        contexts.append(context)

    models = [
        #"openrouter/anthropic/claude-opus-4.5",
        #"openrouter/openrouter/bert-nebulon-alpha",
        "openrouter/x-ai/grok-4.1-fast",
        #"openrouter/google/gemini-3-pro-preview",
        #"openrouter/openai/gpt-5.1",
    ]
    prediction_function = partial(
        run_model,
        scenario=scenario,
        contexts=contexts,
        current_date=current_date,
        final_question=final_question,
    )

    results = []
    with Pool(processes=len(models)) as pool:
        processed = pool.map(prediction_function, models)
        for model_results in processed:
            for result in model_results:
                results.append(result)
    return results
