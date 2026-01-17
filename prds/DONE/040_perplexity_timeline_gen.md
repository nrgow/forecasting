So like a new approach for the construction of the present timeline. This approach will skip the current present timeline generation. It will also skip the the relevant news calculation for now. Instead it will simply connect to a perplexity model via open router. I can add a link to the the basic code. 

> Construct a dense, grounded timeline, 
> including concrete events relating to the topic, 
> as well as statements made by the involved people and organizations,
> going back in time maximum past two years, 
> with heavy focus on the most recent events (today is 2026, January 16),
> for the following topic:
>    "potential American recognition of Somaliland".

So that's the basic prompt structure. So there'll be templates template variables for the current dates. There'll also be a template variable for the topic that the event group is referring to. I think for the moment we can just use the event group title.

> perplexity/sonar-deep-research

So that's the model that we will use via open router. Um I think we can just use the open AI library to interact with this model via open router.
And I'd also like an entry point in main_new.py to be able to generate in these present timelines. They should be saved maybe in a separate file, Jason Lines file, just to make clear that they are a completely different method of generating the timeline. And then because it's skipping the relevant news, when we're generating when we hook it up to the future timeline generation, we should also simply, as I mentioned, skip the skip the relevant news part. Don't even look at it.