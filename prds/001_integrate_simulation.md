This is a bigger one so take your time to plan it.

An EventGroup (i.e. a question on Polymarkets) is _simulationActive_ when we want to track how real-time news affects the probability of that event happening.

_simulationActive_ entails 2 things: 
- *once-off, or occasional*: when the event group is marked as *simulationActive*, we generating a timeline *up to now*, containing all relevant information from wikipedia (or in future other sources). We can call this PresentTimeline. We save it in storage. The present timeline, in the course of generating it, might also allow derivation of associated keyterms that can be used later for query expansion/news relevance judgment.
- *near real-time*: every 15 minutes or so, we scan all new news articles that came from the newsDownloader. we assess for each *simulationActive* EventGroup, if the 

Requirements:
- integrate the initial PresentTimeline generation and storage. You can leverage @src/simulation/generate_present_timeline.py . The list of _simulationActive_ eventGroups is currently stored in mock data in @data/active_event_groups.jsonl. It will be used as an additional step in the current run_pipeline method. E.g. generate_active_present_timelines(*...).
- implement the near realtime part, saving the results in storage.
- We will need to store the news X eventgroup relevance judgments in storage, because this may be interesting to browse in the frontend.

But for the moment, we just have @data/active_event_groups.jsonl. This is the list of EventGroups that should have simulations generated for them.

Please try to think nice abstractions and not just script slop. Because everything is currently very provisional. But this will eventually become more mature and we need flexibility about tech choices, e.g. the choice of storage. So consider hiding information behind Interfaces or classes as appropriate. Strike a balance between simplicity but not underabstracted.

Currently the storage should preferable by the filesystem and appropriate jsonlines files. Later when the exact shape of the flows becomes more fixed, we will move to one or several databases.