Each active event group is associated with a set of open markets.

Each open market relates to the state of the world on a particular date. So I guess that might be the resolution date. Or there may be some metadata fields in the open market object that one might call the event date or something like this.

So when we're generating the future timeline and the future timeline for an active event, we would like to add to the inputs for the future timeline. This event dates. Essentially it's instructing the LLM to generate its future timeline. Only until that event date.

So just to be clear, for an active event group, when generating its future timeline, take the final or the latest event dates or resolution dates over all of its open markets.

