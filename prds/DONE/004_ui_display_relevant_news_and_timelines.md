So we'd like to add a new UI view and make modifications to the existing event group table.

The modification to the event group table is the following. We'd like to add a new column which will display if the event group is considered active. And we'd also like to add a filter above the table to be able to only display active filter active event groups.

And the second modification is the following it's the new view enabling us to visualize the data associated with event groups. In particular, we'd like to see what is for active event groups. What news has been annotated as being relevant for that event group. We'd like to see obviously all open markets for that event group. Descriptions or the events contained in the event group. And finally we'd like to see the currently generated timelines and future timelines for those event groups.

So exactly so we'd like to be able to serve all the relevant data with these views from the API the fast API backend. So that will need a new endpoint. The UI itself. We'll need a URL pattern for this event group display. It will also um navigate to the view of a particular event group. We'd like to click on the ID of the event group in the event group stats table.

