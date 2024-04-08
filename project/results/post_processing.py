from results.models import TripletEvent, TripletEventResults


def filter_event(event: TripletEvent) -> TripletEvent:
    """
    Filter the event to only include the relevant fields
    """
    return TripletEvent(
        main=event.main,
        before=event.before,
        after=event.after,
    )


def filter_result(result: TripletEventResults) -> TripletEventResults:
    """
    Filter the results to only include the relevant fields
    """
    events = []
    for event in result.events:
        events.append(
            TripletEvent(
                main=event.main,
                before=event.before,
                after=event.after,
            )
        )

    return TripletEventResults(
        events=events,
        scores=result.scores,
        scroll_id=result.scroll_id,
        min_score=result.min_score,
        max_score=result.max_score,
        normalized=result.normalized,
    )
