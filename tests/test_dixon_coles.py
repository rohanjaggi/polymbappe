from polymbappe.models.dixon_coles import DixonColesModel, MatchObservation


def test_dixon_coles_fit_and_predict_probabilities_sum_to_one() -> None:
    matches = [
        MatchObservation("A", "B", 1, 0, 10, "World Cup Qualifier"),
        MatchObservation("B", "A", 0, 2, 20, "World Cup Qualifier"),
        MatchObservation("A", "C", 2, 1, 5, "Friendly"),
        MatchObservation("C", "B", 0, 0, 2, "Nations League"),
        MatchObservation("B", "C", 1, 1, 1, "World Cup"),
        MatchObservation("C", "A", 0, 1, 3, "Euro Qualifier"),
    ]
    model = DixonColesModel().fit(matches)
    probs = model.predict_match("A", "B")
    assert probs["home_win"] > 0.0
    assert probs["draw"] > 0.0
    assert probs["away_win"] > 0.0
    assert abs(sum(probs.values()) - 1.0) < 1e-6
