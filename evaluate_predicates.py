class PredicateEvaluator:
    def __init__(self):
        pass

    def evaluate_predicate(self, data, value, operator):
        if operator == ">":  return data > value
        if operator == ">=": return data >= value
        if operator == "<":  return data < value
        if operator == "<=": return data <= value
        if operator == "=":  return data == value
        if operator == "!=": return data != value
        raise ValueError(f"Unknown operator: {operator}")

predicate_evaluator = PredicateEvaluator()