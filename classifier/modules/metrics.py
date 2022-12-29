from sklearn.metrics import accuracy_score, f1_score

def get_metric(metric_name):
    if metric_name == "accuracy":
        return accuracy_score
    if metric_name == "f1macro":
        metric = F1Score(average='macro')
        return metric.get_score

class F1Score:
    def __init__(self, average):
        self.average = average

    def get_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average=self.average)