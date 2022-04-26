from pprint import pprint
from river import datasets


from river import compose
from river import linear_model
from river import metrics
from river import preprocessing



dataset = datasets.Phishing()

for x, y in dataset:
    pprint(x)
    print(y)
    break


# {'age_of_domain': 1,
#  'anchor_from_other_domain': 0.0,
#  'empty_server_form_handler': 0.0,
#  'https': 0.0,
#  'ip_in_url': 1,
#  'is_popular': 0.5,
#  'long_url': 1.0,
#  'popup_window': 0.0,
#  'request_from_other_domain': 0.0}
# True

model = compose.Pipeline(preprocessing.StandardScaler(),
    linear_model.LogisticRegression()
    )

metric = metrics.Accuracy()

for x, y in dataset:
    y_pred = model.predict_one(x)      # make a prediction
    metric = metric.update(y, y_pred)  # update the metric
    model = model.learn_one(x, y)      # make the model learn
# metric
# Accuracy: 89.20%
import pdb; pdb.set_trace()