import joblib
model = joblib.load('D:/study/project/retail_app/model/gradient_boosting_clv.pkl')
print(model.__getstate__()['_sklearn_version'])
