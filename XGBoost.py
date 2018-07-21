from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataset = load_iris()
train_features, test_features, train_labels, test_labels = train_test_split(dataset.data,
                                                                            dataset.target,
                                                                            test_size=0.2,
                                                                            random_state=42)
def classifier(classifier_type, classifier_name):
    classifier_type.fit(train_features, train_labels)
    score = classifier_type.score(test_features, test_labels)
    prediction = classifier_type.predict(test_features)
    print(prediction)
    print("Classifier used", classifier_name, "with accuracy", score*100)

classifier(XGBRegressor(), "XGBoost Regressor")
classifier(XGBClassifier(), "XGBoost Classifier")