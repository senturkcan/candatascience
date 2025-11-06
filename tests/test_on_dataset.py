from test_utils import *


ds, x, y = extract(dataset_path = "wine.data.txt", label_column = 0)

initial_analysis(ds)

main_analysis(ds, x, y, label_column=0)

detector = MLOutlierDetector(x, y)
detector.detect_all()

x,y = detector.remove_rows(indices_to_remove="all")

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

x_train, x_test = encode_and_scale_features(x_train,x_test)
y_train, y_test = encode_labels(y_train, y_test)