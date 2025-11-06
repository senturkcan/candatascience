from test_utils import *


ds, x, y = extract(dataset_path = "wine.data.txt", label_column = 0)

initial_analysis(ds)

main_analysis(ds, x, y, label_column=0)

