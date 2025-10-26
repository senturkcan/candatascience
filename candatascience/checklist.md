End to End Professional Machine Learning Application Checklist

0- enviorment & version control setup
for .venv
for mac: 
python3 -m venv .venv
source .venv/bin/activate
for windows:
python -m venv .venv
.venv\Scripts\activate
(to deactivate write, deactivate )

for conda
conda create -n myenv python=3.12
conda activate myenv
(to deactivate write, conda deactivate )

install and import required libraries. for this library:
First time installation:
pip install git+https://github.com/senturkcan/candatascience.git

To recive updates from GitHub:
pip install --upgrade --force-reinstall git+https://github.com/senturkcan/candatascience.git

1- import (from xlsx,csv,sql)

2-initial information for missing value handleing.
handle missing values and other basic issues like column names

3- exporatory data analysis -> includes feature relationships for feature extraction

4-dataset shuffle &split

5*- encoding (handling categorical data) and maybe discretization

6*- anomaly/ outlier detection and deleting them. Gerçek hatalarsa test set için de yapılabilir.
istatistiksel düzeltmelerse scaling gibidir testle beraber yapılmamalıdır

7*- feature scaling

*: fit on training set apply on test set (to avoid data leakage)

opt. 8- feature extraction
! eğer linear modals, neural nets, svm, pca, lda, autoencoder ile extraction yapılacaksa scaling yapılmış olmalı.
diğerlerinde yapılmamış olması daha iyidir örn. RF, XGBoost. yani 7, 8 sırası buna göre belirlenmelidir ama genelde önce scaling yapılır.

9- modal selection and hyperparameter validation setup
-fit

opt. 10- comparing and combining diffrent modals on validation

11- evaluating the choosen modal cumerically and visually

opt. 12- feature importance metrics for more insights about the dataset

13- prepare the modal for production phase (launch)