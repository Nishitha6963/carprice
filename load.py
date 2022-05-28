# -*- coding: utf-8 -*-

import pickle

with open("random_forest_regression_model.pkl", 'rb') as file:
     rf = pickle.load(file)

rf.predict([[1.5,35000,0,8,0,1,1,1]])[0]


# Present_Price                 1.47
# Kms_Driven                33000.00
# Owner                         0.00
# no_year                       9.00
# Fuel_Type_Diesel              0.00
# Fuel_Type_Petrol              1.00
# Seller_Type_Individual        1.00
# Transmission_Manual           1.00