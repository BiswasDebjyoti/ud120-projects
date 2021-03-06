#!/usr/bin/python

#from poi_email_addresses import poiEmails

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
count =0

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print enron_data
for item in enron_data:
    #print (enron_data[item]['poi'])
    if (enron_data[item]['poi'])==1:
        count+=1

#print count

