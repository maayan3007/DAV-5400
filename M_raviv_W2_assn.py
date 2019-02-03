#!/usr/bin/env python
# coding: utf-8

# In[10]:


import csv
# be sure to update the path below to reflect your own environment!!
# also be sure that the code is properly indented after you paste it!
with open('C:\Python/cars-sample35.txt') as csvfile:
     readCSV = csv.reader(csvfile)
     for row in readCSV:
 # print each row as read by the csv.reader function
         print(row)


# In[12]:


#Create new list for price

price_list=[]

import csv
with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV: 
        price_list.append(row[0]) #allocate all values from row 0

print(price_list)


# In[16]:


#Create new list for Maintenance cost

Maintenance_list=[]

import csv
with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV: 
        Maintenance_list.append(row[1]) #allocate all values from row 1

print(Maintenance_list)


# In[18]:


#Create new list for Number of doors

NUM_Doors_list=[]

import csv
with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV: 
        NUM_Doors_list.append(row[2]) #allocate all values from row 2

print(NUM_Doors_list)


# In[20]:


#Create new list for Number of passengers

NUM_Pass_list=[]

import csv
with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV: 
        NUM_Pass_list.append(row[3]) #allocate all values from row 3

print(NUM_Pass_list)


# In[21]:


#Create new list for Luggage capacity

Luggage_Cap_list=[]

import csv
with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV: 
        Luggage_Cap_list.append(row[4]) #allocate all values from row 4

print(Luggage_Cap_list)


# In[22]:


#Create new list for Safety rating

Safety_rate_list=[]

import csv
with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV: 
        Safety_rate_list.append(row[5]) #allocate all values from row 5

print(Safety_rate_list)


# In[23]:


#Create new list for Classification of vehicle

Class_veh_list=[]

import csv
with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)
    for row in readCSV: 
        Class_veh_list.append(row[6]) #allocate all values from row 6

print(Class_veh_list)


# In[78]:


#create a list of the index of all cars with medium price
#Supported by: https://stackoverflow.com/questions/28182569/get-all-indexes-for-a-python-list

med_price_list=([index for index, value in enumerate(price_list) if value == "med"])

print('The index of cars with medium price is:', med_price_list)


# In[62]:


#create a list of the num of passengers of all cars with medium price

Pass_med_price_list = []

Pass_med_price_list.append(NUM_Pass_list[6])
Pass_med_price_list.append(NUM_Pass_list[16])
Pass_med_price_list.append(NUM_Pass_list[20])
Pass_med_price_list.append(NUM_Pass_list[23])
Pass_med_price_list.append(NUM_Pass_list[26])
Pass_med_price_list.append(NUM_Pass_list[29])



print('The number of passengers for automobile with price medium:',Pass_med_price_list)



# In[79]:


#create a list of the num of passengers of all cars with medium price
#same as before but isn't working, can't figure out how to correct this


Pass_med_price_list = []

for item in med_price_list:
  Pass_med_price_list.append(NUM_Pass_list[item])
      
print (Pass_med_price_list)


# In[47]:


#create a list of the index of all cars with high price and low maintainance:

import csv
with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)
    
    price_paramter='high'  #prameters that will hold the values we target
    maintain_parameter='low'
    
    highP_lowM_list=[]
    
    for index,row in enumerate(readCSV): 
        if row[0]==price_paramter and row[1]!=maintain_parameter:
            highP_lowM_list.append(index) 
    print(highP_lowM_list)
        
 


# In[49]:


#create a list of the index of all cars with two doors and big luggage:

import csv
with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)
    
    doors_paramter='2'     #prameters that will hold the values we target
    luggage_parameter='big'
    
    twoD_bigL_list=[]
    
    for index,row in enumerate(readCSV): 
        if row[2]==doors_paramter and row[4]==luggage_parameter:
            twoD_bigL_list.append(index) 
    print(twoD_bigL_list)
    
    
 


# In[56]:


#create a new list that contain the integer equivalents of the doors values+ convert string to int+ calculate average

with open('C:\Python/cars-sample35.txt') as csvfile:
    readCSV = csv.reader(csvfile)

    Num_doors6 = []
    
    for row in readCSV: 
        NUM_Doors_list = row[2]  # defining the elements index
        Num_doors6.append(NUM_Doors_list)  # store each column in a set
        Num_doors6 = [num.replace('5more','5') for num in Num_doors6] # replace value '5more' to 5
    
    Num_doors6 = [int(i) for i in Num_doors6]
    print(Num_doors6)
    print ("\n")
       
    print('The average number of doors per auto:', sum(Num_doors6) / (len(Num_doors6)))
    
    #assitance of average https://stackoverflow.com/questions/9039961/finding-the-average-of-a-list

