# Traffic-light-management-system
This repository contains files of traffic light management system based on machine learning.

## Basic Idea 

![sample](documentation/samplecity1.png)

suppose we have a city grid as shown above with 4 traffic light nodes.<br/>
n1, n2 , n3 and n4

so our model makes total 8 decision 4 decisions for which side to select for green signal<br/>
plus 4 more decision for green light time signal.

we have to select a minimum time (for ex 30s) our machine can not select green light time below that limit.

Out task is to minimize amount of time cars have to wait on the traffic signal.<br/>
amount of waiting time for given traffic signal is = total car present on signal x number of seconds<br/>
each traffic signal will have 4 waiting time counter for each side of road. so based on that our model will<br/>
decide which side to select for green signal.

## Basic training process.

we will train our model on some number of events.<br/>
Event is defined as a fixed motion where vehicles will pass through node in a fixes (pseudo-random manner).<br/>
reason for keeping event fixed is that using random event everytime will give random result.<br/>
we will use many such fixed events to train our model so our model could handle different situations.

only input our model will receie is number of vehicles present on 4 sides of each traffic node.<br/>
and out model will output 4 sides one for each node and amount of time for each node.

number of nodes depends on size of the grid.



