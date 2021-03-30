# Traffic-light-management-system
This repository contains files of traffic light management system based on Reinforcement Learning.

## Basic Idea 

![sample](documentation/samplecity1.png)

suppose we have a city grid as shown above with 4 traffic light nodes.<br/>
n1, n2 , n3 and n4

Our model makes 4 decisions (one for each node)  for which side to select for green signal<br/>

we have to select a minimum time (for ex 30s) our model can not select green light time below that limit.

Out task is to minimize amount of time vehicles have to wait on the traffic signal.<br/>
Amount of waiting time for given traffic signal is equal to total car present on the signal x number of seconds<br/>
each traffic signal will have 4 waiting time counter for each side of road. So based on that our model will<br/>
decide which side to select for green signal.

## Basic training process.

We trained our model on some number of events.<br/>
Event is defined as a fixed motion where vehicles will pass through node in a fixes (pseudo-random manner).<br/>
reason for keeping event fixed is that using random event everytime will give random result.<br/>
we will use many such fixed events to train our model so our model could handle different situations.

only input our model will receie is number of vehicles present on 4 sides of each traffic node.<br/>
and out model will output 4 sides one for each node and amount of time for each node.

number of nodes depends on size of the grid.

## SUMO for siumlation

We used SUMO open source software to make maps and generate simulation to train our model.

Here are the examples of some of the maps used to train the model.

### Map1 
![map1](maps_images/city2.JPG)

### Map2
![map2](maps_images/city3.JPG)

### Map 3
![map3](maps_images/citymap.JPG)


