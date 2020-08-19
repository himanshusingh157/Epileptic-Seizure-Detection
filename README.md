#Dataset
The original dataset from the reference consists of 5 different folders, each with 100 files, with each file representing a single subject/person. Each file is a recording of brain activity for 23.6 seconds.\
The corresponding time-series is sampled into 4097 data points. Each data point is the value of the EEG recording at a different point in time. So we have total 500 individuals with each has 4097 data points for 23.5 seconds.\
We divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time.\
So now we have 23 x 500 = 11500 pieces of information, each information contains 178 data points for 1 second(column), the last column represents the label y =  {1,2,3,4,5}. The response variable is y in column 179, the Explanatory variables X1, X2, ..., X178. The column y contains the category of the 178-dimensional input vector.
Each class Represents:\
1)Recording of seizure activity\
2)They recorded the EEG from the area where the tumor was located\
3)Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area\
4)eyes closed, means when they were recording the EEG signal the patient had their eyes closed\
5)eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open\
\
All subjects falling in classes 2, 3, 4, and 5 are subjects who did not have epileptic seizure. Only subjects in class 1 have epileptic seizure.
