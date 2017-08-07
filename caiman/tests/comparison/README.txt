# Comparison

We want to improve CaImAn, this will be done by carefully storing and 
comparing the results of CaImAn in respect
to human defined standards ( labeling of neurons ) and to previous 
iteration of CaImAn. 

In this Regard, any pushed code is going to be tested.

## Travis and test

[Travis](https://travis-ci.org/) enables to test the code using predefined functions. 
One of which will be the [comparison function]
(https://docs.google.com/document/d/1Pdft15ZcnLbdaUEftI7Nzak3cPTiLzmYAeKhXlfPFNw/edit?usp=sharing) 
Letting master developers possibility to look in the differences
 whether they are improving or not 
the present version of CaImAn 

_It is good to know that you can test the code yourself using 
the [nosetest](http://nose.readthedocs.io/en/latest/) package_

## Important information

_the ground truth is the dataset to which your output result will be compared to._

It is important to know that you shouldn't push new ground truth, you shouldn't 
tweak the parameters of the function inside of the tests folder 
without letting a master developer know about it. 

If you already know about the fact that your modifications will introduce 
differences in the output results that leads toward an improvement 
of CaImAn, you should call the nosetest function and find your 
comparison results inside the tests/comparison/tests folder. 
You can then rename the X(number) created folder as "tosend" for 
it to be sent to the master developers when doing your push request.

## More information

It is important to know that more information is present about 
this in the documentation of the code. 

* A mind map of how the data is stored is present inside 
the comparison folder. 

* the readme explains how to compare your algorithm

* look at the function in the tests and comparison folders. 

### Investigative comparison
It is a notebook that allows you to go deeper in the understanding of 
the Comparison of two different results using the bokeh library. 
You will have to re-run the comparison pipeline using the notebook.
**Exemple you can generate a groundtruth file from a previous version 
of CaImAn that you like and know well and compare it to a new version 
of CaImAn and investigate the differences in results.**

### Create your ground truth

You can generate your own ground truth by using 
the already implemented: create_gt.py function

1. select your desired parameters inside the params dictionary of the function
2. keep the old ground truth somewhere
3. run the function and then you have your new ground truth.