## GPU Image Processing for AI

### Brainstorming 
- Host something using github actions
- While the project says we need to parallelise a Computer Vision task - we have free reign to decide what task we want this to be
- Would be handy to have a interactive tool for showing off the NN - wish I had a tool to validate during testing


### Neural Network Optimsation
- Try to enforce that all the functions are using float32 rather thatn float64 - force this convention
- Make sure to always use np.array - avoid any unecssary converstion - never use python lists
- Some of our terms seem to be 'blowing up' test our activation functions with large numebrs to ensure this doenst happen
- Any loops check if we can speedup using the numpy functions - for example can this loop utilizise np.dot, np.outer?
- Time a single itteration of training and see if it will scale ok
- Can we train in small batches and save the data? look into this




### References
Neural Network Playlist: https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=SMAt1v60Yihg65gT
Backpropogation Calculus: https://www.3blue1brown.com/lessons/backpropagation-calculus/