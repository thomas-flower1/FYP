## GPU Image Processing for AI

### Brainstorming 
- Need to decide on the task to optimize with GPU
- Have an GUI for interactable configuring a NN


### Neural Network Optimization
- Try to enforce that all the functions are using float32 rather than float64 - force this convention
- Make sure to always use np.array - avoid any conversion
- never use python lists
- Some of our terms seem to be 'blowing up' test our activation functions with large numbers
- Can we train in small batches and save the data?
- batch processing?




### References
Neural Network Playlist: https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&si=SMAt1v60Yihg65gT
Backpropogation Calculus: https://www.3blue1brown.com/lessons/backpropagation-calculus/