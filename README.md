# Flappy Bird Genetic Evolution
Train a neural network using genetic evolution to play flappy bird! For a tutorial on how to get started with your own genetic algorithms, take a look [here](https://threads-iiith.quora.com/Neuro-Evolution-with-Flappy-Bird-Genetic-Evolution-on-Neural-Networks).

## Final Results
![Trained Model](Screenshots/trained_final.gif?raw=true "Trained Flappy Bird")	

## Algorithm Details
* Starts out with a pool of 50 models
* Each iteration based on fitness scores of the models, perform crossover
* Crossover would swap the first layers (input -> hidden) for both the selected parents
* Random mutation ensures that the models are changed at every iteration

## Progress Screenshots
#### Stage 1
Initially all the models would do the "same" wrong thing. So they would all die out quickly.

![Untrained No-Spread](Screenshots/untrained_initial_states_nospread.gif?raw=true "Untrained No-Spread")

#### Stage 2
After some time however, they would start to show more variation, but still perform the wrong moves. This gives us a spread of flappy birds throughout the screen (lengthwise).

![Untrained Spread](Screenshots/untrained_initial_states_spread.gif?raw=true "Untrained Spread")

#### Stage 3
After a bit of training (~1hr) the spread decreases and is more concentrated at the height where there is a hole in the pipes. They start performing a lot better since they now understand when to flap and when not to.

![Trained](Screenshots/trained_set_initial.gif?raw=true "Trained")

#### Stage 4
The model with the maximum fitness can be considered as a trained model and it would perform much better than the average human.

![Final Trained Model](Screenshots/trained_final.gif?raw=true "Final Trained Model")

### Disclaimer
* Based on Flappy Bird clone in pygame, https://github.com/sourabhv/FlapPyBird
