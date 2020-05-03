# Hockey Betting

- Uses Nadaraya-Watson estimator with triangular kernels to predict the home team's score, and the visitor's score of a hockey game
- Points in feature space are more spread out in the direction most correlated to the score
- Highly correlated features are amalgamated into a single dimension in feature space, effectively shrinking the dimension
- The goals arrive according to a Poisson process with its mean predicted by the Nadaraya-Watson estimator
- Able to assign a probability to various outcomes of the game's score