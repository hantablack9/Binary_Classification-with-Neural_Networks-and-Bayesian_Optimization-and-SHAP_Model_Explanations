# Binary Classification with Neural Networks and Bayesian Optimization and SHAP Model Explanations

In the context of neural networks and hyperparameter tuning, Bayesian Optimization offers several advantages over traditional methods like Grid Search or Randomized Search Cross-Validation (CV). Bayesian Optimization is often superior because of the following:

*    Efficient Exploration: Bayesian Optimization intelligently explores hyperparameter spaces using probabilistic surrogate models. It starts with an initial set of configurations, estimating which hyperparameters might perform better. This approach efficiently explores promising regions, reducing iterations.

 *   **Fewer Evaluations:** Grid Search evaluates all possible hyperparameter combinations, becoming impractical with many parameters. Randomized Search CV is better but requires a fixed number of random evaluations. Bayesian Optimization adapts based on prior results, pruning unpromising configurations for quicker convergence.

 *   **Model Performance:** Bayesian Optimization focuses on optimizing specific metrics, leading to models with better overall performance, rather than randomly sampling hyperparameters.

 *   **Adaptive Sampling:** It uses the surrogate model to guide evaluations towards potentially optimal configurations. Grid and Randomized Search sample independently.

 *   **Sequential Nature:** Bayesian Optimization is sequential, adapting as new information comes in. It updates the surrogate model, refining performance estimates.

 *   **Balancing Exploration and Exploitation:** Bayesian Optimization balances exploration (trying new configs) and exploitation (focusing on promising ones) using acquisition functions. This balance is vital, especially for complex models.

 *   **Parallelization:** Bayesian Optimization's adaptability and efficient exploration make it well-suited for parallel execution, effectively distributing evaluations.

## In this notebook, I have attempted to demonstrate the effectiveness of Bayesian Optimization, along with some advanced techniques for preprocessing.
1. Missing values have been **imputed with Decision Trees.**
2. Used the **Kolmogorov-Smirnov test** to evaluate whether the distributions are Gaussian or not, and Used Standard Scaling on the non-Gaussian features.
3. Used a variety of shallow neural network architectures for Binary Classification task.
4. Applied **Bayesian Optimization from kerastuners** class for hyperparameter tuning across a massive search space.
5. Refitted the model with the best parameters to achieve an **Accuracy of 98%** and an **F1-score of 97% and 99% on minority and majority classes, respectively.**
6. **Explained the model using the SHAP** library's functions and visualized feature contributions, **for single and multiple observations.** 
