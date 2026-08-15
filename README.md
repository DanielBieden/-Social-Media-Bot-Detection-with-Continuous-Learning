This is the repository for the student group 3 for the course "Natural Language Processing and Neural Language Models" from the Ulm University.
The subject of this project is social media bot detection with continuous learning.

## Setup Requirements

- The needed packages can be installed with the use of `requirements.txt`
- Additionally, the system library __YAJL__ is required to speed up the streaming of the larger datasets
  - **Windows (Anaconda):** `conda install -c conda-forge yajl`
  - **Ubuntu/Debian Linux:** `sudo apt-get install libyajl2`
  - **macOS:** `brew install yajl`
- The project relies on some not publicly available datasets, which can be obtained by contacting the associated authors
  - the dataset files should be placed in a directory (like 'datasets')

## Datasets
The relevant datasets for this project include:

- Caverlee 11
- Cresci 17
- Cresci 18
- Twibot 20
- Twibot 22

## Experiments
The relevant experiments for the project are located in the `experiments` directory, 
as it includes the baseline classifier evaluation, the hyperparameter search & the continuous learning experiment loop.

### Continuous Learning Experiment Framework

 focuses on the continuous learning experiment loop. It includes the shared runner, the performance matrix, and the classifier strategies used to test different models under the same task order and evaluation setup.

Relevant files can be found in the main folder:

- `experiment_runner.py`: runs the sequential learning loop, trains the selected classifier, stores previous test sets, and evaluates the model after each task.
- `continual_experiment_manager.py`: stores the performance matrix and calculates continual learning metrics such as Average Accuracy, Forgetting Measure, and Average Incremental Accuracy.
- `classifier_strategies.py`: contains the classifier strategies, including the PNN, Confidence Classifier, SVM, Baseline Classifier, and the modified Transformer classifier.
- `continual_experiment_MultiClass.ipynb`: main notebook used for the multiclass continuous learning experiments.

To run the main continuous learning experiment, open:

continual_experiment_MultiClass.ipynb

#### Transformer Experiment 
As part of the course, exploratory experiments involving the transformer architecture where performed and are located in the `transformer_experiments` directory.
