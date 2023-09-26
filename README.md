
# Immersion Measurement

Model for measuring immersion (immersive or not) based on the user's gaze data while watching videos.

This code provides some functions as follows:
- [x] Uses seven ML classifiers such as SVM, kNN, LogisticRegression, DecisionTree, RandomForest, AdaBoost, and NaiveBayes.
- [x] Improves accuracy with feature selection
- [x] Population model vs. individual model

For more details, please see our research paper.

**"Immersion Measurement in Watching Videos Using Eye-tracking Data"**
*IEEE Transactions on Affective Computing* (2022). [[paper](https://doi.org/10.1109/TAFFC.2022.3209311.)]


## Quick Start
```bash
  > git clone https://github.com/jspirit01/ImmersionMeasurement.git
  > cd ImmersionMeasurement
  > python main.py \
    --exp [your_experiment_name] \
    --fs True \
    --load_fs True \
    --print_test False \
    --data dataset/exp1+2_usernorm2_population.csv \
    --load_pretrained_model False \
    --seed 42
```


    
## Dataset
The dataset contains gaze data from 30 participants while watching 14 videos.
Seven statistical values were calculated for each gaze feature type.
A total of 49 features (7 gaze types X 7 statistical values) were used for classification.


## Model Types
### Population Model
- Evaluate using leave-one-person-out cross-validation.
- Calculate the average accuracy for 30 folds (i.e., 30 participants).
- Example of use:
```python
from src.population_loocv import population_model

population_model(
    'dataset/exp1+2_usernorm2_population.csv',
    fs = False,
    load_fs = False,
    print_test = True,
    load_pretrained_model = False,
    save_dir=f'./results/{args.exp}',
    log_path=f'./results/{args.exp}/run.log',
    seed=42)
```

### Individual Model
- Evaluate using leave-one-instance-out cross-validation.
- Calculate the average accuracy for 14 folds for each participant (i.e. 14 videos).
- Example of use:
```python
from src.individual_loocv import individual_model

individual_model(
    'dataset/exp1+2_usernorm2_individual.csv',
    fs = True,
    load_fs = True,
    print_test = True,
    load_pretrained_model = False,
    save_dir=f'./results/{args.exp}',
    log_path=f'./results/{args.exp}/run.log',
    seed=42)
```
## Citation
Please cite as:
```
@ARTICLE{9904895,
  author={Choi, Youjin and Kim, JooYeong and Hong, Jin-Hyuk},
  journal={IEEE Transactions on Affective Computing}, 
  title={Immersion Measurement in Watching Videos Using Eye-tracking Data}, 
  year={2022},
  volume={13},
  number={4},
  pages={1759-1770},
  doi={10.1109/TAFFC.2022.3209311}}
```