# Dual-Path Consistency Learning: A Lightweight Framework for Low-Light Coal Mine Image Enhancement via Stochastic Degradation Perturbations


## Codes
### Requirements
* python3.9
* pytorch==1.8.0
* cuda11.1

### Introduce the trained model
If you want to retrain a new model, you could write the path of the dataset to "train.py" and run "train.py", the final model will be saved to the weights folder, and the intermediate visualization results will be saved to the results folder.

In addition, we also provide "finetune.py" in the code root directory, which aims to make the method proposed in this paper better applicable to more variable scenarios. Specifically, if you are not satisfied with the performance of pretrain model in some other low-light scenes, you could use "finetune.py" to fine-tune the models to get a model whose performance is satisfactory to you.

### Testing
* Prepare the data and put it in the specified folder
* Choose your model as needed 
* Run "test.py"