# dvc-practice

1. GitHub repo creation
2. Clone the GitHub repo
---
3. Run `dvc init`
4. Download MNIST dataset from [here](https://www.kaggle.com/c/digit-recognizer/data)
5. make MNIST dataset to be half [link](https://gist.githubusercontent.com/deep-diver/97be5e5ff9f6a7a00e0579043c165ec6/raw/722553c19448e591e07ebf791f8ed3bf218e9d37/separate_half.py)
6. place MNIST dataset under `data/train.csv`
7. Run `dvc add data/train.csv`
8. Run `dvc remote add my_stroage -d /tmp/dvc-test`
9. Run `git add .`
10. Run `git commit -m "initial commit"`
11. Run `dvc push`
---
12. Create `params.yaml`
13. Create `src/`
14. Create `src/split.py`
15. Run `dvc run -n split -p split.ratio -d src/split.py -d data/train.csv -o data/prepared python src/split.py data/train.csv`
16. Create `src/preprocessing.py`
17. Run `dvc run -n preprocess -d src/preprocessing.py -d data/prepared -o data/preprocessed python src/preprocessing.py data/prepared`
18. Create `src/train.py`
18. Run `dvc run -n train -d src/train.py -d data/preprocessed -o data/model python src/train.py data/preprocessed data/model`
19. Create `src/evaluate.py`
[ ] Run `dvc run -n evaluate -d src/evaluate.py -d data/model
<!-- 20. Run `dvc run -n evaluate -d src/evaluate.py -d data/model -->
---