# dvc-practice

1. GitHub repo creation
2. Clone the GitHub repo
---
3. Run `dvc init`
4. make MNIST dataset to be half [link](https://gist.githubusercontent.com/deep-diver/97be5e5ff9f6a7a00e0579043c165ec6/raw/722553c19448e591e07ebf791f8ed3bf218e9d37/separate_half.py)
5. place MNIST dataset under `data/train.csv`
6. Run `dvc add data/train.csv`
7. Run `dvc remote add my_stroage -d /tmp/dvc-test`
8. Run `git add .`
9. Run `git commit -m "initial commit"`
10. Run `dvc push`
---
11. Create `params.yaml`
12. Create `src/`
13. Create `src/split.py`
14. Run `dvc run -n split -p split.ratio -d src/split.py -d data/train.csv -o data/prepared python src/split.py data/train.csv`
15. Create `src/preprocessing.py`
16. Run `dvc run -n preprocess -d src/preprocessing.py -d data/prepared -o data/features python src/preprocessing.py data/prepared`
17. Run `dvc run -n train -d src/train.py -d data/features -o data/model python src/train.py data/features`
---