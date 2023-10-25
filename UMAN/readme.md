UMAN Method

In UMAN.ipynb file run the cell for the specific task that you want to test (OE or AR, balanced or unbalanced). for AR, the number of instances selected for balanced and unbalanced datasets are written in the comments.

To create several pre-trained models from multiple source, run: UMAN.ipynb cells

Change the number of classes and input feature dimensions inside the code to fit the considered task.

TO TRAIN UMAN: !python main.py --config train-config-office-home.yaml

TO TEST UMAN: !python main.py --config test-config-office-home.yaml
