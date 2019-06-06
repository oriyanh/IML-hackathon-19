root
│   USERS.txt - Our login data
│   README.txt - This file :)
│   project.pdf - the report
│
└───task1
    │   __init__.py - For imports
    │
    ├───src
    │       classifier.py - Classifier you will call - contains "classify" method
    │       ClassifierBase.py - Base class for our classifiers, we use inheritence for simplicity when classifying new data (and unity)
    │       CNNClassifier.py - Convolutional neural network using keras, inherits from ClassifierBase
    │       Commons.py - Constants shared by all files
    │       data_parser.py - Used for pre-processing instances (tweets) before
    │       GenericSKLClassifier.py - Batch for sklearn classifiers, inherits from ClassifierBase
    │       MegaClassifier.py - Container for classify method content
    │       __init__.py - For imports
    │
    └───weights
            tokenizer.pickle - Serialized tokenizer used to carry over tokenization method between runs
            weights.wegh - Keras model of CNN