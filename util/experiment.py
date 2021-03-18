import numpy as np
from models.base_model import SingleDomainModel
from util.data import DataGenerator


def run_experiment(
    src_data,
    target_data,
    model_class=SingleDomainModel,
    model_params={},
    target_labels=0.01,
    patience=5,
    num_experiments=5,
    verbose=False,
    min_epochs=20,
    max_epochs=100,
    val_split=0.2,
    test_split=0.2,
    img_shape=(224, 224),
):
    test_acc = []

    for i in range(num_experiments):
        print("Running test ", i+1)

        d = DataGenerator(
            source_domain=src_data,
            target_domain=target_data,
            val_split=val_split,
            test_split=test_split,
            input_shape=img_shape,
            target_labels=target_labels
        )

        train = d.train_data()
        val = d.val_data()
        test = d.test_data()
        model_params["classes"] = len(d.classes)

        m = model_class(**model_params)
        _ = m.cuda()
        m.train_model(min_epochs, max_epochs, train, val, patience=patience, verbose=verbose)
        loss, acc = m.evaluate(test)
        test_acc.append(acc["class_acc"])

    avg_acc = np.mean(test_acc) * 100
    std = np.std(test_acc) * 100
    print("Avg Acc: ", avg_acc)
    print("Std: ", std)
    return test_acc