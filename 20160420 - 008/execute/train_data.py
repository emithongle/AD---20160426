from libs.store_manage import loadTrainingTestingData
from libs.models import buildModel
from config import model_configs
from sklearn.metrics import accuracy_score
from libs.store_manage import saveModel

def _exec():
    def training(data, cl):
        clf = buildModel(model_configs[cl + '-model'])
        clf.fit(data[cl]['X'], data[cl]['y'])
        return clf

    def testing(data, clf, cl):
        return accuracy_score(data[cl]['y'], clf.predict(data[cl]['X']))

    train, test = loadTrainingTestingData()

    clf_name = training(train, 'name')
    clf_address = training(train, 'address')
    clf_phone = training(train, 'phone')

    acc_name = testing(test, clf_name, 'name')
    acc_address = testing(test, clf_address, 'address')
    acc_phone = testing(test, clf_phone, 'phone')

    saveModel({
        'name': clf_name,
        'address': clf_address,
        'phone': clf_phone
    })

    print('Acc_name = ', acc_name,
          ' - Acc_address = ', acc_address,
          ' - Acc_phone = ', acc_phone)

