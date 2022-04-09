import API_multi as api


net = api.Models(model='ResNet', num_layers=18, num_class=200).net()
dataset = api.Datasets('TinyImageNet', 64)

features = net.get_features(dataset, 10, 0, return_type='mean', verbose=True)
print(len(features))