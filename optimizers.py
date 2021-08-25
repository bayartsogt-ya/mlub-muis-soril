from transformers import AdamW

def create_optimizer_roberta_large(model, learning_rate):
    lr = learning_rate
    multiplier = 0.975
    classifier_lr = 2e-5 # copied from the same notebook comment

    parameters = []
    for layer in range(23,-1,-1):
        layer_params = {
            'params': [p for n,p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier
    classifier_params = {
        'params': [p for n,p in model.named_parameters() if 'layer_norm' in n or 'linear' in n 
                   or 'pooling' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)
    return AdamW(parameters)
