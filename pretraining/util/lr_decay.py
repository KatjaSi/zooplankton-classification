import torch
from transformers import ViTMAEConfig, ViTMAEForPreTraining
import ipdb


def get_layer_id_for_vit(name, num_layers, num_layers_decoder):
    if 'cls_token' in name or 'patch_embeddings' in name:
        return 0
    elif name.startswith('vit.encoder.layer'):
        return int(name.split('.')[3])+1
    elif name.startswith('decoder.decoder_norm') or name.startswith('decoder.decoder_pred'):
        return num_layers - num_layers_decoder
    elif name.startswith('decoder.decoder_layers'):
        return num_layers - (int(name.split('.')[2]) + 1) 
    else:
        return num_layers #'vit.layernorm.weight' decoder.mask_token

def param_groups_lrd(model, 
                        weight_decay=0.05, 
                        no_weight_decay_list=[
                        'vit.embeddings.cls_token', 
                        'decoder.mask_token',
                        'vit.embeddings.patch_embeddings.projection.weight',#TODO: exclude?
                        'vit.embeddings.patch_embeddings.projection.bias' #TODO: exclude?
                        ], 
                        layer_decay=.65):
    param_group_names = {}
    param_groups = {}
    num_layers = len(model.vit.encoder.layer) + 1
    
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    num_decoder_layers = len(model.decoder.decoder_layers) + 1


    for name, param in model.named_parameters():            
        if not param.requires_grad:
            # vit.embeddings.position_embeddings  'decoder.decoder_pos_embed'
            continue

        # no decay: all 1D parameters and model specific ones
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(name, num_layers, num_decoder_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(name)
        param_groups[group_name]["params"].append(param)


    return param_group_names, list(param_groups.values())

if __name__ == "__main__":
    device = torch.device("cuda")
   
    config = ViTMAEConfig(norm_pix_loss=True,  # corresponding vit-mae-base layers
                          mask_ratio=0.75,
                          hidden_size=768,
                          intermediate_size=3072,
                          num_attention_heads=12,
                          num_hidden_layers=12,
                          num_channels=1
                          )
    model = ViTMAEForPreTraining(config).to(device) 
    param_group_names, param_group_values = param_groups_lrd(model)
    ipdb.set_trace()
    pass

