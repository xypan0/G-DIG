from abc import ABC
from collections import OrderedDict
from functools import reduce
import operator
from .layercollection import LayerCollection
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, LlamaPreTrainedModel

class LLamaLayerCollection(LayerCollection):
    def __init__(self, layers=None):
        super().__init__(layers)

    # override
    def from_model(model, ignore_unsupported_layers=False, ignore_layers=[]):
        # print('test')
        lc = LayerCollection()
        for layer, mod in model.named_modules():
            # print(layer, type(layer))
            flag=False
            for l in ignore_layers:
                if l in layer: 
                    flag=True
                    break
            if flag: continue
            mod_class = mod.__class__.__name__
            if mod_class in LayerCollection._known_modules:
                lc.add_layer('%s.%s' % (layer, str(mod)),
                             LayerCollection._module_to_layer(mod))
            elif not ignore_unsupported_layers:
                if len(list(mod.children())) == 0 and len(list(mod.parameters())) > 0:
                    raise Exception('I do not know what to do with layer ' + str(mod))

        return lc
        
if __name__=='__main__':
    model = LlamaForCausalLM.from_pretrained("/home/yarn/Influence/model/llama_2_7b/llama-2-7b-hf")
    lc=LLamaLayerCollection.from_model(model, True)
    print(lc.layers)