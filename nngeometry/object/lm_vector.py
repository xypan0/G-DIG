import torch
from ..llama_layercollection import LayerCollection
from nngeometry.llama_layercollection import LLamaLayerCollection

def random_pvector_dict(layer_collection, device=None):
    """
    Returns a random :class:`nngeometry.object.PVector` object using
    the structure defined by the `layer_collection` parameter, with 
    each components drawn from a normal distribution with mean 0 and standard
    deviation 1.

    The returned `PVector` will internally use a dict representation.

    :param layer_collection: The :class:`nngeometry.layercollection.LayerCollection`
    describing the structure of the random pvector
    """
    v_dict = dict()
    for layer_id, layer in layer_collection.layers.items():
        if layer.bias is not None:
            v_dict[layer_id] = (torch.normal(0, 1, layer.weight.size, device=device),
                                torch.normal(0, 1, layer.bias.size, device=device))
        else:
            v_dict[layer_id] = (torch.normal(0, 1, layer.weight.size, device=device),)
    return PVector(layer_collection, dict_repr=v_dict)


def random_pvector(layer_collection, device=None):
    """
    Returns a random :class:`nngeometry.object.PVector` object using
    the structure defined by the `layer_collection` parameter, with 
    each components drawn from a normal distribution with mean 0 and standard
    deviation 1.

    The returned `PVector` will internally use a flat representation.

    :param layer_collection: The :class:`nngeometry.layercollection.LayerCollection`
    describing the structure of the random pvector
    """
    n_parameters = layer_collection.numel()
    random_v_flat = torch.normal(0, 1, (n_parameters,),
                               device=device)
    return PVector(layer_collection=layer_collection,
                   vector_repr=random_v_flat)


def random_fvector(n_samples, n_output=1, device=None):
    random_v_flat = torch.normal(0, 1, (n_output, n_samples,),
                                device=device)
    return FVector(vector_repr=random_v_flat)


class PVector:
    """
    A vector in parameter space

    :param:
    """
    def __init__(self, layer_collection, vector_repr=None,
                 dict_repr=None):
        self.layer_collection = layer_collection
        self.vector_repr = vector_repr
        self.dict_repr = dict_repr
        self.is_svd=False

    @staticmethod
    def from_model(model):
        """
        Creates a PVector using the current values of the given
        model
        """
        dict_repr = dict()
        layer_collection = LayerCollection.from_model(model)
        l_to_m, _ = layer_collection.get_layerid_module_maps(model)
        for layer_id, layer in layer_collection.layers.items():
            mod = l_to_m[layer_id]
            if layer.bias is not None:
                dict_repr[layer_id] = (mod.weight, mod.bias)
            else:
                dict_repr[layer_id] = (mod.weight,)
        return PVector(layer_collection, dict_repr=dict_repr)

    def copy_to_model(self, model):
        """
        Updates `model` parameter values with the current PVector

        Note. This is an inplace operation
        """
        dict_repr = self.get_dict_representation()
        layer_collection = LayerCollection.from_model(model)
        l_to_m, _ = layer_collection.get_layerid_module_maps(model)
        for layer_id, layer in layer_collection.layers.items():
            mod = l_to_m[layer_id]
            if layer.bias is not None:
                mod.bias.data.copy_(dict_repr[layer_id][1])
            mod.weight.data.copy_(dict_repr[layer_id][0])

    def add_to_model(self, model):
        """
        Updates `model` parameter values by adding the current PVector

        Note. This is an inplace operation
        """
        dict_repr = self.get_dict_representation()
        layer_collection = LayerCollection.from_model(model)
        l_to_m, _ = layer_collection.get_layerid_module_maps(model)
        for layer_id, layer in layer_collection.layers.items():
            mod = l_to_m[layer_id]
            if layer.bias is not None:
                mod.bias.data.add_(dict_repr[layer_id][1])
            mod.weight.data.add_(dict_repr[layer_id][0])

    @staticmethod
    def from_model_grad(model, ignore_layers=[]):
        """
        Creates a PVector using the current values of the `.grad`
        fields of parameters of the given model
        """
        dict_repr = dict()
        layer_collection = LLamaLayerCollection.from_model(model, True, ignore_layers)
        l_to_m, _ = layer_collection.get_layerid_module_maps(model)
        for layer_id, layer in layer_collection.layers.items():
            mod = l_to_m[layer_id]
            if layer.bias is not None:
                dict_repr[layer_id] = (mod.weight.grad, mod.bias.grad)
            else:
                dict_repr[layer_id] = (mod.weight.grad,)
        return PVector(layer_collection, dict_repr=dict_repr)
    
    # @staticmethod
    # def from_lc_grad(layer_collection):
    #     """
    #     Creates a PVector using the current values of the `.grad`
    #     fields of parameters of the given model
    #     """
    #     dict_repr = dict()
    #     layer_collection = LayerCollection.from_model(model)
    #     l_to_m, _ = layer_collection.get_layerid_module_maps(model)
    #     for layer_id, layer in layer_collection.layers.items():
    #         mod = l_to_m[layer_id]
    #         if layer.bias is not None:
    #             dict_repr[layer_id] = (mod.weight.grad, mod.bias.grad)
    #         else:
    #             dict_repr[layer_id] = (mod.weight.grad,)
    #     return PVector(layer_collection, dict_repr=dict_repr)

    def clone(self):
        """
        Returns a clone of the current object
        """
        if self.is_svd:
            raise NotImplementedError
        #     if self.dict_repr is not None:
        #         dict_clone = dict()
        #         for k, v in self.dict_repr.items():
        #             if len(v) == 2:
        #                 dict_clone[k] = ((v[0][0].clone(), v[0][1].clone(), v[0][2].clone()), v[1].clone())
        #             else:
        #                 dict_clone[k] = ((v[0][0].clone(), v[0][1].clone(), v[0][2].clone()), )
        #         return PVector(self.layer_collection, dict_repr=dict_clone)
            
        if self.dict_repr is not None:
            dict_clone = dict()
            for k, v in self.dict_repr.items():
                if len(v) == 2:
                    dict_clone[k] = (v[0].clone(), v[1].clone())
                else:
                    dict_clone[k] = (v[0].clone(),)
            return PVector(self.layer_collection, dict_repr=dict_clone)
        if self.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr.clone())

    def detach(self):
        """
        Detachs the current PVector from the computation graph
        """
        if self.dict_repr is not None:
            dict_detach = dict()
            for k, v in self.dict_repr.items():
                if len(v) == 2:
                    dict_detach[k] = (v[0].detach(), v[1].detach())
                else:
                    dict_detach[k] = (v[0].detach(),)
            return PVector(self.layer_collection, dict_repr=dict_detach)
        if self.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr.detach())

    def get_flat_representation(self):
        """
        Returns a Pytorch 1d tensor of the flatten vector.

        .. warning::
            The ordering in which the parameters are
            flattened can seem to be arbitrary. It is in fact
            the same ordering as specified by the ``layercollection.LayerCollection``
            object.

        :return: a Pytorch Tensor
        """
        if self.vector_repr is not None:
            return self.vector_repr
        elif self.dict_repr is not None:
            return self._dict_to_flat()
        else:
            return NotImplementedError

    def get_dict_representation(self):
        if self.dict_repr is not None:
            return self.dict_repr
        elif self.vector_repr is not None:
            return self._flat_to_dict()
        else:
            return NotImplementedError

    def _dict_to_flat(self):
        parts = []
        for layer_id, layer in self.layer_collection.layers.items():
            parts.append(self.dict_repr[layer_id][0].view(-1))
            if len(self.dict_repr[layer_id]) > 1:
                parts.append(self.dict_repr[layer_id][1].view(-1))
        return torch.cat(parts)

    def _flat_to_dict(self):
        dict_repr = dict()
        for layer_id, layer in self.layer_collection.layers.items():
            start = self.layer_collection.p_pos[layer_id]
            w = self.vector_repr[start:start+layer.weight.numel()] \
                    .view(*layer.weight.size)
            start += layer.weight.numel()
            if layer.bias is not None:
                b = self.vector_repr[start:start+layer.bias.numel()] \
                        .view(*layer.bias.size)
                start += layer.bias.numel()
                dict_repr[layer_id] = (w, b)
            else:
                dict_repr[layer_id] = (w,)
        return dict_repr

    def norm(self, p=2):
        """
        Computes the Lp norm of the PVector
        """
        if self.is_svd:
            if self.dict_repr is not None:
                sum_p = 0
                for l_id, l in self.layer_collection.layers.items():
                    sum_p += (self.dict_repr[l_id][0][1]**p).sum()
                    if l.bias is not None:
                        sum_p += (self.dict_repr[l_id][1]**p).sum()
                return sum_p ** (1/p)
        if self.dict_repr is not None:
            sum_p = 0
            for l_id, l in self.layer_collection.layers.items():
                sum_p += (self.dict_repr[l_id][0]**p).sum()
                if l.bias is not None:
                    sum_p += (self.dict_repr[l_id][1]**p).sum()
            return sum_p ** (1/p)
        else:
            return torch.norm(self.vector_repr, p=p)
    
    def square(self):
        if self.is_svd:
            raise NotImplementedError
        for l_id, l in self.layer_collection.layers.items():
            if l.bias is not None:
                self.dict_repr[l_id]=(self.dict_repr[l_id][0]**2, self.dict_repr[l_id][1]**2)
            else:
                self.dict_repr[l_id]=(self.dict_repr[l_id][0]**2, )

    def __rmul__(self, x):
        # TODO: test
        # scalar multiplication
        if self.dict_repr is not None:
            v_dict = dict()
            for l_id, l in self.layer_collection.layers.items():
                if l.bias:
                    v_dict[l_id] = (x * self.dict_repr[l_id][0],
                                    x * self.dict_repr[l_id][1])
                else:
                    v_dict[l_id] = (x * self.dict_repr[l_id][0],)
            return PVector(self.layer_collection, dict_repr=v_dict)
        else:
            return PVector(self.layer_collection,
                           vector_repr=x * self.vector_repr)

    def __add__(self, other):
        if self.dict_repr is not None and other.dict_repr is not None:
            v_dict = dict()
            for l_id, l in self.layer_collection.layers.items():
                if l.bias is not None:
                    v_dict[l_id] = (self.dict_repr[l_id][0] +
                                    other.dict_repr[l_id][0],
                                    self.dict_repr[l_id][1] +
                                    other.dict_repr[l_id][1])
                else:
                    v_dict[l_id] = (self.dict_repr[l_id][0] +
                                    other.dict_repr[l_id][0],)
            return PVector(self.layer_collection, dict_repr=v_dict)
        elif self.vector_repr is not None and other.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr+other.vector_repr)
        else:
            return PVector(self.layer_collection,
                           vector_repr=(self.get_flat_representation() +
                                        other.get_flat_representation()))

    def __sub__(self, other):
        
        if self.dict_repr is not None and other.dict_repr is not None:
            v_dict = dict()
            # print('in sub')
            for l_id, l in self.layer_collection.layers.items():
                if self.is_svd:
                    u,s,v=self.dict_repr[l_id][0]
                    vec1=u@torch.diag(s)@v.T
                else:
                    vec1=self.dict_repr[l_id][0]
                if other.is_svd:
                    u,s,v=other.dict_repr[l_id][0]
                    vec2=u@torch.diag(s)@v.T
                else:
                    vec2=other.dict_repr[l_id][0]

                if l.bias is not None:
                    v_dict[l_id] = (vec1 - vec2,
                                    self.dict_repr[l_id][1] -
                                    other.dict_repr[l_id][1])
                else:
                    v_dict[l_id] = (vec1 - vec2, )
            return PVector(self.layer_collection, dict_repr=v_dict)
        
        elif self.vector_repr is not None and other.vector_repr is not None:
            return PVector(self.layer_collection,
                           vector_repr=self.vector_repr-other.vector_repr)
        else:
            return PVector(self.layer_collection,
                           vector_repr=(self.get_flat_representation() -
                                        other.get_flat_representation()))
    def svd(self, q=32):
        self.is_svd=True
        for l_id, l in self.layer_collection.layers.items():
            if l.bias is not None:
                self.dict_repr[l_id]=(torch.svd_lowrank(self.dict_repr[l_id][0], q=q), self.dict_repr[l_id][1])
            else:
                self.dict_repr[l_id]=(torch.svd_lowrank(self.dict_repr[l_id][0], q=q),)
    
    def to(self, device):
        if self.is_svd:
            for l_id, l in self.layer_collection.layers.items():
                if l.bias is not None:
                    self.dict_repr[l_id]=((self.dict_repr[l_id][0][0].to(device), self.dict_repr[l_id][0][1].to(device), self.dict_repr[l_id][0][2].to(device)), self.dict_repr[l_id][1].to(device))
                else:
                    self.dict_repr[l_id]=((self.dict_repr[l_id][0][0].to(device), self.dict_repr[l_id][0][1].to(device), self.dict_repr[l_id][0][2].to(device)), )
        else:
            for l_id, l in self.layer_collection.layers.items():
                if l.bias is not None:
                    self.dict_repr[l_id]=(self.dict_repr[l_id][0].to(device), self.dict_repr[l_id][1].to(device))
                else:
                    self.dict_repr[l_id]=(self.dict_repr[l_id][0].to(device), )

    def dot_svd(self, other):
        assert self.is_svd and other.is_svd
        """
        Computes the dot product between `self` and `other`

        :param other: The other `PVector`
        """
        if self.vector_repr is not None or other.vector_repr is not None:
            raise NotImplementedError
            return torch.dot(self.get_flat_representation(),
                             other.get_flat_representation())
        else:
            dot_ = 0
            for l_id, l in self.layer_collection.layers.items():
                if l.bias is not None:
                    dot_ += torch.dot(self.dict_repr[l_id][1],
                                      other.dict_repr[l_id][1])
                u1, s1, v1=self.dict_repr[l_id][0]
                u2, s2, v2=other.dict_repr[l_id][0]
                # dot_ += torch.dot((u1@torch.diag(s1)@v1.T).view(-1),
                #                   (u2@torch.diag(s2)@v2.T).view(-1))
                dot_ += torch.trace(u1@(torch.diag(s1)@(v1.T@v2)@torch.diag(s2))@u2.T)
            return dot_

        
    def dot(self, other):
        """
        Computes the dot product between `self` and `other`

        :param other: The other `PVector`
        """
        if self.vector_repr is not None or other.vector_repr is not None:
            return torch.dot(self.get_flat_representation(),
                             other.get_flat_representation())
        else:
            dot_ = 0
            for l_id, l in self.layer_collection.layers.items():
                if l.bias is not None:
                    dot_ += torch.dot(self.dict_repr[l_id][1],
                                      other.dict_repr[l_id][1])
                dot_ += torch.dot(self.dict_repr[l_id][0].view(-1),
                                  other.dict_repr[l_id][0].view(-1))
            return dot_

    def size(self):
        """
        The size of the PVector, or equivalently the number of
        parameters of the layer collection
        """
        return (self.layer_collection.numel(), )


class FVector:
    """
    A vector in function space
    """
    def __init__(self, vector_repr=None):
        self.vector_repr = vector_repr

    def get_flat_representation(self):
        if self.vector_repr is not None:
            return self.vector_repr
        else:
            return NotImplementedError
