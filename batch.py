import torch 
from torch_geometric.data import Data, Batch

class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                elif key  == 'connected_edge_indices':
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += data.edge_index.shape[1]

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchAE(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchAE, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchAE()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0

        for i, data in enumerate(data_list):
            num_nodes = data.num_nodes
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys:
                item = data[key]
                if key in ['edge_index', 'negative_edge_index']:
                    item = item + cumsum_node
                batch[key].append(item)

            cumsum_node += num_nodes

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.__cat_dim__(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

    def __cat_dim__(self, key):
        return -1 if key in ["edge_index", "negative_edge_index"] else 0
class BatchSubstructContext(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstructContext, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys

        batch = BatchSubstructContext()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]

        for key in keys:
            #print(key)
            batch[key] = []

        #batch.batch = []
        #used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        
        for data in data_list:
            #If there is no context, just skip!!
            if hasattr(data, "x_context"):
                # num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                #batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(torch.full((len(data.overlap_context_substruct_idx), ), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ###batching for the main graph
                #for key in data.keys:
                #    if not "context" in key and not "substruct" in key:
                #        item = data[key]
                #        item = item + cumsum_main if batch.cumsum(key, item) else item
                #        batch[key].append(item)
                
                ###batching for the substructure graph
                for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                    item = data[key]
                    item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    batch[key].append(item) 
                

                ###batching for the context graph
                for key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                    item = data[key]
                    item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                # cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct   
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(
                batch[key], dim=batch.__cat_dim__(key))
        #batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)
        # print(batch)
        return batch.contiguous()

    def __cat_dim__(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx", "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchSubstruct(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstruct, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys
        batch = BatchSubstruct()
        keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "edge_attr_pos_substruct", "edge_index_pos_substruct", "x_pos_substruct","center_pos_substruct_idx","batch","pos_batch"]

        for key in keys:
            batch[key] = []

        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        for data in data_list:
            
            # num_nodes = data.num_nodes
            num_nodes_substruct = len(data.x_substruct)
            num_nodes_context = len(data.x_pos_substruct)

            for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                item = data[key]
                item = item + cumsum_substruct if batch.cumsum(key, item) else item
                batch[key].append(item)

                ###batching for the context graph
            for key in ["edge_attr_pos_substruct", "center_pos_substruct_idx","edge_index_pos_substruct", "x_pos_substruct"]:
                item = data[key]
                item = item + cumsum_context if batch.cumsum(key, item) else item
                batch[key].append(item)
            
            # batch
            for _  in range(len(data["x_substruct"])):
                batch['batch'].append(torch.tensor(i))

            for _  in range(len(data["x_pos_substruct"])):
                batch['pos_batch'].append(torch.tensor(i))

            # cumsum_main += num_nodes
            cumsum_substruct += num_nodes_substruct   
            cumsum_context += num_nodes_context
            i = i + 1

        for key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "edge_attr_pos_substruct", "edge_index_pos_substruct", "x_pos_substruct","center_pos_substruct_idx"]:
            batch[key] = torch.cat(
                batch[key], dim=batch.__cat_dim__(key))

        batch['batch'] = torch.tensor(batch['batch']) 
        batch['pos_batch'] = torch.tensor(batch['pos_batch']) 
        return batch.contiguous()

    def __cat_dim__(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_pos_substruct"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct", "edge_index_pos_substruct", "center_substruct_idx","center_pos_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

class BatchSubstruct_pos_neg(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstruct_pos_neg, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys
        batch = BatchSubstruct_pos_neg()
        keys = ["center_substruct_idx1", "edge_attr_substruct1", "edge_index_substruct1", "x_substruct1",
                "center_substruct_idx2", "edge_attr_substruct2", "edge_index_substruct2", "x_substruct2",
                "edge_attr_pos_substruct", "edge_index_pos_substruct", "x_pos_substruct","center_pos_substruct_idx",
                "edge_attr_neg_substruct", "edge_index_neg_substruct", "x_neg_substruct","center_neg_substruct_idx",
                "batch1","batch2","pos_batch","neg_batch"]

        for key in keys:
            batch[key] = []

        cumsum_substruct1 = 0
        cumsum_substruct2 = 0
        cumsum_context1 = 0
        cumsum_context2 = 0

        i = 0
        for data in data_list:
            
            # num_nodes = data.num_nodes
            num_nodes_substruct1 = len(data.x_substruct1)
            num_nodes_context1 = len(data.x_pos_substruct)
            num_nodes_substruct2 = len(data.x_substruct2)
            num_nodes_context2 = len(data.x_neg_substruct)

            for key in ["center_substruct_idx1", "edge_attr_substruct1", "edge_index_substruct1", "x_substruct1"]:
                item = data[key]
                item = item + cumsum_substruct1 if batch.cumsum(key, item) else item
                batch[key].append(item)
            
            for key in ["center_substruct_idx2", "edge_attr_substruct2", "edge_index_substruct2", "x_substruct2"]:
                item = data[key]
                item = item + cumsum_substruct2 if batch.cumsum(key, item) else item
                batch[key].append(item)

                ###batching for the context graph
            for key in ["edge_attr_pos_substruct", "center_pos_substruct_idx","edge_index_pos_substruct", "x_pos_substruct"]:
                item = data[key]
                item = item + cumsum_context1 if batch.cumsum(key, item) else item
                batch[key].append(item)

            for key in ["edge_attr_neg_substruct", "center_neg_substruct_idx","edge_index_neg_substruct", "x_neg_substruct"]:
                item = data[key]
                item = item + cumsum_context2 if batch.cumsum(key, item) else item
                batch[key].append(item)
            
            # batch
            for _  in range(len(data["x_substruct1"])):
                batch['batch1'].append(torch.tensor(i))
            for _  in range(len(data["x_substruct2"])):
                batch['batch2'].append(torch.tensor(i))

            for _  in range(len(data["x_pos_substruct"])):
                batch['pos_batch'].append(torch.tensor(i))
            for _  in range(len(data["x_neg_substruct"])):
                batch['neg_batch'].append(torch.tensor(i))

            # cumsum_main += num_nodes
            cumsum_substruct1 += num_nodes_substruct1  
            cumsum_context1 += num_nodes_context1
            cumsum_substruct2 += num_nodes_substruct2   
            cumsum_context2 += num_nodes_context2
            i = i + 1

        for key in ["center_substruct_idx1", "edge_attr_substruct1", "edge_index_substruct1", "x_substruct1",
                    "center_substruct_idx2", "edge_attr_substruct2", "edge_index_substruct2", "x_substruct2",
                     "edge_attr_pos_substruct", "edge_index_pos_substruct", "x_pos_substruct","center_pos_substruct_idx",
                     "edge_attr_neg_substruct", "edge_index_neg_substruct", "x_neg_substruct","center_neg_substruct_idx"]:
            batch[key] = torch.cat(
                batch[key], dim=batch.__cat_dim__(key))

        batch['batch1'] = torch.tensor(batch['batch1']) 
        batch['pos_batch'] = torch.tensor(batch['pos_batch']) 
        batch['batch2'] = torch.tensor(batch['batch2']) 
        batch['neg_batch'] = torch.tensor(batch['neg_batch']) 
        return batch.contiguous()

    def __cat_dim__(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct1", "edge_index_substruct2","edge_index_pos_substruct","edge_index_neg_substruct"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct1", "edge_index_substruct2","edge_index_pos_substruct","edge_index_neg_substruct", "center_substruct_idx1","center_pos_substruct_idx","center_substruct_idx2","center_neg_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class BatchSubstruct_pos_neg_pair(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    """
    Specialized batching for substructure context pair!
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchSubstruct_pos_neg, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        #keys = [set(data.keys) for data in data_list]
        #keys = list(set.union(*keys))
        #assert 'batch' not in keys
        batch = BatchSubstruct_pos_neg()
        keys = ["center_substruct_idx1_1", "edge_attr_substruct1_1", "edge_index_substruct1_1", "x_substruct1_1",
                "center_substruct_idx1_2", "edge_attr_substruct1_2", "edge_index_substruct1_2", "x_substruct1_2",
                "center_substruct_idx2_1", "edge_attr_substruct2_1", "edge_index_substruct2_1", "x_substruct2_1",
                "center_substruct_idx2_2", "edge_attr_substruct2_2", "edge_index_substruct2_2", "x_substruct2_2",
                "edge_attr_pos_substruct1", "edge_index_pos_substruct1", "x_pos_substruct1","center_pos_substruct_idx1",
                "edge_attr_pos_substruct2", "edge_index_pos_substruct2", "x_pos_substruct2","center_pos_substruct_idx2",
                "edge_attr_neg_substruct1", "edge_index_neg_substruct1", "x_neg_substruct1","center_neg_substruct_idx1",
                "edge_attr_neg_substruct2", "edge_index_neg_substruct2", "x_neg_substruct2","center_neg_substruct_idx2",
                "batch1_1","batch2_1","pos_batch1","neg_batch1"
                "batch1_2","batch2_2","pos_batch2","neg_batch2"
                ]

        for key in keys:
            batch[key] = []

        cumsum_substruct1_1 = 0
        cumsum_substruct2_1 = 0
        cumsum_substruct1_2 = 0
        cumsum_substruct2_2 = 0
        cumsum_context1_1 = 0
        cumsum_context2_1 = 0
        cumsum_context1_2 = 0
        cumsum_context2_2 = 0

        i = 0
        for data in data_list:
            
            # num_nodes = data.num_nodes
            num_nodes_substruct1_1 = len(data.x_substruct1_1)
            num_nodes_context1_1 = len(data.x_pos_substruct1)
            num_nodes_substruct2_1 = len(data.x_substruct2_1)
            num_nodes_context2_1 = len(data.x_neg_substruct1)
            num_nodes_substruct1_2 = len(data.x_substruct1_2)
            num_nodes_context1_2 = len(data.x_pos_substruct2)
            num_nodes_substruct2_2 = len(data.x_substruct2_2)
            num_nodes_context2_2 = len(data.x_neg_substruct2)

            for key in ["center_substruct_idx1_1", "edge_attr_substruct1_1", "edge_index_substruct1_1", "x_substruct1_1"]:
                item = data[key]
                item = item + cumsum_substruct1_1 if batch.cumsum(key, item) else item
                batch[key].append(item)
            
            for key in ["center_substruct_idx1_2", "edge_attr_substruct1_2", "edge_index_substruct1_2", "x_substruct1_2"]:
                item = data[key]
                item = item + cumsum_substruct1_2 if batch.cumsum(key, item) else item
                batch[key].append(item)
            
            for key in ["center_substruct_idx2_1", "edge_attr_substruct2_1", "edge_index_substruct2_1", "x_substruct2_1"]:
                item = data[key]
                item = item + cumsum_substruct2_1 if batch.cumsum(key, item) else item
                batch[key].append(item)

            for key in ["center_substruct_idx2_2", "edge_attr_substruct2_2", "edge_index_substruct2_2", "x_substruct2_2"]:
                item = data[key]
                item = item + cumsum_substruct2_2 if batch.cumsum(key, item) else item
                batch[key].append(item)

                ###batching for the context graph
            for key in ["edge_attr_pos_substruct1", "center_pos_substruct_idx1","edge_index_pos_substruct1", "x_pos_substruct1"]:
                item = data[key]
                item = item + cumsum_context1_1 if batch.cumsum(key, item) else item
                batch[key].append(item)
            
            for key in ["edge_attr_pos_substruct2", "center_pos_substruct_idx2","edge_index_pos_substruct2", "x_pos_substruct2"]:
                item = data[key]
                item = item + cumsum_context1_2 if batch.cumsum(key, item) else item
                batch[key].append(item)

            for key in ["edge_attr_neg_substruct1", "center_neg_substruct_idx1","edge_index_neg_substruct1", "x_neg_substruct1"]:
                item = data[key]
                item = item + cumsum_context2_1 if batch.cumsum(key, item) else item
                batch[key].append(item)
            
            for key in ["edge_attr_neg_substruct2", "center_neg_substruct_idx2","edge_index_neg_substruct2", "x_neg_substruct2"]:
                item = data[key]
                item = item + cumsum_context2_2 if batch.cumsum(key, item) else item
                batch[key].append(item)
            
            # batch
            for _  in range(len(data["x_substruct1_1"])):
                batch['batch1_1'].append(torch.tensor(i))
            for _  in range(len(data["x_substruct1_2"])):
                batch['batch1_2'].append(torch.tensor(i))
            for _  in range(len(data["x_substruct2_1"])):
                batch['batch2_1'].append(torch.tensor(i))
            for _  in range(len(data["x_substruct2_2"])):
                batch['batch2_2'].append(torch.tensor(i))


            for _  in range(len(data["x_pos_substruct1"])):
                batch['pos_batch1'].append(torch.tensor(i))
            for _  in range(len(data["x_pos_substruct2"])):
                batch['pos_batch2'].append(torch.tensor(i))
            for _  in range(len(data["x_neg_substruct1"])):
                batch['neg_batch1'].append(torch.tensor(i))
            for _  in range(len(data["x_neg_substruct2"])):
                batch['neg_batchs2'].append(torch.tensor(i))

            # cumsum_main += num_nodes
            cumsum_substruct1_1 += num_nodes_substruct1_1 
            cumsum_substruct1_2 += num_nodes_substruct1_2   
            cumsum_context1_1 += num_nodes_context1_1
            cumsum_context1_2 += num_nodes_context1_2
            cumsum_substruct2_1 += num_nodes_substruct2_1   
            cumsum_context2_1 += num_nodes_context2_1
            cumsum_substruct2_2 += num_nodes_substruct2_2   
            cumsum_context2_2 += num_nodes_context2_2
            i = i + 1

        for key in ["center_substruct_idx1_1", "edge_attr_substruct1_1", "edge_index_substruct1_1", "x_substruct1_1",
                    "center_substruct_idx1_2", "edge_attr_substruct1_2", "edge_index_substruct1_2", "x_substruct1_2",
                    "center_substruct_idx2_1", "edge_attr_substruct2_1", "edge_index_substruct2_1", "x_substruct2_1",
                    "center_substruct_idx2_2", "edge_attr_substruct2_2", "edge_index_substruct2_2", "x_substruct2_2",
                    "edge_attr_pos_substruct1", "edge_index_pos_substruct1", "x_pos_substruct1","center_pos_substruct_idx1",
                    "edge_attr_pos_substruct2", "edge_index_pos_substruct2", "x_pos_substruct2","center_pos_substruct_idx2",
                    "edge_attr_neg_substruct1", "edge_index_neg_substruct1", "x_neg_substruct1","center_neg_substruct_idx1",
                    "edge_attr_neg_substruct2", "edge_index_neg_substruct2", "x_neg_substruct2","center_neg_substruct_idx2"]:
            batch[key] = torch.cat(
                batch[key], dim=batch.__cat_dim__(key))

        batch['batch1_1'] = torch.tensor(batch['batch1_1']) 
        batch['batch1_2'] = torch.tensor(batch['batch1_2']) 
        batch['pos_batch1'] = torch.tensor(batch['pos_batch1'])
        batch['pos_batch2'] = torch.tensor(batch['pos_batch2']) 
        batch['batch2_1'] = torch.tensor(batch['batch2_1']) 
        batch['batch2_2'] = torch.tensor(batch['batch2_2'])
        batch['neg_batch1'] = torch.tensor(batch['neg_batch1']) 
        batch['neg_batch2'] = torch.tensor(batch['neg_batch2']) 
        return batch.contiguous()

    def __cat_dim__(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct1_1", "edge_index_substruct2_1","edge_index_pos_substruct1","edge_index_neg_substruct1","edge_index_substruct1_2", "edge_index_substruct2_2","edge_index_pos_substruct2","edge_index_neg_substruct2"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ["edge_index", "edge_index_substruct1_1", "edge_index_substruct2_1","edge_index_pos_substruct1","edge_index_neg_substruct1", "center_substruct_idx1_1","center_pos_substruct_idx1","center_substruct_idx2_1","center_neg_substruct_idx1",
                      "edge_index", "edge_index_substruct1_2", "edge_index_substruct2_2","edge_index_pos_substruct2","edge_index_neg_substruct2", "center_substruct_idx1_2","center_pos_substruct_idx2","center_substruct_idx2_2","center_neg_substruct_idx2"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1

