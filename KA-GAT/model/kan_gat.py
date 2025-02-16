import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
import numpy as np
import copy

from dgl.nn import SortPooling, WeightAndSum, GlobalAttentionPooling, Set2Set, SumPooling, AvgPooling, MaxPooling
from dgl.nn.functional import edge_softmax
from kan import KAN


device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")



class KAN_linear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=False):
        super(KAN_linear,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self,x):

        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        #This should be fused to avoid materializing memory
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        if self.addbias:
            y += self.bias
        
        y = y.view(outshape)
        return y
    


'''
class MultiHeadTransKANBlock(nn.Module):
    def __init__(self, in_node_size, in_edge_size, out_size, grid_size, num_heads):
        super(MultiHeadTransKANBlock, self).__init__()
        self.in_node_size = in_node_size
        self.in_edge_size = in_edge_size
        self.out_size = out_size // num_heads  # Assume out_size is divisible by num_heads
        self.grid_size = grid_size
        self.num_heads = num_heads

        # KAN linear layers for each head
        self.K_v2v = nn.ModuleList([KAN_linear(in_node_size, self.out_size, grid_size) for _ in range(num_heads)])
        self.K_e2v = nn.ModuleList([KAN_linear(in_edge_size, self.out_size, grid_size) for _ in range(num_heads)])
        self.V_v2v = nn.ModuleList([KAN_linear(in_node_size, self.out_size, grid_size) for _ in range(num_heads)])
        self.V_e2v = nn.ModuleList([KAN_linear(in_edge_size, self.out_size, grid_size) for _ in range(num_heads)])
        
        self.linear_update = nn.ModuleList([KAN_linear(self.out_size * 2, self.out_size * 2, grid_size, addbias=True) for _ in range(num_heads)])
        self.layernorm_node = nn.LayerNorm(in_node_size)
        self.layernorm_edge = nn.LayerNorm(in_edge_size)
        self.sigmoid = nn.Sigmoid()
        self.msg_layer = nn.ModuleList([nn.Sequential(KAN_linear(self.out_size * 2, self.out_size, grid_size, addbias=True), nn.LayerNorm(self.out_size)) for _ in range(num_heads)])

    def propagate(self, g, x, edge_feature):
        with g.local_scope():
            residual_x = x  # Save for residual connection
            multi_head_outputs = []
            for i in range(self.num_heads):
                g.ndata[f'K_v{i}'] = self.K_v2v[i](x)
                g.ndata[f'V_v{i}'] = self.V_v2v[i](x)
                g.edata[f'K_E{i}'] = self.K_e2v[i](edge_feature)
                g.edata[f'V_E{i}'] = self.V_e2v[i](edge_feature)
                
                g.apply_edges(lambda edges: self.message_func(edges, i))
                g.update_all(fn.copy_e(f'h{i}', 'm'), fn.sum('m', f'h{i}'))
                multi_head_outputs.append(g.ndata[f'h{i}'])

            #print(len(multi_head_outputs))
            #print(multi_head_outputs[0].shape)
            stacked_tensors = torch.stack(multi_head_outputs)

            # 计算堆叠张量的均值，沿着新创建的堆叠维度（0维）
            mean_tensor = torch.mean(stacked_tensors, dim=0)

            multi_head_output = mean_tensor + residual_x
            return multi_head_output



    def message_func(self, edges, head_idx):
        query_i = torch.cat([edges.src[f'K_v{head_idx}'], edges.dst[f'K_v{head_idx}']], dim=1)
        key_j = torch.cat([edges.data[f'V_E{head_idx}'], edges.data[f'K_E{head_idx}']], dim=1)
        alpha = (query_i * key_j) / math.sqrt(self.out_size * 2)
        alpha = F.softmax(alpha, dim=-1)
        out = torch.cat([edges.dst[f'V_v{head_idx}'], edges.src[f'V_v{head_idx}']], dim=1)
        out = self.linear_update[head_idx](out) * self.sigmoid(alpha)
        return {f'h{head_idx}': F.leaky_relu(self.msg_layer[head_idx](out))}

'''


class Gat_Kan_layer(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats, num_heads, grid_size, bias=True):
        super(Gat_Kan_layer, self).__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats+in_edge_feats, out_node_feats * num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats * num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        self.output_node = KAN(width=[out_node_feats,5,out_node_feats], grid=grid_size, k=3, seed=0)
        self.output_edge = KAN(width=[out_edge_feats,5,out_edge_feats], grid=grid_size, k=3, seed=0)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc_node.weight)
        nn.init.xavier_normal_(self.fc_ni.weight)
        nn.init.xavier_normal_(self.fc_fij.weight)
        nn.init.xavier_normal_(self.fc_nj.weight)
        nn.init.xavier_normal_(self.attn)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    '''
        self.fc_node = KAN_linear(in_node_feats+in_edge_feats, out_node_feats * num_heads, grid_size, addbias=True)
        self.fc_ni = KAN_linear(in_node_feats, out_edge_feats * num_heads, grid_size)
        self.fc_fij = KAN_linear(in_edge_feats, out_edge_feats * num_heads, grid_size)
        self.fc_nj = KAN_linear(in_node_feats, out_edge_feats * num_heads, grid_size)
        self.attn = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_edge_feats)))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.attn)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    '''
    def message_func(self, edges):
        return {'feat': edges.data['feat']}

    def reduce_func(self, nodes):
        num_edges = nodes.mailbox['feat'].size(1)  
        agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  
        return {'agg_feats': agg_feats}
    

    def forward(self, graph, nfeats, efeats, get_attention=False):
        with graph.local_scope():
            graph.ndata['feat'] = nfeats
            graph.edata['feat'] = efeats
            in_degrees = graph.in_degrees().float().unsqueeze(-1)
            in_degrees[in_degrees == 0] = 1 
            f_ni = self.fc_ni(nfeats)# in_node_feats --> out_edge_feats
            f_nj = self.fc_nj(nfeats)# in_node_feats --> out_edge_feats
            f_fij = self.fc_fij(efeats)# in_edge_feats --> out_edge_feats

            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            
            f_out = graph.edata.pop('f_tmp') + f_fij
            if self.bias is not None:
                f_out = f_out + self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)

            graph.send_and_recv(graph.edges(), self.message_func, reduce_func=self.reduce_func)
            m_feats = torch.cat((graph.ndata['feat'],graph.ndata['agg_feats']),dim=1)
            
            graph.edata['a'] = edge_softmax(graph, e)
            
            graph.ndata['h_out'] = self.fc_node(m_feats).view(-1, self._num_heads, self._out_node_feats)
            
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))

            h_out = nn.functional.leaky_relu(graph.ndata['h_out'])
            h_out = h_out.view(-1, self._num_heads, self._out_node_feats)

            h_out = torch.sum(h_out, dim=1)
            f_out = torch.sum(f_out, dim=1)

            out_n = self.output_node(h_out)
            out_e = self.output_edge(f_out)
            if get_attention:
                return out_n, out_e, graph.edata.pop('a')
            else:
                return out_n, out_e



class KAN_GAT(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, hidden_dim, out_1, out_2, gride_size, head, layer_num, pooling):
        super(KAN_GAT, self).__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.hidden_dim = hidden_dim
        self.out_1 = out_1
        self.out_2 = out_2
        self.head = head
        self.layer = layer_num

        self.grid_size = gride_size
        self.pooling = pooling

        self.node_kan_line = KAN_linear(in_node_dim, hidden_dim, gride_size, addbias=False)
        self.edge_kan_line = KAN_linear(in_edge_dim, hidden_dim, gride_size, addbias=False)

        self.attentions = nn.ModuleList()
        

        
        self.attentions.append(Gat_Kan_layer(in_node_feats=in_node_dim,in_edge_feats=in_edge_dim,
                                             out_node_feats=hidden_dim,out_edge_feats=hidden_dim,
                                             num_heads=self.head,grid_size=self.grid_size))
        
        for _ in range(self.layer-1):
            self.attentions.append(Gat_Kan_layer(in_node_feats=hidden_dim,in_edge_feats=hidden_dim,
                                                 out_node_feats=hidden_dim,out_edge_feats=hidden_dim,
                                                 num_heads=self.head,grid_size=self.grid_size))

        self.leaky_relu = nn.LeakyReLU()
        self.sumpool = SumPooling()
        self.avgpool = AvgPooling()
        self.maxpool = MaxPooling()


        out_layers = [
            #KAN_linear(hidden_dim, out_1, gride_size, addbias=False),
            KAN(width=[hidden_dim,5,out_1], grid=gride_size, k=3, seed=0),
            self.leaky_relu,
            #KAN_linear(out_1, out_2, gride_size, addbias=True),
            KAN(width=[out_1,5,out_2], grid=gride_size, k=3, seed=0),
            nn.Sigmoid()
        ]
        self.Readout = nn.Sequential(*out_layers)



    def forward(self, g, node_feature, edge_feature):
        
        '''
        hidden_v = self.node_kan_line(node_feature)
        node_feature = F.leaky_relu(hidden_v)

        hidden_e = self.edge_kan_line(edge_feature)
        edge_feature = F.leaky_relu(hidden_e)
        '''
        for i in range(len(self.attentions)):
            atten = self.attentions[i]
            #node_feature, edge_feature = atten(g, node_feature, edge_feature)
                
            #hidden_v = node_feature.clone().detach()
            #hidden_e = edge_feature.clone().detach()
            node_feature, edge_feature = atten(g, node_feature, edge_feature)

            #node_feature = F.leaky_relu(torch.add(node_feature, hidden_v))
            #edge_feature = F.leaky_relu(torch.add(edge_feature, hidden_e))
            
        
        
        out1 = F.leaky_relu(node_feature)

        if self.pooling == 'avg':
            y = self.avgpool(g, out1)
            
        elif self.pooling == 'max':
            y = self.maxpool(g, out1)
            
        elif self.pooling == 'sum':
            y = self.sumpool(g, out1)
            
        else:
            print('No pooling found!!!!')
        
        out = self.Readout(y)
        
        

        return out
