
from pickle import FROZENSET
from time import time
import torch
import netron
import torch.onnx
from torch.utils import data
from network import Detector
from config import Config
import os
import argparse
# from torchinfo import summary
from torchsummary import summary
import onnx
from onnxsim import simplify
import onnxruntime as ort
import numpy as np
import open3d as o3d
import tensorrt as trt


class model2onnx():
    def __init__(self, torch_file, onnx_file, cfg=None):
        super(model2onnx, self).__init__()
        self.cfg = cfg
        self.torch_file = torch_file
        self.onnx_file = onnx_file
        torch.manual_seed(2021)
        torch.cuda.manual_seed_all(2021)
        np.random.seed(2021)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.input = torch.randn(1,3,1152,1152, dtype=torch.float32)
        # print(self.input) 

    def torch_infer(self, torch_model=None):
        model = Detector()
        model.to(self.device)
        print(model)
        torch_model = torch_model
        checkpoint = torch.load(torch_model, map_location=self.device) 

        model.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint['net'].items()})
        model.eval()     
        points = self.input.to(self.device)

        from thop import profile
        flops, params = profile(model, inputs=(points,))
        print('thop flops:{} GFLOPs, params:{} M'.format(flops/1e9, params/1e6))
        result, fea = model(points)
        print("torch model_out: ",result.shape)
        # np.savetxt('./out/torch_out.txt', result.cpu().detach().numpy())
        # with open('./out/torch_out.txt', 'w') as f:
        #     f.write(str(result.cpu().detach().numpy()))
        return model, result, fea
    
    def export_onnx(self):
        torch.set_default_tensor_type('torch.FloatTensor')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        input = self.input.to(self.device)
        export_onnx_file = self.onnx_file
        model, _ = self.torch_infer(self.torch_file)
        # dynamic_axes = {"input", [0, 1]}
        torch.onnx.export(model,
                        input,
                        export_onnx_file,
                        opset_version=11,
                        export_params=True, 
                        verbose=True,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],)
                        # dynamic_axes=dynamic_axes)

        onnx_model = onnx.load_model(export_onnx_file)
        onnx_model_simp, check = simplify(onnx_model)
        assert check, "Simplified Onnx model could not be validated."
        onnx_simp_path = './onnx_simp.onnx'
        onnx.save_model(onnx_model_simp, onnx_simp_path)

        print("export onnx model finished...")

    def extract_model(self, onnx_file):
        submodel = './submodel_b1.onnx'
        onnx.utils.extract_model(onnx_file, submodel, ['1153'], ['1154'])
        # self.netron_onnx(submodel)

    def infer_onnx(self, onnx_file=None, input=None):
        '''
        args:
            onnx_file="./*.onnx"
            input: numpy.ndarray
        return
            tuple
        '''
        # onnx模型推理
        providers = ort.get_available_providers() 
        # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        print(providers)
        EP_list_tensorrt = [('TensorrtExecutionProvider', 
                            {'trt_max_workspace_size':4294967296, 
                            'trt_fp16_enable':True,
                            'trt_engine_cache_enable':True,
                            'trt_engine_cache_path':'./trt_cache'}
                            ), 
                            ('CUDAExecutionProvider', 
                            {'device_id':0})]

        EP_list_cuda = ['CUDAExecutionProvider']
        cpu_providers = ['CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 0
        sess_options.log_verbosity_level = 1
        # sess_options.enable_profiling = True
        ort_seeion = ort.InferenceSession(onnx_file, sess_options=sess_options, providers=EP_list_cuda)
        input_name = ort_seeion.get_inputs()[0].name
        output_name = ort_seeion.get_outputs()[0].name
        # input = np.random.randn(1, 3, 1152, 1152).astype(np.float32) # 必须np.float32
        # print('onnx input: ', self.input)
        inputs = {input_name: input}
        output = ort_seeion.run(None, inputs) # 若为None则按顺序输出所有的output,即返回[output_0, output_1]
        # print(output)
        # np.savetxt('./out/onnx_out.txt', output)
        # with open('./out/onnx_out.txt', 'w') as f:
        #     f.write(str(output[0]))

        
        return output

    def check_onnx(self, onnx_file):
        # 验证onnx模型格式是否正确 
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)
        onnx.helper.printable_graph(onnx_model.graph)  # 输出计算图
        print("finished output compute graph")
    
    def assert_allclose(self, torch_file=None, onnx_file=None):

        _, torch_out, fea = self.torch_infer(torch_file)
        onnx_out = self.infer_onnx(onnx_file, fea.cpu().detach().numpy())
        # with open('./out/torch_backbone_mlp_out.txt', 'w') as f:
        #     f.write(str(torch_out.cpu().detach().numpy()))
        # with open('./out/onnx_backbone_mlp_out.txt', 'w') as f:
        #     f.write(str(onnx_out[0]))
        
        torch_out = torch_out.cpu().detach().numpy()
        onnx_out = onnx_out[0]
        print(np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-03, atol=1e-05))
        
        
        print("torch model and onnx model have the same output...")
    
    def netron_onnx(self, onnx_file):
        netron.start(onnx_file)

def main():
    # import faulthandler
    # faulthandler.enable()
    ckpt_path = "/workspace/work_dir/K-Lane/ckpt/0903_p28_t3_seg.pth"
    onnx_path = '../ckpt/0903_p28_t3_seg.onnx' 
    onnx_simp_path = "../ckpt/0903_p28_t3_seg_simp.onnx"
    
    model = model2onnx(ckpt_path, onnx_simp_path)
    model.assert_allclose(torch_file=ckpt_path, onnx_file='/workspace/work_dir/K-Lane/submodel_b1.onnx')

if __name__=='__main__':

    def parse_args():
        parser = argparse.ArgumentParser(description="Train klane for dist")
        parser.add_argument('--gpus', default='1', type=str)
        parser.add_argument('--local_rank', default=1, type=int)
        args = parser.parse_args()
        return args
    args = parse_args()   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    main()
    