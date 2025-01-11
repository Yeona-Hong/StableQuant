import tensorrt as trt
import pycuda.driver as cuda
from dataset import AudioDataset
from fairseq.data.dictionary import Dictionary
import os, torch, jiwer, time, torchaudio
from tqdm import tqdm
import numpy as np
import pycuda.autoinit
from decoders.viterbi_decoder import ViterbiDecoder

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_engine(engine_filepath, trt_logger):
    with open(engine_filepath, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def setup_context(engine):
    return engine.create_execution_context()


def load_data(data_path):
    dataset = AudioDataset(data_path) #audio, label
    return dataset

def load_dictionary(data_dir):
    dictionary = Dictionary.load(os.path.join(data_dir, "dict.ltr.txt"))
    return dictionary


def process_text(text):
    text = text.replace(' ', '')
    text = text.replace('|', ' ')
    return ''.join(text)


if __name__ == "__main__":
    
    data_path = './tsv/test.tsv'
    dataset = load_data(data_path)
        
    # Load the dictionary
    data_dir = './dictionary'  # 실제 경로로 변경
    dictionary = load_dictionary(data_dir)
    
    batch_size = 1
    max_seq_length = 600000
    
    engine_path = "./engine/hubert_all.engine"
    
    with open(engine_path, "rb") as f, \
    trt.Runtime(TRT_LOGGER) as runtime, \
    runtime.deserialize_cuda_engine(f.read()) as engine, \
    engine.create_execution_context() as context:
        stream = cuda.Stream()
        print("\nRunning Inference...")
        
        all_hypotheses = []
        all_references = []

        for idx, (x,y) in tqdm(enumerate(dataset)):
            input_shape = x.size()
            input_nbytes = trt.volume(input_shape) * trt.float32.itemsize
            d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(1)]
            tensor_name = engine.get_tensor_name(0)
            context.set_input_shape(tensor_name, input_shape)

            h_output = cuda.pagelocked_empty(tuple(context.get_tensor_shape(engine.get_tensor_name(1))), dtype=np.float32)
            d_output = cuda.mem_alloc(h_output.nbytes)

            # Copy inputs
            x = cuda.register_host_memory(np.ascontiguousarray(x.ravel()))
            cuda.memcpy_htod_async(d_inputs[0], x, stream)
            
            bindings = [int(d_inputs[i]) for i in range(1)] + [int(d_output)]
            
            for i in range(engine.num_io_tensors):
               context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
               
            # Run inference
            context.execute_async_v3(stream_handle=stream.handle)
            # Synchronize the stream
            stream.synchronize()
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            
            logits = np.array(h_output)
            logits = torch.Tensor(logits)
            logits = logits.unsqueeze(0)
            
            
            # Apply log-softmax to get log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
             # Viterbi decoding using the ViterbiDecoder
            viterbi_decoder = ViterbiDecoder(tgt_dict=dictionary)  # tgt_dict 인수 전달
            decoded_results = viterbi_decoder.decode(log_probs)
            
            decoded_texts = []
            for result in decoded_results:
                tokens = result[0]['tokens'].tolist()
                decoded_text = dictionary.string(tokens)
                decoded_text = process_text(decoded_text)
                decoded_texts.append(decoded_text)

            if decoded_texts[0][-1] == ' ':
                decoded_texts[0] = decoded_texts[0][:-1]

            all_hypotheses.append(decoded_texts[0])
            all_references.append(y)
    wer = jiwer.wer(all_references, all_hypotheses)
    print(f"Word Error Rate (WER): {round(wer*100, 2)}")
