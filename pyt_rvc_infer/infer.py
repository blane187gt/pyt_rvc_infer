import os
import shutil
import gc
import torch
import argparse
import requests
from multiprocessing import cpu_count
from pyt_rvc_infer.lib.modules import VC
from pyt_rvc_infer.lib.split_audio import split_silence_nonsilent, adjust_audio_lengths, combine_silence_nonsilent

class Configs:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max



def get_model(voice_model):
    model_dir = os.path.join(os.getcwd(), "models", voice_model)
    model_filename, index_filename = None, None
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            model_filename = file
        if ext == '.index':
            index_filename = file

    if model_filename is None:
        print(f'No model file exists in {model_dir}.')
        return None, None

    return os.path.join(model_dir, model_filename), os.path.join(model_dir, index_filename) if index_filename else ''

def infer_audio(
    model_name,
    audio_path,
    f0_change=0,
    f0_method="rmvpe",
    min_pitch="50",
    max_pitch="1100",
    crepe_hop_length=128,
    index_rate=0.75,
    filter_radius=3,
    rms_mix_rate=0.25,
    protect=0.33,
    split_infer=False,
    min_silence=500,
    silence_threshold=-50,
    seek_step=1,
    keep_silence=100,
    do_formant=False,
    quefrency=0,
    timbre=1,
    f0_autotune=False,
    audio_format="wav",
    resample_sr=0,
    hubert_model_path="hubert_base.pt",
    rmvpe_model_path="rmvpe.pt",
    fcpe_model_path="fcpe.pt"
):
    os.environ["rmvpe_model_path"] = rmvpe_model_path
    os.environ["fcpe_model_path"] = fcpe_model_path
    configs = Configs('cuda:0', True)
    vc = VC(configs)
    pth_path, index_path = get_model(model_name)
    vc_data = vc.get_vc(pth_path, protect, 0.5)
    
    if split_infer:
        inferred_files = []
        temp_dir = os.path.join(os.getcwd(), "separate", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        print("Splitting audio to silence and nonsilent segments.")
        silence_files, nonsilent_files = split_silence_nonsilent(audio_path, min_silence, silence_threshold, seek_step, keep_silence)
        print(f"Total silence segments: {len(silence_files)}.\nTotal nonsilent segments: {len(nonsilent_files)}.")
        for i, nonsilent_file in enumerate(nonsilent_files):
            print(f"Inferring nonsilent audio {i+1}")
            inference_info, audio_data, output_path = vc.vc_single(
                0,
                nonsilent_file,
                f0_change,
                f0_method,
                index_path,
                index_path,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                audio_format,
                crepe_hop_length,
                do_formant,
                quefrency,
                timbre,
                min_pitch,
                max_pitch,
                f0_autotune,
                hubert_model_path
            )
            if inference_info[0] == "Success.":
                print("Inference ran successfully.")
                print(inference_info[1])
                print("Times:\nnpy: %.2fs f0: %.2fs infer: %.2fs\nTotal time: %.2fs" % (*inference_info[2],))
            else:
                print(f"An error occurred while processing.\n{inference_info[0]}")
                return None
            inferred_files.append(output_path)
        print("Adjusting inferred audio lengths.")
        adjusted_inferred_files = adjust_audio_lengths(nonsilent_files, inferred_files)
        print("Combining silence and inferred audios.")
        output_count = 1
        while True:
            output_path = os.path.join(os.getcwd(), "output", f"{os.path.splitext(os.path.basename(audio_path))[0]}{model_name}{f0_method.capitalize()}_{output_count}.{audio_format}")
            if not os.path.exists(output_path):
                break
            output_count += 1
        output_path = combine_silence_nonsilent(silence_files, adjusted_inferred_files, keep_silence, output_path)
        [shutil.move(inferred_file, temp_dir) for inferred_file in inferred_files]
        shutil.rmtree(temp_dir)
    else:
        inference_info, audio_data, output_path = vc.vc_single(
            0,
            audio_path,
            f0_change,
            f0_method,
            index_path,
            index_path,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            audio_format,
            crepe_hop_length,
            do_formant,
            quefrency,
            timbre,
            min_pitch,
            max_pitch,
            f0_autotune,
            hubert_model_path
        )
        if inference_info[0] == "Success.":
            print("Inference ran successfully.")
            print(inference_info[1])
            print("Times:\nnpy: %.2fs f0: %.2fs infer: %.2fs\nTotal time: %.2fs" % (*inference_info[2],))
        else:
            print(f"An error occurred while processing.\n{inference_info[0]}")
            del configs, vc
            gc.collect()
            return inference_info[0]
    
    del configs, vc
    gc.collect()
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Audio Inference CLI using RVC Voice Conversion Model.")
    
    parser.add_argument('-me', '--model_name', type=str, required=True, help="Name of the voice model to use.")
    parser.add_argument('-ap', '--audio_path', type=str, required=True, help="Path to the input audio file.")
    parser.add_argument('-fch', '--f0_change', type=float, default=0.0, help="F0 change value.")
    parser.add_argument('f0m', '--f0_method', type=str, default="rmvpe", help="F0 method to use.")
    parser.add_argument('-min', '--min_pitch', type=str, default="50", help="Minimum pitch value.")
    parser.add_argument('-max', '--max_pitch', type=str, default="1100", help="Maximum pitch value.")
    parser.add_argument('-chp', '--crepe_hop_length', type=int, default=128, help="CREPE hop length.")
    parser.add_argument('-ixr', '--index_rate', type=float, default=0.75, help="Index rate for inference.")
    parser.add_argument('-frs', '--filter_radius', type=int, default=3, help="Filter radius for inference.")
    parser.add_argument('-rmr', '--rms_mix_rate', type=float, default=0.25, help="RMS mix rate for inference.")
    parser.add_argument('-pro', '--protect', type=float, default=0.33, help="Protect value for inference.")
    parser.add_argument('-spt', '--split_infer', action='store_true', help="Whether to split the inference.")
    parser.add_argument('-msl', '--min_silence', type=int, default=500, help="Minimum silence duration in milliseconds.")
    parser.add_argument('-sth', '--silence_threshold', type=int, default=-50, help="Silence threshold in dB.")
    parser.add_argument('-sst', '--seek_step', type=int, default=1, help="Step size for seeking audio segments.")
    parser.add_argument('-kce', '--keep_silence', type=int, default=100, help="Duration of silence to keep in milliseconds.")
    parser.add_argument('-dof', '--do_formant', action='store_true', help="Whether to apply formant shifting.")
    parser.add_argument('-qfen', '--quefrency', type=int, default=0, help="Quefrency value for inference.")
    parser.add_argument('-tme', '--timbre', type=int, default=1, help="Timbre value for inference.")
    parser.add_argument('-ftune', '--f0_autotune', action='store_true', help="Whether to apply F0 autotune.")
    parser.add_argument('-ofor', '--audio_format', type=str, default="wav", help="Output audio format.")
    parser.add_argument('-resr', '--resample_sr', type=int, default=0, help="Resample sample rate.")
    parser.add_argument('-hubpa'  '--hubert_model_path', type=str, default="hubert_base.pt", help="Path to the hubert model.")
    parser.add_argument('-rmvpa', '--rmvpe_model_path', type=str, default="rmvpe.pt", help="Path to the rmvpe model.")
    parser.add_argument('-fcpa', '--fcpe_model_path', type=str, default="fcpe.pt", help="Path to the fcpe model.")
    
    args = parser.parse_args()
    
    # Download models if they don't exist
    download_models()
    
    output_path = infer_audio(
        args.model_name,
        args.audio_path,
        f0_change=args.f0_change,
        f0_method=args.f0_method,
        min_pitch=args.min_pitch,
        max_pitch=args.max_pitch,
        crepe_hop_length=args.crepe_hop_length,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        split_infer=args.split_infer,
        min_silence=args.min_silence,
        silence_threshold=args.silence_threshold,
        seek_step=args.seek_step,
        keep_silence=args.keep_silence,
        do_formant=args.do_formant,
        quefrency=args.quefrency,
        timbre=args.timbre,
        f0_autotune=args.f0_autotune,
        audio_format=args.audio_format,
        resample_sr=args.resample_sr,
        hubert_model_path=args.hubert_model_path,
        rmvpe_model_path=args.rmvpe_model_path,
        fcpe_model_path=args.fcpe_model_path
    )
    
    if output_path:
        print(f"Inference completed. Output saved to: {output_path}")
    else:
        print("Inference failed.")

if __name__ == "__main__":
    main()
