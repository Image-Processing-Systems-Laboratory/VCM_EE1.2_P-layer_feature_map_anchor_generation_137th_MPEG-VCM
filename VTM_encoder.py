import threading
from tqdm import tqdm
import os
import glob
import argparse
from PIL import Image, ImageOps
import subprocess as subp
import sys
from time import sleep

class Worker(threading.Thread):
    def __init__(self, file_path, path, qp):
        super().__init__()
        self.file_path = file_path
        self.path = path
        self.qp = qp

    def run(self):

        file_name = os.path.basename(self.file_path)[:-4]
        
        img = Image.open(self.file_path)

        width = img.size[0]
        height = img.size[1] 

        bitstream_path = self.path.replace("ori", "bit/")
        recon_path = self.path.replace("ori", "rec/")
        temp_path   = self.path.replace("ori", "tmp/")

        stdout_vtm = open(f"{temp_path}vtm_log.txt", 'w')
        stdout_fmp = open(f"{temp_path}ffmpeg_log.txt", 'w')
        
        # Convert png to yuv
        if (os.path.exists(f"{temp_path}{file_name}_yuv.yuv")): os.remove(f"{temp_path}{file_name}_yuv.yuv")
        subp.run(f"ffmpeg -i {self.file_path} -f rawvideo -pix_fmt gray16le -dst_range 1 {temp_path}{file_name}_yuv.yuv", shell = True, stdout = stdout_fmp, stderr = stdout_fmp)         
        # Encoding
        subp.run(f"./EncoderAppStatic -c ./cfg/encoder_intra_vtm.cfg -i {temp_path}{file_name}_yuv.yuv -o \"\" -b {bitstream_path}{file_name}.vvc -q {self.qp} --ConformanceWindowMode=1 -wdt {width} -hgt {height} -f 1 -fr 1 --InternalBitDepth=10 --InputBitDepth=10 --InputChromaFormat=400 --OutputBitDepth=10", stdout = stdout_vtm, shell = True)
        
        # Decoding
        subp.run(f"./DecoderAppStatic -b {bitstream_path}{file_name}.vvc -o {temp_path}{file_name}_rec.yuv", stdout = stdout_vtm, shell = True)
        # Convert yuv to png
        if (os.path.exists(f"{temp_path}{file_name}_rec.png")): os.remove(f"{temp_path}{file_name}_rec.png")
        subp.run(f"ffmpeg -f rawvideo -pix_fmt gray16le -s {width}x{height} -src_range 1 -i {temp_path}{file_name}_rec.yuv -frames 1 -pix_fmt gray16le {recon_path}{file_name}.png", shell = True, stdout = stdout_fmp, stderr = stdout_fmp) 
        
        # Remove tmp files
        try:
            os.remove(f"{temp_path}{file_name}_yuv.yuv")
        except OSError:
            pass
        try:
            os.remove(f"{temp_path}{file_name}_rec.yuv")
        except OSError:
            pass

def run_vtm(path, qp, threads):
    FileList = glob.glob(f"{path}/*.png")
    bitstream_path = path.replace("ori", "bit/")
    recon_path = path.replace("ori", "rec/")
    temp_path   = path.replace("ori", "tmp/")

    os.makedirs(bitstream_path, exist_ok = True)
    os.makedirs(recon_path, exist_ok = True)
    os.makedirs(temp_path,   exist_ok = True)

    for file_path in tqdm(FileList):

        file_name = os.path.basename(file_path)[:-4] 
        
        if os.path.isfile(f"{recon_path}{file_name}.png"):
            print(f"{file_name} skip (exist)")
            continue

        while (threads + 1 < threading.active_count()): sleep(1)
        
        file_path = file_path.encode('utf-8','backslashreplace').decode().replace("\\","/")     

        t = Worker(file_path, path, qp)

        t.start()