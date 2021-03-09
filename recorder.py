import numpy as np
import pyaudio
import threading
from estimate_bpm import BPM_Analyzer
import librosa

class InputRecorder:
    """Simple, cross-platform class to record from the default input device."""
    
    def __init__(self):
        self.RATE = 44100
        self.BUFFERSIZE = 2**12 
        self.secToRecord = 1.4 
        self.kill_threads = False
        self.has_new_audio = False
        self.setup()
        self.bpms = BPM_Analyzer()
        self.mic_sens_dBV = -42.0  # mic sensitivity in dBV + any gain
        self.mic_sens_corr = np.power(10.0, self.mic_sens_dBV / 20.0)  # calculate mic sensitivity conversion factor
        
    def setup(self):
        self.buffers_to_record = int(self.RATE * self.secToRecord / self.BUFFERSIZE)
        if self.buffers_to_record == 0:
            self.buffers_to_record = 1
        self.samples_to_record = int(self.BUFFERSIZE * self.buffers_to_record)
        self.chunks_to_record = int(self.samples_to_record / self.BUFFERSIZE)
        self.sec_per_point = 1. / self.RATE
        
        self.p = pyaudio.PyAudio()
        print("Using default input device: {:s}".format(self.p.get_default_input_device_info()['name']))
        self.in_stream = self.p.open(format=pyaudio.paInt16,
                                     channels=1,
                                     rate=self.RATE,
                                     input=True,
                                     frames_per_buffer=self.BUFFERSIZE)
        
        self.audio = np.empty((self.chunks_to_record * self.BUFFERSIZE), dtype=np.int16)               
    
    def close(self):
        self.kill_threads = True
        self.p.close(self.in_stream)
    
    ### RECORDING AUDIO ###  
    
    def get_audio(self):
        """get a single buffer size worth of audio."""
        audio_string = self.in_stream.read(self.BUFFERSIZE)
        return np.fromstring(audio_string, dtype=np.int16)
        
    def record(self):
        while not self.kill_threads:
            if not self.has_new_audio:
                self.in_stream.start_stream()
                for i in range(self.chunks_to_record):
                    self.audio[i*self.BUFFERSIZE:(i+1)*self.BUFFERSIZE] = self.get_audio()                    
                self.in_stream.stop_stream()
                # self.audio = ((self.audio / np.power(2.0, 15)) * 5.25) * (self.mic_sens_corr)
                self.has_new_audio = True
    
    def start(self):
        self.t = threading.Thread(target=self.record)
        self.t.start()

    def bpm(self, data=None):
        if not data:            
            data = self.audio 
        correlation, bpm = self.bpms.computeWindowBPM(data,self.RATE)
        return correlation, bpm

    def bpm_librosa(self, data=None):
        if not data:            
            data = self.byte_to_float(self.audio)
        onset_env = librosa.onset.onset_strength(data, sr=self.RATE)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.RATE)
        hop_length = 512
        ac = librosa.autocorrelate(onset_env, 2 * self.RATE // hop_length)
        return ac, tempo 

# https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182

    def float_to_byte(self,sig):
        # float32 -> int16(PCM_16) -> byte
        return  self.float2pcm(sig, dtype='int16').tobytes()

    def byte_to_float(self,byte):
        # byte -> int16(PCM_16) -> float32
        return self.pcm2float(np.frombuffer(byte,dtype=np.int16), dtype='float32')

    def pcm2float(self,sig, dtype='float32'):
        """Convert PCM signal to floating point with a range from -1 to 1.
        Use dtype='float32' for single precision.
        Parameters
        ----------
        sig : array_like
            Input array, must have integral type.
        dtype : data type, optional
            Desired (floating point) data type.
        Returns
        -------
        numpy.ndarray
            Normalized floating point data.
        See Also
        --------
        float2pcm, dtype
        """
        sig = np.asarray(sig)
        if sig.dtype.kind not in 'iu':
            raise TypeError("'sig' must be an array of integers")
        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(sig.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max


    def float2pcm(self,sig, dtype='int16'):
        """Convert floating point signal with a range from -1 to 1 to PCM.
        Any signal values outside the interval [-1.0, 1.0) are clipped.
        No dithering is used.
        Note that there are different possibilities for scaling floating
        point numbers to PCM numbers, this function implements just one of
        them.  For an overview of alternatives see
        http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
        Parameters
        ----------
        sig : array_like
            Input array, must have floating point type.
        dtype : data type, optional
            Desired (integer) data type.
        Returns
        -------
        numpy.ndarray
            Integer data, scaled and clipped to the range of the given
            *dtype*.
        See Also
        --------
        pcm2float, dtype
        """
        sig = np.asarray(sig)
        if sig.dtype.kind != 'f':
            raise TypeError("'sig' must be a float array")
        dtype = np.dtype(dtype)
        if dtype.kind not in 'iu':
            raise TypeError("'dtype' must be an integer type")

        i = np.iinfo(dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)        