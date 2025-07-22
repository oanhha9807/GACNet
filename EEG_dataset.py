from torch.utils.data import Dataset
import torch
import numpy as np
# from braindecode.augmentation import FrequencyShift
# import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import FastICA
# Example usage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# import pywt
import pickle
from scipy.fft import fft, fftfreq, rfft
# from scipy.signal import stft
# from numpy.fft import fft
import scipy.signal as signal


class EEGEyeNetDataset(Dataset):
    def __init__(self, data_file, channel_index, transpose = False, fft_method = False, filter_band = False):
        self.data_file = data_file
        print('loading data...')


        with open('/data/oanh/arrays_11.pkl', 'rb') as f:
            data_samples = pickle.load(f)

        min_length = min(sample.shape[1] for sample in data_samples)
        list_noise = []
        for d in range(len(data_samples)):
            if data_samples[d].shape[1] < 100:
                print(d)
                print(data_samples[d].shape)
                list_noise.append(d)
        print('Number of noise sample: ',len(list_noise))
        # a
        # reduced_data_samples = [self.wavelet_transform_reduce(sample, target_length=min_length) for sample in data_samples]
        # reduced_data_samples = np.asarray(reduced_data_samples)
        # print(reduced_data_samples.shape)

        with np.load(self.data_file) as f: # Load the data array
            self.trainX = f['EEG']
            self.trainY = f['labels']


        # self.trainX = np.asarray(data_samples)

        # self.indices_to_remove = [2479, 12683, 15725,15853, 19411,19470, 19954, 20676]
        # self.indices_to_remove = list_noise
        # self.indices_to_remove = list(np.load('/home/oem/oanh/DETRtime/DETRtime/list_noise_detr.npy'))
        # print(self.indices_to_remove)
        # # a
        # self.trainX = np.delete(self.trainX, self.indices_to_remove, axis=0)
        # self.trainY = np.delete(self.trainY, self.indices_to_remove, axis=0)

        if len(channel_index) > 0:
            index_list = channel_index
            print(self.trainX.shape)
            self.trainX = self.trainX[:, :, index_list]
            self.trainX = np.concatenate((self.trainX[:9127, :, :], self.trainX[9523:, :, :]), axis=0)
            self.trainY = np.concatenate((self.trainY[:9127], self.trainY[9523:]), axis=0)
        else:

            # self.trainX = self.trainX[:, 100:, :]
            # self.trainY = self.trainY[:]
            self.trainX = np.concatenate((self.trainX[:9127, :, :], self.trainX[9523:, :, :]), axis=0)
            self.trainY = np.concatenate((self.trainY[:9127], self.trainY[9523:]), axis=0)

            # self.trainX = np.concatenate((self.trainX[:9119, :, :], self.trainX[9514:, :, :]), axis=0)
            # self.trainY = np.concatenate((self.trainY[:9119], self.trainY[9514:]), axis=0)

            # self.trainX = np.concatenate((self.trainX[:, :100, :], self.trainX[:, 350:, :]), axis=1)
            # self.trainX = self.trainX[:, 100:350, :]
            # self.trainY = np.concatenate((self.trainY[:9125], self.trainY[9520:]), axis=0)

        if filter_band:
            # n_channels = 129
            # n_points = 500
            fs = 500  # Tần số lấy mẫu (500 Hz)
            lowcut = 0.5  # Giới hạn dưới (Hz)
            highcut = 40  # Giới hạn trên (Hz)
            order = 4  # Bậc của bộ lọc

            # Thiết kế bộ lọc Butterworth
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = signal.butter(order, [low, high], btype='band')

            # Hàm lọc từng kênh trong một sample
            def filter_sample(sample, b, a):
                filtered_sample = np.zeros_like(sample)
                for i in range(sample.shape[0]):  # Lặp qua các kênh
                    filtered_sample[i] = signal.filtfilt(b, a, sample[i])
                return filtered_sample

            # Áp dụng bộ lọc cho toàn bộ dữ liệu
            eeg_data = self.trainX
            filtered_eeg_data = np.zeros_like(eeg_data)
            for j in range(eeg_data.shape[0]):
                print(j)
                filtered_eeg_data[j] = filter_sample(eeg_data[j], b, a)
                # print( filtered_eeg_data[j].shape)

            print("Kích thước dữ liệu sau khi lọc:", filtered_eeg_data.shape)
            # a
            self.trainX = filtered_eeg_data

        if fft_method:
            print('Apply the FFT Transform')
            all_data = []
            count = 0
            for X in self.trainX:
                print(count)
                count += 1
                X = X.T
                fft_results = []
                for i in range(129):
                    # Tính FFT cho channel i
                    # eeg_data = X[i]
                    fft_result = fft(X[i])
                    real_part = np.real(fft_result)
                    fft_results.append(real_part)
                    # imag_part = np.imag(fft_result)
                    # fft_results.append(imag_part)

                fft_results = np.asarray(fft_results).T
                # print(fft_results.shape)
                all_data.append(fft_results)
            all_data = np.asarray(all_data)

            # print(all_data.shape)


            self.trainX = np.concatenate((self.trainX, all_data),  axis= 1 )
            print('Using FFT Transformation', self.trainX.shape)
            # pri
            # self.trainX = all_data

        print(self.trainX.shape)
        print(self.trainY.shape)
        # a
        print('From 0 to 52 and Denoise by FFT')
        '''
        self.scaler = StandardScaler()
        self.eeg_data_reshaped = self.trainX.reshape(-1, 129)
        # print(self.eeg_data_reshaped.shape)
        
        # a

        self.trainX = self.scaler.fit_transform(self.eeg_data_reshaped)
        # self.trainX = self.scaler.transform(self.trainX)

        self.trainX = self.trainX.reshape(-1, 500, 129)
        '''

        if transpose:
            # self.trainX = np.transpose(self.trainX, (0, 2, 1))[:, np.newaxis, np.newaxis, :, :]
            self.trainX = np.transpose(self.trainX, (0, 2, 1))
            # self.trainX = np.transpose(self.trainX, (0, 2, 1))[:, :, :]

    def __getitem__(self, index):
        # Read a single sample of data from the data array
        X = torch.from_numpy(self.trainX[index]).float()
        y = torch.from_numpy(self.trainY[index,1:3]).float()
        # Return the tensor data
        return (X,y,index)

    def __len__(self):
        # Compute the number of samples in the data array
        return len(self.trainX)

    def apply_wavelet_transform(self, eeg_signal, wavelet='db4', level=4):
        coeffs = pywt.wavedec(eeg_signal, wavelet, level=level)
        # print('level is 5')
        return coeffs

    def denoise_signal(self, coeffs, threshold):
        coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
        return pywt.waverec(coeffs, wavelet = 'db4')

    # def wavelet_transform_reduce(self, sample, wavelet='db4', target_length=None):
    #     channels, timepoints = sample.shape
    #     if target_length is None:
    #         target_length = min_length  # Giảm kích thước về kích thước của mẫu ngắn nhất
    #
    #     # Danh sách để chứa dữ liệu đã giảm kích thước
    #     reduced_sample = np.zeros((channels, target_length))
    #
    #     # Lặp qua từng kênh
    #     for i in range(channels):
    #         # Lấy dữ liệu cho kênh hiện tại
    #         channel_data = sample[i]
    #
    #         # Thực hiện Wavelet Transform
    #         coeffs = pywt.wavedec(channel_data, wavelet, level=None)
    #
    #         # Đảm bảo rằng số lượng hệ số xấp xỉ đủ để phù hợp với kích thước mục tiêu
    #         approx_coeffs = coeffs[0]
    #         if len(approx_coeffs) >= target_length:
    #             reduced_sample[i] = approx_coeffs[:target_length]  # Cắt để phù hợp với kích thước mục tiêu
    #         else:
    #             # Nếu hệ số xấp xỉ không đủ dài, bạn có thể cần phải xử lý đặc biệt (padding hoặc interpolation)
    #             # Dưới đây là một cách đơn giản để lấp đầy dữ liệu thiếu
    #             reduced_sample[i, :len(approx_coeffs)] = approx_coeffs
    #             # reduced_sample[i, len(approx_coeffs):] = approx_coeffs[-1]  # Padding với giá trị cuối cùng
    #
    #     return reduced_sample

    def wavelet_transform_reduce(self, sample, wavelet='db4', target_length=None):
        channels, timepoints = sample.shape
        if target_length is None:
            target_length = min_length  # Giảm kích thước về kích thước của mẫu ngắn nhất

        # Danh sách để chứa dữ liệu đã giảm kích thước
        reduced_sample = sample[:, :target_length]

        # # Lặp qua từng kênh
        # for i in range(channels):
        #     # Lấy dữ liệu cho kênh hiện tại
        #     channel_data = sample[i]
        #
        #     # Thực hiện Wavelet Transform
        #     coeffs = pywt.wavedec(channel_data, wavelet, level=None)
        #
        #     # Đảm bảo rằng số lượng hệ số xấp xỉ đủ để phù hợp với kích thước mục tiêu
        #     approx_coeffs = coeffs[0]
        #     if len(approx_coeffs) >= target_length:
        #         reduced_sample[i] = approx_coeffs[:target_length]  # Cắt để phù hợp với kích thước mục tiêu
        #     else:
        #         # Nếu hệ số xấp xỉ không đủ dài, bạn có thể cần phải xử lý đặc biệt (padding hoặc interpolation)
        #         # Dưới đây là một cách đơn giản để lấp đầy dữ liệu thiếu
        #         reduced_sample[i, :len(approx_coeffs)] = approx_coeffs
        #         # reduced_sample[i, len(approx_coeffs):] = approx_coeffs[-1]  # Padding với giá trị cuối cùng

        return reduced_sample